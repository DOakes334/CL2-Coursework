"""Train and evaluate G2P models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data import (read_tsv, deduplicate_splits, tag_memorisation,
                  build_dataloaders, build_lookup_cache, Vocab)
from model import G2PTransformer, average_checkpoints
from train import train
from evaluate import compute_metrics, error_analysis, print_metrics


# CLI

def parse_args():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--data_dir",  type=Path, default=Path("data"))
    p.add_argument("--lang",      nargs="+", default=["en"])
    p.add_argument("--whitespace_tokenised", action="store_true", default=True)
    p.add_argument("--strategy",  choices=["dedup","tag","none"], default="dedup",
                   help="dedup=Strategy A (recommended); tag=Strategy B (overlap analysis)")
    p.add_argument("--augment_val_to", type=int, default=300)
    p.add_argument("--augment",   action="store_true", default=True,
                   help="Enable grapheme augmentation (default on)")
    p.add_argument("--no_augment",action="store_true")
    p.add_argument("--augment_prob", type=float, default=0.15)
    p.add_argument("--multilingual", action="store_true",
                   help="Prepend <lang:xx> token for multilingual training")
    p.add_argument("--use_cache", action="store_true", default=True,
                   help="Use lookup cache for seen words at test time")
    p.add_argument("--no_cache",  action="store_true")
    # Model
    p.add_argument("--d_model",       type=int,   default=384)
    p.add_argument("--num_heads",     type=int,   default=6)
    p.add_argument("--n_kv_heads",    type=int,   default=None)
    p.add_argument("--num_enc_layers",type=int,   default=6)
    p.add_argument("--num_dec_layers",type=int,   default=6)
    p.add_argument("--d_ff",          type=int,   default=1024)
    p.add_argument("--dropout",       type=float, default=0.15)
    p.add_argument("--layer_drop",    type=float, default=0.1)
    # Training
    p.add_argument("--batch_size",    type=int,   default=128)
    p.add_argument("--grad_accumulation", type=int, default=1)
    p.add_argument("--num_epochs",    type=int,   default=120)
    p.add_argument("--peak_lr",       type=float, default=3e-4)
    p.add_argument("--warmup_epochs", type=int,   default=5)
    p.add_argument("--label_smoothing",type=float,default=0.1)
    p.add_argument("--patience",      type=int,   default=15)
    p.add_argument("--clip_grad_norm",type=float, default=1.0)
    p.add_argument("--weight_decay",  type=float, default=1e-2)
    p.add_argument("--save_top_k",    type=int,   default=3)
    p.add_argument("--dropout_anneal_epoch", type=int, default=25)
    p.add_argument("--dropout_target",type=float, default=0.05)
    # Decoding
    p.add_argument("--beam_size",     type=int,   default=4)
    p.add_argument("--length_penalty",type=float, default=0.6)
    p.add_argument("--no_weight_avg", action="store_true")
    # I/O
    p.add_argument("--output_dir",    type=Path,  default=Path("runs/default"))
    p.add_argument("--eval_only",     action="store_true")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--num_workers",   type=int,   default=0)
    return p.parse_args()


# Data loading

def load_language_data(data_dir, lang, whitespace_tokenised=True):
    lang_dir = data_dir / lang
    prefix = {"en": "eng", "eng": "eng", "es": "spa", "spa": "spa"}.get(lang, lang)

    def _read(path):
        if path.exists():
            ex = read_tsv(path, whitespace_tokenised, lang=lang)
            print(f"  {len(ex):>6}  ← {path}"); return ex
        return []

    train_d = _read(lang_dir / "train.tsv") or _read(data_dir / f"{prefix}_train.tsv")
    val_d   = _read(lang_dir / "val.tsv") or _read(lang_dir / "dev.tsv") or _read(data_dir / f"{prefix}_val.tsv")
    test_d  = _read(lang_dir / "test.tsv") or _read(data_dir / f"{prefix}_test.tsv") or _read(data_dir / "test.tsv")

    if not train_d:
        raise FileNotFoundError(f"No train data for '{lang}' in {lang_dir}")
    return train_d, val_d, test_d


# Decoding

@torch.no_grad()
def decode_test_set(model, test_loader, tgt_vocab, device,
                    beam_size=4, length_penalty=0.6,
                    lookup_cache: dict | None = None,
                    multilingual: bool = False):
    """Decode with cached seen-word pronunciations where available."""
    from data import BOS_IDX, EOS_IDX
    model.eval()
    all_preds, all_refs, all_graphemes, all_seen, all_langs = [], [], [], [], []

    for batch in test_loader:
        src       = batch["src"].to(device)
        graphemes = batch["graphemes"]
        phonemes  = batch["phonemes"]
        seen      = batch["seen_in_train"]
        langs     = batch["langs"]

        neural_indices = []
        batch_preds = [None] * len(graphemes)

        for i, g in enumerate(graphemes):
            if lookup_cache is not None and g in lookup_cache:
                batch_preds[i] = lookup_cache[g]
            else:
                neural_indices.append(i)

        if neural_indices:
            src_sub = src[neural_indices]
            if beam_size <= 1:
                ids_list = model.greedy_decode(src_sub, BOS_IDX, EOS_IDX)
                for j, ids in zip(neural_indices, ids_list):
                    batch_preds[j] = tgt_vocab.decode(ids)
            else:
                for j in neural_indices:
                    ids = model.beam_decode(
                        src[j:j+1], BOS_IDX, EOS_IDX,
                        beam_size=beam_size, length_penalty=length_penalty)
                    batch_preds[j] = tgt_vocab.decode(ids[0])

        all_preds.extend(batch_preds)
        for ph in phonemes: all_refs.append(ph.split())
        all_graphemes.extend(graphemes)
        all_seen.extend(seen)
        all_langs.extend(langs)

    # Stress markers
    for i in range(len(all_preds)):
        all_preds[i] = " ".join(all_preds[i]).replace("ˈ ", "ˈ").replace("ˌ ", "ˌ").split()

    return all_preds, all_refs, all_graphemes, all_seen, all_langs


# Main

def main():
    args = parse_args()
    if args.no_augment: args.augment = False
    if args.no_cache:   args.use_cache = False

    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[Run] Device: {device} | Langs: {args.lang} | Strategy: {args.strategy} | "
          f"Augment: {args.augment} | Cache: {args.use_cache}")

    # Load data
    all_train, all_val, all_test = [], [], []
    for lang in args.lang:
        print(f"\n[Data] {lang}")
        tr, va, te = load_language_data(args.data_dir, lang, args.whitespace_tokenised)
        all_train.extend(tr); all_val.extend(va); all_test.extend(te)
    print(f"\n[Data] Total: train={len(all_train)}, val={len(all_val)}, test={len(all_test)}")

    # Overlap strategy
    if args.strategy == "dedup":
        # Per-language deduplication
        if len(args.lang) > 1:
            new_train, new_val = [], []
            for lang in args.lang:
                lt = [ex for ex in all_train if ex["lang"]==lang]
                lv = [ex for ex in all_val   if ex["lang"]==lang]
                le = [ex for ex in all_test  if ex["lang"]==lang]
                ct, cv, _ = deduplicate_splits(lt, lv, le, args.augment_val_to)
                new_train.extend(ct); new_val.extend(cv)
            all_train, all_val = new_train, new_val
        else:
            all_train, all_val, all_test = deduplicate_splits(
                all_train, all_val, all_test, args.augment_val_to)
    elif args.strategy == "tag":
        all_train, all_val, all_test = tag_memorisation(all_train, all_val, all_test)

    print(f"[Data] After strategy: train={len(all_train)}, val={len(all_val)}, test={len(all_test)}")

    # Build lookup cache from (cleaned) training set
    lookup_cache = build_lookup_cache(all_train) if args.use_cache else None

    # Build dataloaders
    src_vocab, tgt_vocab, train_loader, val_loader, test_loader = build_dataloaders(
        all_train, all_val, all_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment,
        augment_prob=args.augment_prob,
        multilingual=args.multilingual,
    )

    # Build model
    model = G2PTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=args.d_model,
        n_heads=args.num_heads,
        n_kv_heads=args.n_kv_heads,
        n_enc_layers=args.num_enc_layers,
        n_dec_layers=args.num_dec_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        layer_drop=args.layer_drop,
        pad_idx=0,
    ).to(device)

    if torch.cuda.device_count() > 1:
        print(f"[Run] DataParallel across {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Parameters: {n_params:,}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    src_vocab.save(args.output_dir / "src_vocab.txt")
    tgt_vocab.save(args.output_dir / "tgt_vocab.txt")
    (args.output_dir / "config.json").write_text(
        json.dumps(vars(args), default=str, indent=2), encoding="utf-8")

    # Train
    if not args.eval_only:
        train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            tgt_vocab=tgt_vocab,
            device=device,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            peak_lr=args.peak_lr,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            label_smoothing=args.label_smoothing,
            patience=args.patience,
            clip_grad_norm=args.clip_grad_norm,
            grad_accumulation_steps=args.grad_accumulation,
            save_top_k=args.save_top_k,
            dropout_anneal_epoch=args.dropout_anneal_epoch,
            dropout_target=args.dropout_target,
        )

    # Load best / averaged model
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    top_path  = args.output_dir / "top_checkpoints.json"
    if not args.no_weight_avg and top_path.exists():
        top_paths = [p for p in json.loads(top_path.read_text()) if Path(p).exists()]
        if len(top_paths) > 1:
            raw_model = average_checkpoints([Path(p) for p in top_paths], raw_model)
            print(f"[Eval] Weight-averaged {len(top_paths)} checkpoints")
        else:
            ckpt = torch.load(args.output_dir / "best_model.pt", map_location=device)
            raw_model.load_state_dict(ckpt["model_state_dict"])
    else:
        ckpt = torch.load(args.output_dir / "best_model.pt", map_location=device)
        raw_model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Eval] Loaded best model (epoch {ckpt['epoch']}, val={ckpt['val_loss']:.4f})")
    raw_model = raw_model.to(device)

    # Decode
    print(f"\n[Eval] Decoding (beam={args.beam_size}, cache={'on' if lookup_cache else 'off'})...")
    preds, refs, graphemes, seen_flags, langs = decode_test_set(
        raw_model, test_loader, tgt_vocab, device,
        beam_size=args.beam_size, length_penalty=args.length_penalty,
        lookup_cache=lookup_cache,
        multilingual=args.multilingual,
    )

    # Overall metrics
    metrics = compute_metrics(preds, refs, seen_flags)
    print_metrics(metrics, title=f"Test Results ({', '.join(args.lang)})")

    # Per-language metrics
    if len(args.lang) > 1:
        for lang in args.lang:
            idxs = [i for i,l in enumerate(langs) if l == lang]
            lp = [preds[i] for i in idxs]; lr = [refs[i] for i in idxs]
            ls = [seen_flags[i] for i in idxs]
            lm = compute_metrics(lp, lr, ls)
            print_metrics(lm, title=f"Language: {lang} (n={len(idxs)})")

    # Error analysis
    analysis = error_analysis(preds, refs, graphemes, seen_flags)
    print("\nTop substitutions:")
    for s in analysis["top_substitutions"][:12]:
        print(f"  {s['hyp']:8s} -> {s['ref']:8s}  x{s['count']}")
    print("\nHardest examples (unseen):")
    unseen_hard = [e for e in analysis["hardest_examples"] if not e.get("seen_in_train")]
    for ex in unseen_hard[:8]:
        print(f"  {ex['grapheme']:20s} | {ex['pred']:40s} | {ex['ref']} (lev={ex['levenshtein']})")

    # Save outputs
    from evaluate import levenshtein as lev_fn
    with open(args.output_dir / "test_predictions.tsv", "w", encoding="utf-8") as f:
        f.write("grapheme\tpredicted\treference\texact_match\tlevenshtein\tper\tseen_in_train\tlang\n")
        for g, pred, ref, seen, lang in zip(graphemes, preds, refs, seen_flags, langs):
            l = lev_fn(pred, ref); per = l / max(len(ref), 1)
            f.write(f"{g}\t{' '.join(pred)}\t{' '.join(ref)}\t"
                    f"{int(pred==ref)}\t{l}\t{per:.4f}\t{seen}\t{lang}\n")

    (args.output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output_dir / "error_analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[Run] Outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
