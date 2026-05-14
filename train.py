"""Training loop."""

from __future__ import annotations

import heapq
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from model import G2PTransformer
from data import BOS_IDX, EOS_IDX, PAD_IDX, Vocab
from evaluate import compute_metrics


# Loss

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, pad_idx=PAD_IDX, smoothing=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, targets):
        B, T, V = logits.shape
        lf = logits.reshape(-1, V); tf = targets.reshape(-1)
        lp = F.log_softmax(lf, dim=-1)
        with torch.no_grad():
            sd = torch.full_like(lp, self.smoothing / (V - 2))
            sd[:, self.pad_idx] = 0.0
            sd.scatter_(1, tf.unsqueeze(1), self.confidence)
            sd[tf == self.pad_idx] = 0.0
        loss = -(sd * lp).sum(-1)
        non_pad = (tf != self.pad_idx).sum()
        return loss.sum() / non_pad.clamp(min=1)


# LR schedule

def cosine_warmup_schedule(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# Validation

@torch.no_grad()
def validate(model, loader, loss_fn, tgt_vocab, device):
    model.eval()
    total_loss = 0.0; total_tok = 0
    all_preds = []; all_refs = []; all_seen = []
    for batch in loader:
        src = batch["src"].to(device); ti = batch["tgt_input"].to(device)
        to  = batch["tgt_output"].to(device)
        sp  = batch["src_pad_mask"].to(device); tp = batch["tgt_pad_mask"].to(device)
        logits = model(src, ti, sp, tp)
        loss = loss_fn(logits, to)
        n = (to != PAD_IDX).sum().item()
        total_loss += loss.item() * n; total_tok += n
        preds = model.greedy_decode(src, BOS_IDX, EOS_IDX)
        for ids, ref_ph, seen in zip(preds, batch["phonemes"], batch["seen_in_train"]):
            all_preds.append(tgt_vocab.decode(ids))
            all_refs.append(ref_ph.split())
            all_seen.append(seen)
    model.train()
    metrics = compute_metrics(all_preds, all_refs, all_seen)
    metrics["loss"] = total_loss / max(total_tok, 1)
    return metrics


# Top-K checkpoint manager

class TopKCheckpoints:
    def __init__(self, k, output_dir):
        self.k = k; self.output_dir = output_dir; self._heap = []

    def update(self, val_loss, epoch, model_state, opt_state, extra):
        path = str(self.output_dir / f"ckpt_ep{epoch:03d}_loss{val_loss:.4f}.pt")
        torch.save({"epoch": epoch, "model_state_dict": model_state,
                    "optimizer_state_dict": opt_state, "val_loss": val_loss, **extra}, path)
        heapq.heappush(self._heap, (val_loss, path))
        if len(self._heap) > self.k:
            worst_loss, worst_path = max(self._heap, key=lambda x: x[0])
            self._heap = [(l,p) for l,p in self._heap if p != worst_path]
            heapq.heapify(self._heap)
            try: Path(worst_path).unlink(missing_ok=True)
            except Exception: pass

    def best_paths(self): return [p for _,p in sorted(self._heap)]


# Training

def train(
    model: G2PTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tgt_vocab: Vocab,
    device: torch.device,
    output_dir: Path,
    num_epochs: int = 120,
    peak_lr: float = 3e-4,
    weight_decay: float = 1e-2,
    warmup_epochs: int = 5,
    label_smoothing: float = 0.1,
    patience: int = 15,
    clip_grad_norm: float = 1.0,
    grad_accumulation_steps: int = 1,
    save_top_k: int = 3,
    dropout_anneal: bool = True,
    dropout_anneal_epoch: int = 25,
    dropout_target: float = 0.05,
) -> dict:

    output_dir.mkdir(parents=True, exist_ok=True)
    steps_per_epoch = math.ceil(len(train_loader) / grad_accumulation_steps)
    total_steps  = num_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    loss_fn   = LabelSmoothingLoss(model.tgt_vocab_size, PAD_IDX, label_smoothing).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr,
                                  betas=(0.9, 0.98), eps=1e-9, weight_decay=weight_decay)
    scheduler = cosine_warmup_schedule(optimizer, warmup_steps, total_steps)
    ckpt_mgr  = TopKCheckpoints(save_top_k, output_dir)

    history = {"train_loss":[], "val_loss":[], "val_exact_match":[], "val_per":[], "lr":[]}
    best_val_loss = float("inf"); no_improve = 0

    print(f"[Train] {num_epochs} epochs | peak_lr={peak_lr} | warmup={warmup_epochs}ep | "
          f"accum={grad_accumulation_steps} | eff_batch≈{train_loader.batch_size * grad_accumulation_steps}")

    for epoch in range(1, num_epochs + 1):
        model.train()

        if dropout_anneal and epoch == dropout_anneal_epoch:
            for m in model.modules():
                if isinstance(m, nn.Dropout): m.p = dropout_target
            print(f"[Train] Epoch {epoch}: dropout → {dropout_target}")

        epoch_loss = 0.0; epoch_tok = 0
        optimizer.zero_grad()
        t0 = time.time()

        for bi, batch in enumerate(train_loader):
            src = batch["src"].to(device); ti = batch["tgt_input"].to(device)
            to  = batch["tgt_output"].to(device)
            sp  = batch["src_pad_mask"].to(device); tp = batch["tgt_pad_mask"].to(device)

            logits = model(src, ti, sp, tp)
            loss   = loss_fn(logits, to) / grad_accumulation_steps
            loss.backward()

            n = (to != PAD_IDX).sum().item()
            epoch_loss += loss.item() * grad_accumulation_steps * n
            epoch_tok  += n

            if (bi + 1) % grad_accumulation_steps == 0:
                if clip_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()

        train_loss  = epoch_loss / max(epoch_tok, 1)
        current_lr  = optimizer.param_groups[0]["lr"]
        val_metrics = validate(model, val_loader, loss_fn, tgt_vocab, device)
        val_loss    = val_metrics["loss"]
        val_em      = val_metrics["exact_match"]
        val_per     = val_metrics["per"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_exact_match"].append(val_em)
        history["val_per"].append(val_per)
        history["lr"].append(current_lr)

        print(f"Epoch {epoch:3d} | train={train_loss:.4f} | val={val_loss:.4f} | "
              f"EM={val_em:.3f} | PER={val_per:.3f} | lr={current_lr:.2e} | {time.time()-t0:.1f}s")

        ckpt_mgr.update(val_loss, epoch, model.state_dict(), optimizer.state_dict(),
                        {"val_exact_match": val_em, "val_per": val_per})

        if val_loss < best_val_loss:
            best_val_loss = val_loss; no_improve = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss, "val_exact_match": val_em},
                       output_dir / "best_model.pt")
            print(f"  -> Best (val={val_loss:.4f}, EM={val_em:.3f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[Train] Early stopping at epoch {epoch}."); break

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    top_paths = ckpt_mgr.best_paths()
    (output_dir / "top_checkpoints.json").write_text(json.dumps(top_paths, indent=2), encoding="utf-8")
    print(f"[Train] Top-{save_top_k} ckpts: {[Path(p).name for p in top_paths]}")
    return history
