"""Data loading and batching for G2P experiments."""

from __future__ import annotations

import random
import unicodedata
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

PAD = "<PAD>"; UNK = "<UNK>"; BOS = "<BOS>"; EOS = "<EOS>"
SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]
PAD_IDX = 0; UNK_IDX = 1; BOS_IDX = 2; EOS_IDX = 3

# Vocabulary

class Vocab:
    def __init__(self):
        self.token2idx: dict[str, int] = {}
        self.idx2token: list[str] = []
        for tok in SPECIAL_TOKENS:
            self._add(tok)

    def _add(self, token: str) -> int:
        if token not in self.token2idx:
            self.token2idx[token] = len(self.idx2token)
            self.idx2token.append(token)
        return self.token2idx[token]

    def build_from_sequences(self, sequences: list[list[str]]) -> "Vocab":
        for seq in sequences:
            for tok in seq: self._add(tok)
        return self

    def encode(self, tokens, add_bos=False, add_eos=False):
        ids = [self.token2idx.get(t, UNK_IDX) for t in tokens]
        if add_bos: ids = [BOS_IDX] + ids
        if add_eos: ids = ids + [EOS_IDX]
        return ids

    def decode(self, ids, strip_special=True):
        specials = {PAD_IDX, BOS_IDX, EOS_IDX}
        return [
            self.idx2token[i] if i < len(self.idx2token) else UNK
            for i in ids if not (strip_special and i in specials)
        ]

    def __len__(self): return len(self.idx2token)

    def save(self, path: Path):
        path.write_text("\n".join(self.idx2token), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        v = cls.__new__(cls)
        v.idx2token = path.read_text(encoding="utf-8").splitlines()
        v.token2idx = {t: i for i, t in enumerate(v.idx2token)}
        return v

# IPA tokenisation

IPA_COMBINING = set("ːʰʷʲˤˠˁⁿˡ")
TIE_BARS = {"͡", "͜"}
IPA_PROSODIC = {"ˈ", "ˌ", ".", "|", "‖"}

def tokenise_ipa(transcription: str, whitespace_tokenised: bool = True) -> list[str]:
    if whitespace_tokenised:
        tokens = transcription.split()
        result = []
        for tok in tokens:
            if len(tok) > 1 and tok[0] in IPA_PROSODIC:
                result.append(tok[0]); result.append(tok[1:])
            else:
                result.append(tok)
        return [t for t in result if t]
    tokens = []; i = 0; chars = list(transcription)
    while i < len(chars):
        c = chars[i]
        if c in IPA_PROSODIC:
            tokens.append(c); i += 1; continue
        cluster = c; i += 1
        while i < len(chars):
            nc = chars[i]
            if nc in TIE_BARS and i + 1 < len(chars):
                cluster += nc + chars[i + 1]; i += 2
            elif nc in IPA_COMBINING or unicodedata.category(nc) in ("Mn", "Mc"):
                cluster += nc; i += 1
            else: break
        tokens.append(cluster)
    return tokens

def tokenise_grapheme(word: str) -> list[str]:
    tokens = []; i = 0; chars = list(word)
    while i < len(chars):
        cluster = chars[i]; i += 1
        while i < len(chars) and unicodedata.category(chars[i]) in ("Mn", "Mc", "Me"):
            cluster += chars[i]; i += 1
        tokens.append(cluster)
    return tokens

# TSV reading

def read_tsv(path: Path, whitespace_tokenised: bool = True, lang: str | None = None) -> list[dict]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line: continue
            parts = line.split("\t")
            if len(parts) == 2: grapheme, phoneme = parts
            elif len(parts) >= 3: grapheme, phoneme = parts[0], parts[1]
            else: continue
            examples.append({
                "grapheme": grapheme,
                "phoneme": phoneme,
                "grapheme_tokens": tokenise_grapheme(grapheme),
                "phoneme_tokens": tokenise_ipa(phoneme, whitespace_tokenised),
                "lang": lang or "unk",
                "seen_in_train": None,
            })
    return examples

# Split strategies

def deduplicate_splits(train, val, test, augment_val_to=300, seed=42):
    """Remove train/val graphemes that occur in the test split."""
    test_g = {ex["grapheme"] for ex in test}
    clean_train = [ex for ex in train if ex["grapheme"] not in test_g]
    clean_val   = [ex for ex in val   if ex["grapheme"] not in test_g]
    for ex in test: ex["seen_in_train"] = False
    for ex in clean_val: ex["seen_in_train"] = False
    print(f"[Dedup] Removed {len(train)-len(clean_train)} train / "
          f"{len(val)-len(clean_val)} val (appeared in test).")
    if augment_val_to and len(clean_val) < augment_val_to:
        rng = random.Random(seed)
        extras = rng.sample(clean_train, min(augment_val_to - len(clean_val), len(clean_train)))
        clean_val = clean_val + extras
        print(f"[Dedup] Augmented val → {len(clean_val)}.")
    return clean_train, clean_val, test

def tag_memorisation(train, val, test):
    """Strategy B: keep everything, tag seen/unseen."""
    train_g = {ex["grapheme"] for ex in train}
    for ex in test + val: ex["seen_in_train"] = ex["grapheme"] in train_g
    n = sum(1 for ex in test if ex["seen_in_train"])
    print(f"[Tag] Test: {n}/{len(test)} seen ({100*n/max(len(test),1):.1f}%).")
    return train, val, test

# Lookup cache

def build_lookup_cache(train: list[dict]) -> dict[str, list[str]]:
    """Map training graphemes to their most frequent pronunciation."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for ex in train:
        key = " ".join(ex["phoneme_tokens"])
        counts[ex["grapheme"]][key] += 1
    cache = {g: max(prons, key=prons.get).split()
             for g, prons in counts.items()}
    print(f"[Cache] {len(cache)} entries.")
    return cache

# Grapheme augmentation

def augment_grapheme(tokens: list[str], prob: float = 0.15,
                     rng: random.Random | None = None) -> list[str]:
    """Apply swap/delete/duplicate noise."""
    if rng is None: rng = random
    if len(tokens) <= 1: return tokens
    tokens = list(tokens)
    if len(tokens) > 2 and rng.random() < prob:
        i = rng.randint(0, len(tokens) - 2)
        tokens[i], tokens[i+1] = tokens[i+1], tokens[i]
    if len(tokens) > 2 and rng.random() < prob:
        tokens.pop(rng.randint(0, len(tokens) - 1))
    if rng.random() < prob:
        i = rng.randint(0, len(tokens) - 1)
        tokens.insert(i, tokens[i])
    return tokens

# Dataset

class G2PDataset(Dataset):
    def __init__(self, examples, src_vocab, tgt_vocab,
                 augment=False, augment_prob=0.15, multilingual=False):
        self.examples = examples
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.augment = augment
        self.augment_prob = augment_prob
        self.multilingual = multilingual
        self._rng = random.Random(42)

    def __len__(self): return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        g_tokens = list(ex["grapheme_tokens"])
        if self.multilingual:
            lang_tag = f"<lang:{ex.get('lang','unk')}>"
            g_tokens = [lang_tag] + g_tokens
        if self.augment:
            start = 1 if self.multilingual else 0
            g_tokens = g_tokens[:start] + augment_grapheme(
                g_tokens[start:], self.augment_prob, self._rng)
        src_ids    = self.src_vocab.encode(g_tokens)
        tgt_in     = self.tgt_vocab.encode(ex["phoneme_tokens"], add_bos=True)
        tgt_out    = self.tgt_vocab.encode(ex["phoneme_tokens"], add_eos=True)
        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "tgt_input": torch.tensor(tgt_in, dtype=torch.long),
            "tgt_output": torch.tensor(tgt_out, dtype=torch.long),
            "grapheme": ex["grapheme"],
            "phoneme": ex["phoneme"],
            "lang": ex.get("lang", "unk"),
            "seen_in_train": ex.get("seen_in_train", None),
        }

def collate_fn(batch):
    src    = pad_sequence([b["src"]       for b in batch], batch_first=True, padding_value=PAD_IDX)
    tgt_in = pad_sequence([b["tgt_input"] for b in batch], batch_first=True, padding_value=PAD_IDX)
    tgt_out= pad_sequence([b["tgt_output"]for b in batch], batch_first=True, padding_value=PAD_IDX)
    return {
        "src": src, "tgt_input": tgt_in, "tgt_output": tgt_out,
        "src_pad_mask": (src    == PAD_IDX),
        "tgt_pad_mask": (tgt_in == PAD_IDX),
        "graphemes":    [b["grapheme"]     for b in batch],
        "phonemes":     [b["phoneme"]      for b in batch],
        "langs":        [b["lang"]         for b in batch],
        "seen_in_train":[b["seen_in_train"]for b in batch],
    }

def build_dataloaders(train_ex, val_ex, test_ex,
                      batch_size=64, num_workers=0,
                      augment=True, augment_prob=0.15, multilingual=False):
    langs = {ex.get("lang","unk") for ex in train_ex + val_ex + test_ex}
    lang_tags = [f"<lang:{l}>" for l in sorted(langs)]
    src_vocab = Vocab()
    for tag in lang_tags: src_vocab._add(tag)
    src_vocab.build_from_sequences([ex["grapheme_tokens"] for ex in train_ex])
    tgt_vocab = Vocab().build_from_sequences([ex["phoneme_tokens"] for ex in train_ex])
    print(f"[Vocab] src={len(src_vocab)} (incl. {len(lang_tags)} lang tags), tgt={len(tgt_vocab)}")

    def loader(examples, shuffle, aug):
        return DataLoader(
            G2PDataset(examples, src_vocab, tgt_vocab,
                       augment=aug, augment_prob=augment_prob,
                       multilingual=multilingual),
            batch_size=batch_size, shuffle=shuffle,
            collate_fn=collate_fn, num_workers=num_workers,
        )
    return src_vocab, tgt_vocab, loader(train_ex,True,augment), loader(val_ex,False,False), loader(test_ex,False,False)
