"""Evaluation metrics and error analysis."""

from __future__ import annotations

import unicodedata
from collections import defaultdict


# Phone classes

IPA_VOWELS = set("aeiouæœøɛɪʊɔɑɒɐɜɞʏʉɨɘɵɤəɚɝ")


def _is_vowel(phone: str) -> bool:
    """Strip diacritics, then check the base symbol."""
    base = "".join(
        c for c in unicodedata.normalize("NFD", phone)
        if unicodedata.category(c) != "Mn"  # strip combining marks
    )
    return base[:1] in IPA_VOWELS


def weighted_edit_distance(
    hyp: list[str],
    ref: list[str],
    sub_vowel_vowel: float = 0.5,
    sub_cons_cons: float = 0.8,
    sub_cross: float = 1.5,
    ins_cost: float = 1.0,
    del_cost: float = 1.0,
) -> float:
    """
    Edit distance with phone-class-sensitive substitution weights.
    """
    H, R = len(hyp), len(ref)
    dp = [[0.0] * (R + 1) for _ in range(H + 1)]
    for i in range(H + 1):
        dp[i][0] = i * del_cost
    for j in range(R + 1):
        dp[0][j] = j * ins_cost
    for i in range(1, H + 1):
        for j in range(1, R + 1):
            if hyp[i - 1] == ref[j - 1]:
                cost = 0.0
            else:
                hv = _is_vowel(hyp[i - 1])
                rv = _is_vowel(ref[j - 1])
                if hv and rv:
                    cost = sub_vowel_vowel
                elif not hv and not rv:
                    cost = sub_cons_cons
                else:
                    cost = sub_cross
            dp[i][j] = min(
                dp[i - 1][j] + del_cost,
                dp[i][j - 1] + ins_cost,
                dp[i - 1][j - 1] + cost,
            )
    return dp[H][R]


def levenshtein(seq_a: list, seq_b: list) -> int:
    """Standard Levenshtein distance (unit costs)."""
    H, R = len(seq_a), len(seq_b)
    dp = list(range(R + 1))
    for i in range(1, H + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, R + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[j] = min(prev[j] + 1, dp[j - 1] + 1, prev[j - 1] + cost)
    return dp[R]


# Metrics

def compute_metrics(
    preds: list[list[str]],
    refs: list[list[str]],
    seen_flags: list[bool | None] = None,
) -> dict:
    """Compute test metrics and optional seen/unseen breakdowns."""
    assert len(preds) == len(refs)
    n = len(preds)
    if n == 0:
        return {}

    exact_matches = 0
    total_per_num = 0.0
    total_per_den = 0
    total_wed = 0.0
    total_lev = 0
    total_lev_den = 0

    per_by_group: dict[str, list[float]] = defaultdict(list)
    em_by_group: dict[str, list[int]] = defaultdict(list)

    for i, (pred, ref) in enumerate(zip(preds, refs)):
        em = int(pred == ref)
        exact_matches += em

        per_num = levenshtein(pred, ref)
        per_den = max(len(ref), 1)
        per = per_num / per_den

        wed = weighted_edit_distance(pred, ref)
        lev_char = levenshtein(
            list("".join(pred)), list("".join(ref))
        )

        total_per_num += per_num
        total_per_den += per_den
        total_wed += wed
        total_lev += lev_char
        total_lev_den += max(len("".join(ref)), 1)

        if seen_flags is not None and seen_flags[i] is not None:
            group = "seen" if seen_flags[i] else "unseen"
            per_by_group[group].append(per)
            em_by_group[group].append(em)

    metrics = {
        "exact_match": exact_matches / n,
        "per": total_per_num / max(total_per_den, 1),
        "weighted_edit_distance_mean": total_wed / n,
        "char_levenshtein_mean": total_lev / n,
        "n": n,
    }

    if per_by_group:
        for group in ["seen", "unseen"]:
            if group in per_by_group:
                metrics[f"per_{group}"] = sum(per_by_group[group]) / len(per_by_group[group])
                metrics[f"exact_match_{group}"] = sum(em_by_group[group]) / len(em_by_group[group])
                metrics[f"n_{group}"] = len(per_by_group[group])

    return metrics


# Error analysis

def error_analysis(
    preds: list[list[str]],
    refs: list[list[str]],
    graphemes: list[str],
    seen_flags: list[bool | None] = None,
    n_hardest: int = 20,
    n_substitution_pairs: int = 20,
) -> dict:
    """Collect hardest examples and frequent edit operations."""
    records = []
    substitutions: dict[tuple[str, str], int] = defaultdict(int)
    insertions: dict[str, int] = defaultdict(int)
    deletions: dict[str, int] = defaultdict(int)

    for pred, ref, g, seen in zip(
        preds, refs, graphemes,
        seen_flags if seen_flags else [None] * len(preds)
    ):
        lev = levenshtein(pred, ref)
        per = lev / max(len(ref), 1)
        records.append({
            "grapheme": g,
            "pred": " ".join(pred),
            "ref": " ".join(ref),
            "levenshtein": lev,
            "per": per,
            "exact_match": pred == ref,
            "seen_in_train": seen,
        })

        ops = _edit_ops(pred, ref)
        for op, hp, rp in ops:
            if op == "sub":
                substitutions[(hp, rp)] += 1
            elif op == "ins":
                insertions[rp] += 1
            elif op == "del":
                deletions[hp] += 1

    records.sort(key=lambda x: x["levenshtein"], reverse=True)
    hardest = records[:n_hardest]

    top_subs = sorted(substitutions.items(), key=lambda x: x[1], reverse=True)[:n_substitution_pairs]
    top_ins = sorted(insertions.items(), key=lambda x: x[1], reverse=True)[:10]
    top_del = sorted(deletions.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "hardest_examples": hardest,
        "top_substitutions": [
            {"hyp": k[0], "ref": k[1], "count": v} for k, v in top_subs
        ],
        "top_insertions": [{"phone": k, "count": v} for k, v in top_ins],
        "top_deletions": [{"phone": k, "count": v} for k, v in top_del],
    }


def _edit_ops(hyp: list[str], ref: list[str]) -> list[tuple[str, str | None, str | None]]:
    """Backtrack through the Levenshtein table."""
    H, R = len(hyp), len(ref)
    dp = [[0] * (R + 1) for _ in range(H + 1)]
    for i in range(H + 1):
        dp[i][0] = i
    for j in range(R + 1):
        dp[0][j] = j
    for i in range(1, H + 1):
        for j in range(1, R + 1):
            cost = 0 if hyp[i - 1] == ref[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    ops = []
    i, j = H, R
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if hyp[i - 1] == ref[j - 1] else 1):
            op = "match" if hyp[i - 1] == ref[j - 1] else "sub"
            ops.append((op, hyp[i - 1], ref[j - 1]))
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", hyp[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, ref[j - 1]))
            j -= 1

    return ops[::-1]


# Pretty-print

def print_metrics(metrics: dict, title: str = "Results") -> None:
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<35} {v:.4f}")
        else:
            print(f"  {k:<35} {v}")
    print()
