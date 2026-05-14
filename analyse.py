"""Plot and summarise completed runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[Analyse] matplotlib not available; skipping plots.")

try:
    import pandas as pd
    HAS_PD = True
except ImportError:
    HAS_PD = False


# Load helpers

def load_history(output_dir: Path) -> dict | None:
    p = output_dir / "history.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def load_metrics(output_dir: Path) -> dict | None:
    p = output_dir / "test_metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def load_analysis(output_dir: Path) -> dict | None:
    p = output_dir / "error_analysis.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def load_predictions(output_dir: Path) -> "pd.DataFrame | None":
    if not HAS_PD:
        return None
    p = output_dir / "test_predictions.tsv"
    if not p.exists():
        return None
    return pd.read_csv(p, sep="\t")


# Training curves

def plot_training_curves(history: dict, output_dir: Path, label: str = "") -> None:
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train", color="steelblue")
    axes[0].plot(epochs, history["val_loss"], label="Val", color="tomato")
    axes[0].set_title(f"Loss{' — '+label if label else ''}")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["val_exact_match"], color="forestgreen")
    axes[1].set_title("Validation Exact Match")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("EM")
    axes[1].set_ylim(0, 1); axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, history["val_per"], color="darkorange")
    axes[2].set_title("Validation PER")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("PER")
    axes[2].set_ylim(0, 1); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out = output_dir / "training_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[Analyse] Saved {out}")


# Memorisation vs generalisation bar chart

def plot_mem_vs_gen(metrics: dict, output_dir: Path, label: str = "") -> None:
    if not HAS_MPL:
        return
    keys_needed = {"exact_match_seen", "exact_match_unseen", "per_seen", "per_unseen"}
    if not keys_needed.issubset(metrics.keys()):
        print("[Analyse] No memorisation/generalisation breakdown in metrics; skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    groups = ["Seen (memorisation)", "Unseen (generalisation)"]
    em_vals = [metrics["exact_match_seen"], metrics["exact_match_unseen"]]
    per_vals = [metrics["per_seen"], metrics["per_unseen"]]
    colors = ["steelblue", "tomato"]

    bars = axes[0].bar(groups, em_vals, color=colors)
    axes[0].set_ylim(0, 1)
    axes[0].set_title(f"Exact Match{' — '+label if label else ''}")
    axes[0].set_ylabel("EM")
    for bar, v in zip(bars, em_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)

    bars2 = axes[1].bar(groups, per_vals, color=colors)
    axes[1].set_ylim(0, max(per_vals) * 1.2 + 0.05)
    axes[1].set_title(f"PER{' — '+label if label else ''}")
    axes[1].set_ylabel("PER (lower = better)")
    for bar, v in zip(bars2, per_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.002, f"{v:.3f}", ha="center", fontsize=10)

    n_seen = metrics.get("n_seen", "?")
    n_unseen = metrics.get("n_unseen", "?")
    fig.suptitle(f"Memorisation (n={n_seen}) vs Generalisation (n={n_unseen})", fontsize=11)

    plt.tight_layout()
    out = output_dir / "mem_vs_gen.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[Analyse] Saved {out}")


# Substitution heatmap

def plot_substitution_pairs(analysis: dict, output_dir: Path, top_n: int = 20) -> None:
    if not HAS_MPL:
        return
    subs = analysis.get("top_substitutions", [])[:top_n]
    if not subs:
        return

    pairs = [(s["hyp"], s["ref"], s["count"]) for s in subs]
    labels = [f"{h}→{r}" for h, r, _ in pairs]
    counts = [c for _, _, c in pairs]

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.35 + 1)))
    bars = ax.barh(range(len(labels)), counts, color="mediumpurple")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title(f"Top {len(labels)} Substitution Errors")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out = output_dir / "substitution_errors.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[Analyse] Saved {out}")


# Cross-run comparison

def compare_runs(run_dirs: list[Path]) -> None:
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for run_dir in run_dirs:
        history = load_history(run_dir)
        if history is None:
            continue
        label = run_dir.name
        epochs = range(1, len(history["val_loss"]) + 1)
        axes[0].plot(epochs, history["val_loss"], label=label)
        axes[1].plot(epochs, history["val_exact_match"], label=label)

    axes[0].set_title("Validation Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_title("Validation Exact Match"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[0].set_xlabel("Epoch"); axes[1].set_xlabel("Epoch")

    plt.tight_layout()
    out = run_dirs[0].parent / "comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[Analyse] Comparison saved to {out}")


# Print summary table

def print_summary_table(run_dirs: list[Path]) -> None:
    rows = []
    for d in run_dirs:
        m = load_metrics(d)
        if m is None:
            continue
        row = {
            "run": d.name,
            "EM": f"{m.get('exact_match', float('nan')):.3f}",
            "PER": f"{m.get('per', float('nan')):.3f}",
            "WED": f"{m.get('weighted_edit_distance_mean', float('nan')):.3f}",
            "EM_seen": f"{m.get('exact_match_seen', float('nan')):.3f}",
            "EM_unseen": f"{m.get('exact_match_unseen', float('nan')):.3f}",
            "PER_seen": f"{m.get('per_seen', float('nan')):.3f}",
            "PER_unseen": f"{m.get('per_unseen', float('nan')):.3f}",
        }
        rows.append(row)

    if not rows:
        print("No metrics found.")
        return

    col_widths = {k: max(len(k), max(len(r[k]) for r in rows)) for k in rows[0]}
    header = " | ".join(k.ljust(col_widths[k]) for k in col_widths)
    sep = "-" * len(header)
    print(f"\n{'='*len(header)}")
    print(header)
    print(sep)
    for row in rows:
        print(" | ".join(row[k].ljust(col_widths[k]) for k in col_widths))
    print(f"{'='*len(header)}\n")


# CLI

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=Path, nargs="+", required=True)
    p.add_argument("--compare", action="store_true")
    args = p.parse_args()

    for d in args.output_dir:
        print(f"\n{'='*60}")
        print(f"[Analyse] Run: {d}")

        history = load_history(d)
        metrics = load_metrics(d)
        analysis = load_analysis(d)

        if history:
            plot_training_curves(history, d, label=d.name)
        if metrics:
            plot_mem_vs_gen(metrics, d, label=d.name)
            from evaluate import print_metrics
            print_metrics(metrics, title=f"Test metrics: {d.name}")
        if analysis:
            plot_substitution_pairs(analysis, d)
            print("\nTop substitutions:")
            for s in analysis["top_substitutions"][:10]:
                print(f"  {s['hyp']:6s} -> {s['ref']:6s}  x{s['count']}")

    if args.compare and len(args.output_dir) > 1:
        compare_runs(args.output_dir)

    print_summary_table(args.output_dir)


if __name__ == "__main__":
    main()
