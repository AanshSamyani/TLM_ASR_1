#!/usr/bin/env python3
"""Read result JSON files and produce comparison plots.

Usage:
    python analysis/plot_results.py --input_dir results --output_dir results
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load_results(input_dir: str):
    """Load all JSON result files into a list of dicts."""
    results = []
    for path in sorted(glob.glob(os.path.join(input_dir, "*.json"))):
        with open(path) as f:
            data = json.load(f)
            data["_file"] = os.path.basename(path)
            results.append(data)
    return results


# ------------------------------------------------------------------
# Plot 1 — Exp 1 & 2: grouped bar chart  (method × dataset → WER)
# ------------------------------------------------------------------
def plot_method_comparison(results, output_dir: str):
    """Bar chart comparing WER across methods and datasets."""
    # Collect data: { (method, eval_dataset): wer }
    table = {}
    for r in results:
        key = (r["method"], r["eval_dataset"])
        # Prefer exp2 tag if available, else any
        tag = r["_file"]
        if key not in table or "exp2" in tag:
            table[key] = r["wer"]

    if not table:
        print("No results to plot for method comparison.")
        return

    datasets = sorted({k[1] for k in table})
    methods = sorted({k[0] for k in table})
    n_methods = len(methods)
    x = np.arange(len(datasets))
    width = 0.8 / max(n_methods, 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, method in enumerate(methods):
        wers = [table.get((method, d), 0) * 100 for d in datasets]
        bars = ax.bar(x + i * width, wers, width, label=method.upper())
        for bar, w in zip(bars, wers):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{w:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_ylabel("WER (%)")
    ax.set_title("Method Comparison (WER)")
    ax.set_xticks(x + width * (n_methods - 1) / 2)
    ax.set_xticklabels([d.replace("_", " ") for d in datasets])
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    path = os.path.join(output_dir, "method_comparison.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


# ------------------------------------------------------------------
# Plot 2 — Exp 3: P0 sweep line chart
# ------------------------------------------------------------------
def plot_p0_sweep(results, output_dir: str):
    """Line plot of WER vs P0 threshold."""
    # Collect TTL results with different P0 values
    points = []  # (p0, wer, adapted, total)
    no_sel_wer = None
    for r in results:
        if r["method"] != "ttl":
            continue
        tag = r.get("_file", "")
        if "exp3_nosel" in tag:
            no_sel_wer = r["wer"]
            continue
        if "exp3" not in tag:
            continue
        p0 = r.get("config", {}).get("p0")
        if p0 is None:
            continue
        adapted = r.get("adaptation_stats", {}).get("adapted_samples", 0)
        total = r.get("adaptation_stats", {}).get("total_samples", 1)
        points.append((p0, r["wer"], adapted, total))

    if not points:
        print("No P0-sweep results to plot.")
        return

    points.sort()
    p0s = [p[0] for p in points]
    wers = [p[1] * 100 for p in points]
    adapt_frac = [p[2] / max(p[3], 1) * 100 for p in points]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color1 = "tab:blue"
    ax1.plot(p0s, wers, "o-", color=color1, label="WER")
    if no_sel_wer is not None:
        ax1.axhline(no_sel_wer * 100, color=color1, linestyle="--", alpha=0.5,
                     label=f"No selection ({no_sel_wer*100:.1f}%)")
    ax1.set_xlabel("Perplexity threshold P₀")
    ax1.set_ylabel("WER (%)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xscale("log")

    ax2 = ax1.twinx()
    color2 = "tab:orange"
    ax2.bar(p0s, adapt_frac, width=[p * 0.3 for p in p0s], alpha=0.3,
            color=color2, label="% adapted")
    ax2.set_ylabel("% samples adapted", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.set_title("Sample Selection: P₀ Threshold Sweep")
    plt.tight_layout()
    path = os.path.join(output_dir, "p0_sweep.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


# ------------------------------------------------------------------
# Plot 3 — Adaptation statistics summary
# ------------------------------------------------------------------
def plot_adaptation_stats(results, output_dir: str):
    """Bar chart of adapted/skipped sample counts."""
    adapt_results = [
        r for r in results
        if r["method"] in ("ttl", "tent") and r.get("adaptation_stats")
    ]
    if not adapt_results:
        print("No adaptation stats to plot.")
        return

    labels = []
    adapted = []
    skipped = []
    for r in adapt_results:
        s = r["adaptation_stats"]
        lbl = f"{r['method'].upper()}\n{r['eval_dataset']}"
        if r.get("config", {}).get("sample_selection"):
            lbl += "\n(sel)"
        labels.append(lbl)
        adapted.append(s["adapted_samples"])
        skipped.append(s["skipped_samples"])

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))
    ax.bar(x, adapted, label="Adapted")
    ax.bar(x, skipped, bottom=adapted, label="Skipped")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Number of samples")
    ax.set_title("Adaptation Sample Counts")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "adaptation_stats.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


# ------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default="results")
    p.add_argument("--output_dir", default="results")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = load_results(args.input_dir)
    if not results:
        print(f"No JSON results found in {args.input_dir}/")
        return

    print(f"Loaded {len(results)} result files.\n")

    plot_method_comparison(results, args.output_dir)
    plot_p0_sweep(results, args.output_dir)
    plot_adaptation_stats(results, args.output_dir)

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Method':<10} {'Adapt':<20} {'Eval':<20} {'WER':>8}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: (x["eval_dataset"], x["method"])):
        adapt = r.get("adapt_dataset") or "—"
        print(f"{r['method']:<10} {adapt:<20} {r['eval_dataset']:<20} {r['wer']*100:>7.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
