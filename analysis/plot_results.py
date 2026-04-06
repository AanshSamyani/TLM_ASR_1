#!/usr/bin/env python3
"""Read result JSON files and produce comparison plots.

Usage:
    python analysis/plot_results.py --input_dir results --output_dir plots

Plot types (auto-detected from result tags):
  - method_comparison.png   — grouped bar chart: WER per method per dataset
  - p0_sweep.png            — line plot: WER vs P0 threshold
  - adaptation_stats.png    — adapted/skipped sample counts
  - noise_severity.png      — WER vs SNR for each method (exp4)
  - model_scaling.png       — WER across Whisper sizes (exp5)
  - lora_rank_sweep.png     — WER vs LoRA rank (exp6)
  - lora_lr_sweep.png       — WER vs learning rate (exp6)
  - lora_placement.png      — WER by LoRA placement (exp6)
  - entropy_histogram.png   — per-sample entropy distribution
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "base": "#7f8c8d",
    "tent": "#e74c3c",
    "suta": "#f39c12",
    "ttl": "#2ecc71",
}
METHOD_ORDER = ["base", "tent", "suta", "ttl"]


def load_results(input_dir: str):
    """Load all JSON result files into a list of dicts."""
    results = []
    for path in sorted(glob.glob(os.path.join(input_dir, "*.json"))):
        with open(path) as f:
            data = json.load(f)
            data["_file"] = os.path.basename(path)
            results.append(data)
    return results


def _method_color(method):
    return COLORS.get(method, "#3498db")


# ------------------------------------------------------------------
# Plot 1 — Method comparison: grouped bar chart (method x dataset -> WER)
# ------------------------------------------------------------------
def plot_method_comparison(results, output_dir: str):
    """Bar chart comparing WER across methods and datasets."""
    table = {}
    for r in results:
        if r.get("noise_type", "none") != "none":
            continue
        key = (r["method"], r["eval_dataset"])
        tag = r["_file"]
        if key not in table or "exp2" in tag:
            table[key] = r["wer"]

    if not table:
        print("No results to plot for method comparison.")
        return

    datasets = sorted({k[1] for k in table})
    methods = [m for m in METHOD_ORDER if any(k[0] == m for k in table)]
    n_methods = len(methods)
    x = np.arange(len(datasets))
    width = 0.8 / max(n_methods, 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, method in enumerate(methods):
        wers = [table.get((method, d), 0) * 100 for d in datasets]
        bars = ax.bar(x + i * width, wers, width, label=method.upper(),
                      color=_method_color(method))
        for bar, w in zip(bars, wers):
            if w > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{w:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("WER (%)")
    ax.set_title("Method Comparison (WER)")
    ax.set_xticks(x + width * (n_methods - 1) / 2)
    ax.set_xticklabels([d.replace("_", " ").title() for d in datasets])
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    path = os.path.join(output_dir, "method_comparison.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


# ------------------------------------------------------------------
# Plot 2 — P0 sweep line chart
# ------------------------------------------------------------------
def plot_p0_sweep(results, output_dir: str):
    """Line plot of WER vs P0 threshold."""
    points = []
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
    ax1.set_xlabel("Perplexity threshold P0")
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
    ax1.set_title("Sample Selection: P0 Threshold Sweep")
    plt.tight_layout()
    path = os.path.join(output_dir, "p0_sweep.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


# ------------------------------------------------------------------
# Plot 3 — Adaptation statistics
# ------------------------------------------------------------------
def plot_adaptation_stats(results, output_dir: str):
    """Bar chart of adapted/skipped sample counts."""
    adapt_results = [
        r for r in results
        if r["method"] in ("ttl", "tent", "suta") and r.get("adaptation_stats")
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
# Plot 4 — Noise severity (LibriSpeech-C): WER vs SNR
# ------------------------------------------------------------------
def plot_noise_severity(results, output_dir: str):
    """Line plot: WER vs SNR for each method, per noise type."""
    noise_results = [r for r in results if r.get("noise_type", "none") != "none"]
    if not noise_results:
        print("No noise results to plot.")
        return

    noise_types = sorted({r["noise_type"] for r in noise_results})
    fig, axes = plt.subplots(1, len(noise_types), figsize=(7 * len(noise_types), 5),
                             squeeze=False)

    for col, nt in enumerate(noise_types):
        ax = axes[0, col]
        nt_results = [r for r in noise_results if r["noise_type"] == nt]

        # Group by method
        method_data = {}
        for r in nt_results:
            m = r["method"]
            snr = r.get("noise_snr", r.get("noise_snr"))
            if snr is None:
                continue
            method_data.setdefault(m, []).append((snr, r["wer"] * 100))

        for method in METHOD_ORDER:
            if method not in method_data:
                continue
            pts = sorted(method_data[method])
            snrs = [p[0] for p in pts]
            wers = [p[1] for p in pts]
            ax.plot(snrs, wers, "o-", label=method.upper(), color=_method_color(method),
                    linewidth=2, markersize=6)

        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("WER (%)")
        ax.set_title(f"Noise Robustness — {nt.title()} Noise")
        ax.legend()
        ax.invert_xaxis()  # lower SNR = more noise = right side
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "noise_severity.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


# ------------------------------------------------------------------
# Plot 5 — Model scaling: WER across Whisper sizes
# ------------------------------------------------------------------
def plot_model_scaling(results, output_dir: str):
    """Bar chart: WER per model size per method."""
    scaling_results = [r for r in results if "exp5" in r.get("_file", "")]
    if not scaling_results:
        print("No model scaling results to plot.")
        return

    SIZE_ORDER = ["whisper-tiny", "whisper-base", "whisper-small",
                  "whisper-medium", "whisper-large-v3"]

    # Group by (method, model_short)
    table = {}
    for r in scaling_results:
        model_short = r["model"].split("/")[-1]
        table[(r["method"], model_short)] = r["wer"] * 100

    models_present = [m for m in SIZE_ORDER if any(k[1] == m for k in table)]
    methods_present = [m for m in METHOD_ORDER if any(k[0] == m for k in table)]
    n_methods = len(methods_present)
    x = np.arange(len(models_present))
    width = 0.8 / max(n_methods, 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, method in enumerate(methods_present):
        wers = [table.get((method, m), 0) for m in models_present]
        bars = ax.bar(x + i * width, wers, width, label=method.upper(),
                      color=_method_color(method))
        for bar, w in zip(bars, wers):
            if w > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{w:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("WER (%)")
    ax.set_title("Model Scaling — WER by Whisper Size (TEDLIUM)")
    ax.set_xticks(x + width * (n_methods - 1) / 2)
    ax.set_xticklabels([m.replace("whisper-", "") for m in models_present])
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    path = os.path.join(output_dir, "model_scaling.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


# ------------------------------------------------------------------
# Plot 6 — LoRA rank sweep
# ------------------------------------------------------------------
def plot_lora_rank_sweep(results, output_dir: str):
    """Line plot: WER vs LoRA rank."""
    rank_results = [r for r in results if "exp6_rank" in r.get("_file", "")]
    if not rank_results:
        print("No LoRA rank sweep results to plot.")
        return

    pts = []
    for r in rank_results:
        rank = r.get("config", {}).get("lora_rank")
        if rank is not None:
            pts.append((rank, r["wer"] * 100))
    pts.sort()

    fig, ax = plt.subplots(figsize=(7, 5))
    ranks = [p[0] for p in pts]
    wers = [p[1] for p in pts]
    ax.plot(ranks, wers, "o-", color=COLORS["ttl"], linewidth=2, markersize=8)
    for rk, w in zip(ranks, wers):
        ax.annotate(f"{w:.1f}", (rk, w), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)
    ax.set_xlabel("LoRA Rank")
    ax.set_ylabel("WER (%)")
    ax.set_title("LoRA Rank Ablation (TEDLIUM)")
    ax.set_xticks(ranks)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "lora_rank_sweep.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


# ------------------------------------------------------------------
# Plot 7 — LoRA LR sweep
# ------------------------------------------------------------------
def plot_lora_lr_sweep(results, output_dir: str):
    """Line plot: WER vs learning rate."""
    lr_results = [r for r in results if "exp6_lr" in r.get("_file", "")]
    if not lr_results:
        print("No LoRA LR sweep results to plot.")
        return

    pts = []
    for r in lr_results:
        lr = r.get("config", {}).get("lr")
        if lr is not None:
            pts.append((lr, r["wer"] * 100))
    pts.sort()

    fig, ax = plt.subplots(figsize=(7, 5))
    lrs = [p[0] for p in pts]
    wers = [p[1] for p in pts]
    ax.plot(lrs, wers, "o-", color=COLORS["ttl"], linewidth=2, markersize=8)
    for lr_val, w in zip(lrs, wers):
        ax.annotate(f"{w:.1f}", (lr_val, w), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("WER (%)")
    ax.set_title("Learning Rate Sensitivity (TEDLIUM)")
    ax.set_xscale("log")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "lora_lr_sweep.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


# ------------------------------------------------------------------
# Plot 8 — LoRA placement comparison
# ------------------------------------------------------------------
def plot_lora_placement(results, output_dir: str):
    """Bar chart: WER by LoRA placement (encoder / decoder / both)."""
    place_results = [r for r in results if "exp6_place" in r.get("_file", "")]
    if not place_results:
        print("No LoRA placement results to plot.")
        return

    placement_order = ["encoder", "decoder", "both"]
    table = {}
    for r in place_results:
        p = r.get("config", {}).get("lora_placement", "both")
        table[p] = r["wer"] * 100

    placements = [p for p in placement_order if p in table]
    wers = [table[p] for p in placements]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(placements, wers, color=[COLORS["ttl"]] * len(placements), width=0.5)
    for bar, w in zip(bars, wers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{w:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("WER (%)")
    ax.set_title("LoRA Placement Ablation (TEDLIUM)")
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    path = os.path.join(output_dir, "lora_placement.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


# ------------------------------------------------------------------
# Plot 9 — Entropy histogram
# ------------------------------------------------------------------
def plot_entropy_histogram(results, output_dir: str):
    """Histogram of per-sample entropy values from TTL adaptation."""
    all_entropies = []
    for r in results:
        if r["method"] != "ttl":
            continue
        ent = r.get("adaptation_stats", {}).get("raw_entropies", [])
        if ent:
            all_entropies.extend(ent)

    if not all_entropies:
        print("No entropy data to plot histogram.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_entropies, bins=50, color=COLORS["ttl"], alpha=0.8, edgecolor="white")
    ax.axvline(np.mean(all_entropies), color="red", linestyle="--",
               label=f"Mean = {np.mean(all_entropies):.3f}")
    ax.set_xlabel("Entropy (nats)")
    ax.set_ylabel("Count")
    ax.set_title("Per-Sample Decoder Entropy Distribution")
    ax.legend()

    # Add exp(entropy) axis on top
    ax2 = ax.twiny()
    lo, hi = ax.get_xlim()
    ax2.set_xlim(np.exp(lo), np.exp(hi))
    ax2.set_xlabel("exp(Entropy) ≈ Pseudo-perplexity")

    plt.tight_layout()
    path = os.path.join(output_dir, "entropy_histogram.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


# ------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default="results")
    p.add_argument("--output_dir", default="plots")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = load_results(args.input_dir)
    if not results:
        print(f"No JSON results found in {args.input_dir}/")
        return

    print(f"Loaded {len(results)} result files.\n")

    # Run all plotters (each is a no-op if no relevant data)
    plot_method_comparison(results, args.output_dir)
    plot_p0_sweep(results, args.output_dir)
    plot_adaptation_stats(results, args.output_dir)
    plot_noise_severity(results, args.output_dir)
    plot_model_scaling(results, args.output_dir)
    plot_lora_rank_sweep(results, args.output_dir)
    plot_lora_lr_sweep(results, args.output_dir)
    plot_lora_placement(results, args.output_dir)
    plot_entropy_histogram(results, args.output_dir)

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Method':<8} {'Model':<18} {'Adapt':<18} {'Eval':<18} {'Noise':<12} {'WER':>8}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: (x["eval_dataset"], x["method"])):
        adapt = r.get("adapt_dataset") or "-"
        model_s = r["model"].split("/")[-1]
        noise = r.get("noise_type", "none")
        if noise != "none":
            noise = f"{noise}@{r.get('noise_snr', '?')}dB"
        else:
            noise = "-"
        print(f"{r['method']:<8} {model_s:<18} {adapt:<18} "
              f"{r['eval_dataset']:<18} {noise:<12} {r['wer']*100:>7.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
