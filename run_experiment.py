#!/usr/bin/env python3
"""Main entry-point for TTL-ASR experiments.

Usage examples
--------------
# Baseline (no adaptation)
python run_experiment.py --method base --eval_dataset librispeech_other

# TTL with sample selection
python run_experiment.py --method ttl --adapt_dataset tedlium \
    --eval_dataset tedlium --sample_selection --lora_rank 8

# Tent baseline
python run_experiment.py --method tent --adapt_dataset tedlium \
    --eval_dataset tedlium

# Quick smoke test (50 samples)
python run_experiment.py --method ttl --adapt_dataset librispeech_other \
    --eval_dataset librispeech_other --max_samples 50
"""

import argparse
import json
import math
import os
import sys
import time

# Set CUDA_VISIBLE_DEVICES before importing torch so only the chosen GPU is visible.
for _i, _arg in enumerate(sys.argv):
    if _arg == "--gpu" and _i + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[_i + 1]
        break

import torch

from src.models import load_whisper, apply_lora, get_layernorm_params
from src.data import load_asr_dataset, create_dataloader
from src.eval_utils import evaluate
from src.ttl import TTLAdapter
from src.tent import TentAdapter
from src.sample_selection import SampleSelector


def parse_args():
    p = argparse.ArgumentParser(description="TTL-ASR experiments")

    # method
    p.add_argument(
        "--method",
        choices=["base", "ttl", "tent"],
        required=True,
        help="Adaptation method (base = no adaptation).",
    )

    # model
    p.add_argument(
        "--model",
        default="openai/whisper-small",
        help="HuggingFace Whisper model name.",
    )
    p.add_argument("--language", default="en")

    # datasets
    p.add_argument(
        "--adapt_dataset",
        default=None,
        choices=["librispeech_clean", "librispeech_other", "tedlium"],
        help="Dataset to adapt on (required for ttl/tent).",
    )
    p.add_argument(
        "--eval_dataset",
        required=True,
        choices=["librispeech_clean", "librispeech_other", "tedlium"],
        help="Dataset to evaluate on.",
    )
    p.add_argument("--max_samples", type=int, default=None, help="Cap samples (for quick runs).")
    p.add_argument("--max_eval_samples", type=int, default=None, help="Cap eval samples.")

    # TTL / LoRA
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument(
        "--lora_targets",
        nargs="+",
        default=["q_proj", "v_proj"],
        help="Module names to apply LoRA to.",
    )
    p.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    p.add_argument("--adapt_epochs", type=int, default=1)

    # sample selection
    p.add_argument("--sample_selection", action="store_true")
    p.add_argument("--lambda_val", type=float, default=0.10)
    p.add_argument(
        "--p0",
        type=float,
        default=None,
        help="Perplexity threshold for sample selection (default e^3).",
    )

    # Tent-specific
    p.add_argument("--tent_lr", type=float, default=1e-3, help="LR for Tent.")

    # misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="results")
    p.add_argument("--tag", default="", help="Optional tag appended to result filename.")
    p.add_argument("--batch_size", type=int, default=16, help="Batch size for adapt/eval.")
    p.add_argument("--fp16", action="store_true", help="Use float16 for inference.")
    p.add_argument(
        "--gpu",
        default=None,
        help="GPU index to use (e.g. 0, 1, 2, 3). Sets CUDA_VISIBLE_DEVICES.",
    )

    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    if args.method in ("ttl", "tent") and args.adapt_dataset is None:
        args.adapt_dataset = args.eval_dataset
        print(f"--adapt_dataset not set; defaulting to eval_dataset ({args.eval_dataset})")

    # ------------------------------------------------------------------ model
    print(f"\nLoading model: {args.model}")
    model, processor = load_whisper(args.model, device)
    if args.fp16:
        model = model.half()

    # ------------------------------------------------------------------ adapt
    adapt_stats = None
    t0 = time.time()

    if args.method == "ttl":
        model = apply_lora(
            model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            target_modules=args.lora_targets,
        )
        selector = (
            SampleSelector(args.lambda_val, args.p0)
            if args.sample_selection
            else None
        )
        adapter = TTLAdapter(
            model, processor, lr=args.lr,
            sample_selector=selector, device=device,
            language=args.language,
        )
        adapt_ds, _ = load_asr_dataset(args.adapt_dataset, args.max_samples)
        # Sample selection requires per-sample perplexity, so force batch_size=1
        adapt_bs = 1 if args.sample_selection else args.batch_size
        if args.sample_selection and args.batch_size > 1:
            print("Note: sample selection requires batch_size=1 for adaptation")
        adapt_loader = create_dataloader(adapt_ds, processor, batch_size=adapt_bs)
        print(f"\nAdapting with TTL on {args.adapt_dataset} …")
        adapt_stats = adapter.adapt(adapt_loader, n_epochs=args.adapt_epochs)

    elif args.method == "tent":
        ln_params = get_layernorm_params(model)
        adapter = TentAdapter(
            model, processor, lr=args.tent_lr,
            ln_params=ln_params, device=device,
            language=args.language,
        )
        adapt_ds, _ = load_asr_dataset(args.adapt_dataset, args.max_samples)
        adapt_loader = create_dataloader(adapt_ds, processor, batch_size=args.batch_size)
        print(f"\nAdapting with Tent on {args.adapt_dataset} ��")
        adapt_stats = adapter.adapt(adapt_loader, n_epochs=args.adapt_epochs)

    adapt_time = time.time() - t0

    # ------------------------------------------------------------------ eval
    max_eval = args.max_eval_samples or args.max_samples
    eval_ds, _ = load_asr_dataset(args.eval_dataset, max_eval)
    eval_loader = create_dataloader(eval_ds, processor, batch_size=args.batch_size)

    print(f"\nEvaluating on {args.eval_dataset} …")
    t1 = time.time()
    word_error_rate, preds, refs = evaluate(
        model, processor, eval_loader, device, language=args.language,
    )
    eval_time = time.time() - t1

    print(f"\n{'='*50}")
    print(f"Method:  {args.method}")
    print(f"Model:   {args.model}")
    print(f"Adapt:   {args.adapt_dataset or 'n/a'}")
    print(f"Eval:    {args.eval_dataset}")
    print(f"WER:     {word_error_rate:.4f}  ({word_error_rate*100:.2f}%)")
    if adapt_stats:
        print(f"Adapted: {adapt_stats['adapted_samples']}/{adapt_stats['total_samples']} samples")
        print(f"Skipped: {adapt_stats['skipped_samples']}")
        if adapt_stats["perplexities"]:
            mean_ppl = sum(adapt_stats["perplexities"]) / len(adapt_stats["perplexities"])
            print(f"Mean PPL: {mean_ppl:.2f}")
    print(f"Adapt time: {adapt_time:.1f}s | Eval time: {eval_time:.1f}s")
    print(f"{'='*50}\n")

    # ------------------------------------------------------------------ save
    result = {
        "method": args.method,
        "model": args.model,
        "adapt_dataset": args.adapt_dataset,
        "eval_dataset": args.eval_dataset,
        "wer": word_error_rate,
        "config": {
            "lora_rank": args.lora_rank if args.method == "ttl" else None,
            "lora_alpha": args.lora_alpha if args.method == "ttl" else None,
            "lr": args.lr if args.method == "ttl" else args.tent_lr,
            "sample_selection": args.sample_selection,
            "p0": args.p0,
            "lambda_val": args.lambda_val,
            "adapt_epochs": args.adapt_epochs,
            "max_samples": args.max_samples,
            "seed": args.seed,
        },
        "timing": {
            "adapt_seconds": round(adapt_time, 1),
            "eval_seconds": round(eval_time, 1),
        },
    }
    if adapt_stats:
        result["adaptation_stats"] = {
            "total_samples": adapt_stats["total_samples"],
            "adapted_samples": adapt_stats["adapted_samples"],
            "skipped_samples": adapt_stats["skipped_samples"],
            "mean_ppl": (
                round(sum(adapt_stats["perplexities"]) / len(adapt_stats["perplexities"]), 2)
                if adapt_stats["perplexities"]
                else None
            ),
            "mean_loss": (
                round(sum(adapt_stats["losses"]) / len(adapt_stats["losses"]), 4)
                if adapt_stats["losses"]
                else None
            ),
        }

    tag = f"_{args.tag}" if args.tag else ""
    sel_tag = "_sel" if args.sample_selection else ""
    p0_tag = f"_p0{args.p0:.1f}" if args.p0 and args.sample_selection else ""
    fname = f"{args.method}_{args.eval_dataset}{sel_tag}{p0_tag}{tag}.json"
    out_path = os.path.join(args.output_dir, fname)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
