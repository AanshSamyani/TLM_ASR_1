#!/usr/bin/env bash
# =============================================================
# Experiment 2: TTL vs Tent vs Base
#
# Head-to-head comparison of adaptation methods on TEDLIUM
# (domain-shifted TED talks where adaptation should help).
#
# Quick run:  add  --max_samples 100  to each command
# =============================================================
set -euo pipefail
MODEL="openai/whisper-small"
DATASET="tedlium"
GPU_FLAG=()
if [[ "${1:-}" == "--gpu" && -n "${2:-}" ]]; then
    GPU_FLAG=(--gpu "$2")
fi

echo "========================================"
echo "  Dataset: $DATASET"
echo "========================================"

# --- baseline (no adaptation) ---
uv run python run_experiment.py "${GPU_FLAG[@]}" \
    --method base \
    --model "$MODEL" \
    --eval_dataset "$DATASET" \
    --tag exp2

# --- Tent (entropy minimisation, LayerNorm params) ---
uv run python run_experiment.py "${GPU_FLAG[@]}" \
    --method tent \
    --model "$MODEL" \
    --adapt_dataset "$DATASET" \
    --eval_dataset "$DATASET" \
    --tent_lr 1e-3 \
    --tag exp2

# --- TTL (CE on pseudo-labels, LoRA, no sample selection) ---
uv run python run_experiment.py "${GPU_FLAG[@]}" \
    --method ttl \
    --model "$MODEL" \
    --adapt_dataset "$DATASET" \
    --eval_dataset "$DATASET" \
    --lora_rank 8 \
    --lr 5e-5 \
    --tag exp2

echo ""
echo "Experiment 2 finished.  Results in results/"
