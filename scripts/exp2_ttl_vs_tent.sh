#!/usr/bin/env bash
# =============================================================
# Experiment 2: TTL vs Tent vs Base
#
# Head-to-head comparison of adaptation methods on TEDLIUM
# (domain-shifted TED talks where adaptation should help).
#
# Usage:  bash scripts/exp2_ttl_vs_tent.sh --gpu 0 --batch_size 32
# =============================================================
set -euo pipefail
MODEL="openai/whisper-small"
DATASET="tedlium"

# Parse optional flags and forward them to run_experiment.py
EXTRA_FLAGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu|--batch_size)
            EXTRA_FLAGS+=("$1" "$2"); shift 2 ;;
        *)
            echo "Unknown flag: $1"; exit 1 ;;
    esac
done

echo "========================================"
echo "  Dataset: $DATASET"
echo "========================================"

# --- baseline (no adaptation) ---
uv run python run_experiment.py "${EXTRA_FLAGS[@]}" \
    --method base \
    --model "$MODEL" \
    --eval_dataset "$DATASET" \
    --tag exp2

# --- Tent (entropy minimisation, LayerNorm params) ---
uv run python run_experiment.py "${EXTRA_FLAGS[@]}" \
    --method tent \
    --model "$MODEL" \
    --adapt_dataset "$DATASET" \
    --eval_dataset "$DATASET" \
    --tent_lr 1e-3 \
    --tag exp2

# --- TTL (CE on pseudo-labels, LoRA, no sample selection) ---
uv run python run_experiment.py "${EXTRA_FLAGS[@]}" \
    --method ttl \
    --model "$MODEL" \
    --adapt_dataset "$DATASET" \
    --eval_dataset "$DATASET" \
    --lora_rank 8 \
    --lr 5e-5 \
    --tag exp2

echo ""
echo "Experiment 2 finished.  Results in results/"
