#!/usr/bin/env bash
# =============================================================
# Experiment 2: TTL vs Tent vs Base
#
# Head-to-head comparison of adaptation methods on TEDLIUM.
# Includes a learning rate sweep for TTL since entropy
# minimisation needs higher LR than the TLM paper's 5e-5.
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

# --- Tent (entropy minimisation, LayerNorm params, lr=1e-3) ---
uv run python run_experiment.py "${EXTRA_FLAGS[@]}" \
    --method tent \
    --model "$MODEL" \
    --adapt_dataset "$DATASET" \
    --eval_dataset "$DATASET" \
    --tent_lr 1e-3 \
    --tag exp2

# --- TTL entropy — learning rate sweep ---
# Tent uses lr=1e-3; the old TTL default (5e-5) is too low for entropy loss.
for LR in 1e-4 5e-4 1e-3; do
    echo "--- TTL entropy, lr=$LR ---"
    uv run python run_experiment.py "${EXTRA_FLAGS[@]}" \
        --method ttl \
        --ppl_method entropy \
        --model "$MODEL" \
        --adapt_dataset "$DATASET" \
        --eval_dataset "$DATASET" \
        --lora_rank 8 \
        --lr "$LR" \
        --tag "exp2_lr${LR}"
done

# --- TTL entropy with expanded LoRA targets (attn + FF layers) ---
# More LoRA targets = more capacity to reduce entropy
echo "--- TTL entropy, lr=5e-4, expanded LoRA targets ---"
uv run python run_experiment.py "${EXTRA_FLAGS[@]}" \
    --method ttl \
    --ppl_method entropy \
    --model "$MODEL" \
    --adapt_dataset "$DATASET" \
    --eval_dataset "$DATASET" \
    --lora_rank 8 \
    --lr 5e-4 \
    --lora_targets q_proj v_proj k_proj out_proj fc1 fc2 \
    --tag exp2_expanded

echo ""
echo "Experiment 2 finished.  Results in results/"
