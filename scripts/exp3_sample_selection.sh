#!/usr/bin/env bash
# =============================================================
# Experiment 3: TTL ablation — perplexity methods & sample selection
#
# Compares Base, Tent, and TTL variants on TEDLIUM (domain-shifted):
#   (a) TTL with different loss/perplexity methods (ce, entropy, gen)
#   (b) TTL with sample selection at different P0 thresholds
#
# Usage:  bash scripts/exp3_sample_selection.sh --gpu 0 --batch_size 32
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
echo "  Exp3: TTL ablation on $DATASET"
echo "========================================"

# --- 1. baseline (no adaptation) ---
echo "--- Base (no adaptation) ---"
uv run python run_experiment.py "${EXTRA_FLAGS[@]}" \
    --method base \
    --model "$MODEL" \
    --eval_dataset "$DATASET" \
    --tag exp3

# --- 2. Tent (entropy on LayerNorm params) ---
echo "--- Tent ---"
uv run python run_experiment.py "${EXTRA_FLAGS[@]}" \
    --method tent \
    --model "$MODEL" \
    --adapt_dataset "$DATASET" \
    --eval_dataset "$DATASET" \
    --tent_lr 1e-3 \
    --tag exp3

# --- 3. TTL with CE loss (broken baseline — for comparison) ---
echo "--- TTL (CE loss, no selection) ---"
uv run python run_experiment.py "${EXTRA_FLAGS[@]}" \
    --method ttl \
    --ppl_method ce \
    --model "$MODEL" \
    --adapt_dataset "$DATASET" \
    --eval_dataset "$DATASET" \
    --lora_rank 8 \
    --lr 5e-5 \
    --tag exp3

# --- 4. TTL with entropy loss (recommended, no selection) ---
echo "--- TTL (entropy loss, no selection) ---"
uv run python run_experiment.py "${EXTRA_FLAGS[@]}" \
    --method ttl \
    --ppl_method entropy \
    --model "$MODEL" \
    --adapt_dataset "$DATASET" \
    --eval_dataset "$DATASET" \
    --lora_rank 8 \
    --lr 5e-5 \
    --tag exp3

# --- 5. TTL entropy with sample selection — P0 sweep ---
# exp(entropy) values for Whisper on domain-shifted data are typically 1.1–7.0,
# so we sweep P0 in that range.
for P0 in 1.5 2.5 5.0; do
    echo "--- TTL entropy + selection, P0=$P0 ---"
    uv run python run_experiment.py "${EXTRA_FLAGS[@]}" \
        --method ttl \
        --ppl_method entropy \
        --model "$MODEL" \
        --adapt_dataset "$DATASET" \
        --eval_dataset "$DATASET" \
        --lora_rank 8 \
        --lr 5e-5 \
        --sample_selection \
        --p0 "$P0" \
        --tag exp3
done

# --- 6. TTL gen (generation-time perplexity) with sample selection — P0 sweep ---
# Generation perplexity is typically higher (2–50+), so use wider P0 range.
for P0 in 3.0 10.0 30.0; do
    echo "--- TTL gen + selection, P0=$P0 ---"
    uv run python run_experiment.py "${EXTRA_FLAGS[@]}" \
        --method ttl \
        --ppl_method gen \
        --model "$MODEL" \
        --adapt_dataset "$DATASET" \
        --eval_dataset "$DATASET" \
        --lora_rank 8 \
        --lr 5e-5 \
        --sample_selection \
        --p0 "$P0" \
        --tag exp3
done

echo ""
echo "Experiment 3 finished.  Results in results/"
