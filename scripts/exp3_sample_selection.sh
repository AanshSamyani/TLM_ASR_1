#!/usr/bin/env bash
# =============================================================
# Experiment 3 (de-risk): Sample-selection ablation
#
# Varies the perplexity threshold P0 to test whether the
# sample-selection strategy actually helps.  Also includes
# a run with NO selection for comparison.
#
# Usage:  bash scripts/exp3_sample_selection.sh --gpu 0 --batch_size 32
# =============================================================
set -euo pipefail
MODEL="openai/whisper-small"
DATASET="tedlium"          # domain-shifted dataset

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
echo "  Sample-selection ablation on $DATASET"
echo "========================================"

# --- no selection (all samples, weight=1) ---
uv run python run_experiment.py "${EXTRA_FLAGS[@]}" \
    --method ttl \
    --model "$MODEL" \
    --adapt_dataset "$DATASET" \
    --eval_dataset "$DATASET" \
    --lora_rank 8 \
    --lr 5e-5 \
    --tag exp3_nosel

# --- sweep P0 thresholds ---
# e^2 ≈ 7.39,  e^3 ≈ 20.09,  e^4 ≈ 54.60,  e^5 ≈ 148.41
for P0 in 7.39 20.09 54.60 148.41; do
    echo "--- P0 = $P0 ---"
    uv run python run_experiment.py "${EXTRA_FLAGS[@]}" \
        --method ttl \
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
