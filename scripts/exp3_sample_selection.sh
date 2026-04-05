#!/usr/bin/env bash
# =============================================================
# Experiment 3 (de-risk): Sample-selection ablation
#
# Varies the perplexity threshold P0 to test whether the
# sample-selection strategy actually helps.  Also includes
# a run with NO selection for comparison.
#
# Quick run:  add  --max_samples 200
# =============================================================
set -euo pipefail
MODEL="openai/whisper-small"
DATASET="tedlium"          # domain-shifted dataset
GPU_FLAG=()
if [[ "${1:-}" == "--gpu" && -n "${2:-}" ]]; then
    GPU_FLAG=(--gpu "$2")
fi

echo "========================================"
echo "  Sample-selection ablation on $DATASET"
echo "========================================"

# --- no selection (all samples, weight=1) ---
uv run python run_experiment.py "${GPU_FLAG[@]}" \
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
    uv run python run_experiment.py "${GPU_FLAG[@]}" \
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
