#!/usr/bin/env bash
# =============================================================
# Experiment 5: Model scaling across Whisper sizes
#
# Tests base, tent, and ttl on TEDLIUM using:
#   whisper-tiny, whisper-base, whisper-small, whisper-medium, whisper-large-v3
#
# Usage:  bash scripts/exp5_model_scaling.sh --gpu 0 --batch_size 32
# =============================================================
set -euo pipefail
DATASET="tedlium"
TTL_LR="1e-3"

# Parse optional flags
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
echo "  Exp5: Model scaling on $DATASET"
echo "========================================"

for MODEL in openai/whisper-tiny openai/whisper-base openai/whisper-small openai/whisper-medium openai/whisper-large-v3; do
    MODEL_SHORT="${MODEL#openai/}"
    echo ""
    echo "=== $MODEL_SHORT ==="

    # Use fp16 for medium and large to fit in GPU memory
    FP16_FLAG=""
    if [[ "$MODEL_SHORT" == "whisper-medium" || "$MODEL_SHORT" == "whisper-large-v3" ]]; then
        FP16_FLAG="--fp16"
    fi

    echo "--- base ---"
    uv run python run_experiment.py "${EXTRA_FLAGS[@]}" $FP16_FLAG \
        --method base --model "$MODEL" \
        --eval_dataset "$DATASET" --tag exp5

    echo "--- tent ---"
    uv run python run_experiment.py "${EXTRA_FLAGS[@]}" $FP16_FLAG \
        --method tent --model "$MODEL" \
        --adapt_dataset "$DATASET" --eval_dataset "$DATASET" \
        --tent_lr 1e-3 --tag exp5

    echo "--- ttl ---"
    uv run python run_experiment.py "${EXTRA_FLAGS[@]}" $FP16_FLAG \
        --method ttl --ppl_method entropy --model "$MODEL" \
        --adapt_dataset "$DATASET" --eval_dataset "$DATASET" \
        --lora_rank 8 --lr "$TTL_LR" --tag exp5
done

# --- Generate plots ---
echo ""
echo "Generating plots..."
uv run python analysis/plot_results.py --input_dir results --output_dir plots

echo ""
echo "Experiment 5 finished.  Results in results/, plots in plots/"
