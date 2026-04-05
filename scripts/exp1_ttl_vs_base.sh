#!/usr/bin/env bash
# =============================================================
# Experiment 1: TTL vs Base (no adaptation)
#
# Compares Whisper-small with and without TTL adaptation on
# three test sets of increasing domain shift.
#
# Usage:  bash scripts/exp1_ttl_vs_base.sh --gpu 0 --batch_size 32
# =============================================================
set -euo pipefail
MODEL="openai/whisper-small"

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

for DATASET in librispeech_clean librispeech_other tedlium; do
    echo "========================================"
    echo "  Dataset: $DATASET"
    echo "========================================"

    # --- baseline (no adaptation) ---
    uv run python run_experiment.py "${EXTRA_FLAGS[@]}" \
        --method base \
        --model "$MODEL" \
        --eval_dataset "$DATASET" \
        --tag exp1

    # --- TTL (entropy, no sample selection) ---
    uv run python run_experiment.py "${EXTRA_FLAGS[@]}" \
        --method ttl \
        --ppl_method entropy \
        --model "$MODEL" \
        --adapt_dataset "$DATASET" \
        --eval_dataset "$DATASET" \
        --lora_rank 8 \
        --lr 5e-5 \
        --tag exp1

    # --- TTL (entropy, with sample selection) ---
    uv run python run_experiment.py "${EXTRA_FLAGS[@]}" \
        --method ttl \
        --ppl_method entropy \
        --model "$MODEL" \
        --adapt_dataset "$DATASET" \
        --eval_dataset "$DATASET" \
        --lora_rank 8 \
        --lr 5e-5 \
        --sample_selection \
        --tag exp1

done

echo ""
echo "Experiment 1 finished.  Results in results/"
