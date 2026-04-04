#!/usr/bin/env bash
# =============================================================
# Experiment 1: TTL vs Base (no adaptation)
#
# Compares Whisper-small with and without TTL adaptation on
# three test sets of increasing domain shift.
#
# Quick run:   add  --max_samples 100  to each command
# Full run:    remove --max_samples entirely
# =============================================================
set -euo pipefail
MODEL="openai/whisper-small"

for DATASET in librispeech_clean librispeech_other tedlium; do
    echo "========================================"
    echo "  Dataset: $DATASET"
    echo "========================================"

    # --- baseline (no adaptation) ---
    uv run python run_experiment.py \
        --method base \
        --model "$MODEL" \
        --eval_dataset "$DATASET" \
        --tag exp1

    # --- TTL (no sample selection) ---
    uv run python run_experiment.py \
        --method ttl \
        --model "$MODEL" \
        --adapt_dataset "$DATASET" \
        --eval_dataset "$DATASET" \
        --lora_rank 8 \
        --lr 5e-5 \
        --tag exp1

    # --- TTL (with sample selection, P0 = e^3) ---
    uv run python run_experiment.py \
        --method ttl \
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
