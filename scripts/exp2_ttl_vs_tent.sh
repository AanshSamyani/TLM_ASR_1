#!/usr/bin/env bash
# =============================================================
# Experiment 2: TTL vs Tent vs Base
#
# Head-to-head comparison of adaptation methods on the two
# domain-shifted test sets (librispeech_other and tedlium).
#
# Quick run:  add  --max_samples 100  to each command
# =============================================================
set -euo pipefail
MODEL="openai/whisper-small"

for DATASET in librispeech_other tedlium; do
    echo "========================================"
    echo "  Dataset: $DATASET"
    echo "========================================"

    # --- baseline ---
    python run_experiment.py \
        --method base \
        --model "$MODEL" \
        --eval_dataset "$DATASET" \
        --tag exp2

    # --- Tent (entropy minimisation, LayerNorm params) ---
    python run_experiment.py \
        --method tent \
        --model "$MODEL" \
        --adapt_dataset "$DATASET" \
        --eval_dataset "$DATASET" \
        --tent_lr 1e-3 \
        --tag exp2

    # --- TTL (CE on pseudo-labels, LoRA, sample selection) ---
    python run_experiment.py \
        --method ttl \
        --model "$MODEL" \
        --adapt_dataset "$DATASET" \
        --eval_dataset "$DATASET" \
        --lora_rank 8 \
        --lr 5e-5 \
        --sample_selection \
        --tag exp2

done

echo ""
echo "Experiment 2 finished.  Results in results/"
