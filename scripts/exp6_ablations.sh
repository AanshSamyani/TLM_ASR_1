#!/usr/bin/env bash
# =============================================================
# Experiment 6: LoRA ablations
#
# Three ablation studies on TEDLIUM:
#   A) LoRA rank sweep: r = {2, 4, 8, 16, 32}
#   B) Learning rate sweep: lr = {1e-4, 2e-4, 5e-4, 1e-3, 2e-3}
#   C) LoRA placement: encoder-only, decoder-only, both
#
# Usage:  bash scripts/exp6_ablations.sh --gpu 0 --batch_size 32
# =============================================================
set -euo pipefail
MODEL="openai/whisper-small"
DATASET="tedlium"
BASE_LR="1e-3"

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
echo "  Exp6: LoRA ablations on $DATASET"
echo "========================================"

# --- Part A: Rank sweep ---
echo ""
echo "=== Part A: LoRA rank sweep ==="
for RANK in 2 4 8 16 32; do
    echo "--- rank=$RANK ---"
    uv run python run_experiment.py "${EXTRA_FLAGS[@]}" \
        --method ttl --ppl_method entropy --model "$MODEL" \
        --adapt_dataset "$DATASET" --eval_dataset "$DATASET" \
        --lora_rank "$RANK" --lr "$BASE_LR" --tag "exp6_rank"
done

# --- Part B: Learning rate sweep ---
echo ""
echo "=== Part B: Learning rate sweep ==="
for LR in 1e-4 2e-4 5e-4 1e-3 2e-3; do
    echo "--- lr=$LR ---"
    uv run python run_experiment.py "${EXTRA_FLAGS[@]}" \
        --method ttl --ppl_method entropy --model "$MODEL" \
        --adapt_dataset "$DATASET" --eval_dataset "$DATASET" \
        --lora_rank 8 --lr "$LR" --tag "exp6_lr${LR}"
done

# --- Part C: LoRA placement ---
echo ""
echo "=== Part C: LoRA placement ==="
for PLACEMENT in encoder decoder both; do
    echo "--- placement=$PLACEMENT ---"
    uv run python run_experiment.py "${EXTRA_FLAGS[@]}" \
        --method ttl --ppl_method entropy --model "$MODEL" \
        --adapt_dataset "$DATASET" --eval_dataset "$DATASET" \
        --lora_rank 8 --lr "$BASE_LR" \
        --lora_placement "$PLACEMENT" --tag "exp6_place"
done

# --- Generate plots ---
echo ""
echo "Generating plots..."
uv run python analysis/plot_results.py --input_dir results --output_dir plots

echo ""
echo "Experiment 6 finished.  Results in results/, plots in plots/"
