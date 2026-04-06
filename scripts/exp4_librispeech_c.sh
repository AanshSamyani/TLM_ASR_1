#!/usr/bin/env bash
# =============================================================
# Experiment 4: LibriSpeech-C — noise robustness
#
# Tests all methods on LibriSpeech-other with Gaussian and Babble
# noise at 5 severity levels (SNR = 20, 15, 10, 5, 0 dB).
# Also runs clean baselines on all datasets including Common Voice.
#
# Usage:  bash scripts/exp4_librispeech_c.sh --gpu 0 --batch_size 32
# =============================================================
set -euo pipefail
MODEL="openai/whisper-small"
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
echo "  Exp4: LibriSpeech-C noise robustness"
echo "========================================"

# --- Part A: Clean baselines (all datasets, all methods) ---
for DATASET in librispeech_clean librispeech_other tedlium common_voice_en; do
    echo "--- Clean: base on $DATASET ---"
    uv run python run_experiment.py "${EXTRA_FLAGS[@]}" \
        --method base --model "$MODEL" \
        --eval_dataset "$DATASET" --tag exp4

    echo "--- Clean: tent on $DATASET ---"
    uv run python run_experiment.py "${EXTRA_FLAGS[@]}" \
        --method tent --model "$MODEL" \
        --adapt_dataset "$DATASET" --eval_dataset "$DATASET" \
        --tent_lr 1e-3 --tag exp4

    echo "--- Clean: suta on $DATASET ---"
    uv run python run_experiment.py "${EXTRA_FLAGS[@]}" \
        --method suta --model "$MODEL" \
        --eval_dataset "$DATASET" \
        --suta_steps 10 --suta_alpha 0.5 --tag exp4

    echo "--- Clean: ttl on $DATASET ---"
    uv run python run_experiment.py "${EXTRA_FLAGS[@]}" \
        --method ttl --ppl_method entropy --model "$MODEL" \
        --adapt_dataset "$DATASET" --eval_dataset "$DATASET" \
        --lora_rank 8 --lr "$TTL_LR" --tag exp4
done

# --- Part B: Gaussian noise sweep on librispeech_other ---
for SNR in 20 15 10 5 0; do
    echo ""
    echo "=== Gaussian noise, SNR=${SNR} dB ==="

    echo "--- base ---"
    uv run python run_experiment.py "${EXTRA_FLAGS[@]}" \
        --method base --model "$MODEL" \
        --eval_dataset librispeech_other \
        --noise_type gaussian --noise_snr "$SNR" --tag exp4

    echo "--- tent ---"
    uv run python run_experiment.py "${EXTRA_FLAGS[@]}" \
        --method tent --model "$MODEL" \
        --adapt_dataset librispeech_other --eval_dataset librispeech_other \
        --tent_lr 1e-3 \
        --noise_type gaussian --noise_snr "$SNR" --tag exp4

    echo "--- suta ---"
    uv run python run_experiment.py "${EXTRA_FLAGS[@]}" \
        --method suta --model "$MODEL" \
        --eval_dataset librispeech_other \
        --suta_steps 10 --suta_alpha 0.5 \
        --noise_type gaussian --noise_snr "$SNR" --tag exp4

    echo "--- ttl ---"
    uv run python run_experiment.py "${EXTRA_FLAGS[@]}" \
        --method ttl --ppl_method entropy --model "$MODEL" \
        --adapt_dataset librispeech_other --eval_dataset librispeech_other \
        --lora_rank 8 --lr "$TTL_LR" \
        --noise_type gaussian --noise_snr "$SNR" --tag exp4
done

# --- Part C: Babble noise sweep on librispeech_other ---
for SNR in 20 15 10 5 0; do
    echo ""
    echo "=== Babble noise, SNR=${SNR} dB ==="

    echo "--- base ---"
    uv run python run_experiment.py "${EXTRA_FLAGS[@]}" \
        --method base --model "$MODEL" \
        --eval_dataset librispeech_other \
        --noise_type babble --noise_snr "$SNR" --tag exp4

    echo "--- tent ---"
    uv run python run_experiment.py "${EXTRA_FLAGS[@]}" \
        --method tent --model "$MODEL" \
        --adapt_dataset librispeech_other --eval_dataset librispeech_other \
        --tent_lr 1e-3 \
        --noise_type babble --noise_snr "$SNR" --tag exp4

    echo "--- suta ---"
    uv run python run_experiment.py "${EXTRA_FLAGS[@]}" \
        --method suta --model "$MODEL" \
        --eval_dataset librispeech_other \
        --suta_steps 10 --suta_alpha 0.5 \
        --noise_type babble --noise_snr "$SNR" --tag exp4

    echo "--- ttl ---"
    uv run python run_experiment.py "${EXTRA_FLAGS[@]}" \
        --method ttl --ppl_method entropy --model "$MODEL" \
        --adapt_dataset librispeech_other --eval_dataset librispeech_other \
        --lora_rank 8 --lr "$TTL_LR" \
        --noise_type babble --noise_snr "$SNR" --tag exp4
done

# --- Generate plots ---
echo ""
echo "Generating plots..."
uv run python analysis/plot_results.py --input_dir results --output_dir plots

echo ""
echo "Experiment 4 finished.  Results in results/, plots in plots/"
