#!/usr/bin/env bash
# =============================================================
# Phase 1: Run all core experiments sequentially
#
# This runs exp1 through exp4 (clean baselines + noise robustness).
# Each experiment generates plots when it completes.
#
# Usage (with nohup):
#   nohup bash scripts/run_phase1.sh --gpu 0 --batch_size 32 > logs/phase1.log 2>&1 &
# =============================================================
set -euo pipefail

echo "============================================"
echo "  Phase 1: Core Experiments"
echo "  Started: $(date)"
echo "============================================"

echo ""
echo ">>> Running Experiment 1: TTL vs Base"
bash scripts/exp1_ttl_vs_base.sh "$@"

echo ""
echo ">>> Running Experiment 2: TTL vs Tent"
bash scripts/exp2_ttl_vs_tent.sh "$@"

echo ""
echo ">>> Running Experiment 3: Sample Selection Ablation"
bash scripts/exp3_sample_selection.sh "$@"

echo ""
echo ">>> Running Experiment 4: LibriSpeech-C Noise Robustness"
bash scripts/exp4_librispeech_c.sh "$@"

echo ""
echo "============================================"
echo "  Phase 1 complete!"
echo "  Finished: $(date)"
echo "  Results in results/, plots in plots/"
echo "============================================"
