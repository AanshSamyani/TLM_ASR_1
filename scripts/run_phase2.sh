#!/usr/bin/env bash
# =============================================================
# Phase 2: Run ablation & scaling experiments
#
# This runs exp5 (model scaling) and exp6 (LoRA ablations).
# Each experiment generates plots when it completes.
#
# Usage (with nohup):
#   nohup bash scripts/run_phase2.sh --gpu 0 --batch_size 32 > logs/phase2.log 2>&1 &
# =============================================================
set -euo pipefail

echo "============================================"
echo "  Phase 2: Ablations & Model Scaling"
echo "  Started: $(date)"
echo "============================================"

echo ""
echo ">>> Running Experiment 5: Model Scaling"
bash scripts/exp5_model_scaling.sh "$@"

echo ""
echo ">>> Running Experiment 6: LoRA Ablations"
bash scripts/exp6_ablations.sh "$@"

echo ""
echo "============================================"
echo "  Phase 2 complete!"
echo "  Finished: $(date)"
echo "  Results in results/, plots in plots/"
echo "============================================"
