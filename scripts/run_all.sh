#!/usr/bin/env bash
# Run all three experiments sequentially.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "===== Experiment 1: TTL vs Base ====="
bash "$SCRIPT_DIR/exp1_ttl_vs_base.sh"

echo ""
echo "===== Experiment 2: TTL vs Tent ====="
bash "$SCRIPT_DIR/exp2_ttl_vs_tent.sh"

echo ""
echo "===== Experiment 3: Sample Selection Ablation ====="
bash "$SCRIPT_DIR/exp3_sample_selection.sh"

echo ""
echo "===== All experiments done. Generating plots… ====="
uv run python analysis/plot_results.py --input_dir results --output_dir results

echo "Done!  Check results/ for JSON files and PNG plots."
