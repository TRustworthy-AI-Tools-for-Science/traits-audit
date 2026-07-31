#!/bin/bash
# Reproduce every committee-v1 figure from a completed training run.
# Assumes runs/committee_v0_5M/models/ contains all 15 agents x 5 seeds.
#
# Usage:  scripts/make_all_committee_figures.sh [OUT_DIR] [MODELS_DIR]
set -euo pipefail

OUT="${1:-_results/committee_v1}"
MODELS="${2:-runs/committee_v0_5M/models}"
TB="${MODELS%/models}/tb"

mkdir -p "$OUT" "$OUT/threadA" "$OUT/threadB"

ta-committee-analyze corr-random     --output-dir "$OUT"
ta-committee-analyze corr-trained    --models-dir "$MODELS" --output-dir "$OUT"
ta-committee-analyze density         --models-dir "$MODELS" --output-dir "$OUT"
ta-committee-analyze regret          --models-dir "$MODELS" --output-dir "$OUT"
ta-committee-analyze learning-curves --tb-dir     "$TB"     --output-dir "$OUT"
ta-committee-analyze thread-b        --models-dir "$MODELS" --output-dir "$OUT/threadB"
ta-committee-analyze thread-a        --models-dir "$MODELS" \
    --correlation-csv "$OUT/correlation_trained.csv" \
    --regret-json     "$OUT/regret_test.json" \
    --output-dir      "$OUT/threadA"

echo "All figures written under $OUT/"
