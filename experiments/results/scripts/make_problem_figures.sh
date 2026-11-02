#!/bin/bash
# Reproduce the full committee figure set (correlation, density, regret,
# learning curves, thread-a, thread-b) for a benchmark other than Forrester
# — Branin-Currin or color-matching (see committee/problems.py).
#
# thread-a and thread-b run here too: the aggregators and the vote-distance
# penalty take vector actions, so 2-D/3-D go through the same code path as
# Forrester. thread-a reads regret_test.json to pick its best-solo
# reference, so the `regret` step above must run first.
#
# Usage:  scripts/make_problem_figures.sh <branin-currin|color> [OUT_DIR] [MODELS_DIR]
set -euo pipefail

PROBLEM="${1:?usage: make_problem_figures.sh <branin-currin|color> [OUT_DIR] [MODELS_DIR]}"
OUT="${2:-_results/committee_${PROBLEM}}"
MODELS="${3:-runs/committee_${PROBLEM}_5M/models}"
TB="${MODELS%/models}/tb"

mkdir -p "$OUT" "$OUT/threadA" "$OUT/threadB"

ta-committee-analyze corr-random     --problem "$PROBLEM" --output-dir "$OUT"
ta-committee-analyze corr-trained    --problem "$PROBLEM" --models-dir "$MODELS" --output-dir "$OUT"
ta-committee-analyze density         --problem "$PROBLEM" --models-dir "$MODELS" --output-dir "$OUT"
ta-committee-analyze regret          --problem "$PROBLEM" --models-dir "$MODELS" --output-dir "$OUT"
ta-committee-analyze learning-curves --tb-dir "$TB" --output-dir "$OUT"
ta-committee-analyze thread-b        --problem "$PROBLEM" --models-dir "$MODELS" \
    --output-dir "$OUT/threadB"
ta-committee-analyze thread-a        --problem "$PROBLEM" --models-dir "$MODELS" \
    --correlation-csv "$OUT/correlation_trained.csv" \
    --regret-json     "$OUT/regret_test.json" \
    --output-dir      "$OUT/threadA"

echo "All figures written under $OUT/"
