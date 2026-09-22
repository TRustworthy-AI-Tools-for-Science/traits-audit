#!/bin/bash
# Reproduce the core committee figure set (correlation, density, regret,
# learning curves) for a benchmark other than Forrester — Branin-Currin or
# color-matching (see src/traits_audit/committee/problems.py).
#
# Unlike make_all_committee_figures.sh, this does NOT run thread-a/thread-b:
# those (aggregation bake-off, votes-as-features ablation) still assume a
# 1-D acquisition grid and haven't been generalized to 2-D/3-D actions.
#
# Usage:  scripts/make_problem_figures.sh <branin-currin|color> [OUT_DIR] [MODELS_DIR]
set -euo pipefail

PROBLEM="${1:?usage: make_problem_figures.sh <branin-currin|color> [OUT_DIR] [MODELS_DIR]}"
OUT="${2:-_results/committee_${PROBLEM}}"
MODELS="${3:-runs/committee_${PROBLEM}_5M/models}"
TB="${MODELS%/models}/tb"

mkdir -p "$OUT"

ta-committee-analyze corr-random     --problem "$PROBLEM" --output-dir "$OUT"
ta-committee-analyze corr-trained    --problem "$PROBLEM" --models-dir "$MODELS" --output-dir "$OUT"
ta-committee-analyze density         --problem "$PROBLEM" --models-dir "$MODELS" --output-dir "$OUT"
ta-committee-analyze regret          --problem "$PROBLEM" --models-dir "$MODELS" --output-dir "$OUT"
ta-committee-analyze learning-curves --tb-dir "$TB" --output-dir "$OUT"

echo "All figures written under $OUT/"
