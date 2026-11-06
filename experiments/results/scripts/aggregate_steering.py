#!/usr/bin/env python
"""Collect every steering cell's summary and correct for multiple comparisons.

Walks ``ROOT`` for ``steering_summary.json`` files written by
``analyze_steering.py`` (one per problem × condition cell) and writes
``ROOT/steering_aggregate.csv``: one row per (cell, policy, prediction) with the
paired effect, its Wilcoxon p-value, and the Benjamini–Hochberg q-value computed
within each prediction across all cells and policies (as in the spec).

Usage::

    python experiments/results/scripts/aggregate_steering.py ROOT
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

# prediction -> (effect key, p-value key) in steering_summary.json
PREDICTIONS = {
    "P1 steered metrics vs random": ("dS_minus_null", "dS_p"),
    "P2 audit-only metrics vs random": ("dH_minus_null", "dH_p"),
    "P2 audit-only metrics vs state-space": ("dH_minus_state_space", "dH_vs_state_space_p"),
    "P4 A drift vs random": ("A_drift_minus_null", "A_drift_p"),
    "P5 final score vs lcb": ("score_minus_lcb", "score_vs_lcb_p"),
    "P5 final score vs state-space": ("score_minus_state-space", "score_vs_state-space_p"),
    "P6 gap closed vs state-space": ("recovery_closed_minus_state_space", "recovery_closed_vs_state_space_p"),
}


def benjamini_hochberg(p_values):
    """q-values for a list of p-values (NaN entries stay NaN and are not counted)."""
    p = np.asarray(p_values, dtype=float)
    q = np.full(len(p), np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    if len(finite) == 0:
        return q
    order = finite[np.argsort(p[finite])]
    m = len(order)
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.minimum(ranked, 1.0)
    return q


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    root = Path(argv[0])
    rows = []
    for summary_path in sorted(root.rglob("steering_summary.json")):
        cell = summary_path.parent
        summary = json.loads(summary_path.read_text())
        run_path = cell / "steering_run.json"
        run = json.loads(run_path.read_text()) if run_path.exists() else {}
        for policy, entry in summary.items():
            for prediction, (effect_key, p_key) in PREDICTIONS.items():
                if p_key not in entry:
                    continue
                rows.append({
                    "cell": str(cell.relative_to(root)),
                    "problem": run.get("problem"),
                    "condition": run.get("condition"),
                    "policy": policy,
                    "prediction": prediction,
                    "effect": entry.get(effect_key),
                    "p": entry[p_key],
                    "n_seeds": entry.get("n_seeds"),
                })

    for prediction in PREDICTIONS:
        group = [row for row in rows if row["prediction"] == prediction]
        q_values = benjamini_hochberg([row["p"] if row["p"] is not None else np.nan for row in group])
        for row, q in zip(group, q_values, strict=True):
            row["q_bh"] = float(q)

    out = root / "steering_aggregate.csv"
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["cell", "problem", "condition", "policy", "prediction",
                                                "effect", "p", "q_bh", "n_seeds"])
        writer.writeheader()
        writer.writerows(rows)

    significant = [row for row in rows if np.isfinite(row["q_bh"]) and row["q_bh"] < 0.05]
    print(f"{len(rows)} tests from {len({row['cell'] for row in rows})} cells; {len(significant)} with q < 0.05")
    for row in sorted(significant, key=lambda r: (r["prediction"], r["q_bh"])):
        print(f"  {row['prediction']:38s} {row['cell']:32s} {row['policy']:24s} effect {row['effect']:+.4f}"
              f"  q {row['q_bh']:.3g}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
