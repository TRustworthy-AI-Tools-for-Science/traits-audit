"""Aleatoric/epistemic taxonomy class — repairing "total only" coverage.

METRIC_TAXONOMY_AUDIT.md §4.3: every existing calibration/scoring check in
this package scores the *total* predictive distribution against outcomes;
none discriminates the aleatoric/epistemic split itself.
ReducibilityRealisationRatioCheck and AleatoricFloorConsistencyCheck test
each side of that split directly.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..base import AuditCategory, AuditCheck, AuditResult
from ._replicates import build_replicate_groups
from .calibration import _require


class ReducibilityRealisationRatioCheck(AuditCheck):
    """
    Reducibility Realisation Ratio (RRR) — whether the epistemic term means
    what it claims.

    At step t the model claims epistemic variance u_e(x)^2. The loop
    acquires at or near x. At step t+k, the realized drop in total
    predictive variance at x is measured::

        RRR = sum(realized_reduction) / sum(claimed_reduction)

    RRR ~= 1: the epistemic term is honest. RRR < 1: reducibility is
    over-claimed, and an acquisition policy trusting it will over-exploit.
    RRR > 1: the lower-bound regime Jimenez et al. (2025) predict — the
    estimate understates what is actually reducible, and the policy
    under-explores.

    .. important::
       **This check does not perform spatial nearest-neighbor re-query
       matching.** The literature's "acquire at or near x" step requires a
       domain-specific distance metric this package cannot supply
       generically. The three input series must already be paired 1:1 by
       the caller (claim at acquisition time, and the realized before/after
       measurement at the same or a matched point) — this check audits
       already-matched data, it does not chase re-queries itself.

    Parameters
    ----------
    rrr_tolerance : float
        ``passed = abs(RRR - 1) <= rrr_tolerance`` (default 0.3). 1.0 is the
        literature-mandated target here (unlike RSE's contested reference
        point) — only the tolerance width is a free parameter, the same
        status as e.g. ``CalibrationErrorCheck(threshold=0.1)``.

    References
    ----------
    Jimenez, I., Jurgens, D. & Waegeman, W. (2025). arXiv:2505.23506.
    Bengs, V. et al. (2022). NeurIPS 35.

    Required data (kwargs or history keys)
    ----------------------------------------
    ``claimed_epistemic_variance``, ``realized_total_variance_before``,
    ``realized_total_variance_after`` — pre-paired 1:1 by the caller.
    """

    def __init__(self, rrr_tolerance: float = 0.3):
        self.rrr_tolerance = rrr_tolerance

    @property
    def name(self) -> str:
        return "ReducibilityRealisationRatio"

    @property
    def category(self) -> AuditCategory:
        return AuditCategory.EPISTEMIC

    def run(self, history: List[Dict[str, Any]], **kwargs) -> AuditResult:
        claimed = _require("claimed_epistemic_variance", history, kwargs)
        before = _require("realized_total_variance_before", history, kwargs)
        after = _require("realized_total_variance_after", history, kwargs)

        if any(v is None for v in (claimed, before, after)):
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message=(
                    "Skipped — claimed_epistemic_variance / "
                    "realized_total_variance_before / realized_total_variance_after not available."
                ),
            )

        n = min(len(claimed), len(before), len(after))
        if n == 0:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Skipped — empty series.",
            )
        claimed, before, after = claimed[:n], before[:n], after[:n]

        sum_claimed = float(np.sum(claimed))
        if sum_claimed <= 0:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Skipped — non-positive total claimed reduction.",
            )
        realized = before - after
        value = float(np.sum(realized) / sum_claimed)

        return AuditResult(
            name=self.name,
            passed=abs(value - 1.0) <= self.rrr_tolerance,
            category=self.category,
            value=value,
            threshold=self.rrr_tolerance,
            message=f"RRR = {value:.4f}  (1=honest, <1=over-claimed, >1=under-claimed)",
            details={
                "n_pairs": n,
                "sum_realized": float(np.sum(realized)),
                "sum_claimed": sum_claimed,
            },
        )


class AleatoricFloorConsistencyCheck(AuditCheck):
    """
    Aleatoric Floor Consistency (AFC) — ratio of the learned aleatoric
    sigma_a(x) to the observed replicate scatter at repeated inputs.

    Jimenez et al. (2025) show that any deviation of the fitted mean from
    the conditional mean inflates the learned aleatoric term and deflates
    the epistemic one. AFC != 1 detects exactly that contamination — the
    only direct test of it available without ground-truth uncertainty.

    Parameters
    ----------
    afc_tolerance : float
        ``passed = abs(AFC - 1) <= afc_tolerance`` (default 0.3).

    References
    ----------
    Jimenez, I., Jurgens, D. & Waegeman, W. (2025). arXiv:2505.23506.

    Required data
    -------------
    Replicate groups (see ``checks/_replicates.py``), each with both
    ``y_true`` and ``y_pred_std`` populated. Shares ``build_replicate_groups``
    with ``ReplicationShrinkageExponentCheck``/``DarkUncertaintyGapCheck`` —
    a caller supplying ``replicate_groups`` once can drive all three
    together.
    """

    def __init__(self, afc_tolerance: float = 0.3):
        self.afc_tolerance = afc_tolerance

    @property
    def name(self) -> str:
        return "AleatoricFloorConsistency"

    @property
    def category(self) -> AuditCategory:
        return AuditCategory.ALEATORIC_MODEL

    def run(self, history: List[Dict[str, Any]], **kwargs) -> AuditResult:
        groups = build_replicate_groups(history, kwargs)
        groups = [g for g in groups if g.y_pred_std is not None]
        if not groups:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Skipped — no replicate groups with y_pred_std available.",
            )

        per_group = {}
        afcs = []
        for g in groups:
            s_g = float(np.std(g.y_true, ddof=1))
            if s_g <= 0:
                continue
            afc_g = float(np.mean(g.y_pred_std) / s_g)
            per_group[g.key] = afc_g
            afcs.append(afc_g)

        if not afcs:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Skipped — no groups with nonzero replicate scatter.",
            )

        value = float(np.median(afcs))
        return AuditResult(
            name=self.name,
            passed=abs(value - 1.0) <= self.afc_tolerance,
            category=self.category,
            value=value,
            threshold=self.afc_tolerance,
            message=f"AFC (median) = {value:.4f}  (1=consistent, >1=over-declared, <1=under-declared)",
            details={"per_group_afc": per_group, "n_groups": len(afcs)},
        )
