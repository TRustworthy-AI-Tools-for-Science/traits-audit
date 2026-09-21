"""Variability/ignorance taxonomy class: IWF, EnvelopeViolationRate.

See METRIC_TAXONOMY_AUDIT.md §4.5. A single predictive distribution
conflates variability (genuine dispersion) with ignorance (imprecision
about which distribution applies) by construction — both checks here
require the ``CredalSet`` representation (``traits_audit/credal.py``)
rather than a bare mean+std, which is the representational change this
taxonomy class demands.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..base import AuditCategory, AuditCheck, AuditResult
from ..credal import CredalSet
from .calibration import _require


def _build_credal_set(kwargs) -> CredalSet | None:
    if kwargs.get("credal_lower") is not None and kwargs.get("credal_upper") is not None:
        return CredalSet(lower=kwargs["credal_lower"], upper=kwargs["credal_upper"])
    if kwargs.get("y_pred_ensemble") is not None:
        return CredalSet(
            y_pred_ensemble=kwargs["y_pred_ensemble"],
            y_pred_std_ensemble=kwargs.get("y_pred_std_ensemble"),
        )
    return None


class ImprecisionWidthFractionCheck(AuditCheck):
    """
    Imprecision Width Fraction (IWF) — how much of the stated uncertainty is
    imprecision rather than dispersion.

    Given a credal set (the convex hull of an ensemble's predictive CDFs, or
    a Swinburne-Perez bounding set), reports the upper and lower probability
    of a reference interval::

        IWF = (P_upper - P_lower) / P_upper

    IWF -> 0 certifies that a single precise probability distribution is an
    adequate representation and the rest of this suite's single-distribution
    metrics apply. IWF large certifies it is not.

    Kwargs-only: an ensemble/bounding-interval array carries no natural
    "per-step" scalar, so there is no history route for this check (compare
    ``TypeBMassFractionCheck``, similarly kwargs-only for the same reason).

    Parameters
    ----------
    ref_z : float
        When ``ref_lower``/``ref_upper`` are not given explicitly, the
        default reference interval is ``[mean - ref_z*std, mean + ref_z*std]``
        (default 1.0).
    iwf_threshold : float or None
        Maximum acceptable mean IWF. ``None`` (default) disables pass/fail.

    References
    ----------
    Walley, P. (1991). *Statistical Reasoning with Imprecise Probabilities*.
    Chapman and Hall. (source of the upper/lower probability formalism
    ``P_upper``/``P_lower`` reuse here).
    Dubois, D. (2007). HAL hal-03445671.
    Swinburne, T. D. & Perez, D. (2025). *Mach. Learn.: Sci. Technol.*,
    6(1):015008.

    Required data (kwargs)
    -----------------------
    ``y_pred_mean``, ``y_pred_std`` (define the default reference interval
    unless ``ref_lower``/``ref_upper`` given), plus either
    ``(credal_lower, credal_upper)`` or ``y_pred_ensemble``
    (``+ y_pred_std_ensemble`` optional).
    """

    def __init__(self, ref_z: float = 1.0, iwf_threshold: float | None = None):
        self.ref_z = ref_z
        self.iwf_threshold = iwf_threshold

    @property
    def name(self) -> str:
        return "ImprecisionWidthFraction"

    @property
    def category(self) -> AuditCategory:
        return AuditCategory.VARIABILITY_IGNORANCE

    def run(self, history: list[dict[str, Any]], **kwargs) -> AuditResult:
        credal_set = _build_credal_set(kwargs)
        mu = kwargs.get("y_pred_mean")
        sigma = kwargs.get("y_pred_std")

        if credal_set is None or mu is None or sigma is None:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message=(
                    "Skipped — need y_pred_mean, y_pred_std, and either "
                    "(credal_lower, credal_upper) or y_pred_ensemble."
                ),
            )

        mu = np.asarray(mu, dtype=float).ravel()
        sigma = np.asarray(sigma, dtype=float).ravel()
        ref_lower = kwargs.get("ref_lower")
        ref_upper = kwargs.get("ref_upper")
        if ref_lower is None or ref_upper is None:
            ref_lower = mu - self.ref_z * sigma
            ref_upper = mu + self.ref_z * sigma

        p_upper, p_lower = credal_set.reference_probabilities(ref_lower, ref_upper)
        eps = 1e-12
        iwf = (p_upper - p_lower) / np.maximum(p_upper, eps)
        value = float(np.mean(iwf))

        passed = True if self.iwf_threshold is None else value <= self.iwf_threshold
        return AuditResult(
            name=self.name,
            passed=passed,
            category=self.category,
            value=value,
            threshold=self.iwf_threshold,
            message=f"IWF (mean) = {value:.4f}  (0=precise-probability adequate, large=imprecise)",
            details={
                "per_point_iwf": iwf.tolist(),
                "P_upper": p_upper.tolist(),
                "P_lower": p_lower.tolist(),
            },
        )


class EnvelopeViolationRateCheck(AuditCheck):
    """
    Envelope violation rate — the one-sided adequacy of a bounding set: the
    fraction of held-out points outside the credal set's envelope.

    Does not presuppose finite second moments (unlike variance-based
    checks), and returns to Colclough's (1987) worst-case interval, reached
    independently by Swinburne & Perez / Perez et al. thirty-eight years
    later.

    Parameters
    ----------
    max_violation_rate : float
        Maximum acceptable violation rate (default 0.1).

    References
    ----------
    Perez, A. et al. (2025). arXiv:2502.07104.
    Colclough, A. R. (1987). *J. Res. NBS*, 92(3), 167-185.

    Required data (kwargs or history keys for ``y_true``; bounds are kwargs-only)
    -------------------------------------------------------------------------------
    ``y_true`` + either ``(credal_lower, credal_upper)`` or ``y_pred_ensemble``.
    """

    def __init__(self, max_violation_rate: float = 0.1):
        self.max_violation_rate = max_violation_rate

    @property
    def name(self) -> str:
        return "EnvelopeViolationRate"

    @property
    def category(self) -> AuditCategory:
        return AuditCategory.VARIABILITY_IGNORANCE

    def run(self, history: list[dict[str, Any]], **kwargs) -> AuditResult:
        y_true = _require("y_true", history, kwargs)
        credal_set = _build_credal_set(kwargs)

        if y_true is None or credal_set is None:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Skipped — need y_true and either (credal_lower, credal_upper) or y_pred_ensemble.",
            )

        contained = credal_set.contains(y_true)
        value = float(1.0 - np.mean(contained))

        return AuditResult(
            name=self.name,
            passed=value <= self.max_violation_rate,
            category=self.category,
            value=value,
            threshold=self.max_violation_rate,
            message=f"Envelope violation rate = {value:.4f}",
            details={"n_violations": int(np.sum(~contained)), "n_points": len(y_true)},
        )
