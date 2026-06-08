"""Interval coverage and variance alignment checks."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..base import AuditCategory, AuditCheck, AuditResult
from .calibration import _require


class IntervalCoverageCheck(AuditCheck):
    """
    Checks that empirical 1-sigma coverage is close to the expected 68.3 %.

    Parameters
    ----------
    expected_coverage : float
        Expected fraction of true values within ±1 std (default: 0.683).
    tolerance : float
        Acceptable absolute deviation (default: 0.1).

    Required data (kwargs or history keys)
    ----------------------------------------
    ``y_true``, ``y_pred_mean``, ``y_pred_std``
    """

    def __init__(self, expected_coverage: float = 0.683, tolerance: float = 0.1):
        self.expected_coverage = expected_coverage
        self.tolerance = tolerance

    @property
    def name(self) -> str:
        return "IntervalCoverage"

    @property
    def category(self) -> AuditCategory:
        return AuditCategory.ALEATORIC_MODEL

    def run(self, history: List[Dict[str, Any]], **kwargs) -> AuditResult:
        y_true = _require("y_true", history, kwargs)
        mu     = _require("y_pred_mean", history, kwargs)
        sigma  = _require("y_pred_std", history, kwargs)

        if any(v is None for v in (y_true, mu, sigma)):
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Skipped — y_true / y_pred_mean / y_pred_std not available.",
            )

        observed = float((np.abs(y_true - mu) <= sigma).mean())
        delta = abs(observed - self.expected_coverage)

        lo = self.expected_coverage - self.tolerance
        hi = self.expected_coverage + self.tolerance
        return AuditResult(
            name=self.name,
            passed=delta <= self.tolerance,
            category=self.category,
            value=observed,
            threshold=(lo, hi),
            message=(
                f"1σ coverage = {observed:.3f}  "
                f"(expected {self.expected_coverage:.3f} ± {self.tolerance},  "
                f"acceptable band [{lo:.3f}, {hi:.3f}],  Δ = {delta:.3f})"
            ),
            details={"tolerance": self.tolerance, "band_lo": lo, "band_hi": hi},
        )


class VarianceAlignmentCheck(AuditCheck):
    """
    Checks that mean predicted variance is close to mean empirical squared error.

    Ratio = mean(σ²_pred) / mean((y - μ)²).  Passes when ``abs(ratio - 1) <= tolerance``.

    Parameters
    ----------
    tolerance : float
        Maximum acceptable ``abs(ratio - 1)`` (default: 0.5).

    Required data (kwargs or history keys)
    ----------------------------------------
    ``y_true``, ``y_pred_mean``, ``y_pred_std``
    """

    def __init__(self, tolerance: float = 0.5):
        self.tolerance = tolerance

    @property
    def name(self) -> str:
        return "VarianceAlignment"

    @property
    def category(self) -> AuditCategory:
        return AuditCategory.ALEATORIC_MODEL

    def run(self, history: List[Dict[str, Any]], **kwargs) -> AuditResult:
        y_true = _require("y_true", history, kwargs)
        mu     = _require("y_pred_mean", history, kwargs)
        sigma  = _require("y_pred_std", history, kwargs)

        if any(v is None for v in (y_true, mu, sigma)):
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Skipped — y_true / y_pred_mean / y_pred_std not available.",
            )

        ratio = float(np.mean(sigma ** 2) / (np.mean((y_true - mu) ** 2) + 1e-12))

        return AuditResult(
            name=self.name,
            passed=abs(ratio - 1.0) <= self.tolerance,
            category=self.category,
            value=ratio,
            threshold=1.0,
            message=f"Variance ratio (pred/true) = {ratio:.4f}  (tolerance ±{self.tolerance})",
        )
