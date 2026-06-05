"""Calibration checks."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..base import AuditCategory, AuditCheck, AuditResult


def _require(name: str, history: list, kwargs: dict, history_key: str | None = None):
    """
    Pull a named array from kwargs, falling back to extracting from history.
    Returns None if the data is not available — callers must handle this and
    return a Skipped AuditResult rather than raising.
    """
    if name in kwargs and kwargs[name] is not None:
        return np.asarray(kwargs[name]).ravel()
    key = history_key or name
    vals = [h[key] for h in history if key in h]
    if not vals:
        return None
    return np.asarray(vals).ravel()


class CalibrationErrorCheck(AuditCheck):
    """
    Kuleshov (2018) calibration error.

    For each confidence level ``p ∈ (0, 1)``, measures the empirical fraction
    of true values falling within the predicted ``p``-confidence interval, then
    reports the mean absolute deviation from the ideal diagonal.

    Parameters
    ----------
    threshold : float
        Maximum acceptable mean calibration error (default: 0.1).
    n_bins : int
        Number of confidence levels to evaluate (default: 10).

    Required data (kwargs or history keys)
    ----------------------------------------
    ``y_true``, ``y_pred_mean``, ``y_pred_std``
    """

    def __init__(self, threshold: float = 0.1, n_bins: int = 10):
        self.threshold = threshold
        self.n_bins = n_bins

    @property
    def name(self) -> str:
        return "CalibrationError"

    @property
    def category(self) -> AuditCategory:
        return AuditCategory.ALEATORIC_MODEL

    def run(self, history: List[Dict[str, Any]], **kwargs) -> AuditResult:
        from scipy import stats

        y_true = _require("y_true", history, kwargs)
        mu     = _require("y_pred_mean", history, kwargs)
        sigma  = _require("y_pred_std", history, kwargs)

        if any(v is None for v in (y_true, mu, sigma)):
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Skipped — y_true / y_pred_mean / y_pred_std not available.",
            )

        levels = np.linspace(0.0, 1.0, self.n_bins + 2)[1:-1]
        observed = [
            float((np.abs(y_true - mu) <= stats.norm.ppf((1 + p) / 2) * sigma).mean())
            for p in levels
        ]
        ce = float(np.mean(np.abs(np.array(observed) - levels)))

        return AuditResult(
            name=self.name,
            passed=ce <= self.threshold,
            category=self.category,
            value=ce,
            threshold=self.threshold,
            message=f"Mean calibration error = {ce:.4f}",
            details={"confidence_levels": levels.tolist(), "observed_fractions": observed, "calibration_error": ce},
        )
