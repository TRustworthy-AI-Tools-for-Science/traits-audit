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


def _calibration_stats(y_true, mu, sigma, n_bins: int = 10):
    """Self-contained calibration statistics (numpy/scipy only).

    Shared by the three split calibration checks below so each reports a single
    metric as its own pipeline row while computing the full set once. Returns
    ``None`` when there are no valid (finite, positive-sigma) points.
    """
    from scipy import stats as _st

    r = np.asarray(y_true, float).ravel() - np.asarray(mu, float).ravel()
    s = np.asarray(sigma, float).ravel()
    valid = np.isfinite(r) & np.isfinite(s) & (s > 0)
    if not valid.any():
        return None
    r, s = r[valid], s[valid]

    # Kuleshov (2018) calibration error: |observed - expected| coverage, averaged.
    levels = np.linspace(0.0, 1.0, n_bins + 2)[1:-1]
    observed = np.array([float((np.abs(r) <= _st.norm.ppf((1 + p) / 2) * s).mean())
                         for p in levels])
    ce = float(np.mean(np.abs(observed - levels)))
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)  # np.trapz removed in NumPy 2.x
    miscal_area = float(_trapz(np.abs(observed - levels), levels))

    # ENCE (Levi et al. 2022): bin by predicted sigma, compare RMSE to RMV per bin.
    order = np.argsort(s)
    ence_terms = []
    for chunk in np.array_split(order, min(n_bins, max(1, r.size))):
        if chunk.size == 0:
            continue
        rmv = float(np.sqrt(np.mean(s[chunk] ** 2)))
        rmse = float(np.sqrt(np.mean(r[chunk] ** 2)))
        if rmv > 0:
            ence_terms.append(abs(rmv - rmse) / rmv)
    ence = float(np.mean(ence_terms)) if ence_terms else float("nan")

    z = r / s
    within_1std = float(np.mean(np.abs(z) <= 1.0))
    std_cv = float(np.std(s) / max(np.mean(s), 1e-12))
    return {
        "ce": ce,
        "ence": ence,
        "within_1std": within_1std,
        "cal_err_1std": abs(within_1std - 0.68),
        "miscalibration_area": miscal_area,
        "std_cv": std_cv,
        "predicted_pi": levels.tolist(),
        "observed_pi": observed.tolist(),
    }


class KuleshovCalibrationCheck(AuditCheck):
    """Kuleshov (2018) calibration error (CE) as its own pipeline row.

    Equivalent metric to :class:`CalibrationErrorCheck`, reported under the name
    ``KuleshovCalibrationError`` so CE, ENCE, and 1-std error each render as a
    distinct check-grid column.

    Required data (kwargs or history keys): ``y_true``, ``y_pred_mean``, ``y_pred_std``
    """

    def __init__(self, ce_threshold: float = 0.1):
        self.ce_threshold = ce_threshold

    @property
    def name(self) -> str:
        return "KuleshovCalibrationError"

    @property
    def category(self) -> AuditCategory:
        return AuditCategory.ALEATORIC_MODEL

    def run(self, history: List[Dict[str, Any]], **kwargs) -> AuditResult:
        y_true = _require("y_true", history, kwargs)
        mu     = _require("y_pred_mean", history, kwargs)
        sigma  = _require("y_pred_std", history, kwargs)
        if any(v is None for v in (y_true, mu, sigma)):
            return AuditResult(name=self.name, passed=True, category=self.category,
                               message="Skipped — y_true / y_pred_mean / y_pred_std not available.")
        comp = _calibration_stats(y_true, mu, sigma)
        if comp is None:
            return AuditResult(name=self.name, passed=False, category=self.category,
                               message="No valid predictions for calibration.")
        return AuditResult(
            name=self.name, passed=comp["ce"] < self.ce_threshold,
            value=comp["ce"], threshold=self.ce_threshold, category=self.category,
            message=f"CE={comp['ce']:.4f}  miscal_area={comp['miscalibration_area']:.4f}",
            details=comp,
        )


class ENCECheck(AuditCheck):
    """Expected Normalized Calibration Error (Levi et al. 2022) as its own row.

    Bins points by predicted sigma and compares per-bin RMSE to root-mean-variance.

    Required data (kwargs or history keys): ``y_true``, ``y_pred_mean``, ``y_pred_std``
    """

    def __init__(self, ence_threshold: float = 0.1):
        self.ence_threshold = ence_threshold

    @property
    def name(self) -> str:
        return "ENCE"

    @property
    def category(self) -> AuditCategory:
        return AuditCategory.ALEATORIC_MODEL

    def run(self, history: List[Dict[str, Any]], **kwargs) -> AuditResult:
        y_true = _require("y_true", history, kwargs)
        mu     = _require("y_pred_mean", history, kwargs)
        sigma  = _require("y_pred_std", history, kwargs)
        if any(v is None for v in (y_true, mu, sigma)):
            return AuditResult(name=self.name, passed=True, category=self.category,
                               message="Skipped — y_true / y_pred_mean / y_pred_std not available.")
        comp = _calibration_stats(y_true, mu, sigma)
        if comp is None:
            return AuditResult(name=self.name, passed=False, category=self.category,
                               message="No valid predictions for calibration.")
        return AuditResult(
            name=self.name, passed=comp["ence"] < self.ence_threshold,
            value=comp["ence"], threshold=self.ence_threshold, category=self.category,
            message=f"ENCE={comp['ence']:.4f}  std_cv={comp['std_cv']:.4f}",
            details=comp,
        )


class CalibrationError1StdCheck(AuditCheck):
    """1-std predictive-interval coverage error (expected 68% under a Gaussian).

    Required data (kwargs or history keys): ``y_true``, ``y_pred_mean``, ``y_pred_std``
    """

    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "CalibrationError1Std"

    @property
    def category(self) -> AuditCategory:
        return AuditCategory.ALEATORIC_MODEL

    def run(self, history: List[Dict[str, Any]], **kwargs) -> AuditResult:
        y_true = _require("y_true", history, kwargs)
        mu     = _require("y_pred_mean", history, kwargs)
        sigma  = _require("y_pred_std", history, kwargs)
        if any(v is None for v in (y_true, mu, sigma)):
            return AuditResult(name=self.name, passed=True, category=self.category,
                               message="Skipped — y_true / y_pred_mean / y_pred_std not available.")
        comp = _calibration_stats(y_true, mu, sigma)
        if comp is None:
            return AuditResult(name=self.name, passed=False, category=self.category,
                               message="No valid predictions for calibration.")
        return AuditResult(
            name=self.name, passed=comp["cal_err_1std"] < self.threshold,
            value=comp["cal_err_1std"], threshold=self.threshold, category=self.category,
            message=f"within 1σ: {comp['within_1std']:.1%} (expected 68%)",
            details=comp,
        )
