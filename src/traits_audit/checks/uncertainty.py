"""Uncertainty evolution, anomaly, and variance-error correlation checks."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..base import AuditCategory, AuditCheck, AuditResult
from .calibration import _require


def _uncertainties(history: list, kwargs: dict) -> Optional[np.ndarray]:
    """Pull uncertainty series from kwargs or history['uncertainty'] key.

    Returns None if the data is not available — callers must return a skipped
    AuditResult rather than raising.
    """
    if "uncertainties" in kwargs:
        return np.asarray(kwargs["uncertainties"], dtype=float)
    vals = [h["uncertainty"] for h in history if "uncertainty" in h]
    if not vals:
        return None
    return np.asarray(vals, dtype=float)


class UncertaintyEvolutionCheck(AuditCheck):
    """
    Flags if predictive uncertainty trends too steeply downward over iterations.

    The slope is normalised by the mean uncertainty so it is scale-independent.
    A very steep negative slope can indicate model collapse or an overconfident
    posterior.

    Parameters
    ----------
    slope_threshold : float
        Minimum acceptable relative slope per step (default: −0.05 = −5 %/step).

    Required data
    -------------
    ``uncertainties`` kwarg  **or**  ``uncertainty`` key in each history dict.
    """

    def __init__(self, slope_threshold: float = -0.05):
        self.slope_threshold = slope_threshold

    @property
    def name(self) -> str:
        return "UncertaintyEvolution"

    @property
    def category(self) -> AuditCategory:
        return AuditCategory.EPISTEMIC

    def run(self, history: List[Dict[str, Any]], **kwargs) -> AuditResult:
        u = _uncertainties(history, kwargs)
        if u is None:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Skipped — uncertainty series not available.",
            )
        if len(u) < 2:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Too few steps to evaluate trend.",
            )
        slope = float(np.polyfit(np.arange(len(u)), u, 1)[0])
        rel   = slope / (float(np.mean(u)) + 1e-12)

        return AuditResult(
            name=self.name,
            passed=rel >= self.slope_threshold,
            category=self.category,
            value=rel,
            threshold=self.slope_threshold,
            message=f"Relative uncertainty slope = {rel:+.4f} / step",
        )


class UncertaintyAnomalyCheck(AuditCheck):
    """
    Flags steps where uncertainty deviates by more than ``z_threshold`` standard
    deviations from the historical mean.

    Parameters
    ----------
    z_threshold : float
        Z-score threshold (default: 3.0).
    max_anomaly_fraction : float
        Maximum acceptable fraction of anomalous steps (default: 0.05).

    Required data
    -------------
    ``uncertainties`` kwarg  **or**  ``uncertainty`` key in each history dict.
    """

    def __init__(self, z_threshold: float = 3.0, max_anomaly_fraction: float = 0.05):
        self.z_threshold = z_threshold
        self.max_anomaly_fraction = max_anomaly_fraction

    @property
    def name(self) -> str:
        return "UncertaintyAnomalies"

    @property
    def category(self) -> AuditCategory:
        return AuditCategory.EPISTEMIC

    def run(self, history: List[Dict[str, Any]], **kwargs) -> AuditResult:
        u = _uncertainties(history, kwargs)
        if u is None:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Skipped — uncertainty series not available.",
            )
        if len(u) < 3:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Too few steps for anomaly detection.",
            )
        std = np.std(u)
        if std < 1e-12:
            # Constant series — any deviation from the single value is anomalous
            frac = float(np.mean(u != np.mean(u)))
        else:
            z    = (u - np.mean(u)) / std
            mask = np.abs(z) > self.z_threshold
            frac = float(mask.mean())

        return AuditResult(
            name=self.name,
            passed=frac <= self.max_anomaly_fraction,
            category=self.category,
            value=frac,
            threshold=self.max_anomaly_fraction,
            message=f"{frac:.1%} of steps flagged (|z| > {self.z_threshold})",
        )


class VarianceErrorCorrelationCheck(AuditCheck):
    """
    Checks that predicted uncertainty is positively correlated with prediction
    error (Spearman rank correlation).

    A well-calibrated model should assign higher uncertainty where it errs most.
    Negative correlation means the model is most confident where it is most wrong.

    Parameters
    ----------
    min_correlation : float
        Minimum acceptable Spearman ρ (default: 0.0 — any non-negative).

    Required data (kwargs or history keys)
    ----------------------------------------
    ``y_true``, ``y_pred_mean``, ``y_pred_std``
    """

    def __init__(self, min_correlation: float = 0.0):
        self.min_correlation = min_correlation

    @property
    def name(self) -> str:
        return "VarianceErrorCorrelation"

    @property
    def category(self) -> AuditCategory:
        return AuditCategory.EPISTEMIC

    def run(self, history: List[Dict[str, Any]], **kwargs) -> AuditResult:
        from scipy.stats import spearmanr

        y_true = _require("y_true", history, kwargs)
        mu     = _require("y_pred_mean", history, kwargs)
        sigma  = _require("y_pred_std", history, kwargs)

        if any(v is None for v in (y_true, mu, sigma)):
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Skipped — y_true / y_pred_mean / y_pred_std not available.",
            )

        errors = np.abs(y_true - mu)
        stds   = sigma
        rho, pval = spearmanr(stds, errors)
        rho = float(rho)

        return AuditResult(
            name=self.name,
            passed=rho >= self.min_correlation,
            category=self.category,
            value=rho,
            threshold=self.min_correlation,
            message=f"Spearman ρ (std vs |error|) = {rho:.4f}  (p = {float(pval):.3f})",
            details={"p_value": float(pval)},
        )
