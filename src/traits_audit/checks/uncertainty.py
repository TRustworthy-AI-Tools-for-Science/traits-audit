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
    Flags parameter channels whose forward uncertainty trends too steeply downward.

    **Multi-channel**: ``uncertainties`` may be a scalar series ``(steps,)`` or a
    per-channel matrix ``(steps, n_channels)`` (also read from per-step
    ``history['uncertainty']``). Each channel is assessed independently and flagged
    when its least-squares slope is more negative than ``slope_threshold`` × the
    channel mean (scale-independent). A decreasing forward-uncertainty trend is
    physically implausible for a model whose epistemic uncertainty should grow.

    ``value`` = number of flagged channels (0 ⇒ pass). ``details`` carries
    ``n_channels``, ``decreasing_channels`` (list of ``(index, slope)``) and
    ``uncertainty_series`` (the per-step mean, for trend figures).

    Parameters
    ----------
    slope_threshold : float
        Per-channel relative-slope threshold (default −0.01 = −1 %/step). A channel
        is flagged when ``slope < slope_threshold · mean(channel)``.

    Required data
    -------------
    ``uncertainties`` kwarg  **or**  ``uncertainty`` key in each history dict.
    """

    def __init__(self, slope_threshold: float = -0.01):
        self.slope_threshold = slope_threshold

    @property
    def name(self) -> str:
        return "UncertaintyEvolution"

    @property
    def category(self) -> AuditCategory:
        return AuditCategory.EPISTEMIC

    def run(self, history: List[Dict[str, Any]], **kwargs) -> AuditResult:
        raw = kwargs.get("uncertainties")
        if raw is not None:
            u = np.asarray(raw, dtype=float)
        else:
            vals = [h["uncertainty"] for h in history if "uncertainty" in h]
            if not vals:
                return AuditResult(
                    name=self.name, passed=True, category=self.category,
                    message="Skipped — uncertainty series not available.",
                )
            u = np.asarray(vals, dtype=float)

        if u.ndim == 1:
            u = u[:, np.newaxis]

        n_channels = u.shape[1]
        x = np.arange(u.shape[0])
        decreasing = []
        for i in range(n_channels):
            col   = u[:, i]
            valid = np.isfinite(col)
            if valid.sum() < 2:
                continue
            slope = float(np.polyfit(x[valid], col[valid], 1)[0])
            if slope < self.slope_threshold * float(np.mean(col[valid])):
                decreasing.append((i, slope))

        passed = len(decreasing) == 0
        mean_series = np.nanmean(u, axis=1)

        return AuditResult(
            name=self.name,
            passed=passed,
            category=self.category,
            value=float(len(decreasing)),
            threshold=0.0,
            message=(
                "All channels non-decreasing"
                if passed
                else f"{len(decreasing)} channel(s) show decreasing uncertainty"
            ),
            details={
                "n_channels": n_channels,
                "decreasing_channels": decreasing,
                "uncertainty_series": mean_series.tolist(),
            },
        )


class UncertaintyAnomalyCheck(AuditCheck):
    """
    Flags drift: whether more than ``max_anomaly_fraction`` of the current
    uncertainty values lie beyond ``z_threshold`` σ of a **historical baseline**.

    Z-scores the current series against the historical mean/std (not its own), so it
    detects departure from earlier behaviour rather than within-series outliers.
    The current series (``uncertainties`` kwarg or per-step ``history['uncertainty']``)
    is flattened; the baseline is the ``historical_uncertainties`` kwarg. Skipped
    when no baseline is provided.

    Parameters
    ----------
    z_threshold : float
        Z-score threshold (default: 3.0).
    max_anomaly_fraction : float
        Maximum acceptable fraction of anomalous current values (default: 0.05).

    Required data
    -------------
    ``uncertainties`` (current) or ``uncertainty`` key in history dicts.
    ``historical_uncertainties`` (baseline) is optional: when omitted the check
    falls back to within-series z-scoring of the accumulated history.
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
        raw_current = kwargs.get("uncertainties")
        current_u = (
            np.asarray(raw_current, dtype=float).flatten()
            if raw_current is not None
            else np.asarray([h["uncertainty"] for h in history if "uncertainty" in h], dtype=float)
        )
        raw_hist = kwargs.get("historical_uncertainties")
        historical_u = np.asarray(raw_hist, dtype=float).flatten() if raw_hist is not None else np.array([])

        current_u    = current_u[np.isfinite(current_u)]
        historical_u = historical_u[np.isfinite(historical_u)]

        if len(current_u) == 0:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Skipped — no current uncertainty data.",
            )
        if len(historical_u) == 0:
            # No explicit baseline — fall back to within-series z-scoring so the
            # check works without the caller having to supply historical_uncertainties.
            if len(current_u) < 3:
                return AuditResult(
                    name=self.name, passed=True, category=self.category,
                    message="Too few steps for anomaly detection.",
                )
            historical_u = current_u

        hist_mean = float(np.mean(historical_u))
        hist_std  = float(np.std(historical_u))
        if hist_std > 0:
            z = (current_u - hist_mean) / hist_std
            frac  = float(np.mean(np.abs(z) > self.z_threshold))
            max_z = float(np.max(np.abs(z)))
        else:
            different = current_u != hist_mean
            frac  = float(np.mean(different))
            max_z = float(np.max(np.abs(current_u - hist_mean))) if len(current_u) else 0.0

        return AuditResult(
            name=self.name,
            passed=frac < self.max_anomaly_fraction,
            category=self.category,
            value=frac,
            threshold=self.max_anomaly_fraction,
            message=f"Anomalous fraction: {frac:.1%}  max z-score: {max_z:.2f}",
            details={
                "anomalous_fraction": frac,
                "max_z_score": max_z,
                "hist_mean": hist_mean,
                "hist_std": hist_std,
                "current_mean": float(np.mean(current_u)),
                "current_std": float(np.std(current_u)),
            },
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

        if np.std(stds) == 0 or np.std(errors) == 0:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Skipped — constant input array; correlation undefined.",
            )

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
