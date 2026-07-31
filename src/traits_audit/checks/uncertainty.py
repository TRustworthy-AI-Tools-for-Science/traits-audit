"""Uncertainty evolution, anomaly, Mahalanobis OOD, and variance-error correlation checks."""
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


class MahalanobisOODCheck(AuditCheck):
    """
    Flags out-of-distribution (OOD) queries whose calibrated uncertainty is
    *suppressed* rather than appropriately elevated.

    Computes the Mahalanobis distance of each step's ``op_states`` row from the
    empirical distribution of the **reference history** — all steps *except*
    the trailing ``window`` — using a :class:`sklearn.covariance.LedoitWolf`
    shrinkage estimator (robust when the reference is small relative to
    ``op_states.shape[1]``, where the sample covariance would be singular or
    unstable). The trailing window is deliberately excluded from the fit: it
    is exactly what's being tested as possibly OOD, and folding it into the
    reference would let outliers inflate their own baseline and mask
    themselves. A step is OOD when its distance exceeds an **empirical
    bootstrap** threshold — ``n_bootstrap`` resamples of the reference, each
    refit with its own ``LedoitWolf`` estimate, pooled into a null
    distribution of self-distances — rather than a chi-squared critical
    value, since shrinkage distorts that theoretical assumption. The
    bootstrap threshold is ``mean + threshold_sigma · std`` of the pooled
    distances; note this assumes a roughly symmetric null even though
    Mahalanobis-like distances are typically right-skewed, a known
    simplification.

    OOD activity alone is not a problem — active learning is expected to probe
    unexplored regions. What matters is whether the model's calibrated
    uncertainty rises accordingly. ``value`` is the fraction of the last
    ``window`` steps that are OOD. When that fraction exceeds
    ``ood_fraction_threshold``, the check compares mean uncertainty on the
    OOD-flagged window steps against an in-distribution baseline (in-window
    non-OOD steps when there are at least 3, else the whole-history non-OOD
    steps): ``passed=True`` when OOD-step uncertainty is not lower (healthy
    exploration), ``passed=False`` when it is lower (suppression — the model
    is confidently wrong exactly where it is extrapolating).

    Parameters
    ----------
    min_history : int
        Minimum ``op_states`` rows required to fit a covariance estimate
        (default: 20). Fewer rows ⇒ skip.
    threshold_sigma : float
        Bootstrap threshold multiplier (default: 3.0).
    window : int
        Number of most recent steps over which the OOD fraction is computed
        (default: 10). Clipped to the available history length.
    n_bootstrap : int
        Number of bootstrap replicates used to build the null distribution
        (default: 200).
    ood_fraction_threshold : float
        OOD fraction (over ``window``) above which suppression is assessed
        (default: 0.5).
    random_state : int or None
        Seed for the bootstrap resampling (default: None).

    Required data
    -------------
    ``op_states`` kwarg — ndarray of shape (N, D), one row per step
    (same convention as ``LyapunovStabilityCheck``). Optional:
    ``uncertainties`` kwarg or per-step ``uncertainty`` history key, aligned
    1:1 with ``op_states`` rows — without it, suppression cannot be assessed
    and the check reports a degraded pass (``details["suppression_assessed"] = False``)
    rather than failing on missing data.
    """

    def __init__(
        self,
        min_history: int = 20,
        threshold_sigma: float = 3.0,
        window: int = 10,
        n_bootstrap: int = 200,
        ood_fraction_threshold: float = 0.5,
        random_state: Optional[int] = None,
    ):
        self.min_history = min_history
        self.threshold_sigma = threshold_sigma
        self.window = window
        self.n_bootstrap = n_bootstrap
        self.ood_fraction_threshold = ood_fraction_threshold
        self.random_state = random_state

    @property
    def name(self) -> str:
        return "MahalanobisOOD"

    @property
    def category(self) -> AuditCategory:
        return AuditCategory.EPISTEMIC

    def run(self, history: List[Dict[str, Any]], **kwargs) -> AuditResult:
        raw_states = kwargs.get("op_states")
        if raw_states is None:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Skipped — op_states not provided.",
            )

        states = np.asarray(raw_states, dtype=float)
        finite_mask = np.all(np.isfinite(states), axis=1)
        states = states[finite_mask]
        n = states.shape[0]

        if n < self.min_history:
            return AuditResult(
                name=self.name, passed=True, category=self.category, value=None,
                message=(
                    f"Skipped — {n} history point(s) < min_history={self.min_history}."
                ),
            )

        # Reference = history EXCLUDING the trailing window. Fitting the covariance on the
        # window itself would let OOD points inflate their own reference distribution and mask
        # themselves (the window is exactly what we're testing as possibly OOD).
        window_effective = min(self.window, n)
        n_ref = n - window_effective
        min_ref = max(3, states.shape[1] + 1)
        if n_ref < min_ref:
            return AuditResult(
                name=self.name, passed=True, category=self.category, value=None,
                message=(
                    f"Skipped — only {n_ref} reference point(s) outside the last "
                    f"{window_effective}-step window (need >= {min_ref})."
                ),
            )
        reference = states[:n_ref]

        from sklearn.covariance import LedoitWolf

        cov = LedoitWolf().fit(reference)
        diffs = states - cov.location_
        mahalanobis_series = np.sqrt(np.einsum("ij,jk,ik->i", diffs, cov.precision_, diffs))

        rng = np.random.default_rng(self.random_state)
        pooled = np.empty(self.n_bootstrap * n_ref)
        for b in range(self.n_bootstrap):
            idx = rng.integers(0, n_ref, size=n_ref)
            sample = reference[idx]
            cov_b = LedoitWolf().fit(sample)
            diffs_b = sample - cov_b.location_
            pooled[b * n_ref:(b + 1) * n_ref] = np.sqrt(
                np.einsum("ij,jk,ik->i", diffs_b, cov_b.precision_, diffs_b)
            )

        threshold = float(np.mean(pooled) + self.threshold_sigma * np.std(pooled))
        is_ood = mahalanobis_series > threshold

        window_ood = is_ood[-window_effective:]
        ood_fraction = float(np.mean(window_ood))

        details: Dict[str, Any] = {
            "mahalanobis_series": mahalanobis_series.tolist(),
            "threshold": threshold,
            "is_ood": window_ood.tolist(),
            "ood_fraction": ood_fraction,
            "window_effective": window_effective,
            "n_reference": n_ref,
            "n_bootstrap": self.n_bootstrap,
            "shrinkage_": float(cov.shrinkage_),
            "n_ood_total": int(is_ood.sum()),
        }

        if ood_fraction <= self.ood_fraction_threshold:
            return AuditResult(
                name=self.name,
                passed=True,
                category=self.category,
                value=ood_fraction,
                threshold=self.ood_fraction_threshold,
                message=f"OOD fraction (last {window_effective} steps) = {ood_fraction:.2f} — below threshold.",
                details={**details, "suppression_assessed": False},
            )

        # Uncertainty-suppression assessment — reuse the uncertainties/history["uncertainty"] convention.
        raw_u = kwargs.get("uncertainties")
        u = (
            np.asarray(raw_u, dtype=float).flatten()
            if raw_u is not None
            else np.asarray([h["uncertainty"] for h in history if "uncertainty" in h], dtype=float)
        )
        if u.size == finite_mask.size:
            u = u[finite_mask]

        if u.size != n:
            return AuditResult(
                name=self.name,
                passed=True,
                category=self.category,
                value=ood_fraction,
                threshold=self.ood_fraction_threshold,
                message=(
                    f"OOD fraction (last {window_effective} steps) = {ood_fraction:.2f} — "
                    "exceeds threshold, but uncertainty data unavailable for suppression check."
                ),
                details={**details, "suppression_assessed": False},
            )

        window_u = u[-window_effective:]
        in_window_id_u = window_u[~window_ood]
        if in_window_id_u.size >= 3:
            id_baseline_mean = float(np.mean(in_window_id_u))
            id_baseline_mode = "window"
        else:
            whole_id_u = u[~is_ood]
            if whole_id_u.size == 0:
                return AuditResult(
                    name=self.name,
                    passed=True,
                    category=self.category,
                    value=ood_fraction,
                    threshold=self.ood_fraction_threshold,
                    message=(
                        f"OOD fraction (last {window_effective} steps) = {ood_fraction:.2f} — "
                        "exceeds threshold, but no in-distribution steps remain for a baseline."
                    ),
                    details={**details, "suppression_assessed": False},
                )
            id_baseline_mean = float(np.mean(whole_id_u))
            id_baseline_mode = "whole_history"

        ood_window_mean = float(np.mean(window_u[window_ood]))
        suppressed = ood_window_mean < id_baseline_mean

        details.update({
            "suppression_assessed": True,
            "id_baseline_mode": id_baseline_mode,
            "id_baseline_mean": id_baseline_mean,
            "ood_window_mean": ood_window_mean,
        })

        return AuditResult(
            name=self.name,
            passed=not suppressed,
            category=self.category,
            value=ood_fraction,
            threshold=self.ood_fraction_threshold,
            message=(
                f"OOD fraction (last {window_effective} steps) = {ood_fraction:.2f}  "
                f"OOD-step uncertainty={ood_window_mean:.4g} vs ID baseline={id_baseline_mean:.4g} "
                f"({id_baseline_mode}) — "
                + ("suppression detected" if suppressed else "healthy exploration")
            ),
            details=details,
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
