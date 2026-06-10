"""Proper scoring rules for probabilistic regression: CRPS, NLL, Interval Score."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from ..base import AuditCategory, AuditCheck, AuditResult
from .calibration import _require


class CRPSCheck(AuditCheck):
    """
    Continuous Ranked Probability Score (CRPS) for Gaussian forecasts.

    CRPS is a proper scoring rule that simultaneously rewards calibration and
    sharpness.  For a Gaussian predictive distribution N(μ_i, σ_i²) and
    observation y_i the closed-form expression is::

        CRPS_i = σ_i · [2φ(z_i) + z_i(2Φ(z_i) − 1) − 1/√π]

    where z_i = (y_i − μ_i) / σ_i and φ, Φ are the standard-normal PDF/CDF.
    **Lower values are better.**  A perfectly calibrated Gaussian model achieves
    ``mean(CRPS) ≈ mean(σ) / √π ≈ 0.564 · mean(σ)``, which is stored in
    ``details["crps_reference"]`` for comparison.

    .. note::
       CRPS is scale-dependent (proportional to σ).  The default
       ``threshold=None`` means the check always passes — it reports the value
       for monitoring and trending purposes.  Set a problem-specific threshold
       to enable pass/fail detection (e.g. ``threshold=0.7 * typical_sigma``).

    Parameters
    ----------
    threshold : float or None
        Maximum acceptable mean CRPS.  ``None`` (default) disables pass/fail.

    References
    ----------
    Gneiting, T. & Raftery, A. E. (2007). Strictly proper scoring rules,
    prediction, and estimation. *JASA*, 102(477), 359–378.
    https://doi.org/10.1198/016214506000001437

    Gneiting, T., Balabdaoui, F. & Raftery, A. E. (2007). Probabilistic
    forecasts, calibration and sharpness. *JRSS-B*, 69(2), 243–268.

    Required data (kwargs or history keys)
    ----------------------------------------
    ``y_true``, ``y_pred_mean``, ``y_pred_std``
    """

    def __init__(self, threshold: Optional[float] = None):
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "CRPS"

    @property
    def category(self) -> AuditCategory:
        return AuditCategory.ALEATORIC_MODEL

    def run(self, history: List[Dict[str, Any]], **kwargs) -> AuditResult:
        from scipy.stats import norm

        y_true = _require("y_true", history, kwargs)
        mu     = _require("y_pred_mean", history, kwargs)
        sigma  = _require("y_pred_std", history, kwargs)

        if any(v is None for v in (y_true, mu, sigma)):
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Skipped — y_true / y_pred_mean / y_pred_std not available.",
            )

        n = len(y_true)
        if n < 5:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Too few samples for CRPS estimate (need ≥ 5).",
            )

        sigma_safe = np.maximum(sigma, 1e-12)
        z = (y_true - mu) / sigma_safe
        crps = sigma_safe * (2.0 * norm.pdf(z) + z * (2.0 * norm.cdf(z) - 1.0) - 1.0 / math.sqrt(math.pi))
        mean_crps = float(np.mean(crps))
        crps_reference = float(np.mean(sigma_safe)) / math.sqrt(math.pi)

        if self.threshold is None:
            passed = True
            thr_str = "none (reporting only)"
        else:
            passed = mean_crps <= self.threshold
            thr_str = f"{self.threshold:.4f}"

        return AuditResult(
            name=self.name,
            passed=passed,
            category=self.category,
            value=mean_crps,
            threshold=self.threshold,
            message=(
                f"Mean CRPS = {mean_crps:.4f}  "
                f"(calibrated reference ≈ {crps_reference:.4f},  threshold = {thr_str})"
            ),
            details={
                "mean_crps": mean_crps,
                "crps_reference": crps_reference,
                "n_samples": n,
            },
        )


class NegativeLogLikelihoodCheck(AuditCheck):
    """
    Gaussian negative log-likelihood (NLL) as a proper scoring rule.

    For a Gaussian predictive distribution N(μ_i, σ_i²) and observation y_i::

        NLL_i = 0.5 · log(2π) + log(σ_i) + 0.5 · ((y_i − μ_i) / σ_i)²

    **Lower values are better.**  A perfectly calibrated Gaussian model with
    unit residuals achieves NLL ≈ 0.5 · log(2π) + 0.5 ≈ 1.419, stored in
    ``details["nll_reference"]`` for comparison.  Overconfident models
    (σ too small) produce large z² terms that drive NLL higher.

    .. note::
       NLL is scale-dependent (the log(σ) term depends on the absolute scale of
       predictions).  The default ``threshold=None`` means the check always
       passes — it reports the value for monitoring and trending purposes.
       Set a problem-specific threshold to enable pass/fail detection.

    Parameters
    ----------
    threshold : float or None
        Maximum acceptable mean NLL.  ``None`` (default) disables pass/fail.

    References
    ----------
    Good, I. J. (1952). Rational decisions. *JRSS-B*, 14(1), 107–114.

    Lakshminarayanan, B., Pritzel, A. & Blundell, C. (2017). Simple and
    scalable predictive uncertainty estimation using deep ensembles. *NeurIPS*.

    Required data (kwargs or history keys)
    ----------------------------------------
    ``y_true``, ``y_pred_mean``, ``y_pred_std``
    """

    def __init__(self, threshold: Optional[float] = None):
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "NegativeLogLikelihood"

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

        n = len(y_true)
        if n < 5:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Too few samples for NLL estimate (need ≥ 5).",
            )

        sigma_safe = np.maximum(sigma, 1e-12)
        z = (y_true - mu) / sigma_safe
        nll_per_sample = 0.5 * math.log(2.0 * math.pi) + np.log(sigma_safe) + 0.5 * z ** 2
        mean_nll = float(np.mean(nll_per_sample))
        nll_reference = 0.5 * math.log(2.0 * math.pi) + float(np.mean(np.log(sigma_safe))) + 0.5

        if self.threshold is None:
            passed = True
            thr_str = "none (reporting only)"
        else:
            passed = mean_nll <= self.threshold
            thr_str = f"{self.threshold:.4f}"

        return AuditResult(
            name=self.name,
            passed=passed,
            category=self.category,
            value=mean_nll,
            threshold=self.threshold,
            message=(
                f"Mean NLL = {mean_nll:.4f}  "
                f"(calibrated reference ≈ {nll_reference:.4f},  threshold = {thr_str})"
            ),
            details={
                "mean_nll": mean_nll,
                "nll_reference": nll_reference,
                "n_samples": n,
            },
        )


class IntervalScoreCheck(AuditCheck):
    """
    Winkler interval score — proper scoring rule for interval forecasts.

    For a prediction interval [l_i, u_i] at nominal coverage 1 − α::

        IS_i = (u_i − l_i)
               + (2/α) · max(l_i − y_i, 0)
               + (2/α) · max(y_i − u_i, 0)

    Intervals are constructed as [μ_i ± z_{1−α/2} · σ_i].  **Lower is better.**
    The score penalises both unnecessary width and coverage failures jointly.
    A perfectly calibrated Gaussian model achieves an expected score of
    ``2 · z_{1-α/2} · mean(σ) + 2 · φ(z) / α`` (stored in
    ``details["is_reference"]``).

    .. note::
       Interval Score is scale-dependent (proportional to σ).  The default
       ``threshold=None`` means the check always passes — it reports the value
       for monitoring and trending purposes.  Set a problem-specific threshold
       to enable pass/fail detection.

    Parameters
    ----------
    alpha : float
        Significance level (default 0.1 → 90 % intervals).
    threshold : float or None
        Maximum acceptable mean interval score.  ``None`` (default) disables
        pass/fail.

    References
    ----------
    Winkler, R. L. (1972). A decision-theoretic approach to interval estimation.
    *JASA*, 67(337), 187–191.

    Gneiting, T. & Raftery, A. E. (2007). Strictly proper scoring rules,
    prediction, and estimation. *JASA*, 102(477), 359–378.
    https://doi.org/10.1198/016214506000001437

    Required data (kwargs or history keys)
    ----------------------------------------
    ``y_true``, ``y_pred_mean``, ``y_pred_std``
    """

    def __init__(self, alpha: float = 0.1, threshold: Optional[float] = None):
        self.alpha = alpha
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "IntervalScore"

    @property
    def category(self) -> AuditCategory:
        return AuditCategory.ALEATORIC_MODEL

    def run(self, history: List[Dict[str, Any]], **kwargs) -> AuditResult:
        from scipy.stats import norm

        y_true = _require("y_true", history, kwargs)
        mu     = _require("y_pred_mean", history, kwargs)
        sigma  = _require("y_pred_std", history, kwargs)

        if any(v is None for v in (y_true, mu, sigma)):
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Skipped — y_true / y_pred_mean / y_pred_std not available.",
            )

        n = len(y_true)
        if n < 5:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Too few samples for Interval Score estimate (need ≥ 5).",
            )

        sigma_safe = np.maximum(sigma, 1e-12)
        z_crit = float(norm.ppf(1.0 - self.alpha / 2.0))
        lo = mu - z_crit * sigma_safe
        hi = mu + z_crit * sigma_safe

        width = hi - lo
        penalty_lo = np.maximum(lo - y_true, 0.0)
        penalty_hi = np.maximum(y_true - hi, 0.0)
        is_per_sample = width + (2.0 / self.alpha) * (penalty_lo + penalty_hi)
        mean_is = float(np.mean(is_per_sample))

        mean_sigma = float(np.mean(sigma_safe))
        is_reference = mean_sigma * (2.0 * z_crit + 2.0 * float(norm.pdf(z_crit)) / self.alpha)
        if self.threshold is None:
            passed = True
            thr_str = "none (reporting only)"
        else:
            passed = mean_is <= self.threshold
            thr_str = f"{self.threshold:.4f}"

        return AuditResult(
            name=self.name,
            passed=passed,
            category=self.category,
            value=mean_is,
            threshold=self.threshold,
            message=(
                f"Mean Interval Score = {mean_is:.4f}  "
                f"(calibrated reference ≈ {is_reference:.4f},  "
                f"alpha = {self.alpha},  threshold = {thr_str})"
            ),
            details={
                "mean_is": mean_is,
                "is_reference": is_reference,
                "alpha": self.alpha,
                "z_critical": z_crit,
                "n_samples": n,
            },
        )
