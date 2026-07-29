"""Proper scoring rules for probabilistic regression: CRPS, NLL, Interval Score."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import norm as _norm

from ..base import AuditCategory, AuditCheck, AuditResult
from .calibration import _require


def _report_only(threshold: Optional[float], value: float) -> "tuple[bool, str]":
    """Return (passed, threshold_str) for checks that are report-only when threshold=None."""
    if threshold is None:
        return True, "none (reporting only)"
    return value <= threshold, f"{threshold:.4f}"


def _gaussian_nll(residual: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Per-sample Gaussian NLL: 0.5*log(2π) + log(σ) + 0.5*(r/σ)²."""
    sigma_safe = np.maximum(sigma, 1e-12)
    return 0.5 * math.log(2.0 * math.pi) + np.log(sigma_safe) + 0.5 * (residual / sigma_safe) ** 2


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
        crps = sigma_safe * (2.0 * _norm.pdf(z) + z * (2.0 * _norm.cdf(z) - 1.0) - 1.0 / math.sqrt(math.pi))
        mean_crps = float(np.mean(crps))
        crps_reference = float(np.mean(sigma_safe)) / math.sqrt(math.pi)
        passed, thr_str = _report_only(self.threshold, mean_crps)

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
        nll_per_sample = _gaussian_nll(y_true - mu, sigma_safe)
        mean_nll = float(np.mean(nll_per_sample))
        nll_reference = 0.5 * math.log(2.0 * math.pi) + float(np.mean(np.log(sigma_safe))) + 0.5
        passed, thr_str = _report_only(self.threshold, mean_nll)

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
       **Read this score as Gneiting & Raftery's sharpness-subject-to-
       calibration answer, not as an independent row next to a separate
       sharpness metric.** Gneiting & Raftery (2007) formulate the goal of
       probabilistic forecasting as "maximize sharpness subject to
       calibration" — reporting sharpness and calibration as two independent
       numbers admits degenerate readings (their own worked example: three
       interval forecasts all had close-to-nominal coverage, the narrowest
       of the three scored best on sharpness alone, yet a *wider* interval
       won on interval score because the narrowest one collapsed toward a
       point forecast exactly where the true conditional variance was
       highest). The Winkler interval score combines width and coverage
       into one proper number and is the closest thing this package
       implements to that constrained figure — prefer it over reporting
       sharpness and coverage separately when a single headline number is
       needed.

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
        z_crit = float(_norm.ppf(1.0 - self.alpha / 2.0))
        lo = mu - z_crit * sigma_safe
        hi = mu + z_crit * sigma_safe

        width = hi - lo
        penalty_lo = np.maximum(lo - y_true, 0.0)
        penalty_hi = np.maximum(y_true - hi, 0.0)
        is_per_sample = width + (2.0 / self.alpha) * (penalty_lo + penalty_hi)
        mean_is = float(np.mean(is_per_sample))

        mean_sigma = float(np.mean(sigma_safe))
        is_reference = mean_sigma * (2.0 * z_crit + 2.0 * float(_norm.pdf(z_crit)) / self.alpha)
        passed, thr_str = _report_only(self.threshold, mean_is)

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


class ScoreDecompositionCheck(AuditCheck):
    """
    Proper-score calibration/refinement decomposition — a principled
    replacement for the ECE family, built from the Gaussian NLL this suite
    already computes.

    Guo et al. (2017) established ECE as the conventional calibration
    metric, but it is not a proper scoring rule, has no corresponding
    proper-score decomposition, is discontinuous, is not distribution-free
    estimable, and its binned estimator lower-bounds the true quantity with
    an unbounded gap (see ``CalibrationErrorCheck``'s references). This
    check instead applies the DeGroot & Fienberg (1983) calibration/
    refinement decomposition directly to Gaussian NLL, binning by predicted
    sigma the same way ``ENCECheck`` does:

    - ``UNC`` — NLL of the "climatological" constant forecast
      ``N(mean(y_true), std(y_true))``: the best you could do knowing
      nothing but the marginal outcome distribution.
    - Bin by predicted sigma into ``n_bins`` groups; **recalibrate** each bin
      to its own empirical residual mean/std (the best a forecaster with
      exactly this bin's information could do).
    - ``CAL`` — mean NLL improvement from recalibrating: how much of the
      original score is attributable to miscalibration.
    - ``REF`` — ``UNC`` minus the recalibrated NLL: how much of the
      climatological uncertainty the model's binning genuinely resolves.

    ``details["identity_residual"]`` reports
    ``mean(NLL_original) - (UNC - REF + CAL)``, which is exactly 0 by
    construction — a sanity check a caller can inspect.

    ``value = CAL``. **Lower is better** (less miscalibration).

    Parameters
    ----------
    n_bins : int
        Number of sigma-ordered bins (default 10).
    cal_threshold : float or None
        Maximum acceptable CAL. ``None`` (default) disables pass/fail —
        reports for monitoring, like ``CRPSCheck``.

    References
    ----------
    DeGroot, M. H. & Fienberg, S. E. (1983). The comparison and evaluation
    of forecasters. *The Statistician*, 32(1), 12-22.
    Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. (2017). On calibration
    of modern neural networks. *ICML*.

    Required data (kwargs or history keys)
    ----------------------------------------
    ``y_true``, ``y_pred_mean``, ``y_pred_std``
    """

    def __init__(self, n_bins: int = 10, cal_threshold: Optional[float] = None):
        self.n_bins = n_bins
        self.cal_threshold = cal_threshold

    @property
    def name(self) -> str:
        return "ScoreDecomposition"

    @property
    def category(self) -> AuditCategory:
        return AuditCategory.ALEATORIC_MODEL

    def run(self, history: List[Dict[str, Any]], **kwargs) -> AuditResult:
        y_true = _require("y_true", history, kwargs)
        mu = _require("y_pred_mean", history, kwargs)
        sigma = _require("y_pred_std", history, kwargs)

        if any(v is None for v in (y_true, mu, sigma)):
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Skipped — y_true / y_pred_mean / y_pred_std not available.",
            )

        n = len(y_true)
        if n < 3 * self.n_bins:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message=f"Skipped — need >= {3 * self.n_bins} samples for {self.n_bins} bins (n={n}).",
            )

        sigma_c = float(np.std(y_true, ddof=0))
        if sigma_c <= 0:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Skipped — y_true has zero variance; climatological forecast undefined.",
            )
        mu_c = float(np.mean(y_true))
        unc = float(np.mean(_gaussian_nll(y_true - mu_c, np.full(n, sigma_c))))

        residual = y_true - mu
        nll_original = _gaussian_nll(residual, sigma)
        overall_nll_original = float(np.mean(nll_original))

        order = np.argsort(sigma)
        weighted_orig, weighted_recal, total = 0.0, 0.0, 0
        for chunk in np.array_split(order, self.n_bins):
            if chunk.size == 0:
                continue
            bin_r = residual[chunk]
            bin_std = float(np.std(bin_r, ddof=0))
            if bin_std <= 0:
                bin_std = 1e-12
            recal_r = bin_r - np.mean(bin_r)
            nll_orig_bin = float(np.mean(nll_original[chunk]))
            nll_recal_bin = float(np.mean(_gaussian_nll(recal_r, np.full(chunk.size, bin_std))))
            weighted_orig += nll_orig_bin * chunk.size
            weighted_recal += nll_recal_bin * chunk.size
            total += chunk.size

        mean_recal = weighted_recal / total
        mean_orig = weighted_orig / total
        cal = mean_orig - mean_recal
        ref = unc - mean_recal
        identity_residual = overall_nll_original - (unc - ref + cal)

        passed = True if self.cal_threshold is None else cal <= self.cal_threshold
        return AuditResult(
            name=self.name,
            passed=passed,
            category=self.category,
            value=cal,
            threshold=self.cal_threshold,
            message=f"CAL={cal:.4f}  REF={ref:.4f}  UNC={unc:.4f}",
            details={
                "UNC": unc,
                "REF": ref,
                "CAL": cal,
                "identity_residual": identity_residual,
                "n_bins": self.n_bins,
            },
        )
