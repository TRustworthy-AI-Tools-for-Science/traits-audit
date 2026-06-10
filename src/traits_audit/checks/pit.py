"""Probability Integral Transform uniformity check."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..base import AuditCategory, AuditCheck, AuditResult
from .calibration import _require


class PITUniformityCheck(AuditCheck):
    """
    Probability Integral Transform (PIT) uniformity test.

    For a Gaussian predictive distribution N(μ_i, σ_i²), the PIT value is::

        U_i = Φ((y_i − μ_i) / σ_i)

    If the predictive distribution is correctly specified, U_1, …, U_n are
    i.i.d. Uniform(0, 1).  This check applies a one-sample
    Kolmogorov–Smirnov test and **passes if the p-value ≥ alpha** (i.e. the
    null hypothesis of uniformity is not rejected).

    The PIT test is a more powerful distributional calibration diagnostic than
    marginal-coverage checks: it detects asymmetric miscalibration, systematic
    biases, and heavy-tailed or light-tailed predictive distributions that
    produce the same mean coverage but different PIT histograms.

    Parameters
    ----------
    alpha : float
        Significance level for the KS test (default 0.05).

    References
    ----------
    Dawid, A. P. (1984). Present position and potential developments: Some
    personal views on statistical theory and practice. *JRSS-A*, 147(2),
    278–292. https://doi.org/10.2307/2981683

    Diebold, F. X., Gunther, T. A. & Tay, A. S. (1998). Evaluating density
    forecasts with applications to financial risk management. *International
    Economic Review*, 39(4), 863–883.

    Required data (kwargs or history keys)
    ----------------------------------------
    ``y_true``, ``y_pred_mean``, ``y_pred_std``
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    @property
    def name(self) -> str:
        return "PITUniformity"

    @property
    def category(self) -> AuditCategory:
        return AuditCategory.ALEATORIC_MODEL

    def run(self, history: List[Dict[str, Any]], **kwargs) -> AuditResult:
        from scipy.stats import kstest, norm

        y_true = _require("y_true", history, kwargs)
        mu     = _require("y_pred_mean", history, kwargs)
        sigma  = _require("y_pred_std", history, kwargs)

        if any(v is None for v in (y_true, mu, sigma)):
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Skipped — y_true / y_pred_mean / y_pred_std not available.",
            )

        n = len(y_true)
        if n < 20:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Too few samples for PIT uniformity test (need ≥ 20).",
            )

        sigma_safe = np.maximum(sigma, 1e-12)
        pit = norm.cdf((y_true - mu) / sigma_safe)

        ks_stat, p_value = kstest(pit, "uniform")
        ks_stat = float(ks_stat)
        p_value = float(p_value)
        passed = p_value >= self.alpha

        return AuditResult(
            name=self.name,
            passed=passed,
            category=self.category,
            value=p_value,
            threshold=self.alpha,
            message=(
                f"PIT KS p-value = {p_value:.4f}  "
                f"(KS stat = {ks_stat:.4f},  threshold ≥ {self.alpha},  "
                f"{'PASS' if passed else 'FAIL — non-uniform PIT'})"
            ),
            details={
                "ks_statistic": ks_stat,
                "p_value": p_value,
                "n_samples": n,
                "alpha": self.alpha,
            },
        )
