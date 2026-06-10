"""Conformal prediction coverage validity check."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..base import AuditCategory, AuditCheck, AuditResult
from .calibration import _require


class ConformalCoverageCheck(AuditCheck):
    """
    Distribution-free interval validity check based on split conformal prediction.

    Normalised nonconformity scores ``s_i = |y_i − ŷ_i| / σ_i`` are computed
    for each sample.  The **conformal quantile** q̂ at the target coverage level
    ``1−α`` is estimated from their empirical distribution with the finite-sample
    correction:

    .. math::

        \\hat{q} = \\operatorname{Quantile}\\!\\left(s,\\;
                   \\frac{\\lceil (n+1)(1-\\alpha) \\rceil}{n}\\right)

    q̂ is then compared to the expected Gaussian critical value ``z_{1−α/2}``
    (the value one would use for perfectly Gaussian predictions) to form the
    **calibration ratio**:

    .. math::

        r = \\hat{q} \\,/\\, z_{1-\\alpha/2}

    * **r ≈ 1** — model is well-calibrated; parametric intervals are already valid.
    * **r > 1** — model is overconfident; intervals need widening by factor r.
    * **r < 1** — model is over-dispersed; intervals are wider than necessary.

    The check reports r as its primary value and passes when ``r ≤ max_q_ratio``,
    flagging overconfidence.  The accompanying empirical coverage at q̂ and the
    raw q̂ value are stored in ``details`` for diagnostics.

    Unlike parametric calibration checks, this method makes no distributional
    assumption and provides a finite-sample coverage guarantee when applied to
    an independent calibration set (exchangeability suffices).

    Parameters
    ----------
    target_coverage : float
        Target marginal coverage level ``1−α`` (default: 0.9).
    max_q_ratio : float
        Maximum acceptable conformal-to-Gaussian quantile ratio r = q̂/z_{1−α/2}.
        Values significantly above 1 indicate overconfidence (default: 1.5).

    Required data (kwargs or history keys)
    ----------------------------------------
    ``y_true``, ``y_pred_mean``, ``y_pred_std``

    References
    ----------
    Angelopoulos, A. N. & Bates, S. (2021). A gentle introduction to conformal
    prediction and distribution-free uncertainty quantification.
    *arXiv:2107.07511*.

    Vovk, V., Gammerman, A. & Shafer, G. (2005). *Algorithmic Learning in a
    Statistical World*. Springer.
    """

    def __init__(self, target_coverage: float = 0.9, max_q_ratio: float = 1.5):
        self.target_coverage = target_coverage
        self.max_q_ratio = max_q_ratio

    @property
    def name(self) -> str:
        return "ConformalCoverage"

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
        if n < 10:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Too few samples for conformal coverage estimate (need ≥ 10).",
            )

        alpha = 1.0 - self.target_coverage
        scores = np.abs(y_true - mu) / np.maximum(sigma, 1e-12)

        # Finite-sample corrected conformal quantile (Vovk et al. 2005)
        level = min(float(np.ceil((n + 1) * (1.0 - alpha)) / n), 1.0)
        try:
            q_hat = float(np.quantile(scores, level, method="higher"))
        except TypeError:  # NumPy < 1.22
            q_hat = float(np.quantile(scores, level, interpolation="higher"))
        empirical_coverage = float(np.mean(scores <= q_hat))

        # Ratio of conformal quantile to expected Gaussian critical value
        z_expected = float(norm.ppf(1.0 - alpha / 2.0))
        q_ratio = q_hat / max(z_expected, 1e-12)

        passed = q_ratio <= self.max_q_ratio

        return AuditResult(
            name=self.name,
            passed=passed,
            category=self.category,
            value=q_ratio,
            threshold=self.max_q_ratio,
            message=(
                f"Conformal q-ratio = {q_ratio:.3f}  "
                f"(q̂ = {q_hat:.3f},  z_{{1-α/2}} = {z_expected:.3f},  "
                f"coverage = {empirical_coverage:.1%},  "
                f"target ≤ {self.max_q_ratio:.2f})"
            ),
            details={
                "q_hat": q_hat,
                "z_expected": z_expected,
                "q_ratio": q_ratio,
                "empirical_coverage": empirical_coverage,
                "target_coverage": self.target_coverage,
                "n_samples": n,
                "alpha": alpha,
            },
        )
