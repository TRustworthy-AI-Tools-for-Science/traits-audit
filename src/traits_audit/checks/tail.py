"""Cross-cutting: tail index (not a taxonomy-class member — see docstring)."""
from __future__ import annotations

from typing import Any

import numpy as np

from ..base import AuditCategory, AuditCheck, AuditResult
from .calibration import _require


class TailIndexCheck(AuditCheck):
    """
    Hill estimator of the tail index alpha-hat on normalized residuals
    ``z = (y - mu) / sigma``, flagging when ``alpha_hat < 2`` (variance does
    not exist).

    Bailey (2017) found tail indices of 1.6-3.5 across ~41,000 published
    measurements — below 2, variance is not merely large but undefined.
    Every variance-based object in this suite (sharpness, coverage,
    calibration RMSCE, the quadrature combination implicit in any sigma)
    presupposes finite second moments. This check is cheap (roughly ten
    lines of actual computation) and, per METRIC_TAXONOMY_AUDIT.md §5,
    "arguably the highest-value single addition" — it tells a user whether
    the rest of the suite means anything.

    .. note::
       This is explicitly a cross-cutting diagnostic, **not** a member of
       any of the eight taxonomy classes in METRIC_TAXONOMY_AUDIT.md §3 (it
       conditions whether the variance-based metrics in those classes are
       valid, rather than discriminating a class itself) — ``UNKNOWN`` is
       therefore the correct category here, not a fallback.

    Parameters
    ----------
    tail_fraction : float
        Fraction of the (sorted) residuals used as the tail sample for the
        Hill estimator (default 0.1).
    alpha_threshold : float
        Minimum acceptable alpha-hat (default 2.0 — the finite-variance
        boundary).

    References
    ----------
    Bailey, D. C. (2017). Not normal: the uncertainties of scientific
    measurements. *R. Soc. Open Sci.*, 4:160600.
    Hill, B. M. (1975). A simple general approach to inference about the
    tail of a distribution. *Ann. Stat.*, 3(5), 1163-1174.

    Required data (kwargs or history keys)
    ----------------------------------------
    ``y_true``, ``y_pred_mean``, ``y_pred_std``
    """

    def __init__(self, tail_fraction: float = 0.1, alpha_threshold: float = 2.0):
        self.tail_fraction = tail_fraction
        self.alpha_threshold = alpha_threshold

    @property
    def name(self) -> str:
        return "TailIndex"

    @property
    def category(self) -> AuditCategory:
        return AuditCategory.UNKNOWN

    def run(self, history: list[dict[str, Any]], **kwargs) -> AuditResult:
        y_true = _require("y_true", history, kwargs)
        mu = _require("y_pred_mean", history, kwargs)
        sigma = _require("y_pred_std", history, kwargs)

        if any(v is None for v in (y_true, mu, sigma)):
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Skipped — y_true / y_pred_mean / y_pred_std not available.",
            )

        n = len(y_true)
        if n < 50:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message=f"Skipped — too few samples for a Hill estimate (n={n} < 50).",
            )

        sigma_safe = np.maximum(sigma, 1e-12)
        z = (y_true - mu) / sigma_safe
        abs_z = np.sort(np.abs(z))[::-1]

        k = max(int(self.tail_fraction * n), 10)
        k = min(k, n - 1)
        threshold_val = abs_z[k]
        if threshold_val <= 0:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Skipped — degenerate tail (threshold value is zero).",
            )

        log_ratios = np.log(abs_z[:k] / threshold_val)
        mean_log_ratio = float(np.mean(log_ratios))
        if mean_log_ratio <= 0:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Skipped — degenerate tail (all tail values equal threshold; Hill estimator undefined).",
            )
        alpha_hat = 1.0 / mean_log_ratio

        return AuditResult(
            name=self.name,
            passed=alpha_hat >= self.alpha_threshold,
            category=self.category,
            value=alpha_hat,
            threshold=self.alpha_threshold,
            message=(
                f"Hill alpha_hat = {alpha_hat:.3f} "
                + ("(variance undefined — alpha < 2)" if alpha_hat < 2.0 else "(finite variance)")
            ),
            details={"k": k, "n_samples": n, "tail_fraction": self.tail_fraction},
        )
