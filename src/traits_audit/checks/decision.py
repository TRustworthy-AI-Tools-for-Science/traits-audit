"""Locus-in-the-chain taxonomy class: Decision Flip Rate (DFR).

See METRIC_TAXONOMY_AUDIT.md §4.7. Kamal et al. (2021)'s decision
uncertainty locates uncertainty at the point of use rather than in the
data — the complement of ``StageVarianceAttributionCheck``, which locates
it in the analysis pipeline.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..base import AuditCategory, AuditCheck, AuditResult


class DecisionFlipRateCheck(AuditCheck):
    """
    Decision Flip Rate (DFR) — resamples the reported predictive
    distribution and records the fraction of resamples under which a
    downstream decision changes (the next query point, an accept/reject, a
    ranking).

    Not degenerate under any of the three evaluation currencies
    (METRIC_TAXONOMY_AUDIT.md §7.7): a marginal predictor flips every
    decision, a point mass flips on every near-tie. Loss-aware by
    construction.

    Parameters
    ----------
    n_resamples : int
        Number of resamples drawn when ``y_pred_samples`` is not given
        directly (default 200).
    seed : int
        RNG seed for resampling.
    max_flip_rate : float or None
        Maximum acceptable flip rate. ``None`` (default) disables
        pass/fail.

    References
    ----------
    Kamal, A. et al. (2021). *J. Vis.*, 24, 1-16.

    Required data (kwargs)
    -----------------------
    ``decision_fn`` (``Callable[[np.ndarray], Hashable]``), plus either
    ``y_pred_samples`` (``(n_resamples, n_points)``, precomputed) or
    ``y_pred_mean`` + ``y_pred_std`` (resampled internally as Gaussian).
    ``decision_fn`` is a live callable, so this check is kwargs-only (no
    history route), the same status as ``LyapunovStabilityCheck``'s
    ``surrogate_fn`` kwarg.
    """

    def __init__(self, n_resamples: int = 200, seed: int = 0, max_flip_rate: float | None = None):
        self.n_resamples = n_resamples
        self.seed = seed
        self.max_flip_rate = max_flip_rate

    @property
    def name(self) -> str:
        return "DecisionFlipRate"

    @property
    def category(self) -> AuditCategory:
        return AuditCategory.LOCUS_IN_CHAIN

    def run(self, history: list[dict[str, Any]], **kwargs) -> AuditResult:
        decision_fn: Callable | None = kwargs.get("decision_fn")
        mu = kwargs.get("y_pred_mean")
        if decision_fn is None or mu is None:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Skipped — decision_fn and y_pred_mean not both provided.",
            )
        mu = np.asarray(mu, dtype=float).ravel()

        samples = kwargs.get("y_pred_samples")
        if samples is not None:
            samples = np.asarray(samples, dtype=float)
        else:
            sigma = kwargs.get("y_pred_std")
            if sigma is None:
                return AuditResult(
                    name=self.name, passed=True, category=self.category,
                    message="Skipped — need y_pred_samples or y_pred_std to resample.",
                )
            sigma = np.asarray(sigma, dtype=float).ravel()
            rng = np.random.default_rng(self.seed)
            samples = rng.normal(mu, sigma, size=(self.n_resamples, len(mu)))

        reference = decision_fn(mu)
        flips = np.mean([decision_fn(s) != reference for s in samples])
        value = float(flips)

        passed = True if self.max_flip_rate is None else value <= self.max_flip_rate
        return AuditResult(
            name=self.name,
            passed=passed,
            category=self.category,
            value=value,
            threshold=self.max_flip_rate,
            message=f"Decision flip rate = {value:.4f}  (reference decision = {reference!r})",
            details={"reference_decision": repr(reference), "n_resamples": samples.shape[0]},
        )
