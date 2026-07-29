"""Locus-in-the-chain taxonomy class: Stage Variance Attribution (SVA)
pipeline wrapper.

See METRIC_TAXONOMY_AUDIT.md §4.7. The real computation is
:func:`traits_audit.pipeline_attribution.run_stage_variance_attribution` —
a standalone function, since its natural output is a per-stage vector, not
a scalar. This module is a thin, lossy ``AuditCheck`` wrapper around it for
pipeline/report integration.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..base import AuditCategory, AuditCheck, AuditResult
from ..pipeline_attribution import run_stage_variance_attribution


class StageVarianceAttributionCheck(AuditCheck):
    """
    Thin pipeline adapter around
    :func:`traits_audit.pipeline_attribution.run_stage_variance_attribution`.
    See that function for the real computation (Saltelli/Jansen two-matrix
    Sobol estimator over analysis-chain stages).

    ``value = max(interaction_gap)`` — the headline "how bad is the
    additive-per-source-budget assumption" scalar. The full per-stage
    ``first_order``/``total_effect``/``interaction_gap`` vectors are always
    in ``details``, never dropped: METRIC_TAXONOMY_AUDIT.md §6.2 makes the
    total-minus-first-order gap the criterion for when an additive budget is
    inapplicable, and an implementation reporting only first-order indices
    reproduces the exact failure this metric exists to detect.

    Parameters
    ----------
    n_mc : int
        Monte Carlo sample count (default 1024).
    seed : int
        RNG seed.
    max_interaction_gap : float or None
        Maximum acceptable ``max(interaction_gap)``. ``None`` (default)
        disables pass/fail.

    References
    ----------
    Sobol, I. M. (2001). *Math. Comput. Simul.*, 55(1-3), 271-280.
    Saltelli, A. et al. (2010). *Comput. Phys. Commun.*, 181(2), 259-270.
    Jansen, M. J. W. (1999). *Comput. Phys. Commun.*, 117(1-2), 35-43.
    Pipeline framing: Bonneau, G. P. et al. (2014). Overview and
    state-of-the-art of uncertainty visualization. In *Scientific
    Visualization*. Springer.
    See also: van den Boorn, D. et al. (2022). *Adv. Theory Simul.*,
    5(9), 2200615.

    Required data (kwargs)
    -----------------------
    ``chain_fn`` (``Callable[..., float]``), ``stages``
    (``List[traits_audit.pipeline_attribution.StageUncertainty]``). Kwargs-only
    (live callables cannot be JSON-serialized or carried per-step in
    history) — only the resulting numeric vectors go into ``details``.
    """

    def __init__(self, n_mc: int = 1024, seed: int = 0, max_interaction_gap: Optional[float] = None):
        self.n_mc = n_mc
        self.seed = seed
        self.max_interaction_gap = max_interaction_gap

    @property
    def name(self) -> str:
        return "StageVarianceAttribution"

    @property
    def category(self) -> AuditCategory:
        return AuditCategory.LOCUS_IN_CHAIN

    def run(self, history: List[Dict[str, Any]], **kwargs) -> AuditResult:
        chain_fn = kwargs.get("chain_fn")
        stages = kwargs.get("stages")

        if chain_fn is None or not stages:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Skipped — chain_fn and stages not both provided.",
            )

        result = run_stage_variance_attribution(chain_fn, stages, n_mc=self.n_mc, seed=self.seed)
        value = float(result.interaction_gap.max())

        passed = True if self.max_interaction_gap is None else value <= self.max_interaction_gap
        return AuditResult(
            name=self.name,
            passed=passed,
            category=self.category,
            value=value,
            threshold=self.max_interaction_gap,
            message=f"max(interaction_gap) = {value:.4f}  across {len(result.stage_names)} stages",
            details={
                "stage_names": result.stage_names,
                "first_order": result.first_order.tolist(),
                "total_effect": result.total_effect.tolist(),
                "interaction_gap": result.interaction_gap.tolist(),
                "n_mc": result.n_mc,
            },
        )
