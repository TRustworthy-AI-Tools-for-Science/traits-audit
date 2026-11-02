"""
Stage Variance Attribution (SVA) — locus-in-the-chain taxonomy class
(METRIC_TAXONOMY_AUDIT.md §4.7).

This is a standalone diagnostic, not an ``AuditCheck``: its natural output is
a per-stage vector (first-order and total-effect Sobol indices, plus the
interaction gap between them), not a single scalar. This follows the
precedent already set by ``_mechanism_check.py`` — a diagnostic whose output
doesn't fit the ``AuditCheck.run(history, **kwargs) -> AuditResult`` contract
is a standalone function, not forced into the ABC. A thin, lossy
``AuditCheck`` wrapper (``StageVarianceAttributionCheck``, reporting only
``max(interaction_gap)``) lives in ``checks/attribution.py`` for pipeline
convenience; this module is the real computation.

Perturb each analysis-chain stage's inputs/parameters by their own stated
uncertainty via Monte Carlo (JCGM 101 style), freeze all other stages, and
report first-order (S_i) AND total-effect (S_Ti) Sobol indices of the final
scalar with respect to stage — using the Saltelli (2010) estimator for S_i
and the Jansen (1999) estimator for S_Ti, the standard pairing (as used by
e.g. SALib).

Reporting ``interaction_gap = S_Ti - S_i`` is not optional: per
METRIC_TAXONOMY_AUDIT.md §6.2, this gap is the criterion for when an additive
per-source uncertainty budget is inapplicable, and §6.3 records that
attribution fails entirely in the tail (interaction dominates as first-order
indices vanish) — an implementation reporting only first-order indices
reproduces the exact failure this metric exists to detect.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class StageUncertainty:
    """One analysis-chain stage's perturbation source.

    ``sample_fn(rng) -> value`` draws one perturbed value of this stage's
    inputs/parameters (e.g. a baseline-subtraction offset drawn from its own
    stated uncertainty). ``value`` is passed to ``chain_fn`` as the keyword
    argument named ``name`` on each Monte Carlo draw.
    """
    name: str
    sample_fn: Callable[[np.random.Generator], Any]


@dataclass
class SVAResult:
    stage_names: list[str]
    first_order: np.ndarray
    total_effect: np.ndarray
    interaction_gap: np.ndarray
    n_mc: int


def _draw_matrix(stages: list[StageUncertainty], n_mc: int, rng: np.random.Generator) -> np.ndarray:
    """(n_mc, n_stages) object array of independently drawn stage values."""
    k = len(stages)
    M = np.empty((n_mc, k), dtype=object)
    for j, stage in enumerate(stages):
        for i in range(n_mc):
            M[i, j] = stage.sample_fn(rng)
    return M


def _evaluate(chain_fn: Callable[..., float], stages: list[StageUncertainty], M: np.ndarray) -> np.ndarray:
    names = [s.name for s in stages]
    out = np.empty(M.shape[0], dtype=float)
    for i in range(M.shape[0]):
        out[i] = chain_fn(**dict(zip(names, M[i], strict=False)))
    return out


def run_stage_variance_attribution(
    chain_fn: Callable[..., float],
    stages: list[StageUncertainty],
    n_mc: int = 1024,
    seed: int = 0,
) -> SVAResult:
    """Saltelli/Jansen two-matrix Monte Carlo estimator of first-order and
    total-effect Sobol indices for ``chain_fn``'s output with respect to each
    stage in ``stages``.

    ``chain_fn`` must accept one keyword argument per stage name and return a
    scalar. Draws two independent (n_mc, n_stages) sample matrices A, B; for
    each stage i, forms AB^i (A with column i replaced by B's column i);
    evaluates ``chain_fn`` on all rows of A, B, and each AB^i.
    """
    rng = np.random.default_rng(seed)
    k = len(stages)
    names = [s.name for s in stages]

    A = _draw_matrix(stages, n_mc, rng)
    B = _draw_matrix(stages, n_mc, rng)

    f_A = _evaluate(chain_fn, stages, A)
    f_B = _evaluate(chain_fn, stages, B)

    var_total = float(np.var(np.concatenate([f_A, f_B]), ddof=0))
    var_total_safe = var_total if var_total > 0 else 1e-300

    first_order = np.empty(k)
    total_effect = np.empty(k)

    for i in range(k):
        AB_i = A.copy()
        AB_i[:, i] = B[:, i]
        f_ABi = _evaluate(chain_fn, stages, AB_i)

        # Saltelli (2010) first-order estimator.
        first_order[i] = float(np.mean(f_B * (f_ABi - f_A))) / var_total_safe
        # Jansen (1999) total-effect estimator.
        total_effect[i] = float(np.mean((f_A - f_ABi) ** 2)) / (2.0 * var_total_safe)

    interaction_gap = total_effect - first_order

    return SVAResult(
        stage_names=names,
        first_order=first_order,
        total_effect=total_effect,
        interaction_gap=interaction_gap,
        n_mc=n_mc,
    )
