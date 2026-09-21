"""
Generic moving-block bootstrap for a scalar statistic of an autocorrelated
1-D series.

This is a general-purpose extraction of the moving-block resampling pattern
already used (in a DMDc/Gramian-specific form) by ``trajectory.py``'s
``_block_bootstrap_stats``. It is deliberately NOT used to refactor that
working code — this module exists so new statistics (currently just
``ResidualPersistenceHalfLifeCheck``) can reuse the same block-bootstrap
machinery without touching the DMDc-specific implementation.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np


def moving_block_bootstrap_ci(
    values: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    block_len: int = 8,
    n_boot: int = 200,
    seed: int = 42,
) -> tuple[float, float] | None:
    """95% CI for ``stat_fn(values)`` via moving-block bootstrap, which
    respects the series' autocorrelation (unlike i.i.d. resampling).
    Returns ``None`` if ``len(values) <= block_len``."""
    values = np.asarray(values, dtype=float).ravel()
    n = len(values)
    if n <= block_len:
        return None

    rng = np.random.default_rng(seed)
    n_blocks_needed = int(np.ceil(n / block_len))
    starts_possible = n - block_len + 1

    stats = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, starts_possible, size=n_blocks_needed)
        sample = np.concatenate([values[s:s + block_len] for s in starts])[:n]
        stats[b] = stat_fn(sample)

    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)
