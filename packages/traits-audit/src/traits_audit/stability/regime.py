"""Regime splitting and background-trend extraction for trajectories."""
from __future__ import annotations

import numpy as np

from .dmdc import detrend_trajectory

__all__ = ["split_regime_labels", "extract_background_trend"]


def split_regime_labels(n_steps: int, split_fraction: float = 0.5) -> np.ndarray:
    """Per-step regime label (0 = early, 1 = late) for a trajectory of
    length n_steps, split at split_fraction (0 < split_fraction < 1).

    Deliberately a single configurable split-fraction into exactly 2
    regimes, not changepoint detection: the empirically-confirmed
    early/late asymmetry observed in battery-forecast's Experiment 4 is a 2-regime phenomenon
    by construction (it was found BY splitting at the midpoint), so a
    changepoint-detection dependency isn't justified by what's actually
    been observed. --split-fraction lets a later analysis explore e.g. a
    30/70 split without code changes.
    """
    if n_steps < 2:
        raise ValueError(f"n_steps must be >= 2, got {n_steps}")
    cut = int(round(n_steps * split_fraction))
    cut = max(1, min(cut, n_steps - 1))  # keep both regimes non-empty
    return np.array([0] * cut + [1] * (n_steps - cut))


def extract_background_trend(raw_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a recorded raw-state trajectory into its causal-EMA background
    TREND and the residual fluctuations around it -- the same detrending
    transform applied before every DMDc fit
    (traits_audit.stability.dmdc.detrend_trajectory), but here the discarded half
    (the trend) is what we actually keep and analyze, rather than the
    residual.

    ``detrend_trajectory(raw_state, None)`` -- called ONCE over the WHOLE
    trajectory, not fresh per-window -- gives one
    continuous causal-EMA baseline (regimes=None collapses the internal
    RegimeDetrender's k-means clustering to a single group, i.e. a plain
    causal EMA with alpha=0.1 and a 20-step expanding-mean warmup). The
    trend is then just what got subtracted: trend = raw_state - residual.

    General precedent for extracting a trend as its own analyzable signal
    (not merely a nuisance to remove) is Hodrick & Prescott (1997) -- see
    its original reference list. This transform is causal/one-sided
    (unlike HP's two-sided filter), which matters less here since the
    point is a real-time-consistent trend, not a retrospective smooth.

    Returns
    -------
    (trend, residual) : both shape (T, D), same shape as raw_state;
    trend + residual == raw_state exactly (a plain subtraction identity,
    not an approximation).
    """
    raw_state = np.asarray(raw_state, dtype=np.float64)
    residual = detrend_trajectory(raw_state, None)
    trend = raw_state - residual
    return trend, residual
