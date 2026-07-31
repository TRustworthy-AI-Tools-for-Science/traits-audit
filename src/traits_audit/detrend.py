"""Per-regime EMA detrending of a streaming state trajectory.

Pure numpy/scipy.  No dependency on any GPR surrogate, battery oracle, or
plotting library — callers pass plain ``(state, regime)`` pairs in, one at a
time, and get a detrended state back.

References
----------
[HHK20]  Hirsh, S. M., Harris, K. D., Kutz, J. N., and Brunton, B. W. (2020).
         Centering data improves the dynamic mode decomposition. SIAM J.
         Applied Dynamical Systems, 19(3), 1920-1955. doi:10.1137/19M1289881.
         Shows DMD on centered (mean-subtracted) data is not equivalent to
         DMD on raw data — motivates detrending *before* fitting
         :func:`traits_audit.dmdc.fit_dmdc`/:func:`traits_audit.dmdc.fit_dmdc_pairs`,
         not after.
[SKA24]  Seenivasaharagavan, G. S., Korda, M., Arbabi, H., and Mezić, I.
         (2024). Clarifying the effect of mean subtraction on Dynamic Mode
         Decomposition. SIAM J. Applied Dynamical Systems.
         doi:10.1137/23M1569940. Formalizes when/why mean subtraction
         changes the recovered DMD spectrum; motivates treating the detrend
         reference choice (global mean vs. per-regime EMA) as a modeling
         decision, not a cosmetic preprocessing step.
[BHG21]  Baptista, M. L., Henriques, E. M. P., and Goebel, K. (2021). A
         self-organizing map and a normalizing multi-layer perceptron
         approach to baselining in prognostics under dynamic regimes.
         Neurocomputing, 456, 268-287. Direct precedent for this class's
         design: cluster observations into operating regimes, then
         baseline/normalize per regime rather than globally, because
         degradation signals conflated with regime-induced variation are
         difficult to track otherwise.
[ROB59]  Roberts, S. W. (1959). Control chart tests based on geometric
         moving averages. Technometrics, 1(3), 239-250.
         doi:10.1080/00401706.1959.10489860. Foundational EWMA reference;
         motivates the exponential-decay per-group reference update in
         :meth:`RegimeDetrender.update` over a plain running mean.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.cluster.vq import kmeans2, vq

__all__ = ["DetrendResult", "RegimeDetrender"]


@dataclass
class DetrendResult:
    """Return value of :meth:`RegimeDetrender.update`.

    Attributes
    ----------
    detrended : np.ndarray
        ``state - x_t_ref``, same shape as ``state``. NaN slots in ``state``
        propagate as NaN here — a downstream DMDc/audit consumer drops or
        imputes them.
    is_warmup : bool
        True while regime groups have not been fit yet (the reference is the
        global mean of observed states, not a per-regime EMA).
    group : int
        Index of the regime group this observation was assigned to, or -1
        during warmup.
    """
    detrended: np.ndarray
    is_warmup: bool
    group: int


class RegimeDetrender:
    """Per-regime EMA detrending of a streaming state trajectory.

    Groups observations by nearest k-means centroid in a whitened "regime
    vector" space (e.g. an experimental protocol, a control input, an
    operating condition), fit once on the first ``warmup`` calls, then
    detrends each new state against its group's exponential moving average.
    Regime vectors may have any dimensionality — nothing here assumes a
    fixed length or any particular semantics.

    This combines two separately-established practices rather than
    implementing one paper's proposed method: (1) centering/detrending state
    data before fitting a DMD-style linear operator, shown to change what
    the fit recovers ([HHK20], [SKA24]); and (2) partitioning data by
    discrete operating regime before baselining, standard in the
    degradation/PHM literature ([BHG21]) because pooling multiple regimes
    into one global baseline conflates regime-specific behavior with the
    drift you actually want to track. The EMA reference update ([ROB59]) is
    the standard recency-weighted mechanism for a per-regime baseline that
    should adapt as more data for that regime arrives.

    Parameters
    ----------
    alpha : float
        EMA decay for the per-group reference; time constant is
        approximately ``1/alpha`` calls to that group.
    n_groups : int
        Target number of k-means regime groups; clipped to the number of
        unique regime vectors seen during warmup if fewer.
    warmup : int
        Number of calls before regime groups are fit. Before this many
        calls, ``update()`` detrends against the running global mean of all
        observed states (``is_warmup=True``, ``group=-1``).
    """

    def __init__(self, alpha: float = 0.1, n_groups: int = 8, warmup: int = 20):
        self._alpha = float(alpha)
        self._n_groups = int(n_groups)
        self._warmup = int(warmup)
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated regime/EMA state so the trajectory restarts fresh."""
        self._regimes: list[np.ndarray] = []
        self._states: list[np.ndarray] = []
        self._centroids: np.ndarray | None = None
        self._whiten_std: np.ndarray | None = None
        self._group_ema: dict[int, np.ndarray] = {}
        self._global_sum: np.ndarray | None = None
        self._global_count: int = 0
        self._n_calls: int = 0

    def _fit_groups(self) -> None:
        """One-time k-means fit on the buffered warmup regimes (whitened so
        differing units/scales across regime dimensions don't dominate) and
        seed each group's EMA with the mean warmup state of its members.
        Frozen after warmup; later regimes snap to the nearest centroid.
        Deterministic (seed).
        """
        regimes = np.asarray(self._regimes, dtype=float)
        if regimes.shape[0] < 1:
            return
        std = regimes.std(axis=0)
        std[std < 1e-12] = 1.0  # avoid divide-by-zero on constant dims
        self._whiten_std = std
        w = regimes / std
        n_unique = len(np.unique(w, axis=0))
        k = max(1, min(self._n_groups, n_unique))
        centroids, labels = kmeans2(w, k, seed=0, minit="++", missing="warn")
        self._centroids = np.asarray(centroids, dtype=float)
        states = np.array([np.nan_to_num(s) for s in self._states])
        global_mean = self._global_sum / max(self._global_count, 1)
        for g in range(k):
            mask = labels == g
            self._group_ema[g] = (
                states[mask].mean(axis=0) if mask.any() else global_mean
            )

    def _nearest_group(self, regime: np.ndarray) -> int:
        """Nearest regime-group centroid for ``regime`` (whitened space)."""
        if self._centroids is None:
            return -1
        w = (regime / self._whiten_std)[None, :]
        code, _ = vq(w, self._centroids)
        return int(code[0])

    def update(self, state: np.ndarray, regime: np.ndarray) -> DetrendResult:
        """Detrend ``state`` against its regime group's reference, and
        update internal accumulators with this observation.

        ``x_t_ref`` is the per-regime-group EMA (decay ``alpha``); during
        the first ``warmup`` calls it is the global mean of observed states.
        NaN state slots propagate as NaN in the detrended vector.
        """
        state = np.asarray(state, dtype=float)
        regime = np.asarray(regime, dtype=float)
        safe = np.nan_to_num(state)
        if self._global_sum is None or self._global_sum.shape != state.shape:
            self._global_sum = np.zeros_like(state, dtype=float)
            self._global_count = 0
        self._global_sum = self._global_sum + safe
        self._global_count += 1
        # Buffer regimes/states only until the (one-time) k-means fit;
        # afterwards they are dead weight, so stop growing them.
        if self._centroids is None:
            self._regimes.append(regime)
            self._states.append(safe)

        self._n_calls += 1
        global_mean = self._global_sum / max(self._global_count, 1)
        if self._n_calls <= self._warmup:
            if self._n_calls == self._warmup:
                self._fit_groups()
            return DetrendResult(state - global_mean, True, -1)

        if self._centroids is None:  # <warmup calls happened before groups fit
            self._fit_groups()
        group = self._nearest_group(regime)
        prev = self._group_ema.get(group, global_mean)
        a = self._alpha
        self._group_ema[group] = a * safe + (1.0 - a) * np.nan_to_num(prev)
        return DetrendResult(state - prev, False, group)
