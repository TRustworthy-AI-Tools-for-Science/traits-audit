"""Largest Lyapunov exponent (LLE) estimation from a recorded trajectory.

Rosenstein et al. (1993) average log-divergence curve + an explicit
linear-region slope fit, rolling-window variant, and an
Eckmann & Ruelle (1985)-style regime classification.

Eigenvalues of a fitted linear operator can never detect chaos (it is
intrinsically nonlinear: stretch-and-fold), which is why the LLE is estimated
directly from the trajectory. Note this remains an approximation to Eckmann &
Ruelle's taxonomy, which technically requires the sign pattern of the whole
Lyapunov spectrum, not just the largest exponent (see Margazoglou & Magri
2023 for a nonlinear spectrum).

References
----------
Rosenstein, M. T., Collins, J. J., and De Luca, C. J. (1993). A practical
    method for calculating largest Lyapunov exponents from small data sets.
    Physica D, 65(1-2), 117-134.
Wolf, A., Swift, J. B., Swinney, H. L., and Vastano, J. A. (1985).
    Determining Lyapunov exponents from a time series. Physica D, 16(3), 285-317.
Eckmann, J.-P. and Ruelle, D. (1985). Ergodic theory of chaos and strange
    attractors. Rev. Mod. Phys., 57(3), 617-656.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import wilcoxon

__all__ = [
    "rosenstein_divergence",
    "compute_lyapunov_exponent",
    "rolling_lyapunov_exponent",
    "classify_lle_regime",
]


def rosenstein_divergence(y: np.ndarray, maxt: int = 500, window: int = 10) -> np.ndarray:
    """Average log-divergence curve of near-neighbor trajectories
    (Rosenstein et al. 1993), euclidean metric.

    For each point, the nearest neighbor at a nonzero distance and temporal
    separation greater than ``window`` (the Theiler window) is found by a KD-tree
    search; the mean of ``log(distance)`` between the two trajectories is then
    tracked forward ``0 .. maxt-1`` steps.

    Bit-for-bit compatible with ``nolitsa.lyapunov.mle(y, maxt, window,
    metric="euclidean")`` (verified in this package's tests).

    Parameters
    ----------
    y : (T, D) ndarray
    maxt : int
        Maximum number of lags; clamped to ``T - window - 1``.
    window : int
        Theiler window. Needs ``T > 2*window + 3`` (the neighbor search looks
        at up to ``2*window + 3`` candidates per point) or ``ValueError`` is
        raised.

    Returns
    -------
    d : (maxt,) ndarray
    """
    y = np.asarray(y, dtype=np.float64)
    m = len(y)
    tree = cKDTree(y)
    maxnum = (window + 1) + 1 + (window + 1)
    if maxnum >= m:
        raise ValueError("maxnum is bigger than array length.")

    dists = np.empty(m)
    index = np.empty(m, dtype=int)
    for i, x in enumerate(y):
        for k in range(2, maxnum + 2):
            dist, idx = tree.query(x, k=k, p=2)
            valid = (np.abs(idx - i) > window) & (dist > 0)
            if np.count_nonzero(valid):
                dists[i] = dist[valid][0]
                index[i] = idx[valid][0]
                break
            if k == (maxnum + 1):
                raise Exception("Could not find any near neighbor with a "
                                "nonzero distance.  Try increasing the "
                                "value of maxnum.")

    maxt = min(m - window - 1, maxt)
    d = np.empty(maxt)
    d[0] = np.mean(np.log(dists))
    for t in range(1, maxt):
        t1 = np.arange(t, m)
        t2 = index[:-t] + t
        # Sometimes the nearest point would be farther than (m - maxt)
        # in time.  Such trajectories need to be omitted.
        valid = t2 < m
        t1, t2 = t1[valid], t2[valid]
        diff = y[t1] - y[t2]
        # Sequential accumulation over columns (not ``np.sum``'s pairwise
        # summation) so results are bit-identical to nolitsa's euclidean_dist.
        sq = np.zeros(len(diff))
        for j in range(diff.shape[1]):
            sq += diff[:, j] * diff[:, j]
        d[t] = np.mean(np.log(np.sqrt(sq)))
    return d


def compute_lyapunov_exponent(
    aug_states: np.ndarray,
    maxt: int = 30,
    window: int = 10,
    fit_start_frac: float = 0.1,
    fit_end_frac: float = 0.6,
) -> dict:
    """Largest Lyapunov exponent (LLE) point-estimate from a single
    recorded trajectory, via :func:`rosenstein_divergence` (Rosenstein et al.
    1993 average log-divergence curve), plus the explicit linear-region
    slope fit that the divergence curve itself does not provide.

    :func:`rosenstein_divergence` takes the (T, D) aug_states array directly
    and returns a 1-D array d[k] = mean log-divergence at lag k. It does not
    directly estimate the MLE; the MLE is estimated by linearly fitting the
    average divergence with time.

    The LLE point estimate here is the slope of d[k] vs k over a SIMPLE,
    EXPLICIT, config-selected index range -- fit_start_frac/fit_end_frac
    define [fit_start, fit_end) as a fraction of len(d); skips the
    earliest few lags (dominated by short-range correlation/noise, not
    genuine divergence) and stops well before the tail (avoiding the
    flattening once trajectories saturate at the attractor diameter).
    This is a simple, fully-documented choice, not a claim of an optimal
    automatic region-detector.

    Returns
    -------
    dict with keys "lle" (the fitted slope; >0 is chaos-indicative),
    "divergence_curve", "fit_slope" (== "lle"), "fit_intercept",
    "fit_range" (the actual (start, end) indices used), "r2_of_fit" (how
    linear the fitted region actually was -- a low value means the
    "linear region" assumption is shaky for this trajectory and the LLE
    point estimate should be treated with more skepticism).
    """
    aug_states = np.asarray(aug_states, dtype=np.float64)
    T = len(aug_states)
    # The neighbor search (see rosenstein_divergence) uses
    # maxnum = 2*window + 3 and raises
    # ValueError if maxnum >= len(y) -- i.e. it needs T > 2*window + 3,
    # independent of maxt. Confirmed empirically: window=10 (the default)
    # on a 15-step regime-half slice raises "maxnum is bigger than array
    # length" even though maxt was already clamped down to 3. Both window
    # and maxt must be clamped relative to T for short slices to be safe.
    window_eff = max(1, min(window, (T - 4) // 2))
    maxt_eff = max(3, min(maxt, T - window_eff - 2))
    d = rosenstein_divergence(aug_states, maxt=maxt_eff, window=window_eff)
    k = np.arange(len(d))
    i0 = max(1, int(round(fit_start_frac * len(d))))
    i1 = max(i0 + 2, int(round(fit_end_frac * len(d))))
    i1 = min(i1, len(d))
    coeffs = np.polyfit(k[i0:i1], d[i0:i1], 1)
    fitted = np.polyval(coeffs, k[i0:i1])
    ss_res = np.sum((d[i0:i1] - fitted) ** 2)
    ss_tot = np.sum((d[i0:i1] - d[i0:i1].mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {
        "lle": float(coeffs[0]),
        "divergence_curve": d,
        "fit_slope": float(coeffs[0]),
        "fit_intercept": float(coeffs[1]),
        "fit_range": (i0, i1),
        "r2_of_fit": r2,
    }


def rolling_lyapunov_exponent(
    aug_states: np.ndarray,
    window: int = 40,
    stride: int = 10,
    maxt: int = 20,
    lle_window: int = 8,
    fit_start_frac: float = 0.1,
    fit_end_frac: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling Lyapunov-exponent estimate over trailing windows of the
    trajectory, at a coarser stride than the per-timestep DMDc curve --
    each estimate is itself a KDTree neighbor search over `window` points
    (compute_lyapunov_exponent), so sliding it one step at a time the way
    the DMDc curve does would be far more expensive for no real benefit
    (the underlying question -- does the trajectory become more or less
    prone to chaotic/sensitive-dependence behavior over its lifetime -- is
    a slow, large-scale question, not one that needs single-timestep
    resolution).

    This directly probes whether specific time windows are more or less
    susceptible to chaotic/unstable behavior -- in particular, the
    hypothesis that once the AL policy has converged (small updates, a
    near-settled posterior), the system behaves as if it has settled onto
    an attractor: nearby trajectories should stop diverging (LLE
    decreasing, ideally going negative -- contraction) rather than
    continuing to spread apart the way an unconverged, actively-exploring
    early phase might.

    Returns
    -------
    (t_indices, lle_values) : t_indices are the window END index into
    aug_states for each estimate (the same t-index convention as
    stability_convergence_windowed_pydmd's curves, so the two can be
    plotted against/compared with each other directly); lle_values[i] is
    NaN wherever that window's fit failed.
    """
    aug_states = np.asarray(aug_states, dtype=np.float64)
    T = len(aug_states)
    t_indices, lle_values = [], []
    for t in range(window, T + 1, stride):
        start = t - window
        try:
            result = compute_lyapunov_exponent(
                aug_states[start:t], maxt=maxt, window=lle_window,
                fit_start_frac=fit_start_frac, fit_end_frac=fit_end_frac,
            )
            lle_values.append(result["lle"])
        except Exception:
            lle_values.append(np.nan)
        t_indices.append(t)
    return np.array(t_indices), np.array(lle_values)


def classify_lle_regime(values: np.ndarray, alpha: float = 0.05) -> tuple[str, float, float]:
    """Classify a sample of Lyapunov-exponent-like values into one of
    Eckmann & Ruelle (1985)'s 3-way taxonomy (see module docstring) via a
    one-sample Wilcoxon signed-rank test against 0, with
    an ADAPTIVE test direction (not a hardcoded alternative="greater"): the alternative is "greater" if the sample
    median is > 0, else "less" -- this is a genuine correctness fix, not
    just a generalization, since always testing "is it greater than 0"
    would give the wrong one-sided p-value whenever the observed median is
    negative. For a positive pooled median LLE this is the plain
    one-sided "greater" test.

    Labels
    ------
    "COLLAPSING"       : median < 0 and credibly so (p < alpha) -- the
                          strange-attractor analogue of a stable fixed
                          point/limit cycle, contraction dominates.
    "CHAOTIC"           : median > 0 and credibly so (p < alpha) --
                          sensitive dependence, divergence dominates.
    "PERIODIC-LIKE"     : not credibly different from 0 (p >= alpha) --
                          neither confirmed contraction nor divergence.
    "INSUFFICIENT_DATA" : fewer than 3 finite values -- Wilcoxon is not
                          meaningful below this.

    Note: this remains an approximation to Eckmann & Ruelle's taxonomy,
    which technically requires the sign PATTERN OF THE WHOLE Lyapunov
    spectrum, not just the largest exponent -- see the module docstring's note
    on the spectrum (Margazoglou & Magri 2023).

    Returns
    -------
    (label, median, p_value)
    """
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if len(finite) < 3:
        return "INSUFFICIENT_DATA", float("nan"), float("nan")
    median = float(np.median(finite))
    alternative = "greater" if median > 0 else "less"
    try:
        _, p_value = wilcoxon(finite, alternative=alternative)
        p_value = float(p_value)
    except ValueError:
        # All-zero or otherwise degenerate sample -- wilcoxon raises.
        return "PERIODIC-LIKE", median, float("nan")
    if p_value < alpha:
        label = "CHAOTIC" if median > 0 else "COLLAPSING"
    else:
        label = "PERIODIC-LIKE"
    return label, median, p_value
