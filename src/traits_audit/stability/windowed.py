"""Windowed / sliding DMDc stability curves and detrended-frame R² scoring.

Generic counterparts of what battery-forecast's Experiment 3/4 used locally.
Pure numpy except :func:`stability_convergence_windowed_pydmd`, which needs
the optional ``pydmd`` dependency (imported lazily).
"""
from __future__ import annotations

import numpy as np

from .dmdc import detrend_trajectory, dmdc_r2, fit_dmdc

__all__ = [
    "score_dmdc_r2",
    "stability_convergence_windowed",
    "stability_convergence_windowed_pydmd",
]


def score_dmdc_r2(aug, actions, A_r, B_r, U_r, regimes=None):
    """R² of a DMDc fit, scored in the same detrended frame fit_dmdc(...,
    detrend=True) fits in internally.

    fit_dmdc replaces its local aug_states with a detrended copy before the
    SVD/regression but never returns that detrended array (see the
    traits_audit.stability.dmdc module docstring) -- dmdc_r2 scored directly against
    the original raw aug is a coordinate-frame mismatch: dmdc_r2's ss_tot is
    mean-centered but ss_res is not, so a large near-constant offset that
    detrending removes (but that A_r/B_r were fit without) blows up ss_res
    without inflating ss_tot, producing wildly negative R² (reproduced: this
    was the root cause of a "27/27 policy x model fits UNRELIABLE, R² -22 to
    -442" result previously reported by battery-forecast's exp3).

    Reuses traits_audit.stability.dmdc.detrend_trajectory -- the exact (private)
    helper fit_dmdc calls internally -- rather than reimplementing the
    transform locally, so this can never silently drift out of sync with
    whatever fit_dmdc actually does to its input before fitting.
    """
    aug_detrended = detrend_trajectory(np.asarray(aug, dtype=float), regimes)
    return dmdc_r2(aug_detrended, actions, A_r, B_r, U_r)


def stability_convergence_windowed(aug_states, actions, min_obs: int = 5,
                                     n_components: int = 8,
                                     window: int | None = None):
    """Same as :func:`traits_audit.stability.dmdc.stability_convergence`, but each step's fit
    uses only the last ``window`` transitions instead of the full growing
    prefix from step 0.

    ``stability_convergence`` refits on ``aug_states[:t+1]`` for every
    ``t`` -- an ever-growing prefix, so each point late in the curve is an
    increasingly diluted cumulative fit rather than a read of the *current*
    dynamics regime. Windowing keeps each point local, mirroring the same
    reasoning already applied to LyapunovStabilityCheck.window (a separate,
    surrogate-Jacobian stability signal).

    Calls the same ``fit_dmdc`` that :func:`stability_convergence` does, just
    on a windowed slice instead of a growing prefix.

    ``window=None`` reproduces ``stability_convergence`` exactly (growing
    prefix, unbounded).
    """
    aug_states = np.asarray(aug_states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    T = min(len(aug_states) - 1, len(actions))
    out = []
    for t in range(min_obs, T + 1):
        start = max(0, t - window) if window is not None else 0
        try:
            A_t, _, _ = fit_dmdc(aug_states[start:t + 1], actions[start:t], n_components)
            out.append(float(np.abs(np.linalg.eigvals(A_t)).max()))
        except Exception:
            out.append(np.nan)
    return np.array(out)


def stability_convergence_windowed_pydmd(
    aug_states: np.ndarray,
    actions: np.ndarray,
    min_obs: int = 5,
    window: int | None = None,
    svd_rank: int = 4,
    rank_mode: str = "fixed",
    energy_threshold: float = 0.90,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
    """Same loop shape as stability_convergence_windowed
    (slice aug_states[start:t+1]/actions[start:t], fit, take
    |eigvals().max()|), but the per-window fit is pydmd.DMDc instead of
    traits_audit.stability.dmdc.fit_dmdc -- using an established package for
    the actual DMDc linear algebra while keeping the windowing SCHEDULE
    (just a slicing loop, not "the algorithm") under local control.
    Requires the optional ``pydmd`` dependency (``traits-audit[stability]``).

    PyDMD's DMDc wants features-first arrays: X is (n_features,
    n_snapshots), I is (n_control, n_snapshots-1) -- the opposite
    convention from traits_audit's (T, n), hence the transposes below.

    This is the "tangent line" view of the trajectory: at each time t, fit
    a local linear (DMDc) model to the trailing window, and characterize
    it by two numbers -- |lambda_max(t)| (the tangent's "slope": is the
    local linearization contracting, |.|<1, or expanding, |.|>1?) and
    r2(t) (how good that tangent actually is: does a linear model explain
    this window's one-step-ahead transitions well at all?). Tracking both
    together, rather than either alone, is what actually lets you ask "is
    this part of the trajectory well-approximated as locally linear, and
    if so, is that local linear approximation stable or not" -- a low
    r2(t) means lambda_max(t) at that point isn't trustworthy as a
    stability read (the tangent itself is a poor local fit).

    Confirmed empirically (see tests): .eigs gives the fitted dynamics
    matrix's eigenvalues directly. .operator.as_numpy_array is NOT
    generally a full-feature-space matrix, though -- it's the reduced
    (r, r) operator in whatever rank the SVD truncation to svd_rank
    actually produced (confirmed: shape (r, r), not (n, n), whenever
    svd_rank < n -- the untruncated case svd_rank == n was checked first
    and misleadingly made the two coincide). .B, in contrast, IS already
    a full-feature-space (n, m) matrix regardless of truncation. To get a
    full-feature-space dynamics matrix consistent with B's convention, the
    reduced operator must be reconstructed via the basis: A_full = basis @
    A_reduced @ basis.T -- exactly the same U_r @ A_r @ U_r.T convention
    traits_audit-based code elsewhere already uses.

    Each window is detrended (traits_audit.stability.dmdc.detrend_trajectory, the
    same causal-EMA-baseline transform traits_audit's own fit_dmdc applies
    internally by default) before fitting -- confirmed empirically this
    matters here just as much as it did for the coordinate-frame R^2 bug
    already found and fixed once elsewhere (score_dmdc_r2):
    real ECM state features carry a non-trivial near-constant offset
    relative to their own short-window variance, and scoring/fitting
    against the raw (non-detrended) window inflates residuals without
    inflating the variance they're compared against, understating both
    the fit quality (R^2) and, since the fit itself changes, the
    stability read (lambda_max) too. Without this, this function's
    lambda_max was also inconsistent with `_regime_fit_r2`'s
    `fit_dmdc(..., detrend=True)` (the default) used elsewhere in this
    same file.

    Rank selection (`rank_mode`) -- confirmed empirically (real trajectory
    data, all 3 datasets, several policies): a FIXED `svd_rank=8` against
    this 14-D augmented state forces in several near-noise-floor SVD modes
    (the real singular-value spectrum typically has only ~3-4 modes above
    the noise floor, e.g. [13.7, 4-6, 2-4, 0.7-0.9, ...] dropping sharply
    after the 4th), which `lstsq` then overfits, degrading one-step-ahead
    R^2 (sometimes catastrophically, e.g. R^2 = -1882). Lowering the fixed
    rank to 4 roughly DOUBLES-TO-TRIPLES post-warmup median R^2, uniformly
    across every dataset/policy combination tested -- hence the new
    default below. Two modes are exposed, matching `pydmd.utils.
    _compute_rank`'s own two conventions (reused directly here, not
    re-derived, so this can never silently drift out of sync with what
    PyDMD's own float-`svd_rank` energy mode would independently compute):
      - `"fixed"` (default): `svd_rank` used as-is, a plain fixed rank cap.
      - `"energy"`: per window, after detrending, resolve the smallest
        rank whose cumulative singular-value energy (of the window's
        future-snapshot block, i.e. the same matrix PyDMD's own `DMDc.fit`
        truncates internally) is >= `energy_threshold`, capped at
        `svd_rank` (read as "max rank" in this mode).
    Separately confirmed: the per-policy differences in windowed-fit R^2
    are REAL, ongoing dynamical signal, NOT a detrending-warmup artifact
    -- e.g. QBC-Agree genuinely stabilizes to R^2~=0.75-0.9 by t~=27-30 and
    stays there, while SS/Random never stabilize even out to t=50+; the
    detrender's own warmup=20 boundary shows no discontinuity in any of
    these curves, and disabling detrending entirely collapses ALL
    policies (including QBC-Agree) to uniformly catastrophic R^2 --
    confirming detrending is necessary and is NOT the source of the
    per-policy split once applied. This means callers must NOT pool or
    smooth over per-policy differences in this function's output -- they
    are the finding, not noise around it.

    Returns
    -------
    (lambda_max_curve, r2_curve, log_eig_spectra, effective_rank,
    raw_eigs) : the first two and the fourth all shape (T-min_obs+1,),
    NaN at any step where the window's fit failed (e.g. too little data).
    log_eig_spectra is a same-length list of 1-D arrays (ragged -- window
    rank can vary step-to-step under either rank_mode), each holding the
    FULL sorted log|eigs| of that window's fitted operator (not just the
    max) -- a "local linear stability spectrum" diagnostic. IMPORTANT
    CAVEAT (do not drop when reusing this array): this is NOT a Lyapunov
    spectrum and must never be presented or labeled as one -- it is the
    same class of object this module's own docstring already argues is
    insufficient to detect chaos (eigenvalues of a fitted
    linear/piecewise-linear operator can only ever show exponential
    growth/decay/rotation, never the nonlinear stretch-and-fold that
    produces genuine sensitive dependence), just reporting the full
    window's eigenvalue set instead of only the largest. See the module
    docstring's "Limitations / future work" section (Margazoglou & Magri
    2023) for why a true nonlinear Lyapunov spectrum is not computed
    here. effective_rank is the per-window RESOLVED DMDc rank actually
    used (`dmdc.basis.shape[1]` after the fit, in BOTH modes) -- lets a
    caller directly check e.g. "does policy X need fewer modes than
    policy Y." Note this can fall BELOW the requested/capped rank even
    under `rank_mode="fixed"` on short/early windows: PyDMD's own
    internal truncation is `min(requested_rank, actual_matrix_rank)`, not
    an exact request. raw_eigs is a same-length list of 1-D COMPLEX
    arrays (`dmdc.eigs` itself, unmodified -- same object log_eig_spectra
    is derived from) -- the raw complex poles, for a pole-plot diagnostic
    (unit-circle position: inside = contracting, outside = expanding,
    imaginary part = oscillation). Empty array / NaN at any step where
    the fit failed, matching the convention of the other outputs.
    """
    from pydmd import DMDc
    from pydmd.utils import _compute_rank

    if rank_mode not in ("fixed", "energy"):
        raise ValueError(f"rank_mode must be 'fixed' or 'energy', got {rank_mode!r}")
    aug_states = np.asarray(aug_states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    T = min(len(aug_states) - 1, len(actions))
    lambda_max_out, r2_out, log_eig_spectra, rank_out, raw_eigs_out = [], [], [], [], []
    for t in range(min_obs, T + 1):
        start = max(0, t - window) if window is not None else 0
        try:
            state_win = detrend_trajectory(aug_states[start:t + 1], None)
            X = state_win.T   # (n, k+1)
            U = actions[start:t].T          # (m, k)

            if rank_mode == "energy":
                # Y is the future-snapshot block -- the same matrix
                # DMDc.fit truncates internally, so resolving rank from
                # its SVD here mirrors PyDMD's own convention exactly.
                Y = state_win[1:].T
                s = np.linalg.svd(Y, compute_uv=False)
                r = _compute_rank(s, Y.shape[0], Y.shape[1], float(energy_threshold))
                # _compute_rank returns numpy.int64, not a plain int.
                # CRITICAL: pydmd.utils._compute_rank's own internal
                # dispatch (called again, inside DMDc.fit, on whatever we
                # pass as svd_rank) checks `isinstance(svd_rank, int)` --
                # a numpy.int64 fails that check and silently falls
                # through to the `else: rank = min(rows, cols)` branch
                # (NO truncation at all), rather than being treated as a
                # fixed-rank request. Confirmed empirically: passing the
                # un-cast numpy.int64 here made a resolved rank of 2
                # silently become an untruncated basis of shape (n, n)
                # instead of (n, 2), with no exception raised anywhere.
                # int(...) below is not a cosmetic cast -- it is the fix.
                r = int(max(1, min(r, svd_rank)))
            else:
                r = int(svd_rank)

            dmdc = DMDc(svd_rank=r)
            dmdc.fit(X, U)
            lambda_max_out.append(float(np.abs(dmdc.eigs).max()))
            log_eig_spectra.append(np.sort(np.log(np.abs(dmdc.eigs)))[::-1])
            rank_out.append(float(dmdc.basis.shape[1]))
            raw_eigs_out.append(np.asarray(dmdc.eigs))

            A_full = dmdc.basis @ dmdc.operator.as_numpy_array @ dmdc.basis.T
            Z_true = state_win[1:]
            Z_pred = (A_full @ state_win[:-1].T).T + (dmdc.B @ actions[start:t].T).T
            ss_res = float(np.sum((Z_true - Z_pred) ** 2))
            ss_tot = float(np.sum((Z_true - Z_true.mean(axis=0)) ** 2))
            r2_out.append(1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))
        except Exception:
            lambda_max_out.append(np.nan)
            r2_out.append(np.nan)
            log_eig_spectra.append(np.array([]))
            rank_out.append(np.nan)
            raw_eigs_out.append(np.array([], dtype=complex))
    return (np.array(lambda_max_out), np.array(r2_out), log_eig_spectra,
            np.array(rank_out), raw_eigs_out)
