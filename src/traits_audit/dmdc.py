"""Dynamic Mode Decomposition with Control (DMDc) and stability diagnostics.

Pure numpy/scipy.  No dependency on any GPR surrogate, battery oracle, or
plotting library — callers assemble the augmented state trajectory and pass
plain arrays in.

References
----------
[VCH06]   Vachtsevanos, G., Lewis, F., Roemer, M., Hess, A., and Wu, B.
          (2006). Intelligent Fault Diagnosis and Prognosis for Engineering
          Systems. Wiley. ISBN 978-0-471-72999-0. General PHM/RUL framing;
          motivates treating ``1 - |λ_max|`` as a stability-margin health
          proxy and bounding perturbation transients (see
          :func:`perturbation_response`).
[MOO18]   Moosavi, A. and Sandu, A. (2018). A state-space approach to
          analyze structural uncertainty in physical models. Metrologia,
          55(1), S1-S12. doi:10.1088/1681-7575/aa8f53. LTI SSM
          identification from discrepancy trajectories; motivates
          :func:`fit_dmdc` / :func:`fit_dmdc_pairs` and the bootstrap
          confidence intervals in :func:`bootstrap_eig_ci` (A_r is uncertain
          because it is fit from finite data).
[MOR81]   Moore, B. C. (1981). Principal component analysis in linear
          systems: Controllability, observability, and model reduction.
          IEEE Trans. Automatic Control, 26(1), 17-32.
          doi:10.1109/TAC.1981.1102568. Gramian-based observability/
          controllability analysis; motivates :func:`compute_gramians`
          (the discrete-time Stein equations solved there are the
          conventional discrete analogue of Moore's continuous-time
          equations).
[PRO16]   Proctor, J. L., Brunton, S. L., and Kutz, J. N. (2016). Dynamic
          mode decomposition with control. SIAM J. Applied Dynamical
          Systems, 15(1), 142-161. doi:10.1137/15M1013857. Primary
          reference for DMDc; covers rank selection (used throughout this
          module, e.g. in :func:`fit_dmdc` and :func:`rank_sensitivity`) and
          the relationship between DMDc modes and full-state reconstruction.
[SCH10]   Schmid, P. J. (2010). Dynamic mode decomposition of numerical and
          experimental data. J. Fluid Mechanics, 656, 5-28.
          doi:10.1017/S0022112010001217. Foundational DMD reference for
          mode interpretation (oscillation frequency, growth/decay rate);
          motivates :func:`modal_decomposition`.
[TRE05]   Trefethen, L. N. and Embree, M. (2005). Spectra and Pseudospectra:
          The Behavior of Nonnormal Matrices and Operators. Princeton
          University Press. ISBN 978-0-691-11946-5. Pseudospectra theory;
          motivates :func:`pseudospectrum`.

Detrending
----------
``fit_dmdc``/``fit_dmdc_pairs`` center the augmented-state trajectory before
fitting, by default — see :mod:`traits_audit.detrend` (which cites [HHK20]
and [SKA24] motivating pre-DMD centering) for the detrending model itself.
Pass ``detrend=False`` to fit on raw, uncentered data.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from .detrend import RegimeDetrender

__all__ = [
    "fit_dmdc",
    "fit_dmdc_pairs",
    "dmdc_r2",
    "stability_convergence",
    "perturbation_response",
    "pseudospectrum",
    "equilibrium_state",
    "compute_gramians",
    "bootstrap_eig_ci",
    "modal_decomposition",
    "rank_sensitivity",
]


def _detrend_trajectory(states: np.ndarray, regimes: np.ndarray | None = None) -> np.ndarray:
    """Chronologically detrend a contiguous state trajectory via a fresh
    :class:`~traits_audit.detrend.RegimeDetrender`.

    ``regimes`` defaults to a trivial constant vector when omitted, which
    collapses the detrender's regime clustering to a single group — i.e. a
    causal (expanding-window, then EMA) global baseline.
    """
    states = np.asarray(states, dtype=np.float64)
    if regimes is None:
        regimes = np.zeros((len(states), 1))
    else:
        regimes = np.asarray(regimes, dtype=np.float64)
    detrender = RegimeDetrender()
    out = np.empty_like(states)
    for i in range(len(states)):
        out[i] = detrender.update(states[i], regimes[i]).detrended
    return out


def _detrend_pairs_pooled(aug_t: np.ndarray, aug_t1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Order-invariant centering for explicit (possibly resampled, out-of-order)
    transition pairs: subtract the pooled mean of ``aug_t`` from both arrays.

    Unlike :func:`_detrend_trajectory`, this does not assume temporal order —
    appropriate for :func:`fit_dmdc_pairs`, whose own contract makes no such
    assumption (bootstrap-resampled pairs may repeat rows and appear in any
    order).
    """
    aug_t = np.asarray(aug_t, dtype=np.float64)
    aug_t1 = np.asarray(aug_t1, dtype=np.float64)
    mean_t = aug_t.mean(axis=0)
    return aug_t - mean_t, aug_t1 - mean_t


def fit_dmdc(aug_states: np.ndarray, actions: np.ndarray, n_components: int = 8,
             regimes: np.ndarray | None = None, detrend: bool = True):
    """Fit a reduced-order DMDc model ``z_{t+1} = A_r z_t + B_r a_t``.

    Parameters
    ----------
    aug_states : np.ndarray, shape (T, n)
        Augmented state trajectory ``s̃_0 ... s̃_{T-1}``.
    actions : np.ndarray, shape (T-1, m) or (T, m)
        Action applied at each step; only the first ``T-1`` rows are used.
    n_components : int
        Target SVD truncation rank ``r``; clipped to ``min(n_components, n, T-1)``.
    regimes : np.ndarray, shape (T, k), optional
        Per-step regime vector forwarded to :func:`_detrend_trajectory`
        (ignored if ``detrend=False``). Defaults to a trivial constant
        vector — a causal global-EMA baseline — when omitted.
    detrend : bool
        Center ``aug_states`` via :func:`_detrend_trajectory` before fitting
        (default ``True``). Pass ``False`` to fit on raw data.

    Returns
    -------
    (A_r, B_r, U_r) : tuple of np.ndarray
        ``A_r`` is ``(r, r)``, ``B_r`` is ``(r, m)``, ``U_r`` is ``(n, r)``.

    References
    ----------
    [MOO18] : SSM identification approach.
    [PRO16] : rank-r truncation via SVD.
    """
    aug_states = np.asarray(aug_states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if detrend:
        aug_states = _detrend_trajectory(aug_states, regimes)
    T_fit = len(aug_states) - 1
    n = aug_states.shape[1]
    m = actions.shape[1]
    # lstsq system is (T_fit) × (r + m).  It must be overdetermined (T_fit > r + m),
    # otherwise lstsq returns a minimum-norm interpolant with R² = 1 and
    # eigenvalues that are numerical artifacts, not physical stability.
    r_max_od = max(1, T_fit - m - 1)   # largest r that keeps the system overdetermined
    r = min(n_components, n, T_fit, r_max_od)

    U, _, _ = np.linalg.svd(aug_states.T, full_matrices=False)
    U_r = U[:, :r]                       # (n, r)

    Z = aug_states @ U_r                 # (T, r)
    X_fit = np.hstack([Z[:T_fit], actions[:T_fit]])   # (T_fit, r+m)
    Y_fit = Z[1:T_fit + 1]                              # (T_fit, r)
    W, _, _, _ = np.linalg.lstsq(X_fit, Y_fit, rcond=None)

    A_r = W[:r].T                        # (r, r)
    B_r = W[r:].T                        # (r, m)
    return A_r, B_r, U_r


def fit_dmdc_pairs(aug_t: np.ndarray, aug_t1: np.ndarray, actions: np.ndarray,
                    n_components: int = 8, detrend: bool = True):
    """Fit DMDc on explicit (possibly non-contiguous) transition pairs.

    Unlike :func:`fit_dmdc`, this does not assume ``aug_t1[i] == aug_t[i+1]``,
    so it can be used on bootstrap-resampled transitions.

    Parameters
    ----------
    aug_t, aug_t1 : np.ndarray, shape (N, n)
        Current and next augmented states for each of ``N`` transitions.
    actions : np.ndarray, shape (N, m)
        Action applied at each transition.
    n_components : int
        Target SVD truncation rank ``r``.
    detrend : bool
        Center ``aug_t``/``aug_t1`` via :func:`_detrend_pairs_pooled` before
        fitting (default ``True``). This is an order-invariant pooled-mean
        subtraction, not :func:`fit_dmdc`'s causal streaming detrend — pairs
        make no temporal-order guarantee (e.g. bootstrap resamples), so a
        causal detrender would be inappropriate here. Pass ``False`` to fit
        on raw data, e.g. when the source trajectory was already detrended
        upstream (see :func:`bootstrap_eig_ci`).

    Returns
    -------
    (A_r, B_r, U_r)

    References
    ----------
    [MOO18] : bootstrap resampling of transition pairs for structural
              uncertainty in the fitted SSM.
    """
    aug_t = np.asarray(aug_t, dtype=np.float64)
    aug_t1 = np.asarray(aug_t1, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if detrend:
        aug_t, aug_t1 = _detrend_pairs_pooled(aug_t, aug_t1)
    n = aug_t.shape[1]
    m = actions.shape[1]
    N = len(aug_t)
    r_max_od = max(1, N - m - 1)
    r = min(n_components, n, N, r_max_od)

    U, _, _ = np.linalg.svd(aug_t.T, full_matrices=False)
    U_r = U[:, :r]

    Z_t = aug_t @ U_r
    Z_t1 = aug_t1 @ U_r
    X_fit = np.hstack([Z_t, actions])
    W, _, _, _ = np.linalg.lstsq(X_fit, Z_t1, rcond=None)

    A_r = W[:r].T
    B_r = W[r:].T
    return A_r, B_r, U_r


def dmdc_r2(aug_states: np.ndarray, actions: np.ndarray, A_r: np.ndarray,
            B_r: np.ndarray, U_r: np.ndarray) -> float:
    """Coefficient of determination of the fitted DMDc model in the r-dim subspace.

    Returns ``nan`` if the projected trajectory has zero variance.
    """
    aug_states = np.asarray(aug_states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    Z = aug_states @ U_r
    T = len(Z) - 1
    Z_pred = Z[:T] @ A_r.T + actions[:T] @ B_r.T
    ss_res = np.sum((Z[1:] - Z_pred) ** 2)
    ss_tot = np.sum((Z[1:] - Z[1:].mean(axis=0)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def stability_convergence(aug_states: np.ndarray, actions: np.ndarray,
                           min_obs: int = 5, n_components: int = 8) -> np.ndarray:
    """``|λ_max(A_r(t))|`` for growing prefixes of the trajectory.

    Fits DMDc using only data up to step ``t`` for ``t = min_obs ... T``,
    tracking how the identified dynamics evolve as evidence accumulates.

    References
    ----------
    [MOO18] : repeated SSM identification as data accumulates.
    """
    aug_states = np.asarray(aug_states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    T = min(len(aug_states) - 1, len(actions))
    out = []
    for t in range(min_obs, T + 1):
        try:
            A_t, _, _ = fit_dmdc(aug_states[:t + 1], actions[:t], n_components)
            out.append(float(np.abs(np.linalg.eigvals(A_t)).max()))
        except Exception:
            out.append(np.nan)
    return np.array(out)


def perturbation_response(A_r: np.ndarray, n_steps: int = 20, n_samples: int = 200,
                           seed: int = 42) -> np.ndarray:
    """Free response ``z_k = A_r^k δz`` to random unit-norm perturbations.

    Even when ``|λ_max| < 1``, a non-normal ``A_r`` can show transient
    amplification before eventual decay.

    Returns
    -------
    np.ndarray, shape (n_samples, n_steps + 1)
        ``||z_k||`` for each sample and step.

    References
    ----------
    [VCH06] : PHM/resilience framing — peak amplification as a fragility indicator.
    """
    rng = np.random.default_rng(seed)
    r = A_r.shape[0]
    curves = np.zeros((n_samples, n_steps + 1))
    for i in range(n_samples):
        dz = rng.normal(size=r)
        dz /= np.linalg.norm(dz)
        z = dz
        curves[i, 0] = np.linalg.norm(z)
        for k in range(1, n_steps + 1):
            z = A_r @ z
            curves[i, k] = np.linalg.norm(z)
    return curves


def pseudospectrum(A_r: np.ndarray, grid_n: int = 60, eps_levels=None):
    """Smallest singular value of ``(zI - A_r)`` on a grid of the complex plane.

    Parameters
    ----------
    A_r : np.ndarray, shape (r, r)
    grid_n : int
        Grid resolution per axis.
    eps_levels : sequence of float, optional
        Levels at which the ε-pseudospectrum contour would be drawn by the
        caller (not used internally; included for API documentation).

    Returns
    -------
    (RE, IM, sigma_min) : tuple of np.ndarray, each shape (grid_n, grid_n)

    References
    ----------
    [TRE05] : pseudospectrum theory and sensitivity analysis of the dynamics operator.
    """
    if eps_levels is None:
        eps_levels = [0.1, 0.3, 0.5]
    r = A_r.shape[0]
    re = np.linspace(-1.5, 1.5, grid_n)
    im = np.linspace(-1.5, 1.5, grid_n)
    RE, IM = np.meshgrid(re, im)
    sigma_min = np.zeros_like(RE)
    I = np.eye(r)
    for i in range(grid_n):
        for j in range(grid_n):
            z = RE[i, j] + 1j * IM[i, j]
            sigma_min[i, j] = np.linalg.svd(z * I - A_r, compute_uv=False)[-1]
    return RE, IM, sigma_min


def equilibrium_state(A_r: np.ndarray, B_r: np.ndarray, U_r: np.ndarray,
                       a_star: np.ndarray) -> np.ndarray:
    """Fixed point ``s̃* = U_r (I - A_r)^{-1} B_r a*`` under sustained action ``a*``.

    Requires ``A_r`` to be stable (``|λ_max| < 1``); raises if ``I - A_r`` is
    singular.

    References
    ----------
    [VCH06] : equilibrium health state under a sustained policy action (PHM framing).
    """
    r = A_r.shape[0]
    z_star = np.linalg.solve(np.eye(r) - A_r, B_r @ a_star)
    return U_r @ z_star


def compute_gramians(A_r: np.ndarray, B_r: np.ndarray):
    """Discrete-time controllability and observability Gramians.

    ``W_c`` solves ``A_r W_c A_r^T - W_c = -B_r B_r^T`` (controllability);
    ``W_o`` solves ``A_r^T W_o A_r - W_o = -I`` (full-state observability,
    ``C = I``). High ``cond(W_c)`` → under-actuated modes; high
    ``cond(W_o)`` → unobservable modes.

    If ``A_r`` has spectral radius ≥ 1 the Lyapunov equation is ill-posed.
    In that case ``A_r`` is rescaled to spectral radius 0.99 before solving;
    the resulting Gramians characterise the stabilised system and still give
    meaningful relative controllability/observability comparisons across
    policies.

    Returns
    -------
    (W_c, W_o) : tuple of np.ndarray, each shape (r, r)

    References
    ----------
    [MOR81] : Gramian-based observability / controllability analysis; the discrete
              Stein equations solved here are the conventional discrete analogue of
              Moore's continuous-time equations.
    """
    r = A_r.shape[0]
    rho = float(np.abs(np.linalg.eigvals(A_r)).max())
    if rho >= 1.0:
        A_r = A_r * (0.99 / rho)
    W_c = solve_discrete_lyapunov(A_r, B_r @ B_r.T)
    W_o = solve_discrete_lyapunov(A_r.T, np.eye(r))
    return W_c, W_o


def bootstrap_eig_ci(aug_states: np.ndarray, actions: np.ndarray, n_boot: int = 200,
                      n_components: int = 8, seed: int = 42, regimes: np.ndarray | None = None,
                      detrend: bool = True):
    """Bootstrap 95% CI on eigenvalue magnitudes via paired-transition resampling.

    Resamples ``(s̃_t, s̃_{t+1}, a_t)`` transitions with replacement and refits
    DMDc on each resample, building an empirical distribution of
    ``|λ_i(A_r)|`` for each of the ``r`` modes.

    When ``detrend``, the full trajectory is centered ONCE via
    :func:`_detrend_trajectory` before resampling begins (rather than
    re-running a causal streaming detrend on each shuffled/repeated-row
    resample, which would make the detrend fit an artifact of resample order
    rather than the trajectory).

    Returns
    -------
    (ci_lo, ci_hi) : tuple of np.ndarray, each shape (r,)
        2.5th and 97.5th percentile eigenvalue magnitudes across bootstraps.

    References
    ----------
    [MOO18] : structural uncertainty in SSMs motivates bootstrap CI on A_r.
    """
    aug_states = np.asarray(aug_states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if detrend:
        aug_states = _detrend_trajectory(aug_states, regimes)
    rng = np.random.default_rng(seed)
    T_pairs = len(aug_states) - 1
    mags = []
    for _ in range(n_boot):
        idx = rng.integers(0, T_pairs, size=T_pairs)
        aug_t = aug_states[idx]
        aug_t1 = aug_states[idx + 1]
        acts = actions[idx]
        try:
            A_b, _, _ = fit_dmdc_pairs(aug_t, aug_t1, acts, n_components=n_components, detrend=False)
            mags.append(np.abs(np.linalg.eigvals(A_b)))
        except Exception:
            continue
    if not mags:
        r = min(n_components, aug_states.shape[1], T_pairs)
        nan = np.full(r, np.nan)
        return nan, nan
    max_r = max(len(m) for m in mags)
    mags = [np.pad(m, (0, max_r - len(m)), constant_values=np.nan) for m in mags]
    mags = np.array(mags)
    lo, hi = np.nanpercentile(mags, [2.5, 97.5], axis=0)
    return lo, hi


def modal_decomposition(A_r: np.ndarray, U_r: np.ndarray, feature_names=None):
    """Eigendecompose ``A_r`` and project modes back to the original feature space.

    Returns a list of dicts, one per mode, sorted by descending ``|λ|``:
    ``{"eigenvalue", "magnitude", "frequency_rad", "damping", "mode_orig",
    "dominant_feature"}``. ``frequency_rad`` is ``arctan2(Im(λ), Re(λ))``,
    the per-step phase rotation of a damped spiral mode; ``damping`` is
    ``log(|λ|)`` (negative = decaying).

    References
    ----------
    [SCH10] : DMD mode interpretation — projecting reduced-order modes back through
              U_r, oscillation frequency and growth/decay rate.
    """
    eigvals, eigvecs = np.linalg.eig(A_r)
    order = np.argsort(-np.abs(eigvals))
    modes = []
    for i in order:
        lam = eigvals[i]
        v = eigvecs[:, i]
        mode_orig = (U_r @ v).real
        dom_idx = int(np.argmax(np.abs(mode_orig)))
        dominant_feature = (
            feature_names[dom_idx] if feature_names is not None and dom_idx < len(feature_names)
            else dom_idx
        )
        modes.append({
            "eigenvalue": lam,
            "magnitude": float(np.abs(lam)),
            "frequency_rad": float(np.arctan2(lam.imag, lam.real)),
            "damping": float(np.log(np.abs(lam))) if np.abs(lam) > 0 else float("-inf"),
            "mode_orig": mode_orig,
            "dominant_feature": dominant_feature,
        })
    return modes


def rank_sensitivity(aug_states: np.ndarray, actions: np.ndarray, r_values) -> dict:
    """Fit DMDc at each rank in ``r_values``; return ``{r: eigenvalues}``.

    Used to confirm that stability conclusions (``|λ_max|``) are not
    artifacts of the SVD truncation rank choice.

    References
    ----------
    [PRO16] : rank-selection sensitivity analysis for DMDc.
    """
    out = {}
    for r in r_values:
        try:
            A_r, _, _ = fit_dmdc(aug_states, actions, n_components=r)
            out[r] = np.linalg.eigvals(A_r)
        except Exception:
            out[r] = np.array([])
    return out
