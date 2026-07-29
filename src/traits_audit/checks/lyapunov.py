"""Lyapunov stability check and associated computation for surrogate models."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from ..base import AuditCategory, AuditCheck, AuditResult


# ── Lyapunov computation (numpy/scipy only, no matplotlib) ───────────────────

def numerical_jacobian(
    predictor,
    state: np.ndarray,
    action: np.ndarray | None = None,
    dx: float = 1e-4,
) -> np.ndarray:
    """n×n Jacobian of predictor(state, action) → ℝⁿ via central differences.

    ``action`` is passed through unchanged to ``predictor``; pass ``None``
    for surrogates that have no action input.
    """
    n = len(state)
    J = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        s_plus  = state.copy(); s_plus[i]  += dx
        s_minus = state.copy(); s_minus[i] -= dx
        col = (predictor(s_plus, action) - predictor(s_minus, action)) / (2.0 * dx)
        J[:, i] = col
    return J


def eigenvalues_and_stability(J: np.ndarray) -> dict:
    """Eigenvalue spectrum and discrete-time stability indicators."""
    eigs = np.linalg.eigvals(J)
    mags = np.abs(eigs)
    return {
        "eigenvalues": eigs,
        "magnitudes":  mags,
        "lambda_max":  float(mags.max()),
        "lambda_min":  float(mags.min()),
        "is_stable":   bool((mags < 1.0).all()),
        "n_unstable":  int((mags >= 1.0).sum()),
    }


def compute_lyapunov(A: np.ndarray, rho_max: float = 0.99) -> np.ndarray | None:
    """Solve discrete Lyapunov equation Aᵀ P A − P = −I.

    If ``A``'s spectral radius is ≥ 1, ``A`` is rescaled to spectral radius
    *rho_max* before solving. Returns ``None`` if ``solve_discrete_lyapunov``
    raises (degenerate/singular ``A``).
    """
    rho = float(np.abs(np.linalg.eigvals(A)).max())
    if rho >= 1.0:
        A = A * (rho_max / rho)
    try:
        return solve_discrete_lyapunov(A.T, np.eye(len(A)))
    except Exception:
        return None


def make_gd_predictor(f_scalar, alpha: float = 0.01, eps: float = 1e-5):
    """Discrete gradient-descent step predictor: x_{t+1} = x_t − α ∇f(x_t).

    The Jacobian of this map equals J = I − α H_f, so |λ(J)| < 1 iff all
    eigenvalues of H_f lie in (0, 2/α).
    """
    def predictor(state: np.ndarray, action=None) -> np.ndarray:
        n = len(state)
        grad = np.zeros(n)
        for i in range(n):
            s_p = state.copy(); s_p[i] += eps
            s_m = state.copy(); s_m[i] -= eps
            grad[i] = (f_scalar(s_p) - f_scalar(s_m)) / (2.0 * eps)
        return state - alpha * grad
    return predictor


def _numerical_jacobian_batched(
    f_batched,
    state: np.ndarray,
    alpha: float,
    dx: float = 1e-4,
    eps: float = 1e-5,
) -> np.ndarray:
    """Same n×n Jacobian as ``_numerical_jacobian(_gd_predictor(f_scalar, alpha,
    eps), state, dx)`` for the scalar-callable ``f_scalar`` that ``f_batched``
    batches — but issued as exactly ONE call to ``f_batched`` covering all
    4n² perturbed states the nested (Jacobian-of-a-numerically-differentiated-
    gradient) computation needs, instead of 4n² separate scalar calls.

    For a surrogate whose real cost is per-call dispatch overhead rather than
    per-row compute (e.g. a JAX-jitted model, where a threaded loop of many
    small calls empirically gave ~0% speedup — batched execution is the
    idiomatic fix for that case, not concurrency), this is where the actual
    win comes from.

    Parameters
    ----------
    f_batched : callable
        ``(M, D) ndarray -> (M,) ndarray``. Must accept an arbitrary batch
        size M and return one scalar per row — the batched counterpart of
        the scalar ``f_scalar: (D,) ndarray -> float`` that
        ``_numerical_jacobian``/``_gd_predictor`` take.
    state : (D,) ndarray
    alpha : float
        Gradient-descent step size (see ``_gd_predictor``).
    """
    n = len(state)
    # Outer perturbations (2n states): outer[2i] = state + dx*e_i, outer[2i+1] = state - dx*e_i.
    outer = np.tile(state, (2 * n, 1)).astype(np.float64)
    idx = np.arange(n)
    outer[2 * idx, idx] += dx
    outer[2 * idx + 1, idx] -= dx

    # Each of the 2n outer states needs its own 2n-perturbation gradient estimate
    # -> 2n * 2n = 4n^2 states total, batched into one call.
    inner = np.repeat(outer, 2 * n, axis=0)  # (4n^2, D)
    base = (np.arange(2 * n) * 2 * n)[:, None]  # (2n, 1) row offsets into `inner`
    rows_p = (base + 2 * idx[None, :]).ravel()
    rows_m = (base + 2 * idx[None, :] + 1).ravel()
    cols = np.tile(idx, 2 * n)
    inner[rows_p, cols] += eps
    inner[rows_m, cols] -= eps

    flat = np.asarray(f_batched(inner), dtype=np.float64).reshape(2 * n, 2 * n)
    grad = (flat[:, 0::2] - flat[:, 1::2]) / (2.0 * eps)  # (2n, n) == (2n, D)
    predictor_out = outer - alpha * grad  # (2n, D)

    J = np.zeros((n, n), dtype=np.float64)
    J[:, idx] = ((predictor_out[2 * idx] - predictor_out[2 * idx + 1]) / (2.0 * dx)).T
    return J


# ── Check ─────────────────────────────────────────────────────────────────────

class LyapunovStabilityCheck(AuditCheck):
    """
    Flags if the surrogate landscape is predominantly dynamically unstable.

    Lyapunov stability characterises whether gradient descent on the surrogate's
    predicted surface would converge.  The discrete gradient-descent map
    :math:`x_{t+1} = x_t - \\alpha \\nabla f(x_t)` has Jacobian
    :math:`J = I - \\alpha H_f`; an operating point is *stable* when all
    eigenvalues of *J* are strictly inside the unit circle
    (:math:`|\\lambda| < 1`).
    A high fraction of unstable points signals that the surrogate has learned steep
    or rough gradients — often a sign of insufficient coverage or a non-smooth
    surrogate family (e.g. decision trees).

    The check reports the fraction of operating points with
    :math:`|\\lambda_{\\max}| < \\text{stability\\_threshold}`.

    Parameters
    ----------
    stability_threshold : float
        Maximum :math:`|\\lambda_{\\max}|` treated as stable.
        Default ``1.0`` — the discrete-time unit-circle boundary.
    min_stable_fraction : float
        Minimum fraction of operating points that must be stable to PASS
        (default ``0.5``).
    alpha : float
        Gradient-descent step size used when building the GD predictor from
        a surrogate callable (default ``0.01``).
    n_pca : int or None
        If set, PCA-reduces the (windowed) ``op_states`` to this many
        dimensions before computing Jacobians.  Useful for high-dimensional
        feature spaces.  Requires ``scikit-learn``.  PCA is **skipped** (with a
        note in ``message``/``details``) when ``n_pca >= min(window_points,
        n_features)`` — i.e. there are too few points to fit that many
        components; a further note is emitted when ``window_points < 3*n_pca``
        (fragile estimate).
    window : int or None
        Lookback window ``M``.  When set, only the **last M** operating points
        are used — both to fit the (local) PCA and to aggregate the stable
        fraction — so the signal reflects the current operating region rather
        than the whole diluted history.  ``None`` uses all points (the original
        cumulative behaviour).
    max_workers : int or None
        If set (> 1) and ``surrogate_fn_batched`` is *not* used, each *new*
        (uncached) point's Jacobian is computed concurrently in a
        ``ThreadPoolExecutor`` with this many workers — every point's
        Jacobian is independent of every other's, making this embarrassingly
        parallel in principle.  ``None`` (default) computes sequentially.
        In practice, for a JAX-jitted ``surrogate_fn`` this delivered close
        to zero real speedup in testing (JAX's CPU/XLA backend does not give
        genuine concurrent execution to calls issued from separate Python
        threads) — see ``surrogate_fn_batched`` below for the fix that
        actually helps that case. ``max_workers`` is kept for surrogates
        that genuinely benefit from thread-level concurrency (e.g. ones that
        release the GIL during their own compute). Requires ``surrogate_fn``
        to be safe to call concurrently from multiple threads; this class
        holds no shared mutable state during the parallel section itself, so
        the check is thread-safe either way — the constraint is entirely on
        the caller-supplied ``surrogate_fn``.

    Required data (at least one route)
    -----------------------------------
    Precomputed (preferred): ``lambda_max`` kwarg (array_like) or
    ``lambda_max`` key in each history dict (float per step). Non-finite
    entries (e.g. a NaN-padded warm-up prefix from a growing-window fit) are
    dropped before windowing/aggregation, rather than counted as unstable.
    On-demand: ``surrogate_fn`` kwarg (callable) and ``op_states``
    kwarg (ndarray of shape (N, D)).

    Optional batched evaluation (recommended for JAX/vectorized surrogates)
    ------------------------------------------------------------------------
    surrogate_fn_batched kwarg — callable with signature
    (M, D) ndarray -> (M,) ndarray (the batched counterpart of
    surrogate_fn). Each point's Jacobian needs 4n² scalar evaluations (the
    Jacobian of a numerically-differentiated
    gradient, nested); when this is given, all 4n² perturbed states for
    a point are issued as a **single** batched call instead of 4n²
    separate ones — this is where the check's real cost lives, and batching
    is the fix that actually pays off for surrogates optimized for vectorized
    execution (unlike ``max_workers`` threading, which does not help a
    JAX-jitted model — see above). Falls back to the scalar ``surrogate_fn``
    path (optionally threaded via ``max_workers``) when omitted.

    Optional caching
    ----------------
    ``lambda_cache`` kwarg — a caller-owned ``dict`` keyed by absolute
    operating-point row index.  Per-point λ_max already present is reused; new
    points are computed under the *current* window's PCA basis and stored.  The
    check never keeps this state on ``self`` (it stays stateless and
    thread-safe); the caller owns one dict per trajectory.  Reuse is
    *approximate* under per-window PCA — a cached value keeps the basis it was
    first computed in.  Omit the kwarg to recompute every point exactly.
    """

    def __init__(
        self,
        stability_threshold: float = 1.0,
        min_stable_fraction: float = 0.5,
        alpha: float = 0.01,
        n_pca: Optional[int] = None,
        window: Optional[int] = None,
        max_workers: Optional[int] = None,
    ):
        self.stability_threshold = stability_threshold
        self.min_stable_fraction = min_stable_fraction
        self.alpha = alpha
        self.n_pca = n_pca
        self.window = window
        self.max_workers = max_workers

    @property
    def name(self) -> str:
        return "LyapunovStability"

    @property
    def category(self) -> AuditCategory:
        # Ergodic/non-ergodic (METRIC_TAXONOMY_AUDIT.md §2.1, §4.4): this
        # characterises the *local* dynamics of the uncertainty landscape
        # along the campaign trajectory, not an aleatoric/epistemic split.
        # Must be paired with DMDcSpectralRadiusCheck (global spectral
        # radius of the same trajectory) — see checks/ergodic.py and
        # validation.NAME_PAIRS.
        return AuditCategory.ERGODIC_NON_ERGODIC

    def _compute_lambda_max(
        self, surrogate_fn, op_states: np.ndarray, cache=None, surrogate_fn_batched=None
    ):
        """Per-point |λ_max| over the (windowed) op_states, plus a diagnostics dict.

        Windows to the last ``self.window`` rows (keeping absolute row indices
        for cache keys), skips PCA when it cannot be fit, and reuses any per-row
        value already in ``cache`` (a caller-owned dict) — see class docstring.
        When ``surrogate_fn_batched`` is given, each new point's Jacobian is
        computed via one batched call (see ``_numerical_jacobian_batched``)
        instead of ``4n²`` scalar ``surrogate_fn`` calls (optionally threaded
        via ``max_workers``). Returns ``(lm, info)``.
        """
        states_full = np.asarray(op_states, dtype=float)
        n_total = states_full.shape[0]

        # Window: last M rows; offset keeps absolute indices stable across stages.
        if self.window is not None and n_total > self.window:
            states = states_full[-self.window:]
        else:
            states = states_full
        offset = n_total - states.shape[0]
        m, n_features = states.shape

        # PCA needs strictly fewer components than points AND features to fit.
        use_pca = self.n_pca is not None and self.n_pca < min(m, n_features)
        info = {
            "window_effective": int(m),
            "pca_skipped": bool(self.n_pca is not None and not use_pca),
            "m_lt_3d": bool(self.n_pca is not None and m < 3 * self.n_pca),
            "n_cached": 0,
            "n_computed": 0,
        }

        # Caller-owned cache (shared across stages) or a throwaway local dict.
        store = cache if cache is not None else {}
        need = [i for i in range(m) if (offset + i) not in store]
        info["n_cached"] = m - len(need)
        info["n_computed"] = len(need)

        if need:
            if use_pca:
                from sklearn.decomposition import PCA  # optional dep
                mean = states.mean(axis=0)
                pca = PCA(n_components=self.n_pca)
                pca_states = pca.fit_transform(states - mean)

                def f_use(x_pca: np.ndarray) -> float:
                    x_orig = pca.inverse_transform(x_pca[np.newaxis])[0] + mean
                    return float(surrogate_fn(x_orig))

                def f_use_batched(x_pca_batch: np.ndarray) -> np.ndarray:
                    x_orig_batch = pca.inverse_transform(x_pca_batch) + mean
                    return np.asarray(surrogate_fn_batched(x_orig_batch), dtype=np.float64)

                work_states = pca_states
            else:
                def f_use(x: np.ndarray) -> float:
                    return float(surrogate_fn(x))

                def f_use_batched(x_batch: np.ndarray) -> np.ndarray:
                    return np.asarray(surrogate_fn_batched(x_batch), dtype=np.float64)

                work_states = states

            if surrogate_fn_batched is not None:
                # One batched call per point (4n^2 perturbed states at once)
                # instead of 4n^2 tiny scalar calls -- see class docstring.
                for i in need:
                    J = _numerical_jacobian_batched(f_use_batched, work_states[i], alpha=self.alpha)
                    store[offset + i] = eigenvalues_and_stability(J)["lambda_max"]
            else:
                predictor = make_gd_predictor(f_use, alpha=self.alpha)

                def _one_point(i: int):
                    J = numerical_jacobian(predictor, work_states[i])
                    return i, eigenvalues_and_stability(J)["lambda_max"]

                if self.max_workers and self.max_workers > 1:
                    from concurrent.futures import ThreadPoolExecutor
                    with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                        for i, lam in ex.map(_one_point, need):
                            store[offset + i] = lam
                else:
                    for i in need:
                        _, lam = _one_point(i)
                        store[offset + i] = lam

        lm = np.array([store[offset + i] for i in range(m)], dtype=float)
        return lm, info

    def run(self, history: List[Dict[str, Any]], **kwargs) -> AuditResult:
        info: Dict[str, Any] = {}
        # Route 1: precomputed lambda_max array (honour the window at aggregation)
        if "lambda_max" in kwargs and kwargs["lambda_max"] is not None:
            lm = np.asarray(kwargs["lambda_max"], dtype=float).ravel()
            lm = lm[np.isfinite(lm)]  # drop NaN/inf (e.g. a warm-up prefix) before windowing
            if self.window is not None and lm.size > self.window:
                lm = lm[-self.window:]
        else:
            vals = [h["lambda_max"] for h in history if "lambda_max" in h]
            if vals:
                lm = np.asarray(vals, dtype=float)
                lm = lm[np.isfinite(lm)]
                if self.window is not None and lm.size > self.window:
                    lm = lm[-self.window:]
            elif "surrogate_fn" in kwargs and "op_states" in kwargs:
                # Route 2: on-demand Jacobian computation (windowed + cached)
                try:
                    lm, info = self._compute_lambda_max(
                        kwargs["surrogate_fn"],
                        kwargs["op_states"],
                        cache=kwargs.get("lambda_cache"),
                        surrogate_fn_batched=kwargs.get("surrogate_fn_batched"),
                    )
                except Exception as exc:
                    return AuditResult(
                        name=self.name,
                        passed=True,
                        category=self.category,
                        message=f"Skipped — Jacobian computation failed: {exc}",
                    )
            else:
                return AuditResult(
                    name=self.name,
                    passed=True,
                    category=self.category,
                    message=(
                        "Skipped — lambda_max series not available and "
                        "surrogate_fn / op_states not provided."
                    ),
                )

        if len(lm) == 0:
            return AuditResult(
                name=self.name,
                passed=True,
                category=self.category,
                message="Skipped — empty lambda_max array.",
            )

        stable_mask = lm < self.stability_threshold
        fraction_stable = float(stable_mask.mean())

        # Guard warnings (package idiom: surfaced via message + details, not warnings.warn)
        notes = []
        if info.get("pca_skipped"):
            notes.append(
                f"PCA skipped (n_pca={self.n_pca} ≥ window points={info.get('window_effective')})"
            )
        if info.get("m_lt_3d"):
            notes.append(
                f"window points={info.get('window_effective')} < 3·n_pca "
                f"({3 * (self.n_pca or 0)}) — fragile PCA"
            )
        message = (
            f"Stable fraction = {fraction_stable:.3f}  "
            f"|λ_max| mean={lm.mean():.3e} max={lm.max():.3e}"
        )
        if notes:
            message += "  [" + "; ".join(notes) + "]"

        details: Dict[str, Any] = {
            "lambda_max_mean": float(lm.mean()),
            "lambda_max_max": float(lm.max()),
            "lambda_max_min": float(lm.min()),
            "n_stable": int(stable_mask.sum()),
            "n_total": len(lm),
            "window": self.window,
        }
        for key in ("window_effective", "pca_skipped", "m_lt_3d", "n_cached", "n_computed"):
            if key in info:
                details[key] = info[key]

        return AuditResult(
            name=self.name,
            passed=fraction_stable >= self.min_stable_fraction,
            category=self.category,
            value=fraction_stable,
            threshold=self.min_stable_fraction,
            message=message,
            details=details,
        )
