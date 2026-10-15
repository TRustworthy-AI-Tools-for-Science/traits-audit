"""Core Lyapunov / Jacobian numerics shared across traits-audit.

Pure numpy/scipy (no matplotlib). Single source of truth for what used to be
copied independently in ``_viz.py``, ``_lyapunov.py``,
``checks/lyapunov.py`` and :mod:`traits_audit.stability.dmdc`, and in
battery-forecast's ``exp3_stability_analysis._stable_lyapunov_P``.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

__all__ = [
    "numerical_jacobian",
    "numerical_jacobian_batched",
    "eigenvalues_and_stability",
    "rescale_to_stable",
    "lyapunov_P",
    "compute_lyapunov",
    "make_gd_predictor",
]


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


def numerical_jacobian_batched(
    f_batched,
    state: np.ndarray,
    alpha: float,
    dx: float = 1e-4,
    eps: float = 1e-5,
) -> np.ndarray:
    """Same n×n Jacobian as ``numerical_jacobian(make_gd_predictor(f_scalar,
    alpha, eps), state, None, dx)`` for the scalar-callable ``f_scalar`` that
    ``f_batched`` batches — but issued as exactly ONE call to ``f_batched``
    covering all 4n² perturbed states the nested (Jacobian-of-a-numerically-
    differentiated-gradient) computation needs, instead of 4n² separate
    scalar calls.

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
        :func:`make_gd_predictor` takes.
    state : (D,) ndarray
    alpha : float
        Gradient-descent step size (see :func:`make_gd_predictor`).
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


def rescale_to_stable(A: np.ndarray, rho_max: float = 0.99) -> np.ndarray:
    """Return ``A`` unchanged if its spectral radius is < 1, else ``A``
    rescaled to spectral radius ``rho_max``.

    The discrete Lyapunov/Stein equations have no positive-definite solution
    for ``rho(A) >= 1``; rescaling keeps V(x) = xᵀPx (and the Gramians) on one
    consistent, comparable scale across stable and unstable operating points.
    """
    rho = float(np.abs(np.linalg.eigvals(A)).max())
    if rho >= 1.0:
        return A * (rho_max / rho)
    return A


def lyapunov_P(A: np.ndarray, rho_max: float = 0.99) -> np.ndarray:
    """Solve the discrete Lyapunov equation Aᵀ P A − P = −I for P.

    ``A`` is first passed through :func:`rescale_to_stable`. Raises whatever
    ``scipy.linalg.solve_discrete_lyapunov`` raises on a degenerate ``A``;
    see :func:`compute_lyapunov` for the non-raising variant.
    """
    A = rescale_to_stable(A, rho_max)
    return solve_discrete_lyapunov(A.T, np.eye(len(A)))


def compute_lyapunov(A: np.ndarray, rho_max: float = 0.99) -> np.ndarray | None:
    """Like :func:`lyapunov_P`, but returns ``None`` instead of raising if
    ``solve_discrete_lyapunov`` itself fails (e.g. a degenerate/singular
    ``A``)."""
    A = rescale_to_stable(A, rho_max)  # outside the try: an eigvals failure still raises
    try:
        return solve_discrete_lyapunov(A.T, np.eye(len(A)))
    except Exception:
        return None


def make_gd_predictor(f_scalar, alpha: float = 0.01, eps: float = 1e-5):
    """Build a discrete gradient-descent step predictor for Lyapunov analysis.

    Models x_{t+1} = x_t − α ∇f(x_t).  The Jacobian of this map equals
    J = I − α H_f, so |λ(J)| < 1 iff all eigenvalues of H_f lie in (0, 2/α).

    **Purely real poles**: Because f_scalar is a real scalar function, its
    Hessian H_f is always real and symmetric, and J = I − α H_f inherits
    that symmetry.  All eigenvalues of J are therefore guaranteed to be purely
    real regardless of the surrogate architecture.

    Parameters
    ----------
    f_scalar : callable
        ``f_scalar(state: np.ndarray) → float``
    alpha : float
        Gradient-descent step size used to define the dynamical system.
    eps : float
        Finite-difference step for computing ∇f numerically.
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
