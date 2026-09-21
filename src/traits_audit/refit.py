"""
Refit-sweep helpers for Procedural/Data Variance Share and the Misspecification
Residual Floor learning curve.

``fit_fn`` is entirely user-supplied — traits-audit does not own the surrogate
model, so these helpers only orchestrate calling it repeatedly with controlled
variation (seed only, data only, or nested subset size) and collecting the
resulting predictions. Signature contract for ``fit_fn``::

    fit_fn(X_train, y_train, *, seed=None) -> predict_fn
    predict_fn(X_eval) -> y_pred                    # (n_eval,) array, or
    predict_fn(X_eval) -> (y_pred_mean, y_pred_std)  # duck-typed tuple

``nested_subset_curve`` accepts either return shape from ``predict_fn``; when
a ``(mean, std)`` tuple is returned it scores held-out Gaussian NLL, otherwise
it scores held-out residual variance directly.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def refit_sweep_seed(
    fit_fn: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    k: int,
    base_seed: int = 0,
) -> np.ndarray:
    """Refit ``k`` times varying ONLY the seed; ``X_train``/``y_train`` are
    identical each time. Returns a ``(k, n_eval)`` prediction matrix. Feeds
    ``ProceduralVarianceShareCheck``."""
    preds = []
    for i in range(k):
        predict_fn = fit_fn(X_train, y_train, seed=base_seed + i)
        out = predict_fn(X_eval)
        mean = out[0] if isinstance(out, tuple) else out
        preds.append(np.asarray(mean, dtype=float).ravel())
    return np.vstack(preds)


def refit_sweep_bootstrap(
    fit_fn: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    k: int,
    seed: int = 0,
    fixed_seed: int = 0,
) -> np.ndarray:
    """Refit ``k`` times on a bootstrap resample of ``(X_train, y_train)``;
    ``fit_fn``'s own seed is HELD FIXED at ``fixed_seed`` so only the data
    varies. Returns a ``(k, n_eval)`` prediction matrix. Feeds
    ``DataVarianceShareCheck``."""
    rng = np.random.default_rng(seed)
    n = len(y_train)
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train, dtype=float)
    preds = []
    for _ in range(k):
        idx = rng.integers(0, n, size=n)
        predict_fn = fit_fn(X_train[idx], y_train[idx], seed=fixed_seed)
        out = predict_fn(X_eval)
        mean = out[0] if isinstance(out, tuple) else out
        preds.append(np.asarray(mean, dtype=float).ravel())
    return np.vstack(preds)


def nested_subset_curve(
    fit_fn: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    subset_fracs: Sequence[float],
    reps: int = 3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each fraction ``f`` in ``subset_fracs``, draw ``reps`` random
    subsets of size ``round(f * N)`` (a distinct seed per rep), fit, and
    score held-out performance on ``(X_eval, y_eval)``.

    Duck-types on ``predict_fn``'s return: a ``(mean, std)`` tuple is scored
    by Gaussian NLL; a mean-only array is scored by held-out residual
    variance directly (not an excess over any in-sample floor -- subtracting
    one would remove part of the very asymptote
    ``MisspecificationResidualFloorCheck``'s curve fit is trying to
    estimate, since that floor IS the quantity the fitted ``c`` parameter
    should converge to as N grows).

    Returns ``(Ns, nll_mean_per_N, nll_values_per_rep_matrix)`` where ``Ns``
    is ``(len(subset_fracs),)``, the mean curve is the same shape, and the
    matrix is ``(len(subset_fracs), reps)`` for bootstrap CI purposes. Feeds
    ``MisspecificationResidualFloorCheck``."""
    rng = np.random.default_rng(seed)
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train, dtype=float)
    y_eval = np.asarray(y_eval, dtype=float).ravel()
    n = len(y_train)

    Ns = np.array([max(2, round(f * n)) for f in subset_fracs])
    values = np.empty((len(subset_fracs), reps))

    for i, N in enumerate(Ns):
        for j in range(reps):
            idx = rng.choice(n, size=int(N), replace=False)
            predict_fn = fit_fn(X_train[idx], y_train[idx], seed=int(rng.integers(0, 2**31 - 1)))
            out = predict_fn(X_eval)
            if isinstance(out, tuple):
                mean, std = out
                mean = np.asarray(mean, dtype=float).ravel()
                std = np.maximum(np.asarray(std, dtype=float).ravel(), 1e-12)
                z = (y_eval - mean) / std
                values[i, j] = float(np.mean(0.5 * np.log(2 * np.pi) + np.log(std) + 0.5 * z**2))
            else:
                mean = np.asarray(out, dtype=float).ravel()
                residual = y_eval - mean
                values[i, j] = float(np.var(residual))

    return Ns.astype(float), values.mean(axis=1), values
