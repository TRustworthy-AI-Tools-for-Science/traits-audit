"""Lyapunov stability check for surrogate model landscape analysis."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..base import AuditCategory, AuditCheck, AuditResult


# ── Pure computation helpers (numpy-only, no matplotlib) ─────────────────────

def _numerical_jacobian(
    predictor,
    state: np.ndarray,
    dx: float = 1e-4,
) -> np.ndarray:
    """n×n Jacobian of predictor(state) → ℝⁿ via central differences."""
    n = len(state)
    J = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        s_plus = state.copy(); s_plus[i] += dx
        s_minus = state.copy(); s_minus[i] -= dx
        J[:, i] = (predictor(s_plus) - predictor(s_minus)) / (2.0 * dx)
    return J


def _eigenvalues(J: np.ndarray) -> np.ndarray:
    return np.abs(np.linalg.eigvals(J))


def _gd_predictor(f_scalar, alpha: float = 0.01, eps: float = 1e-5):
    """Wrap f: ℝⁿ → ℝ as the discrete gradient-descent step x_{t+1} = x_t − α∇f."""
    def predictor(state: np.ndarray) -> np.ndarray:
        n = len(state)
        grad = np.zeros(n)
        for i in range(n):
            s_p = state.copy(); s_p[i] += eps
            s_m = state.copy(); s_m[i] -= eps
            grad[i] = (f_scalar(s_p) - f_scalar(s_m)) / (2.0 * eps)
        return state - alpha * grad
    return predictor


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
        If set, PCA-reduces ``op_states`` to this many dimensions before
        computing Jacobians.  Useful for high-dimensional feature spaces.
        Requires ``scikit-learn``.

    Required data (at least one route)
    -----------------------------------
    Precomputed (preferred): ``lambda_max`` kwarg (array_like) or
    ``lambda_max`` key in each history dict (float per step).
    On-demand: ``surrogate_fn`` kwarg (callable) and ``op_states``
    kwarg (ndarray of shape (N, D)).
    """

    def __init__(
        self,
        stability_threshold: float = 1.0,
        min_stable_fraction: float = 0.5,
        alpha: float = 0.01,
        n_pca: Optional[int] = None,
    ):
        self.stability_threshold = stability_threshold
        self.min_stable_fraction = min_stable_fraction
        self.alpha = alpha
        self.n_pca = n_pca

    @property
    def name(self) -> str:
        return "LyapunovStability"

    @property
    def category(self) -> AuditCategory:
        return AuditCategory.EPISTEMIC

    def _compute_lambda_max(
        self, surrogate_fn, op_states: np.ndarray
    ) -> np.ndarray:
        states = np.asarray(op_states, dtype=float)

        if self.n_pca is not None and states.shape[1] > self.n_pca:
            from sklearn.decomposition import PCA  # optional dep
            mean = states.mean(axis=0)
            pca = PCA(n_components=self.n_pca)
            pca_states = pca.fit_transform(states - mean)

            def f_use(x_pca: np.ndarray) -> float:
                x_orig = pca.inverse_transform(x_pca[np.newaxis])[0] + mean
                return float(surrogate_fn(x_orig))

            work_states = pca_states
        else:
            def f_use(x: np.ndarray) -> float:
                return float(surrogate_fn(x))

            work_states = states

        predictor = _gd_predictor(f_use, alpha=self.alpha)
        lm = []
        for state in work_states:
            J = _numerical_jacobian(predictor, state)
            lm.append(float(_eigenvalues(J).max()))
        return np.array(lm)

    def run(self, history: List[Dict[str, Any]], **kwargs) -> AuditResult:
        # Route 1: precomputed lambda_max array
        if "lambda_max" in kwargs and kwargs["lambda_max"] is not None:
            lm = np.asarray(kwargs["lambda_max"], dtype=float).ravel()
        else:
            vals = [h["lambda_max"] for h in history if "lambda_max" in h]
            if vals:
                lm = np.asarray(vals, dtype=float)
            elif "surrogate_fn" in kwargs and "op_states" in kwargs:
                # Route 2: on-demand Jacobian computation
                try:
                    lm = self._compute_lambda_max(
                        kwargs["surrogate_fn"],
                        kwargs["op_states"],
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

        return AuditResult(
            name=self.name,
            passed=fraction_stable >= self.min_stable_fraction,
            category=self.category,
            value=fraction_stable,
            threshold=self.min_stable_fraction,
            message=(
                f"Stable fraction = {fraction_stable:.3f}  "
                f"|λ_max| mean={lm.mean():.3e} max={lm.max():.3e}"
            ),
            details={
                "lambda_max_mean": float(lm.mean()),
                "lambda_max_max": float(lm.max()),
                "lambda_max_min": float(lm.min()),
                "n_stable": int(stable_mask.sum()),
                "n_total": len(lm),
            },
        )
