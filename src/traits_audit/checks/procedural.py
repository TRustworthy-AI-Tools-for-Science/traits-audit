"""Model/approximation/misspecification/procedural taxonomy class.

See METRIC_TAXONOMY_AUDIT.md §4.6. This class has no metrological
antecedent for its procedural sub-term and no metric anywhere prior to
this. PVS/DVS decompose what a deep ensemble reports as a single variance
number into its optimizer-variability and data-sufficiency components; MRF
tests whether more data would actually close a misspecification gap.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..base import AuditCategory, AuditCheck, AuditResult
from ..refit import nested_subset_curve, refit_sweep_bootstrap, refit_sweep_seed


def _mean_pointwise_variance(matrix: np.ndarray) -> float:
    return float(np.mean(np.var(np.asarray(matrix, dtype=float), axis=0)))


class ProceduralVarianceShareCheck(AuditCheck):
    """
    Procedural Variance Share (PVS) — the fraction of a deployed ensemble's
    variance attributable purely to optimizer/seed variability, holding the
    training data fixed.

    Jimenez et al. (2025) find that deep ensembles capture the procedural
    component of estimation variance and *not* the data component, so a deep
    ensemble should score PVS ~= 1. That makes PVS a falsifiable test of that
    claim and, independently, a warning label: an ensemble spread with
    PVS ~= 1 is telling you about optimizer variability, not data
    sufficiency, and should not drive acquisition.

    Two data routes:

    - Route 1 (precomputed): ``y_pred_procedural`` kwarg, ``(k, n_eval)``.
    - Route 2 (on-demand): ``fit_fn``, ``X_train``, ``y_train``, ``X_eval``
      kwargs -> :func:`traits_audit.refit.refit_sweep_seed`.

    Both routes also need ``y_pred_ensemble`` ``(n_models, n_eval)`` — the
    deployed ensemble's own predictions (``Var_ensemble``).

    ``PVS = mean_point(var(y_pred_procedural)) / mean_point(var(y_pred_ensemble))``.

    Parameters
    ----------
    k_refits : int
        Number of seed-varied refits for Route 2 (default 20).
    base_seed : int
        Base seed for Route 2's sweep.
    pvs_tolerance : float or None
        If set, ``passed = abs(PVS - 1) <= pvs_tolerance``. ``None``
        (default) disables pass/fail — PVS ~= 1 is the literature's
        *falsifiable claim*, not something this check should fail by
        default.

    References
    ----------
    Jimenez, I., Jurgens, D. & Waegeman, W. (2025). arXiv:2505.23506.
    Lakshminarayanan, B., Pritzel, A. & Blundell, C. (2017). NeurIPS 30.

    Required data (kwargs)
    -----------------------
    Route 1: ``y_pred_procedural``, ``y_pred_ensemble``.
    Route 2: ``fit_fn``, ``X_train``, ``y_train``, ``X_eval``, ``y_pred_ensemble``.
    """

    def __init__(self, k_refits: int = 20, base_seed: int = 0, pvs_tolerance: float | None = None):
        self.k_refits = k_refits
        self.base_seed = base_seed
        self.pvs_tolerance = pvs_tolerance

    @property
    def name(self) -> str:
        return "ProceduralVarianceShare"

    @property
    def category(self) -> AuditCategory:
        return AuditCategory.MODEL_PROCEDURAL

    def _get_procedural_matrix(self, kwargs) -> np.ndarray | None:
        if kwargs.get("y_pred_procedural") is not None:
            return np.asarray(kwargs["y_pred_procedural"], dtype=float)
        required = ("fit_fn", "X_train", "y_train", "X_eval")
        if all(kwargs.get(k) is not None for k in required):
            return refit_sweep_seed(
                kwargs["fit_fn"], kwargs["X_train"], kwargs["y_train"], kwargs["X_eval"],
                self.k_refits, self.base_seed,
            )
        return None

    def run(self, history: list[dict[str, Any]], **kwargs) -> AuditResult:
        ensemble = kwargs.get("y_pred_ensemble")
        procedural = self._get_procedural_matrix(kwargs)

        if ensemble is None or procedural is None:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message=(
                    "Skipped — need y_pred_ensemble plus either y_pred_procedural "
                    "or (fit_fn, X_train, y_train, X_eval)."
                ),
            )

        var_ensemble = _mean_pointwise_variance(ensemble)
        var_procedural = _mean_pointwise_variance(procedural)
        if var_ensemble <= 0:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Skipped — ensemble has zero variance.",
            )

        value = var_procedural / var_ensemble
        passed = True if self.pvs_tolerance is None else abs(value - 1.0) <= self.pvs_tolerance
        return AuditResult(
            name=self.name,
            passed=passed,
            category=self.category,
            value=value,
            threshold=self.pvs_tolerance,
            message=f"PVS = {value:.4f}  (var_procedural={var_procedural:.4g}, var_ensemble={var_ensemble:.4g})",
            details={"var_procedural": var_procedural, "var_ensemble": var_ensemble},
        )


class DataVarianceShareCheck(AuditCheck):
    """
    Data Variance Share (DVS) — the complement of PVS: variance from
    resampling the *data* (bootstrap), holding the optimizer seed fixed.

    Report PVS and DVS side by side (see
    :func:`traits_audit.validation.find_unpaired_checks`, which pairs them
    advisorially) — their ratio decomposes what a deep ensemble currently
    reports as a single variance number.

    Two data routes:

    - Route 1 (precomputed): ``y_pred_data`` kwarg, ``(k, n_eval)``.
    - Route 2 (on-demand): ``fit_fn``, ``X_train``, ``y_train``, ``X_eval``
      kwargs -> :func:`traits_audit.refit.refit_sweep_bootstrap`.

    Both routes also need ``y_pred_ensemble``.

    Parameters
    ----------
    k_refits : int
        Number of bootstrap refits for Route 2 (default 20).
    fixed_seed : int
        Seed held fixed across Route 2's bootstrap sweep.
    dvs_tolerance : float or None
        Report-only by default (``None``), matching ``ProceduralVarianceShareCheck``.

    References
    ----------
    Jimenez, I., Jurgens, D. & Waegeman, W. (2025). arXiv:2505.23506.

    Required data (kwargs)
    -----------------------
    Route 1: ``y_pred_data``, ``y_pred_ensemble``.
    Route 2: ``fit_fn``, ``X_train``, ``y_train``, ``X_eval``, ``y_pred_ensemble``.
    """

    def __init__(self, k_refits: int = 20, fixed_seed: int = 0, dvs_tolerance: float | None = None):
        self.k_refits = k_refits
        self.fixed_seed = fixed_seed
        self.dvs_tolerance = dvs_tolerance

    @property
    def name(self) -> str:
        return "DataVarianceShare"

    @property
    def category(self) -> AuditCategory:
        return AuditCategory.MODEL_PROCEDURAL

    def _get_data_matrix(self, kwargs) -> np.ndarray | None:
        if kwargs.get("y_pred_data") is not None:
            return np.asarray(kwargs["y_pred_data"], dtype=float)
        required = ("fit_fn", "X_train", "y_train", "X_eval")
        if all(kwargs.get(k) is not None for k in required):
            return refit_sweep_bootstrap(
                kwargs["fit_fn"], kwargs["X_train"], kwargs["y_train"], kwargs["X_eval"],
                self.k_refits, seed=self.fixed_seed, fixed_seed=self.fixed_seed,
            )
        return None

    def run(self, history: list[dict[str, Any]], **kwargs) -> AuditResult:
        ensemble = kwargs.get("y_pred_ensemble")
        data_matrix = self._get_data_matrix(kwargs)

        if ensemble is None or data_matrix is None:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message=(
                    "Skipped — need y_pred_ensemble plus either y_pred_data "
                    "or (fit_fn, X_train, y_train, X_eval)."
                ),
            )

        var_ensemble = _mean_pointwise_variance(ensemble)
        var_data = _mean_pointwise_variance(data_matrix)
        if var_ensemble <= 0:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Skipped — ensemble has zero variance.",
            )

        value = var_data / var_ensemble
        passed = True if self.dvs_tolerance is None else abs(value - 1.0) <= self.dvs_tolerance
        return AuditResult(
            name=self.name,
            passed=passed,
            category=self.category,
            value=value,
            threshold=self.dvs_tolerance,
            message=f"DVS = {value:.4f}  (var_data={var_data:.4g}, var_ensemble={var_ensemble:.4g})",
            details={"var_data": var_data, "var_ensemble": var_ensemble},
        )


class MisspecificationResidualFloorCheck(AuditCheck):
    """
    Misspecification Residual Floor (MRF) — the non-vanishing component of a
    learning curve: neither aleatoric nor asymptotically reducible.

    Fits ``a * N^(-gamma) + c`` to a held-out learning curve over nested
    training subsets. ``c_hat = 0`` means more data will close the gap;
    ``c_hat > 0`` falsifies that assumption — expected to dominate in the
    low-noise regime (Swinburne & Perez 2025).

    Two data routes:

    - Route 1 (precomputed): ``learning_curve = (Ns, nll_values)``.
    - Route 2 (on-demand): ``fit_fn``, ``X_train``, ``y_train``, ``X_eval``,
      ``y_eval`` kwargs -> :func:`traits_audit.refit.nested_subset_curve`.

    Parameters
    ----------
    subset_fracs : sequence of float
        Nested-subset fractions for Route 2 (default
        ``(0.1,0.2,0.4,0.6,0.8,1.0)``).
    reps : int
        Repetitions per fraction for Route 2 (default 3).
    seed : int
        RNG seed for Route 2 and the bootstrap CI.
    mrf_threshold : float or None
        Maximum acceptable c_hat. ``None`` (default) disables pass/fail.

    References
    ----------
    Swinburne, T. D. & Perez, D. (2025). *Mach. Learn.: Sci. Technol.*,
    6(1):015008.
    Perez, A. et al. (2025). arXiv:2502.07104.

    Required data (kwargs)
    -----------------------
    Route 1: ``learning_curve``.
    Route 2: ``fit_fn``, ``X_train``, ``y_train``, ``X_eval``, ``y_eval``.
    """

    def __init__(
        self,
        subset_fracs=(0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
        reps: int = 3,
        seed: int = 0,
        mrf_threshold: float | None = None,
    ):
        self.subset_fracs = tuple(subset_fracs)
        self.reps = reps
        self.seed = seed
        self.mrf_threshold = mrf_threshold

    @property
    def name(self) -> str:
        return "MisspecificationResidualFloor"

    @property
    def category(self) -> AuditCategory:
        return AuditCategory.MODEL_PROCEDURAL

    def _get_curve(self, kwargs):
        if kwargs.get("learning_curve") is not None:
            Ns, nll = kwargs["learning_curve"]
            return np.asarray(Ns, dtype=float), np.asarray(nll, dtype=float), None
        required = ("fit_fn", "X_train", "y_train", "X_eval", "y_eval")
        if all(kwargs.get(k) is not None for k in required):
            Ns, curve, matrix = nested_subset_curve(
                kwargs["fit_fn"], kwargs["X_train"], kwargs["y_train"],
                kwargs["X_eval"], kwargs["y_eval"], self.subset_fracs, self.reps, self.seed,
            )
            return Ns, curve, matrix
        return None, None, None

    @staticmethod
    def _fit_curve(Ns: np.ndarray, values: np.ndarray):
        from scipy.optimize import curve_fit

        def model(N, a, gamma, c):
            return a * N ** (-gamma) + c

        # Heuristic initial guess from a log-log linear fit on the extremes.
        try:
            slope = (np.log(values[-1] + 1e-12) - np.log(values[0] + 1e-12)) / (
                np.log(Ns[-1]) - np.log(Ns[0])
            )
        except Exception:
            slope = -0.5
        p0 = [max(values[0] - values[-1], 1e-6), max(-slope, 1e-3), max(values[-1], 0.0)]
        bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
        popt, _ = curve_fit(model, Ns, values, p0=p0, bounds=bounds, maxfev=10000)
        return popt  # a, gamma, c

    def run(self, history: list[dict[str, Any]], **kwargs) -> AuditResult:
        Ns, curve, matrix = self._get_curve(kwargs)
        if Ns is None or len(Ns) < 4:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message=(
                    "Skipped — need >= 4 (N, value) points via learning_curve or "
                    "(fit_fn, X_train, y_train, X_eval, y_eval)."
                ),
            )

        try:
            a_hat, gamma_hat, c_hat = self._fit_curve(Ns, curve)
        except Exception as exc:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message=f"Skipped — curve fit failed: {exc}",
            )

        c_ci = None
        if matrix is not None and matrix.shape[1] >= 2:
            rng = np.random.default_rng(self.seed)
            boot_c = []
            for _ in range(200):
                rep_idx = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
                resampled_curve = matrix[:, rep_idx].mean(axis=1)
                try:
                    _, _, c_b = self._fit_curve(Ns, resampled_curve)
                    boot_c.append(c_b)
                except Exception:
                    continue
            if len(boot_c) >= 10:
                c_ci = tuple(float(x) for x in np.percentile(boot_c, [2.5, 97.5]))

        passed = True if self.mrf_threshold is None else c_hat <= self.mrf_threshold
        return AuditResult(
            name=self.name,
            passed=passed,
            category=self.category,
            value=float(c_hat),
            threshold=self.mrf_threshold,
            message=f"c_hat = {c_hat:.4f}  (a_hat={a_hat:.4f}, gamma_hat={gamma_hat:.4f}, CI={c_ci})",
            details={
                "a_hat": float(a_hat), "gamma_hat": float(gamma_hat), "c_hat": float(c_hat),
                "c_ci": c_ci, "Ns": Ns.tolist(), "nll_curve": curve.tolist(),
            },
        )
