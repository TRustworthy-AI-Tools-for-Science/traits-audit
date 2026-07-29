import numpy as np
import pytest

from traits_audit.checks.procedural import (
    ProceduralVarianceShareCheck,
    DataVarianceShareCheck,
    MisspecificationResidualFloorCheck,
)


def _ridge_fit_fn(l2=1e-3):
    def fit_fn(X, y, *, seed=None):
        rng = np.random.default_rng(seed)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_features = X.shape[1]
        A = X.T @ X + l2 * np.eye(n_features)
        b = X.T @ y
        # Seed-dependent tiny perturbation models "optimizer/init variability"
        # on data that is otherwise held fixed.
        noise = rng.standard_normal(n_features) * 0.02 if seed is not None else 0.0
        w = np.linalg.solve(A, b) + noise

        def predict(Xe):
            return np.asarray(Xe, dtype=float) @ w

        return predict

    return fit_fn


def _synthetic_data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2))
    true_w = np.array([2.0, -1.0])
    y = X @ true_w + rng.standard_normal(n) * 0.05
    return X, y


# ── ProceduralVarianceShareCheck ─────────────────────────────────────────────

def test_pvs_near_one_when_ensemble_is_literally_a_seed_sweep():
    X, y = _synthetic_data()
    X_eval = np.random.default_rng(1).standard_normal((15, 2))
    fit_fn = _ridge_fit_fn()

    from traits_audit.refit import refit_sweep_seed
    ensemble = refit_sweep_seed(fit_fn, X, y, X_eval, k=25, base_seed=100)

    check = ProceduralVarianceShareCheck(k_refits=25, base_seed=0)
    result = check.run([], fit_fn=fit_fn, X_train=X, y_train=y, X_eval=X_eval, y_pred_ensemble=ensemble)
    assert result.value == pytest.approx(1.0, rel=0.5)


def test_pvs_near_zero_when_ensemble_spread_is_from_data_not_seed():
    # A small, noisy training set so bootstrap resampling meaningfully moves
    # the fit -- with n=60 (as in _synthetic_data) the regression is so
    # well-determined that bootstrap variance is negligible next to the
    # fixed 0.02-scale seed noise, which would give the opposite result.
    X, y = _synthetic_data(n=12, seed=0)
    y = y + np.random.default_rng(2).standard_normal(12) * 1.0
    X_eval = np.random.default_rng(1).standard_normal((15, 2))
    fit_fn = _ridge_fit_fn()

    from traits_audit.refit import refit_sweep_bootstrap
    ensemble = refit_sweep_bootstrap(fit_fn, X, y, X_eval, k=25, seed=100, fixed_seed=None)

    check = ProceduralVarianceShareCheck(k_refits=25, base_seed=0, pvs_tolerance=None)
    result = check.run([], fit_fn=fit_fn, X_train=X, y_train=y, X_eval=X_eval, y_pred_ensemble=ensemble)
    # procedural (seed-only) variance should be much smaller than the
    # ensemble's data-driven bootstrap variance
    assert result.value < 0.5


def test_pvs_route1_and_route2_agree():
    X, y = _synthetic_data()
    X_eval = np.random.default_rng(1).standard_normal((15, 2))
    fit_fn = _ridge_fit_fn()

    from traits_audit.refit import refit_sweep_seed
    procedural = refit_sweep_seed(fit_fn, X, y, X_eval, k=25, base_seed=0)
    ensemble = refit_sweep_seed(fit_fn, X, y, X_eval, k=25, base_seed=100)

    route1 = ProceduralVarianceShareCheck().run(
        [], y_pred_procedural=procedural, y_pred_ensemble=ensemble
    )
    route2 = ProceduralVarianceShareCheck(k_refits=25, base_seed=0).run(
        [], fit_fn=fit_fn, X_train=X, y_train=y, X_eval=X_eval, y_pred_ensemble=ensemble
    )
    assert route1.value == pytest.approx(route2.value, rel=1e-9)


def test_pvs_skips_without_data():
    result = ProceduralVarianceShareCheck().run([])
    assert result.passed
    assert "Skipped" in result.message


def test_pvs_report_only_by_default():
    ensemble = np.ones((5, 10))
    procedural = np.ones((5, 10)) * 100  # wildly different value
    result = ProceduralVarianceShareCheck().run([], y_pred_procedural=procedural, y_pred_ensemble=ensemble)
    assert result.passed  # zero-variance ensemble -> skipped, still passed=True


# ── DataVarianceShareCheck ───────────────────────────────────────────────────

def test_dvs_skips_without_data():
    result = DataVarianceShareCheck().run([])
    assert result.passed
    assert "Skipped" in result.message


def test_dvs_computes_from_precomputed_matrices():
    rng = np.random.default_rng(0)
    ensemble = rng.standard_normal((10, 20))
    data_matrix = rng.standard_normal((10, 20)) * 0.5
    result = DataVarianceShareCheck().run([], y_pred_data=data_matrix, y_pred_ensemble=ensemble)
    assert result.value == pytest.approx(0.25, rel=0.5)


# ── MisspecificationResidualFloorCheck ──────────────────────────────────────

def test_mrf_zero_floor_from_precomputed_curve():
    Ns = np.array([10.0, 20.0, 40.0, 80.0, 160.0, 320.0])
    rng = np.random.default_rng(0)
    curve = 5.0 * Ns ** (-0.5) + rng.normal(scale=0.001, size=len(Ns))
    result = MisspecificationResidualFloorCheck().run([], learning_curve=(Ns, curve))
    assert result.value == pytest.approx(0.0, abs=0.05)


def test_mrf_nonzero_floor_from_precomputed_curve():
    Ns = np.array([10.0, 20.0, 40.0, 80.0, 160.0, 320.0])
    rng = np.random.default_rng(0)
    curve = 5.0 * Ns ** (-0.5) + 0.3 + rng.normal(scale=0.001, size=len(Ns))
    result = MisspecificationResidualFloorCheck(mrf_threshold=0.1).run([], learning_curve=(Ns, curve))
    assert not result.passed
    assert result.value == pytest.approx(0.3, abs=0.05)


def test_mrf_skips_with_too_few_points():
    Ns = np.array([10.0, 20.0])
    curve = np.array([1.0, 0.5])
    result = MisspecificationResidualFloorCheck().run([], learning_curve=(Ns, curve))
    assert result.passed
    assert "Skipped" in result.message


def test_mrf_route1_and_route2_agree():
    fit_fn = _ridge_fit_fn()
    X, y = _synthetic_data(n=200, seed=2)
    X_eval, y_eval = _synthetic_data(n=30, seed=3)

    from traits_audit.refit import nested_subset_curve
    subset_fracs = (0.2, 0.4, 0.6, 0.8, 1.0)
    Ns, curve, _ = nested_subset_curve(fit_fn, X, y, X_eval, y_eval, subset_fracs, reps=3, seed=0)

    route1 = MisspecificationResidualFloorCheck(subset_fracs=subset_fracs, seed=0).run(
        [], learning_curve=(Ns, curve)
    )
    route2 = MisspecificationResidualFloorCheck(subset_fracs=subset_fracs, reps=3, seed=0).run(
        [], fit_fn=fit_fn, X_train=X, y_train=y, X_eval=X_eval, y_eval=y_eval
    )
    assert route1.value == pytest.approx(route2.value, abs=1e-6)
