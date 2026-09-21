import numpy as np

from traits_audit.checks.ergodic import (
    DMDcSpectralRadiusCheck,
    EnsembleIndependenceDeficitCheck,
    ResidualPersistenceHalfLifeCheck,
)

# ── EnsembleIndependenceDeficitCheck ────────────────────────────────────────

def test_eid_independent_members_near_zero():
    rng = np.random.default_rng(0)
    n_models, n_points = 8, 200
    y_true = rng.standard_normal(n_points)
    ensemble = y_true[None, :] + rng.standard_normal((n_models, n_points))  # iid residuals
    result = EnsembleIndependenceDeficitCheck(eid_threshold=0.5).run([], y_true=y_true, y_pred_ensemble=ensemble)
    assert result.passed
    assert result.value < 0.2


def test_eid_shared_bias_pushes_toward_one():
    rng = np.random.default_rng(0)
    n_models, n_points = 8, 200
    y_true = rng.standard_normal(n_points)
    common_bias = rng.standard_normal(n_points) * 3.0  # dominates over small iid noise
    ensemble = y_true[None, :] + common_bias[None, :] + rng.standard_normal((n_models, n_points)) * 0.05
    result = EnsembleIndependenceDeficitCheck(eid_threshold=0.5).run([], y_true=y_true, y_pred_ensemble=ensemble)
    assert not result.passed
    assert result.value > 0.5


def test_eid_skips_without_ensemble():
    result = EnsembleIndependenceDeficitCheck().run([], y_true=np.array([1.0, 2.0]))
    assert result.passed
    assert "Skipped" in result.message


def test_eid_reads_y_true_from_history_ensemble_from_kwarg():
    rng = np.random.default_rng(0)
    n_points = 100
    y_true = rng.standard_normal(n_points)
    ensemble = y_true[None, :] + rng.standard_normal((5, n_points))
    history = [{"y_true": v} for v in y_true]
    result = EnsembleIndependenceDeficitCheck().run(history, y_pred_ensemble=ensemble)
    assert result.value is not None


# ── DMDcSpectralRadiusCheck ──────────────────────────────────────────────────

def test_dmdc_spectral_radius_passes_below_threshold():
    result = DMDcSpectralRadiusCheck(stability_threshold=1.0).run([], rho_A=0.8)
    assert result.passed
    assert result.value == 0.8


def test_dmdc_spectral_radius_fails_above_threshold():
    result = DMDcSpectralRadiusCheck(stability_threshold=1.0).run([], rho_A=1.2)
    assert not result.passed


def test_dmdc_spectral_radius_skips_without_rho_a():
    result = DMDcSpectralRadiusCheck().run([])
    assert result.passed
    assert "Skipped" in result.message


# ── ResidualPersistenceHalfLifeCheck ────────────────────────────────────────

def _ar1(phi, n, seed):
    rng = np.random.default_rng(seed)
    y = np.empty(n)
    y[0] = rng.standard_normal()
    for t in range(1, n):
        y[t] = phi * y[t - 1] + np.sqrt(1 - phi**2) * rng.standard_normal()
    return y


def test_half_life_fast_decay_passes():
    series = _ar1(phi=0.3, n=300, seed=0)
    result = ResidualPersistenceHalfLifeCheck(seed=0).run([], residuals_at_fixed_x=series)
    assert result.passed
    assert result.value < 1.0


def test_half_life_near_unit_root_fails():
    # phi=0.995's true 1/e crossing lag is ~200 steps, far beyond the
    # check's max_lag cap of 40 — so even with a well-estimated ACF (n=500
    # keeps estimation bias/noise small), the crossing is never observed
    # within max_lag. That is correctly reported as unresolved persistence
    # exceeding the campaign, the failure mode this metric exists to catch.
    series = _ar1(phi=0.995, n=500, seed=0)
    result = ResidualPersistenceHalfLifeCheck(seed=0).run([], residuals_at_fixed_x=series)
    assert not result.passed
    assert result.value > 1.0


def test_half_life_skips_short_series():
    result = ResidualPersistenceHalfLifeCheck(block_len=8).run([], residuals_at_fixed_x=np.arange(10.0))
    assert result.passed
    assert "Skipped" in result.message


def test_half_life_reads_from_history():
    series = _ar1(phi=0.3, n=300, seed=1)
    history = [{"residuals_at_fixed_x": v} for v in series]
    result = ResidualPersistenceHalfLifeCheck(seed=0).run(history)
    assert result.value is not None
