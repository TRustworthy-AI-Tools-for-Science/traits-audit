"""
Tests for all built-in AuditChecks.

Data helpers use fixed seeds for reproducibility.  Each check gets:
  - a pass case (well-calibrated / healthy data)
  - a fail case (clearly miscalibrated / pathological data)
  - a skip case (required keys not provided → passed=True, "Skipped" in message)
  - at least one history-route test (data via on_step history dicts, not kwargs)
"""
import numpy as np
import pytest

from traits_audit.checks import (
    CalibrationErrorCheck,
    ConformalCoverageCheck,
    CRPSCheck,
    NegativeLogLikelihoodCheck,
    PITUniformityCheck,
    IntervalScoreCheck,
    IntervalCoverageCheck,
    LyapunovStabilityCheck,
    MahalanobisOODCheck,
    VarianceAlignmentCheck,
    UncertaintyEvolutionCheck,
    UncertaintyAnomalyCheck,
    VarianceErrorCorrelationCheck,
)


# ── shared data builders ─────────────────────────────────────────────────────

def _calibrated(n=400, seed=0):
    """y_true drawn from N(mu, sigma) — model is perfectly calibrated."""
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal(n)
    sigma = np.abs(rng.standard_normal(n)) + 0.5
    y_true = mu + sigma * rng.standard_normal(n)
    return dict(y_true=y_true, y_pred_mean=mu, y_pred_std=sigma)


def _overconfident(n=400, seed=0):
    """sigma is 100x too small; true noise is O(1)."""
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal(n)
    sigma = np.ones(n) * 0.01
    y_true = mu + rng.standard_normal(n)
    return dict(y_true=y_true, y_pred_mean=mu, y_pred_std=sigma)


# ── CalibrationErrorCheck ────────────────────────────────────────────────────

def test_calibration_passes_on_calibrated_data():
    check = CalibrationErrorCheck(threshold=0.1)
    result = check.run([], **_calibrated())
    assert result.passed
    assert result.value < 0.1


def test_calibration_fails_on_overconfident_data():
    check = CalibrationErrorCheck(threshold=0.1)
    result = check.run([], **_overconfident())
    assert not result.passed
    assert result.value > 0.1


def test_calibration_skips_when_no_data():
    result = CalibrationErrorCheck().run([])
    assert result.passed
    assert "Skipped" in result.message


def test_calibration_details_contains_levels_and_fractions():
    result = CalibrationErrorCheck().run([], **_calibrated())
    assert "confidence_levels" in result.details
    assert "observed_fractions" in result.details
    assert len(result.details["confidence_levels"]) == len(result.details["observed_fractions"])


def test_calibration_reads_from_history():
    rng = np.random.default_rng(1)
    n = 300
    mu = rng.standard_normal(n)
    sigma = np.abs(rng.standard_normal(n)) + 0.5
    y_true = mu + sigma * rng.standard_normal(n)
    history = [
        {"y_true": float(y_true[i]), "y_pred_mean": float(mu[i]), "y_pred_std": float(sigma[i])}
        for i in range(n)
    ]
    result = CalibrationErrorCheck(threshold=0.1).run(history)
    assert result.passed


# ── IntervalCoverageCheck ────────────────────────────────────────────────────

def test_coverage_passes_on_calibrated_data():
    result = IntervalCoverageCheck(tolerance=0.1).run([], **_calibrated())
    assert result.passed
    assert abs(result.value - 0.683) <= 0.1


def test_coverage_fails_on_overconfident_data():
    result = IntervalCoverageCheck(tolerance=0.1).run([], **_overconfident())
    assert not result.passed
    assert result.value < 0.683 - 0.1


def test_coverage_skips_when_no_data():
    result = IntervalCoverageCheck().run([])
    assert result.passed
    assert "Skipped" in result.message


def test_coverage_reads_from_history():
    rng = np.random.default_rng(2)
    n = 300
    mu = rng.standard_normal(n)
    sigma = np.abs(rng.standard_normal(n)) + 0.5
    y_true = mu + sigma * rng.standard_normal(n)
    history = [
        {"y_true": float(y_true[i]), "y_pred_mean": float(mu[i]), "y_pred_std": float(sigma[i])}
        for i in range(n)
    ]
    result = IntervalCoverageCheck(tolerance=0.1).run(history)
    assert result.passed


# ── VarianceAlignmentCheck ───────────────────────────────────────────────────

def test_variance_alignment_passes_on_calibrated_data():
    result = VarianceAlignmentCheck(tolerance=0.5).run([], **_calibrated())
    assert result.passed
    assert abs(result.value - 1.0) <= 0.5


def test_variance_alignment_fails_on_overconfident_data():
    result = VarianceAlignmentCheck(tolerance=0.5).run([], **_overconfident())
    assert not result.passed
    # Predicted variance (0.01²) << empirical variance (~1) → ratio << 1
    assert result.value < 0.5


def test_variance_alignment_skips_when_no_data():
    result = VarianceAlignmentCheck().run([])
    assert result.passed
    assert "Skipped" in result.message


# ── UncertaintyEvolutionCheck (multi-channel: value = #decreasing channels) ───

def test_evolution_passes_on_gentle_decline():
    u = np.linspace(1.0, 0.5, 100)  # slope ≈ −0.005 > −0.01·mean → not flagged
    result = UncertaintyEvolutionCheck().run([], uncertainties=u)
    assert result.passed
    assert result.value == 0.0


def test_evolution_fails_on_collapse():
    u = np.linspace(1.0, 0.01, 10)  # steep: slope ≈ −0.11 ≪ −0.01·mean → flagged
    result = UncertaintyEvolutionCheck().run([], uncertainties=u)
    assert not result.passed
    assert result.value == 1.0  # one channel flagged


def test_evolution_multichannel_flags_only_declining():
    steps = 30
    u = np.column_stack([np.full(steps, 0.8), np.linspace(1.0, 0.05, steps)])
    result = UncertaintyEvolutionCheck().run([], uncertainties=u)
    assert not result.passed
    assert result.value == 1.0
    assert result.details["n_channels"] == 2
    assert len(result.details["uncertainty_series"]) == steps


def test_evolution_respects_slope_threshold():
    u = np.linspace(1.0, 0.5, 100)  # ~−0.7 %/step relative
    assert UncertaintyEvolutionCheck(slope_threshold=-0.005).run([], uncertainties=u).passed is False
    assert UncertaintyEvolutionCheck(slope_threshold=-0.05).run([], uncertainties=u).passed is True


def test_evolution_passes_on_increasing_uncertainty():
    u = np.linspace(0.1, 1.0, 30)
    assert UncertaintyEvolutionCheck().run([], uncertainties=u).passed


def test_evolution_single_step_not_flagged():
    result = UncertaintyEvolutionCheck().run([], uncertainties=[0.5])
    assert result.passed
    assert result.value == 0.0


def test_evolution_skips_when_no_data():
    result = UncertaintyEvolutionCheck().run([])
    assert result.passed
    assert "Skipped" in result.message


def test_evolution_reads_from_history():
    u = np.linspace(1.0, 0.6, 60)  # slope ≈ −0.0068 > −0.01·mean → not flagged
    history = [{"uncertainty": float(v)} for v in u]
    assert UncertaintyEvolutionCheck().run(history).passed


# ── UncertaintyAnomalyCheck (z-score vs HISTORICAL baseline) ──────────────────

def test_anomaly_passes_when_current_matches_baseline():
    rng = np.random.default_rng(0)
    hist = 0.5 + 0.05 * rng.standard_normal(50)
    cur  = 0.5 + 0.05 * rng.standard_normal(20)
    result = UncertaintyAnomalyCheck().run(
        [], uncertainties=cur, historical_uncertainties=hist
    )
    assert result.passed
    assert result.value < 0.05


def test_anomaly_fails_with_spikes_vs_baseline():
    rng = np.random.default_rng(1)
    hist = 1.0 + 0.05 * rng.standard_normal(50)
    cur = np.concatenate([1.0 + 0.05 * rng.standard_normal(18), [100.0, 100.0]])
    result = UncertaintyAnomalyCheck(z_threshold=3.0, max_anomaly_fraction=0.05).run(
        [], uncertainties=cur, historical_uncertainties=hist
    )
    assert not result.passed
    assert result.value > 0.05


def test_anomaly_falls_back_to_within_series_without_baseline():
    # When no historical_uncertainties provided, check should fall back to
    # within-series z-scoring and return a real value rather than skipping.
    result = UncertaintyAnomalyCheck().run([], uncertainties=[1.0, 2.0, 3.0])
    assert result.passed
    assert result.value is not None


def test_anomaly_skips_when_no_data():
    result = UncertaintyAnomalyCheck().run([])
    assert result.passed
    assert "Skipped" in result.message


def test_anomaly_details_keys():
    rng = np.random.default_rng(2)
    hist = 0.5 + 0.05 * rng.standard_normal(40)
    cur = 0.5 + 0.05 * rng.standard_normal(20)
    result = UncertaintyAnomalyCheck().run(
        [], uncertainties=cur, historical_uncertainties=hist
    )
    for key in ("anomalous_fraction", "max_z_score", "hist_mean",
                "hist_std", "current_mean", "current_std"):
        assert key in result.details


def test_anomaly_reads_current_from_history():
    rng = np.random.default_rng(3)
    hist = 0.5 + 0.05 * rng.standard_normal(30)
    u = 0.5 + 0.05 * rng.standard_normal(20)
    history = [{"uncertainty": float(v)} for v in u]
    assert UncertaintyAnomalyCheck().run(history, historical_uncertainties=hist).passed


# ── VarianceErrorCorrelationCheck ────────────────────────────────────────────

def test_variance_error_correlation_passes_on_perfectly_correlated():
    # errors proportional to sigma → Spearman ρ = 1.0
    n = 50
    mu = np.zeros(n)
    sigma = np.linspace(0.1, 1.0, n)
    y_true = mu + sigma * 2          # errors = 2*sigma (perfectly rank-correlated)
    result = VarianceErrorCorrelationCheck(min_correlation=0.0).run(
        [], y_true=y_true, y_pred_mean=mu, y_pred_std=sigma
    )
    assert result.passed
    assert result.value == pytest.approx(1.0, abs=1e-6)


def test_variance_error_correlation_fails_on_perfectly_anticorrelated():
    # errors inversely proportional to sigma → Spearman ρ = -1.0
    n = 50
    mu = np.zeros(n)
    sigma = np.linspace(0.1, 1.0, n)       # ascending
    y_true = mu + sigma[::-1]              # errors = reversed sigma (descending)
    result = VarianceErrorCorrelationCheck(min_correlation=0.0).run(
        [], y_true=y_true, y_pred_mean=mu, y_pred_std=sigma
    )
    assert not result.passed
    assert result.value == pytest.approx(-1.0, abs=1e-6)


def test_variance_error_correlation_skips_when_no_data():
    result = VarianceErrorCorrelationCheck().run([])
    assert result.passed
    assert "Skipped" in result.message


def test_variance_error_correlation_details_contains_pvalue():
    result = VarianceErrorCorrelationCheck().run([], **_calibrated(n=100))
    assert "p_value" in result.details
    assert 0.0 <= result.details["p_value"] <= 1.0


def test_variance_error_correlation_reads_from_history():
    n = 50
    mu = np.zeros(n)
    sigma = np.linspace(0.1, 1.0, n)
    y_true = mu + sigma * 2
    history = [
        {"y_true": float(y_true[i]), "y_pred_mean": float(mu[i]), "y_pred_std": float(sigma[i])}
        for i in range(n)
    ]
    result = VarianceErrorCorrelationCheck(min_correlation=0.0).run(history)
    assert result.passed


# ── ConformalCoverageCheck ───────────────────────────────────────────────────

def test_conformal_passes_on_calibrated_data():
    # Well-calibrated data: y_true ~ N(mu, sigma) → q̂ ≈ z_{1-α/2} → q_ratio ≈ 1
    result = ConformalCoverageCheck(target_coverage=0.9, max_q_ratio=1.5).run([], **_calibrated(n=500))
    assert result.passed
    assert result.value <= 1.5


def test_conformal_fails_on_overconfident_data():
    # sigma is 100× too small → normalised residuals ≫ 1 → q̂ ≫ z → q_ratio ≫ 1
    result = ConformalCoverageCheck(target_coverage=0.9, max_q_ratio=1.5).run([], **_overconfident(n=500))
    assert not result.passed
    assert result.value > 1.5


def test_conformal_skips_when_no_data():
    result = ConformalCoverageCheck().run([])
    assert result.passed
    assert "Skipped" in result.message


def test_conformal_skips_too_few_samples():
    rng = np.random.default_rng(0)
    mu = rng.standard_normal(5)
    sigma = np.ones(5) * 0.5
    y_true = mu + sigma * rng.standard_normal(5)
    result = ConformalCoverageCheck().run([], y_true=y_true, y_pred_mean=mu, y_pred_std=sigma)
    assert result.passed
    assert "Too few" in result.message


def test_conformal_details_keys():
    result = ConformalCoverageCheck().run([], **_calibrated(n=200))
    for key in ("q_hat", "z_expected", "q_ratio", "empirical_coverage", "n_samples", "alpha"):
        assert key in result.details
    assert result.details["n_samples"] == 200


def test_conformal_reads_from_history():
    rng = np.random.default_rng(2)
    n = 300
    mu = rng.standard_normal(n)
    sigma = np.abs(rng.standard_normal(n)) + 0.5
    y_true = mu + sigma * rng.standard_normal(n)
    history = [
        {"y_true": float(y_true[i]), "y_pred_mean": float(mu[i]), "y_pred_std": float(sigma[i])}
        for i in range(n)
    ]
    result = ConformalCoverageCheck(target_coverage=0.9, max_q_ratio=1.5).run(history)
    assert result.passed


# ── LyapunovStabilityCheck ───────────────────────────────────────────────────

def test_lyapunov_passes_on_stable_precomputed():
    # All |λ_max| < 1 → fraction_stable = 1.0
    lm = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    result = LyapunovStabilityCheck(min_stable_fraction=0.5).run([], lambda_max=lm)
    assert result.passed
    assert result.value == pytest.approx(1.0)


def test_lyapunov_fails_on_unstable_precomputed():
    # All |λ_max| > 1 → fraction_stable = 0.0
    lm = np.array([2.0, 5.0, 10.0, 45.0, 100.0])
    result = LyapunovStabilityCheck(min_stable_fraction=0.5).run([], lambda_max=lm)
    assert not result.passed
    assert result.value == pytest.approx(0.0)


def test_lyapunov_mixed_fraction():
    # 3 stable / 5 total = 0.6 → PASS with threshold 0.5
    lm = np.array([0.2, 0.8, 0.99, 1.5, 3.0])
    result = LyapunovStabilityCheck(min_stable_fraction=0.5).run([], lambda_max=lm)
    assert result.passed
    assert result.value == pytest.approx(0.6)


def test_lyapunov_custom_stability_threshold():
    # With threshold=10, values [2, 5] are "stable"
    lm = np.array([2.0, 5.0, 15.0])
    result = LyapunovStabilityCheck(stability_threshold=10.0, min_stable_fraction=0.5).run(
        [], lambda_max=lm
    )
    assert result.passed
    assert result.value == pytest.approx(2 / 3)


def test_lyapunov_reads_from_history():
    lm_vals = [0.2, 0.4, 0.6, 0.8]
    history = [{"lambda_max": v} for v in lm_vals]
    result = LyapunovStabilityCheck(min_stable_fraction=0.5).run(history)
    assert result.passed
    assert result.value == pytest.approx(1.0)


def test_lyapunov_skips_when_no_data():
    result = LyapunovStabilityCheck().run([])
    assert result.passed
    assert "Skipped" in result.message


def test_lyapunov_skips_on_empty_array():
    result = LyapunovStabilityCheck().run([], lambda_max=np.array([]))
    assert result.passed
    assert "Skipped" in result.message


def test_lyapunov_details_keys():
    lm = np.array([0.5, 1.5, 3.0])
    result = LyapunovStabilityCheck().run([], lambda_max=lm)
    for key in ("lambda_max_mean", "lambda_max_max", "lambda_max_min", "n_stable", "n_total"):
        assert key in result.details
    assert result.details["n_total"] == 3
    assert result.details["n_stable"] == 1


def test_lyapunov_drops_nan_prefix_instead_of_counting_unstable():
    # A NaN-padded warm-up prefix (e.g. from a growing-window DMDc fit) must
    # not be counted as "unstable" — it should be excluded entirely, so the
    # verdict reflects only the valid entries.
    lm = np.array([np.nan, np.nan, 0.3, 0.4, 0.5])
    result = LyapunovStabilityCheck(min_stable_fraction=0.5).run([], lambda_max=lm)
    assert result.passed
    assert result.details["n_total"] == 3
    assert result.details["n_stable"] == 3


def test_lyapunov_from_surrogate_callable():
    # Flat quadratic bowl: f(x) = sum(x**2). Hessian = 2I, so GD Jacobian = I - 2α*I.
    # With alpha=0.01: J = (1 - 0.02)*I → |λ| = 0.98 < 1 → all stable.
    def f(x):
        return float(np.sum(x**2))

    rng = np.random.default_rng(42)
    op_states = rng.standard_normal((5, 3))
    result = LyapunovStabilityCheck(alpha=0.01, min_stable_fraction=0.5).run(
        [], surrogate_fn=f, op_states=op_states
    )
    assert result.passed
    assert result.value > 0.5


def test_lyapunov_callable_unstable():
    # f(x) = -sum(x**2): inverted bowl. Hessian = -2I, J = I + 2α*I → |λ| > 1.
    def f(x):
        return float(-np.sum(x**2))

    rng = np.random.default_rng(0)
    op_states = rng.standard_normal((5, 2)) * 0.1  # small perturbations
    result = LyapunovStabilityCheck(alpha=0.01, min_stable_fraction=0.5).run(
        [], surrogate_fn=f, op_states=op_states
    )
    assert not result.passed


# ── MahalanobisOODCheck ──────────────────────────────────────────────────────

def _mahalanobis_data(seed, n_id=40, d=2, window=10, shift=20.0):
    """(n_id + window, d) array: an in-distribution cluster followed by a shifted
    OOD tail occupying the whole trailing window."""
    rng = np.random.default_rng(seed)
    id_cluster = rng.standard_normal((n_id, d))
    tail = rng.standard_normal((window, d)) * 0.5 + shift
    return np.vstack([id_cluster, tail]), n_id, window


def test_mahalanobis_skips_when_op_states_absent():
    result = MahalanobisOODCheck().run([])
    assert result.passed
    assert result.value is None
    assert "Skipped" in result.message


def test_mahalanobis_skips_below_min_history():
    op_states = np.random.default_rng(0).standard_normal((10, 2))
    result = MahalanobisOODCheck(min_history=20).run([], op_states=op_states)
    assert result.passed
    assert result.value is None
    assert "Skipped" in result.message


def test_mahalanobis_passes_on_pure_in_distribution_data():
    op_states = np.random.default_rng(0).standard_normal((30, 2))
    result = MahalanobisOODCheck(min_history=20, window=10, n_bootstrap=50, random_state=0).run(
        [], op_states=op_states, uncertainties=np.full(30, 0.1)
    )
    assert result.passed
    assert result.value == pytest.approx(0.0)


def test_mahalanobis_passes_when_ood_exploration_has_healthy_uncertainty():
    op_states, n_id, window = _mahalanobis_data(seed=42)
    uncertainties = np.concatenate([np.full(n_id, 0.1), np.full(window, 1.0)])
    result = MahalanobisOODCheck(min_history=20, window=window, n_bootstrap=50, random_state=0).run(
        [], op_states=op_states, uncertainties=uncertainties
    )
    assert result.passed
    assert result.value == pytest.approx(1.0)
    assert result.details["suppression_assessed"] is True


def test_mahalanobis_fails_when_ood_uncertainty_is_suppressed():
    op_states, n_id, window = _mahalanobis_data(seed=42)
    uncertainties = np.concatenate([np.full(n_id, 1.0), np.full(window, 0.05)])
    result = MahalanobisOODCheck(min_history=20, window=window, n_bootstrap=50, random_state=0).run(
        [], op_states=op_states, uncertainties=uncertainties
    )
    assert not result.passed
    assert result.value == pytest.approx(1.0)
    assert "suppression" in result.message


def test_mahalanobis_degraded_pass_without_uncertainty_data():
    op_states, _, _ = _mahalanobis_data(seed=42)
    result = MahalanobisOODCheck(min_history=20, window=10, n_bootstrap=50, random_state=0).run(
        [], op_states=op_states
    )
    assert result.passed
    assert result.value == pytest.approx(1.0)
    assert result.details["suppression_assessed"] is False


def test_mahalanobis_uses_in_window_baseline_when_enough_id_points_in_window():
    rng = np.random.default_rng(7)
    n_id, d, window = 40, 2, 10
    id_cluster = rng.standard_normal((n_id, d))
    ood_part = rng.standard_normal((6, d)) * 0.5 + 20.0
    id_part = rng.standard_normal((4, d))
    op_states = np.vstack([id_cluster, ood_part, id_part])
    uncertainties = np.concatenate([np.full(n_id, 0.1), np.full(6, 1.0), np.full(4, 0.1)])

    result = MahalanobisOODCheck(min_history=20, window=window, n_bootstrap=50, random_state=1).run(
        [], op_states=op_states, uncertainties=uncertainties
    )
    assert result.passed
    assert result.details["id_baseline_mode"] == "window"


def test_mahalanobis_details_keys():
    op_states, n_id, window = _mahalanobis_data(seed=42)
    uncertainties = np.concatenate([np.full(n_id, 0.1), np.full(window, 1.0)])
    result = MahalanobisOODCheck(min_history=20, window=window, n_bootstrap=50, random_state=0).run(
        [], op_states=op_states, uncertainties=uncertainties
    )
    for key in (
        "mahalanobis_series", "threshold", "is_ood", "ood_fraction", "window_effective",
        "n_reference", "n_bootstrap", "shrinkage_", "n_ood_total", "suppression_assessed",
        "id_baseline_mode", "id_baseline_mean", "ood_window_mean",
    ):
        assert key in result.details
    assert len(result.details["mahalanobis_series"]) == op_states.shape[0]


def test_mahalanobis_reads_uncertainty_from_history():
    op_states, n_id, window = _mahalanobis_data(seed=42)
    u = np.concatenate([np.full(n_id, 1.0), np.full(window, 0.05)])
    history = [{"uncertainty": float(v)} for v in u]
    result = MahalanobisOODCheck(min_history=20, window=window, n_bootstrap=50, random_state=0).run(
        history, op_states=op_states
    )
    assert not result.passed


# ── CRPSCheck ────────────────────────────────────────────────────────────────

def test_crps_passes_no_threshold_calibrated():
    # No threshold → always passes; calibrated data should give low CRPS.
    result = CRPSCheck().run([], **_calibrated(n=500))
    assert result.passed
    assert result.value is not None
    assert result.value > 0


def test_crps_calibrated_lower_than_overconfident():
    # Calibrated CRPS should be lower than overconfident CRPS (sigma 100x too small).
    cal = CRPSCheck().run([], **_calibrated(n=500))
    over = CRPSCheck().run([], **_overconfident(n=500))
    assert cal.value < over.value


def test_crps_fails_when_threshold_exceeded():
    # Explicit low threshold; overconfident data (CRPS ≈ MAE >> 0) should fail.
    result = CRPSCheck(threshold=0.3).run([], **_overconfident(n=500))
    assert not result.passed
    assert result.value > 0.3


def test_crps_passes_when_threshold_met():
    result = CRPSCheck(threshold=10.0).run([], **_calibrated(n=500))
    assert result.passed


def test_crps_skips_when_no_data():
    result = CRPSCheck().run([])
    assert result.passed
    assert "Skipped" in result.message


def test_crps_skips_too_few_samples():
    rng = np.random.default_rng(0)
    mu = rng.standard_normal(4)
    result = CRPSCheck().run([], y_true=mu + 0.1, y_pred_mean=mu, y_pred_std=np.ones(4))
    assert result.passed
    assert "Too few" in result.message


def test_crps_details_keys():
    result = CRPSCheck().run([], **_calibrated(n=200))
    for key in ("mean_crps", "crps_reference", "n_samples"):
        assert key in result.details
    assert result.details["n_samples"] == 200
    assert result.details["crps_reference"] > 0


def test_crps_reads_from_history():
    rng = np.random.default_rng(3)
    n = 300
    mu = rng.standard_normal(n)
    sigma = np.abs(rng.standard_normal(n)) + 0.5
    y_true = mu + sigma * rng.standard_normal(n)
    history = [
        {"y_true": float(y_true[i]), "y_pred_mean": float(mu[i]), "y_pred_std": float(sigma[i])}
        for i in range(n)
    ]
    result = CRPSCheck().run(history)
    assert result.passed
    assert result.value > 0


# ── NegativeLogLikelihoodCheck ───────────────────────────────────────────────

def test_nll_passes_no_threshold_calibrated():
    result = NegativeLogLikelihoodCheck().run([], **_calibrated(n=500))
    assert result.passed
    assert result.value is not None


def test_nll_calibrated_lower_than_overconfident():
    cal = NegativeLogLikelihoodCheck().run([], **_calibrated(n=500))
    over = NegativeLogLikelihoodCheck().run([], **_overconfident(n=500))
    assert cal.value < over.value


def test_nll_fails_when_threshold_exceeded():
    result = NegativeLogLikelihoodCheck(threshold=2.0).run([], **_overconfident(n=500))
    assert not result.passed
    assert result.value > 2.0


def test_nll_passes_when_threshold_met():
    result = NegativeLogLikelihoodCheck(threshold=100.0).run([], **_calibrated(n=500))
    assert result.passed


def test_nll_skips_when_no_data():
    result = NegativeLogLikelihoodCheck().run([])
    assert result.passed
    assert "Skipped" in result.message


def test_nll_skips_too_few_samples():
    rng = np.random.default_rng(0)
    mu = rng.standard_normal(4)
    result = NegativeLogLikelihoodCheck().run([], y_true=mu + 0.1, y_pred_mean=mu, y_pred_std=np.ones(4))
    assert result.passed
    assert "Too few" in result.message


def test_nll_details_keys():
    result = NegativeLogLikelihoodCheck().run([], **_calibrated(n=200))
    for key in ("mean_nll", "nll_reference", "n_samples"):
        assert key in result.details
    assert result.details["n_samples"] == 200


def test_nll_reads_from_history():
    rng = np.random.default_rng(4)
    n = 300
    mu = rng.standard_normal(n)
    sigma = np.abs(rng.standard_normal(n)) + 0.5
    y_true = mu + sigma * rng.standard_normal(n)
    history = [
        {"y_true": float(y_true[i]), "y_pred_mean": float(mu[i]), "y_pred_std": float(sigma[i])}
        for i in range(n)
    ]
    result = NegativeLogLikelihoodCheck().run(history)
    assert result.passed
    assert result.value is not None


# ── PITUniformityCheck ───────────────────────────────────────────────────────

def test_pit_passes_on_calibrated_data():
    # Large n gives enough power; calibrated PIT should be uniform.
    result = PITUniformityCheck(alpha=0.05).run([], **_calibrated(n=500))
    assert result.passed
    assert result.value >= 0.05


def test_pit_fails_on_overconfident_data():
    # sigma 100x too small → PIT values cluster near 0 and 1 → non-uniform → FAIL.
    result = PITUniformityCheck(alpha=0.05).run([], **_overconfident(n=500))
    assert not result.passed
    assert result.value < 0.05


def test_pit_skips_when_no_data():
    result = PITUniformityCheck().run([])
    assert result.passed
    assert "Skipped" in result.message


def test_pit_skips_too_few_samples():
    rng = np.random.default_rng(0)
    mu = rng.standard_normal(15)
    result = PITUniformityCheck().run([], y_true=mu + 0.1, y_pred_mean=mu, y_pred_std=np.ones(15))
    assert result.passed
    assert "Too few" in result.message


def test_pit_details_keys():
    result = PITUniformityCheck().run([], **_calibrated(n=200))
    for key in ("ks_statistic", "p_value", "n_samples", "alpha"):
        assert key in result.details
    assert result.details["n_samples"] == 200
    assert 0.0 <= result.details["ks_statistic"] <= 1.0
    assert 0.0 <= result.details["p_value"] <= 1.0


def test_pit_reads_from_history():
    rng = np.random.default_rng(5)
    n = 300
    mu = rng.standard_normal(n)
    sigma = np.abs(rng.standard_normal(n)) + 0.5
    y_true = mu + sigma * rng.standard_normal(n)
    history = [
        {"y_true": float(y_true[i]), "y_pred_mean": float(mu[i]), "y_pred_std": float(sigma[i])}
        for i in range(n)
    ]
    result = PITUniformityCheck(alpha=0.05).run(history)
    assert result.passed


# ── IntervalScoreCheck ────────────────────────────────────────────────────────

def test_interval_score_passes_no_threshold_calibrated():
    result = IntervalScoreCheck().run([], **_calibrated(n=500))
    assert result.passed
    assert result.value is not None
    assert result.value > 0


def test_interval_score_calibrated_lower_than_overconfident():
    # Overconfident model incurs heavy coverage penalty (2/alpha * violations).
    cal = IntervalScoreCheck().run([], **_calibrated(n=500))
    over = IntervalScoreCheck().run([], **_overconfident(n=500))
    assert cal.value < over.value


def test_interval_score_fails_when_threshold_exceeded():
    result = IntervalScoreCheck(threshold=1.0).run([], **_overconfident(n=500))
    assert not result.passed
    assert result.value > 1.0


def test_interval_score_passes_when_threshold_met():
    result = IntervalScoreCheck(threshold=1000.0).run([], **_calibrated(n=500))
    assert result.passed


def test_interval_score_skips_when_no_data():
    result = IntervalScoreCheck().run([])
    assert result.passed
    assert "Skipped" in result.message


def test_interval_score_skips_too_few_samples():
    rng = np.random.default_rng(0)
    mu = rng.standard_normal(4)
    result = IntervalScoreCheck().run([], y_true=mu + 0.1, y_pred_mean=mu, y_pred_std=np.ones(4))
    assert result.passed
    assert "Too few" in result.message


def test_interval_score_details_keys():
    result = IntervalScoreCheck().run([], **_calibrated(n=200))
    for key in ("mean_is", "is_reference", "alpha", "z_critical", "n_samples"):
        assert key in result.details
    assert result.details["n_samples"] == 200
    assert result.details["alpha"] == pytest.approx(0.1)


def test_interval_score_reads_from_history():
    rng = np.random.default_rng(6)
    n = 300
    mu = rng.standard_normal(n)
    sigma = np.abs(rng.standard_normal(n)) + 0.5
    y_true = mu + sigma * rng.standard_normal(n)
    history = [
        {"y_true": float(y_true[i]), "y_pred_mean": float(mu[i]), "y_pred_std": float(sigma[i])}
        for i in range(n)
    ]
    result = IntervalScoreCheck().run(history)
    assert result.passed
    assert result.value > 0
