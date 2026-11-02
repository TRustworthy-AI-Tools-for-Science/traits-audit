"""Tests for traits_audit.stability.lle / regime (ported from battery-forecast exp4)."""
import numpy as np
import pytest

from traits_audit.stability.detrend import RegimeDetrender
from traits_audit.stability.lle import (
    classify_lle_regime,
    compute_lyapunov_exponent,
    rolling_lyapunov_exponent,
    rosenstein_divergence,
)
from traits_audit.stability.regime import extract_background_trend, split_regime_labels


class TestSplitRegimeLabels:
    def test_midpoint_split_default(self):
        labels = split_regime_labels(10, split_fraction=0.5)
        assert list(labels) == [0] * 5 + [1] * 5

    def test_configurable_split_fraction(self):
        labels = split_regime_labels(10, split_fraction=0.3)
        assert labels.sum() == 7  # 30% early (3), 70% late (7)

    def test_both_regimes_nonempty_at_extreme_fractions(self):
        labels = split_regime_labels(4, split_fraction=0.99)
        assert 0 in labels and 1 in labels

    def test_raises_on_too_short_input(self):
        with pytest.raises(ValueError):
            split_regime_labels(1)


class TestComputeLyapunovExponent:
    def test_known_shape_on_trivial_input(self):
        """Drift-guard: calls the REAL nolitsa.lyapunov.mle on a trivial
        synthetic input and asserts the returned divergence_curve is
        non-empty and that the slope-fit machinery runs without error and
        returns a finite float. If a future nolitsa version changes mle's
        signature or return shape, this fails loudly."""
        rng = np.random.default_rng(0)
        aug = rng.standard_normal((60, 4))
        result = compute_lyapunov_exponent(aug, maxt=15, window=5)
        assert np.isfinite(result["lle"])
        assert len(result["divergence_curve"]) > 0
        assert result["fit_range"][1] > result["fit_range"][0]

    def test_positive_lle_on_an_expansive_synthetic_system(self):
        """Regression sanity: a mildly expansive linear system (|eigenvalue|
        > 1, so nearby trajectories genuinely diverge) should give a
        clearly positive LLE point estimate -- confirms the fit_range
        slope actually picks up real divergence, not just noise."""
        rng = np.random.default_rng(3)
        T = 80
        z = np.zeros((T, 3))
        z[0] = rng.standard_normal(3) * 0.01
        A = np.array([[1.15, 0.0, 0.0], [0.0, 0.9, 0.05], [0.0, -0.05, 0.9]])
        for t in range(1, T):
            z[t] = A @ z[t - 1] + rng.standard_normal(3) * 1e-4
            # keep it from overflowing while still genuinely expanding
            if np.linalg.norm(z[t]) > 10:
                z[t] *= 0.1
        result = compute_lyapunov_exponent(z, maxt=20, window=5)
        assert result["lle"] > 0.02

    def test_maxt_clamped_for_short_trajectories(self):
        """maxt_eff must never make nolitsa.lyapunov.mle raise on a short
        (~15-step) regime-half slice -- confirms the maxt/window clamp
        logic actually protects against the short-trajectory case this
        script hits on real regime-split data."""
        aug = np.random.default_rng(1).standard_normal((15, 4))
        result = compute_lyapunov_exponent(aug, maxt=30, window=10)
        assert "lle" in result  # must not raise


class TestRollingLyapunovExponent:
    def test_t_indices_match_window_stride_schedule(self):
        """Confirms the window-end index schedule is exactly
        range(window, T+1, stride) -- the same t-index convention as
        stability_convergence_windowed_pydmd's curves, so the two can be
        plotted against each other directly."""
        rng = np.random.default_rng(0)
        T, window, stride = 100, 40, 10
        aug = rng.standard_normal((T, 3))
        t_indices, lle_values = rolling_lyapunov_exponent(
            aug, window=window, stride=stride, maxt=15, lle_window=5
        )
        assert list(t_indices) == list(range(window, T + 1, stride))
        assert len(lle_values) == len(t_indices)

    def test_empty_when_trajectory_shorter_than_window(self):
        """A trajectory shorter than the rolling window must degrade
        gracefully to empty arrays, not raise -- this is the realistic
        case for a short regime-half slice or a short test fixture."""
        aug = np.random.default_rng(2).standard_normal((20, 3))
        t_indices, lle_values = rolling_lyapunov_exponent(
            aug, window=40, stride=10, maxt=15, lle_window=5
        )
        assert len(t_indices) == 0
        assert len(lle_values) == 0

    def test_lle_decreases_when_dynamics_transition_from_expansive_to_contractive(self):
        """Regression sanity, directly testing the 'settles onto an
        attractor' hypothesis this function exists to probe: build a
        trajectory that is genuinely expansive (diverging) in its first
        half and genuinely contractive (converging to a fixed point) in
        its second half, and confirm the rolling LLE is higher in windows
        drawn from the expansive half than in windows drawn from the
        contractive half."""
        rng = np.random.default_rng(5)
        T = 160
        z = np.zeros((T, 2))
        z[0] = rng.standard_normal(2) * 0.01
        for t in range(1, T):
            if t < T // 2:
                A = np.array([[1.12, 0.0], [0.0, 1.08]])  # expansive
            else:
                A = np.array([[0.85, 0.0], [0.0, 0.80]])  # contractive
            z[t] = A @ z[t - 1] + rng.standard_normal(2) * 1e-4
            if np.linalg.norm(z[t]) > 5:
                z[t] *= 0.1
        t_indices, lle_values = rolling_lyapunov_exponent(
            z, window=40, stride=20, maxt=15, lle_window=5
        )
        early_mask = t_indices <= T // 2
        late_mask = t_indices > T // 2
        assert early_mask.any() and late_mask.any()
        assert np.nanmean(lle_values[early_mask]) > np.nanmean(lle_values[late_mask])


class TestExtractBackgroundTrend:
    def test_trend_plus_residual_reconstructs_raw_state_exactly(self):
        """Regression guard: trend + residual must equal raw_state exactly
        (a plain subtraction identity, not an approximation) -- confirms
        extract_background_trend doesn't silently lose or duplicate signal."""
        rng = np.random.default_rng(0)
        raw_state = rng.standard_normal((30, 4)) + np.linspace(0, 5, 30)[:, None]
        trend, residual = extract_background_trend(raw_state)
        assert np.allclose(trend + residual, raw_state)

    def test_output_shape_matches_input(self):
        raw_state = np.random.default_rng(1).standard_normal((25, 6))
        trend, residual = extract_background_trend(raw_state)
        assert trend.shape == raw_state.shape
        assert residual.shape == raw_state.shape

    def test_trend_matches_independently_constructed_regime_detrender_baseline(self):
        """Drift-guard against a future _detrend_trajectory semantics
        change: trend[i] must equal an INDEPENDENTLY built
        RegimeDetrender().update(...) baseline (state - detrended) at
        hand-picked steps, using regimes=None's documented all-zero-vector
        semantics directly (not reaching into _detrend_trajectory again)."""
        rng = np.random.default_rng(2)
        raw_state = rng.standard_normal((15, 3)) + np.linspace(0, 3, 15)[:, None]
        trend, _ = extract_background_trend(raw_state)

        detrender = RegimeDetrender()
        expected_baseline = np.empty_like(raw_state)
        zero_regime = np.zeros((1,))
        for i in range(len(raw_state)):
            result = detrender.update(raw_state[i], zero_regime)
            expected_baseline[i] = raw_state[i] - result.detrended

        assert np.allclose(trend, expected_baseline)


class TestClassifyLleRegime:
    def test_collapsing_on_credibly_negative_values(self):
        rng = np.random.default_rng(0)
        values = -0.5 + rng.standard_normal(30) * 0.05
        label, median, p = classify_lle_regime(values)
        assert label == "COLLAPSING"
        assert median < 0
        assert p < 0.05

    def test_chaotic_on_credibly_positive_values(self):
        rng = np.random.default_rng(1)
        values = 0.5 + rng.standard_normal(30) * 0.05
        label, median, p = classify_lle_regime(values)
        assert label == "CHAOTIC"
        assert median > 0
        assert p < 0.05

    def test_periodic_like_on_noisy_near_zero_values(self):
        rng = np.random.default_rng(2)
        values = rng.standard_normal(30) * 0.5  # centered at 0, noisy
        label, median, p = classify_lle_regime(values)
        assert label == "PERIODIC-LIKE"
        assert p >= 0.05

    def test_insufficient_data_below_three_values(self):
        assert classify_lle_regime(np.array([0.1, 0.2]))[0] == "INSUFFICIENT_DATA"
        assert classify_lle_regime(np.array([]))[0] == "INSUFFICIENT_DATA"

    def test_insufficient_data_ignores_non_finite_values(self):
        """Only finite values count toward the >=3 threshold."""
        values = np.array([0.1, np.nan, np.inf, -np.inf])
        assert classify_lle_regime(values)[0] == "INSUFFICIENT_DATA"

    def test_adaptive_direction_reproduces_h6a_common_case(self):
        """The common case seen in this project's real data so far (a
        positive pooled median LLE) must reproduce what a hardcoded
        alternative="greater" Wilcoxon test would have given -- confirming
        the adaptive-direction fix is backward-compatible in that case."""
        from scipy.stats import wilcoxon
        rng = np.random.default_rng(3)
        values = 0.3 + rng.standard_normal(20) * 0.1
        label, median, p = classify_lle_regime(values)
        _, p_hardcoded = wilcoxon(values, alternative="greater")
        assert label == "CHAOTIC"
        assert p == pytest.approx(p_hardcoded)



class TestRosensteinDivergence:
    """The in-package Rosenstein implementation must reproduce
    ``nolitsa.lyapunov.mle`` (the backend this replaced) exactly."""

    @staticmethod
    def _trajectories():
        rng = np.random.default_rng(0)
        out = []
        out.append(rng.standard_normal((60, 14)))                      # noise
        x = np.empty((120, 2)); x[0] = (0.3, 0.1)                      # Henon map (chaotic)
        for i in range(1, 120):
            x[i] = (1 - 1.4 * x[i - 1, 0] ** 2 + x[i - 1, 1], 0.3 * x[i - 1, 0])
        out.append(x)
        out.append(np.cumsum(rng.standard_normal((80, 5)), axis=0))    # random walk
        return out

    @pytest.mark.parametrize("maxt,window", [(10, 3), (30, 10), (15, 5)])
    def test_matches_nolitsa_mle(self, maxt, window):
        lyapunov = pytest.importorskip("nolitsa.lyapunov")
        for y in self._trajectories():
            if len(y) <= 2 * window + 3:
                continue
            ours = rosenstein_divergence(y, maxt=maxt, window=window)
            ref = lyapunov.mle(y, maxt=maxt, window=window, metric="euclidean")
            assert ours.shape == ref.shape
            np.testing.assert_array_equal(ours, ref)

    def test_raises_when_series_too_short_for_window(self):
        y = np.random.default_rng(1).standard_normal((10, 3))
        with pytest.raises(ValueError, match="maxnum"):
            rosenstein_divergence(y, maxt=5, window=10)

    def test_henon_map_has_positive_lle(self):
        y = self._trajectories()[1]
        res = compute_lyapunov_exponent(y, maxt=15, window=5)
        assert res["lle"] > 0
