"""Tests for traits_audit.detrend — per-regime EMA state detrending.

Ported from battery-oracle's PyBaMMOracle._detrend_state tests when the
detrending logic moved into this package as RegimeDetrender, generalized to
accept regime vectors of any dimensionality (not hardcoded to 6).
"""
import numpy as np
import pytest

from traits_audit.detrend import DetrendResult, RegimeDetrender


class TestRegimeDetrender:
    def test_config_wires_to_attrs(self):
        d = RegimeDetrender(alpha=0.2, n_groups=4, warmup=10)
        assert d._alpha == pytest.approx(0.2)
        assert d._n_groups == 4
        assert d._warmup == 10

    def test_warmup_then_group_ema(self):
        """First `warmup` calls use the global mean; afterwards k-means
        groups drive a per-group EMA. NaN state slots propagate as NaN in
        the detrended vector. reset() clears all accumulators."""
        d = RegimeDetrender(alpha=0.1, n_groups=3, warmup=5)
        regimes = [np.array([100., 50., 1., .5, 100., 1.]),
                   np.array([140., 70., 1., .5, 100., 1.])]
        warmups, groups = [], []
        for i in range(12):
            state = np.array([0.1 + 0.001 * i, 1.0, 0.9, np.nan, np.nan, np.nan])
            result = d.update(state, regimes[i % 2])
            assert isinstance(result, DetrendResult)
            warmups.append(result.is_warmup)
            groups.append(result.group)
            assert np.isnan(result.detrended[3:]).all()  # NaN slots propagate
        # First 5 calls are warmup (global-mean reference, group -1); then grouped.
        assert warmups[:5] == [True] * 5
        assert warmups[5] is False and groups[5] >= 0
        assert d._centroids is not None
        # 2 unique regimes -> at most 2 groups even though n_groups=3.
        assert d._centroids.shape[0] == 2
        d.reset()
        assert d._centroids is None
        assert d._global_sum is None
        assert d._group_ema == {}

    def test_detrended_equals_state_minus_reference(self):
        """Sanity check the arithmetic: detrended == state - reference."""
        d = RegimeDetrender(alpha=0.5, n_groups=2, warmup=2)
        regime = np.array([1.0, 2.0])
        r0 = d.update(np.array([10.0, 20.0]), regime)
        np.testing.assert_allclose(r0.detrended, np.array([0.0, 0.0]))  # first call: state - state
        r1 = d.update(np.array([12.0, 24.0]), regime)
        global_mean = np.array([11.0, 22.0])  # mean of [10,20] and [12,24]
        np.testing.assert_allclose(r1.detrended, np.array([12.0, 24.0]) - global_mean)

    @pytest.mark.parametrize("regime_dim", [1, 3, 8])
    def test_handles_arbitrary_regime_dimensionality(self, regime_dim):
        """Regime vectors are not hardcoded to any particular length — this
        is the regression guard for the battery-oracle `protocol[:6]` gap
        that silently dropped the 7th (T_ambient_K) dimension."""
        rng = np.random.default_rng(0)
        d = RegimeDetrender(alpha=0.1, n_groups=4, warmup=6)
        two_regimes = [rng.standard_normal(regime_dim), rng.standard_normal(regime_dim) + 10]
        for i in range(15):
            state = np.array([float(i), float(i) * 2.0])
            result = d.update(state, two_regimes[i % 2])
            assert result.detrended.shape == state.shape
        assert d._centroids is not None
        assert d._centroids.shape[1] == regime_dim

    def test_reset_allows_fresh_trajectory(self):
        d = RegimeDetrender(alpha=0.1, n_groups=2, warmup=3)
        for i in range(5):
            d.update(np.array([float(i)]), np.array([0.0]))
        assert d._n_calls == 5
        d.reset()
        assert d._n_calls == 0
        assert d._regimes == []
        assert d._states == []
        assert d._whiten_std is None

    def test_state_shape_change_reallocates_global_sum(self):
        """If the caller's state vector grows/shrinks mid-trajectory (e.g. a
        schema change), the global accumulator is reallocated rather than
        raising a shape-mismatch error."""
        d = RegimeDetrender(alpha=0.1, n_groups=2, warmup=10)
        d.update(np.array([1.0, 2.0]), np.array([0.0]))
        result = d.update(np.array([1.0, 2.0, 3.0]), np.array([0.0]))
        assert result.detrended.shape == (3,)
