"""Tests for traits_audit.stability.action_hygiene (ported from battery-forecast exp4)."""
import numpy as np
import pytest

from traits_audit.stability.dmdc import detrend_trajectory
from traits_audit.stability.action_hygiene import (
    compute_global_action_kept_mask,
    drop_degenerate_action_columns,
)


class TestComputeGlobalActionKeptMask:
    def test_column_real_in_only_one_trajectory_is_kept_globally(self):
        """A column non-degenerate in AT LEAST ONE trajectory is kept for
        EVERY trajectory (policy_a varies column 0; policy_b varies nothing)."""
        rng = np.random.default_rng(1)
        T = 20
        varying_col = rng.uniform(-1, 1, size=T)
        constant_col = np.full(T, 5.0)
        mask = compute_global_action_kept_mask([
            np.column_stack([varying_col, constant_col]),
            np.column_stack([constant_col, constant_col]),
        ])
        assert list(mask) == [True, False]

    def test_columns_real_in_different_trajectories_are_ored_together(self):
        rng = np.random.default_rng(2)
        T = 20
        a = np.column_stack([rng.uniform(-1, 1, T), np.full(T, 1.0), np.full(T, 1.0)])
        b = np.column_stack([np.full(T, 1.0), np.full(T, 1.0), rng.uniform(-1, 1, T)])
        assert list(compute_global_action_kept_mask([a, b])) == [True, False, True]

    def test_skips_none_and_too_short_arrays(self):
        rng = np.random.default_rng(3)
        good = np.column_stack([rng.uniform(-1, 1, 10), np.full(10, 2.0)])
        mask = compute_global_action_kept_mask([None, np.zeros((1, 2)), good])
        assert list(mask) == [True, False]

    def test_raises_when_no_trajectories_have_usable_actions(self):
        with pytest.raises(RuntimeError):
            compute_global_action_kept_mask([])
        with pytest.raises(RuntimeError):
            compute_global_action_kept_mask([None, np.zeros((1, 2))])

    def test_feature_names_only_affect_the_message(self, capsys):
        a = np.column_stack([np.linspace(0, 1, 10), np.ones(10)])
        mask = compute_global_action_kept_mask([a], feature_names=["C_rate", "T"])
        assert list(mask) == [True, False]
        out = capsys.readouterr().out
        assert "C_rate" in out and "kept" in out


class TestDropDegenerateActionColumns:
    def test_all_real_columns_unchanged(self):
        rng = np.random.default_rng(0)
        actions = rng.standard_normal((20, 3))
        reduced, mask = drop_degenerate_action_columns(actions)
        assert reduced.shape == actions.shape
        assert np.array_equal(reduced, actions)
        assert mask.all()

    def test_all_degenerate_returns_single_zero_column(self):
        """Confirmed real case: nasa/QBC-Agree, n_real_cols=0/6 -- must not
        return a 0-width array (fit_dmdc/DMDc both require
        actions.shape[1] >= 1)."""
        actions = np.full((15, 4), 3.7)  # exactly constant, every column
        reduced, mask = drop_degenerate_action_columns(actions)
        assert reduced.shape == (15, 1)
        assert np.all(reduced == 0)
        assert not mask.any()

    def test_mixed_real_and_degenerate_columns(self):
        """Mirrors real production data exactly: 1 genuinely-varying
        column plus several columns that are constant but NOT exactly
        zero-variance (std ~1e-10) -- confirmed in real data as low as
        1e-16, so a floor-based check (not an exact ==0 check) is
        required to catch this."""
        rng = np.random.default_rng(1)
        T = 30
        real_col = rng.standard_normal(T)
        near_constant_col = 5.0 + rng.standard_normal(T) * 1e-10
        exactly_constant_col = np.full(T, 2.0)
        actions = np.column_stack([real_col, near_constant_col, exactly_constant_col])
        reduced, mask = drop_degenerate_action_columns(actions)
        assert list(mask) == [True, False, False]
        assert reduced.shape == (T, 1)
        assert np.array_equal(reduced[:, 0], real_col)

    def test_std_floor_is_configurable(self):
        rng = np.random.default_rng(2)
        col = 1.0 + rng.standard_normal(20) * 1e-4
        actions = col[:, None]
        # default floor (1e-6) keeps this column (std ~1e-4 > 1e-6)
        _, mask_default = drop_degenerate_action_columns(actions)
        assert mask_default[0]
        # a stricter floor drops it
        _, mask_strict = drop_degenerate_action_columns(actions, std_floor=1.0)
        assert not mask_strict[0]

    def test_regression_guard_fix_prevents_astronomically_large_B_r(self):
        """The most important test: reproduces the REAL bug on a
        controlled fixture (1 real + 5 degenerate action columns, mirrors
        this project's actual 6-D action vector), confirming that fitting
        WITHOUT the fix via pydmd.DMDc (the exact backend
        stability_convergence_windowed_pydmd uses) produces an
        astronomically large B -- confirmed this session to be a
        pydmd.DMDc-specific mechanism (its own DMDBUnknownOperator SVDs
        the stacked [state; control] matrix with svd_rank_omega=-1 by
        default, i.e. no truncation, and inverts every singular value via
        np.reciprocal with no floor) -- while fitting WITH the fix
        (degenerate columns dropped first) produces a B of sane
        magnitude. This is what proves the fix addresses the diagnosed
        mechanism, not just that the helper behaves reasonably alone.

        Separately confirmed (not asserted below, but worth recording):
        traits_audit.dmdc.fit_dmdc's own numpy.linalg.lstsq(rcond=None)
        does NOT reproduce this severe blowup on the same fixture (its
        SVD-based pseudoinverse already floors near-zero singular values
        near machine precision) -- B_r stays sane (~0.13) even on raw,
        undropped degenerate columns. The fix is still applied at BOTH
        call sites (this function's own _regime_fit_r2 uses fit_dmdc) for
        consistency and to avoid wasting fitted rank on columns that
        carry zero information, but pydmd.DMDc is the call site where
        this bug is actually severe."""
        rng = np.random.default_rng(3)
        T = 60
        n_state = 8
        real_action = rng.uniform(-1, 1, size=T)
        # 5 degenerate columns: numerically constant, floating-point-noise
        # level variation (matches real data's confirmed std as low as 1e-16)
        degenerate = np.tile([10.0, 0.5, 0.2, 3.0, 1.5], (T, 1)) \
            + rng.standard_normal((T, 5)) * 1e-13
        actions_raw = np.column_stack([real_action, degenerate])

        aug_states = rng.standard_normal((T + 1, n_state)) * 0.1
        for t in range(1, T + 1):
            aug_states[t] = 0.7 * aug_states[t - 1] + 0.05 * real_action[t - 1]

        state_win = detrend_trajectory(aug_states, None)
        X = state_win.T
        U_bad = actions_raw[:T].T
        from pydmd import DMDc
        dmdc_bad = DMDc(svd_rank=4)
        dmdc_bad.fit(X, U_bad)
        assert np.abs(dmdc_bad.B).max() > 1e6, (
            "fixture failed to reproduce the real bug -- pydmd.DMDc's B "
            "should be astronomically large when fit on raw (undropped) "
            "degenerate action columns"
        )

        actions_fixed, kept_mask = drop_degenerate_action_columns(actions_raw)
        assert list(kept_mask) == [True, False, False, False, False, False]
        U_fixed = actions_fixed[:T].T
        dmdc_fixed = DMDc(svd_rank=4)
        dmdc_fixed.fit(X, U_fixed)
        assert np.abs(dmdc_fixed.B).max() < 100
