"""Tests for traits_audit.trajectory — TrajectoryRecord and analyze_trajectory.

Focuses on the block-bootstrap CI for the Gramian eigenvalue ratio (``cond_Wc``,
the paper's headline mechanism statistic — not ``rho_A_joint``) and the
``T >= 4d+1`` estimability flag, both added alongside the pre-existing
``rho_A_joint`` bootstrap.
"""
import numpy as np
import pytest

from traits_audit.trajectory import TrajectoryRecord, analyze_trajectory


def _make_record(T=80, warmup=10, seed=0):
    """Synthetic AL trajectory: 2-D state driven by actions, plus a 2-D
    uncertainty vector ``[sigma_ep, sigma_al]`` — sigma_ep decays (reducible),
    sigma_al hovers around a fixed floor (irreducible).
    """
    rng = np.random.default_rng(seed)
    rec = TrajectoryRecord(domain="unit_test", policy="LCB")
    z = rng.standard_normal(2) * 0.1
    A = 0.85 * np.eye(2)
    for t in range(T):
        a = rng.standard_normal(2) * 0.1
        z = A @ z + a + rng.standard_normal(2) * 1e-3
        sigma_ep = 1.0 * np.exp(-t / 20.0) + 0.01 * rng.standard_normal()
        sigma_al = 0.3 + 0.01 * rng.standard_normal()
        rec.add(state=z, uncertainty=[sigma_ep, sigma_al], action=a, is_warmup=(t < warmup))
    return rec.finalize()


class TestAnalyzeTrajectoryEstimability:
    def test_min_length_ok_true_for_long_trajectory(self):
        rec = _make_record(T=80)
        result = analyze_trajectory(rec, n_components=4, n_boot=20)
        assert result.min_length_ok is True

    def test_min_length_ok_false_for_short_trajectory(self):
        rec = _make_record(T=12, warmup=0)
        result = analyze_trajectory(rec, n_components=4, n_boot=10)
        assert result.min_length_ok is False


class TestAnalyzeTrajectoryCondWcCi:
    def test_cond_wc_ci_is_valid_interval(self):
        rec = _make_record(T=80, seed=1)
        result = analyze_trajectory(rec, n_components=4, n_boot=30, block_len=8)
        assert result.cond_Wc_ci is not None
        lo, hi = result.cond_Wc_ci
        assert lo > 0 and hi > 0
        assert lo <= hi

    def test_cond_wc_point_estimate_within_or_near_ci(self):
        # The point estimate (fit on the full trajectory) should be broadly
        # consistent with the resampled distribution, not wildly outside it.
        rec = _make_record(T=80, seed=4)
        result = analyze_trajectory(rec, n_components=4, n_boot=50, block_len=8)
        lo, hi = result.cond_Wc_ci
        # Allow generous slack: block bootstrap on a short autocorrelated
        # series is noisy, this just guards against a badly broken CI.
        assert lo / 10 <= result.cond_Wc <= hi * 10

    def test_cond_wc_ci_none_when_bootstrap_disabled(self):
        rec = _make_record(T=80, seed=2)
        result = analyze_trajectory(rec, n_components=4, block_bootstrap=False)
        assert result.cond_Wc_ci is None
        assert result.rho_joint_ci is None

    def test_no_crash_when_trajectory_shorter_than_block_len(self):
        # Regression: block bootstrap used to call rng.integers(0, T -
        # block_len, ...) unconditionally, raising ValueError when T <=
        # block_len (e.g. a tiny demo smoke run). Should degrade gracefully.
        rec = _make_record(T=5, warmup=0, seed=9)
        result = analyze_trajectory(rec, n_components=4, n_boot=20, block_len=8)
        assert result.cond_Wc_ci is None
        assert result.rho_joint_ci is None


class TestAnalyzeTrajectoryWcEigenvectors:
    def test_shape_matches_augmented_dim_and_rank(self):
        rec = _make_record(T=80, seed=6)
        result = analyze_trajectory(rec, n_components=4, n_boot=0, block_bootstrap=False)
        n_aug = 2 + 2  # state dim + uncertainty dim
        assert result.Wc_eigenvectors.shape[0] == n_aug
        assert result.Wc_eigenvectors.shape[1] == len(result.Wc_singular_values)

    def test_eigenvectors_are_unit_norm(self):
        # Wc_eigenvectors is a projection (U_joint @ U_wc) of an orthonormal
        # basis through U_joint (itself orthonormal, from fit_dmdc's SVD), so
        # columns should still have unit norm.
        rec = _make_record(T=100, seed=7)
        result = analyze_trajectory(rec, n_components=4, n_boot=0, block_bootstrap=False)
        norms = np.linalg.norm(result.Wc_eigenvectors, axis=0)
        np.testing.assert_allclose(norms, 1.0, atol=1e-8)

    def test_eigenvalue_ratio_separates_decaying_from_floor_component(self):
        # sigma_ep decays toward 0 (reducible) while sigma_al hovers near a
        # fixed floor (irreducible) — the Gramian should find a genuine
        # separation, i.e. cond_Wc well above 1.
        rec = _make_record(T=100, seed=7)
        result = analyze_trajectory(rec, n_components=4, n_boot=0, block_bootstrap=False)
        assert result.cond_Wc > 10.0


class TestAnalyzeTrajectoryBasics:
    def test_uncertainty_dim_and_scalar_warning(self):
        rec = _make_record(T=60, seed=3)
        result = analyze_trajectory(rec, n_components=4, n_boot=10)
        assert result.uncertainty_dim == 2
        assert result.scalar_uncertainty_warning is False

    def test_warmup_steps_excluded_from_fit(self):
        rec = _make_record(T=60, warmup=15, seed=5)
        result = analyze_trajectory(rec, n_components=4, n_boot=0, block_bootstrap=False)
        assert result.warmup_excluded == 15
