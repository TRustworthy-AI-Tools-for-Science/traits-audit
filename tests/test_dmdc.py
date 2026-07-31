"""Tests for traits_audit.dmdc — DMDc fitting and stability diagnostics.

Synthetic systems use a known stable/unstable A_r so that eigenvalue-based
assertions are exact, not just "didn't crash".
"""
import numpy as np
import pytest

from traits_audit.dmdc import (
    bootstrap_eig_ci,
    compute_gramians,
    dmdc_r2,
    equilibrium_state,
    fit_dmdc,
    fit_dmdc_pairs,
    modal_decomposition,
    perturbation_response,
    pseudospectrum,
    rank_sensitivity,
    stability_convergence,
)


# ── Synthetic trajectory builders ────────────────────────────────────────────

def _simulate(A, B, n_steps=60, n_action=2, seed=0):
    """Simulate z_{t+1} = A z_t + B a_t + noise; return (states, actions)."""
    rng = np.random.default_rng(seed)
    r = A.shape[0]
    z = rng.standard_normal(r) * 0.1
    states = [z]
    actions = []
    for _ in range(n_steps):
        a = rng.standard_normal(n_action) * 0.1
        z = A @ z + B @ a + rng.standard_normal(r) * 1e-4
        states.append(z)
        actions.append(a)
    actions.append(np.zeros(n_action))  # pad to len(states)
    return np.array(states), np.array(actions)


def _stable_complex_A():
    """2x2 rotation-scaling matrix with eigenvalues 0.8*exp(+-i*pi/4) (complex, stable)."""
    theta = np.pi / 4
    return 0.8 * np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])


def _unstable_A():
    return np.array([[1.5, 0.0], [0.0, 1.2]])


@pytest.fixture
def stable_traj():
    A = _stable_complex_A()
    B = np.array([[1.0, 0.0], [0.0, 1.0]])
    states, actions = _simulate(A, B, n_steps=80, seed=1)
    return A, B, states, actions


@pytest.fixture
def unstable_traj():
    A = _unstable_A()
    B = np.array([[1.0, 0.0], [0.0, 1.0]])
    states, actions = _simulate(A, B, n_steps=80, seed=2)
    return A, B, states, actions


# ── fit_dmdc ──────────────────────────────────────────────────────────────────

class TestFitDmdc:
    def test_shapes(self, stable_traj):
        _, _, states, actions = stable_traj
        A_r, B_r, U_r = fit_dmdc(states, actions, n_components=2)
        assert A_r.shape == (2, 2)
        assert B_r.shape == (2, 2)
        assert U_r.shape == (2, 2)

    def test_recovers_complex_eigenvalues(self, stable_traj):
        # detrend=False: this test verifies core SVD/lstsq recovery of known
        # ground-truth dynamics, a concern orthogonal to detrending (tested
        # separately in TestDetrending below).
        A_true, _, states, actions = stable_traj
        A_r, _, U_r = fit_dmdc(states, actions, n_components=2, detrend=False)
        # Project A_r back into original coords; should approximate A_true
        A_full = U_r @ A_r @ U_r.T
        np.testing.assert_allclose(A_full, A_true, atol=0.05)

    def test_eigenvalues_are_complex_for_rotation_matrix(self, stable_traj):
        _, _, states, actions = stable_traj
        A_r, _, _ = fit_dmdc(states, actions, n_components=2, detrend=False)
        eigs = np.linalg.eigvals(A_r)
        assert np.any(np.abs(eigs.imag) > 1e-6)

    def test_rank_clipped_to_t_fit(self, stable_traj):
        _, _, states, actions = stable_traj
        # Ask for a rank far larger than n or T-1; should clip without error.
        A_r, B_r, U_r = fit_dmdc(states[:4], actions[:4], n_components=50)
        assert A_r.shape[0] <= 3  # min(n=2, T_fit=3)

    def test_unstable_system_has_eigenvalue_above_one(self, unstable_traj):
        _, _, states, actions = unstable_traj
        A_r, _, _ = fit_dmdc(states, actions, n_components=2, detrend=False)
        assert np.abs(np.linalg.eigvals(A_r)).max() > 1.0


# ── fit_dmdc_pairs ────────────────────────────────────────────────────────────

class TestFitDmdcPairs:
    def test_matches_fit_dmdc_on_contiguous_pairs(self, stable_traj):
        # detrend=False on both sides: fit_dmdc and fit_dmdc_pairs use
        # different *default* detrending algorithms (causal streaming vs.
        # order-invariant pooled-mean, see TestDetrending), so this test
        # isolates the core SVD/lstsq equivalence the two entry points share.
        _, _, states, actions = stable_traj
        T = len(states) - 1
        A_r1, B_r1, U_r1 = fit_dmdc(states, actions, n_components=2, detrend=False)
        A_r2, B_r2, U_r2 = fit_dmdc_pairs(states[:T], states[1:T + 1], actions[:T],
                                           n_components=2, detrend=False)
        # Subspaces may differ in sign but A_full should match closely.
        A_full1 = U_r1 @ A_r1 @ U_r1.T
        A_full2 = U_r2 @ A_r2 @ U_r2.T
        np.testing.assert_allclose(A_full1, A_full2, atol=1e-6)

    def test_shapes_with_explicit_pairs(self):
        rng = np.random.default_rng(3)
        aug_t = rng.standard_normal((30, 4))
        aug_t1 = rng.standard_normal((30, 4))
        actions = rng.standard_normal((30, 2))
        A_r, B_r, U_r = fit_dmdc_pairs(aug_t, aug_t1, actions, n_components=3)
        assert A_r.shape == (3, 3)
        assert B_r.shape == (3, 2)
        assert U_r.shape == (4, 3)


# ── detrending ────────────────────────────────────────────────────────────────

class TestDetrending:
    def test_fit_dmdc_detrend_changes_fit_on_mean_shifted_trajectory(self, stable_traj):
        _, _, states, actions = stable_traj
        shifted = states + np.array([5.0, -5.0])
        A_raw, _, U_raw = fit_dmdc(shifted, actions, n_components=2, detrend=False)
        A_dt, _, U_dt = fit_dmdc(shifted, actions, n_components=2, detrend=True)
        A_full_raw = U_raw @ A_raw @ U_raw.T
        A_full_dt = U_dt @ A_dt @ U_dt.T
        assert not np.allclose(A_full_raw, A_full_dt, atol=1e-3)

    def test_fit_dmdc_pairs_pooled_detrend_recovers_shifted_system(self, stable_traj):
        A_true, _, states, actions = stable_traj
        shifted = states + np.array([10.0, -10.0])
        T = len(shifted) - 1
        A_r, _, U_r = fit_dmdc_pairs(shifted[:T], shifted[1:T + 1], actions[:T],
                                      n_components=2, detrend=True)
        A_full = U_r @ A_r @ U_r.T
        np.testing.assert_allclose(A_full, A_true, atol=0.1)

    def test_bootstrap_eig_ci_finite_with_detrending(self, stable_traj):
        _, _, states, actions = stable_traj
        lo, hi = bootstrap_eig_ci(states, actions, n_boot=20, n_components=2, seed=0)
        assert np.all(np.isfinite(lo))
        assert np.all(np.isfinite(hi))
        assert np.all(lo <= hi)


# ── dmdc_r2 ───────────────────────────────────────────────────────────────────

class TestDmdcR2:
    def test_high_r2_for_well_fit_linear_system(self, stable_traj):
        _, _, states, actions = stable_traj
        A_r, B_r, U_r = fit_dmdc(states, actions, n_components=2, detrend=False)
        r2 = dmdc_r2(states, actions, A_r, B_r, U_r)
        assert r2 > 0.95

    def test_low_r2_for_random_dynamics_matrix(self, stable_traj):
        _, _, states, actions = stable_traj
        _, B_r, U_r = fit_dmdc(states, actions, n_components=2)
        rng = np.random.default_rng(7)
        A_random = rng.standard_normal((2, 2)) * 5
        r2 = dmdc_r2(states, actions, A_random, B_r, U_r)
        assert r2 < 0.5

    def test_nan_when_zero_variance(self):
        states = np.zeros((10, 2))
        actions = np.zeros((10, 2))
        A_r = np.zeros((2, 2))
        B_r = np.zeros((2, 2))
        U_r = np.eye(2)
        r2 = dmdc_r2(states, actions, A_r, B_r, U_r)
        assert np.isnan(r2)


# ── stability_convergence ────────────────────────────────────────────────────

class TestStabilityConvergence:
    def test_output_length(self, stable_traj):
        _, _, states, actions = stable_traj
        conv = stability_convergence(states, actions, min_obs=5, n_components=2)
        T = min(len(states) - 1, len(actions))
        assert len(conv) == T - 5 + 1

    def test_values_below_one_for_stable_system(self, stable_traj):
        _, _, states, actions = stable_traj
        conv = stability_convergence(states, actions, min_obs=10, n_components=2)
        finite = conv[np.isfinite(conv)]
        assert finite.size > 0
        # The fitted A_r(t) should converge near the true |lambda| ~ 0.8.
        assert finite[-1] < 1.0

    def test_handles_fit_failures_gracefully(self):
        # Degenerate trajectory: too few points relative to min_obs → some
        # iterations may fail internally but should not raise.
        states = np.zeros((6, 2))
        actions = np.zeros((6, 2))
        conv = stability_convergence(states, actions, min_obs=5, n_components=2)
        assert isinstance(conv, np.ndarray)


# ── perturbation_response ────────────────────────────────────────────────────

class TestPerturbationResponse:
    def test_shape(self):
        A_r = 0.5 * np.eye(3)
        curves = perturbation_response(A_r, n_steps=10, n_samples=20, seed=0)
        assert curves.shape == (20, 11)

    def test_decays_for_contractive_matrix(self):
        A_r = 0.3 * np.eye(2)
        curves = perturbation_response(A_r, n_steps=15, n_samples=50, seed=0)
        assert curves[:, -1].mean() < curves[:, 0].mean()

    def test_initial_norm_is_one(self):
        A_r = np.eye(2)
        curves = perturbation_response(A_r, n_steps=5, n_samples=10, seed=0)
        np.testing.assert_allclose(curves[:, 0], 1.0)

    def test_deterministic_with_seed(self):
        A_r = 0.5 * np.array([[0.9, 0.1], [-0.1, 0.9]])
        c1 = perturbation_response(A_r, n_steps=5, n_samples=10, seed=42)
        c2 = perturbation_response(A_r, n_steps=5, n_samples=10, seed=42)
        np.testing.assert_array_equal(c1, c2)


# ── pseudospectrum ────────────────────────────────────────────────────────────

class TestPseudospectrum:
    def test_shapes(self):
        A_r = np.array([[0.5, 0.1], [-0.1, 0.5]])
        RE, IM, sigma_min = pseudospectrum(A_r, grid_n=10)
        assert RE.shape == (10, 10)
        assert IM.shape == (10, 10)
        assert sigma_min.shape == (10, 10)

    def test_sigma_min_near_eigenvalue_is_small(self):
        A_r = np.diag([0.5, -0.5])
        RE, IM, sigma_min = pseudospectrum(A_r, grid_n=61)
        # Grid point closest to lambda=0.5 (on the real axis) should have a
        # near-zero smallest singular value of (zI - A_r).
        idx = np.unravel_index(np.argmin(np.abs(RE - 0.5) + np.abs(IM)), RE.shape)
        assert sigma_min[idx] < 0.05

    def test_sigma_min_nonnegative(self):
        A_r = np.array([[0.2, 0.9], [-0.9, 0.2]])
        _, _, sigma_min = pseudospectrum(A_r, grid_n=15)
        assert np.all(sigma_min >= 0)


# ── equilibrium_state ─────────────────────────────────────────────────────────

class TestEquilibriumState:
    def test_fixed_point_satisfies_equation(self):
        A_r = np.diag([0.5, 0.2])
        B_r = np.eye(2)
        U_r = np.eye(2)
        a_star = np.array([1.0, 2.0])
        s_star = equilibrium_state(A_r, B_r, U_r, a_star)
        # z* = (I - A)^-1 B a*; check it's a fixed point: A z* + B a* == z*
        z_star = s_star  # U_r is identity here
        np.testing.assert_allclose(A_r @ z_star + B_r @ a_star, z_star, atol=1e-10)

    def test_zero_action_gives_origin(self):
        A_r = np.diag([0.5, 0.3])
        B_r = np.eye(2)
        U_r = np.eye(2)
        s_star = equilibrium_state(A_r, B_r, U_r, np.zeros(2))
        np.testing.assert_allclose(s_star, 0.0, atol=1e-10)

    def test_raises_on_singular_system(self):
        A_r = np.eye(2)  # I - A_r is singular
        B_r = np.eye(2)
        U_r = np.eye(2)
        with pytest.raises(np.linalg.LinAlgError):
            equilibrium_state(A_r, B_r, U_r, np.array([1.0, 1.0]))


# ── compute_gramians ──────────────────────────────────────────────────────────

class TestComputeGramians:
    def test_shapes(self):
        A_r = np.diag([0.5, 0.3])
        B_r = np.eye(2)
        W_c, W_o = compute_gramians(A_r, B_r)
        assert W_c.shape == (2, 2)
        assert W_o.shape == (2, 2)

    def test_gramians_are_symmetric_positive_definite(self):
        A_r = np.array([[0.4, 0.1], [-0.1, 0.4]])
        B_r = np.eye(2)
        W_c, W_o = compute_gramians(A_r, B_r)
        np.testing.assert_allclose(W_c, W_c.T, atol=1e-10)
        np.testing.assert_allclose(W_o, W_o.T, atol=1e-10)
        assert np.all(np.linalg.eigvalsh(W_c) > 0)
        assert np.all(np.linalg.eigvalsh(W_o) > 0)

    def test_underactuated_direction_increases_controllability_cond(self):
        A_r = np.diag([0.5, 0.5])
        B_full = np.eye(2)
        B_underactuated = np.array([[1.0], [0.0]])  # only controls first mode
        _, _ = compute_gramians(A_r, B_full)
        W_c_full, _ = compute_gramians(A_r, B_full)
        W_c_under, _ = compute_gramians(A_r, B_underactuated)
        assert np.linalg.cond(W_c_under) > np.linalg.cond(W_c_full)

    def test_unstable_A_r_does_not_warn_or_raise(self):
        # ρ(A_r) > 1 → would normally trigger LinAlgWarning from scipy.
        # The stabilisation rescale must suppress it entirely.
        import warnings
        from scipy.linalg import LinAlgWarning
        A_r = np.array([[1.5, 0.2], [-0.2, 1.3]])   # spectral radius > 1
        B_r = np.ones((2, 1))
        with warnings.catch_warnings():
            warnings.simplefilter("error", LinAlgWarning)
            W_c, W_o = compute_gramians(A_r, B_r)   # must not raise
        assert W_c.shape == (2, 2)
        assert W_o.shape == (2, 2)

    def test_marginally_stable_A_r_returns_finite_gramians(self):
        # Eigenvalue exactly on the unit circle (|λ| = 1).
        A_r = np.array([[1.0, 0.0], [0.0, 0.9]])
        B_r = np.eye(2)
        W_c, W_o = compute_gramians(A_r, B_r)
        assert np.all(np.isfinite(W_c))
        assert np.all(np.isfinite(W_o))


# ── bootstrap_eig_ci ──────────────────────────────────────────────────────────

class TestBootstrapEigCi:
    def test_shapes(self, stable_traj):
        _, _, states, actions = stable_traj
        ci_lo, ci_hi = bootstrap_eig_ci(states, actions, n_boot=20, n_components=2, seed=0)
        assert ci_lo.shape == (2,)
        assert ci_hi.shape == (2,)

    def test_ci_lo_below_ci_hi(self, stable_traj):
        _, _, states, actions = stable_traj
        ci_lo, ci_hi = bootstrap_eig_ci(states, actions, n_boot=30, n_components=2, seed=1)
        assert np.all(ci_lo <= ci_hi)

    def test_true_magnitude_within_ci(self, stable_traj):
        A_true, _, states, actions = stable_traj
        ci_lo, ci_hi = bootstrap_eig_ci(states, actions, n_boot=100, n_components=2, seed=2)
        true_mag = np.abs(np.linalg.eigvals(A_true)).max()
        assert ci_lo.min() <= true_mag <= ci_hi.max() + 0.1

    def test_deterministic_with_seed(self, stable_traj):
        _, _, states, actions = stable_traj
        ci1 = bootstrap_eig_ci(states, actions, n_boot=15, n_components=2, seed=5)
        ci2 = bootstrap_eig_ci(states, actions, n_boot=15, n_components=2, seed=5)
        np.testing.assert_array_equal(ci1[0], ci2[0])
        np.testing.assert_array_equal(ci1[1], ci2[1])


# ── modal_decomposition ───────────────────────────────────────────────────────

class TestModalDecomposition:
    def test_sorted_by_descending_magnitude(self):
        A_r = np.diag([0.2, 0.8])
        U_r = np.eye(2)
        modes = modal_decomposition(A_r, U_r)
        mags = [m["magnitude"] for m in modes]
        assert mags == sorted(mags, reverse=True)

    def test_dominant_feature_named(self):
        A_r = np.diag([0.2, 0.8])
        U_r = np.eye(2)
        modes = modal_decomposition(A_r, U_r, feature_names=["alpha", "beta"])
        assert modes[0]["dominant_feature"] == "beta"

    def test_dominant_feature_index_when_no_names(self):
        A_r = np.diag([0.2, 0.8])
        U_r = np.eye(2)
        modes = modal_decomposition(A_r, U_r)
        assert modes[0]["dominant_feature"] == 1

    def test_complex_mode_has_nonzero_frequency(self, stable_traj):
        _, _, states, actions = stable_traj
        A_r, _, U_r = fit_dmdc(states, actions, n_components=2)
        modes = modal_decomposition(A_r, U_r)
        assert any(abs(m["frequency_rad"]) > 1e-6 for m in modes)

    def test_damping_negative_for_stable_mode(self):
        A_r = np.diag([0.5])
        U_r = np.eye(1)
        modes = modal_decomposition(A_r, U_r)
        assert modes[0]["damping"] < 0


# ── rank_sensitivity ──────────────────────────────────────────────────────────

class TestRankSensitivity:
    def test_returns_one_entry_per_rank(self, stable_traj):
        _, _, states, actions = stable_traj
        out = rank_sensitivity(states, actions, [1, 2])
        assert set(out.keys()) == {1, 2}

    def test_eigenvalue_count_matches_rank(self, stable_traj):
        _, _, states, actions = stable_traj
        out = rank_sensitivity(states, actions, [1, 2])
        assert len(out[1]) == 1
        assert len(out[2]) == 2

    def test_empty_array_on_failure(self):
        # n_components larger than what the degenerate trajectory supports
        # should still produce a (possibly empty) array, not raise.
        out = rank_sensitivity(np.zeros((2, 2)), np.zeros((2, 2)), [5])
        assert 5 in out
        assert isinstance(out[5], np.ndarray)
