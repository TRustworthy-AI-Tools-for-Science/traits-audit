"""Tests for traits_audit.stability.lyapunov_eq and the backward-compat module aliases."""
import numpy as np
import pytest
from scipy.linalg import solve_discrete_lyapunov

from traits_audit.stability.lyapunov_eq import (
    compute_lyapunov,
    eigenvalues_and_stability,
    lyapunov_P,
    make_gd_predictor,
    numerical_jacobian,
    numerical_jacobian_batched,
    rescale_to_stable,
)


def test_numerical_jacobian_of_linear_map_recovers_matrix():
    A = np.array([[0.5, 0.2], [-0.1, 0.9]])
    J = numerical_jacobian(lambda s, a: A @ s, np.array([0.3, -0.2]))
    np.testing.assert_allclose(J, A, atol=1e-8)


def test_numerical_jacobian_passes_action_through_unchanged():
    seen = []
    def pred(s, a):
        seen.append(a)
        return s
    numerical_jacobian(pred, np.zeros(2), action="act")
    assert set(map(str, seen)) == {"act"}


def test_gd_predictor_jacobian_is_I_minus_alpha_hessian():
    H = np.array([[2.0, 0.5], [0.5, 1.0]])
    f = lambda x: 0.5 * x @ H @ x
    J = numerical_jacobian(make_gd_predictor(f, alpha=0.1), np.array([0.4, -0.3]), None, dx=1e-3)
    np.testing.assert_allclose(J, np.eye(2) - 0.1 * H, atol=1e-4)


def test_batched_jacobian_matches_scalar_nested_computation():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((4, 4))
    f_scalar = lambda x: float(np.sin(x @ W @ x) + 0.1 * np.sum(x ** 3))
    f_batched = lambda X: np.array([f_scalar(r) for r in X])
    state = rng.standard_normal(4)
    J_seq = numerical_jacobian(make_gd_predictor(f_scalar, alpha=0.02), state)
    J_bat = numerical_jacobian_batched(f_batched, state, alpha=0.02)
    np.testing.assert_allclose(J_bat, J_seq, atol=1e-6)


def test_eigenvalues_and_stability_flags():
    out = eigenvalues_and_stability(np.diag([0.5, 1.2]))
    assert out["lambda_max"] == pytest.approx(1.2)
    assert out["lambda_min"] == pytest.approx(0.5)
    assert out["is_stable"] is False and out["n_unstable"] == 1


def test_rescale_to_stable_only_touches_unstable_matrices():
    stable = np.diag([0.5, 0.2])
    assert rescale_to_stable(stable) is stable
    rescaled = rescale_to_stable(np.diag([2.0, 1.0]), rho_max=0.9)
    assert np.abs(np.linalg.eigvals(rescaled)).max() == pytest.approx(0.9)


def test_lyapunov_P_solves_the_stein_equation():
    A = np.array([[0.5, 0.2], [-0.1, 0.7]])
    P = lyapunov_P(A)
    np.testing.assert_allclose(A.T @ P @ A - P, -np.eye(2), atol=1e-10)
    np.testing.assert_allclose(P, solve_discrete_lyapunov(A.T, np.eye(2)))


def test_lyapunov_P_rescales_unstable_A_before_solving():
    A = np.diag([1.5, 0.5])
    P = lyapunov_P(A, rho_max=0.99)
    A_s = A * (0.99 / 1.5)
    np.testing.assert_allclose(A_s.T @ P @ A_s - P, -np.eye(2), atol=1e-8)


def test_compute_lyapunov_returns_none_when_solver_raises(monkeypatch):
    import traits_audit.stability.lyapunov_eq as mod
    def boom(*a, **k):
        raise np.linalg.LinAlgError("singular")
    monkeypatch.setattr(mod, "solve_discrete_lyapunov", boom)
    assert compute_lyapunov(np.eye(2) * 0.5) is None
    with pytest.raises(np.linalg.LinAlgError):
        lyapunov_P(np.eye(2) * 0.5)


def test_compute_lyapunov_does_not_swallow_eigvals_failures():
    with pytest.raises(Exception):
        compute_lyapunov(np.array([[np.nan, 0.0], [0.0, 1.0]]))


def test_old_module_paths_alias_the_new_modules():
    import traits_audit
    import traits_audit.dmdc as old_dmdc
    import traits_audit.detrend as old_detrend
    from traits_audit.stability import dmdc as new_dmdc, detrend as new_detrend
    assert old_dmdc is new_dmdc and old_detrend is new_detrend
    assert traits_audit.dmdc is new_dmdc
    assert old_dmdc._detrend_trajectory is new_dmdc.detrend_trajectory
    from traits_audit.dmdc import fit_dmdc
    from traits_audit.detrend import RegimeDetrender              # noqa: F401
    assert fit_dmdc is new_dmdc.fit_dmdc and RegimeDetrender is new_detrend.RegimeDetrender
