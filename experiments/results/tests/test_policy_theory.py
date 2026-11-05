"""The state-space policies implement their derivation (state_space_bayes_derivation.md).

Synthetic inputs only; each test checks one relationship the derivation states.
"""
from __future__ import annotations

import numpy as np
import pytest
from results.policies import BayesianDynamicsPolicy, StateSpacePolicy
from traits_audit.stability.dmdc import detrend_trajectory


class _NoModel:
    """Surrogate stub: the tests below never score candidates through it."""

    def predict_with_uncertainty(self, X):
        X = np.atleast_2d(X)
        return {"mean": np.zeros((len(X), 1)), "std": np.zeros((len(X), 1))}


def _random_walk(n_steps, state_dim, action_dim, seed):
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(n_steps, state_dim)).cumsum(axis=0)
    actions = rng.uniform(-1.0, 1.0, size=(n_steps, action_dim))
    return states, actions


def _linear_system(n_steps, offset, M, B, seed):
    """s_{t+1} = offset + M (s_t - offset) + B a_t, noise-free."""
    rng = np.random.default_rng(seed)
    states = [offset.copy()]
    actions = []
    for _ in range(n_steps):
        a = rng.uniform(-1.0, 1.0, size=B.shape[1])
        actions.append(a)
        states.append(offset + M @ (states[-1] - offset) + B @ a)
    return np.array(states), np.array(actions)


# ── StateSpacePolicy: fit and prediction share the detrended frame (§4.1) ──

@pytest.mark.parametrize("window", [None, 8])
def test_state_space_current_state_detrended_like_the_fit(window):
    states, actions = _random_walk(30, 3, 2, seed=0)
    policy = StateSpacePolicy(_NoModel(), num_action_features=2, min_fit_obs=5, n_components=3,
                              window=window)
    for s, a in zip(states[:-1], actions[:-1], strict=True):
        policy.observe(s, a)

    current = states[-1]
    current_detrended, _ = policy._detrending_frame(current)

    fitted_plus_current = np.vstack([policy._fitted_aug_states, current])
    assert np.allclose(current_detrended, detrend_trajectory(fitted_plus_current)[-1])
    if window is not None:
        assert len(policy._fitted_aug_states) == window + 1


def test_state_space_prediction_is_in_the_fitted_frame():
    """A large offset must not leak into the prediction.

    The dynamics are fitted to deviations from a baseline.  Applying them to
    the raw state (the frame mistake) misses by roughly the offset; adding the
    baseline back lands near the true next state.
    """
    offset = np.array([50.0, -30.0])
    M = 0.6 * np.eye(2)
    B = np.array([[1.0], [0.5]])
    states, actions = _linear_system(40, offset, M, B, seed=1)
    policy = StateSpacePolicy(_NoModel(), num_action_features=1, min_fit_obs=5, n_components=2)
    for s, a in zip(states[:-1], actions, strict=True):
        policy.observe(s, a)

    current = states[-1]
    action = np.array([0.7])
    truth = offset + M @ (current - offset) + B @ action

    predicted = policy._predict_next_aug(current, action)
    raw_frame = policy._U_r @ (policy._A_r @ (current @ policy._U_r) + policy._B_r @ action)

    error = np.linalg.norm(predicted - truth)
    assert error < 0.05 * np.linalg.norm(offset)
    assert np.linalg.norm(raw_frame - truth) > 10 * error


# ── BayesianDynamicsPolicy: full Kalman update of the posterior (§9, §10) ──

def _batch_posterior(policy, states, actions):
    """Posterior over every completed transition, in the policy's (fixed) basis."""
    Z = states @ policy._U_r
    X = np.hstack([Z[:-1], actions[:len(states) - 1]])
    G = policy.lambda_reg * np.eye(X.shape[1]) + X.T @ X
    W = np.linalg.solve(G, X.T @ Z[1:])
    pending = np.concatenate([Z[-1], actions[len(states) - 1]])
    G_inv_with_pending = np.linalg.inv(G + np.outer(pending, pending))
    return W, G_inv_with_pending


def test_bayes_recursive_posterior_equals_batch_posterior():
    states, actions = _random_walk(25, 3, 2, seed=2)
    policy = BayesianDynamicsPolicy(_NoModel(), num_action_features=2, min_fit_obs=5, n_components=3)
    for s, a in zip(states, actions, strict=True):
        policy.observe(s, a)

    W, G_inv = _batch_posterior(policy, states, actions)
    assert np.allclose(policy._W, W, rtol=1e-6, atol=1e-8)
    assert np.allclose(policy._G_inv, G_inv, rtol=1e-6, atol=1e-8)


def test_bayes_window_posterior_uses_only_the_last_transitions():
    window = 6
    states, actions = _random_walk(25, 3, 2, seed=3)
    policy = BayesianDynamicsPolicy(_NoModel(), num_action_features=2, min_fit_obs=5, n_components=3,
                                    window=window)
    for s, a in zip(states, actions, strict=True):
        policy.observe(s, a)

    W, G_inv = _batch_posterior(policy, states[-(window + 1):], actions[-(window + 1):])
    assert np.allclose(policy._W, W, rtol=1e-6, atol=1e-8)
    assert np.allclose(policy._G_inv, G_inv, rtol=1e-6, atol=1e-8)


def test_bayes_learns_dynamics_after_an_underdetermined_start():
    """With 3 transitions and 5 unknowns per output the first fit is underdetermined.

    The posterior mean must keep learning afterwards: the one-step prediction
    error on a noise-free linear system falls far below the first fit's.
    """
    M = np.array([[0.8, 0.1, 0.0], [0.0, 0.7, 0.2], [0.1, 0.0, 0.6]])
    B = np.array([[1.0, 0.0], [0.5, 1.0], [0.0, 0.5]])
    states, actions = _linear_system(40, np.zeros(3), M, B, seed=4)
    policy = BayesianDynamicsPolicy(_NoModel(), num_action_features=2, min_fit_obs=4, n_components=3)

    def prediction_error():
        errors = []
        for s, a in zip(states[:-1], actions, strict=True):
            z_next = policy._A_r @ (s @ policy._U_r) + policy._B_r @ a
            errors.append(np.linalg.norm(policy._U_r @ z_next - (M @ s + B @ a)))
        return float(np.mean(errors))

    for i, (s, a) in enumerate(zip(states[:-1], actions, strict=True)):
        policy.observe(s, a)
        if i + 1 == policy.min_fit_obs:
            error_after_first_fit = prediction_error()
    error_at_end = prediction_error()

    assert error_at_end < 0.01 * error_after_first_fit


# ── Subspace residual uses an orthogonal projector (§13) ──

def test_subspace_residual_is_zero_inside_the_subspace_of_an_augmented_basis():
    states, actions = _random_walk(12, 3, 2, seed=5)
    rng = np.random.default_rng(6)
    policy = BayesianDynamicsPolicy(_NoModel(), num_action_features=2, min_fit_obs=5, n_components=2)
    for s, a in zip(states, actions, strict=True):
        policy.observe(s, a, uncertainty_vector=rng.uniform(0.0, 1.0, size=2))

    U_s = policy._U_r[:3]
    assert not np.allclose(U_s.T @ U_s, np.eye(U_s.shape[1]))   # the case §13 is about

    inside = U_s @ np.array([0.3, -1.2])
    assert policy._subspace_residual(inside) == pytest.approx(0.0, abs=1e-10)
    for _ in range(20):
        residual = policy._subspace_residual(rng.normal(size=3))
        assert 0.0 <= residual <= 1.0 + 1e-12


# ── Audit health vector: unreported checks are neutral (audit derivation §2.3) ──

def test_unreported_checks_neither_reward_nor_penalise_health():
    from results.policies import NEUTRAL_AUDIT_HEALTH, make_audit_health_vector
    from results.policies.state_space_audit import _health_score

    empty = make_audit_health_vector([])
    assert np.array_equal(empty, NEUTRAL_AUDIT_HEALTH)
    assert _health_score(empty) == 0.0

    perfect = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=float)
    assert _health_score(perfect) == 2.0


# ── Information bonus is the D-optimal gain log(1 + h) (§11.3) ──

def test_info_gain_is_log_one_plus_leverage():
    states, actions = _random_walk(6, 3, 2, seed=7)
    policy = BayesianDynamicsPolicy(_NoModel(), num_action_features=2, min_fit_obs=5, n_components=3)
    for s, a in zip(states, actions, strict=True):
        policy.observe(s, a)

    z = states[-1] @ policy._U_r
    for action in (np.array([0.1, 0.0]), np.array([10.0, -10.0])):
        x = np.concatenate([z, action])
        assert policy._info_gain(z, action) == pytest.approx(np.log1p(x @ policy._G_inv @ x))


def test_info_gain_still_ranks_candidates_under_a_vague_prior():
    """With the precision at the prior alone, h/(1+h) is ~1 for every candidate;
    log(1 + h) still separates candidates whose leverage differs tenfold."""
    states, actions = _random_walk(6, 3, 2, seed=8)
    policy = BayesianDynamicsPolicy(_NoModel(), num_action_features=2, min_fit_obs=5, n_components=3)
    for s, a in zip(states, actions, strict=True):
        policy.observe(s, a)
    policy._G_inv = np.eye(policy._G_inv.shape[0]) / policy.lambda_reg

    z = np.zeros(policy._U_r.shape[1])
    near, far = np.array([1.0, 0.0]), np.array([10.0, 0.0])      # leverage ratio 100
    h_near = np.expm1(policy._info_gain(z, near))
    h_far = np.expm1(policy._info_gain(z, far))

    assert h_near / (1 + h_near) > 0.99 and h_far / (1 + h_far) > 0.99
    assert policy._info_gain(z, far) - policy._info_gain(z, near) == pytest.approx(np.log(100), rel=1e-2)


# ── Run loops record the degradation of the observed state ──

@pytest.mark.parametrize("policy_class", [StateSpacePolicy, BayesianDynamicsPolicy])
def test_run_records_degradation_of_the_observed_state(policy_class):
    M = 0.9 * np.eye(2)
    B = np.array([[1.0], [0.5]])
    candidates = [np.array([-1.0]), np.array([0.0]), np.array([1.0])]
    policy = policy_class(_NoModel(), num_action_features=1, min_fit_obs=4, n_components=2,
                          degradation_fn=lambda s: float(np.sum(s)))
    state = {"s": np.array([5.0, -2.0])}

    def oracle(action):
        state["s"] = M @ state["s"] + B @ action
        return state["s"].copy()

    history = policy.run(np.array([5.0, -2.0]), candidates, oracle, n_iterations=8)

    for degradation, observed in zip(history["degradation"], history["observed_states"][1:], strict=True):
        assert degradation == float(np.sum(observed))


# ── Steering health: smooth and bounded (audit derivation §4.2) ──

def test_steering_health_is_bounded_decreasing_and_smooth():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "steer_state_space.py"
    spec = importlib.util.spec_from_file_location("steer_state_space", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    health = module.make_health_function(scale=2.0)

    values = [health([z, z]) for z in (-1000.0, -2.0, 0.0, 2.0, 1000.0)]
    assert all(-1.0 <= v <= 1.0 for v in values)
    assert values == sorted(values, reverse=True)            # more badness, less health
    assert health([0.0]) == 0.0
    assert health([1000.0]) > health([4.0]) - 0.05           # saturated, not clipped to a cliff
    assert health([4.0]) < health([2.0]) < health([0.0])     # still informative past k
