"""Tests for the committee benchmark problems and the d-dimensional env.

The Forrester tests pin the contract that existing trained models rely on:
the default env must produce exactly the observations the original 1-D
implementation did. Skipped if gymnasium isn't installed.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gymnasium")


def test_forrester_env_matches_legacy_1d_computation():
    """First observation == hand-rolled legacy computation, bit for bit."""
    from traits_audit._cal_demo import BootstrapSurrogate, oracle
    from traits_audit.committee.env import STATE_DIM, CommitteeEnv
    from traits_audit.committee.rewards import REWARD_REGISTRY

    env = CommitteeEnv(reward_computer=REWARD_REGISTRY["CRPS"]())
    obs, _ = env.reset(seed=5)
    assert obs.shape == (STATE_DIM,) == (414,)
    assert env.action_space.shape == (1,)

    rng = np.random.default_rng(5)
    x = rng.uniform(0.0, 1.0, size=20)
    y = oracle(x, rng)
    surrogate = BootstrapSurrogate(
        degree=5, n_estimators=30, std_scale=0.7, rng=np.random.default_rng(5 + 2**31),
    )
    surrogate.fit(x, y)
    mu, sigma = surrogate.predict(np.linspace(0.0, 1.0, 200))
    np.testing.assert_array_equal(obs[:200], mu.astype(np.float32))
    np.testing.assert_array_equal(obs[200:400], sigma.astype(np.float32))
    np.testing.assert_array_equal(env.x_obs, x)


def test_poly_surrogate_reduces_to_bootstrap_surrogate_in_1d():
    from traits_audit._cal_demo import BootstrapSurrogate
    from traits_audit.committee.problems import PolyBootstrapSurrogate

    rng = np.random.default_rng(0)
    x, y = rng.uniform(size=25), rng.normal(size=25)
    legacy = BootstrapSurrogate(degree=5, n_estimators=10, rng=np.random.default_rng(1)).fit(x, y)
    poly = PolyBootstrapSurrogate(dim=1, degree=5, n_estimators=10, rng=np.random.default_rng(1)).fit(x, y)
    grid = np.linspace(0, 1, 50)
    for a, b in zip(legacy.predict(grid), poly.predict(grid)):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("dim, degree, n_terms", [(2, 5, 21), (3, 3, 20)])
def test_poly_surrogate_feature_count(dim, degree, n_terms):
    from traits_audit.committee.problems import PolyBootstrapSurrogate

    assert len(PolyBootstrapSurrogate(dim=dim, degree=degree)._exponents) == n_terms


def test_branin_currin_env():
    pytest.importorskip("botorch")
    from traits_audit.committee.env import CommitteeEnv
    from traits_audit.committee.problems import get_problem
    from traits_audit.committee.rewards import REWARD_REGISTRY

    env = CommitteeEnv(reward_computer=REWARD_REGISTRY["CRPS"](), problem="branin-currin")
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape == (2 * 225 + 4 + 25,)
    assert env.action_space.shape == (2,)
    obs, r, _, _, info = env.step(np.array([0.3, 0.7], dtype=np.float32))
    assert np.isfinite(obs).all() and np.isfinite(r)
    assert env.x_obs.shape == (21, 2) and env.y_raw_obs.shape == (21, 2)
    # Rewards and the surrogate see Branin only, scaled.
    np.testing.assert_allclose(env.y_obs, env.y_raw_obs[:, 0] / 50.0)
    np.testing.assert_allclose(info["x_q"], [0.3, 0.7], rtol=1e-6)
    # Branin's global minimum (0.398) at one of its three optima.
    clean = get_problem("branin-currin").clean(np.array([[0.5428, 0.1517]]))
    assert clean[0, 0] == pytest.approx(0.3979, abs=1e-3)


def _color_problem_or_skip():
    from traits_audit.committee.problems import get_problem

    problem = get_problem("color")
    try:
        problem.rgb_max  # triggers the simulator import
    except ImportError:
        pytest.skip("self-driving-lab-demo not installed")
    return problem


def test_color_env():
    problem = _color_problem_or_skip()
    from traits_audit.committee.env import CommitteeEnv
    from traits_audit.committee.rewards import REWARD_REGISTRY

    env = CommitteeEnv(reward_computer=REWARD_REGISTRY["CRPS"](), problem=problem)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape == (2 * 512 + 4 + 64,)
    assert env.action_space.shape == (3,)
    obs, r, _, _, _ = env.step(np.array([0.5, 0.5, 0.5], dtype=np.float32))
    assert np.isfinite(obs).all() and np.isfinite(r)
    # Frechet distance is 0 at the simulator's target inputs.
    target = problem._sdl.get_target_inputs()
    x_target = np.array([[target["R"], target["G"], target["B"]]]) / problem.rgb_max
    assert problem.clean(x_target)[0, 0] == pytest.approx(0.0, abs=1e-9)


def test_mahalanobis_reward_handles_multidimensional_queries():
    from traits_audit.committee.rewards import REWARD_REGISTRY

    rng = np.random.default_rng(0)
    x_before = rng.uniform(0.0, 0.2, size=(40, 2))
    x_after = np.vstack([x_before, [[1.0, 1.0]]])  # far outside the cluster
    sigma = np.ones(41)
    r = REWARD_REGISTRY["MahalanobisOOD"]().reward(
        None, None, None, None, None, None,
        x_before=x_before, x_after=x_after,
        sigma_series_before=sigma[:40], sigma_series_after=sigma,
    )
    assert r < 0  # one more OOD query in the trailing window -> penalised
