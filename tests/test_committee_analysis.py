"""Smoke tests for the post-training analysis pipeline.

These tests are intentionally tiny (1 episode, 20 steps) so they run in <1s
and don't need any trained models on disk. The goal is to catch wiring
regressions, not to validate the science.

Skipped if stable-baselines3 isn't installed.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gymnasium")
sb3 = pytest.importorskip("stable_baselines3")


def test_random_rollout_correlation_smoke():
    """End-to-end: random rollouts -> KxK matrix -> clustering."""
    from traits_audit.committee.analysis.correlation import (
        cluster_agents,
        random_rollout_correlation,
    )
    from traits_audit.committee.rewards import REWARD_REGISTRY

    k = len(REWARD_REGISTRY)
    result = random_rollout_correlation(
        n_episodes=2, episode_length=20, seed=123,
    )
    assert result.matrix.shape == (k, k)
    # Diagonal is 1 (self-correlation), or 0 for a reward that stayed constant
    # over this short rollout — sparse ones like MahalanobisOOD often do, and
    # _correlation_matrix maps their NaN rows to 0.
    diag = np.diag(result.matrix)
    assert np.all(np.isclose(diag, 1.0, atol=1e-9) | (diag == 0.0))
    # Off-diagonal symmetry.
    np.testing.assert_allclose(result.matrix, result.matrix.T, atol=1e-9)

    labels, n_clusters = cluster_agents(result.matrix, threshold=0.5)
    assert labels.shape == (k,)
    assert 1 <= n_clusters <= k


def test_score_trace_matches_direct_reward_call():
    """score_trace must be equivalent to running the reward computer manually.

    This is the contract that lets the correlation matrix be trusted: each
    column of the matrix is the same per-step reward stream the env would
    have produced at training time.
    """
    from traits_audit.committee.analysis.rollouts import (
        random_policy,
        run_rollout,
        score_trace,
    )
    from traits_audit.committee.rewards import REWARD_REGISTRY

    rng = np.random.default_rng(7)
    trace = run_rollout(random_policy(rng), seed=7, episode_length=15)
    scored = score_trace(trace)

    # Spot-check CRPS — the (y, mu, sigma) path shared by the check-based
    # computers. The signal-based ones are covered by the test below.
    rc = REWARD_REGISTRY["CRPS"]()
    expected = []
    w = trace.warmstart_n
    for t in range(trace.n_steps):
        i = w + t
        expected.append(rc.reward(
            trace.y_obs[:i], trace.mu_hist[:i], trace.sigma_hist[:i],
            trace.y_obs[:i + 1], trace.mu_hist[:i + 1], trace.sigma_hist[:i + 1],
        ))
    np.testing.assert_allclose(scored["CRPS"], np.asarray(expected), atol=1e-10)


# MahalanobisOOD also reads the extras but is left out: its check draws an
# unseeded bootstrap (random_state=None) for the OOD threshold, so two calls on
# the same data need not agree exactly.
@pytest.mark.parametrize("name", ["UncertaintyEvolution", "UncertaintyAnomaly"])
def test_score_trace_reproduces_env_reward_for_signal_rewards(name):
    """Offline scoring of the signal-based rewards must match what env.step()
    returned at training time. They read x / sigma-series extras, so a
    score_trace that drops those extras silently scores them as all-zero.
    """
    from traits_audit.committee.analysis.rollouts import RolloutTrace, score_trace
    from traits_audit.committee.env import CommitteeEnv
    from traits_audit.committee.rewards import REWARD_REGISTRY

    env = CommitteeEnv(reward_computer=REWARD_REGISTRY[name](), episode_length=40)
    env.reset(seed=11)
    rng = np.random.default_rng(11)
    online, x_q = [], []
    for _ in range(40):
        _obs, r, _term, _trunc, info = env.step(rng.uniform(0.0, 1.0, size=1))
        online.append(r)
        x_q.append(info["x_q"])

    trace = RolloutTrace(
        x_obs=env.x_obs,
        y_obs=env.y_obs,
        mu_hist=np.asarray(env._mu_history, dtype=float),
        sigma_hist=np.asarray(env._sigma_history, dtype=float),
        x_queries=np.asarray(x_q, dtype=float),
        warmstart_n=env.warmstart_n,
    )
    np.testing.assert_allclose(score_trace(trace)[name], online, atol=1e-10)
