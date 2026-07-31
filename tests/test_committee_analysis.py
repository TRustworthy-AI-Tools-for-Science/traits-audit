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
    """End-to-end: random rollouts -> 9x9 matrix -> clustering."""
    from traits_audit.committee.analysis.correlation import (
        cluster_agents,
        random_rollout_correlation,
    )

    result = random_rollout_correlation(
        n_episodes=2, episode_length=20, seed=123,
    )
    assert result.matrix.shape == (9, 9)
    # Diagonal must be 1 (self-correlation) modulo tiny float noise.
    np.testing.assert_allclose(np.diag(result.matrix), 1.0, atol=1e-9)
    # Off-diagonal symmetry.
    np.testing.assert_allclose(result.matrix, result.matrix.T, atol=1e-9)

    labels, n_clusters = cluster_agents(result.matrix, threshold=0.5)
    assert labels.shape == (9,)
    assert 1 <= n_clusters <= 9


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

    # Spot-check CRPS only — the path is shared across all 9 computers.
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
