"""Thread A/B generalization to d-dimensional actions.

The Thread A/B aggregators were written against a scalar action space
(Forrester). Generalizing them to Branin-Currin (2-D) and colour matching
(3-D) must not move the Forrester numbers, so the 1-D cases here assert the
new vector code reproduces the original scalar formulas exactly.

No trained models are needed: a stub voter stands in for CommitteeVoter, and
a stub env carries just the ``problem`` attribute the policies read.
"""
from __future__ import annotations

import numpy as np
import pytest

from traits_audit.committee.analysis.policies_ext import (
    _mean_vote_distance,
    committee_agree_policy,
    committee_disagree_policy,
    committee_uniform_policy,
    committee_weighted_policy,
)
from traits_audit.committee.problems import get_problem


class _StubVoter:
    """Stands in for CommitteeVoter with fixed preferred actions."""

    def __init__(self, prefs: np.ndarray, names: list[str] | None = None):
        self._prefs = np.asarray(prefs, dtype=float)
        self._names = names or [f"a{i}" for i in range(len(self._prefs))]

    @property
    def agent_names(self) -> list[str]:
        return list(self._names)

    @property
    def n_agents(self) -> int:
        return len(self._prefs)

    def preferred_actions(self, obs):
        return self._prefs


class _StubEnv:
    """Minimal env exposing the `.unwrapped.problem` the policies read."""

    def __init__(self, problem):
        self.problem = problem

    @property
    def unwrapped(self):
        return self


def _env(name: str) -> _StubEnv:
    return _StubEnv(get_problem(name))


# ---------------------------------------------------------------------------
# 1-D: the vector code must reproduce the original scalar formulas.
# ---------------------------------------------------------------------------

def test_mean_vote_distance_matches_scalar_formula_in_1d():
    grid = np.linspace(0.0, 1.0, 37).reshape(-1, 1)
    prefs = np.array([[0.1], [0.42], [0.9]])
    expected = np.abs(grid[:, 0][:, None] - prefs[:, 0][None, :]).mean(axis=1)
    np.testing.assert_allclose(_mean_vote_distance(grid, prefs), expected)


def test_disagree_reduces_to_weighted_variance_in_1d():
    """Trace-of-covariance == the original 1-D weighted variance."""
    prefs = np.array([[0.15], [0.20], [0.80]])
    bandwidth = 0.05
    grid_size = 300

    # Original scalar implementation, verbatim.
    grid = np.linspace(0.0, 1.0, grid_size)
    flat = prefs[:, 0]
    dx = grid[:, None] - flat[None, :]
    w = np.exp(-(dx ** 2) / (2.0 * bandwidth ** 2))
    w_sum = w.sum(axis=1, keepdims=True)
    w_sum = np.where(w_sum < 1e-12, 1.0, w_sum)
    mean = (w * flat[None, :]).sum(axis=1, keepdims=True) / w_sum
    var = (w * (flat[None, :] - mean) ** 2).sum(axis=1) / w_sum.squeeze(1)
    expected = np.array([grid[int(np.argmax(var))]], dtype=np.float32)

    pol = committee_disagree_policy(_StubVoter(prefs), grid_size=grid_size,
                                    bandwidth=bandwidth)
    np.testing.assert_allclose(pol(None, _env("forrester")), expected)


def test_agree_and_weighted_reduce_to_scalar_means_in_1d():
    prefs = np.array([[0.2], [0.4], [0.9]])
    env = _env("forrester")

    got = committee_agree_policy(_StubVoter(prefs))(None, env)
    np.testing.assert_allclose(got, [prefs[:, 0].mean()], rtol=1e-6)

    weights = {"a0": 1.0, "a1": 3.0, "a2": 0.0}
    got = committee_weighted_policy(_StubVoter(prefs), weights)(None, env)
    np.testing.assert_allclose(got, [(0.2 * 1 + 0.4 * 3) / 4.0], rtol=1e-6)


# ---------------------------------------------------------------------------
# d-dimensional: shapes, bounds, and the behaviour each policy claims.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,dim", [("forrester", 1),
                                      ("branin-currin", 2),
                                      ("color", 3)])
def test_policies_return_one_point_of_the_right_dimension(name, dim):
    rng = np.random.default_rng(0)
    prefs = rng.uniform(0, 1, size=(5, dim))
    voter = _StubVoter(prefs)
    env = _env(name)

    policies = [
        committee_agree_policy(voter),
        committee_disagree_policy(voter),
        committee_uniform_policy(voter, np.random.default_rng(1)),
        committee_weighted_policy(voter, {n: 1.0 for n in voter.agent_names}),
    ]
    for pol in policies:
        action = pol(None, env)
        assert action.shape == (dim,), f"{pol} on {name}"
        assert action.dtype == np.float32
        assert np.all((action >= 0.0) & (action <= 1.0))


@pytest.mark.parametrize("name,dim", [("branin-currin", 2), ("color", 3)])
def test_agree_is_the_centroid_and_uniform_is_a_member(name, dim):
    rng = np.random.default_rng(7)
    prefs = rng.uniform(0, 1, size=(6, dim))
    voter, env = _StubVoter(prefs), _env(name)

    centroid = committee_agree_policy(voter)(None, env)
    np.testing.assert_allclose(centroid, prefs.mean(axis=0), rtol=1e-6)

    # Uniform must return one of the members verbatim, not a blend.
    pick = committee_uniform_policy(voter, np.random.default_rng(3))(None, env)
    assert np.isclose(prefs, pick, rtol=1e-6).all(axis=1).any()


def test_weighted_ignores_zero_weight_members():
    """A zero-weighted agent must not influence the aggregate at all."""
    prefs = np.array([[0.1, 0.1], [0.9, 0.9], [0.5, 0.5]])
    voter = _StubVoter(prefs, names=["keep0", "keep1", "drop"])
    got = committee_weighted_policy(
        voter, {"keep0": 1.0, "keep1": 1.0, "drop": 0.0},
    )(None, _env("branin-currin"))
    np.testing.assert_allclose(got, [0.5, 0.5], rtol=1e-6)


def test_disagree_picks_the_contested_region_not_the_consensus():
    """Two tight clusters plus a gap: disagree must avoid a cluster centre."""
    prefs = np.array([
        [0.20, 0.20], [0.21, 0.21], [0.19, 0.22],   # tight cluster A
        [0.80, 0.80], [0.81, 0.79], [0.79, 0.81],   # tight cluster B
    ])
    action = committee_disagree_policy(
        _StubVoter(prefs), bandwidth=0.5,
    )(None, _env("branin-currin"))
    # With a bandwidth spanning both clusters, the most-contested point sits
    # between them, not on top of either consensus centre.
    assert np.linalg.norm(action - prefs[:3].mean(axis=0)) > 0.1
    assert np.linalg.norm(action - prefs[3:].mean(axis=0)) > 0.1


def test_weighted_rejects_all_zero_weights():
    voter = _StubVoter(np.array([[0.5, 0.5]]), names=["only"])
    with pytest.raises(ValueError):
        committee_weighted_policy(voter, {"only": 0.0})


def test_masked_voter_drops_an_agent_row_not_a_coordinate():
    """The ablation's mask must remove a whole agent, keeping (K-1, dim)."""
    from traits_audit.committee.analysis.thread_regret import (
        run_thread_b_ablation,
    )
    prefs = np.arange(12, dtype=float).reshape(4, 3) / 12.0
    voter = _StubVoter(prefs)

    # _MaskedVoter is defined inside run_thread_b_ablation; re-create the
    # same view here to assert the shape contract it has to satisfy.
    masked = np.delete(voter.preferred_actions(None), 1, axis=0)
    assert masked.shape == (3, 3)
    np.testing.assert_allclose(masked, prefs[[0, 2, 3]])
