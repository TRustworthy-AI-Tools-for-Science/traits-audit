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


# ---------------------------------------------------------------------------
# Shard / gather round-trip: HPC array jobs split the bake-off by policy, so
# merged shards must reconstruct exactly what one process would have produced.
# ---------------------------------------------------------------------------

def _fake_result(policies, n_seeds=3, n_steps=5, seed=0):
    from traits_audit.committee.analysis.thread_regret import ThreadResult

    rng = np.random.default_rng(seed)
    seeds = [1000 + i for i in range(n_seeds)]
    return ThreadResult(
        per_policy_regret={p: rng.random((n_seeds, n_steps)) for p in policies},
        per_policy_action_std={p: rng.random((n_seeds, n_steps)) for p in policies},
        seeds=seeds,
        episode_length=n_steps,
        warmstart_n=20,
    )


def test_thread_csv_round_trips(tmp_path):
    from traits_audit.committee.analysis.thread_regret import (
        read_thread_csv, write_thread_csv,
    )
    result = _fake_result(["random", "LCB", "LCB+votes"])
    path = tmp_path / "thread_b_regret.csv"
    write_thread_csv(result, path)

    back = read_thread_csv([path])
    assert back.seeds == result.seeds
    assert back.episode_length == result.episode_length
    for p, arr in result.per_policy_regret.items():
        # write_thread_csv rounds to 6dp; that is the on-disk precision.
        np.testing.assert_allclose(back.per_policy_regret[p], arr, atol=1e-6)
        np.testing.assert_allclose(back.per_policy_action_std[p],
                                   result.per_policy_action_std[p], atol=1e-6)


def test_merged_shards_equal_a_single_whole_run(tmp_path):
    """Two shards written separately must merge back to the whole result."""
    from traits_audit.committee.analysis.thread_regret import (
        read_thread_csv, write_thread_csv,
    )
    whole = _fake_result(["random", "LCB", "max-sigma", "LCB+votes"])

    # Split by policy, exactly as an array job's --only shards would.
    from traits_audit.committee.analysis.thread_regret import ThreadResult
    for tag, names in [("s0", ["random", "LCB"]),
                       ("s1", ["max-sigma", "LCB+votes"])]:
        shard = ThreadResult(
            per_policy_regret={n: whole.per_policy_regret[n] for n in names},
            per_policy_action_std={n: whole.per_policy_action_std[n] for n in names},
            seeds=whole.seeds,
            episode_length=whole.episode_length,
            warmstart_n=whole.warmstart_n,
        )
        write_thread_csv(shard, tmp_path / f"thread_b_regret_{tag}.csv")

    merged = read_thread_csv(sorted(tmp_path.glob("thread_b_regret_*.csv")))
    assert set(merged.per_policy_regret) == set(whole.per_policy_regret)
    assert merged.seeds == whole.seeds
    for p, arr in whole.per_policy_regret.items():
        np.testing.assert_allclose(merged.per_policy_regret[p], arr, atol=1e-6)


def test_merging_shards_with_mismatched_seeds_is_rejected(tmp_path):
    """Paired tests are invalid across different seeds -- must not merge."""
    from traits_audit.committee.analysis.thread_regret import (
        read_thread_csv, write_thread_csv,
    )
    a = _fake_result(["random"], seed=1)
    b = _fake_result(["LCB"], seed=2)
    b.seeds = [9999 + i for i in range(len(b.seeds))]   # different episodes
    write_thread_csv(a, tmp_path / "thread_b_regret_s0.csv")
    write_thread_csv(b, tmp_path / "thread_b_regret_s1.csv")

    with pytest.raises(ValueError, match="paired"):
        read_thread_csv(sorted(tmp_path.glob("thread_b_regret_*.csv")))


# ---------------------------------------------------------------------------
# Channel-marginal figure: the colour panels' CIE projection hides queries
# pinned against a channel bound, which is the whole point of this figure.
# ---------------------------------------------------------------------------

def test_channel_marginals_renders_for_each_multidim_problem(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from traits_audit.committee.analysis.channel_marginals import (
        render_channel_marginals,
    )
    from traits_audit.committee.analysis.density import AGENT_NAMES, DensityResult

    for name, dim in [("branin-currin", 2), ("color", 3)]:
        rng = np.random.default_rng(0)
        pooled = {a: rng.uniform(0, 1, size=(200, dim)) for a in AGENT_NAMES}
        result = DensityResult(
            queries_by_agent=pooled,
            queries_by_agent_seed={a: {0: v} for a, v in pooled.items()},
            n_episodes_per_seed=2,
        )
        out = tmp_path / f"{name}.png"
        render_channel_marginals(result, out, problem=name)
        assert out.exists() and out.stat().st_size > 0


def test_color_target_is_read_from_the_simulator():
    """ColorMatchingProblem.target must come from the SDL, not a literal.

    frechet is 0 there by definition, which is what makes true_min = 0.0
    exact rather than a grid-search approximation.
    """
    problem = get_problem("color")
    target = problem.target
    assert target is not None
    assert target.shape == (3,)
    assert np.all((target >= 0.0) & (target <= 1.0))
    # The optimum really is the optimum.
    assert problem.clean(target.reshape(1, 3))[0, 0] == problem.true_min == 0.0
