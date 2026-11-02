"""Branin-Currin with BOTH objectives audited (``branin-currin-mo``).

The committee env historically reduced every problem to one audited objective
(``_y_raw[:, 0] / y_scale``), so on Branin-Currin it fitted a surrogate to
Branin alone and discarded Currin. ``branin-currin-mo`` is a sibling problem
that keeps both: two surrogates, two uncertainty streams, and each of the 15
audit rewards scored per objective and averaged.

The single-objective ``branin-currin`` arm must stay bit-identical — 75 trained
SAC models depend on its 479-dim observation, and SB3 only raises a shape
mismatch at ``SAC.load``, i.e. during analysis, long after the training run
that produced them.
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Regression guards for the existing arms.
# ---------------------------------------------------------------------------

def test_single_objective_state_dims_unchanged():
    """The 75 saved models depend on these exact numbers."""
    from traits_audit.committee.problems import get_problem

    assert get_problem("forrester").state_dim == 414
    assert get_problem("branin-currin").state_dim == 479
    assert get_problem("color").state_dim == 1092


def test_branin_currin_still_audits_one_objective():
    """n_objectives must be explicit, not inferred from objective_names.

    BraninCurrinProblem already *names* two objectives while auditing only
    Branin; inferring the count would silently flip it to 932 dims.
    """
    from traits_audit.committee.problems import get_problem

    p = get_problem("branin-currin")
    assert p.objective_names == ("branin", "currin")
    assert p.n_objectives == 1
    np.testing.assert_allclose(p.y_scales, [50.0])


# ---------------------------------------------------------------------------
# The new problem.
# ---------------------------------------------------------------------------

def test_branin_currin_mo_problem_attributes():
    pytest.importorskip("botorch")
    from traits_audit.committee.problems import get_problem

    p = get_problem("branin-currin-mo")
    assert p.n_objectives == 2
    # 2 obj x 2 (mu, sigma) x 225 grid + (1 + 3*2) summary + 5^2 histogram
    assert p.state_dim == 932 == 4 * 225 + 7 + 25
    # Per-objective scaling is load-bearing: observations are not normalised
    # before SAC sees them, so a shared scale would leave Currin's mu/sigma
    # block ~20x smaller than Branin's and effectively invisible.
    np.testing.assert_allclose(p.y_scales, [50.0, 2.6])
    assert p.dim == 2 and p.degree == 5


def test_branin_currin_mo_env():
    pytest.importorskip("botorch")
    from traits_audit.committee.env import CommitteeEnv
    from traits_audit.committee.problems import get_problem
    from traits_audit.committee.rewards import REWARD_REGISTRY

    p = get_problem("branin-currin-mo")
    env = CommitteeEnv(reward_computer=REWARD_REGISTRY["CRPS"](), problem=p)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape == (932,)
    assert env.action_space.shape == (2,)

    obs, r, _, _, info = env.step(np.array([0.3, 0.7], dtype=np.float32))
    assert np.isfinite(obs).all() and np.isfinite(r)
    assert env.x_obs.shape == (21, 2) and env.y_raw_obs.shape == (21, 2)
    # Both objectives kept, each divided by its own scale.
    assert env.y_obs.shape == (21, 2)
    np.testing.assert_allclose(env.y_obs, env.y_raw_obs / p.y_scales)
    np.testing.assert_allclose(info["x_q"], [0.3, 0.7], rtol=1e-6)


def test_mo_surrogate_matches_two_independent_fits():
    """_MultiSurrogate must be exactly two separate fits, with no axis mixing."""
    pytest.importorskip("botorch")
    from traits_audit.committee.problems import get_problem

    p = get_problem("branin-currin-mo")
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, size=(40, 2))
    Y = p.clean(x)[:, :2] / p.y_scales

    multi = p.make_surrogate(degree=3, n_estimators=8, std_scale=1.0,
                             rng=np.random.default_rng(5))
    multi.fit(x, Y)
    g = rng.uniform(0, 1, size=(25, 2))
    mu, sigma = multi.predict(g)
    assert mu.shape == sigma.shape == (25, 2)

    for j, sub in enumerate(multi.surrogates):
        mu_j, sigma_j = sub.predict(g)
        np.testing.assert_allclose(mu[:, j], mu_j)
        np.testing.assert_allclose(sigma[:, j], sigma_j)


# ---------------------------------------------------------------------------
# Reward reduction.
# ---------------------------------------------------------------------------

def test_mo_reward_average_reduces_to_single_objective():
    """With identical objectives, averaging must equal the 1-objective reward.

    Catches a sum-vs-mean slip and the MahalanobisOOD double-count in one
    assertion: its check reads only the queried-x geometry, so it is
    objective-independent and summing would multiply it by n_objectives.
    """
    from traits_audit.committee.env import CommitteeEnv
    from traits_audit.committee.rewards import REWARD_REGISTRY

    rng = np.random.default_rng(0)
    n = 40
    y = rng.normal(0, 1, n)
    mu = y + rng.normal(0, 0.1, n)
    sg = np.abs(rng.normal(0.4, 0.05, n))
    Y, MU, SG = (np.stack([a, a], axis=1) for a in (y, mu, sg))

    class _P:
        def __init__(self, k):
            self.n_objectives = k

    for name, cls in REWARD_REGISTRY.items():
        rc = cls()
        one = CommitteeEnv.__new__(CommitteeEnv)
        one.reward_computer, one.problem = rc, _P(1)
        two = CommitteeEnv.__new__(CommitteeEnv)
        two.reward_computer, two.problem = rc, _P(2)

        r1 = one._compute_reward(y[:-1], mu[:-1], sg[:-1], y, mu, sg, None, None)
        r2 = two._compute_reward(Y[:-1], MU[:-1], SG[:-1], Y, MU, SG, None, None)
        assert r1 == pytest.approx(r2, abs=1e-12), name


def test_mo_reward_differs_when_objectives_differ():
    """Sanity check the other direction: the second stream must matter."""
    from traits_audit.committee.env import CommitteeEnv
    from traits_audit.committee.rewards import REWARD_REGISTRY

    rng = np.random.default_rng(3)
    n = 40
    y0 = rng.normal(0, 1, n)
    y1 = rng.normal(0, 1, n)          # an unrelated second objective
    mu = np.stack([y0 + 0.05, y1 * 0.5], axis=1)
    sg = np.abs(rng.normal(0.4, 0.05, (n, 2)))
    Y = np.stack([y0, y1], axis=1)

    class _P:
        n_objectives = 2

    env = CommitteeEnv.__new__(CommitteeEnv)
    env.reward_computer, env.problem = REWARD_REGISTRY["CRPS"](), _P()
    both = env._compute_reward(Y[:-1], mu[:-1], sg[:-1], Y, mu, sg, None, None)

    env.problem = type("P1", (), {"n_objectives": 1})()
    only0 = env._compute_reward(Y[:-1, 0], mu[:-1, 0], sg[:-1, 0],
                                Y[:, 0], mu[:, 0], sg[:, 0], None, None)
    assert both != pytest.approx(only0, abs=1e-12)


# ---------------------------------------------------------------------------
# Hypervolume scoring.
# ---------------------------------------------------------------------------

def test_mo_hypervolume_matches_mobo_demo():
    """The deliberate 6-line copy must track the demo's original."""
    pytest.importorskip("botorch")
    from traits_audit._mobo_demo import _REF_POINT, _hypervolume
    from traits_audit.committee.problems import get_problem

    p = get_problem("branin-currin-mo")
    np.testing.assert_allclose(p.ref_point, _REF_POINT.numpy())

    rng = np.random.default_rng(0)
    for _ in range(3):
        Y = p.clean(rng.uniform(0, 1, size=(20, 2)))
        assert p.hypervolume(Y) == pytest.approx(_hypervolume(Y), rel=1e-9)


def test_mo_ref_point_dominates_part_of_the_objective_space():
    """A clean value outside the ref point contributes zero HV, silently."""
    pytest.importorskip("botorch")
    from traits_audit.committee.problems import _grid, get_problem

    p = get_problem("branin-currin-mo")
    Y = p.clean(_grid(40, 2))[:, :2]
    assert np.any(np.all(Y < p.ref_point, axis=1))
    assert p.hv_max > 0


def test_single_objective_problems_have_no_ref_point():
    """hypervolume() must refuse rather than invent a reference."""
    from traits_audit.committee.problems import get_problem

    p = get_problem("forrester")
    assert p.ref_point is None
    with pytest.raises(ValueError, match="ref_point"):
        p.hypervolume(np.zeros((3, 2)))


def test_mo_trace_to_regret_is_monotone_non_increasing():
    """Hypervolume only grows, so log10(hv_max - HV) only shrinks."""
    pytest.importorskip("botorch")
    from traits_audit.committee.analysis.regret import _trace_to_regret
    from traits_audit.committee.analysis.rollouts import random_policy, run_rollout
    from traits_audit.committee.problems import get_problem

    p = get_problem("branin-currin-mo")
    trace = run_rollout(random_policy(np.random.default_rng(1)), seed=0,
                        episode_length=10, problem=p)
    sr = _trace_to_regret(trace, 20, 10, p)
    assert sr.shape == (10,)
    assert np.all(np.isfinite(sr))
    assert np.all(np.diff(sr) <= 1e-9)


# ---------------------------------------------------------------------------
# Downstream analysis paths.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("policy_name", ["random", "lcb", "max_sigma"])
def test_mo_baseline_policies_return_one_valid_point(policy_name):
    """The grid-search baselines must reduce (n_grid, n_obj) to a scalar score.

    Without the reduction, argmin over a 2-column array returns a flat index
    and the policy indexes the grid out of bounds.
    """
    pytest.importorskip("botorch")
    from traits_audit.committee.analysis import rollouts as ro
    from traits_audit.committee.problems import get_problem

    factory = {"random": lambda: ro.random_policy(np.random.default_rng(0)),
               "lcb": ro.lcb_policy,
               "max_sigma": ro.max_sigma_policy}[policy_name]
    trace = ro.run_rollout(factory(), seed=0, episode_length=6,
                           problem=get_problem("branin-currin-mo"))
    assert trace.x_queries.shape == (6, 2)
    assert np.all((trace.x_queries >= 0.0) & (trace.x_queries <= 1.0))


def test_reduce_objectives_passes_1d_through_unchanged():
    from traits_audit.committee.analysis.rollouts import _reduce_objectives

    a = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(_reduce_objectives(a), a)
    b = np.array([[1.0, 10.0], [2.0, 20.0]])
    np.testing.assert_allclose(_reduce_objectives(b), [11.0, 22.0])


def test_mo_score_trace_is_finite():
    """score_trace must mirror the env's averaging, not feed checks 2-D input.

    A 2-D array would not raise -- it would return a plausible but wrong
    number, quietly corrupting the cross-agent correlation matrices.
    """
    pytest.importorskip("botorch")
    from traits_audit.committee.analysis.rollouts import (
        random_policy, run_rollout, score_trace,
    )
    from traits_audit.committee.problems import get_problem

    trace = run_rollout(random_policy(np.random.default_rng(2)), seed=1,
                        episode_length=8, problem=get_problem("branin-currin-mo"))
    scored = score_trace(trace)
    assert len(scored) == 15
    for name, arr in scored.items():
        assert arr.shape == (8,), name
        assert np.all(np.isfinite(arr)), name


def test_mo_correlation_matrix_is_well_formed():
    pytest.importorskip("botorch")
    from traits_audit.committee.analysis.correlation import random_rollout_correlation

    result = random_rollout_correlation(n_episodes=2, episode_length=8, seed=0,
                                        problem="branin-currin-mo")
    M = result.matrix
    assert M.shape == (15, 15)
    assert np.all(np.isfinite(M))
    np.testing.assert_allclose(M, M.T, atol=1e-9)
    assert np.all(np.abs(M) <= 1.0 + 1e-9)
    # Diagonal is 1 except for agents whose reward stream is constant over
    # this short episode (zero variance -> correlation undefined -> 0). That
    # is pre-existing behaviour, not multi-objective specific: the
    # single-objective arm has the same at this episode length.
    diag = np.diag(M)
    assert np.all((np.abs(diag - 1.0) < 1e-9) | (diag == 0.0))
    assert (diag == 1.0).sum() >= 13
