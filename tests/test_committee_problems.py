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


# -- true_min / regret exactness -------------------------------------------

def test_true_min_exact_values():
    """Branin-Currin and color's true_min are exact literature/analytic
    values (verified externally via multistart optimisation and the
    simulator's own zero-distance-at-target property), not grid-search
    approximations — see the comments on each Problem subclass."""
    from traits_audit.committee.problems import get_problem

    assert get_problem("forrester").true_min == pytest.approx(-6.02074005501676)
    assert get_problem("branin-currin").true_min == pytest.approx(0.3978873577297666, abs=1e-9)
    assert get_problem("color").true_min == 0.0


def test_forrester_true_min_matches_regret_module_constant():
    """ForresterProblem.true_min must stay bit-identical to regret.py's
    module-level FORRESTER_TRUE_MIN, which thread_regret.py still imports
    directly and independently."""
    from traits_audit.committee.analysis.regret import FORRESTER_TRUE_MIN
    from traits_audit.committee.problems import ForresterProblem

    assert ForresterProblem().true_min == FORRESTER_TRUE_MIN


def test_trace_to_regret_matches_legacy_forrester_computation():
    """The generalized _trace_to_regret(..., problem) must reproduce the
    original hardcoded-Forrester formula bit for bit."""
    from traits_audit.committee.analysis.regret import (
        FORRESTER_TRUE_MIN, _forrester_clean, _trace_to_regret,
    )
    from traits_audit.committee.analysis.rollouts import random_policy, run_rollout
    from traits_audit.committee.problems import ForresterProblem

    trace = run_rollout(random_policy(np.random.default_rng(3)), seed=3, episode_length=25)
    new = _trace_to_regret(trace, warmstart_n=20, episode_length=25, problem=ForresterProblem())

    f_clean = _forrester_clean(trace.x_obs)
    old = np.zeros(25)
    for t in range(25):
        cutoff = 20 + t + 1
        old[t] = float(np.min(f_clean[:cutoff])) - FORRESTER_TRUE_MIN
    np.testing.assert_array_equal(old, new)


@pytest.mark.parametrize("problem_name", ["branin-currin", "color"])
def test_trace_to_regret_nonnegative_and_monotone(problem_name):
    if problem_name == "color":
        _color_problem_or_skip()
    else:
        pytest.importorskip("botorch")
    from traits_audit.committee.analysis.regret import _trace_to_regret
    from traits_audit.committee.analysis.rollouts import random_policy, run_rollout
    from traits_audit.committee.problems import get_problem

    problem = get_problem(problem_name)
    trace = run_rollout(
        random_policy(np.random.default_rng(1)), seed=1, episode_length=12, problem=problem,
    )
    sr = _trace_to_regret(trace, warmstart_n=20, episode_length=12, problem=problem)
    assert np.all(sr >= -1e-9)
    assert np.all(np.diff(sr) <= 1e-9)  # running min can only improve


# -- rollout/correlation/density generalize to any dim ---------------------

@pytest.mark.parametrize("policy_name", ["random", "lcb", "max_sigma"])
@pytest.mark.parametrize("problem_name", ["forrester", "branin-currin", "color"])
def test_baseline_policies_finite_on_every_problem(problem_name, policy_name):
    if problem_name == "color":
        _color_problem_or_skip()
    elif problem_name == "branin-currin":
        pytest.importorskip("botorch")
    from traits_audit.committee.analysis.rollouts import (
        lcb_policy, max_sigma_policy, random_policy, run_rollout, score_trace,
    )

    policy = {
        "random": random_policy(np.random.default_rng(1)),
        "lcb": lcb_policy(),
        "max_sigma": max_sigma_policy(),
    }[policy_name]
    trace = run_rollout(policy, seed=1, episode_length=8, problem=problem_name)
    scored = score_trace(trace)
    assert all(np.isfinite(v).all() for v in scored.values())


def _write_fake_tb_scalar(tb_dir, run_name: str, tag: str, steps, values):
    """Write a minimal real TB event file so learning_curves.py's
    EventAccumulator can read it back — avoids depending on a real SAC run."""
    from tensorboard.summary.writer.event_file_writer import EventFileWriter
    from tensorboard.compat.proto import event_pb2, summary_pb2

    run_dir = tb_dir / run_name / "SAC_1"
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = EventFileWriter(str(run_dir))
    for step, value in zip(steps, values):
        summary = summary_pb2.Summary(
            value=[summary_pb2.Summary.Value(tag=tag, simple_value=float(value))]
        )
        writer.add_event(event_pb2.Event(step=int(step), summary=summary))
    writer.close()


def test_load_learning_curves_skips_missing_agents_instead_of_crashing():
    """A registry-wide subcommand must not go all-or-nothing on 15 agents'
    TB logs — one missing (still-training) agent used to crash the whole
    figure; it should now be skipped with the rest still plotted."""
    pytest.importorskip("tensorboard")
    import tempfile
    from pathlib import Path
    from traits_audit.committee.analysis.learning_curves import load_learning_curves

    with tempfile.TemporaryDirectory() as d:
        tb_dir = Path(d)
        steps = list(range(0, 1000, 100))
        _write_fake_tb_scalar(tb_dir, "CRPS_seed0", "rollout/ep_rew_mean", steps, range(10))
        # NLL_seed0 deliberately absent.
        result = load_learning_curves(tb_dir, seeds=[0])
        assert list(result.smoothed_by_agent_seed.keys()) == ["CRPS"]
        assert result.step_grid.min() == 0.0 and result.step_grid.max() == 900.0


def test_render_learning_curves_figure_scales_panel_grid():
    """Regression for the pre-existing 3x3-hardcode bug: a result with only
    a subset of agents (or, previously, 15 > 9) must not silently drop any
    agent from the figure."""
    pytest.importorskip("tensorboard")
    import tempfile
    from pathlib import Path
    from traits_audit.committee.analysis.learning_curves import (
        LearningCurveResult, render_learning_curves_figure,
    )

    grid = np.linspace(0, 900, 10)
    agents = [f"agent{i}" for i in range(12)]  # > 9, not a multiple of 5
    result = LearningCurveResult(
        step_grid=grid,
        smoothed_by_agent_seed={a: {0: np.zeros(10)} for a in agents},
        raw_by_agent_seed={a: {0: (grid, np.zeros(10))} for a in agents},
    )
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "lc.png"
        render_learning_curves_figure(result, out)
        assert out.exists()


def test_write_density_csv_1d_format_unchanged():
    """dim=1 (the default) must keep the exact pre-generalization CSV shape:
    header 'agent,seed,x', one row 'agent,seed,value' per query."""
    from traits_audit.committee.analysis.density import DensityResult, write_density_csv

    result = DensityResult(
        queries_by_agent={"CRPS": np.array([0.1, 0.2])},
        queries_by_agent_seed={"CRPS": {0: np.array([0.1, 0.2])}},
        n_episodes_per_seed=1,
    )

    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "density.csv"
        write_density_csv(result, out)
        assert out.read_text() == "agent,seed,x\nCRPS,0,0.100000\nCRPS,0,0.200000\n"
