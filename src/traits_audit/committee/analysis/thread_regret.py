"""Thread B (votes-as-features) and Thread A (vote-aggregators) regret bake-offs.

Both threads share the same evaluation harness:
    - 20 episode seeds, paired across all policies.
    - 100 acquisition steps per episode, warm-start 20.
    - Clean simple regret on the problem's audited objective (shares
      ``regret._trace_to_regret``, so it matches the v0 regret curves).
    - Paired Wilcoxon signed-rank at terminal step against a chosen reference.

Both run on any benchmark in ``committee/problems.py`` — pass ``problem=``.
The committee's preferred actions are vectors throughout (see
:mod:`.votes`), so 1-D Forrester, 2-D Branin-Currin and 3-D colour matching
all go through the same code path, and the best-solo reference is read from
the problem's own ``regret_test.json`` rather than assumed.

Per-step diagnostics also captured for the A3 / B3 figures:
    - committee_action_std[t]   — std of the 9 preferred actions at obs_t,
      averaged later across seeds.
    - The figures live in :mod:`.thread_figures`.

This module is intentionally separate from :mod:`.regret` so the v0 outputs
on disk are not perturbed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

import numpy as np

from traits_audit.committee.env import (
    CommitteeEnv,
    DEFAULT_EPISODE_LENGTH,
    DEFAULT_WARMSTART,
)
from traits_audit.committee.problems import ForresterProblem, Problem, get_problem
from traits_audit.committee.rewards import REWARD_REGISTRY
from traits_audit.committee.analysis.regret import (
    AGENT_NAMES,
    _trace_to_regret,
)
from traits_audit.committee.analysis.rollouts import (
    Policy,
    RolloutTrace,
    lcb_policy,
    max_sigma_policy,
    random_policy,
    sac_policy,
)
from traits_audit.committee.analysis.votes import CommitteeVoter
from traits_audit.committee.analysis.policies_ext import (
    committee_agree_policy,
    committee_disagree_policy,
    committee_uniform_policy,
    committee_weighted_policy,
    independence_weights,
    inverse_regret_weights,
    lcb_with_votes_policy,
    max_sigma_with_votes_policy,
)


# ---------------------------------------------------------------------------
# Rollout-with-diagnostics: drop-in for run_rollout that also captures the
# committee action-std per step (needed for the disagreement diagnostic).
# ---------------------------------------------------------------------------

def _resolve_problem(problem: Union[Problem, str, None]) -> Problem:
    """Accept a Problem, a registry name, or None (-> Forrester)."""
    if isinstance(problem, str):
        return get_problem(problem)
    if problem is None:
        return ForresterProblem()
    return problem


@dataclass
class DiagnosticRolloutTrace:
    trace: RolloutTrace
    committee_action_std: np.ndarray  # shape (episode_length,)


def _run_rollout_with_diagnostics(
    policy: Policy,
    voter: Optional[CommitteeVoter],
    seed: int,
    episode_length: int = DEFAULT_EPISODE_LENGTH,
    warmstart_n: int = DEFAULT_WARMSTART,
    problem: Union[Problem, str, None] = None,
) -> DiagnosticRolloutTrace:
    """Run policy; per step record committee spread of preferred actions at obs.

    Mirrors :func:`traits_audit.committee.analysis.rollouts.run_rollout` but
    intercepts the obs after each step. If ``voter`` is None, action-std is
    filled with NaN (still legal for plotting; A3 will skip it).

    The recorded spread is the root-mean-square per-axis std of the (K, dim)
    preferred actions — identical to ``np.std(prefs)`` when dim == 1, and in
    higher dimensions the square root of the same trace-of-covariance that
    :func:`~.policies_ext.committee_disagree_policy` maximises, so the A3
    diagnostic and the disagree aggregator measure the same quantity.
    """
    dummy_reward = REWARD_REGISTRY["CRPS"]()
    env = CommitteeEnv(
        reward_computer=dummy_reward,
        episode_length=episode_length,
        warmstart_n=warmstart_n,
        problem=problem,
    )
    obs, _ = env.reset(seed=seed)
    x_q_list: list[np.ndarray] = []
    action_std = np.full(episode_length, np.nan, dtype=np.float64)

    for t in range(episode_length):
        if voter is not None:
            prefs = np.atleast_2d(voter.preferred_actions(obs))
            action_std[t] = float(np.sqrt(prefs.var(axis=0).sum()))
        action = policy(obs, env)
        obs, _r, terminated, truncated, info = env.step(action)
        x_q_list.append(info["x_q"])
        if terminated or truncated:
            break

    trace = RolloutTrace(
        x_obs=np.asarray(env.x_obs, dtype=float),
        y_obs=np.asarray(env.y_obs, dtype=float),
        mu_hist=np.asarray(env._mu_history, dtype=float),
        sigma_hist=np.asarray(env._sigma_history, dtype=float),
        x_queries=np.asarray(x_q_list, dtype=float),
        warmstart_n=warmstart_n,
    )
    return DiagnosticRolloutTrace(trace=trace, committee_action_std=action_std)


# ---------------------------------------------------------------------------
# Shared evaluation harness.
# ---------------------------------------------------------------------------

@dataclass
class ThreadResult:
    """Per-policy regret + diagnostics on the shared episode seeds.

    per_policy_regret : dict[str, np.ndarray]
        Shape (n_seeds, episode_length).
    per_policy_action_std : dict[str, np.ndarray]
        Shape (n_seeds, episode_length). NaN for policies evaluated without
        a voter attached.
    seeds, episode_length, warmstart_n : metadata.
    """

    per_policy_regret: dict[str, np.ndarray]
    per_policy_action_std: dict[str, np.ndarray]
    seeds: list[int]
    episode_length: int
    warmstart_n: int


def _bakeoff(
    policy_specs: dict[str, Callable[[int], Policy]],
    voter: Optional[CommitteeVoter],
    n_episode_seeds: int,
    episode_length: int,
    warmstart_n: int,
    rng_seed: int,
    problem: Problem,
    only: Optional[Iterable[str]] = None,
) -> ThreadResult:
    """Run every policy in ``policy_specs`` on the same episode seeds.

    Each spec maps name -> factory(episode_seed) -> Policy. Episode seeds
    are sampled deterministically from ``rng_seed`` so reruns are paired.

    ``only`` restricts the run to a subset of the policy names — one shard
    of an HPC array job. The episode seeds do not depend on it, so shards
    recombine (see :func:`read_thread_csv`) into the same paired result a
    single process would have produced.
    """
    # Seeds are drawn before any policy filtering, so a sharded run and a
    # whole run see byte-identical episode seeds and stay paired.
    rng = np.random.default_rng(rng_seed)
    seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(n_episode_seeds)]

    if only is not None:
        wanted = list(only)
        unknown = [n for n in wanted if n not in policy_specs]
        if unknown:
            raise KeyError(
                f"unknown policy name(s) {unknown}; "
                f"available: {sorted(policy_specs)}"
            )
        policy_specs = {n: policy_specs[n] for n in wanted}

    per_regret: dict[str, np.ndarray] = {}
    per_std: dict[str, np.ndarray] = {}

    for name, factory in policy_specs.items():
        print(f"[bakeoff] {name} ...")
        sr_rows = []
        std_rows = []
        for es in seeds:
            pol = factory(es)
            diag = _run_rollout_with_diagnostics(
                pol, voter, seed=es,
                episode_length=episode_length, warmstart_n=warmstart_n,
                problem=problem,
            )
            sr_rows.append(
                _trace_to_regret(diag.trace, warmstart_n, episode_length, problem)
            )
            std_rows.append(diag.committee_action_std)
        per_regret[name] = np.stack(sr_rows, axis=0)
        per_std[name] = np.stack(std_rows, axis=0)

    return ThreadResult(
        per_policy_regret=per_regret,
        per_policy_action_std=per_std,
        seeds=seeds,
        episode_length=episode_length,
        warmstart_n=warmstart_n,
    )


# ---------------------------------------------------------------------------
# Thread B runner.
# ---------------------------------------------------------------------------

def run_thread_b(
    models_dir: Path,
    n_episode_seeds: int = 20,
    episode_length: int = DEFAULT_EPISODE_LENGTH,
    warmstart_n: int = DEFAULT_WARMSTART,
    rng_seed: int = 0,
    committee_solo_seed: int = 0,
    vote_weight: float = 1.0,
    problem: Union[Problem, str, None] = None,
    only: Optional[Iterable[str]] = None,
) -> tuple[ThreadResult, CommitteeVoter]:
    """Paired regret bake-off for vote-augmented baselines.

    Policies:
        random, max-sigma, max-sigma+votes, LCB, LCB+votes.

    Returns (result, voter). The voter is returned so the caller can re-use
    it for ablation runs (Thread B's permutation-importance figure).

    ``problem`` selects the benchmark (default Forrester); ``models_dir``
    must hold models trained on that same problem.
    """
    problem = _resolve_problem(problem)
    voter = CommitteeVoter(models_dir=Path(models_dir),
                          committee_solo_seed=committee_solo_seed)

    specs: dict[str, Callable[[int], Policy]] = {
        "random": lambda es: random_policy(np.random.default_rng(es + 1)),
        "max-sigma": lambda es: max_sigma_policy(),
        "max-sigma+votes": lambda es: max_sigma_with_votes_policy(
            voter, vote_weight=vote_weight,
        ),
        "LCB": lambda es: lcb_policy(),
        "LCB+votes": lambda es: lcb_with_votes_policy(
            voter, vote_weight=vote_weight,
        ),
    }
    result = _bakeoff(
        specs, voter,
        n_episode_seeds=n_episode_seeds,
        episode_length=episode_length,
        warmstart_n=warmstart_n,
        rng_seed=rng_seed,
        problem=problem,
        only=only,
    )
    return result, voter


def run_thread_b_ablation(
    voter: CommitteeVoter,
    n_episode_seeds: int = 20,
    episode_length: int = DEFAULT_EPISODE_LENGTH,
    warmstart_n: int = DEFAULT_WARMSTART,
    rng_seed: int = 0,
    vote_weight: float = 1.0,
    policy: str = "LCB+votes",
    problem: Union[Problem, str, None] = None,
    only_agents: Optional[Iterable[str]] = None,
) -> dict[str, np.ndarray]:
    """Leave-one-agent-out ablation for a vote-augmented policy.

    For each agent k, build a sub-voter view that drops k, run the augmented
    policy with the (n-1)-member committee, and record terminal SR.

    ``policy`` selects which augmented policy to ablate:
        - "LCB+votes" (default) — pair to lcb_with_votes_policy.
        - "max-sigma+votes"     — pair to max_sigma_with_votes_policy.

    Returns dict[agent_name] -> terminal_regret_array (shape n_episode_seeds).
    """
    problem = _resolve_problem(problem)
    rng = np.random.default_rng(rng_seed)
    seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(n_episode_seeds)]

    class _MaskedVoter:
        """View of `voter` with one agent's contribution removed."""
        def __init__(self, base: CommitteeVoter, mask_idx: int) -> None:
            self._base = base
            self._mask = mask_idx
        @property
        def agent_names(self): return self._base.agent_names
        @property
        def n_agents(self): return self._base.n_agents - 1
        def preferred_actions(self, obs):
            # axis=0 drops the agent's row; without it np.delete would
            # flatten the (K, dim) array and remove a single coordinate.
            prefs = self._base.preferred_actions(obs)
            return np.delete(prefs, self._mask, axis=0)

    if policy == "LCB+votes":
        make_policy = lambda sub: lcb_with_votes_policy(
            sub, vote_weight=vote_weight,
        )
    elif policy == "max-sigma+votes":
        make_policy = lambda sub: max_sigma_with_votes_policy(
            sub, vote_weight=vote_weight,
        )
    else:
        raise ValueError(f"Unknown ablation target policy: {policy!r}")

    # The mask index must stay the agent's position in the *full* committee,
    # so filter the loop rather than the committee itself.
    wanted = set(voter.agent_names) if only_agents is None else set(only_agents)
    unknown = wanted - set(voter.agent_names)
    if unknown:
        raise KeyError(f"unknown agent(s) to ablate: {sorted(unknown)}")

    terminal: dict[str, np.ndarray] = {}
    for k, name in enumerate(voter.agent_names):
        if name not in wanted:
            continue
        print(f"[ablation:{policy}] drop {name} ...")
        sub = _MaskedVoter(voter, k)
        pol_factory = lambda es, _sub=sub: make_policy(_sub)
        sr = np.zeros(len(seeds), dtype=float)
        for i, es in enumerate(seeds):
            pol = pol_factory(es)
            diag = _run_rollout_with_diagnostics(
                pol, voter=None, seed=es,
                episode_length=episode_length, warmstart_n=warmstart_n,
                problem=problem,
            )
            sr[i] = _trace_to_regret(
                diag.trace, warmstart_n, episode_length, problem
            )[-1]
        terminal[name] = sr
    return terminal


# ---------------------------------------------------------------------------
# Thread A runner.
# ---------------------------------------------------------------------------

def run_thread_a(
    models_dir: Path,
    correlation_csv: Path,
    regret_json: Path,
    n_episode_seeds: int = 20,
    episode_length: int = DEFAULT_EPISODE_LENGTH,
    warmstart_n: int = DEFAULT_WARMSTART,
    rng_seed: int = 0,
    committee_solo_seed: int = 0,
    problem: Union[Problem, str, None] = None,
    only: Optional[Iterable[str]] = None,
) -> tuple[ThreadResult, CommitteeVoter, dict[str, float], dict[str, float], str]:
    """Aggregator bake-off.

    Policies:
        random, best-solo, committee-uniform (v0 baseline),
        committee-agree, committee-disagree,
        committee-weighted-by-independence, committee-weighted-by-inv-regret.

    The best-solo reference is whichever agent has the lowest mean terminal
    SR in ``regret_json`` — it is *not* PITUniformity in general. It wins on
    Forrester but is among the worst agents on Branin-Currin, so hardcoding
    it would silently compare every aggregator against a poor baseline.

    Returns (result, voter, indep_weights, invreg_weights, best_solo_agent).
    The weight dicts are returned so the A2 figure can plot them against solo
    regret without re-reading the source files; the agent name is returned so
    the A1 figure can label and test against the right reference.
    """
    import json

    from stable_baselines3 import SAC

    problem = _resolve_problem(problem)
    voter = CommitteeVoter(models_dir=Path(models_dir),
                          committee_solo_seed=committee_solo_seed)
    indep_w = independence_weights(correlation_csv, voter.agent_names)
    invreg_w = inverse_regret_weights(regret_json, voter.agent_names)

    solo_means = json.loads(Path(regret_json).read_text())["solo_means"]
    # Restrict to agents this committee actually holds, then take the best.
    best_solo = min(
        voter.agent_names,
        key=lambda a: float(solo_means.get(f"solo:{a}", np.inf)),
    )
    print(f"[thread-a] best-solo reference: {best_solo} "
          f"(terminal SR {float(solo_means[f'solo:{best_solo}']):.4g})")

    # Skip the replay buffer at load time — same reason as in CommitteeVoter.
    best_solo_model = SAC.load(
        str(Path(models_dir) / f"{best_solo}_seed{committee_solo_seed}.zip"),
        custom_objects={"buffer_size": 1},
    )

    specs: dict[str, Callable[[int], Policy]] = {
        "random": lambda es: random_policy(np.random.default_rng(es + 1)),
        f"best-solo:{best_solo}": lambda es: sac_policy(best_solo_model),
        "committee:uniform": lambda es: committee_uniform_policy(
            voter, np.random.default_rng(es + 999),
        ),
        "committee:agree": lambda es: committee_agree_policy(voter),
        "committee:disagree": lambda es: committee_disagree_policy(voter),
        "committee:weighted-indep": lambda es: committee_weighted_policy(
            voter, indep_w,
        ),
        "committee:weighted-invreg": lambda es: committee_weighted_policy(
            voter, invreg_w,
        ),
    }
    result = _bakeoff(
        specs, voter,
        n_episode_seeds=n_episode_seeds,
        episode_length=episode_length,
        warmstart_n=warmstart_n,
        rng_seed=rng_seed,
        problem=problem,
        only=only,
    )
    return result, voter, indep_w, invreg_w, best_solo


# ---------------------------------------------------------------------------
# Statistical tests.
# ---------------------------------------------------------------------------

def paired_terminal_test(
    result: ThreadResult,
    a: str,
    b: str,
) -> dict:
    """Paired Wilcoxon at terminal step: a vs b."""
    from scipy.stats import wilcoxon

    T = result.episode_length - 1
    av = result.per_policy_regret[a][:, T]
    bv = result.per_policy_regret[b][:, T]
    diffs = av - bv
    if np.allclose(diffs, 0):
        return {"a": a, "b": b, "a_mean": float(av.mean()),
                "b_mean": float(bv.mean()),
                "p_value": 1.0, "stat": 0.0,
                "note": "differences all zero"}
    stat, p = wilcoxon(av, bv, alternative="two-sided")
    return {
        "a": a, "b": b,
        "a_mean": float(av.mean()), "b_mean": float(bv.mean()),
        "p_value": float(p), "stat": float(stat),
    }


def read_thread_csv(paths: Iterable[Path]) -> ThreadResult:
    """Rebuild a :class:`ThreadResult` from one or more shard CSVs.

    Inverse of :func:`write_thread_csv`. Reading several paths merges their
    policies into one result, which is how the HPC path works: each array
    task bakes off a subset of the policies onto the *same* episode seeds
    and writes its own CSV, then a gather step gets back exactly the result
    a single-process run would have produced.

    Raises if the shards disagree on the episode seeds or their order —
    every figure here is a *paired* comparison, so silently merging shards
    run on different seeds would invalidate every Wilcoxon test.
    """
    import csv as _csv

    per_regret: dict[str, dict[int, dict[int, float]]] = {}
    per_std: dict[str, dict[int, dict[int, float]]] = {}
    seed_order: list[int] = []
    seen: set[int] = set()

    for path in paths:
        with Path(path).open() as fh:
            for row in _csv.DictReader(fh):
                policy = row["policy"]
                es, t = int(row["episode_seed"]), int(row["step"])
                if es not in seen:
                    seen.add(es)
                    seed_order.append(es)
                per_regret.setdefault(policy, {}).setdefault(es, {})[t] = \
                    float(row["simple_regret"])
                std_raw = row.get("action_std", "")
                per_std.setdefault(policy, {}).setdefault(es, {})[t] = (
                    float(std_raw) if std_raw else np.nan
                )
    if not per_regret:
        raise ValueError("no rows found in the given thread CSV(s)")

    # Every policy must cover the same seeds; pin the order from first sight
    # so row i of one policy's array is the same episode as row i of another.
    episode_length = 1 + max(
        t for pol in per_regret.values() for row in pol.values() for t in row
    )
    for policy, by_seed in per_regret.items():
        missing = set(seed_order) - set(by_seed)
        if missing:
            raise ValueError(
                f"policy {policy!r} is missing episode seeds {sorted(missing)}; "
                "shards must be run on the same seeds to stay paired"
            )

    def _stack(src: dict[str, dict[int, dict[int, float]]]) -> dict[str, np.ndarray]:
        out = {}
        for policy, by_seed in src.items():
            arr = np.full((len(seed_order), episode_length), np.nan)
            for i, es in enumerate(seed_order):
                for t, v in by_seed[es].items():
                    arr[i, t] = v
            out[policy] = arr
        return out

    return ThreadResult(
        per_policy_regret=_stack(per_regret),
        per_policy_action_std=_stack(per_std),
        seeds=seed_order,
        episode_length=episode_length,
        warmstart_n=DEFAULT_WARMSTART,
    )


def write_thread_csv(result: ThreadResult, output_path: Path) -> None:
    """Long-form CSV: policy, episode_seed, step, simple_regret, action_std."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["policy,episode_seed,step,simple_regret,action_std"]
    for policy, arr in result.per_policy_regret.items():
        std_arr = result.per_policy_action_std[policy]
        for i, es in enumerate(result.seeds):
            for t in range(result.episode_length):
                std_val = std_arr[i, t]
                std_str = f"{std_val:.6f}" if not np.isnan(std_val) else ""
                rows.append(
                    f"{policy},{es},{t},{arr[i, t]:.6f},{std_str}"
                )
    output_path.write_text("\n".join(rows) + "\n")
