"""Thread B (votes-as-features) and Thread A (vote-aggregators) regret bake-offs.

Both threads share the same evaluation harness:
    - 20 episode seeds, paired across all policies.
    - 100 acquisition steps per episode, warm-start 20.
    - Clean-Forrester simple regret (matches v0 regret.py).
    - Paired Wilcoxon signed-rank at terminal step against a chosen reference.

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
from typing import Callable, Iterable, Optional

import numpy as np

from traits_audit.committee.env import (
    CommitteeEnv,
    DEFAULT_EPISODE_LENGTH,
    DEFAULT_WARMSTART,
)
from traits_audit.committee.rewards import REWARD_REGISTRY
from traits_audit.committee.analysis.regret import (
    AGENT_NAMES,
    FORRESTER_TRUE_MIN,
    _forrester_clean,
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
) -> DiagnosticRolloutTrace:
    """Run policy; per step record committee std of preferred actions at obs.

    Mirrors :func:`traits_audit.committee.analysis.rollouts.run_rollout` but
    intercepts the obs after each step. If ``voter`` is None, action-std is
    filled with NaN (still legal for plotting; A3 will skip it).
    """
    dummy_reward = REWARD_REGISTRY["CRPS"]()
    env = CommitteeEnv(
        reward_computer=dummy_reward,
        episode_length=episode_length,
        warmstart_n=warmstart_n,
    )
    obs, _ = env.reset(seed=seed)
    x_q_list: list[float] = []
    action_std = np.full(episode_length, np.nan, dtype=np.float64)

    for t in range(episode_length):
        if voter is not None:
            prefs = voter.preferred_actions(obs)
            action_std[t] = float(np.std(prefs))
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
) -> ThreadResult:
    """Run every policy in ``policy_specs`` on the same episode seeds.

    Each spec maps name -> factory(episode_seed) -> Policy. Episode seeds
    are sampled deterministically from ``rng_seed`` so reruns are paired.
    """
    rng = np.random.default_rng(rng_seed)
    seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(n_episode_seeds)]

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
            )
            sr_rows.append(_trace_to_regret(diag.trace, warmstart_n, episode_length))
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
) -> tuple[ThreadResult, CommitteeVoter]:
    """Paired regret bake-off for vote-augmented baselines.

    Policies:
        random, max-sigma, max-sigma+votes, LCB, LCB+votes.

    Returns (result, voter). The voter is returned so the caller can re-use
    it for ablation runs (Thread B's permutation-importance figure).
    """
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
) -> dict[str, np.ndarray]:
    """Leave-one-agent-out ablation for a vote-augmented policy.

    For each agent k, build a sub-voter view that drops k, run the augmented
    policy with the (n-1)-member committee, and record terminal SR.

    ``policy`` selects which augmented policy to ablate:
        - "LCB+votes" (default) — pair to lcb_with_votes_policy.
        - "max-sigma+votes"     — pair to max_sigma_with_votes_policy.

    Returns dict[agent_name] -> terminal_regret_array (shape n_episode_seeds).
    """
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
            prefs = self._base.preferred_actions(obs)
            return np.delete(prefs, self._mask)

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

    terminal: dict[str, np.ndarray] = {}
    for k, name in enumerate(voter.agent_names):
        print(f"[ablation:{policy}] drop {name} ...")
        sub = _MaskedVoter(voter, k)
        pol_factory = lambda es, _sub=sub: make_policy(_sub)
        sr = np.zeros(len(seeds), dtype=float)
        for i, es in enumerate(seeds):
            pol = pol_factory(es)
            diag = _run_rollout_with_diagnostics(
                pol, voter=None, seed=es,
                episode_length=episode_length, warmstart_n=warmstart_n,
            )
            sr[i] = _trace_to_regret(diag.trace, warmstart_n, episode_length)[-1]
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
) -> tuple[ThreadResult, CommitteeVoter, dict[str, float], dict[str, float]]:
    """Aggregator bake-off.

    Policies:
        random, best-solo (PITUniformity), committee-uniform (v0 baseline),
        committee-agree, committee-disagree,
        committee-weighted-by-independence, committee-weighted-by-inv-regret.

    Returns (result, voter, indep_weights, invreg_weights). The weight dicts
    are returned so the A2 figure can plot them against solo regret without
    re-reading the source files.
    """
    from stable_baselines3 import SAC

    voter = CommitteeVoter(models_dir=Path(models_dir),
                          committee_solo_seed=committee_solo_seed)
    indep_w = independence_weights(correlation_csv, voter.agent_names)
    invreg_w = inverse_regret_weights(regret_json, voter.agent_names)

    # Best-solo (PITUniformity, from v0 results). Skip the replay buffer at
    # load time — same reason as in CommitteeVoter.
    best_solo_model = SAC.load(
        str(Path(models_dir) / f"PITUniformity_seed{committee_solo_seed}.zip"),
        custom_objects={"buffer_size": 1},
    )

    specs: dict[str, Callable[[int], Policy]] = {
        "random": lambda es: random_policy(np.random.default_rng(es + 1)),
        "best-solo:PITUniformity": lambda es: sac_policy(best_solo_model),
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
    )
    return result, voter, indep_w, invreg_w


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
