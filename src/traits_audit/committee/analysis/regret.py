"""Simple-regret comparison: random / max-sigma / LCB / 9 solo / committee.

Deferred item 6 of the v0 plan. Simple regret at step t for one episode is::

    SR(t) = min_{i <= t} f_clean(x_i)  -  f*

evaluated on the *clean* Forrester (the noiseless objective), not on the
noisy y_obs. Using noisy observations would let a lucky noise draw push SR
below 0; standard BO convention is to score the chosen x against the clean
oracle. f* is computed numerically once.

For each policy:
    20 rollout seeds x 100 steps x (warmstart 20) per the plan.

Committee = uniform pick across the 9 trained policies at each step.

Statistical test on terminal simple regret (step T):
    Paired Wilcoxon signed-rank, committee vs best-solo (the solo agent with
    the lowest mean SR at step T). Two-sided. p < 0.05 declares significance.

Output: a long-format CSV (one row per (policy, seed, step)) and a figure
of mean SR with 95% CI bands.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from traits_audit._example import oracle as forrester_oracle
from traits_audit.committee.env import DEFAULT_EPISODE_LENGTH, DEFAULT_WARMSTART
from traits_audit.committee.rewards import REWARD_REGISTRY
from traits_audit.committee.analysis.rollouts import (
    lcb_policy,
    max_sigma_policy,
    random_policy,
    run_rollout,
    sac_policy,
)


AGENT_NAMES: list[str] = list(REWARD_REGISTRY.keys())


def _forrester_clean(x: np.ndarray) -> np.ndarray:
    """Noiseless Forrester evaluated at x."""
    return (6.0 * x - 2.0) ** 2 * np.sin(12.0 * x - 4.0)


def _forrester_min(grid_size: int = 100_000) -> float:
    """Numerically locate min of the clean Forrester benchmark on [0, 1]."""
    xs = np.linspace(0.0, 1.0, grid_size)
    return float(_forrester_clean(xs).min())


FORRESTER_TRUE_MIN = _forrester_min()


@dataclass
class RegretResult:
    """Simple regret per (policy, seed, step).

    Attributes
    ----------
    per_policy : dict[str, np.ndarray]
        Each value has shape (n_seeds, episode_length). Entry [s, t] is the
        running minimum y observed by step t (warm-start included) minus
        the true Forrester minimum.
    seeds : list[int]
    episode_length : int
    warmstart_n : int
    """

    per_policy: dict[str, np.ndarray]
    seeds: list[int]
    episode_length: int
    warmstart_n: int


def _trace_to_regret(trace, warmstart_n: int, episode_length: int) -> np.ndarray:
    """Per-step simple regret along the acquisition trajectory.

    Score *clean* Forrester at each queried x and take the running min,
    so noise can't drive regret negative.
    """
    f_clean = _forrester_clean(trace.x_obs)
    sr = np.zeros(episode_length, dtype=float)
    for t in range(episode_length):
        cutoff = warmstart_n + t + 1
        sr[t] = float(np.min(f_clean[:cutoff])) - FORRESTER_TRUE_MIN
    return sr


def _make_committee_policy(models: list, rng: np.random.Generator) -> Callable:
    """Uniform-random pick across the 9 frozen policies at each step.

    Each step: every policy proposes; one is chosen uniformly. Reuses the
    SB3 ``predict(obs, deterministic=True)`` interface.
    """
    n = len(models)
    def _pi(obs: np.ndarray, env) -> np.ndarray:
        # Cheap version: just sample which agent to use this step and
        # ask only that one (predictions are deterministic anyway).
        idx = int(rng.integers(0, n))
        action, _ = models[idx].predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(1)
    return _pi


def run_regret(
    models_dir: Path,
    seeds: list[int],
    n_episode_seeds: int = 20,
    episode_length: int = DEFAULT_EPISODE_LENGTH,
    warmstart_n: int = DEFAULT_WARMSTART,
    rng_seed: int = 0,
    committee_solo_seed: int = 0,
) -> RegretResult:
    """Compute simple regret for all comparator policies.

    Policies:
      - random
      - max-sigma
      - LCB (kappa=2)
      - each of the 9 solo SAC policies (using training seed ``committee_solo_seed``)
      - committee (uniform pick across 9 policies at ``committee_solo_seed``)

    For statistical apples-to-apples, every policy is evaluated on the same
    ``n_episode_seeds`` episode seeds.
    """
    from stable_baselines3 import SAC

    rng = np.random.default_rng(rng_seed)
    episode_seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(n_episode_seeds)]

    # Load the solo models at the requested training seed for committee + solo.
    models: dict[str, object] = {}
    for agent in AGENT_NAMES:
        p = models_dir / f"{agent}_seed{committee_solo_seed}.zip"
        if not p.exists():
            raise FileNotFoundError(f"Missing model: {p}")
        models[agent] = SAC.load(str(p))

    per_policy: dict[str, np.ndarray] = {}

    def _rollout_grid(policy_factory) -> np.ndarray:
        """Run policy on every episode_seed, stack regret rows."""
        sr_rows = []
        for es in episode_seeds:
            pol = policy_factory(es)
            tr = run_rollout(pol, seed=es,
                             episode_length=episode_length,
                             warmstart_n=warmstart_n)
            sr_rows.append(_trace_to_regret(tr, warmstart_n, episode_length))
        return np.stack(sr_rows, axis=0)

    # Stateless comparators: factory ignores the seed (policy itself uses env state).
    print("[regret] random ...")
    per_policy["random"] = _rollout_grid(
        lambda es: random_policy(np.random.default_rng(es + 1))
    )
    print("[regret] max-sigma ...")
    per_policy["max-sigma"] = _rollout_grid(lambda es: max_sigma_policy())
    print("[regret] LCB ...")
    per_policy["LCB"] = _rollout_grid(lambda es: lcb_policy())

    for agent in AGENT_NAMES:
        print(f"[regret] solo:{agent} ...")
        per_policy[f"solo:{agent}"] = _rollout_grid(lambda es, a=agent: sac_policy(models[a]))

    print("[regret] committee ...")
    per_policy["committee"] = _rollout_grid(
        lambda es: _make_committee_policy(
            list(models.values()), np.random.default_rng(es + 999)
        )
    )

    return RegretResult(
        per_policy=per_policy,
        seeds=episode_seeds,
        episode_length=episode_length,
        warmstart_n=warmstart_n,
    )


def paired_test(result: RegretResult) -> dict:
    """Paired Wilcoxon signed-rank: committee vs best-solo at terminal step.

    Best-solo := the solo policy with the lowest *mean* terminal regret.
    """
    from scipy.stats import wilcoxon

    T = result.episode_length - 1
    solo_means = {
        name: float(arr[:, T].mean())
        for name, arr in result.per_policy.items()
        if name.startswith("solo:")
    }
    best_solo = min(solo_means, key=solo_means.get)
    committee = result.per_policy["committee"][:, T]
    best_arr = result.per_policy[best_solo][:, T]
    # Wilcoxon requires no all-zero differences; jitter with a tiny noise if so.
    diffs = committee - best_arr
    if np.allclose(diffs, 0):
        return {
            "best_solo": best_solo,
            "committee_mean": float(committee.mean()),
            "best_solo_mean": solo_means[best_solo],
            "p_value": 1.0,
            "stat": 0.0,
            "note": "differences all zero",
        }
    stat, p = wilcoxon(committee, best_arr, alternative="two-sided")
    return {
        "best_solo": best_solo,
        "committee_mean": float(committee.mean()),
        "best_solo_mean": solo_means[best_solo],
        "p_value": float(p),
        "stat": float(stat),
        "solo_means": solo_means,
    }


def render_regret_figure(result: RegretResult, output_path: Path) -> None:
    """Mean simple regret over time per policy with 95% CI bands.

    Solo agents in the Okabe-Ito palette; baselines (random / max-sigma) muted
    in dashed grey; committee in vermillion as the headline policy.
    """
    import matplotlib.pyplot as plt
    from traits_audit.committee.analysis import style as st

    fig, ax = plt.subplots(figsize=(10, 6.2))
    xs = np.arange(result.episode_length)

    def _band(arr, label, color, lw=1.6, ls="-", alpha=1.0,
              alpha_band=0.0, zorder=2):
        mean = arr.mean(axis=0)
        ax.plot(xs, mean, label=label, color=color, lw=lw, ls=ls,
                alpha=alpha, zorder=zorder)
        if alpha_band > 0:
            sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
            ax.fill_between(xs, mean - 1.96 * sem, mean + 1.96 * sem,
                            color=color, alpha=alpha_band,
                            zorder=zorder - 1)

    # Solo agents in the Okabe-Ito palette, no CI band.
    for agent in AGENT_NAMES:
        _band(result.per_policy[f"solo:{agent}"], f"solo:{agent}",
              color=st.policy_color(f"solo:{agent}"),
              lw=1.8, zorder=1)

    # Baselines muted + dashed, no band.
    _band(result.per_policy["random"], "random",
          **st.RANDOM_STYLE)
    _band(result.per_policy["max-sigma"], "max-σ",
          **st.BASELINE_STYLE)
    _band(result.per_policy["LCB"], "LCB",
          color=st.BLACK, lw=2.0, ls="-")
    _band(result.per_policy["committee"], "committee",
          **st.HEADLINE_STYLE)

    ax.set_xlabel("acquisition step", fontsize=st.LABEL_FS)
    ax.set_ylabel("simple regret", fontsize=st.LABEL_FS)
    ax.set_title(f"Simple regret ({len(result.seeds)} episode seeds)",
                 fontsize=st.TITLE_FS)
    ax.legend(fontsize=st.LEGEND_FS, ncol=2, loc="lower left",
              framealpha=0.92)
    st.style_axes(ax, xlim=(0, 100), ylog=True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def write_regret_csv(result: RegretResult, output_path: Path) -> None:
    """Long-form CSV: policy, episode_seed, step, simple_regret."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["policy,episode_seed,step,simple_regret"]
    for policy, arr in result.per_policy.items():
        for i, es in enumerate(result.seeds):
            for t in range(result.episode_length):
                rows.append(f"{policy},{es},{t},{arr[i, t]:.6f}")
    output_path.write_text("\n".join(rows) + "\n")
