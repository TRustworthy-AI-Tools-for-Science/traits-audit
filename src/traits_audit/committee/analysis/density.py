"""Per-agent query-density distributions — the headline 9-panel figure.

Deferred item 4 of the v0 plan. For each of the 9 agents, runs N deployment
rollouts under that agent's own trained policy (uniform-pick aggregation
disabled), and renders the histogram of acquisition queries x in [0, 1].
Each panel is overlaid with the pre-registered signature from
``predicted_styles.md`` for visual comparison.

Aggregation across seeds: the user trained 5 seeds per agent. We pool
queries from all (agent, seed) pairs into the same per-agent histogram,
and also draw seed-level mean +/- spread (KDE per seed, then envelope).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from traits_audit.committee.rewards import REWARD_REGISTRY
from traits_audit.committee.analysis.rollouts import (
    run_rollout,
    sac_policy,
)


AGENT_NAMES: list[str] = list(REWARD_REGISTRY.keys())

# Short pre-registered signatures (compressed from predicted_styles.md
# column 3). Shown as panel sub-titles so the figure is self-contained.
PREDICTED_SIGNATURES: dict[str, str] = {
    "CRPS": "Exploit low-sigma regions",
    "NLL": "Trust mu (low z^2)",
    "IntervalScore": "Safe interval (moderate sigma)",
    "CalibrationError": "Fix worst miscalibrated bin",
    "ConformalCoverage": "Absorb tail (avoid new outliers)",
    "PITUniformity": "Fill quantile gaps (broadest spread)",
    "IntervalCoverage": "Regime-dependent (sign flips)",
    "VarianceAlignment": "Balance global sigma^2 / MSE",
    "VarErrCorrelation": "Rank-align sigma with error (most distinct)",
}


@dataclass
class DensityResult:
    """Per-agent pooled queries + per-seed traces.

    Attributes
    ----------
    queries_by_agent : dict[str, np.ndarray]
        Concatenated x-queries across all (seed, episode), per agent.
    queries_by_agent_seed : dict[str, dict[int, np.ndarray]]
        Same data broken out per training seed (for envelope rendering).
    n_episodes_per_seed : int
    """

    queries_by_agent: dict[str, np.ndarray]
    queries_by_agent_seed: dict[str, dict[int, np.ndarray]]
    n_episodes_per_seed: int


def run_density_rollouts(
    models_dir: Path,
    seeds: list[int],
    n_episodes_per_seed: int = 50,
    episode_length: int = 100,
    rng_seed: int = 0,
) -> DensityResult:
    """For every (agent, training_seed), run N rollouts under that policy.

    Returns the pooled x-queries per agent.
    """
    from stable_baselines3 import SAC

    rng = np.random.default_rng(rng_seed)
    by_agent: dict[str, list[np.ndarray]] = {n: [] for n in AGENT_NAMES}
    by_agent_seed: dict[str, dict[int, np.ndarray]] = {
        n: {} for n in AGENT_NAMES
    }

    for agent in AGENT_NAMES:
        for s in seeds:
            path = models_dir / f"{agent}_seed{s}.zip"
            if not path.exists():
                raise FileNotFoundError(f"Missing model: {path}")
            model = SAC.load(str(path))
            per_seed_queries: list[np.ndarray] = []
            for _ in range(n_episodes_per_seed):
                ep_seed = int(rng.integers(0, 2**31 - 1))
                trace = run_rollout(
                    policy=sac_policy(model),
                    seed=ep_seed,
                    episode_length=episode_length,
                )
                per_seed_queries.append(trace.x_queries)
            pooled = np.concatenate(per_seed_queries)
            by_agent[agent].append(pooled)
            by_agent_seed[agent][s] = pooled
            print(f"[density] {agent} seed={s}: "
                  f"{len(pooled)} queries, mean x = {pooled.mean():.3f}")

    return DensityResult(
        queries_by_agent={n: np.concatenate(by_agent[n]) for n in AGENT_NAMES},
        queries_by_agent_seed=by_agent_seed,
        n_episodes_per_seed=n_episodes_per_seed,
    )


def _forrester(x: np.ndarray) -> np.ndarray:
    return (6.0 * x - 2.0) ** 2 * np.sin(12.0 * x - 4.0)


def render_headline_figure(
    result: DensityResult,
    output_path: Path,
    n_bins: int = 30,
) -> None:
    """Render the 9-panel headline figure.

    Each panel: histogram of queries pooled across seeds, with per-seed
    histograms overlaid as thin lines (so you can see seed-to-seed spread).
    Pre-registered signature printed as the sub-title. The clean Forrester
    oracle is overlaid on a twin axis so query densities can be read
    against the function landscape.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharex=True)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    x_grid = np.linspace(0.0, 1.0, 400)
    f_grid = _forrester(x_grid)

    for ax, agent in zip(axes.ravel(), AGENT_NAMES):
        pooled = result.queries_by_agent[agent]
        ax.hist(pooled, bins=edges, density=True, color="C0", alpha=0.45,
                label="pooled (5 seeds)")
        for s, qs in result.queries_by_agent_seed[agent].items():
            counts, _ = np.histogram(qs, bins=edges, density=True)
            ax.step(edges[:-1], counts, where="post",
                    color="black", alpha=0.25, linewidth=0.7)
        ax.set_xlim(0.0, 1.0)
        ax.set_title(agent, fontsize=15)
        ax.set_ylabel("density", fontsize=13)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(x_grid, f_grid, color="0.55", linewidth=1.0, alpha=0.6,
                 label="Forrester f(x)")
        # Forrester overlay is visual context only; suppress its axis.
        ax2.set_ylabel("")
        ax2.tick_params(axis="y", which="both",
                        left=False, right=False,
                        labelleft=False, labelright=False)

    for ax in axes[-1]:
        ax.set_xlabel("x (acquisition query)", fontsize=13)
    # fig.suptitle(
    #     f"Per-agent query densities vs pre-registered signatures "
    #     f"({result.n_episodes_per_seed} eps x 5 seeds per agent) "
    #     f"- Forrester oracle overlaid (gray)",
    #     fontsize=12,
    # )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def write_density_csv(result: DensityResult, output_path: Path) -> None:
    """Tidy CSV: agent, seed, x. One row per query."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["agent,seed,x"]
    for agent, by_seed in result.queries_by_agent_seed.items():
        for s, xs in by_seed.items():
            for x in xs:
                rows.append(f"{agent},{s},{x:.6f}")
    output_path.write_text("\n".join(rows) + "\n")
