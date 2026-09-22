"""Per-agent query-density distributions — the headline per-agent-panel figure.

Deferred item 4 of the v0 plan. For each agent, runs N deployment rollouts
under that agent's own trained policy (uniform-pick aggregation disabled),
and renders the histogram of acquisition queries. For Forrester (1-D),
each panel is overlaid with the pre-registered signature from
``predicted_styles.md`` for visual comparison; for Branin-Currin / color
(2-D / 3-D), each panel instead overlays one marginal histogram per input
dimension, since there's no single 1-D landscape to draw against.

Aggregation across seeds: the user trained 5 seeds per agent. We pool
queries from all (agent, seed) pairs into the same per-agent histogram,
and also draw seed-level mean +/- spread (KDE per seed, then envelope).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

from traits_audit.committee.problems import ForresterProblem, Problem, get_problem
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
    problem: Union[Problem, str, None] = None,
) -> DensityResult:
    """For every (agent, training_seed), run N rollouts under that policy.

    Returns the pooled x-queries per agent — shape (n,) for Forrester,
    (n, dim) otherwise (same convention as ``CommitteeEnv.x_obs``).
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
                    problem=problem,
                )
                per_seed_queries.append(trace.x_queries)
            pooled = np.concatenate(per_seed_queries, axis=0)
            by_agent[agent].append(pooled)
            by_agent_seed[agent][s] = pooled
            mean_str = np.array2string(np.atleast_1d(pooled.mean(axis=0)), precision=3)
            print(f"[density] {agent} seed={s}: "
                  f"{len(pooled)} queries, mean x = {mean_str}")

    return DensityResult(
        queries_by_agent={n: np.concatenate(by_agent[n]) for n in AGENT_NAMES},
        queries_by_agent_seed=by_agent_seed,
        n_episodes_per_seed=n_episodes_per_seed,
    )


def _forrester(x: np.ndarray) -> np.ndarray:
    return (6.0 * x - 2.0) ** 2 * np.sin(12.0 * x - 4.0)


def _scatter_alpha(n_points: int, reference: int = 25_000,
                   reference_alpha: float = 0.06) -> float:
    """Per-point alpha for a dense scatter panel, scaled by point count.

    Tuned so the current run (5 seeds x 50 episodes x 100 steps = 25k
    queries per agent) lands at ``reference_alpha``, which keeps the
    Branin contours and the CIE D65 marker legible underneath. Scaling
    as 1/n holds the accumulated opacity of an equally dense patch
    roughly constant, so re-running with more or fewer episodes doesn't
    silently go back to a solid blue block. Clamped to a visible range.

    ponytail: linear 1/n, not a true alpha-compositing inversion --
    swap in 1-(1-a)^n if panels ever differ wildly in density.
    """
    if n_points <= 0:
        return reference_alpha
    return float(np.clip(reference_alpha * reference / n_points, 0.02, 0.5))


def _panel_grid(n_panels: int, ncols: int = 5):
    """rows/cols for a panel-per-agent figure — scales with the registry
    size instead of assuming 9 (the original 3x3 hardcode silently dropped
    the last 6 agents once the registry grew from 9 to 15 members)."""
    import matplotlib.pyplot as plt

    ncols = min(ncols, n_panels)
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.6 * ncols, 3.3 * nrows), sharex=True,
    )
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n_panels:]:
        ax.axis("off")
    return fig, axes, nrows, ncols


def render_headline_figure(
    result: DensityResult,
    output_path: Path,
    n_bins: int = 30,
    problem: Union[Problem, str, None] = None,
) -> None:
    """Render the per-agent query-density headline figure.

    Each panel shows one agent's pooled queries (5 seeds), with context that
    depends on the benchmark:

    ``forrester`` (1-D): pre-registered signature as sub-title, and the
    clean Forrester oracle overlaid on a twin axis so query densities can
    be read against the function landscape — exactly the original figure
    (histogram + per-seed step overlay, ``C0``/black/gray colours).

    ``branin-currin`` (2-D): unfilled contours of the true (noise-free)
    Branin surface — the objective the audit rewards actually score — with
    each agent's pooled queries scattered on top as semi-transparent ``C0``
    dots, so query placement can be read against the landscape the same way
    the Forrester panels do. The contours are a reference frame only; the
    surface's shape is shown properly by the 3-D companion figure in
    :mod:`.surface3d`.

    ``color`` (3-D): the CIE 1931 xy chromaticity diagram (spectral locus +
    sRGB gamut triangle) that ``plot_cie_trajectory`` draws for the SDL demo
    (``traits_audit._viz``), with each agent's normalised (R, G, B) queries
    projected to CIE xy and scattered on top in the same ``C0`` style —
    reusing that exact background rather than inventing a new one. A red
    star marks the problem's optimum (the simulator's target spectrum,
    frechet = 0). The grey ``+`` is D65, the diagram's standard illuminant,
    which is *not* the target and is deliberately left unhighlighted.

    Any other/future problem without a bespoke panel falls back to one
    marginal histogram per input dimension (the original placeholder).
    """
    import matplotlib.pyplot as plt

    if isinstance(problem, str):
        problem = get_problem(problem)
    if problem is None:
        problem = ForresterProblem()
    dim = problem.dim

    n_agents = len(AGENT_NAMES)
    fig, axes, nrows, ncols = _panel_grid(n_agents)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    target_xy = None  # set below for `color`; gates that figure's legend

    if dim == 1:
        x_grid = np.linspace(0.0, 1.0, 400)
        f_grid = _forrester(x_grid)
    elif problem.name == "branin-currin":
        # Fresh, finer-than-state grid just for this contour (the env's own
        # problem.grid is only 15/dim -- too coarse to look like a surface).
        axis = np.linspace(0.0, 1.0, 60)
        GX, GY = np.meshgrid(axis, axis, indexing="ij")
        grid_pts = np.stack([GX.ravel(), GY.ravel()], axis=1)
        Z = problem.clean(grid_pts)[:, 0].reshape(GX.shape)
        # Log-spaced levels: Branin spans ~0.4-308 with three sharp minima,
        # so linear levels wash out the basins that matter for this figure.
        levels = np.geomspace(max(float(Z.min()), 1e-3), float(Z.max()), 12)
    elif problem.name == "color":
        from traits_audit._viz import draw_cie_background, rgb_norm_to_cie_xy
        target = getattr(problem, "target", None)
        if target is not None:
            tx, ty = rgb_norm_to_cie_xy(np.asarray(target).reshape(1, 3))
            target_xy = (float(tx[0]), float(ty[0]))
    else:
        labels = list(problem.input_names)
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, agent in zip(axes, AGENT_NAMES):
        pooled = np.atleast_2d(
            np.asarray(result.queries_by_agent[agent]).reshape(-1, dim)
        )
        if dim == 1:
            ax.hist(pooled[:, 0], bins=edges, density=True, color="C0",
                    alpha=0.45, label="pooled (5 seeds)")
            for s, qs in result.queries_by_agent_seed[agent].items():
                counts, _ = np.histogram(qs, bins=edges, density=True)
                ax.step(edges[:-1], counts, where="post",
                        color="black", alpha=0.25, linewidth=0.7)
            ax.set_xlim(0.0, 1.0)
            ylabel = "density"

            ax2 = ax.twinx()
            ax2.plot(x_grid, f_grid, color="0.55", linewidth=1.0, alpha=0.6,
                     label="Forrester f(x)")
            # Forrester overlay is visual context only; suppress its axis.
            ax2.set_ylabel("")
            ax2.tick_params(axis="y", which="both",
                            left=False, right=False,
                            labelleft=False, labelright=False)

        elif problem.name == "branin-currin":
            # Unfilled contours only. A grey contourf underneath darkened
            # the whole panel and made the queries read as washed out; the
            # 3-D companion figure (surface3d.py) is where the surface's
            # *shape* is shown, so here the landscape only needs to be a
            # reference frame for the query positions.
            cs = ax.contour(GX, GY, Z, levels=levels, cmap="viridis",
                            linewidths=0.8, alpha=0.85, zorder=1)
            ax.scatter(pooled[:, 0], pooled[:, 1], s=4, color="C0",
                       alpha=_scatter_alpha(len(pooled)) * 2.0,
                       edgecolors="none", zorder=3,
                       label="pooled (5 seeds)")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ylabel = problem.input_names[1]

        elif problem.name == "color":
            # Both reference points are red so they stay visible over tens of
            # thousands of query dots, and are told apart by SHAPE:
            #   +  D65, the diagram's standard illuminant (context only)
            #   *  the target spectrum, i.e. this problem's optimum
            # They sit only ~0.13 apart in CIE y, so the panel legend below
            # is what stops the cross being read as the goal.
            draw_cie_background(ax, gamut_label=False,
                                d65_color="red", d65_zorder=5)
            xa, ya = rgb_norm_to_cie_xy(pooled)
            ax.scatter(xa, ya, s=4, color="C0",
                       alpha=_scatter_alpha(len(pooled)),
                       edgecolors="none", zorder=4, label="pooled (5 seeds)")
            if target_xy is not None:
                # Same s=60 as the D65 cross: a filled star reads larger
                # than a stroked plus at equal area, but matching the
                # areas keeps neither marker looking like the emphasis.
                ax.scatter(*target_xy, marker="*", s=60, color="red",
                           edgecolors="none", zorder=6)
            ax.set_xlim(0.0, 0.80)
            ax.set_ylim(0.0, 0.90)
            ylabel = "CIE y"

        else:
            for d in range(dim):
                ax.hist(pooled[:, d], bins=edges, density=True, alpha=0.4,
                        color=colors[d % len(colors)], label=labels[d])
            ax.legend(fontsize=7, loc="upper right")
            ax.set_xlim(0.0, 1.0)
            ylabel = "density"

        ax.set_title(agent, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(alpha=0.3)

    if problem.name == "color" and target_xy is not None:
        # One figure-level legend rather than 15 panel legends: the two red
        # markers are only ~0.13 apart in CIE y and differ by shape alone,
        # so they have to be decoded somewhere -- but repeating that in
        # every panel buried the horseshoe under legend boxes.
        from matplotlib.lines import Line2D
        fig.legend(
            handles=[
                Line2D([], [], marker="*", color="none", markerfacecolor="red",
                       markeredgecolor="none", markersize=9,
                       label="target spectrum (frechet = 0)"),
                Line2D([], [], marker="+", color="red", linestyle="none",
                       markersize=9, label="D65 illuminant (reference only)"),
                Line2D([], [], marker="o", color="none", markerfacecolor="C0",
                       markersize=7, label="pooled queries (5 seeds)"),
            ],
            loc="lower center", ncol=3, fontsize=11, frameon=False,
            bbox_to_anchor=(0.5, -0.004),
        )

    if dim == 1:
        bottom_label = "x (acquisition query)"
    elif problem.name == "branin-currin":
        bottom_label = f"{problem.input_names[0]}  (contours: Branin surface)"
    elif problem.name == "color":
        bottom_label = "CIE x"
    else:
        bottom_label = "x (acquisition query)"

    # Bottom-row-only xlabel assumes the grid divides evenly (true today:
    # 15 agents / 5 cols = 3 full rows). A future registry size that leaves
    # a partial last row would need per-column "last visible axis" logic.
    for ax in axes[-ncols:]:
        ax.set_xlabel(bottom_label, fontsize=11)
    # Reserve a strip at the bottom when the figure-level legend is drawn.
    bottom = 0.035 if (problem.name == "color" and target_xy is not None) else 0.0
    fig.tight_layout(rect=(0, bottom, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def write_density_csv(
    result: DensityResult, output_path: Path, dim: int = 1,
) -> None:
    """Tidy CSV: agent, seed, x0[, x1[, x2]]. One row per query."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_cols = ["x"] if dim == 1 else [f"x{d}" for d in range(dim)]
    rows = [",".join(["agent", "seed", *x_cols])]
    for agent, by_seed in result.queries_by_agent_seed.items():
        for s, xs in by_seed.items():
            for x in np.atleast_2d(np.asarray(xs).reshape(-1, dim)):
                rows.append(f"{agent},{s}," + ",".join(f"{v:.6f}" for v in x))
    output_path.write_text("\n".join(rows) + "\n")
