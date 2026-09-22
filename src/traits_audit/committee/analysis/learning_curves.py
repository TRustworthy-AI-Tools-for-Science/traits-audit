"""Learning-curve figure for the SAC committee members.

For each agent: load ``rollout/ep_rew_mean`` from the TensorBoard event file
of every (agent, seed) run, apply a light EMA (alpha=0.6, TB-style),
interpolate to a shared step grid, and plot median + IQR band across seeds.
Styling matches :mod:`density.py` (panel grid, title 15, label 13, tick 11).
Dimension-agnostic: TB scalar logs carry no reference to input dimension, so
this file needs no problem-specific handling — it works unchanged for any
benchmark trained via ``ta-committee-train --problem ...``.

Each agent's reward stream was z-scored on its own running statistics during
training, so absolute reward magnitudes are NOT comparable across panels.
The figure answers "is this agent learning?" per panel, not "which agent
earned more."

Missing (agent, seed) logs — e.g. a still-running or since-failed HPC task —
are skipped with a printed warning rather than crashing the whole figure;
an agent with zero available seeds is dropped from the figure entirely.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from traits_audit.committee.rewards import REWARD_REGISTRY


AGENT_NAMES: list[str] = list(REWARD_REGISTRY.keys())


@dataclass
class LearningCurveResult:
    """Per-agent learning curves on a shared step grid.

    Attributes
    ----------
    step_grid : np.ndarray, shape (G,)
        Common step axis (intersection across all (agent, seed) ranges).
    smoothed_by_agent_seed : dict[str, dict[int, np.ndarray]]
        EMA-smoothed reward per agent and seed, on ``step_grid``.
    raw_by_agent_seed : dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]
        Pre-interpolation (steps, values) for diagnostics.
    """

    step_grid: np.ndarray
    smoothed_by_agent_seed: dict[str, dict[int, np.ndarray]]
    raw_by_agent_seed: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]


def _load_scalar(event_dir: Path, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, values) for ``tag`` from a TB event directory.

    The directory should contain the single ``SAC_1/`` (or equivalent)
    subdirectory holding ``events.out.tfevents.*``.
    """
    from tensorboard.backend.event_processing import event_accumulator

    # SB3 writes events.out under {run_dir}/SAC_1/
    candidates = list(event_dir.glob("SAC_*"))
    if not candidates:
        raise FileNotFoundError(f"No SAC_* subdir in {event_dir}")
    inner = candidates[0]
    ea = event_accumulator.EventAccumulator(
        str(inner),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        raise KeyError(f"Tag {tag!r} not in {inner}; "
                       f"available: {ea.Tags()['scalars']}")
    events = ea.Scalars(tag)
    steps = np.asarray([e.step for e in events], dtype=float)
    values = np.asarray([e.value for e in events], dtype=float)
    return steps, values


def _ema(values: np.ndarray, alpha: float) -> np.ndarray:
    """TensorBoard-style EMA: smoothed[t] = alpha * smoothed[t-1] + (1-alpha) * v[t]."""
    out = np.empty_like(values)
    if len(values) == 0:
        return out
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * out[i - 1] + (1.0 - alpha) * values[i]
    return out


def load_learning_curves(
    tb_dir: Path,
    seeds: list[int],
    tag: str = "rollout/ep_rew_mean",
    smoothing_alpha: float = 0.6,
    n_grid: int = 500,
) -> LearningCurveResult:
    """Load and align every (agent, seed) curve that's actually available.

    The shared step grid spans the intersection of all *loaded* step ranges
    so we never extrapolate. Linear interpolation onto ``n_grid`` evenly
    spaced points within that intersection.
    """
    raw: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    s_min = -np.inf
    s_max = np.inf
    for agent in AGENT_NAMES:
        agent_raw: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for s in seeds:
            run_dir = tb_dir / f"{agent}_seed{s}"
            try:
                steps, values = _load_scalar(run_dir, tag)
            except (FileNotFoundError, KeyError) as exc:
                print(f"[learning-curves] skipping {agent} seed={s}: {exc}")
                continue
            agent_raw[s] = (steps, values)
            s_min = max(s_min, float(steps.min()))
            s_max = min(s_max, float(steps.max()))
        if agent_raw:
            raw[agent] = agent_raw
    if not raw:
        raise FileNotFoundError(f"No (agent, seed) TB logs found under {tb_dir}")
    if s_min > s_max:
        raise ValueError(
            f"Loaded runs' step ranges don't overlap (max start {s_min:.0f} > "
            f"min end {s_max:.0f}) — likely a still-running or partial job."
        )

    grid = np.linspace(s_min, s_max, n_grid)
    smoothed_per_seed: dict[str, dict[int, np.ndarray]] = {a: {} for a in raw}
    for agent, agent_raw in raw.items():
        for s, (steps, values) in agent_raw.items():
            smooth = _ema(values, smoothing_alpha)
            smoothed_per_seed[agent][s] = np.interp(grid, steps, smooth)

    return LearningCurveResult(
        step_grid=grid,
        smoothed_by_agent_seed=smoothed_per_seed,
        raw_by_agent_seed=raw,
    )


def render_learning_curves_figure(
    result: LearningCurveResult,
    output_path: Path,
    title_tag: str = "rollout/ep_rew_mean",
) -> None:
    """Panel grid: median + IQR band across seeds per agent.

    Styling mirrors :func:`density.render_headline_figure` — same panel
    sizing (scales with agent count, not a hardcoded 3x3), same
    title/label/tick fontsizes, light grid. Only agents with at least one
    loaded seed appear (see ``load_learning_curves``).
    """
    import matplotlib.pyplot as plt
    from traits_audit.committee.analysis import style as st
    from traits_audit.committee.analysis.density import _panel_grid

    agents = list(result.smoothed_by_agent_seed.keys())

    # Project font: serif (matches traits_audit._viz._RCPARAMS). Scoped to
    # this figure via rc_context so we don't mutate global rcParams.
    with plt.rc_context({"font.family": "serif"}):
        fig, axes, nrows, ncols = _panel_grid(len(agents))
        # Scale step axis to thousands.
        grid_K = result.step_grid / 1e3
        x_max = float(grid_K.max())

        for ax, agent in zip(axes, agents):
            per_seed = result.smoothed_by_agent_seed[agent]
            stacked = np.stack(list(per_seed.values()), axis=0)   # (n_seeds, G)
            median = np.median(stacked, axis=0)
            q25 = np.quantile(stacked, 0.25, axis=0)
            q75 = np.quantile(stacked, 0.75, axis=0)

            n_seeds = len(per_seed)
            ax.fill_between(grid_K, q25, q75, color=st.BLUE, alpha=0.22,
                            label=f"IQR ({n_seeds} seed{'s' if n_seeds != 1 else ''})")
            ax.plot(grid_K, median, color=st.BLUE, lw=2.2,
                    label="median")
            ax.axhline(0.0, color=st.NEUTRAL_LIGHT, lw=0.9, ls=":")
            ax.set_xlim(0, x_max)
            ax.set_title(agent, fontsize=st.TITLE_FS)
            ax.set_ylabel("smoothed reward", fontsize=st.LABEL_FS)
            ax.tick_params(axis="both", labelsize=st.TICK_FS)
            ax.grid(alpha=0.3)

        # See density.py's _panel_grid caller for the same even-grid caveat.
        for ax in axes[-ncols:]:
            ax.set_xlabel("training steps (K)", fontsize=st.LABEL_FS)

        fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def write_learning_curves_csv(
    result: LearningCurveResult, output_path: Path
) -> None:
    """Tidy CSV: agent, seed, step, smoothed_reward (one row per grid point)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["agent,seed,step,smoothed_reward"]
    for agent, per_seed in result.smoothed_by_agent_seed.items():
        for s, vals in per_seed.items():
            for st_val, v in zip(result.step_grid, vals):
                rows.append(f"{agent},{s},{int(st_val)},{v:.6f}")
    output_path.write_text("\n".join(rows) + "\n")
