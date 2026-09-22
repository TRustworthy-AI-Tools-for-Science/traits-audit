"""3-D surface view of where each agent queries on a 2-D benchmark.

Companion to the contour panels in :mod:`.density`, not a replacement. The
contour view answers "where in the input square?"; this one answers "at what
*objective value* did the agent spend its queries?" -- the vertical axis
carries information the top-down view has to encode as shading, which 25k
semi-transparent dots tend to bury.

Each query is lifted onto the true (noise-free) surface, so a point's height
is the value that query actually returned. Reading the figure: points in the
basins are good, points on the ridge are wasted.

Two rendering decisions the Branin surface forces:

* **log z.** Branin spans 0.4 to 308 on [0, 1]^2, nearly three decades. On a
  linear axis the three minima -- the whole point of the benchmark -- are
  crushed into an invisible film on the floor, so the z-axis is log-scaled
  by plotting log10(z) and relabelling the ticks in original units.
* **subsampling.** 25k markers per panel in a 3-D axes is slow to render and
  reads as an opaque shell that hides the surface underneath. A seeded
  random subsample keeps the distribution honest while leaving the surface
  visible; ``max_points`` controls it, and the panel says how many are shown.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from traits_audit.committee.analysis import style as st
from traits_audit.committee.analysis.density import AGENT_NAMES, DensityResult
from traits_audit.committee.problems import Problem, get_problem


def _log_ticks(zmin: float, zmax: float) -> tuple[list[float], list[str]]:
    """Decade ticks in log10 space, labelled in the original units."""
    lo, hi = int(np.floor(np.log10(zmin))), int(np.ceil(np.log10(zmax)))
    vals = [v for v in range(lo, hi + 1)]
    labels = [f"$10^{{{v}}}$" if abs(v) > 2 else f"{10.0 ** v:g}" for v in vals]
    return [float(v) for v in vals], labels


def render_surface3d(
    result: DensityResult,
    output_path: Path,
    problem: Union[Problem, str, None] = None,
    grid_n: int = 70,
    max_points: int = 1500,
    elev: float = 28.0,
    azim: float = -128.0,
    seed: int = 0,
    log_z: bool = True,
) -> None:
    """Per-agent 3-D surface with that agent's queries drawn on it.

    Parameters
    ----------
    result
        Pooled queries, as produced by ``run_density_rollouts`` or read back
        from ``query_density.csv``.
    problem
        A 2-D benchmark (only ``dim == 2`` can be drawn this way).
    grid_n
        Surface mesh resolution. Independent of the problem's own state grid
        -- this is for looking at, not for the agent.
    max_points
        Markers drawn per panel, subsampled without replacement.
    elev, azim
        Camera. The default looks down the ridge so all three Branin basins
        stay visible.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers '3d')

    if isinstance(problem, str):
        problem = get_problem(problem)
    if problem is None:
        raise ValueError("render_surface3d needs an explicit problem")
    if problem.dim != 2:
        raise ValueError(
            f"render_surface3d is for 2-D problems; {problem.name!r} is "
            f"{problem.dim}-D. Use the density panels instead."
        )

    axis = np.linspace(0.0, 1.0, grid_n)
    GX, GY = np.meshgrid(axis, axis, indexing="ij")
    Z = problem.clean(np.stack([GX.ravel(), GY.ravel()], axis=1))[:, 0]
    Z = Z.reshape(GX.shape)

    # Guard the log transform against a non-positive floor (Branin's is
    # ~0.398, but Currin or a future 2-D problem need not be).
    floor = float(Z[Z > 0].min()) if np.any(Z > 0) else 1.0
    def _z(v):
        return np.log10(np.maximum(v, floor)) if log_z else v

    Zp = _z(Z)
    rng = np.random.default_rng(seed)
    # "In a basin" = bottom decile of the surface's own value distribution,
    # so the threshold adapts to the problem instead of being a magic number.
    basin_cut = float(np.percentile(Z, 10))

    n = len(AGENT_NAMES)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(4.1 * ncols, 3.7 * nrows))

    for i, agent in enumerate(AGENT_NAMES):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        # Wireframe-ish surface: a light mesh reads as "the landscape"
        # without the opaque fill that a dense marker cloud then hides.
        ax.plot_surface(GX, GY, Zp, cmap="viridis", alpha=0.30,
                        linewidth=0, antialiased=True, rstride=1, cstride=1,
                        zorder=1)
        ax.contour(GX, GY, Zp, levels=8, colors="0.35", linewidths=0.4,
                   alpha=0.5, offset=float(Zp.min()), zorder=0)

        pooled = np.asarray(result.queries_by_agent[agent]).reshape(-1, 2)
        shown = pooled
        if len(pooled) > max_points:
            shown = pooled[rng.choice(len(pooled), max_points, replace=False)]
        zq = problem.clean(shown)[:, 0]
        # Dark = a good (low) query, light = a wasted one. Colouring by
        # height rather than a flat red gives the cloud a depth cue, so
        # "points in the basins" is readable without rotating the figure.
        ax.scatter(shown[:, 0], shown[:, 1], _z(zq),
                   s=3.5, c=_z(zq), cmap="autumn", alpha=0.55,
                   edgecolors="none", depthshade=False, zorder=5)

        ax.set_title(agent, fontsize=12, pad=0)
        ax.set_xlabel(problem.input_names[0], fontsize=9, labelpad=-6)
        ax.set_ylabel(problem.input_names[1], fontsize=9, labelpad=-6)
        ax.tick_params(axis="both", labelsize=7, pad=-2)
        ax.view_init(elev=elev, azim=azim)
        if log_z:
            ticks, labels = _log_ticks(floor, float(Z.max()))
            ax.set_zticks(ticks)
            ax.set_zticklabels(labels, fontsize=7)
        # Two numbers the eye cannot reliably read off a 3-D cloud: the
        # typical queried value, and how often the agent actually got into
        # a basin (bottom decile of the surface).
        in_basin = (zq <= basin_cut).mean() * 100
        ax.text2D(0.02, 0.92, f"median f = {np.median(zq):,.1f}",
                  transform=ax.transAxes, fontsize=8, color=st.BLACK)
        ax.text2D(0.02, 0.85, f"in basin = {in_basin:.1f}%",
                  transform=ax.transAxes, fontsize=8, color=st.VERMILLION)

    obj = problem.objective_names[0]
    fig.suptitle(
        f"Where each agent queried, on the true {obj} surface  "
        f"(z log-scaled; {max_points:,} of "
        f"{len(next(iter(result.queries_by_agent.values()))):,} queries shown; "
        f"\"in basin\" = f below {basin_cut:,.1f}, the surface's 10th percentile)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
