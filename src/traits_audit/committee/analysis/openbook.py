"""Two objectives as the facing pages of an open book.

A companion to the flat twin-contour panels in :mod:`.density`, for
two-objective problems. Both landscapes are drawn as contour planes hinged at
90 degrees -- one on the floor, one on the back wall -- with the *same* query
cloud projected flat onto each. Reading it: a cloud sitting in a floor basin
while the wall shows it high on the second objective is an agent chasing one
objective at the other's expense.

Why this and not just the flat version: with two contour families on one set
of axes, legibility rests entirely on the two objectives having different
topologies (Branin's closed rings vs Currin's smooth ramp). Giving each its
own plane removes that dependency -- no amount of visual similarity between
the landscapes can make them ambiguous. The cost is real and worth stating:
perspective compresses the far half of each page, so comparing *densities*
across agents is harder here than on the flat panels, and a 3-D scatter goes
opaque well below the full 25k queries, so points are subsampled. Use this as
the explanatory figure and the flat panels as the density workhorse.

Implementation note: matplotlib's ``contour(..., zdir=...)`` re-interprets the
Z array as *positions* rather than values, so it cannot place a value-contour
onto an arbitrary plane. The contours are therefore computed on a throwaway
2-D figure and their path vertices placed into each plane by hand, via
``allsegs``/``to_rgba`` (``.collections`` was removed in matplotlib 3.10).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from traits_audit.committee.analysis.density import AGENT_NAMES, DensityResult
from traits_audit.committee.problems import Problem, get_problem


#: Colormaps for the two pages. Distinct hue families so the floor and the
#: wall stay separable even where the two landscapes overlap in shape.
FLOOR_CMAP = "autumn"
WALL_CMAP = "winter"


def _contour_paths(GX, GY, Z, levels, cmap):
    """2-D contour computed off-screen -> [(verts (N,2), rgba)].

    Kept separate from the drawing so each plane's geometry is computed once
    and reused across all 15 panels rather than per panel.
    """
    import matplotlib.pyplot as plt

    tmp = plt.figure()
    try:
        ax = tmp.add_subplot(111)
        cs = ax.contour(GX, GY, Z, levels=levels, cmap=cmap)
        colors = cs.to_rgba(cs.levels)
        out = [
            (np.asarray(v), color)
            for segs, color in zip(cs.allsegs, colors)
            for v in segs
            if len(v) > 1
        ]
    finally:
        plt.close(tmp)
    return out


def render_openbook_figure(
    result: DensityResult,
    output_path: Path,
    problem: Union[Problem, str, None] = None,
    grid_n: int = 60,
    max_points: int = 1200,
    elev: float = 20.0,
    azim: float = -60.0,
    seed: int = 0,
) -> None:
    """Per-agent open-book panel: objective 0 on the floor, objective 1 on the wall.

    Parameters
    ----------
    result
        Pooled queries, as produced by ``run_density_rollouts`` or read back
        from ``query_density.csv``.
    problem
        A 2-D, 2-objective benchmark.
    max_points
        Markers drawn per plane, subsampled without replacement. A 3-D scatter
        turns into an opaque sheet well before the full query count.
    elev, azim
        Camera. The default looks into the hinge so both pages are legible.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers '3d')

    if isinstance(problem, str):
        problem = get_problem(problem)
    if problem is None:
        raise ValueError("render_openbook_figure needs an explicit problem")
    if problem.dim != 2 or problem.n_objectives != 2:
        raise ValueError(
            f"render_openbook_figure needs a 2-D, 2-objective problem; "
            f"{problem.name!r} is {problem.dim}-D with "
            f"{problem.n_objectives} objective(s)."
        )

    axis = np.linspace(0.0, 1.0, grid_n)
    GX, GY = np.meshgrid(axis, axis, indexing="ij")
    clean = problem.clean(np.stack([GX.ravel(), GY.ravel()], axis=1))
    Z0 = clean[:, 0].reshape(GX.shape)
    Z1 = clean[:, 1].reshape(GX.shape)

    # Objective 0 (Branin-like) spans decades and needs log levels or its
    # basins vanish; objective 1 does not.
    lv0 = np.geomspace(max(float(Z0.min()), 1e-3), float(Z0.max()), 9)
    lv1 = np.linspace(float(Z1.min()), float(Z1.max()), 9)
    floor = _contour_paths(GX, GY, Z0, lv0, FLOOR_CMAP)
    wall = _contour_paths(GX, GY, Z1, lv1, WALL_CMAP)

    rng = np.random.default_rng(seed)
    n_agents = len(AGENT_NAMES)
    ncols = 5
    nrows = int(np.ceil(n_agents / ncols))
    fig = plt.figure(figsize=(4.2 * ncols, 4.15 * nrows))

    n_total = len(next(iter(result.queries_by_agent.values())))
    for i, agent in enumerate(AGENT_NAMES):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        pooled = np.asarray(result.queries_by_agent[agent]).reshape(-1, 2)
        shown = pooled
        if len(pooled) > max_points:
            shown = pooled[rng.choice(len(pooled), max_points, replace=False)]
        x1, x2 = shown[:, 0], shown[:, 1]

        # Floor page, z = 0: objective 0 over (x1, x2).
        for verts, color in floor:
            ax.plot(verts[:, 0], verts[:, 1], 0, color=color, lw=0.9, zorder=2)
        ax.scatter(x1, x2, np.zeros_like(x1), s=2, c="C0", alpha=0.22,
                   edgecolors="none", depthshade=False, zorder=1)

        # Wall page, y = 1: objective 1 over the same (x1, x2), with x2 now
        # drawn vertically. Same queries, mirrored.
        for verts, color in wall:
            ax.plot(verts[:, 0], np.ones(len(verts)), verts[:, 1],
                    color=color, lw=0.9, zorder=2)
        ax.scatter(x1, np.ones_like(x1), x2, s=2, c="C0", alpha=0.22,
                   edgecolors="none", depthshade=False, zorder=1)

        # Hide only the third (x-normal) pane, so the result reads as two
        # hinged pages rather than the corner of a box. The remaining panes,
        # grid and ticks are kept -- without them the planes lose their scale.
        ax.xaxis.pane.set_visible(False)
        for a3 in (ax.xaxis, ax.yaxis, ax.zaxis):
            a3.pane.set_edgecolor("0.85")
            a3.pane.set_alpha(0.5)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_zticks([0, 0.5, 1])
        ax.tick_params(labelsize=6, pad=-3)
        ax.set_xlabel(f"${problem.input_names[0]}$", fontsize=8, labelpad=-8)
        ax.set_ylabel(f"${problem.input_names[1]}$", fontsize=8, labelpad=-8)
        ax.set_zlabel(f"${problem.input_names[1]}$", fontsize=8, labelpad=-8)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, 1, 1))
        ax.set_title(agent, fontsize=10, pad=-1)

    fig.suptitle(
        f"Query placement against both objectives — floor: "
        f"{problem.objective_names[0]}, back wall: {problem.objective_names[1]}. "
        f"{min(max_points, n_total):,} of {n_total:,} queries shown per panel.",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
