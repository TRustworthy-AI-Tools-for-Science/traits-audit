"""Per-channel query marginals for the colour-matching benchmark.

The headline density figure plots colour queries on the CIE 1931
chromaticity diagram, which is the natural way to *show* colour but is a
3 -> 2 projection that divides out intensity. Two things are invisible in
it:

  * the whole R = G = B diagonal of the RGB cube collapses onto the single
    D65 point, so structure appears to converge there that is not really
    converging at all;
  * a policy pinned against a channel's upper bound looks unremarkable.

That second one matters here: 12 of the 15 agents spend 75-100% of their
queries at B > 0.95. The audited objective is very nearly a function of B
alone (Spearman rho with frechet: B +0.83, R +0.03, G -0.12), and the
surrogate's standardised residual |y-mu|/sigma is *lowest* in the high-B
band -- i.e. the surrogate looks best calibrated in the region furthest
from the optimum. The calibration-style rewards therefore pay agents to
sit exactly where the true objective is worst.

This module draws that directly: one panel per agent, one histogram per
input channel, with the target's channel values marked. It is a separate
figure (``query_channels.png``) rather than a replacement for the CIE one
-- the chromaticity view still shows *which colours* are being sampled,
which the marginals cannot.

Reuses ``DensityResult`` from :mod:`.density`, so it renders from the
``query_density.csv`` already on disk without re-running any rollouts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from traits_audit.committee.analysis import style as st
from traits_audit.committee.analysis.density import (
    AGENT_NAMES,
    DensityResult,
    _panel_grid,
)
from traits_audit.committee.problems import Problem, get_problem


# One colour per input channel. For the colour problem these are deliberately
# the literal R/G/B rather than the Okabe-Ito palette: a reader should not
# have to consult a legend to know which curve is the blue channel. Other
# problems fall back to the accessible palette.
RGB_CHANNEL_COLORS = ("#D55E00", "#009E73", "#0072B2")  # Okabe-Ito vermillion/green/blue


def render_channel_marginals(
    result: DensityResult,
    output_path: Path,
    problem: Union[Problem, str, None] = None,
    n_bins: int = 40,
    ceiling: float = 0.95,
) -> None:
    """One panel per agent; one histogram per input channel.

    Parameters
    ----------
    result
        Pooled queries, as produced by ``run_density_rollouts`` or read back
        from ``query_density.csv``.
    problem
        Benchmark whose ``input_names``/``dim`` label the axes; also supplies
        ``target`` (if it has one) for the dashed reference lines.
    ceiling
        Queries above this fraction of a channel's range count as "at the
        bound" for the per-panel annotation. The annotation is what makes
        the saturation legible at a glance.
    """
    import matplotlib.pyplot as plt

    if isinstance(problem, str):
        problem = get_problem(problem)
    if problem is None:
        raise ValueError("render_channel_marginals needs an explicit problem")

    dim = problem.dim
    names = list(problem.input_names)
    colors = (RGB_CHANNEL_COLORS if names == ["R", "G", "B"]
              else st.OKABE_ITO[1:1 + dim])
    target = getattr(problem, "target", None)

    fig, axes, nrows, ncols = _panel_grid(len(AGENT_NAMES))
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    for ax, agent in zip(axes, AGENT_NAMES):
        pooled = np.asarray(result.queries_by_agent[agent]).reshape(-1, dim)
        for d in range(dim):
            ax.hist(pooled[:, d], bins=edges, density=True, histtype="step",
                    linewidth=1.8, color=colors[d], label=names[d], zorder=3)
        if target is not None:
            for d in range(dim):
                ax.axvline(float(target[d]), color=colors[d], ls=":",
                           lw=1.2, alpha=0.9, zorder=2)

        # Name the saturated channels in the panel: this is the whole point
        # of the figure, and reading it off a log-ish histogram is fiddly.
        at_bound = [(names[d], (pooled[:, d] > ceiling).mean() * 100)
                    for d in range(dim)]
        hot = [f"{n} {pc:.0f}%" for n, pc in at_bound if pc >= 50.0]
        if hot:
            ax.text(0.03, 0.95, "at bound: " + ", ".join(hot),
                    transform=ax.transAxes, fontsize=8, va="top",
                    color=st.VERMILLION,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec="none", alpha=0.75))

        ax.set_title(agent, fontsize=13)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylabel("density", fontsize=11)
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(alpha=0.3)

    for ax in axes[-ncols:]:
        ax.set_xlabel("normalised channel value", fontsize=11)

    handles = [
        plt.Line2D([], [], color=colors[d], lw=1.8, label=names[d])
        for d in range(dim)
    ]
    if target is not None:
        handles.append(plt.Line2D([], [], color="0.35", ls=":", lw=1.4,
                                  label="target value"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.004))

    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
