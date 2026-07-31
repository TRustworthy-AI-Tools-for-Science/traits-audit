"""Figures for Thread B (votes-as-features) and Thread A (aggregators).

B1: paired regret curves (LCB vs LCB+votes, max-sigma vs max-sigma+votes).
B2: leave-one-out terminal-regret ablation for LCB+votes.
B2-max-sigma: analogous leave-one-out ablation for max-sigma+votes.
A1: aggregator regret curves with paired Wilcoxon p-values in the subheader.
A2: weight-vs-solo-regret scatter for both weighting schemes.
A3: committee disagreement (action-std) over time twin-axed with the
    winning aggregator's regret curve.

All plots use :mod:`.style` for the Okabe-Ito palette and consistent fonts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from traits_audit.committee.analysis import style as st
from traits_audit.committee.analysis.thread_regret import (
    ThreadResult,
    paired_terminal_test,
)


# ---------------------------------------------------------------------------
# Shared helpers.
# ---------------------------------------------------------------------------

def _band(ax, xs, arr, label, color, lw=1.6, alpha=1.0, ls="-",
          alpha_band=0.0, zorder=2):
    """Line + optional CI band. alpha_band=0 disables the band."""
    mean = arr.mean(axis=0)
    ax.plot(xs, mean, label=label, color=color, lw=lw, ls=ls,
            alpha=alpha, zorder=zorder)
    if alpha_band > 0:
        sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        ax.fill_between(
            xs, mean - 1.96 * sem, mean + 1.96 * sem,
            color=color, alpha=alpha_band, zorder=zorder - 1,
        )


def _format_p(p: float) -> str:
    """Compact p-value string for subheaders."""
    if p < 1e-3:
        return f"p={p:.1e}"
    return f"p={p:.3f}"


# ---------------------------------------------------------------------------
# Thread B figures.
# ---------------------------------------------------------------------------

def render_b1(result: ThreadResult, output_path: Path) -> dict:
    """B1: paired regret curves for the 4 baselines.

    Subheader: paired Wilcoxon p-values for both LCB+votes vs LCB and
    max-sigma+votes vs max-sigma.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6.2))
    xs = np.arange(result.episode_length)

    _band(ax, xs, result.per_policy_regret["random"], "random",
          **st.RANDOM_STYLE, zorder=1)
    # max-σ pair (vermillion): dashed baseline + solid headline winner.
    _band(ax, xs, result.per_policy_regret["max-sigma"], "max-σ",
          color=st.HEADLINE, lw=2.0, ls="--", zorder=2)
    _band(ax, xs, result.per_policy_regret["max-sigma+votes"],
          "max-σ+votes", color=st.HEADLINE, lw=2.6, ls="-", zorder=5)
    # LCB pair (Okabe-Ito blue): dashed baseline + solid +votes.
    _band(ax, xs, result.per_policy_regret["LCB"], "LCB",
          color=st.BLUE, lw=2.0, ls="--", zorder=3)
    _band(ax, xs, result.per_policy_regret["LCB+votes"], "LCB+votes",
          color=st.BLUE, lw=2.6, ls="-", zorder=4)

    test_lcb = paired_terminal_test(result, "LCB+votes", "LCB")
    test_msv = paired_terminal_test(result, "max-sigma+votes", "max-sigma")

    subheader = (
        f"max-σ+votes vs max-σ: {_format_p(test_msv['p_value'])} "
        f"(mean {test_msv['a_mean']:.3f} vs {test_msv['b_mean']:.3f})\n"
        f"LCB+votes vs LCB: {_format_p(test_lcb['p_value'])} "
        f"(mean {test_lcb['a_mean']:.3f} vs {test_lcb['b_mean']:.3f})"
    )

    ax.set_xlabel("acquisition step", fontsize=st.LABEL_FS)
    ax.set_ylabel("simple regret", fontsize=st.LABEL_FS)
    # Title + subheader in a single string so matplotlib lays them out together.
    ax.set_title(
        f"Votes-as-features paired regret ({len(result.seeds)} seeds)\n"
        + subheader,
        fontsize=st.TITLE_FS,
    )
    ax.title.set_multialignment("center")

    ax.legend(fontsize=st.LEGEND_FS, loc="lower left", framealpha=0.92)
    st.style_axes(ax, xlim=(0, 100), ylog=True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return {"LCB+votes_vs_LCB": test_lcb, "MaxSigma+votes_vs_MaxSigma": test_msv}


def render_ablation(
    ablation: dict[str, np.ndarray],
    baseline_terminal: np.ndarray,
    agent_names: list[str],
    output_path: Path,
    headline: str,
) -> None:
    """Leave-one-agent-out ablation bars for a vote-augmented policy.

    Bar height = (mean terminal SR with k dropped) minus (mean terminal SR
    of the full ``headline`` policy). Positive => removing k *hurts*.
    """
    import matplotlib.pyplot as plt

    baseline_mean = float(np.mean(baseline_terminal))
    deltas = {
        name: float(np.mean(arr) - baseline_mean) for name, arr in ablation.items()
    }
    order = sorted(agent_names, key=lambda n: deltas[n], reverse=True)
    vals = [deltas[n] for n in order]
    colors = [st.VERMILLION if v > 0 else st.BLUISH_GREEN for v in vals]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y_pos = np.arange(len(order))
    ax.barh(y_pos, vals, color=colors, alpha=0.88)
    ax.axvline(0, color=st.BLACK, lw=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(order, fontsize=st.TICK_FS)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta$ terminal SR  (drop-k minus full committee)",
                  fontsize=st.LABEL_FS)
    headline_display = headline.replace("max-sigma", "max-σ")
    ax.set_title(
        f"Committee-member importance (leave-one-out for {headline_display})",
        fontsize=st.TITLE_FS,
    )
    ax.tick_params(axis="x", labelsize=st.TICK_FS)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def render_b2(ablation, baseline_terminal, agent_names, output_path):
    """Back-compat wrapper for the LCB+votes ablation."""
    render_ablation(ablation, baseline_terminal, agent_names, output_path,
                    headline="LCB+votes")


# ---------------------------------------------------------------------------
# Thread A figures.
# ---------------------------------------------------------------------------

def render_a1(
    result: ThreadResult,
    output_path: Path,
    reference: str = "best-solo:PITUniformity",
) -> dict:
    """A1: regret curves for all aggregators; paired Wilcoxon p in subheader.

    Returns dict mapping aggregator -> paired test against ``reference``.
    """
    import matplotlib.pyplot as plt

    aggregator_order = [
        "random",
        reference,
        "committee:uniform",
        "committee:agree",
        "committee:disagree",
        "committee:weighted-invreg",
        "committee:weighted-indep",
    ]
    # Each entry: (color, lw, ls). CI bands are off everywhere in this view.
    style_map = {
        "random":                     (st.NEUTRAL_FAINT, 2.0, ":"),
        reference:                    (st.BLACK,         2.2, "-"),
        "committee:uniform":          (st.NEUTRAL_LIGHT, 2.0, "--"),
        "committee:agree":            (st.ORANGE,        2.0, "-"),
        "committee:disagree":         (st.SKY_BLUE,      2.0, "-"),
        "committee:weighted-invreg":  (st.HEADLINE,      2.6, "-"),
        "committee:weighted-indep":   (st.REDDISH_PURPLE,2.0, "-"),
    }

    fig, ax = plt.subplots(figsize=(11, 6.5))
    xs = np.arange(result.episode_length)
    for name in aggregator_order:
        if name not in result.per_policy_regret:
            continue
        color, lw, ls = style_map[name]
        _band(ax, xs, result.per_policy_regret[name], name,
              color=color, lw=lw, ls=ls,
              zorder=3 if name == reference else 2)

    # Compute p-values vs reference for the subheader.
    tests: dict[str, dict] = {}
    bar_names = [n for n in aggregator_order if n in result.per_policy_regret
                 and n != reference]
    for n in bar_names:
        tests[n] = paired_terminal_test(result, n, reference)

    short = {
        "committee:weighted-invreg": "weighted-invreg",
        "committee:weighted-indep":  "weighted-indep",
        "committee:uniform":         "uniform",
        "committee:agree":           "agree",
        "committee:disagree":        "disagree",
        "random":                    "random",
    }
    pieces = []
    for n in bar_names:
        t = tests[n]
        pieces.append(f"{short.get(n, n)} {_format_p(t['p_value'])}")
    # Split subheader across two lines for readability.
    half = (len(pieces) + 1) // 2
    subheader = (
        f"vs {reference}:  " + "   ".join(pieces[:half]) + "\n"
        + "   ".join(pieces[half:])
    )

    ax.set_xlabel("acquisition step", fontsize=st.LABEL_FS)
    ax.set_ylabel("simple regret", fontsize=st.LABEL_FS)
    ax.set_title(
        f"Aggregator regret ({len(result.seeds)} seeds)\n" + subheader,
        fontsize=st.TITLE_FS,
    )
    ax.title.set_multialignment("center")
    ax.legend(fontsize=st.LEGEND_FS, loc="lower left", ncol=2,
              framealpha=0.92)
    st.style_axes(ax, xlim=(0, 100), ylog=True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return tests


def render_a2(
    indep_w: dict[str, float],
    invreg_w: dict[str, float],
    solo_regret_terminal: dict[str, float],
    output_path: Path,
) -> None:
    """A2: scatter of aggregator weight (normalized) vs solo terminal SR.

    Two panels, one per weighting scheme. Negative trend = good weighting.
    """
    import matplotlib.pyplot as plt

    agents = list(solo_regret_terminal.keys())

    def _normalize(w):
        s = sum(w[a] for a in agents)
        return {a: w[a] / s for a in agents}

    indep_n = _normalize(indep_w)
    invreg_n = _normalize(invreg_w)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, w, title in zip(
        axes,
        [indep_n, invreg_n],
        ["independence-weighted", "inverse-regret-weighted"],
    ):
        for a in agents:
            ax.scatter(w[a], solo_regret_terminal[a], s=110,
                       color=st.BLUE, alpha=0.88, edgecolor=st.BLACK,
                       linewidth=0.6)
            ax.annotate(a, (w[a], solo_regret_terminal[a]),
                        xytext=(7, 7), textcoords="offset points",
                        fontsize=st.ANNOT_FS)
        ax.axvline(1.0 / len(agents), color=st.NEUTRAL_LIGHT, ls=":",
                   lw=1.4, label=f"uniform 1/{len(agents)}")
        ax.set_xlabel("aggregator weight (normalized)", fontsize=st.LABEL_FS)
        ax.set_title(title, fontsize=st.TITLE_FS)
        ax.tick_params(axis="both", labelsize=st.TICK_FS)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=st.LEGEND_FS, loc="best")
    axes[0].set_ylabel("solo terminal SR (lower = better)", fontsize=st.LABEL_FS)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def render_a3(
    result: ThreadResult,
    output_path: Path,
    overlay_policy: Optional[str] = None,
) -> None:
    """A3: committee action-std vs the winning aggregator's regret.

    Twin-axed: left = std of preferred actions, right = log SR.
    """
    import matplotlib.pyplot as plt

    xs = np.arange(result.episode_length)
    std_arr = result.per_policy_action_std.get("committee:uniform")
    if std_arr is None or np.all(np.isnan(std_arr)):
        std_arr = result.per_policy_action_std.get("committee:agree")
    std_mean = np.nanmean(std_arr, axis=0)

    if overlay_policy is None:
        T = result.episode_length - 1
        candidates = {
            k: v[:, T].mean() for k, v in result.per_policy_regret.items()
            if k.startswith("committee:")
        }
        overlay_policy = min(candidates, key=candidates.get)

    overlay_arr = result.per_policy_regret[overlay_policy]
    overlay_mean = overlay_arr.mean(axis=0)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(xs, std_mean, color=st.BLUE, lw=2.2,
             label="mean committee action-std")
    ax1.set_xlabel("acquisition step", fontsize=st.LABEL_FS)
    ax1.set_ylabel("std of 9 preferred actions",
                   color=st.BLUE, fontsize=st.LABEL_FS)
    ax1.tick_params(axis="y", labelcolor=st.BLUE, labelsize=st.TICK_FS)
    ax1.tick_params(axis="x", labelsize=st.TICK_FS)
    ax1.set_xlim(0, 100)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(xs, overlay_mean, color=st.HEADLINE, lw=2.4,
             label=f"{overlay_policy} mean SR")
    ax2.set_yscale("log")
    ax2.set_ylabel(f"{overlay_policy} simple regret (log)",
                   color=st.HEADLINE, fontsize=st.LABEL_FS)
    ax2.tick_params(axis="y", labelcolor=st.HEADLINE, labelsize=st.TICK_FS)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               fontsize=st.LEGEND_FS, loc="lower left", framealpha=0.92)

    ax1.set_title(f"Disagreement diagnostic  ({len(result.seeds)} seeds)",
                  fontsize=st.TITLE_FS)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
