"""Figure generation for regime-aware stability and Lyapunov-exponent chaos
probing of trajectories.

Pure rendering: every function takes plain numpy/dict data and an ``out_dir``,
and returns the saved figure path. No DMDc/PyDMD fitting, regime-splitting,
or hypothesis-test logic lives here.

Data are grouped by a ``policy`` label (the inner grouping, e.g. an
acquisition policy); label order and colors/markers come from an injectable
:class:`~traits_audit.stability.viz.style.PlotStyle`. Every public function
renders under the publication ``rcParams`` (see
:func:`~traits_audit.stability.viz.common.publication_style`) without
changing global matplotlib state. Axis labels that are domain specific
(iteration index, the name of the traced quantity) are keyword parameters.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import (
    DOUBLE_COL,
    SINGLE_COL,
    box as _box,
    draw_unit_circle as _draw_unit_circle,
    publication_style,
    save_fig as _save,
    set_square as _set_square,
)
from .style import PlotStyle, resolve_style

__all__ = [
    "plot_early_vs_late_lambda_max",
    "plot_lle_distribution",
    "plot_local_linearity_evolution",
    "plot_rolling_lyapunov_evolution",
    "plot_trend_evolution",
    "plot_trend_rolling_lyapunov_evolution",
    "plot_local_linear_stability_spectrum",
    "plot_effective_rank_evolution",
    "plot_poles",
]


def _color_violins_by_policy(violin_result: dict, policies: list[str], style: PlotStyle) -> None:
    """Recolor a matplotlib ``ax.violinplot(...)`` result per-policy,
    using the style's policy colors -- ``violinplot`` has no
    built-in per-body color argument, so every body/median/min/max/bar
    defaults to one uniform color unless recolored after the fact.
    Without this, the x-axis policy LABELS would be the only per-policy cue
    (position only, no color).

    Parameters
    ----------
    violin_result : the dict returned by ``ax.violinplot(...)``
        (keys: "bodies", "cmeans"/"cmedians"/"cmaxes"/"cmins"/"cbars",
        whichever were requested).
    policies : the policy name at each violin position, in the SAME
        left-to-right order the data was passed to ``violinplot`` in
        (violinplot has no policy-name argument of its own, so this list
        must exactly match the caller's own ``data = [d[p] for p in
        policies]`` construction).
    style : supplies the per-policy colors.
    """
    for body, policy in zip(violin_result["bodies"], policies):
        color = style.color(policy)
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.5)
    for key in ("cmeans", "cmedians", "cmaxes", "cmins", "cbars"):
        collection = violin_result.get(key)
        if collection is None:
            continue
        colors = [style.color(p) for p in policies]
        collection.set_color(colors)


# ── early vs. late lambda_max per policy ─────────────────────────────────

@publication_style
def plot_early_vs_late_lambda_max(
    early_late_by_policy: dict,
    out_dir: Path,
    *,
    ylabel: str = "median |λ_max| (windowed, PyDMD)",
    xticklabels: tuple[str, str] = ("early regime", "late regime"),
    style: PlotStyle | None = None,
) -> Path:
    """``fig1``: paired early-regime -> late-regime median |lambda_max|,
    one line per policy -- the headline visual for a regime-shift finding.

    Parameters
    ----------
    early_late_by_policy : dict[policy] -> {"early": list[float], "late": list[float]}
    """
    style = resolve_style(style)
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, SINGLE_COL))
    x = [0, 1]
    for policy in style.policies(early_late_by_policy):
        d = early_late_by_policy[policy]
        early_vals = [v for v in d["early"] if np.isfinite(v)]
        late_vals = [v for v in d["late"] if np.isfinite(v)]
        if not early_vals or not late_vals:
            continue
        y = [np.median(early_vals), np.median(late_vals)]
        color = style.color(policy)
        marker = style.marker(policy)
        ax.plot(x, y, marker=marker, color=color, label=policy, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(list(xticklabels))
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    _box(ax)
    ax.legend(frameon=False, fontsize=7, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    return _save(fig, out_dir, "fig1_early_vs_late_lambda_max")


# ── Lyapunov-exponent distribution ───────────────────────────────────────

@publication_style
def plot_lle_distribution(
    lle_by_policy: dict, out_dir: Path, *, style: PlotStyle | None = None
) -> Path:
    """``fig3``: whole-trajectory Lyapunov-exponent (LLE) distribution by
    policy, violin style, with a horizontal reference line at LLE=0 -- the
    chaos boundary (a credibly positive LLE is chaos-indicative; the
    reference line is what makes that boundary visually explicit, since
    |lambda_max| plots have no equivalent single-value boundary to draw
    this way).

    Parameters
    ----------
    lle_by_policy : dict[policy] -> array of per-trajectory LLE point estimates
    """
    style = resolve_style(style)
    policies = style.policies(lle_by_policy)
    data = [lle_by_policy[p] for p in policies]
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, SINGLE_COL))
    ax.axhline(0, color="k", lw=1.0, alpha=0.6, linestyle="--")
    violin_result = ax.violinplot(data, showmedians=True)
    _color_violins_by_policy(violin_result, policies, style)
    ax.set_xticks(range(1, len(policies) + 1))
    ax.set_xticklabels(policies, rotation=30, ha="right")
    ax.set_ylabel("Lyapunov exponent (LLE)")
    ax.grid(True, alpha=0.3)
    _box(ax)
    fig.tight_layout()
    return _save(fig, out_dir, "fig3_lle_distribution")


# ── local linearity evolution over time ──────────────────────────────────

@publication_style
def plot_local_linearity_evolution(
    lambda_max_evolution: dict,
    r2_evolution: dict,
    out_dir: Path,
    *,
    xlabel: str = "AL iteration",
    lambda_max_ylabel: str = "|λ_max(t)| (windowed, PyDMD)",
    style: PlotStyle | None = None,
) -> Path:
    """``fig4``: the "tangent line" view of the trajectory -- two stacked
    panels sharing the same x-axis (iteration index), one line per
    policy: top panel is R^2(t) (how good the local linear "tangent" is
    at each point), bottom panel is |lambda_max(t)| (that tangent's
    "slope" -- is the local linearization contracting or expanding?).
    Reading them together is the point: a low R^2(t) means the
    lambda_max(t) value at that same point isn't a trustworthy stability
    read, since the local linear fit itself is poor there.

    R^2's y-axis is clipped to [-2, 1.05]: early in the trajectory the
    windowed fit has very few points relative to its rank (the window
    hasn't yet grown to full size), which is a near-singular regime that
    can send R^2 to enormous negative values (R^2 is unbounded below --
    unlike, e.g., a correlation coefficient). That's a real, expected
    numerical property of a poor fit, not a bug, but left unclipped it
    swamps the y-axis and makes every other, well-behaved point on the
    curve illegible. Clipping is a display-only choice.

    Parameters
    ----------
    lambda_max_evolution, r2_evolution : dict[policy] -> (t_array, median,
        q25, q75), median-over-seeds curves plus the interquartile spread
        across seeds/datasets at each timestep, shown as a shaded band --
        shows not just the typical curve but how much runs actually
        disagree.
    """
    style = resolve_style(style)
    fig, (ax_r2, ax_lm) = plt.subplots(
        2, 1, figsize=(DOUBLE_COL, DOUBLE_COL * 0.8), sharex=True
    )
    for policy in style.policies(lambda_max_evolution):
        color = style.color(policy)
        t_r2, r2_vals, r2_q25, r2_q75 = r2_evolution[policy]
        ax_r2.plot(t_r2, r2_vals, color=color, alpha=0.8, label=policy)
        ax_r2.fill_between(t_r2, r2_q25, r2_q75, color=color, alpha=0.10, linewidth=0)
        t_lm, lm_vals, lm_q25, lm_q75 = lambda_max_evolution[policy]
        ax_lm.plot(t_lm, lm_vals, color=color, alpha=0.8)
        ax_lm.fill_between(t_lm, lm_q25, lm_q75, color=color, alpha=0.10, linewidth=0)
    ax_r2.set_ylim(-2, 1.05)
    ax_r2.axhline(0.5, color="k", lw=0.8, alpha=0.4, linestyle="--")
    ax_r2.set_ylabel("windowed R² (t)")
    ax_r2.grid(True, alpha=0.3)
    _box(ax_r2)
    ax_lm.axhline(1.0, color="k", lw=0.8, alpha=0.4, linestyle="--")
    # Same early-transient-blowup rationale as R^2's clip above: a
    # percentile-based (not hardcoded) cap, since |lambda_max|'s "normal"
    # scale varies by dataset/policy, unlike R^2's natural [.,1] ceiling.
    all_lm = np.concatenate([v[1] for v in lambda_max_evolution.values()])
    finite_lm = all_lm[np.isfinite(all_lm)]
    if len(finite_lm):
        ax_lm.set_ylim(0, max(2.0, 1.2 * np.percentile(finite_lm, 95)))
    ax_lm.set_xlabel(xlabel)
    ax_lm.set_ylabel(lambda_max_ylabel)
    ax_lm.grid(True, alpha=0.3)
    _box(ax_lm)
    ax_r2.legend(frameon=False, fontsize=7, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    return _save(fig, out_dir, "fig4_local_linearity_evolution")


# ── rolling Lyapunov exponent evolution ──────────────────────────────────

@publication_style
def plot_rolling_lyapunov_evolution(
    rolling_lle_pooled: dict,
    out_dir: Path,
    *,
    xlabel: str = "AL iteration (rolling-window end)",
    style: PlotStyle | None = None,
) -> Path:
    """``fig5``: rolling Lyapunov-exponent estimate vs. iteration, one
    line per policy, with a horizontal reference at LLE=0 (the chaos
    boundary). Visualizes whether the system becomes progressively less
    prone to sensitive-dependence/chaotic divergence as the run
    converges ("settles onto an attractor"): the lines should trend
    downward (ideally crossing below zero) as iteration increases, rather
    than staying flat or trending upward.

    Parameters
    ----------
    rolling_lle_pooled : dict[policy] -> (t_array, median, q25, q75),
        median-over-seeds curve plus interquartile spread.
    """
    style = resolve_style(style)
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, SINGLE_COL))
    ax.axhline(0, color="k", lw=1.0, alpha=0.6, linestyle="--")
    for policy in style.policies(rolling_lle_pooled):
        color = style.color(policy)
        t, lle, q25, q75 = rolling_lle_pooled[policy]
        ax.plot(t, lle, color=color,
                marker=style.marker(policy), markersize=3,
                alpha=0.8, label=policy)
        ax.fill_between(t, q25, q75, color=color, alpha=0.10, linewidth=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("rolling Lyapunov exponent (LLE)")
    ax.grid(True, alpha=0.3)
    _box(ax)
    ax.legend(frameon=False, fontsize=7, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    return _save(fig, out_dir, "fig5_rolling_lyapunov_evolution")


# ── trend evolution over time ────────────────────────────────────────────

@publication_style
def plot_trend_evolution(
    trend_evolution: dict,
    out_dir: Path,
    *,
    xlabel: str = "AL iteration",
    ylabel: str = "background-drift trend\n(mean ECM charge parameter)",
    style: PlotStyle | None = None,
) -> Path:
    """``fig6``: a background-drift TREND's value vs. iteration, one line
    per policy -- the FULL, untrimmed trend from t=0 (the trend is defined
    over the whole trajectory, not a windowed derivative of it, so
    trimming it would arbitrarily discard the earliest part).

    Parameters
    ----------
    trend_evolution : dict[policy] -> (t_array, median, q25, q75),
        time-aligned, starting at t=0, plus the interquartile spread across
        seeds/datasets.
    xlabel, ylabel : axis labels (the defaults reproduce the original
        battery-experiment wording; pass your own quantity's name).
    """
    style = resolve_style(style)
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, SINGLE_COL))
    for policy in style.policies(trend_evolution):
        color = style.color(policy)
        t, vals, q25, q75 = trend_evolution[policy]
        ax.plot(t, vals, color=color, alpha=0.85, label=policy)
        ax.fill_between(t, q25, q75, color=color, alpha=0.10, linewidth=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    _box(ax)
    ax.legend(frameon=False, fontsize=7, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    return _save(fig, out_dir, "fig6_trend_evolution")


# ── trend rolling-Lyapunov evolution ─────────────────────────────────────

@publication_style
def plot_trend_rolling_lyapunov_evolution(
    trend_rolling_lle_pooled: dict,
    out_dir: Path,
    *,
    xlabel: str = "AL iteration (rolling-window end)",
    ylabel: str = "rolling Lyapunov exponent\nof background-drift trend",
    style: PlotStyle | None = None,
) -> Path:
    """``fig7``: exact structural mirror of :func:`plot_rolling_lyapunov_evolution`,
    applied to the background-drift TREND instead of the raw
    (detrended-window) signal. Does the trend's own rolling Lyapunov
    exponent settle/collapse (trend downward, ideally negative) as the run
    progresses, stay periodic-like, or keep diverging? Horizontal
    reference line at LLE=0.

    Parameters
    ----------
    trend_rolling_lle_pooled : dict[policy] -> (t_array, median, q25,
        q75), pooled identically to ``rolling_lle_pooled``, on the
        trend series instead of the raw signal.
    """
    style = resolve_style(style)
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, SINGLE_COL))
    ax.axhline(0, color="k", lw=1.0, alpha=0.6, linestyle="--")
    for policy in style.policies(trend_rolling_lle_pooled):
        color = style.color(policy)
        t, lle, q25, q75 = trend_rolling_lle_pooled[policy]
        ax.plot(t, lle, color=color,
                marker=style.marker(policy), markersize=3,
                alpha=0.8, label=policy)
        ax.fill_between(t, q25, q75, color=color, alpha=0.10, linewidth=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    _box(ax)
    ax.legend(frameon=False, fontsize=7, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    return _save(fig, out_dir, "fig7_trend_rolling_lyapunov_evolution")


# ── local linear stability spectrum ──────────────────────────────────────

@publication_style
def plot_local_linear_stability_spectrum(
    log_eig_spectrum_by_policy: dict, out_dir: Path, *, style: PlotStyle | None = None
) -> Path:
    """Supplementary diagnostic (cf. Margazoglou & Magri, 2023): the FULL
    per-window log|eigenvalue| spread of the windowed DMD fit, not just the
    max (lambda_max), pooled by policy.

    NON-NEGOTIABLE FRAMING CAVEAT: this is NOT a Lyapunov spectrum and
    must never be labeled or read as one. It is the same class of object
    that is insufficient to detect chaos (eigenvalues of a fitted
    linear/piecewise-linear operator can only ever show exponential
    growth/decay/rotation, never the nonlinear stretch-and-fold that
    produces genuine sensitive dependence) -- reporting the full window's
    eigenvalue set instead of only the largest does not change that. Kept
    here strictly as a "local linear stability spectrum".

    Parameters
    ----------
    log_eig_spectrum_by_policy : dict[policy] -> flattened array of every
        finite log|eig| value from every window of every trajectory for
        that policy (not time-aligned).

    Y-axis is clipped to the 1st-99th percentile of the POOLED values,
    padded 10% -- a few near-singular windows (rank-deficient fits early in
    a trajectory, or windows where a genuinely near-zero eigenvalue sends
    log|eig| to a large negative value) can otherwise swamp the axis.
    Display-only.
    """
    style = resolve_style(style)
    policies = style.policies(log_eig_spectrum_by_policy)
    data = [log_eig_spectrum_by_policy[p] for p in policies]
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, SINGLE_COL))
    ax.axhline(0, color="k", lw=0.8, alpha=0.4, linestyle="--")
    violin_result = ax.violinplot(data, showmedians=True)
    _color_violins_by_policy(violin_result, policies, style)
    ax.set_xticks(range(1, len(policies) + 1))
    ax.set_xticklabels(policies, rotation=30, ha="right")
    ax.set_ylabel("local linear stability spectrum\nlog|eig(A_window)| (NOT a Lyapunov spectrum)",
                   fontsize=8)
    all_vals = np.concatenate(data) if data else np.array([])
    finite_vals = all_vals[np.isfinite(all_vals)]
    if len(finite_vals):
        lo, hi = np.percentile(finite_vals, [1, 99])
        pad = 0.1 * (hi - lo) if hi > lo else 1.0
        ax.set_ylim(lo - pad, hi + pad)
    ax.grid(True, alpha=0.3)
    _box(ax)
    fig.tight_layout()
    return _save(fig, out_dir, "fig9_local_linear_stability_spectrum")


# ── effective DMDc rank evolution ────────────────────────────────────────

@publication_style
def plot_effective_rank_evolution(
    effective_rank_evolution: dict,
    out_dir: Path,
    *,
    xlabel: str = "AL iteration",
    style: PlotStyle | None = None,
) -> Path:
    """``fig10``: per-window RESOLVED DMDc rank vs. iteration, one line per
    policy -- lets a reader directly check whether some policy needs fewer
    modes than another. Applies identically whether the windowed fit used a
    fixed rank (where the resolved rank can still fall below the requested
    cap on short/early windows -- the truncation is
    min(requested, actual_matrix_rank)) or an energy-based rank (where the
    resolved rank varies by how much real signal each window's
    singular-value spectrum actually has).

    A dedicated figure rather than a 3rd panel on the local-linearity
    figure: that one has a defined "tangent-line" narrative (slope +
    goodness-of-fit) that a 3rd, related-but-distinct quantity would
    dilute -- "how many modes does this window need" is its own question.

    Parameters
    ----------
    effective_rank_evolution : dict[policy] -> (t_array, median, q25,
        q75), time-aligned (same convention as ``lambda_max_evolution`` /
        ``r2_evolution`` of :func:`plot_local_linearity_evolution`), plus
        the interquartile spread across seeds/datasets.
    """
    style = resolve_style(style)
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, SINGLE_COL))
    for policy in style.policies(effective_rank_evolution):
        color = style.color(policy)
        t, ranks, q25, q75 = effective_rank_evolution[policy]
        ax.plot(t, ranks, color=color,
                marker=style.marker(policy), markersize=3,
                alpha=0.8, label=policy)
        ax.fill_between(t, q25, q75, color=color, alpha=0.10, linewidth=0)
    ax.axhline(1, color="k", lw=0.8, alpha=0.4, linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("effective DMDc rank (windowed fit)")
    ax.grid(True, alpha=0.3)
    _box(ax)
    ax.legend(frameon=False, fontsize=7, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    return _save(fig, out_dir, "fig10_effective_rank_evolution")


# ── pole plot ────────────────────────────────────────────────────────────

@publication_style
def plot_poles(
    raw_eigs_by_policy: dict, out_dir: Path, *, style: PlotStyle | None = None
) -> Path:
    """``fig11``: complex eigenvalues of the windowed DMDc fit, pooled
    across every window of every trajectory, one color-coded dot cloud
    per policy on a single unit-circle panel.

    The windowed fit produces one eigenvalue set PER WINDOW, so this figure
    pools ALL windows' poles into one dot cloud per policy on a SINGLE
    panel. Reading it: points inside the unit circle are locally
    contracting, outside are locally expanding; a nonzero imaginary part
    indicates local oscillatory (rotating) behavior -- a genuinely
    different question from a Lyapunov-exponent-based chaos test
    (eigenvalues of a fitted linear operator can show growth/decay/rotation
    but never confirm or rule out genuine nonlinear sensitive dependence).

    Parameters
    ----------
    raw_eigs_by_policy : dict[policy] -> flattened complex array of every
        eigenvalue from every window of every trajectory for that policy
        (not time-aligned).

    Axis limits are clipped to the 1st-99th percentile of the POOLED
    real/imaginary parts, padded 10%: a handful of near-singular
    early windows can produce eigenvalues orders of magnitude outside the
    unit-circle scale this figure exists to show, and left unclipped they
    swamp the axes into an unreadable sliver at the origin. Display-only.
    """
    style = resolve_style(style)
    # DOUBLE_COL: an up-to-9-entry external legend needs the extra width.
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, SINGLE_COL))
    _draw_unit_circle(ax)
    all_real, all_imag = [], []
    for policy in style.policies(raw_eigs_by_policy):
        eigs = raw_eigs_by_policy[policy]
        ax.scatter(eigs.real, eigs.imag, c=[style.color(policy)],
                   marker=style.marker(policy), s=10, alpha=0.35,
                   linewidths=0, label=policy)
        all_real.extend(eigs.real)
        all_imag.extend(eigs.imag)
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.axvline(0, color="k", lw=0.5, alpha=0.3)
    all_real = np.asarray(all_real)
    all_imag = np.asarray(all_imag)
    finite_real = all_real[np.isfinite(all_real)]
    finite_imag = all_imag[np.isfinite(all_imag)]
    if len(finite_real) and len(finite_imag):
        re_lo, re_hi = np.percentile(finite_real, [1, 99])
        im_lo, im_hi = np.percentile(finite_imag, [1, 99])
        # Keep the unit circle itself always in view regardless of the
        # percentile bounds, since it's the whole point of reference.
        re_lo, re_hi = min(re_lo, -1.2), max(re_hi, 1.2)
        im_lo, im_hi = min(im_lo, -1.2), max(im_hi, 1.2)
        re_pad = 0.1 * (re_hi - re_lo)
        im_pad = 0.1 * (im_hi - im_lo)
        ax.set_xlim(re_lo - re_pad, re_hi + re_pad)
        ax.set_ylim(im_lo - im_pad, im_hi + im_pad)
    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    _set_square(ax)
    ax.grid(True, alpha=0.3)
    _box(ax)
    ax.legend(frameon=False, fontsize=6, bbox_to_anchor=(1.02, 1.0), loc="upper left",
              markerscale=2)
    fig.tight_layout()
    return _save(fig, out_dir, "fig11_pole_plot")
