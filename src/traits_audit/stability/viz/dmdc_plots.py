"""Figure functions for DMDc-based dynamical-stability analyses.

Pure rendering: every function takes plain numpy/dict data and an ``out_dir``,
and returns the saved figure path(s). No DMDc fitting or hypothesis-test logic
lives here; see :mod:`traits_audit.dmdc` for the analysis itself.

Every multi-panel figure is split into one file per panel (``figX_a``,
``figX_b``, ...) so each PNG is a single, self-contained, publication-ready
figure rather than a composite. Figures that are already a single panel keep
a plain ``figX`` stem.

Data are grouped by two labels: ``policy`` (inner grouping) and ``model``
(outer grouping). Which labels are drawn, in which order, and with which
colors / markers / linestyles is controlled by the optional ``style``
argument (a :class:`~traits_audit.stability.viz.style.PlotStyle`); the default
style draws every label present in the data with automatic colors.
Rendering happens under the publication ``rcParams`` (see
:func:`traits_audit.stability.viz.common.publication_style`) without changing
global matplotlib state.

References
----------
[VCH06] Vachtsevanos et al. (2006), Intelligent Fault Diagnosis and
        Prognosis for Engineering Systems, Wiley, ISBN 978-0-471-72999-0.
        Safe-envelope/RUL framing: V_safe boundary in the stability plane.
[MOO18] Moosavi & Sandu (2018), Metrologia, 55(1):S1-S12,
        doi:10.1088/1681-7575/aa8f53. Motivates eigenvalue time series under
        finite-trajectory uncertainty.
[MOR81] Moore (1981), IEEE TAC, 26(1):17-32. Gramian-based observability/
        controllability: motivates plot_gramians.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Patch

from .common import (
    DOUBLE_COL,
    SINGLE_COL,
    box,
    draw_unit_circle,
    letters,
    maybe_log,
    publication_style,
    save_fig,
    set_square,
    signed_log_colorbar,
)
from .style import PlotStyle, resolve_style

__all__ = [
    "plot_poles",
    "plot_poles_combined",
    "plot_stability_plane",
    "plot_lyapunov_timeseries",
    "plot_lyapunov_all_models",
    "plot_gramians",
    "plot_convergence_robustness",
    "plot_matrix_heatmaps",
    "plot_coupling_network_combined",
    "plot_modal_equations",
    "plot_rank_dataset",
    "plot_dmdc_quality",
]

#: marker size per model in :func:`plot_poles_combined` (unknown models get 20)
_DEFAULT_MODEL_SIZES = {"Var-GPR": 12, "Var-FN": 22, "HSGP": 36}

_FRAGILITY_LABELS = [
    "high fragility (σ_min small — near-singular, sensitive to perturbation)",
    "medium fragility",
    "low fragility (σ_min large — far from any pole)",
]


@publication_style
def plot_poles(eig_dict: dict, out_dir: Path, stem: str = "fig1",
               style: PlotStyle | None = None) -> list[Path]:
    """Eigenvalues of post-hoc (and, where available, raw-state and native)
    A_r, one figure per model: ``{stem}_a``, ``{stem}_b``, ... (one letter per
    model drawn, in style order).

    Filled markers are the "posthoc" fit. Hollow markers (``raw_state``) are a
    secondary fit refit from each trajectory's saved raw state, a cross-check
    on whether using recorded vs. freshly re-queried uncertainty changes the
    fitted poles materially. The small grey '+' markers ("native") are an
    online-fitted A_r that may live on a wider, augmented state than
    posthoc/raw_state, so its eigenvalues are **not** on a directly comparable
    footing (deliberately drawn in one flat neutral color, not per-policy
    color, so they read as a separate reference layer) -- informational only.

    Parameters
    ----------
    eig_dict : dict[model] -> dict[policy] -> {"posthoc": complex ndarray,
        "raw_state": complex ndarray or None, "native": complex ndarray or None}

    References
    ----------
    [SUBSP21]: complex eigenvalue pairs from the general (non-symmetric) A_r.
    """
    style = resolve_style(style)
    models = style.models(eig_dict)
    paths = []
    for letter, model in zip(letters(len(models)), models):
        fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))
        draw_unit_circle(ax)
        has_native = False
        for policy in style.policies(eig_dict[model]):
            data = eig_dict[model].get(policy)
            if data is None:
                continue
            posthoc = data.get("posthoc")
            raw_state = data.get("raw_state")
            native = data.get("native")
            color = style.color(policy)
            marker = style.marker(policy)
            if posthoc is not None and len(posthoc):
                ax.scatter(posthoc.real, posthoc.imag, c=[color], marker=marker,
                           s=22, alpha=0.6, linewidths=0, label=policy)
            if raw_state is not None and len(raw_state):
                ax.scatter(raw_state.real, raw_state.imag, facecolors="none",
                           edgecolors=[color], marker=marker, s=40, linewidths=1.0)
            if native is not None and len(native):
                has_native = True
                ax.scatter(native.real, native.imag, c="0.5", marker="+",
                           s=18, alpha=0.5, linewidths=0.8)
        ax.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax.axvline(0, color="k", lw=0.5, alpha=0.3)
        ax.set_xlabel("Re(λ)")
        ax.set_ylabel("Im(λ)")
        ax.set_title(model)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        box(ax)
        handles, labels = ax.get_legend_handles_labels()
        if has_native:
            handles.append(plt.Line2D([0], [0], marker="+", color="0.5",
                                       linestyle="", markersize=5))
            labels.append("native (informational)")
        ax.legend(handles, labels, frameon=False, fontsize=7,
                  bbox_to_anchor=(1.02, 1.0), loc="upper left")
        fig.tight_layout()
        paths.append(save_fig(fig, out_dir, f"{stem}_{letter}"))
    return paths


@publication_style
def plot_poles_combined(eig_dict: dict, out_dir: Path,
                        model_sizes: dict | None = None,
                        style: PlotStyle | None = None) -> Path:
    """All policies × all models on one pole plot (fig1_d).

    Model encoded by marker size (``model_sizes``, default Var-GPR/Var-FN/HSGP
    = 12/22/36, any other model 20); policy by color/shape.

    References
    ----------
    [SUBSP21]: overview of pole placement across the full policy × model grid.
    """
    style = resolve_style(style)
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))
    draw_unit_circle(ax)
    size_by_model = _DEFAULT_MODEL_SIZES if model_sizes is None else model_sizes
    for model in style.models(eig_dict):
        for policy in style.policies(eig_dict[model]):
            data = eig_dict[model].get(policy)
            if data is None or data.get("posthoc") is None:
                continue
            eigs = data["posthoc"]
            if len(eigs) == 0:
                continue
            ax.scatter(eigs.real, eigs.imag, c=[style.color(policy)],
                       marker=style.marker(policy), s=size_by_model.get(model, 20),
                       alpha=0.5, linewidths=0)
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.axvline(0, color="k", lw=0.5, alpha=0.3)
    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    box(ax)
    handles = [plt.Line2D([0], [0], marker=style.marker(p), color="w",
                           markerfacecolor=style.color(p), markersize=6, label=p)
               for p in style.policies(pp for m in eig_dict.values() for pp in m)]
    ax.legend(handles=handles, frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    return save_fig(fig, out_dir, "fig1_d")


# ── fig2 — stability plane ──────────────────────────────────────────────────

@publication_style
def plot_stability_plane(plane_data: dict, out_dir: Path, cmap: str = "viridis",
                         style: PlotStyle | None = None) -> list[Path]:
    """Lyapunov V(s̃) contours in 2-D PCA space with per-policy trajectories,
    one figure per dataset: ``fig2_a``, ``fig2_b``, ...

    Parameters
    ----------
    plane_data : dict[dataset] -> {
        "gx", "gy", "lyap": meshgrid + V(x) values,
        "v_safe": float,
        "trajectories": dict[policy] -> {"pc1", "pc2", "lyap_deriv"},
    }

    References
    ----------
    [SUBSP21]: top-2 PCA directions as the stability-plane basis.
    [VCH18]  : safe-envelope boundary V = V_safe.
    """
    style = resolve_style(style)
    datasets = list(plane_data.keys())
    paths = []
    for ds in datasets:
        d = plane_data[ds]
        fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))
        cf = ax.contourf(d["gx"], d["gy"], d["lyap"], levels=15, cmap=cmap, alpha=0.6)
        ax.contour(d["gx"], d["gy"], d["lyap"], levels=[d["v_safe"]], colors="k", linewidths=1.2)
        legend_handles = []
        for policy, traj in d.get("trajectories", {}).items():
            color = style.color(policy, default=style.blue)
            ax.scatter(traj["pc1"], traj["pc2"], c=traj["lyap_deriv"], cmap=cmap,
                       s=14, edgecolors=[color], linewidths=0.6, vmin=-1, vmax=1)
            ax.plot(traj["pc1"][0], traj["pc2"][0], "o", color=color, ms=6)
            ax.plot(traj["pc1"][-1], traj["pc2"][-1], "*", color=color, ms=9)
            legend_handles.append(
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                           markeredgecolor=color, markersize=5, label=policy)
            )
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        # PC1 and PC2 have different natural scales (explained variance differs),
        # so _set_square makes the physical box square without constraining data aspect.
        set_square(ax)
        ax.grid(True, alpha=0.3)
        box(ax)
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label="V(s̃)")
        if legend_handles:
            ax.legend(handles=legend_handles, frameon=False, fontsize=6, ncol=2,
                      loc="upper left", bbox_to_anchor=(1.12, 1.0), title="Policy")
        fig.tight_layout()
        slug = ds.lower().replace("-", "_").replace(" ", "_")
        paths.append(save_fig(fig, out_dir, f"fig2_{slug}"))
    return paths


# ── fig3 — Lyapunov time series ─────────────────────────────────────────────

@publication_style
def plot_lyapunov_timeseries(series: dict, v_safe: float, out_dir: Path,
                             style: PlotStyle | None = None) -> list[Path]:
    """V(s̃_t) over AL iterations, one figure per GPR model: ``fig3_a``,
    ``fig3_b``, ``fig3_c``.

    Parameters
    ----------
    series : dict[model] -> dict[policy] -> {"mean", "std_epistemic", "std_aleatoric"}

    References
    ----------
    [DIE18]: bootstrap-derived uncertainty band on V(s̃_t).
    """
    style = resolve_style(style)
    models = style.models(series)
    paths = []
    for letter, model in zip(letters(len(models)), models):
        fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))
        all_means = []
        for policy in style.policies(series[model]):
            d = series[model].get(policy)
            if d is None:
                continue
            mean = np.asarray(d["mean"])
            all_means.extend(mean[np.isfinite(mean) & (mean > 0)].tolist())
            t = np.arange(len(mean))
            color = style.color(policy)
            ax.plot(t, mean, color=color, marker=style.marker(policy), ms=3, label=policy)
            std_a = np.asarray(d.get("std_aleatoric", np.zeros_like(mean)))
            ax.fill_between(t, mean - std_a, mean + std_a, color=color, alpha=0.10, linewidth=0)
            std_e = np.asarray(d.get("std_epistemic", np.zeros_like(mean)))
            ax.fill_between(t, mean - std_e, mean + std_e, color=color, alpha=0.20, linewidth=0)
        ax.axhline(v_safe, color="0.4", lw=0.8, ls="--")
        maybe_log(ax, all_means, axis="y")
        ax.set_xlabel("AL iteration")
        ax.set_ylabel("V(s̃)")
        ax.set_title(model)
        set_square(ax)
        ax.grid(True, alpha=0.3)
        box(ax)
        ax.legend(frameon=False, fontsize=7, ncol=2)
        fig.tight_layout()
        paths.append(save_fig(fig, out_dir, f"fig3_{letter}"))
    return paths


@publication_style
def plot_lyapunov_all_models(series: dict, out_dir: Path,
                             style: PlotStyle | None = None) -> Path:
    """All 3 GPR models overlaid per policy panel (fig3_d; line style = model,
    color = policy).

    References
    ----------
    [DIE18]: comparing identified-SSM Lyapunov trajectories across surrogates.
    """
    style = resolve_style(style)
    policies = style.policies(p for m in series for p in series[m])
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))
    all_means = []
    for model in style.models(series):
        for policy in policies:
            d = series[model].get(policy)
            if d is None:
                continue
            mean = np.asarray(d["mean"])
            all_means.extend(mean[np.isfinite(mean) & (mean > 0)].tolist())
            t = np.arange(len(mean))
            ax.plot(t, mean, color=style.color(policy), linestyle=style.linestyle(model),
                     lw=1.2, alpha=0.85)
    maybe_log(ax, all_means, axis="y")
    ax.set_xlabel("AL iteration")
    ax.set_ylabel("V(s̃)")
    set_square(ax)
    ax.grid(True, alpha=0.3)
    box(ax)
    policy_handles = [plt.Line2D([0], [0], color=style.color(p), label=p) for p in policies]
    model_handles = [plt.Line2D([0], [0], color="k", linestyle=style.linestyle(m), label=m)
                      for m in style.models(series)]
    leg1 = ax.legend(handles=policy_handles, frameon=False, fontsize=7, loc="upper left",
                      bbox_to_anchor=(1.02, 1.0), title="Policy")
    ax.add_artist(leg1)
    ax.legend(handles=model_handles, frameon=False, fontsize=7, loc="lower left",
              bbox_to_anchor=(1.02, 0.0), title="Model")
    fig.tight_layout()
    return save_fig(fig, out_dir, "fig3_d")


# ── fig4 — Gramians ──────────────────────────────────────────────────────────

@publication_style
def plot_gramians(gramians: dict, out_dir: Path,
                  stability_margin: dict | None = None,
                  style: PlotStyle | None = None) -> list[Path]:
    """log10 condition number of controllability/observability Gramians, one
    figure per GPR model: ``fig4_a``, ``fig4_b``, ``fig4_c``.

    Unstable policies (``stability_margin[policy] < 0``, i.e. spectral radius
    ≥ 1) are highlighted with a red x-axis label and a shaded column so the
    reader can immediately see which policies drove the system outside the
    stable regime.  Gramian values for those policies are computed on a
    rescaled (ρ = 0.99) version of A_r — they quantify
    controllability/observability structure but do not imply stability.

    Parameters
    ----------
    gramians : dict[model] -> dict[policy] -> {"cond_c", "cond_o"}
    stability_margin : dict[policy] -> float, optional
        ``1 − |λ_max(A_r)|`` per policy (averaged across seeds / models as
        appropriate).  Negative values flag unstable policies.  When omitted
        no stability annotation is drawn.

    References
    ----------
    [CHA25]: Gramian-based observability/controllability analysis.
    """
    style = resolve_style(style)
    models = style.models(gramians)
    paths = []
    for letter, model in zip(letters(len(models)), models):
        fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))
        policies = style.policies(gramians[model])
        x = np.arange(len(policies))
        cond_c = [np.log10(max(gramians[model][p]["cond_c"], 1e-12)) for p in policies]
        cond_o = [np.log10(max(gramians[model][p]["cond_o"], 1e-12)) for p in policies]
        ax.bar(x - 0.2, cond_c, width=0.4, label=r"$\mathrm{cond}(W_c)$", color=style.blue)
        ax.bar(x + 0.2, cond_o, width=0.4, label=r"$\mathrm{cond}(W_o)$", color=style.orange)

        # ── stability annotation ──────────────────────────────────────────
        unstable_added = False
        for i, policy in enumerate(policies):
            if stability_margin is not None and stability_margin.get(policy, 1.0) < 0:
                ax.axvspan(i - 0.5, i + 0.5, color=style.red, alpha=0.12, zorder=0,
                           label="unstable (ρ≥1)" if not unstable_added else "")
                unstable_added = True

        ax.set_xticks(x)
        labels = ax.set_xticklabels(policies, rotation=45, ha="right", fontsize=7)
        if stability_margin is not None:
            for label, policy in zip(labels, policies):
                if stability_margin.get(policy, 1.0) < 0:
                    label.set_color(style.red)

        ax.set_ylabel(r"$\log_{10}(\mathrm{cond})$")
        ax.set_title(model)
        set_square(ax)
        ax.grid(True, alpha=0.3)
        box(ax)
        ax.legend(frameon=False)
        fig.tight_layout()
        paths.append(save_fig(fig, out_dir, f"fig4_{letter}"))
    return paths


# ── fig5 — convergence + perturbation robustness ────────────────────────────

@publication_style
def plot_convergence_robustness(conv_robust: dict, out_dir: Path,
                                style: PlotStyle | None = None) -> list[Path]:
    """One independent figure per (model, type) pair.

    Convergence panels: ``fig5_a1`` (Var-GPR), ``fig5_a2`` (Var-FN), ``fig5_a3`` (HSGP).
    Perturbation panels: ``fig5_b1``, ``fig5_b2``, ``fig5_b3`` — same model order.

    Parameters
    ----------
    conv_robust : dict[model] -> dict[policy] -> {
        "convergence": 1-D array (|λ_max| vs step),
        "convergence_std": 1-D array,
        "perturbation_mean": 1-D array, "perturbation_std": 1-D array,
    }

    References
    ----------
    [VCH18]: transient amplification as a fragility indicator.
    """
    style = resolve_style(style)
    models = style.models(conv_robust)
    paths = []
    for num, model in enumerate(models, start=1):
        # ── fig5_aX — stability convergence ─────────────────────────────────
        fig_a, ax_l = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))
        all_conv = []
        for policy in style.policies(conv_robust[model]):
            d = conv_robust[model].get(policy)
            if d is None:
                continue
            color = style.color(policy)
            conv = np.asarray(d["convergence"])
            all_conv.extend(conv[np.isfinite(conv) & (conv > 0)].tolist())
            t = np.arange(len(conv))
            ax_l.plot(t, conv, color=color, marker=style.marker(policy), ms=3, label=policy)
            cstd = np.asarray(d.get("convergence_std", np.zeros_like(conv)))
            ax_l.fill_between(t, conv - cstd, conv + cstd, color=color, alpha=0.15, linewidth=0)
        ax_l.axhline(1.0, color="k", lw=0.8, ls="--")
        ax_l.set_xlabel("AL step")
        ax_l.set_ylabel("|λ_max(A_r(t))|")
        ax_l.set_title(f"{model} — convergence")
        set_square(ax_l)
        maybe_log(ax_l, all_conv, axis="y")
        ax_l.legend(frameon=False, fontsize=6, ncol=2)
        ax_l.grid(True, alpha=0.3)
        box(ax_l)
        fig_a.tight_layout()
        paths.append(save_fig(fig_a, out_dir, f"fig5_a{num}"))

        # ── fig5_bX — perturbation response ─────────────────────────────────
        fig_b, ax_r = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))
        all_pert = []
        for policy in style.policies(conv_robust[model]):
            d = conv_robust[model].get(policy)
            if d is None:
                continue
            color = style.color(policy)
            pm = np.asarray(d["perturbation_mean"])
            all_pert.extend(pm[np.isfinite(pm) & (pm > 0)].tolist())
            ps = np.asarray(d.get("perturbation_std", np.zeros_like(pm)))
            k = np.arange(len(pm))
            ax_r.plot(k, pm, color=color, lw=1.2, label=policy)
            ax_r.fill_between(k, pm - ps, pm + ps, color=color, alpha=0.15, linewidth=0)
        ax_r.set_xlabel("steps after perturbation")
        ax_r.set_ylabel("||A_r^k δz||")
        ax_r.set_title(f"{model} — perturbation")
        set_square(ax_r)
        maybe_log(ax_r, all_pert, axis="y")
        ax_r.legend(frameon=False, fontsize=6, ncol=2)
        ax_r.grid(True, alpha=0.3)
        box(ax_r)
        fig_b.tight_layout()
        paths.append(save_fig(fig_b, out_dir, f"fig5_b{num}"))

    return paths


# ── fig12 — plating/SEI decomposition vs stability convergence ─────────────

@publication_style
def plot_matrix_heatmaps(A_full: np.ndarray, B_full: np.ndarray, feature_names: list,
                          action_names: list, mode_results: list, out_dir: Path,
                          policy_label: str = "", cmap: str = "RdBu_r",
                          style: PlotStyle | None = None) -> list[Path]:
    """``fig7_a``: A_full heatmap. ``fig7_b``: B_full heatmap.
    ``fig7_c``: coupling network. ``fig7_d``: dominant modes.

    References
    ----------
    [DIE18]: A_full = U_r A_r U_r^T reconstructed coupling structure.
    """
    style = resolve_style(style)
    suffix = f"_{policy_label.lower().replace('-', '_')}" if policy_label else ""

    # SymLogNorm: takes log10(|value|) for the colour mapping, then puts the
    # original sign back on automatically (a linear region of half-width
    # linthresh around 0 keeps the sign transition continuous instead of
    # blowing up at exactly zero) — a signed-log scale instead of the
    # previous linear vmin=-vmax/vmax scale, which let large entries wash
    # out all the smaller (but still meaningful) coupling structure.
    fig_a, ax_a = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))
    vmax = np.abs(A_full).max() or 1.0
    nz = np.abs(A_full)[np.abs(A_full) > 0]
    linthresh_a = float(np.quantile(nz, 0.1)) if nz.size else vmax * 1e-3
    norm_a = SymLogNorm(linthresh=max(linthresh_a, vmax * 1e-6), vmin=-vmax, vmax=vmax, base=10)
    im = ax_a.imshow(A_full, cmap=cmap, norm=norm_a)
    ax_a.set_xticks(range(len(feature_names)))
    ax_a.set_yticks(range(len(feature_names)))
    ax_a.set_xticklabels(feature_names, rotation=90, fontsize=5)
    ax_a.set_yticklabels(feature_names, fontsize=5)
    ax_a.set_title(f"A_full {policy_label}")
    signed_log_colorbar(im, ax_a, vmax, norm_a.linthresh)
    box(ax_a)
    fig_a.tight_layout()

    fig_b, ax_b = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))
    vmax_b = np.abs(B_full).max() or 1.0
    nz_b = np.abs(B_full)[np.abs(B_full) > 0]
    linthresh_b = float(np.quantile(nz_b, 0.1)) if nz_b.size else vmax_b * 1e-3
    norm_b = SymLogNorm(linthresh=max(linthresh_b, vmax_b * 1e-6), vmin=-vmax_b, vmax=vmax_b, base=10)
    im2 = ax_b.imshow(B_full, cmap=cmap, norm=norm_b, aspect="auto")
    ax_b.set_yticks(range(len(feature_names)))
    ax_b.set_yticklabels(feature_names, fontsize=5)
    ax_b.set_xticks(range(len(action_names)))
    ax_b.set_xticklabels(action_names, rotation=90, fontsize=6)
    ax_b.set_title("B_full")
    signed_log_colorbar(im2, ax_b, vmax_b, norm_b.linthresh)
    box(ax_b)
    fig_b.tight_layout()

    fig_d, ax_modes = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))
    top = mode_results[:3]
    labels = [f"mode {i+1}" for i in range(len(top))]
    mags = [m["magnitude"] for m in top]
    colors = [style.green if m["magnitude"] < 1 else style.red for m in top]
    ax_modes.bar(labels, mags, color=colors)
    for i, m in enumerate(top):
        ax_modes.annotate(str(m["dominant_feature"]), (i, mags[i]),
                           ha="center", va="bottom", fontsize=6, rotation=30)
    ax_modes.axhline(1.0, color="k", lw=0.8, ls="--")
    ax_modes.set_ylabel("|λ|")
    ax_modes.set_title("Dominant modes")
    set_square(ax_modes)
    ax_modes.grid(True, alpha=0.3)
    box(ax_modes)
    fig_d.tight_layout()

    return [
        save_fig(fig_a, out_dir, f"fig7_a{suffix}"),
        save_fig(fig_b, out_dir, f"fig7_b{suffix}"),
        save_fig(fig_d, out_dir, f"fig7_d{suffix}"),
    ]


# ── fig8 — modal equations ──────────────────────────────────────────────────

@publication_style
def plot_coupling_network_combined(
    policy_a_fulls: dict,
    feature_names: list,
    out_dir: Path,
    style: PlotStyle | None = None,
) -> Path:
    """``fig7_c``: coupling network graph with all policies overlaid.

    Each policy's edges are drawn in its color from ``style``.
    Edge width scales with |A_full[i,j]| / policy_max. Only edges above
    50% of each policy's own maximum coupling strength are shown.
    Nodes are shared across all policies (same feature layout).

    Parameters
    ----------
    policy_a_fulls : dict[str, np.ndarray]
        Mapping of policy name → A_full (n_feat × n_feat) reconstruction
        in original feature space (U_r @ A_r @ U_r.T).
    feature_names : list[str]
        Feature labels for the n nodes.
    out_dir : Path
        Directory to write fig7_c.png.
    """
    style = resolve_style(style)
    fig_c, ax_net = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))
    n = len(feature_names)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs, ys = np.cos(angles), np.sin(angles)

    legend_handles = []
    policies_present = style.policies(policy_a_fulls)
    for policy in policies_present:
        A_full = policy_a_fulls[policy]
        color = style.color(policy, default=style.blue)
        vmax = np.abs(A_full).max() or 1.0
        threshold = 0.5 * vmax
        for i in range(n):
            for j in range(n):
                if i == j or abs(A_full[i, j]) < threshold:
                    continue
                lw = 1.2 * abs(A_full[i, j]) / vmax
                ax_net.plot([xs[i], xs[j]], [ys[i], ys[j]], color=color,
                            alpha=0.45, lw=lw, solid_capstyle="round")
        legend_handles.append(
            Patch(color=color, label=policy, alpha=0.8)
        )

    ax_net.scatter(xs, ys, s=18, color="k", zorder=5)
    pad = 0.18
    for i, name in enumerate(feature_names):
        ax_net.annotate(
            name,
            (xs[i] * (1 + pad), ys[i] * (1 + pad)),
            fontsize=4.5,
            ha="center",
            va="center",
        )

    ax_net.set_xlim(-1.6, 1.6)
    ax_net.set_ylim(-1.6, 1.6)
    ax_net.set_aspect("equal")
    ax_net.axis("off")

    if legend_handles:
        leg = ax_net.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(4, len(legend_handles)),
            frameon=False,
            fontsize=6,
        )
        fig_c.tight_layout()
        return save_fig(fig_c, out_dir, "fig7_c", extra_artists=[leg])

    fig_c.tight_layout()
    return save_fig(fig_c, out_dir, "fig7_c")


@publication_style
def plot_modal_equations(mode_results: dict, out_dir: Path,
                         style: PlotStyle | None = None) -> Path:
    """Top-2 dominant mode magnitudes per policy with feature loadings.

    Parameters
    ----------
    mode_results : dict[policy] -> list of mode dicts (from
        ``traits_audit.dmdc.modal_decomposition``), already sorted by |λ|.
    """
    style = resolve_style(style)
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))
    policies = style.policies(mode_results)
    x = np.arange(len(policies))
    width = 0.35
    for k in range(2):
        mags, colors, labels = [], [], []
        for p in policies:
            modes = mode_results[p]
            if k < len(modes):
                m = modes[k]
                mags.append(m["magnitude"])
                osc = abs(m["eigenvalue"].imag) > 1e-6
                colors.append(style.orange if osc else style.blue)
                labels.append(str(m["dominant_feature"]))
            else:
                mags.append(0.0)
                colors.append(style.neutral)
                labels.append("")
        offs = x + (k - 0.5) * width
        ax.bar(offs, mags, width=width, color=colors, alpha=0.8)
        for xi, lab in zip(offs, labels):
            if lab:
                ax.annotate(lab, (xi, 0.02), rotation=90, fontsize=5, va="bottom", ha="center")
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=45, ha="right")
    ax.set_ylabel("|λ_dominant|")
    handles = [plt.Rectangle((0, 0), 1, 1, color=style.blue, label="non-oscillatory"),
               plt.Rectangle((0, 0), 1, 1, color=style.orange, label="oscillatory (complex)")]
    ax.legend(handles=handles, frameon=False, fontsize=7)
    set_square(ax)
    ax.grid(True, alpha=0.3)
    box(ax)
    fig.tight_layout()
    return save_fig(fig, out_dir, "fig8")


# ── fig9 — H3 scatter ───────────────────────────────────────────────────────


@publication_style
def plot_rank_dataset(rank_sens: dict, default_r: int, cross_dataset: dict, out_dir: Path,
                      style: PlotStyle | None = None) -> list[Path]:
    """``fig10_a``: |λ_max| vs truncation rank.
    ``fig10_b``: stability margin per policy × dataset.

    Parameters
    ----------
    rank_sens : dict[policy] -> dict[r] -> |λ_max| (float)
    cross_dataset : dict[policy] -> dict[dataset] -> {"mean", "std"}

    References
    ----------
    [SUBSP21]: rank-robustness check for the low-rank subspace.
    """
    style = resolve_style(style)
    fig_a, ax_l = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))
    all_lam = []
    for policy, by_r in rank_sens.items():
        rs = sorted(by_r.keys())
        vals = [by_r[r] for r in rs]
        all_lam.extend([v for v in vals if v > 0 and np.isfinite(v)])
        ax_l.plot(rs, vals, color=style.color(policy, default=style.blue),
                  marker=style.marker(policy, default="o"), label=policy)
    ax_l.axvline(default_r, color="k", lw=0.8, ls="--")
    ax_l.set_xlabel("truncation rank r")
    ax_l.set_ylabel("|λ_max(A_r)|")
    set_square(ax_l)
    maybe_log(ax_l, all_lam, axis="y")
    ax_l.legend(frameon=False, fontsize=6, ncol=2)
    ax_l.grid(True, alpha=0.3)
    box(ax_l)
    fig_a.tight_layout()

    fig_b, ax_r = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))
    policies = list(cross_dataset.keys())
    datasets = sorted({ds for p in cross_dataset.values() for ds in p})
    x = np.arange(len(policies))
    width = 0.8 / max(len(datasets), 1)
    for i, ds in enumerate(datasets):
        means = [cross_dataset[p].get(ds, {}).get("mean", np.nan) for p in policies]
        stds = [cross_dataset[p].get(ds, {}).get("std", 0.0) for p in policies]
        ax_r.bar(x + (i - len(datasets) / 2) * width, means, width=width, yerr=stds,
                 label=ds, capsize=2)
    ax_r.set_xticks(x)
    ax_r.set_xticklabels(policies, rotation=45, ha="right")
    ax_r.set_ylabel("stability margin")
    set_square(ax_r)
    ax_r.legend(frameon=False, fontsize=7)
    ax_r.grid(True, alpha=0.3)
    box(ax_r)
    fig_b.tight_layout()

    return [save_fig(fig_a, out_dir, "fig10_a"), save_fig(fig_b, out_dir, "fig10_b")]


# ── fig11 — DMDc quality + pseudospectra ────────────────────────────────────



@publication_style
def plot_dmdc_quality(r2_dict: dict, pseudospectra: dict, out_dir: Path,
                       reference_model: str | None = None,
                       style: PlotStyle | None = None) -> list[Path]:
    """``fig11_a``: DMDc R² bar chart.
    ``fig11_b_<model>``: pseudospectra contours per policy (small-multiples
    grid), one figure per GPR model.
    ``fig11_c_<model>``: all policies overlaid on a single coordinate system,
    one figure per GPR model.

    Parameters
    ----------
    r2_dict : dict[model] -> dict[policy] -> float
    pseudospectra : dict[model] -> dict[policy] -> {"RE", "IM", "sigma_min",
        "eigenvalues"}
    reference_model : unused, kept for backward-compatible call sites.

    References
    ----------
    [CHA25]: pseudospectrum sensitivity of poles near the unit circle.
    """
    style = resolve_style(style)
    fragility_colors = [style.red, style.orange, style.green]   # small sigma_min (fragile) -> large (safe)
    fig_a, ax_r2 = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))
    models = style.models(r2_dict)
    policies = sorted({p for m in r2_dict.values() for p in m})
    x = np.arange(len(policies))
    width = 0.8 / max(len(models), 1)
    for i, model in enumerate(models):
        vals = [r2_dict[model].get(p, np.nan) for p in policies]
        ax_r2.bar(x + (i - len(models) / 2) * width, vals, width=width,
                  color=style.model_color(model, default=style.blue), edgecolor="k", linewidth=0.5, alpha=0.85)
    ax_r2.axhline(0.5, color="k", lw=0.8, ls="--")
    ax_r2.set_xticks(x)
    ax_r2.set_xticklabels(policies, rotation=45, ha="right")
    ax_r2.set_ylabel("DMDc R²")
    set_square(ax_r2)
    box(ax_r2)
    model_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=style.model_color(m, default=style.blue),
                                    edgecolor="k", linewidth=0.5, label=m)
                      for m in models]
    ax_r2.legend(handles=model_handles, frameon=False, fontsize=7)
    ax_r2.grid(True, alpha=0.3)
    fig_a.tight_layout()

    # ── fig11_b / fig11_c: pseudospectra, one figure-set per model ──────────
    # Levels are derived from each model's own sigma_min distribution rather
    # than fixed absolute thresholds: a highly non-normal / near-singular A_r
    # can have sigma_min below 0.1 (or even 0.5) everywhere on the grid, which
    # made the old fixed levels=[0, 0.1, 0.3, 0.5] fill the entire panel with
    # one colour and show no structure at all. Three discrete, semantically
    # fixed colours (red/orange/green = high/medium/low fragility) replace the
    # continuous "Slipstream_r" colormap sampling, which produced muddy
    # intermediate hues (salmon/teal/grey) that weren't meaningful as discrete
    # bands.
    models = style.models(pseudospectra)
    paths = [save_fig(fig_a, out_dir, "fig11_a")]
    legend_patches = [Patch(facecolor=c, edgecolor="none", alpha=0.55, label=lab)
                       for c, lab in zip(fragility_colors, _FRAGILITY_LABELS)]

    for model in models:
        ps_model = pseudospectra[model]
        if not ps_model:
            continue
        # Uniform axis limits across policies so every subplot has the same
        # extent and physical area (set_aspect("equal") then makes them identical).
        # The grid extent alone isn't enough: eigenvalues can sit outside the
        # contour grid (or right at its edge), which cropped poles/zeros in
        # the rendered panel — fold the eigenvalue extent into the limit too,
        # then add a margin so nothing sits flush against the axes edge.
        re_min = min(d["RE"].min() for d in ps_model.values())
        re_max = max(d["RE"].max() for d in ps_model.values())
        im_min = min(d["IM"].min() for d in ps_model.values())
        im_max = max(d["IM"].max() for d in ps_model.values())
        half = max(abs(re_min), abs(re_max), abs(im_min), abs(im_max), 1.2) * 1.05

        all_sigma = np.concatenate([np.asarray(d["sigma_min"]).ravel() for d in ps_model.values()])
        finite = all_sigma[np.isfinite(all_sigma) & (all_sigma >= 0)]
        if finite.size and finite.max() > finite.min():
            levels = np.unique(np.quantile(finite, [0.0, 0.1, 0.4, 1.0]))
        else:
            levels = None
        eps = float(np.quantile(finite, 0.25)) if finite.size else 0.1

        # ── fig11_b_<model>: small-multiples grid ───────────────────────────
        n_pol = len(ps_model)
        ncols = 3
        nrows = max(1, int(np.ceil(n_pol / ncols)))
        cell = DOUBLE_COL / ncols      # physical size per cell (inches)
        fig_b = plt.figure(figsize=(DOUBLE_COL, cell * nrows + 0.6))
        gs = fig_b.add_gridspec(nrows, ncols, hspace=0.5, wspace=0.3)
        for idx, (policy, d) in enumerate(ps_model.items()):
            ax = fig_b.add_subplot(gs[idx // ncols, idx % ncols])
            ax.patch.set_alpha(0.0)    # transparent panel background
            draw_unit_circle(ax)
            if levels is not None and len(levels) > 1:
                ncolors = len(levels) - 1
                ax.contourf(d["RE"], d["IM"], d["sigma_min"], levels=levels,
                            colors=fragility_colors[:ncolors], alpha=0.55)
            eigs = d.get("eigenvalues")
            if eigs is not None:
                ax.scatter(eigs.real, eigs.imag, s=8, color=style.color(policy, default=style.blue))
            ax.set_xlim(-half, half)
            ax.set_ylim(-half, half)
            ax.set_aspect("equal")
            ax.set_title(policy, fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
            box(ax)
        fig_b.patch.set_alpha(0.0)
        fig_b.suptitle(f"Pseudospectra ({model})", fontsize=9)
        # Legend goes below the grid (not above) so it never overlaps the
        # panels — bbox_inches="tight" + extra_artists expands the canvas
        # downward to fit it.
        leg_b = fig_b.legend(handles=legend_patches, loc="upper center",
                              bbox_to_anchor=(0.5, 0.0), ncol=1, frameon=False, fontsize=6.5)
        slug = model.lower().replace("-", "_")
        paths.append(save_fig(fig_b, out_dir, f"fig11_b_{slug}", transparent=True, extra_artists=[leg_b]))

        # ── fig11_c_<model>: all policies on a single coordinate system ─────
        # Fragility boundary (one contour line at the model's 25th-percentile
        # sigma_min) instead of filled regions, so overlapping policies stay legible.
        fig_c, ax_c = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))
        ax_c.patch.set_alpha(0.0)
        draw_unit_circle(ax_c)
        for policy, d in ps_model.items():
            color = style.color(policy, default=style.blue)
            ax_c.contour(d["RE"], d["IM"], d["sigma_min"], levels=[eps],
                         colors=[color], linewidths=1.2, alpha=0.85)
            eigs = d.get("eigenvalues")
            if eigs is not None:
                ax_c.scatter(eigs.real, eigs.imag, s=14, color=color,
                             marker=style.marker(policy, default="o"), alpha=0.8)
        ax_c.set_xlabel("Re(λ)")
        ax_c.set_ylabel("Im(λ)")
        ax_c.set_title(model)
        set_square(ax_c)
        box(ax_c)
        policy_handles = [plt.Line2D([0], [0], marker=style.marker(p, default="o"), color="w",
                                      markerfacecolor=style.color(p, default=style.blue), markersize=6, label=p)
                           for p in style.policies(ps_model)]
        leg_c = ax_c.legend(handles=policy_handles, frameon=False, fontsize=6, ncol=2,
                             loc="upper left", bbox_to_anchor=(1.02, 1.0), title="Policy")
        ax_c.grid(True, alpha=0.3)
        fig_c.patch.set_alpha(0.0)
        paths.append(save_fig(fig_c, out_dir, f"fig11_c_{slug}", transparent=True, extra_artists=[leg_c]))

    return paths
