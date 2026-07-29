"""Visualisation utilities for the traits-audit uncertainty audit.

All public ``plot_*`` functions write publication-ready PNG figures to disk
at 300 dpi.  Private helpers (prefixed ``_``) produce Plotly
or matplotlib objects returned to callers (MLflow, demo scripts).

A single ``_RCPARAMS`` block is applied at import time so every figure
produced by this module inherits the same typography and line weights.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

import matplotlib.pyplot as plt

from .checks.lyapunov import (
    numerical_jacobian,
    eigenvalues_and_stability,
    compute_lyapunov,
    make_gd_predictor,
)

# ── Publication rcParams (applied once at import) ───────────────────────────

_RCPARAMS: Dict[str, Any] = {
    "font.family":     "serif",
    "font.size":       10,
    "axes.titlesize":  11,
    "axes.labelsize":  10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.5,
    "axes.linewidth":  0.8,
    "figure.dpi":      300,
}
plt.rcParams.update(_RCPARAMS)


# ── Plotly figure constants ─────────────────────────────────────────────────

#: Short labels for check names used on the x-axis of the check-grid heatmap
#: and the metric-correlation matrix. Every ``AuditCheck.name`` in the
#: package should have an entry here — a missing one falls back to the full
#: name (check grid) or a hard 12-char truncation (correlation matrix), both
#: of which get visually cramped or ambiguous fast.
_CHECK_ABBREV: Dict[str, str] = {
    # Total-predictive-distribution checks (pre-existing).
    "CalibrationError":            "CE",
    "KuleshovCalibrationError":    "KCE",
    "ENCE":                        "ENCE",
    "CalibrationError1Std":        "CE1Sig",
    "ConformalCoverage":           "CC",
    "CRPS":                        "CRPS",
    "IntervalCoverage":            "IC",
    "IntervalScore":               "IS",
    "NegativeLogLikelihood":       "NLL",
    "PITUniformity":               "PITU",
    "UncertaintyEvolution":        "UE",
    "UncertaintyAnomalies":        "UA",
    "VarianceAlignment":           "VA",
    "VarianceErrorCorrelation":    "VEC",
    "LyapunovStability":           "LYS",
    "MahalanobisOOD":              "MOOD",
    # Taxonomy-audit additions (METRIC_TAXONOMY_AUDIT.md §4).
    "SignedBias":                  "SB",
    "ReplicationShrinkageExponent": "RSE",
    "DarkUncertaintyGap":          "DUG",
    "TypeBMassFraction":           "TBMF",
    "ReducibilityRealisationRatio": "RRR",
    "AleatoricFloorConsistency":   "AFC",
    "EnsembleIndependenceDeficit": "EID",
    "DMDcSpectralRadius":          "DMDcRho",
    "ResidualPersistenceHalfLife": "RHL",
    "ImprecisionWidthFraction":    "IWF",
    "EnvelopeViolationRate":       "EV",
    "ProceduralVarianceShare":     "PVS",
    "DataVarianceShare":           "DVS",
    "MisspecificationResidualFloor": "MRF",
    "StageVarianceAttribution":    "SVA",
    "DecisionFlipRate":            "DFR",
    "TailIndex":                   "TI",
    "ScoreDecomposition":          "SD",
}

#: Per-step scalar keys recorded by the audit hook and shown in the state heatmap.
_STATE_KEYS = ["uncertainty", "pool_sigma_mean", "pool_sigma_max", "abs_error"]


# ── Private save helper ─────────────────────────────────────────────────────

def _save(fig, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {stem}.png")


# ── Matplotlib plot functions ───────────────────────────────────────────────

def plot_poles(
    all_eigenvalues: np.ndarray,
    model_label: str,
    out_dir: Path,
) -> None:
    """Eigenvalue diagram (fig1_poles).

    When all eigenvalues are purely real (max|Im(λ)| < 1e-8 · max|λ|) renders
    a 1-D strip plot along the real axis — GD-predictor Jacobians are always
    real-symmetric so this case is common.  Otherwise falls back to the
    standard complex unit-circle diagram.
    """
    eigs  = np.asarray(all_eigenvalues)
    mags  = np.abs(eigs)
    max_m = float(mags.max()) if len(mags) else 1.0
    purely_real = (
        max_m == 0.0
        or float(np.abs(eigs.imag).max()) < 1e-8 * max_m
    )

    if purely_real:
        re = eigs.real
        stable   = np.abs(re) < 1.0
        unstable = ~stable

        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        rng_spread = float(np.abs(re).max()) * 1.15
        lim = max(1.3, rng_spread)
        jitter = np.random.default_rng(0).uniform(-0.08, 0.08, size=len(re))

        if stable.any():
            ax.scatter(re[stable], jitter[stable],
                       c="C0", s=22, alpha=0.7, linewidths=0,
                       label=f"Stable |λ|<1  ({stable.sum()})")
        if unstable.any():
            ax.scatter(re[unstable], jitter[unstable],
                       c="C3", s=22, alpha=0.8, linewidths=0,
                       label=f"Unstable |λ|≥1  ({unstable.sum()})")

        ax.axvline(-1.0, color="k", lw=0.8, ls="--", alpha=0.55)
        ax.axvline(+1.0, color="k", lw=0.8, ls="--", alpha=0.55,
                   label="Stability boundary (±1)")
        ax.axhline(0.0,  color="k", lw=0.4, alpha=0.25)

        out_view = int((np.abs(re) > lim).sum())
        if out_view:
            ax.text(0.97, 0.05,
                    f"{out_view} pole(s) outside view  "
                    f"[{re.min():.2f}, {re.max():.2f}]",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7, color="C3")

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-0.35, 0.35)
        ax.set_xlabel("Re(λ)  [Im(λ) ≡ 0 — GD on real scalar f]")
        ax.set_yticks([])
        ax.set_title(model_label, fontsize=9)
        ax.grid(True, alpha=0.3, axis="x")
        fig.tight_layout()
        ax.legend(frameon=False, fontsize=7, ncol=1,
                   bbox_to_anchor=(1.02, 0.5), loc="center left")

    else:
        fig, ax = plt.subplots(figsize=(3.5, 3.5))

        lim = max(1.5, min(max_m * 1.15, 5.0))

        theta = np.linspace(0, 2 * np.pi, 300)
        ax.plot(np.cos(theta), np.sin(theta), "k--", lw=0.8, alpha=0.5,
                label="Unit circle")

        in_view = (np.abs(eigs.real) <= lim) & (np.abs(eigs.imag) <= lim)
        n_out   = int((~in_view).sum())

        if in_view.any():
            ax.scatter(eigs[in_view].real, eigs[in_view].imag,
                       c="C0", s=18, alpha=0.7, linewidths=0, label=model_label)
        else:
            ax.scatter([], [], c="C0", s=18, linewidths=0, label=model_label)

        if n_out:
            msg = (f"{n_out} pole(s) outside view\n"
                   f"|λ| ∈ [{mags.min():.2e}, {mags.max():.2e}]")
            ax.text(0.03, 0.03, msg,
                    transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=7, color="C0", alpha=0.85)

        ax.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax.axvline(0, color="k", lw=0.5, alpha=0.3)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("Re(λ)")
        ax.set_ylabel("Im(λ)")
        ax.set_aspect("equal")
        ax.grid(False)
        fig.tight_layout()
        ax.legend(frameon=False, fontsize=7,
                   bbox_to_anchor=(1.02, 0.5), loc="center left")

    _save(fig, out_dir, "fig1_poles")


def plot_stability_contours(
    P: np.ndarray | None,
    states: np.ndarray,
    lambda_max: np.ndarray,
    model_label: str,
    out_dir: Path,
) -> None:
    """2-D PCA projection of the Lyapunov stability landscape (fig2)."""
    from sklearn.decomposition import PCA

    n_comp = min(2, states.shape[1], len(states) - 1)
    pca = PCA(n_components=n_comp)
    z = pca.fit_transform(states - states.mean(axis=0))
    z2 = z[:, 1] if z.shape[1] > 1 else np.zeros(len(z))

    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    if P is not None and n_comp == 2:
        V = pca.components_.T[:, :2]
        P2 = V.T @ P @ V
        lim = np.abs(z).max() * 1.2
        gs = np.linspace(-lim, lim, 120)
        gx, gy = np.meshgrid(gs, gs)
        gpts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        lyap = np.einsum("ni,ij,nj->n", gpts, P2, gpts).reshape(gx.shape)
        cf = ax.contourf(gx, gy, lyap, levels=15, cmap="YlOrRd", alpha=0.65)
        ax.contour(gx,  gy, lyap, levels=15, colors="k", linewidths=0.4, alpha=0.4)
        plt.colorbar(cf, ax=ax, label="V(x) = xᵀPx")

    # Use TwoSlopeNorm centred at the stability boundary (|λ|=1) so that
    # blue=stable, white=boundary, red=unstable regardless of the actual spread.
    from matplotlib.colors import TwoSlopeNorm
    lm_vals = np.asarray(lambda_max, dtype=float)
    lm_lo = float(np.percentile(lm_vals, 2))
    lm_hi = float(np.percentile(lm_vals, 98))
    # Guarantee that the centre (1.0) lies strictly inside [vmin, vmax].
    lm_lo = min(lm_lo, 0.98)
    lm_hi = max(lm_hi, 1.02)
    norm = TwoSlopeNorm(vcenter=1.0, vmin=lm_lo, vmax=lm_hi)
    sc = ax.scatter(z[:, 0], z2, c=lambda_max, cmap="coolwarm",
                    s=18, edgecolors="k", linewidths=0.3, norm=norm,
                    label="|λ_max|")
    plt.colorbar(sc, ax=ax, label="|λ_max|")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(model_label)
    ax.legend(frameon=False)
    ax.grid(False)
    fig.tight_layout()
    _save(fig, out_dir, "fig2_stability_contours")


def plot_stability_vs_uncertainty(
    lambda_max: np.ndarray,
    surrogate_std: np.ndarray,
    model_label: str,
    out_dir: Path,
) -> None:
    """Scatter: |λ_max| vs surrogate posterior std (fig3)."""
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.scatter(surrogate_std, lambda_max,
               c="C0", s=20, alpha=0.6, linewidths=0, label=model_label)
    ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.6, label="Stability boundary")
    ax.set_xlabel("Surrogate posterior std")
    ax.set_ylabel("|λ_max|")
    ax.grid(False)
    fig.tight_layout()
    ax.legend(frameon=False, fontsize=7,
              bbox_to_anchor=(0.5, 1.02), loc="lower center", ncol=1)
    _save(fig, out_dir, "fig3_stability_vs_unc")


def plot_pareto_frontier(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    x_label: str,
    y_label: str,
    model_label: str,
    out_dir: Path,
    minimize_x: bool = True,
    minimize_y: bool = True,
    color_vals: np.ndarray | None = None,
    color_label: str = "AL step",
    true_front_x: np.ndarray | None = None,
    true_front_y: np.ndarray | None = None,
) -> None:
    """Pareto frontier in 2-D objective space (fig7).

    Highlights the non-dominated set of queried points.  Optionally overlays
    the known true Pareto front as a dashed reference line via ``true_front_x``
    and ``true_front_y``.
    """
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) == 0:
        return
    cv = np.asarray(color_vals, dtype=float)[valid] if color_vals is not None else None

    xp = x if minimize_x else -x
    yp = y if minimize_y else -y
    n = len(x)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if (
                i != j
                and xp[j] <= xp[i]
                and yp[j] <= yp[i]
                and (xp[j] < xp[i] or yp[j] < yp[i])
            ):
                dominated[i] = True
                break
    pareto_idx = np.where(~dominated)[0]
    order = np.argsort(x[pareto_idx])
    px, py = x[pareto_idx][order], y[pareto_idx][order]

    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    if cv is not None:
        sc = ax.scatter(
            x, y, c=cv, cmap="viridis_r", s=22, alpha=0.75,
            linewidths=0.3, edgecolors="none", zorder=2,
        )
        plt.colorbar(sc, ax=ax, label=color_label, fraction=0.046, pad=0.04)
        ax.scatter(
            px, py, s=50, facecolors="none", edgecolors="C1",
            linewidths=1.2, zorder=4, label="Pareto-optimal",
        )
    else:
        ax.scatter(
            x[dominated], y[dominated], c="C0", s=16, alpha=0.4,
            linewidths=0, label="Dominated",
        )
        ax.scatter(
            px, py, c="C1", s=30, zorder=3, linewidths=0.5,
            edgecolors="k", label="Pareto-optimal",
        )

    for k in range(len(px) - 1):
        ax.plot([px[k], px[k + 1]], [py[k], py[k]],
                color="C1", lw=1.0, ls="--", alpha=0.8)
        ax.plot([px[k + 1], px[k + 1]], [py[k], py[k + 1]],
                color="C1", lw=1.0, ls="--", alpha=0.8)

    if true_front_x is not None and true_front_y is not None:
        tfx = np.asarray(true_front_x, dtype=float)
        tfy = np.asarray(true_front_y, dtype=float)
        order = np.argsort(tfx)
        ax.plot(tfx[order], tfy[order], color="k", lw=1.0, ls=":", alpha=0.6, label="True front")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(frameon=False)
    ax.grid(False)
    ax.set_title(model_label)
    fig.tight_layout()
    _save(fig, out_dir, "fig7_pareto_frontier")


def plot_cie_trajectory(
    lhs_points: np.ndarray,
    al_points: np.ndarray,
    y_lhs: np.ndarray,
    y_al: np.ndarray,
    out_dir: Path,
    model_label: str = "",
) -> None:
    """CIE 1931 xy chromaticity diagram showing the acquisition trajectory (fig7).

    LED (R, G, B) intensities (normalised to [0, 1]) are converted to CIE XYZ
    via the sRGB/D65 matrix and projected to (x, y) chromaticity.  LHS initial
    samples and the active-learning trajectory are overlaid on the standard
    horse-shoe locus with the sRGB gamut triangle.

    Parameters
    ----------
    lhs_points : array (n_init, 3)
        Normalised (r, g, b) ∈ [0, 1]³ for the initial LHS samples.
    al_points : array (n_iter, 3)
        Normalised (r, g, b) for each AL-queried point, in acquisition order.
    y_lhs : array (n_init,)
        Fréchet distance at each LHS point (used for colour scale).
    y_al : array (n_iter,)
        Fréchet distance at each AL point.
    out_dir : Path
        Output directory.
    model_label : str
        Figure title.
    """
    # ── CIE 1931 spectral locus xy (380–780 nm, 10 nm steps) ─────────────────
    _locus_x = np.array([
        0.17411, 0.17396, 0.17383, 0.17367, 0.17343,
        0.16892, 0.16437, 0.15659, 0.14399, 0.12413,
        0.09136, 0.04539, 0.00823, 0.01385, 0.07420,
        0.15464, 0.22952, 0.30162, 0.37291, 0.44420,
        0.51259, 0.57536, 0.62704, 0.66575, 0.69149,
        0.70888, 0.72367, 0.73480, 0.74302, 0.74862,
        0.75138, 0.75368, 0.75518, 0.75636, 0.75718,
        0.75775, 0.75814, 0.75841, 0.75860, 0.75874,
        0.75883,
    ])
    _locus_y = np.array([
        0.00496, 0.00494, 0.00481, 0.00476, 0.00482,
        0.00810, 0.01086, 0.01765, 0.02975, 0.05782,
        0.13279, 0.29505, 0.53837, 0.75016, 0.83380,
        0.81604, 0.75430, 0.69232, 0.62488, 0.55093,
        0.48633, 0.42384, 0.37283, 0.33370, 0.30807,
        0.29083, 0.27597, 0.26516, 0.25704, 0.25161,
        0.24899, 0.24682, 0.24531, 0.24413, 0.24327,
        0.24279, 0.24234, 0.24216, 0.24197, 0.24186,
        0.24176,
    ])

    # sRGB gamut primaries + white point (D65)
    _srgb_r = (0.6400, 0.3300)
    _srgb_g = (0.3000, 0.6000)
    _srgb_b = (0.1500, 0.0600)
    _d65    = (0.3127, 0.3290)

    # sRGB → XYZ (D65) matrix
    _M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])

    def _to_xy(rgb_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convert (N, 3) normalised RGB to CIE xy chromaticity."""
        rgb = np.clip(rgb_norm, 0, 1)
        XYZ = rgb @ _M.T                   # (N, 3)
        s = XYZ.sum(axis=1, keepdims=True)
        s = np.where(s < 1e-9, 1.0, s)
        xy = XYZ[:, :2] / s
        return xy[:, 0], xy[:, 1]

    lhs = np.asarray(lhs_points, dtype=float)
    al  = np.asarray(al_points,  dtype=float)
    xl, yl = _to_xy(lhs)
    xa, ya = _to_xy(al)

    all_y = np.concatenate([np.asarray(y_lhs), np.asarray(y_al)]).astype(float)
    vmin, vmax = float(np.nanmin(all_y)), float(np.nanmax(all_y))

    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    # Spectral locus + purple line
    lx = np.append(_locus_x, _locus_x[0])
    ly = np.append(_locus_y, _locus_y[0])
    ax.plot(lx, ly, color="k", lw=0.8, zorder=1)
    ax.plot([_locus_x[-1], _locus_x[0]], [_locus_y[-1], _locus_y[0]],
            color="k", lw=0.8, ls="--", zorder=1)

    # sRGB gamut triangle
    gx = [_srgb_r[0], _srgb_g[0], _srgb_b[0], _srgb_r[0]]
    gy = [_srgb_r[1], _srgb_g[1], _srgb_b[1], _srgb_r[1]]
    ax.plot(gx, gy, color="gray", lw=0.8, ls=":", zorder=2, label="sRGB gamut")
    ax.scatter(*_d65, marker="+", s=60, color="gray", zorder=3)

    cmap = plt.cm.viridis_r

    # Acquisition trajectory (line connecting AL points in order)
    if len(xa) > 1:
        ax.plot(xa, ya, color="0.6", lw=0.7, zorder=3)

    # LHS initial samples
    sc_lhs = ax.scatter(
        xl, yl, c=np.asarray(y_lhs, dtype=float),
        cmap=cmap, vmin=vmin, vmax=vmax,
        marker="s", s=40, linewidths=0.5, edgecolors="k",
        zorder=4, label="LHS init",
    )

    # AL-queried points
    sc_al = ax.scatter(
        xa, ya, c=np.asarray(y_al, dtype=float),
        cmap=cmap, vmin=vmin, vmax=vmax,
        marker="o", s=28, linewidths=0.3, edgecolors="k",
        zorder=5, label="AL query",
    )

    # Best found (lowest Fréchet)
    best_idx = int(np.argmin(y_al))
    ax.scatter(xa[best_idx], ya[best_idx],
               marker="*", s=200, color="gold", edgecolors="k",
               linewidths=0.8, zorder=6, label="Best")

    plt.colorbar(sc_al, ax=ax, label="Fréchet distance", fraction=0.04, pad=0.04)

    ax.set_xlim(0.0, 0.80)
    ax.set_ylim(0.0, 0.90)
    ax.set_xlabel("CIE x")
    ax.set_ylabel("CIE y")
    ax.set_title(model_label or "CIE 1931 acquisition trajectory")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(False)
    fig.tight_layout()
    _save(fig, out_dir, "fig7_cie_trajectory")


def plot_uncertainty_evolution(
    uncertainties: np.ndarray,
    model_label: str,
    out_dir: Path,
) -> None:
    """Per-step surrogate uncertainty over the AL loop (fig4)."""
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.plot(np.arange(len(uncertainties)), uncertainties, color="C0")
    ax.set_xlabel("Step")
    ax.set_ylabel("Surrogate std")
    ax.set_title(model_label, fontsize=9)
    ax.grid(False)
    fig.tight_layout()
    _save(fig, out_dir, "fig4_uncertainty_evolution")


def plot_lyapunov_evolution(
    lambda_max_seq: np.ndarray,
    uncertainties: np.ndarray,
    model_label: str,
    out_dir: Path,
) -> None:
    """Dual-axis: |λ_max| and surrogate std over AL steps (fig5)."""
    n = len(lambda_max_seq)
    steps = np.arange(n)

    fig, ax1 = plt.subplots(figsize=(3.5, 3.5))
    ax2 = ax1.twinx()

    lm = np.asarray(lambda_max_seq, dtype=float)
    uc = np.asarray(uncertainties, dtype=float)

    l1, = ax1.plot(steps, lm, color="C1", label="|λ_max|")
    ax1.axhline(1.0, color="C1", lw=0.6, ls="--", alpha=0.5)
    ax1.set_xlabel("AL step")
    ax1.set_ylabel("|λ_max| (stability)", color="C1")
    ax1.tick_params(axis="y", labelcolor="C1")

    l2, = ax2.plot(steps, uc, color="C0", label="Surrogate std")
    ax2.set_ylabel("Surrogate std (uncertainty)", color="C0")
    ax2.tick_params(axis="y", labelcolor="C0")

    ax1.grid(False)
    ax1.set_title(model_label)
    fig.tight_layout()
    fig.legend(handles=[l1, l2], loc="center left",
               bbox_to_anchor=(1.12, 0.5), bbox_transform=fig.transFigure,
               frameon=False, fontsize=_RCPARAMS["legend.fontsize"])
    _save(fig, out_dir, "fig5_lyapunov_evolution")


def plot_audit_evolution(
    pipeline,
    history: list[dict],
    model_label: str,
    out_dir: Path,
    snapshot_every: int = 5,
) -> None:
    """Per-check metric vs AL step, one subplot per check (fig6).

    Each subplot shows the metric value at every snapshot step.  Dots are
    coloured green (pass), red (fail), or grey (report-only — the check
    reported a value but has no configured threshold, e.g. ``CRPSCheck()``
    with the default ``threshold=None``; a bare "always green" dot there
    would misrepresent an unevaluated score as a healthy one).  A dashed
    black horizontal line marks the pass/fail threshold where one is
    defined; for ``IntervalCoverage`` the acceptable band bounds are drawn
    as two lines.  Report-only checks show no threshold line, since they
    have none.

    ``pipeline`` is re-run from scratch at every snapshot step, so pass a
    lightweight subset (fast checks only) rather than the demo's full
    pipeline once it includes refit sweeps or bootstrap CIs — those are
    affordable once per run, not once per snapshot times every AL step.
    """
    n_steps = len(history)
    if n_steps < snapshot_every:
        return

    snap_steps = list(range(snapshot_every, n_steps + 1, snapshot_every))
    if snap_steps[-1] != n_steps:
        snap_steps.append(n_steps)

    records: dict[str, tuple[list, list]] = {}
    status_at: dict[str, list[str]] = {}
    thresholds: dict[str, Any] = {}
    tolerances: dict[str, float] = {}

    for k in snap_steps:
        sub = history[:k]
        try:
            report = pipeline.run(sub)
        except Exception:
            continue
        for r in report.results:
            if r.value is None:
                continue
            _, status = _result_status(r)
            records.setdefault(r.name, ([], []))[0].append(k)
            records[r.name][1].append(r.value)
            status_at.setdefault(r.name, []).append(status)
            if r.name not in thresholds and r.threshold is not None:
                thresholds[r.name] = r.threshold
            if r.name not in tolerances and r.details:
                tol = r.details.get("tolerance")
                if tol is not None:
                    tolerances[r.name] = tol

    if not records:
        return

    check_names = list(records.keys())
    n_checks = len(check_names)
    ncols = 3
    nrows = (n_checks + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0, nrows * 1.9))
    axes_flat = np.array(axes).flatten()

    _STATUS_COLOR = {"pass": "#27ae60", "fail": "#c0392b", "report_only": "#9a9a9a", "skipped": "#9a9a9a"}

    for i, name in enumerate(check_names):
        ax = axes_flat[i]
        xs, ys = records[name]
        statuses = status_at.get(name, ["pass"] * len(xs))
        colors = [_STATUS_COLOR[s] for s in statuses]
        ax.plot(xs, ys, color="C0", lw=1.2)
        ax.scatter(xs, ys, c=colors, s=18, zorder=3)

        t = thresholds.get(name)
        if t is not None:
            if isinstance(t, tuple):
                ax.axhline(t[0], color="k", lw=0.8, ls="--", alpha=0.5)
                ax.axhline(t[1], color="k", lw=0.8, ls="--", alpha=0.5)
            elif name in tolerances:
                tol = tolerances[name]
                ax.axhline(float(t) - tol, color="k", lw=0.8, ls="--", alpha=0.5)
                ax.axhline(float(t) + tol, color="k", lw=0.8, ls="--", alpha=0.5)
            else:
                ax.axhline(float(t), color="k", lw=0.8, ls="--", alpha=0.5)

        title = name.replace("Check", "")
        if all(s in ("report_only", "skipped") for s in statuses):
            title += " (report-only)"
        ax.set_title(title, fontsize=_RCPARAMS["legend.fontsize"])
        ax.set_xlabel("Step")
        ax.tick_params(labelsize=_RCPARAMS["xtick.labelsize"])
        ax.grid(False)

    for j in range(n_checks, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"{model_label} — Audit checks over AL steps",
                 fontsize=_RCPARAMS["font.size"])
    fig.tight_layout()
    _save(fig, out_dir, "fig6_audit_evolution")


def plot_convergence(
    best_vals: np.ndarray,
    query_counts: np.ndarray,
    y_label: str,
    model_label: str,
    out_dir: Path,
    maximise: bool = False,
    fig_title="fig8_convergence",
) -> None:
    """Running best task objective vs cumulative AL queries (fig8).

    Parameters
    ----------
    best_vals :
        Running best objective value at each query (already accumulated;
        pass ``np.maximum.accumulate`` or ``np.minimum.accumulate`` of the
        raw observations before calling).
    query_counts :
        Cumulative query index, same length as ``best_vals``.
    y_label :
        Axis label for the objective, including units.
    model_label :
        Surrogate label shown in the figure title.
    maximise :
        If ``True``, the objective is being maximised (e.g. capacity);
        if ``False``, minimised (e.g. error, Fréchet distance). Used only to
        validate that ``best_vals`` was actually pre-accumulated in the
        claimed direction — a caller that passes raw per-step values instead
        of a running best (the ``np.maximum``/``np.minimum.accumulate`` this
        docstring asks for) gets a warning naming the fix, rather than a
        silently spiky "convergence" plot.
    """
    best_vals = np.asarray(best_vals, dtype=float)
    query_counts = np.asarray(query_counts, dtype=float)
    valid = np.isfinite(best_vals)
    if not valid.any():
        return

    diffs = np.diff(best_vals[valid])
    wrong_direction = (diffs < -1e-9) if maximise else (diffs > 1e-9)
    if wrong_direction.any():
        import warnings
        accumulator = "np.maximum.accumulate" if maximise else "np.minimum.accumulate"
        warnings.warn(
            f"plot_convergence(maximise={maximise}) for '{model_label}' received "
            f"best_vals that are not monotonic in the claimed direction "
            f"({int(wrong_direction.sum())}/{len(diffs)} steps go the wrong way) — "
            f"this should be a running best; pre-accumulate with {accumulator}() "
            "before calling.",
            stacklevel=2,
        )

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.plot(query_counts[valid], best_vals[valid], color="C0", label=model_label)
    # Seed baseline: dashed horizontal at the initial best value
    baseline = best_vals[valid][0]
    ax.axhline(baseline, color="k", lw=0.8, ls="--", alpha=0.5, label="Seed baseline")
    ax.set_xlabel("Cumulative AL queries")
    ax.set_ylabel(y_label)
    ax.grid(False)
    fig.tight_layout()
    ax.legend(frameon=False, fontsize=7,
              bbox_to_anchor=(0.5, 1.02), loc="lower center", ncol=1)
    _save(fig, out_dir, fig_title)


# ── Heatmap intensity helper ────────────────────────────────────────────────

def _result_status(result: Any) -> "tuple[float, str]":
    """Map an AuditResult to a (intensity, status) pair.

    ``status`` is one of four values, not two — this is the load-bearing
    fix for the "report-only checks render as deeply passing" problem:
    a check with ``threshold=None`` (``CRPSCheck``, ``TailIndexCheck``, …)
    always sets ``passed=True`` by convention, but that is not the same
    claim as "deeply passing" — it means the check was never asked to judge
    anything. Likewise a ``Skipped — …`` message is a pass in
    ``AuditResult.passed`` but means "no data", not "healthy". Collapsing
    either into the same green as a genuinely evaluated, healthy result is
    what produced an overconfident model's NLL (13.47, badly failing by any
    reasonable bar) painting the identical colour as a calibrated model's
    NLL (0.26).

    Returns
    -------
    (intensity, status)
        ``intensity`` is only meaningful when ``status in ("pass", "fail")``:
        1.0 = deeply passing, 0.5 = at the threshold boundary, 0.0 = deeply
        failing. ``status`` is one of ``"pass"``, ``"fail"``,
        ``"report_only"`` (threshold is None — never evaluated against a
        criterion), or ``"skipped"`` (required data was unavailable).
    """
    if isinstance(result.message, str) and result.message.startswith("Skipped"):
        return 0.5, "skipped"

    v = result.value
    t = result.threshold
    if t is None:
        return 0.5, "report_only"
    if v is None:
        return (1.0 if result.passed else 0.0), ("pass" if result.passed else "fail")

    name = result.name
    details = result.details or {}
    if name == "CalibrationError":
        # Lower is better; PASS if v ≤ t
        signed = (t - v) / max(abs(t), 1e-6)
    elif name == "UncertaintyAnomalies":
        # Lower is better; PASS if v ≤ t
        signed = (t - v) / max(abs(t), 1e-6)
    elif name == "UncertaintyEvolution":
        # Higher (less negative) is better; PASS if v ≥ t. Display-span
        # heuristic, not a duplicated check parameter (UncertaintyEvolution
        # always reports threshold=0.0, which carries no natural scale).
        span = max(abs(t) * 3, 0.10)
        signed = (v - t) / span
    elif name == "VarianceErrorCorrelation":
        # Higher is better; PASS if v ≥ t. Same status as above — min_correlation
        # sets the pass boundary but not a natural display scale.
        span = max(1.0 - t, 0.30)
        signed = (v - t) / span
    elif name == "IntervalCoverage":
        # t is (lo, hi) band; derive target and tolerance from it. The
        # check always emits a tuple threshold, so the scalar branch below
        # is a defensive fallback only, using the check's own default
        # rather than a hardcoded value that could disagree with it.
        if isinstance(t, tuple):
            target = (t[0] + t[1]) / 2
            tol = max((t[1] - t[0]) / 2, 1e-6)
        else:
            tol = details.get("tolerance", 0.1)
            target = t
        signed = (tol - abs(v - target)) / tol
    elif name == "VarianceAlignment":
        # Toward-target (ideal ratio = 1.0 = t); tolerance read from the
        # check's own details rather than hardcoded, so this can never
        # silently disagree with VarianceAlignmentCheck(tolerance=...).
        tol = details.get("tolerance", 0.5)
        signed = (tol - abs(v - t)) / tol
    else:
        return (1.0 if result.passed else 0.0), ("pass" if result.passed else "fail")

    intensity = float(np.clip(0.5 + 0.5 * np.clip(signed, -1.0, 1.0), 0.0, 1.0))
    return intensity, ("pass" if result.passed else "fail")


# ── Plotly interactive figures ──────────────────────────────────────────────

def _fig_check_grid(
    stage_reports: "list[tuple[str, Any]]",
    run_name: str,
) -> Any:
    """Plotly heatmap: rows = audit checks, cols = pipeline stages.

    Cell intensity encodes how far the metric sits from the pass/fail
    threshold: dark green = deeply passing, white = at threshold, dark red =
    deeply failing. Two further states are distinct from both: a check with
    no configured threshold ("report-only", e.g. ``CRPSCheck()`` by default)
    and a check that was skipped for lack of data. Neither is a verdict, so
    neither gets a colorscale colour — both render as an empty (NaN) cell
    over the grey background, ringed by a hollow marker from the overlay
    traces below, so they cannot be mistaken for "deeply passing" green.
    """
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError:
        return None

    # Union of check names across every stage report, ordered by first
    # appearance. Using stage 0's names alone (the previous approach) silently
    # dropped any check that is absent from the first stage report entirely
    # -- e.g. a demo that merges a second, ensemble-only AuditPipeline's
    # results into the *final* report only (CAMD's EnsembleIndependenceDeficit
    # etc.): those checks never appear in hook.intermediate_reports at all,
    # so stage 0 doesn't know about them, and they vanished from the grid
    # without so much as a "skipped" marker.
    check_names: list = []
    for _, rep in stage_reports:
        for r in rep.results:
            if r.name not in check_names:
                check_names.append(r.name)
    abbrevs = [_CHECK_ABBREV.get(n, n) for n in check_names]
    stage_labels = [label for label, _ in stage_reports]

    # Build [stage][check] intermediate arrays then transpose to [check][stage].
    #
    # report-only/skipped cells get NO on-heatmap text (Heatmap.textfont.color
    # is a single scalar in this plotly version, not a per-cell array, so
    # there is no way to grey just those cells' numbers in the heatmap trace
    # itself). Their value is instead drawn by the overlay scatter traces
    # below, which do support their own per-trace text colour.
    z_by_stage, text_by_stage, hover_by_stage, status_by_stage, celltext_by_stage = [], [], [], [], []
    for label, rep in stage_reports:
        results_by_name = {r.name: r for r in rep.results}
        z_row, text_row, hover_row, status_row, celltext_row = [], [], [], [], []
        for name in check_names:
            result = results_by_name.get(name)
            if result is None:
                # Not evaluated at this stage at all (as opposed to evaluated
                # and explicitly Skipped) -- from the viewer's perspective
                # there is equally no data here, so render it identically.
                z_row.append(np.nan)
                status_row.append("skipped")
                celltext_row.append("—")
                text_row.append("")
                hover_row.append(
                    f"<b>{name}</b><br>Stage: {label}<br>"
                    "— SKIPPED — not evaluated at this stage<br>"
                    "<i>This check is only computed at a later stage "
                    "(e.g. it needs data only available once the run ends).</i>"
                )
                continue
            intensity, status = _result_status(result)
            report_or_skip = status in ("report_only", "skipped")
            z_row.append(np.nan if report_or_skip else intensity)
            status_row.append(status)
            if result.value is None:
                cell = "—"
            else:
                v = result.value
                for fmt in (".3f", ".2f", ".1f", ".0f"):
                    s = format(v, fmt)
                    if len(s) <= 5:
                        cell = s
                        break
                else:
                    raw = f"{v:.0e}"
                    mantissa, exp_part = raw.split("e")
                    exp_sign = exp_part[0]
                    exp_digits = exp_part[1:].lstrip("0") or "0"
                    cell = f"{mantissa}e{exp_digits}" if exp_sign == "+" else f"{mantissa}e-{exp_digits}"
            celltext_row.append(cell)
            text_row.append("" if report_or_skip else cell)
            t = result.threshold
            if t is None:
                thresh = "—"
            elif isinstance(t, tuple):
                thresh = f"[{t[0]:.3f}, {t[1]:.3f}]"
            else:
                thresh = f"{t:.3f}"
            status_label = {
                "pass": "✓ PASS", "fail": "✗ FAIL",
                "report_only": "◦ REPORT ONLY — no threshold configured",
                "skipped": "— SKIPPED — data unavailable",
            }[status]
            hover_row.append(
                f"<b>{result.name}</b><br>"
                f"Stage: {label}<br>"
                f"Value: {celltext_row[-1]}<br>"
                f"Threshold: {thresh}<br>"
                f"{status_label}<br>"
                f"<i>{result.message}</i>"
            )
        z_by_stage.append(z_row)
        text_by_stage.append(text_row)
        hover_by_stage.append(hover_row)
        status_by_stage.append(status_row)
        celltext_by_stage.append(celltext_row)

    # Transpose so rows = checks, cols = stages.
    n_checks = len(check_names)
    n_stages = len(stage_labels)
    z        = [[z_by_stage[s][c]        for s in range(n_stages)] for c in range(n_checks)]
    text     = [[text_by_stage[s][c]     for s in range(n_stages)] for c in range(n_checks)]
    hover    = [[hover_by_stage[s][c]    for s in range(n_stages)] for c in range(n_checks)]
    status   = [[status_by_stage[s][c]   for s in range(n_stages)] for c in range(n_checks)]
    celltext = [[celltext_by_stage[s][c] for s in range(n_stages)] for c in range(n_checks)]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=stage_labels,
        y=abbrevs,
        text=text,
        customdata=hover,
        texttemplate="<b>%{text}</b>",
        textfont=dict(size=11),
        colorscale=[
            [0.00, "#7b0000"],
            [0.25, "#c0392b"],
            [0.45, "#f0a09a"],
            [0.50, "#f5f5f5"],
            [0.55, "#9fd49f"],
            [0.75, "#27ae60"],
            [1.00, "#1a5e35"],
        ],
        zmin=0, zmax=1,
        showscale=False,
        hovertemplate="%{customdata}<extra></extra>",
        xgap=2,
        ygap=2,
    ))

    # Hollow-marker overlays for the two non-verdict states: a scatter trace
    # keyed to the same categorical (stage, check) coordinates as the
    # heatmap, so it lines up exactly without any pixel-coordinate
    # arithmetic. Each also gets its own legend entry, which the bare
    # heatmap otherwise has none of (showscale=False). The cell's numeric
    # value is drawn here too (mode="markers+text"), in this trace's own
    # muted colour, since the heatmap's own text was left blank for these
    # cells above.
    for state, symbol, color, legend_name in (
        ("report_only", "square-open", "#7a7a7a", "Report-only (no threshold)"),
        ("skipped", "circle-open", "#8888c0", "Skipped (no data)"),
    ):
        xs, ys, vals = [], [], []
        for ci in range(n_checks):
            for si in range(n_stages):
                if status[ci][si] == state:
                    xs.append(stage_labels[si])
                    ys.append(abbrevs[ci])
                    vals.append(celltext[ci][si])
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers+text",
                marker=dict(symbol=symbol, size=32, color=color, line=dict(width=2, color=color)),
                text=vals,
                textfont=dict(size=11, color=color),
                name=legend_name,
                showlegend=True,
                hoverinfo="skip",
            ))

    fig.update_layout(
        title=dict(
            text=f"",
            font=dict(size=15),
        ),
        xaxis=dict(title="", side="top", tickfont=dict(size=13)),
        yaxis=dict(title="Audit check", tickfont=dict(size=13), autorange="reversed"),
        height=max(260, n_checks * 44 + 100),
        width=max(600, n_stages * 40 + 200),
        margin=dict(l=150, r=20, t=90, b=20),
        plot_bgcolor="#dcdcdc",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
    )
    return fig


def _split_stage_reports_by_trackability(
    stage_reports: "list[tuple[str, Any]]",
) -> "tuple[list, list]":
    """Partition check names into step-trackable vs final-report-only.

    A check is "final-only" if it is skipped (or absent — see
    ``_fig_check_grid``'s union-of-names handling) at *every* intermediate
    stage and only produces a real value in the last stage report. This is
    the normal, structural situation for any check that needs data only
    available once the loop ends (``AuditHook.on_step`` only ever receives
    the per-step kwargs the loop chooses to pass it; a held-out test set,
    an ensemble evaluation, a replication arm, or a refit sweep is
    deliberately computed once, post-loop, rather than re-run at every
    ``check_every`` snapshot, which would be prohibitively expensive for
    e.g. bootstrap refit checks). Mixing both kinds of check into one grid
    makes the grid mostly empty cells for any pipeline with more final-only
    checks than step-trackable ones — this split is what lets each half be
    rendered at a size that matches how much real data it actually has.

    Returns
    -------
    (trackable_names, final_only_names)
        Both preserve the order names first appear in the final report.
    """
    if len(stage_reports) < 2:
        return [r.name for r in stage_reports[-1][1].results], []
    *intermediate, (_, final_report) = stage_reports
    trackable = set()
    for _, rep in intermediate:
        for r in rep.results:
            _, status = _result_status(r)
            if status != "skipped":
                trackable.add(r.name)
    trackable_names, final_only_names = [], []
    for r in final_report.results:
        (trackable_names if r.name in trackable else final_only_names).append(r.name)
    return trackable_names, final_only_names


def _filter_stage_reports(stage_reports: "list[tuple[str, Any]]", names: "list") -> "list":
    """Copy of ``stage_reports`` with each report's ``.results`` filtered to
    ``names`` (preserving ``names``' order). A stage missing a name entirely
    (e.g. an intermediate report from a pipeline that doesn't yet know about
    a final-only check) simply omits it — ``_fig_check_grid`` already
    renders an absent check the same as an explicitly skipped one.
    """
    from .base import AuditReport

    name_set = set(names)
    order = {n: i for i, n in enumerate(names)}
    out = []
    for label, rep in stage_reports:
        by_name = {r.name: r for r in rep.results if r.name in name_set}
        filtered = sorted(by_name.values(), key=lambda r: order[r.name])
        out.append((label, AuditReport(results=filtered, metadata=rep.metadata)))
    return out


def check_grid_figures(
    stage_reports: "list[tuple[str, Any]]",
    run_name: str,
) -> "tuple[Any, Any]":
    """Build the check-grid figure(s) for a run, splitting step-trackable
    checks from final-report-only ones (see
    :func:`_split_stage_reports_by_trackability`) so neither drowns the
    other: a pipeline with e.g. 12 step-trackable and 18 final-only checks
    would otherwise render one 30-row grid where 18 rows are empty hollow
    circles across every intermediate column but the last, which is both
    hard to read and easy to mistake for something being broken rather than
    working as designed.

    Returns
    -------
    (fig_trackable, fig_final_only)
        ``fig_trackable`` covers every stage exactly as ``_fig_check_grid``
        always has. ``fig_final_only`` is a single-column grid (``None`` if
        every check turned out to be step-trackable) — same visual language
        (colour, hollow markers for report-only checks among the final-only
        set), just one column wide since there is only ever one stage's
        worth of data for these checks.
    """
    trackable_names, final_only_names = _split_stage_reports_by_trackability(stage_reports)
    fig_trackable = _fig_check_grid(_filter_stage_reports(stage_reports, trackable_names), run_name)
    fig_final_only = None
    if final_only_names:
        final_stage = [stage_reports[-1]]
        fig_final_only = _fig_check_grid(
            _filter_stage_reports(final_stage, final_only_names),
            f"{run_name} — final-report-only checks",
        )
        if fig_final_only is not None:
            # The single-column grid's default width formula is sized for
            # the check names alone; widen it enough for the title text too.
            title_width = 14 * len(fig_final_only.layout.title.text) + 40
            fig_final_only.update_layout(width=max(fig_final_only.layout.width, title_width))
    return fig_trackable, fig_final_only


def _fig_state_heatmap(
    history: "list[dict]",
    run_name: str,
) -> Any:
    """Plotly heatmap: x = state-vector components, y = AL step.

    Each column is independently min-max normalised so components with
    different scales remain visible.
    """
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError:
        return None

    n_steps = len(history)
    matrix = np.array(
        [[h.get(k, np.nan) for k in _STATE_KEYS] for h in history]
    )

    col_min = np.nanmin(matrix, axis=0)
    col_max = np.nanmax(matrix, axis=0)
    norm = (matrix - col_min) / np.where(col_max - col_min > 1e-12, col_max - col_min, 1)

    hover = [
        [
            f"<b>{_STATE_KEYS[j]}</b><br>"
            f"Step: {i}<br>"
            f"Raw value: {matrix[i, j]:.4f}<br>"
            f"Normalised: {norm[i, j]:.3f}"
            for j in range(len(_STATE_KEYS))
        ]
        for i in range(n_steps)
    ]
    text = [
        [f"{matrix[i, j]:.3f}" for j in range(len(_STATE_KEYS))]
        for i in range(n_steps)
    ]

    fig = go.Figure(go.Heatmap(
        z=norm,
        x=_STATE_KEYS,
        y=list(range(n_steps)),
        text=text if n_steps <= 60 else None,
        texttemplate="%{text}" if n_steps <= 60 else None,
        customdata=hover,
        colorscale="Viridis",
        zmin=0, zmax=1,
        hovertemplate="%{customdata}<extra></extra>",
        colorbar=dict(
            title=dict(text="Normalised<br>value", side="right"),
            thickness=14,
        ),
        xgap=1,
        ygap=0,
    ))

    fig.update_layout(
        title=dict(
            text=(
                f"Uncertainty state vector — {run_name}<br>"
                f"<sup>Columns independently normalised · hover for raw values</sup>"
            ),
            font=dict(size=14),
        ),
        xaxis=dict(title="Uncertainty state component", tickfont=dict(size=12)),
        yaxis=dict(
            title="Active learning step",
            autorange="reversed",
            tickfont=dict(size=10),
        ),
        height=max(420, n_steps * 14 + 140),
        margin=dict(l=70, r=80, t=100, b=60),
    )
    return fig


def _fig_pareto_scenarios(
    pareto_data: "dict[str, list[tuple[float, float, str]]]",
    scenario_styles: "dict[str, dict] | None" = None,
) -> Any:
    """Pareto frontier of (CalibrationError, MAE) across all scenarios and stages.

    Parameters
    ----------
    pareto_data :
        ``{scenario_name: [(ece, mae, stage_label), …]}``
    scenario_styles :
        Per-scenario visual style dicts with keys ``color``, ``marker``,
        ``label``.  Falls back to a generic style for unknown names.
    """
    import matplotlib.patches as mpatches

    styles = scenario_styles or {}

    all_ece, all_mae = [], []
    for pts in pareto_data.values():
        for ece, mae, _ in pts:
            all_ece.append(ece)
            all_mae.append(mae)

    x = np.array(all_ece)
    y = np.array(all_mae)
    n = len(x)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and x[j] <= x[i] and y[j] <= y[i] and (x[j] < x[i] or y[j] < y[i]):
                dominated[i] = True
                break

    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    pt_idx = 0
    for sname, pts in pareto_data.items():
        style = styles.get(sname, {"color": "C4", "marker": "x", "label": sname})
        eces = [p[0] for p in pts]
        maes = [p[1] for p in pts]
        n_pts = len(pts)
        ax.plot(eces, maes, color=style["color"], lw=0.8, alpha=0.5, ls="-")
        for k in range(n_pts):
            is_dom = dominated[pt_idx]
            ax.scatter(
                [eces[k]], [maes[k]],
                c=style["color"], s=(30 if not is_dom else 14),
                marker=style["marker"], zorder=(4 if not is_dom else 2),
                alpha=(0.9 if not is_dom else 0.35),
                linewidths=0.5,
                edgecolors="k" if not is_dom else "none",
            )
            pt_idx += 1

    pareto_idx = np.where(~dominated)[0]
    if len(pareto_idx) > 0:
        order = np.argsort(x[pareto_idx])
        px, py = x[pareto_idx][order], y[pareto_idx][order]
        for k in range(len(px) - 1):
            ax.plot([px[k], px[k + 1]], [py[k], py[k]],
                    color="k", lw=1.2, ls="--", alpha=0.7)
            ax.plot([px[k + 1], px[k + 1]], [py[k], py[k + 1]],
                    color="k", lw=1.2, ls="--", alpha=0.7)

    handles = [
        mpatches.Patch(
            color=styles.get(n, {"color": "C4"})["color"],
            label=styles.get(n, {"label": n})["label"],
        )
        for n in pareto_data
    ]

    ax.set_xlabel("Calibration Error (ECE)")
    ax.set_ylabel("Mean absolute error (MAE)")
    ax.set_box_aspect(1) 
    ax.grid(False)
    ax.legend(handles=handles, frameon=False,
            fontsize=_RCPARAMS["legend.fontsize"])
    fig.tight_layout()
    return fig


def _fig_calibration_curve(result: Any) -> Optional[Any]:
    """Calibration reliability diagram for ``CalibrationErrorCheck`` results.

    Reads ``confidence_levels`` and ``observed_fractions`` from
    ``result.details``.  Returns ``None`` if the details are absent.
    """
    d = result.details
    expected = d.get("confidence_levels")
    observed = d.get("observed_fractions")
    if expected is None or observed is None:
        return None

    expected = np.asarray(expected)
    observed = np.asarray(observed)
    ce = d.get("calibration_error", float("nan"))

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Perfect calibration")
    ax.plot(expected, observed, color="C0", label="Observed")
    ax.fill_between(
        expected, expected, observed,
        alpha=0.15, color="C1", label="Miscalibration",
    )
    if not np.isnan(ce):
        ax.text(
            0.05, 0.95, f"CE = {ce:.4f}",
            transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            fontsize=_RCPARAMS["legend.fontsize"],
        )
    ax.set_xlabel("Expected coverage")
    ax.set_ylabel("Observed coverage")
    ax.legend(frameon=False)
    ax.grid(False)
    fig.tight_layout()
    return fig


def _fig_calibration_curves_all(
    scenario_results: Dict[str, Any],
    scenario_styles: Dict[str, Any],
) -> Optional[Any]:
    """Reliability-diagram grid — one panel per calibration scenario.

    Grid size is adaptive (not fixed 2×2): a hardcoded 2×2 grid silently
    dropped any scenario past the fourth via ``zip`` truncation, with no
    indication in the figure that anything was missing.
    """
    names = list(scenario_results.keys())
    if not names:
        return None

    ncols = 2 if len(names) <= 4 else 3
    nrows = (len(names) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7, 7), squeeze=False)
    axes_flat = list(axes.flat)

    for i, (ax, name) in enumerate(zip(axes_flat, names)):
        result = scenario_results[name]
        d = result.details or {}
        expected = d.get("confidence_levels")
        observed = d.get("observed_fractions")
        if expected is None or observed is None:
            ax.set_visible(False)
            continue

        expected = np.asarray(expected)
        observed = np.asarray(observed)
        ce = d.get("calibration_error", float("nan"))
        style = scenario_styles.get(name, {"color": "C4", "label": name})

        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
        ax.plot(expected, observed, color=style["color"], lw=1.5)
        ax.fill_between(expected, expected, observed, alpha=0.15, color=style["color"])
        if not np.isnan(ce):
            ax.text(
                0.05, 0.95, f"CE = {ce:.4f}",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
        ax.set_title(style["label"], fontsize=_RCPARAMS["axes.titlesize"])
        ax.set_xlabel("Expected coverage")
        ax.set_ylabel("Observed coverage")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(False)

    for ax in axes_flat[len(names):]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig


def _fig_metric_correlations(
    intermediate_reports: "list[Any]",
    run_name: str,
) -> Any:
    """Matplotlib figure: pairwise correlations of audit check metrics over time.

    Computes Spearman correlations between all combinations of audit check
    result values (CalibrationError, IntervalCoverage, VarianceAlignment, etc.)
    across all intermediate pipeline stages.

    Parameters
    ----------
    intermediate_reports : list[AuditReport]
        Intermediate reports from hook.intermediate_reports or a full report list.
    run_name : str
        Scenario name for figure title.

    Returns
    -------
    matplotlib figure object, or None if insufficient data
    """
    from scipy.stats import spearmanr

    if not intermediate_reports:
        return None

    # Extract all check names and their values across stages
    check_names = []
    stage_values: dict[str, list[float]] = {}
    
    for report in intermediate_reports:
        if not hasattr(report, 'results'):
            continue
        for result in report.results:
            check_name = result.name
            if check_name not in check_names:
                check_names.append(check_name)
            if check_name not in stage_values:
                stage_values[check_name] = []
            
            # Use the result value if available, otherwise skip
            if result.value is not None:
                stage_values[check_name].append(float(result.value))
            else:
                stage_values[check_name].append(np.nan)
    
    # Filter to only checks with enough genuine (non-NaN) values to
    # correlate. A check that's Skipped at every intermediate stage (e.g. a
    # final-report-only check like RSE/DUG/AFC, which need kwargs only
    # available at hook.on_end() -- see check_grid_figures' identical
    # "trackable vs final-only" split) still appears in every intermediate
    # report's .results as an AuditResult, just with value=None, so it was
    # appending np.nan every time rather than being absent. The previous
    # filter (`len(stage_values[name]) > 0`) is true for ANY such list, NaN
    # or not, so these checks were silently included as an axis label with
    # zero real data behind it -- rendered as a blank/zero row rather than
    # excluded, which looks like "reported but always uncorrelated" instead
    # of "never had data to correlate in the first place". Requiring >= 3
    # genuine values matches the >2-valid-points gate the correlation loop
    # below already needs to compute anything for this check at all.
    available_checks = [
        name for name in check_names
        if name in stage_values
        and np.sum(~np.isnan(np.asarray(stage_values[name], dtype=float))) >= 3
    ]
    
    if len(available_checks) < 2:
        return None

    # Ensure all check value lists have the same length
    n_stages = len(intermediate_reports)
    for check_name in available_checks:
        vals = stage_values[check_name]
        if len(vals) < n_stages:
            # Pad with NaN if necessary
            stage_values[check_name] = vals + [np.nan] * (n_stages - len(vals))

    # Compute the correlation matrix between all checks
    n_checks = len(available_checks)
    corr_matrix = np.zeros((n_checks, n_checks))
    
    for i, check1 in enumerate(available_checks):
        for j, check2 in enumerate(available_checks):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                v1 = np.array(stage_values[check1], dtype=float)
                v2 = np.array(stage_values[check2], dtype=float)
                
                # Only compute if both have sufficient valid data, and
                # neither is constant over that data -- a check that reports
                # the same value at every snapshot (e.g. a report-only check,
                # or a trend check pinned at 0) has an undefined rank
                # correlation with anything; spearmanr would return NaN for
                # it anyway (handled below), but only after emitting a
                # ConstantInputWarning on every such pair, so this is
                # checked directly rather than relying on that fallback.
                valid = ~(np.isnan(v1) | np.isnan(v2))
                if valid.sum() > 2 and np.std(v1[valid]) > 0 and np.std(v2[valid]) > 0:
                    try:
                        rho, _ = spearmanr(v1[valid], v2[valid])
                        corr_matrix[i, j] = np.abs(float(rho)) if not np.isnan(rho) else 0.0
                    except Exception:
                        corr_matrix[i, j] = 0.0
                else:
                    corr_matrix[i, j] = 0.0

    # Create figure
    fig_size = min(max(5.0, n_checks * 0.6), 12.0)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.95))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=0)
    masked_data = np.ma.masked_where(mask, corr_matrix)
    
    # Plot heatmap
    im = ax.imshow(masked_data, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    
    # Set ticks and labels
    ax.set_xticks(range(n_checks))
    ax.set_yticks(range(n_checks))
    
    # Abbreviate check names for display. Same fallback as _fig_check_grid
    # (the bare name) rather than a hard truncation — _CHECK_ABBREV now
    # covers every shipped check, so this only matters for a future
    # third-party check with no registered abbreviation.
    check_abbrevs = [_CHECK_ABBREV.get(name, name) for name in available_checks]
    ax.set_xticklabels(check_abbrevs, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(check_abbrevs, fontsize=10)
    
    # Add grid
    ax.set_xticks(np.arange(n_checks) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_checks) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.8, alpha=0.4)
    
    ax.set_title(
        f"Audit check correlations — {run_name}\n"
        f"(Spearman ρ across {n_stages} pipeline stages)",
        fontsize=12, weight="bold", pad=15,
    )
    ax.set_xlabel("Audit check", fontsize=11, weight="bold")
    ax.set_ylabel("Audit check", fontsize=11, weight="bold")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman ρ", rotation=270, labelpad=18, fontsize=10, weight="bold")
    
    fig.tight_layout()
    
    return fig




# ── Composition-space exploration figure ────────────────────────────────────

#: Pauling electronegativities used to encode binary-compound composition space.
_EN: Dict[str, float] = {
    "H": 2.20, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55, "N": 3.04,
    "O": 3.44, "F": 3.98, "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.90,
    "P": 2.19, "S": 2.58, "Cl": 3.16, "K": 0.82, "Ca": 1.00, "Sc": 1.36,
    "Ti": 1.54, "V": 1.63, "Cr": 1.66, "Mn": 1.55, "Fe": 1.83, "Co": 1.88,
    "Ni": 1.91, "Cu": 1.90, "Zn": 1.65, "Ga": 1.81, "Ge": 2.01, "As": 2.18,
    "Se": 2.55, "Br": 2.96, "Rb": 0.82, "Sr": 0.95, "Y": 1.22, "Zr": 1.33,
    "Nb": 1.60, "Mo": 2.16, "Tc": 1.90, "Ru": 2.20, "Rh": 2.28, "Pd": 2.20,
    "Ag": 1.93, "Cd": 1.69, "In": 1.78, "Sn": 1.96, "Sb": 2.05, "Te": 2.10,
    "I": 2.66, "Cs": 0.79, "Ba": 0.89, "La": 1.10, "Ce": 1.12, "Pr": 1.13,
    "Nd": 1.14, "Sm": 1.17, "Eu": 1.20, "Gd": 1.20, "Tb": 1.10, "Dy": 1.22,
    "Ho": 1.23, "Er": 1.24, "Tm": 1.25, "Yb": 1.10, "Lu": 1.27, "Hf": 1.30,
    "Ta": 1.50, "W": 2.36, "Re": 1.90, "Os": 2.20, "Ir": 2.20, "Pt": 2.28,
    "Au": 2.54, "Hg": 2.00, "Tl": 1.62, "Pb": 2.33, "Bi": 2.02, "Ac": 1.10,
    "Th": 1.30, "U": 1.38,
}

_EL_RE = re.compile(r"([A-Z][a-z]?)[\d.]*")


def _parse_en_pair(formula: str):
    """Return (en_low, en_high) for a binary formula, or None."""
    elems = list(dict.fromkeys(_EL_RE.findall(str(formula))))
    if len(elems) != 2:
        return None
    ea, eb = _EN.get(elems[0], 0.0), _EN.get(elems[1], 0.0)
    if ea == 0.0 or eb == 0.0:
        return None
    return (min(ea, eb), max(ea, eb))


def plot_exploration_campaign(
    df_all: Any,
    feat: list,
    target: str,
    seed_df: Any,
    queried_batches: list,
    model_label: str,
    out_dir: Path,
) -> None:
    """Materials exploration map and chemical-space coverage (fig9).

    Left panel — composition space (Pauling EN axes, shown as a hexbin density
    map) when the dataframe has a ``Composition`` column, otherwise a 2-D PCA
    projection.  Each queried batch is overlaid as coloured circles (plasma,
    dark = early, bright = late).

    Right panel — cumulative coverage and per-step batch novelty, both
    computed in the **same 2-D space** as the left panel.  Computing in 2-D
    avoids the curse of dimensionality that inflates the 1-NN radius in
    high-dimensional feature space and causes coverage to saturate at 100 %
    immediately.

    * **Coverage** (solid line): fraction of the full pool whose nearest
      neighbour in the queried set falls within the coverage radius.
    * **Batch novelty** (bars): fraction of each queried batch that lies
      outside the coverage radius of all previously queried points.
    """
    import matplotlib.colors as mcolors
    import matplotlib.ticker as mticker
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    has_comp = "Composition" in df_all.columns

    # ── Build 2-D coordinates (EN or PCA) ────────────────────────────────────
    batch_coords_2d: list[np.ndarray] = []

    if has_comp:
        pairs_all = df_all["Composition"].map(_parse_en_pair)
        valid_all = pairs_all.notna()
        xs_all = np.array([v[0] for v in pairs_all[valid_all]], dtype=float)
        ys_all = np.array([v[1] for v in pairs_all[valid_all]], dtype=float)

        pairs_seed = seed_df["Composition"].map(_parse_en_pair)
        valid_seed = pairs_seed.notna()
        xs_seed = np.array([v[0] for v in pairs_seed[valid_seed]], dtype=float)
        ys_seed = np.array([v[1] for v in pairs_seed[valid_seed]], dtype=float)

        xs_q, ys_q, step_q = [], [], []
        for k, batch in enumerate(queried_batches):
            pts = []
            for _, row in batch.iterrows():
                pair = _parse_en_pair(str(row.get("Composition", "")))
                if pair:
                    pts.append(pair)
                    xs_q.append(pair[0])
                    ys_q.append(pair[1])
                    step_q.append(k)
            batch_coords_2d.append(
                np.array(pts, dtype=float) if pts else np.empty((0, 2))
            )

        xlabel = "Pauling EN  (electropositive)"
        ylabel = "Pauling EN  (electronegative)"
        panel_title = "Composition space"

    else:
        from sklearn.decomposition import PCA

        common_feat = [c for c in feat if c in df_all.columns]
        X_all = df_all[common_feat].values.astype(float)
        nc = min(2, X_all.shape[1], len(X_all) - 1)
        pca = PCA(n_components=nc)
        Z_all = pca.fit_transform(X_all)
        xs_all = Z_all[:, 0]
        ys_all = Z_all[:, 1] if nc > 1 else np.zeros(len(Z_all))

        common_s = [c for c in feat if c in seed_df.columns]
        Z_seed = pca.transform(seed_df[common_s].values.astype(float))
        xs_seed = Z_seed[:, 0]
        ys_seed = Z_seed[:, 1] if nc > 1 else np.zeros(len(Z_seed))

        xs_q, ys_q, step_q = [], [], []
        for k, batch in enumerate(queried_batches):
            common_b = [c for c in feat if c in batch.columns]
            if common_b:
                Z_b = pca.transform(batch[common_b].values.astype(float))
                b_x = Z_b[:, 0]
                b_y = Z_b[:, 1] if Z_b.shape[1] > 1 else np.zeros(len(Z_b))
                pts = np.column_stack([b_x, b_y])
            else:
                pts = np.empty((0, 2))
            batch_coords_2d.append(pts)
            for i in range(len(pts)):
                xs_q.append(pts[i, 0])
                ys_q.append(pts[i, 1])
                step_q.append(k)

        xlabel = "PC 1"
        ylabel = "PC 2"
        panel_title = "Feature space  (PCA)"

    xs_q = np.array(xs_q, dtype=float)
    ys_q = np.array(ys_q, dtype=float)
    step_q = np.array(step_q, dtype=int)
    n_steps = len(queried_batches)
    step_norm = mcolors.Normalize(vmin=0, vmax=max(1, n_steps - 1))

    # ── Coverage / novelty via coarse EN/PCA grid ────────────────────────────
    # Grid-based metrics avoid the NN-radius degeneracy: a fine radius makes
    # coverage trivially low and novelty trivially 100% because each point
    # covers a tiny neighbourhood.  A coarse grid (12×12 bins) assigns each
    # point to a cell; coverage = cumulative fraction of non-empty pool cells
    # visited; novelty = fraction of each batch landing in cells not yet seen.
    # With ~50-80 non-empty cells in a 12×12 grid, both metrics vary over a
    # meaningful range throughout the 50-step campaign.
    cov_vals: list[float] = []
    nov_vals: list[float] = []
    coords_all_2d = (
        np.column_stack([xs_all, ys_all]) if len(xs_all) else np.empty((0, 2))
    )
    coords_seed_2d = (
        np.column_stack([xs_seed, ys_seed]) if len(xs_seed) else np.empty((0, 2))
    )

    if len(coords_all_2d) >= 2 and len(coords_seed_2d) >= 1:
        _N_BINS = 12
        _x_edges = np.linspace(coords_all_2d[:, 0].min(),
                               coords_all_2d[:, 0].max() + 1e-9, _N_BINS + 1)
        _y_edges = np.linspace(coords_all_2d[:, 1].min(),
                               coords_all_2d[:, 1].max() + 1e-9, _N_BINS + 1)

        def _cells(xy: np.ndarray) -> set:
            if len(xy) == 0:
                return set()
            xi = np.clip(np.searchsorted(_x_edges, xy[:, 0], side="right") - 1,
                         0, _N_BINS - 1)
            yi = np.clip(np.searchsorted(_y_edges, xy[:, 1], side="right") - 1,
                         0, _N_BINS - 1)
            return set(zip(xi.tolist(), yi.tolist()))

        _pool_cells = _cells(coords_all_2d)
        n_pool_cells = max(len(_pool_cells), 1)
        _visited = _cells(coords_seed_2d)

        for batch_xy in batch_coords_2d:
            batch_cells = _cells(batch_xy)
            new_cells = batch_cells - _visited
            nov_vals.append(
                len(new_cells) / max(len(batch_cells), 1) if batch_cells else 0.0
            )
            _visited |= batch_cells
            cov_vals.append(len(_visited & _pool_cells) / n_pool_cells)

    with plt.rc_context(_RCPARAMS):
        fig = plt.figure(figsize=(11, 3.5))
        gs = GridSpec(1, 2, figure=fig, width_ratios=[1.4, 1.0], wspace=0.52)

        # ── Left panel: exploration map ─────────────────────────────────────
        ax1 = fig.add_subplot(gs[0])

        if has_comp and len(xs_all):
            # Hexbin shows material density — cleaner than individual dots
            # because many OQMD compounds share identical Pauling EN coordinates.
            # No colorbar: lighter hex = fewer compounds, darker = denser cluster.
            ax1.hexbin(xs_all, ys_all, gridsize=28, cmap="Greys",
                       mincnt=1, linewidths=0.15, alpha=0.70, zorder=1)
        elif len(xs_all):
            ax1.scatter(xs_all, ys_all, c="grey", s=4, alpha=0.2,
                        linewidths=0, rasterized=True, zorder=1)

        if len(xs_seed):
            ax1.scatter(xs_seed, ys_seed, s=30, marker="D",
                        facecolors="#2c3e8c", edgecolors="white", linewidths=0.6,
                        zorder=3, label="Seed")

        if len(xs_q):
            sc = ax1.scatter(xs_q, ys_q, c=step_q, cmap="plasma", norm=step_norm,
                             s=20, marker="o", edgecolors="k", linewidths=0.3,
                             alpha=0.85, zorder=4, label="Queried")
            cb = plt.colorbar(sc, ax=ax1, fraction=0.034, pad=0.01, shrink=0.75)
            cb.set_label("AL step", fontsize=7)
            cb.ax.tick_params(labelsize=6)

        if has_comp:
            ax1.set_xlim(0.75, 4.15)
            ax1.set_ylim(0.75, 4.15)

        ax1.set_xlabel(xlabel, labelpad=3)
        ax1.set_ylabel(ylabel, labelpad=2)
        ax1.set_title(None)
        ax1.legend(fontsize=7, framealpha=0.85, loc="upper left",
                   handletextpad=0.3, borderpad=0.4)
        ax1.grid(False)

        # ── Right panel: coverage + novelty ─────────────────────────────────
        ax2 = fig.add_subplot(gs[1])

        if cov_vals:
            al_steps = np.arange(1, len(cov_vals) + 1)
            cov_arr = np.array(cov_vals)
            nov_arr = np.array(nov_vals)

            ax2.bar(al_steps, nov_arr, width=0.75, color="C1",
                    alpha=0.45, zorder=2)
            ax2.fill_between(al_steps, 0, cov_arr, alpha=0.18,
                             color="C0", linewidth=0)
            ax2.plot(al_steps, cov_arr, color="C0", lw=1.8, zorder=3)

            ax2.set_ylim(0, 1.05)
            ax2.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda y, _: f"{y:.0%}")
            )
            legend_handles = [
                Line2D([0], [0], color="C0", lw=1.8, label="Coverage"),
                Patch(facecolor="C1", alpha=0.55, label="Batch novelty"),
            ]
            ax2.legend(handles=legend_handles, fontsize=7,
                       framealpha=0.85, loc="upper right")

        ax2.set_xlabel("AL step", labelpad=3)
        ax2.set_ylabel("Fraction of pool  /  batch", labelpad=2)
        ax2.set_title(None)
        ax2.grid(False)

        fig.tight_layout()
        _save(fig, out_dir, "fig9_exploration_campaign")


def plot_discovery_rate(
    y_true_per_batch: list,
    df_all_target: "np.ndarray",
    stability_threshold: float,
    model_label: str,
    out_dir: Path,
) -> None:
    """Cumulative stable materials discovered vs random baseline (fig11).

    The primary evaluation metric of Montoya et al. (2020): how many
    stable/near-stable materials does the AL agent find per DFT calculation,
    compared to selecting candidates at random?

    A material is counted as "discovered stable" if its true target value
    (delta_e) falls at or below ``stability_threshold``.  The threshold is
    set to the 25th percentile of the full pool so the figure is meaningful
    for both real OQMD data and the synthetic fallback.

    Parameters
    ----------
    y_true_per_batch :
        List of 1-D arrays, one per AL step, each containing the true target
        values for that step's queried batch.
    df_all_target :
        True target values for every material in the full pool (seed + cand).
    stability_threshold :
        Materials with target ≤ threshold are considered stable.
    model_label, out_dir :
        Forwarded to title and ``_save``.
    """
    import matplotlib.ticker as mticker

    n_pool = len(df_all_target)
    n_stable_total = int((np.asarray(df_all_target) <= stability_threshold).sum())
    stable_frac = n_stable_total / max(n_pool, 1)

    cum_found: list[int] = []
    cum_queries: list[int] = []
    running = 0
    running_q = 0
    for batch_y in y_true_per_batch:
        b = np.asarray(batch_y, dtype=float)
        running += int((b <= stability_threshold).sum())
        running_q += len(b)
        cum_found.append(running)
        cum_queries.append(running_q)

    if not cum_queries:
        return

    cum_found_arr  = np.array(cum_found,   dtype=float)
    cum_q_arr      = np.array(cum_queries, dtype=float)

    # Random baseline: E[found at k queries] = k × stable_frac (hypergeometric)
    rand_exp = cum_q_arr * stable_frac
    rand_std = np.sqrt(cum_q_arr * stable_frac * (1.0 - stable_frac))

    with plt.rc_context(_RCPARAMS):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.5))

        # ── Left: absolute count ────────────────────────────────────────────
        ax1.plot(cum_q_arr, cum_found_arr, color="C0", lw=1.5, label=model_label)
        ax1.plot(cum_q_arr, rand_exp, color="k", lw=1.0, ls="--", alpha=0.65,
                 label="Random baseline")
        ax1.fill_between(cum_q_arr,
                         np.maximum(rand_exp - rand_std, 0), rand_exp + rand_std,
                         color="k", alpha=0.10, linewidth=0)
        ax1.set_xlabel("Cumulative AL queries")
        ax1.set_ylabel("Stable materials found")

        ax1.legend(frameon=False, fontsize=7,
                   bbox_to_anchor=(0.5, 1.02), loc="lower center", ncol=1)
        ax1.text(0.98, 0.05,
                 f"Stable in pool: {n_stable_total}/{n_pool} ({stable_frac:.1%})",
                 transform=ax1.transAxes, ha="right", va="bottom", fontsize=7,
                 color="grey")
        ax1.grid(False)

        # ── Right: enrichment factor = AL found / random expected ──────────
        # Scale-invariant metric: > 1 means outperforming random, regardless
        # of pool size or absolute number of queries.
        enrichment = cum_found_arr / np.where(rand_exp > 0, rand_exp, np.nan)

        ax2.plot(cum_q_arr, enrichment, color="C0", lw=1.5, label=model_label)
        ax2.axhline(1.0, color="k", lw=1.0, ls="--", alpha=0.65, label="Random (= 1×)")
        ax2.set_xlabel("Cumulative AL queries")
        ax2.set_ylabel("Enrichment factor  (AL / random)")

        ax2.legend(frameon=False, fontsize=7,
                   bbox_to_anchor=(0.5, 1.02), loc="lower center", ncol=1)
        ax2.grid(False)

        # Annotate final enrichment
        final_enrich = float(enrichment[np.isfinite(enrichment)][-1]) if np.any(np.isfinite(enrichment)) else 1.0
        ax2.text(0.98, 0.95,
                 f"Final: {final_enrich:.1f}× random",
                 transform=ax2.transAxes, ha="right", va="top", fontsize=8, color="C0")

        fig.tight_layout()
        _save(fig, out_dir, "fig11_discovery_rate")


# ── Convenience runner ──────────────────────────────────────────────────────

def run_lyapunov_analysis(
    predictor,
    op_states: np.ndarray,
    gp_std_fn,
    model_label: str,
    out_dir: Path,
    dx: float = 1e-4,
) -> dict:
    """Run the full Lyapunov analysis for one surrogate model.

    Parameters
    ----------
    predictor : callable
        ``(state, action=None) → np.ndarray``
    op_states : np.ndarray, shape (N, D)
        Operating points in state space.
    gp_std_fn : callable
        ``(state: np.ndarray) → float``
    model_label, out_dir, dx :
        Forwarded to figure and CSV helpers.

    Returns
    -------
    dict with keys ``lambda_max``, ``gp_std``, ``eigenvalues``, ``P``,
    ``csv_path``.
    """
    import csv

    N = len(op_states)
    mean_state = op_states.mean(axis=0)

    all_eigs: list[np.ndarray] = []
    lambda_max_list: list[float] = []
    gp_std_list: list[float] = []
    rows: list[dict] = []

    print(f"  Running Lyapunov analysis — {N} operating points, D={op_states.shape[1]} …")
    for i, state in enumerate(op_states):
        J = numerical_jacobian(predictor, state, action=None, dx=dx)
        stab = eigenvalues_and_stability(J)
        gp_std = float(gp_std_fn(state))

        all_eigs.append(stab["eigenvalues"])
        lambda_max_list.append(stab["lambda_max"])
        gp_std_list.append(gp_std)
        rows.append({
            "model":        model_label,
            "op_point_idx": i,
            "lambda_max":   stab["lambda_max"],
            "gp_std_at_op_point": gp_std,
            "is_stable":    stab["is_stable"],
            "n_unstable":   stab["n_unstable"],
        })

    lambda_max_arr = np.array(lambda_max_list)
    gp_std_arr     = np.array(gp_std_list)
    all_eigs_flat  = np.concatenate(all_eigs)

    J_mean = numerical_jacobian(predictor, mean_state, action=None, dx=dx)
    P = compute_lyapunov(J_mean)
    if P is not None:
        rho_mean = float(np.abs(np.linalg.eigvals(J_mean)).max())
        if rho_mean < 1.0:
            print("  Lyapunov matrix P computed (mean operating point is stable)")
        else:
            print(f"  Mean operating point is unstable (|λ_max|={rho_mean:.3f}) — "
                  "P computed from a rescaled J_mean (spectral radius 0.99) "
                  "for a comparably-scaled contour")
    else:
        print("  Lyapunov solve failed (degenerate J_mean) — P omitted from contour plot")

    plot_poles(all_eigs_flat, model_label, out_dir)
    plot_stability_contours(P, op_states, lambda_max_arr, model_label, out_dir)
    plot_stability_vs_uncertainty(lambda_max_arr, gp_std_arr, model_label, out_dir)

    csv_path = out_dir / "lyapunov_stability.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved lyapunov_stability.csv ({N} rows)")

    n_stable = sum(r["is_stable"] for r in rows)
    print(f"  Stable: {n_stable}/{N}  "
          f"|λ_max| mean={lambda_max_arr.mean():.3f} "
          f"max={lambda_max_arr.max():.3f}")

    return {
        "lambda_max": lambda_max_arr,
        "gp_std":     gp_std_arr,
        "eigenvalues": all_eigs_flat,
        "P":           P,
        "csv_path":    csv_path,
        "n_stable":    n_stable,
    }


def run_dmdc_lyapunov_analysis(
    aug_states: np.ndarray,
    model_label: str,
    out_dir: Path,
    n_components: int = 5,
    gp_std_seq: np.ndarray | None = None,
    actions: np.ndarray | None = None,
    min_obs: int | None = None,
) -> dict:
    """DMDc-based Lyapunov analysis on an augmented state trajectory.

    Fits a reduced-order dynamics matrix ``A_r`` directly from the AL query
    trajectory via DMDc.  Unlike :func:`run_lyapunov_analysis` (which
    differentiates a scalar surrogate to get ``J = I − α H_f``), ``A_r`` is a
    *general* (non-symmetric) real matrix and will generally have complex
    conjugate eigenvalues representing spiral modes in the joint
    state/uncertainty space.

    The augmented state should be ``[surrogate-feature coords | posterior std]``
    — analogous to battery-forecast's ``[ECM means | ECM stds]``.

    Parameters
    ----------
    aug_states : np.ndarray, shape (T, D)
        Per-step augmented state trajectory.
    model_label : str
        Used in figure titles and CSV.
    out_dir : Path
        Output directory for figures and ``lyapunov_stability.csv``.
    n_components : int
        DMDc rank (clipped internally to ``min(n_components, D, T-1)``).
    gp_std_seq : np.ndarray, optional
        Per-step surrogate posterior std for :func:`plot_stability_vs_uncertainty`.
    actions : np.ndarray, optional
        Per-step action array, shape ``(T, m)``.  Defaults to a column of ones.
    min_obs : int, optional
        Minimum observations before fitting DMDc in the convergence sweep.
        Defaults to ``n_components + 2``.

    Returns
    -------
    dict with keys ``lambda_max`` (per-step array), ``eigenvalues``,
    ``P``, ``A_r``, ``B_r``, ``U_r``, ``csv_path``.
    """
    import csv as _csv
    from traits_audit import dmdc as dm

    aug_states = np.asarray(aug_states, dtype=np.float64)
    T = len(aug_states)
    if actions is None:
        actions = np.ones((T, 1))
    actions = np.asarray(actions, dtype=np.float64)
    if min_obs is None:
        min_obs = n_components + 2

    A_r, B_r, U_r = dm.fit_dmdc(aug_states, actions, n_components)
    eigs = np.linalg.eigvals(A_r)
    lm_final = float(np.abs(eigs).max())

    # Per-step lambda_max via growing-prefix DMDc
    conv = dm.stability_convergence(
        aug_states, actions, min_obs=min_obs, n_components=n_components,
    )
    lambda_max_arr = np.concatenate([np.full(min_obs, np.nan), conv])[:T]
    lm_filled = np.where(np.isfinite(lambda_max_arr), lambda_max_arr, lm_final)

    P = compute_lyapunov(A_r)
    if P is not None:
        rho = lm_final
        if rho < 1.0:
            print("  Lyapunov matrix P computed (final A_r is stable)")
        else:
            print(f"  Final A_r unstable (|λ_max|={rho:.3f}) — "
                  "P from rescaled A_r (spectral radius 0.99)")
    else:
        print("  Lyapunov solve failed — P omitted from contour plot")

    print(f"  Running DMDc Lyapunov analysis — T={T} steps, "
          f"D={aug_states.shape[1]} → r={A_r.shape[0]} …")

    plot_poles(eigs, model_label, out_dir)
    # Use raw aug_states for the scatter so the 2-D PCA reflects the true
    # trajectory geometry. Pass P=None since P lives in the r-dim DMDc space,
    # not the original D-dim space, so the contour overlay would be invalid.
    plot_stability_contours(None, aug_states, lm_filled, model_label, out_dir)
    if gp_std_seq is not None:
        std_arr = np.asarray(gp_std_seq, dtype=float)
        n = min(len(lm_filled), len(std_arr))
        plot_stability_vs_uncertainty(lm_filled[:n], std_arr[:n], model_label, out_dir)

    rows = [
        {
            "model":        model_label,
            "op_point_idx": i,
            "lambda_max":   float(lm_filled[i]),
            "gp_std":       float(gp_std_seq[i]) if gp_std_seq is not None and i < len(gp_std_seq) else float("nan"),
            "is_stable":    bool(lm_filled[i] < 1.0),
        }
        for i in range(T)
    ]
    csv_path = out_dir / "lyapunov_stability.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved lyapunov_stability.csv ({T} rows)")

    n_stable = sum(r["is_stable"] for r in rows)
    print(f"  Stable: {n_stable}/{T}  |λ_max| final={lm_final:.3f}")

    return {
        "lambda_max": lm_filled,
        "gp_std":     np.asarray(gp_std_seq) if gp_std_seq is not None else np.array([]),
        "eigenvalues": eigs,
        "P":           P,
        "A_r": A_r, "B_r": B_r, "U_r": U_r,
        "csv_path": csv_path,
    }
