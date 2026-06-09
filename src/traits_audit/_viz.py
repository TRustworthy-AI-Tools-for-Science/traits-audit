"""Visualisation utilities for the traits-audit uncertainty audit.

All public ``plot_*`` functions write publication-ready PNG figures to disk
at 300 dpi.  Private helpers (prefixed ``_``) produce Plotly
or matplotlib objects returned to callers (MLflow, demo scripts).

A single ``_RCPARAMS`` block is applied at import time so every figure
produced by this module inherits the same typography and line weights.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

import matplotlib.pyplot as plt

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

#: Short labels for check names used on the x-axis of the check-grid heatmap.
_CHECK_ABBREV: Dict[str, str] = {
    "CalibrationError":         "CalibError",
    "IntervalCoverage":         "IntCoverage",
    "VarianceAlignment":        "VarAlignment",
    "UncertaintyEvolution":     "UncEvolution",
    "UncertaintyAnomalies":     "UncAnomalies",
    "VarianceErrorCorrelation": "VarErrCorr",
}

#: Per-step scalar keys recorded by the audit hook and shown in the state heatmap.
_STATE_KEYS = ["uncertainty", "pool_sigma_mean", "pool_sigma_max", "abs_error"]


# ── Core Lyapunov functions ─────────────────────────────────────────────────

def numerical_jacobian(
    predictor,
    state: np.ndarray,
    action: np.ndarray | None = None,
    dx: float = 1e-4,
) -> np.ndarray:
    """n×n Jacobian of predictor(state, action) → ℝⁿ via central differences."""
    n = len(state)
    J = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        s_plus  = state.copy(); s_plus[i]  += dx
        s_minus = state.copy(); s_minus[i] -= dx
        col = (predictor(s_plus, action) - predictor(s_minus, action)) / (2.0 * dx)
        J[:, i] = col
    return J


def eigenvalues_and_stability(J: np.ndarray) -> dict:
    """Eigenvalue spectrum and discrete-time stability indicators."""
    eigs = np.linalg.eigvals(J)
    mags = np.abs(eigs)
    return {
        "eigenvalues": eigs,
        "magnitudes":  mags,
        "lambda_max":  float(mags.max()),
        "lambda_min":  float(mags.min()),
        "is_stable":   bool((mags < 1.0).all()),
        "n_unstable":  int((mags >= 1.0).sum()),
    }


def compute_lyapunov(A: np.ndarray) -> np.ndarray | None:
    """Solve discrete Lyapunov equation Aᵀ P A − P = −I.

    Returns P or ``None`` when the system is unstable.
    """
    if np.abs(np.linalg.eigvals(A)).max() >= 1.0:
        return None
    try:
        return solve_discrete_lyapunov(A.T, np.eye(len(A)))
    except Exception:
        return None


def make_gd_predictor(f_scalar, alpha: float = 0.01, eps: float = 1e-5):
    """Discrete gradient-descent step predictor: x_{t+1} = x_t − α ∇f(x_t)."""
    def predictor(state: np.ndarray, action=None) -> np.ndarray:
        n = len(state)
        grad = np.zeros(n)
        for i in range(n):
            s_p = state.copy(); s_p[i] += eps
            s_m = state.copy(); s_m[i] -= eps
            grad[i] = (f_scalar(s_p) - f_scalar(s_m)) / (2.0 * eps)
        return state - alpha * grad
    return predictor


# ── Private save helper ─────────────────────────────────────────────────────

def _save(fig, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── Matplotlib plot functions ───────────────────────────────────────────────

def plot_poles(
    all_eigenvalues: np.ndarray,
    model_label: str,
    out_dir: Path,
) -> None:
    """Eigenvalues on the complex unit circle (fig1_poles)."""
    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    eigs  = np.asarray(all_eigenvalues)
    mags  = np.abs(eigs)
    max_m = float(mags.max())

    lim = max(1.5, max_m * 1.15) if max_m <= 5.0 else 1.5

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
    ax.legend(frameon=False)
    ax.grid(False)
    fig.tight_layout()
    _save(fig, out_dir, "fig1_poles")
    print("  Saved fig1_poles.png")


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
    print("  Saved fig2_stability_contours.png")


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
    ax.legend(frameon=False)
    ax.grid(False)
    fig.tight_layout()
    _save(fig, out_dir, "fig3_stability_vs_unc")
    print("  Saved fig3_stability_vs_unc.png")


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
) -> None:
    """Pareto frontier in 2-D objective space (uncertainty × performance) (fig7).

    Highlights the non-dominated set of queried points: those achieving the
    best simultaneous trade-off between both objectives.
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

    fig, ax = plt.subplots(figsize=(3.5, 2.625))

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

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(frameon=False)
    ax.grid(False)
    ax.set_title(model_label)
    fig.tight_layout()
    _save(fig, out_dir, "fig7_pareto_frontier")
    print("  Saved fig7_pareto_frontier.png")


def plot_uncertainty_evolution(
    uncertainties: np.ndarray,
    model_label: str,
    out_dir: Path,
) -> None:
    """Per-step surrogate uncertainty over the AL loop (fig4)."""
    fig, ax = plt.subplots(figsize=(3.5, 2.625))
    ax.plot(np.arange(len(uncertainties)), uncertainties, color="C0", label=model_label)
    ax.set_xlabel("Step")
    ax.set_ylabel("Surrogate std")
    ax.legend(frameon=False)
    ax.grid(False)
    fig.tight_layout()
    _save(fig, out_dir, "fig4_uncertainty_evolution")
    print("  Saved fig4_uncertainty_evolution.png")


def plot_lyapunov_evolution(
    lambda_max_seq: np.ndarray,
    uncertainties: np.ndarray,
    model_label: str,
    out_dir: Path,
) -> None:
    """Dual-axis: |λ_max| and surrogate std over AL steps (fig5)."""
    n = len(lambda_max_seq)
    steps = np.arange(n)

    fig, ax1 = plt.subplots(figsize=(3.5, 2.625))
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

    fig.legend(handles=[l1, l2], loc="upper right",
               bbox_to_anchor=(1.0, 1.0), bbox_transform=ax1.transAxes,
               frameon=False, fontsize=_RCPARAMS["legend.fontsize"])
    ax1.grid(False)
    ax1.set_title(model_label)
    fig.tight_layout()
    _save(fig, out_dir, "fig5_lyapunov_evolution")
    print("  Saved fig5_lyapunov_evolution.png")


def plot_audit_evolution(
    pipeline,
    history: list[dict],
    model_label: str,
    out_dir: Path,
    snapshot_every: int = 5,
) -> None:
    """6-panel figure: each audit check metric vs AL step (fig6)."""
    n_steps = len(history)
    if n_steps < snapshot_every:
        return

    snap_steps = list(range(snapshot_every, n_steps + 1, snapshot_every))
    if snap_steps[-1] != n_steps:
        snap_steps.append(n_steps)

    records: dict[str, tuple[list, list]] = {}
    pass_at: dict[str, list[bool]] = {}

    for k in snap_steps:
        sub = history[:k]
        try:
            report = pipeline.run(sub)
        except Exception:
            continue
        for r in report.results:
            if r.value is None:
                continue
            records.setdefault(r.name, ([], []))[0].append(k)
            records[r.name][1].append(r.value)
            pass_at.setdefault(r.name, []).append(r.passed)

    if not records:
        return

    check_names = list(records.keys())
    n_checks = len(check_names)
    ncols = 3
    nrows = (n_checks + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0, nrows * 1.9))
    axes_flat = np.array(axes).flatten()

    for i, name in enumerate(check_names):
        ax = axes_flat[i]
        xs, ys = records[name]
        passed = pass_at.get(name, [True] * len(xs))
        colors = ["#27ae60" if p else "#c0392b" for p in passed]
        ax.plot(xs, ys, color="C0", lw=1.2)
        ax.scatter(xs, ys, c=colors, s=18, zorder=3)
        ax.set_title(name.replace("Check", ""),
                     fontsize=_RCPARAMS["legend.fontsize"])
        ax.set_xlabel("Step")
        ax.tick_params(labelsize=_RCPARAMS["xtick.labelsize"])
        ax.grid(False)

    for j in range(n_checks, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"{model_label} — Audit checks over AL steps",
                 fontsize=_RCPARAMS["font.size"])
    fig.tight_layout()
    _save(fig, out_dir, "fig6_audit_evolution")
    print("  Saved fig6_audit_evolution.png")


def plot_convergence(
    best_vals: np.ndarray,
    query_counts: np.ndarray,
    y_label: str,
    model_label: str,
    out_dir: Path,
    maximise: bool = False,
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
        if ``False``, minimised (e.g. error, Fréchet distance).
    """
    best_vals = np.asarray(best_vals, dtype=float)
    query_counts = np.asarray(query_counts, dtype=float)
    valid = np.isfinite(best_vals)
    if not valid.any():
        return

    fig, ax = plt.subplots(figsize=(3.5, 2.625))
    ax.plot(query_counts[valid], best_vals[valid], color="C0", label=model_label)
    # Seed baseline: dashed horizontal at the initial best value
    baseline = best_vals[valid][0]
    ax.axhline(baseline, color="k", lw=0.8, ls="--", alpha=0.5, label="Seed baseline")
    ax.set_xlabel("Cumulative AL queries")
    ax.set_ylabel(y_label)
    ax.legend(frameon=False)
    ax.grid(False)
    fig.tight_layout()
    _save(fig, out_dir, "fig8_convergence")
    print("  Saved fig8_convergence.png")


# ── Heatmap intensity helper ────────────────────────────────────────────────

def _result_intensity(result: Any) -> float:
    """Map an AuditResult to a continuous [0, 1] intensity.

    Returns
    -------
    float
        1.0 = deeply passing (large positive margin from threshold)
        0.5 = exactly at the threshold boundary
        0.0 = deeply failing (large negative margin)
    """
    v = result.value
    t = result.threshold
    if v is None or t is None:
        return 1.0 if result.passed else 0.0

    name = result.name
    if name == "CalibrationError":
        # Lower is better; PASS if v ≤ t
        signed = (t - v) / max(abs(t), 1e-6)
    elif name == "UncertaintyAnomalies":
        # Lower is better; PASS if v ≤ t
        signed = (t - v) / max(abs(t), 1e-6)
    elif name == "UncertaintyEvolution":
        # Higher (less negative) is better; PASS if v ≥ t
        span = max(abs(t) * 3, 0.10)
        signed = (v - t) / span
    elif name == "VarianceErrorCorrelation":
        # Higher is better; PASS if v ≥ t
        span = max(1.0 - t, 0.30)
        signed = (v - t) / span
    elif name == "IntervalCoverage":
        # t is (lo, hi) band; derive target and tolerance from it
        if isinstance(t, tuple):
            target = (t[0] + t[1]) / 2
            tol = max((t[1] - t[0]) / 2, 1e-6)
        else:
            target, tol = t, 0.15
        signed = (tol - abs(v - target)) / tol
    elif name == "VarianceAlignment":
        # Toward-target (ideal ratio = 1.0 = t); tolerance ≈ 0.5
        tol = 0.50
        signed = (tol - abs(v - t)) / tol
    else:
        return 1.0 if result.passed else 0.0

    return float(np.clip(0.5 + 0.5 * np.clip(signed, -1.0, 1.0), 0.0, 1.0))


# ── Plotly interactive figures ──────────────────────────────────────────────

def _fig_check_grid(
    stage_reports: "list[tuple[str, Any]]",
    run_name: str,
) -> Any:
    """Plotly heatmap: rows = audit checks, cols = pipeline stages.

    Cell intensity encodes how far the metric sits from the pass/fail
    threshold: dark green = deeply passing, white = at threshold,
    dark red = deeply failing.
    """
    import plotly.graph_objects as go

    check_names = [r.name for r in stage_reports[0][1].results]
    abbrevs = [_CHECK_ABBREV.get(n, n) for n in check_names]
    stage_labels = [label for label, _ in stage_reports]

    # Build [stage][check] intermediate arrays then transpose to [check][stage].
    z_by_stage, text_by_stage, hover_by_stage = [], [], []
    for label, rep in stage_reports:
        z_row, text_row, hover_row = [], [], []
        for result in rep.results:
            z_row.append(_result_intensity(result))
            text_row.append(f"{result.value:.3f}" if result.value is not None else "—")
            t = result.threshold
            if t is None:
                thresh = "—"
            elif isinstance(t, tuple):
                thresh = f"[{t[0]:.3f}, {t[1]:.3f}]"
            else:
                thresh = f"{t:.3f}"
            hover_row.append(
                f"<b>{result.name}</b><br>"
                f"Stage: {label}<br>"
                f"Value: {text_row[-1]}<br>"
                f"Threshold: {thresh}<br>"
                f"{'✓ PASS' if result.passed else '✗ FAIL'}<br>"
                f"<i>{result.message}</i>"
            )
        z_by_stage.append(z_row)
        text_by_stage.append(text_row)
        hover_by_stage.append(hover_row)

    # Transpose so rows = checks, cols = stages.
    n_checks = len(check_names)
    n_stages = len(stage_labels)
    z     = [[z_by_stage[s][c]     for s in range(n_stages)] for c in range(n_checks)]
    text  = [[text_by_stage[s][c]  for s in range(n_stages)] for c in range(n_checks)]
    hover = [[hover_by_stage[s][c] for s in range(n_stages)] for c in range(n_checks)]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=stage_labels,
        y=abbrevs,
        text=text,
        customdata=hover,
        texttemplate="<b>%{text}</b>",
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

    fig.update_layout(
        title=dict(
            text=f"Audit check summary — {run_name}",
            font=dict(size=15),
        ),
        xaxis=dict(title="Pipeline stage", side="top", tickfont=dict(size=12)),
        yaxis=dict(title="Audit check", tickfont=dict(size=12), autorange="reversed"),
        height=max(300, n_checks * 65 + 120),
        width=max(500, n_stages * 160 + 200),
        margin=dict(l=130, r=30, t=100, b=30),
        plot_bgcolor="#f8f9fa",
    )
    return fig


def _fig_state_heatmap(
    history: "list[dict]",
    run_name: str,
) -> Any:
    """Plotly heatmap: x = state-vector components, y = AL step.

    Each column is independently min-max normalised so components with
    different scales remain visible.
    """
    import plotly.graph_objects as go

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

    fig, ax = plt.subplots(figsize=(3.5, 2.625))

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
    ax.legend(handles=handles, frameon=False,
              fontsize=_RCPARAMS["legend.fontsize"])
    ax.set_xlabel("CalibrationError (ECE)")
    ax.set_ylabel("Mean absolute error")
    ax.grid(False)
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
    """2×2 reliability-diagram grid — one panel per calibration scenario."""
    names = list(scenario_results.keys())
    if not names:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.25))
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
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[len(names):]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig


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
            "gp_std":       gp_std,
            "is_stable":    stab["is_stable"],
            "n_unstable":   stab["n_unstable"],
        })

    lambda_max_arr = np.array(lambda_max_list)
    gp_std_arr     = np.array(gp_std_list)
    all_eigs_flat  = np.concatenate(all_eigs)

    J_mean = numerical_jacobian(predictor, mean_state, action=None, dx=dx)
    P = compute_lyapunov(J_mean)
    if P is not None:
        print("  Lyapunov matrix P computed (mean operating point is stable)")
    else:
        print("  Mean operating point is unstable — P omitted from contour plot")

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
    }
