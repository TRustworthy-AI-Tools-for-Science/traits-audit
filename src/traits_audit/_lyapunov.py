"""Lyapunov stability utilities for arbitrary-dimensional surrogate models.

Core functions (`numerical_jacobian`, `eigenvalues_and_stability`,
`compute_lyapunov`) are adapted from
``battery_forecast.bin.exp3_stability_analysis`` and generalised to work
with any state dimension and any scalar predictor.

The key addition over exp3 is `make_gd_predictor`, which wraps a scalar
surrogate f(x) → ℝ into the discrete gradient-descent map
x_{t+1} = x_t − α ∇f(x_t), whose Jacobian J = I − α H_f connects
surrogate curvature to dynamical stability.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       10,
    "axes.titlesize":  11,
    "axes.labelsize":  10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.5,
    "axes.linewidth":  0.8,
    "grid.alpha":      0.3,
    "figure.dpi":      300,
})


# ── Core Lyapunov functions ────────────────────────────────────────────────────

def numerical_jacobian(
    predictor,
    state: np.ndarray,
    action: np.ndarray | None = None,
    dx: float = 1e-4,
) -> np.ndarray:
    """n×n Jacobian of predictor(state, action) → ℝⁿ via central differences.

    ``action`` is passed through unchanged to ``predictor``; pass ``None``
    for surrogates that have no action input.
    """
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


def compute_lyapunov(A: np.ndarray, rho_max: float = 0.99) -> np.ndarray | None:
    """Solve discrete Lyapunov equation Aᵀ P A − P = −I.

    If ``A``'s spectral radius is ≥ 1 (the equation has no positive-definite
    solution in that case), ``A`` is rescaled to spectral radius *rho_max*
    before solving — the same approach used by
    :func:`traits_audit.dmdc.compute_gramians` — so V(x) = xᵀPx stays on a
    comparable scale across stable and unstable operating points. Returns
    ``None`` only if ``solve_discrete_lyapunov`` itself raises (e.g. a
    degenerate/singular ``A``).
    """
    rho = float(np.abs(np.linalg.eigvals(A)).max())
    if rho >= 1.0:
        A = A * (rho_max / rho)
    try:
        return solve_discrete_lyapunov(A.T, np.eye(len(A)))
    except Exception:
        return None


# ── Gradient-descent predictor ─────────────────────────────────────────────────

def make_gd_predictor(f_scalar, alpha: float = 0.01, eps: float = 1e-5):
    """Build a discrete gradient-descent step predictor for Lyapunov analysis.

    Models x_{t+1} = x_t − α ∇f(x_t).  The Jacobian of this map equals
    J = I − α H_f, so |λ(J)| < 1 iff all eigenvalues of H_f lie in (0, 2/α).

    **Purely real poles**: Because f_scalar is a real scalar function, its
    Hessian H_f is always real and symmetric, and J = I − α H_f inherits
    that symmetry.  All eigenvalues of J are therefore guaranteed to be purely
    real regardless of the surrogate architecture.  :func:`plot_poles`
    automatically renders a 1-D strip plot in this case.

    Parameters
    ----------
    f_scalar : callable
        ``f_scalar(state: np.ndarray) → float``
    alpha : float
        Gradient-descent step size used to define the dynamical system.
    eps : float
        Finite-difference step for computing ∇f numerically.
    """
    def predictor(state: np.ndarray, action=None) -> np.ndarray:
        n = len(state)
        grad = np.zeros(n)
        for i in range(n):
            s_p = state.copy(); s_p[i] += eps
            s_m = state.copy(); s_m[i] -= eps
            grad[i] = (f_scalar(s_p) - f_scalar(s_m)) / (2.0 * eps)
        return state - alpha * grad
    return predictor


# ── Figures ────────────────────────────────────────────────────────────────────

def _save(fig, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_poles(
    all_eigenvalues: np.ndarray,
    model_label: str,
    out_dir: Path,
) -> None:
    """Eigenvalue diagram for the GD-predictor Jacobian J = I − αH_f.

    For gradient-descent predictors built on a real scalar surrogate, H_f is
    always real and symmetric, so all eigenvalues of J are purely real.  In
    that case this function renders a 1-D strip plot along the real axis —
    much more readable than an empty complex plane.

    When significant imaginary parts are present (max|Im(λ)| > 1e-8·max|λ|)
    the function falls back to the full complex unit-circle diagram.
    """
    eigs = np.asarray(all_eigenvalues)
    mags = np.abs(eigs)
    max_m = float(mags.max()) if len(mags) else 1.0
    purely_real = (
        max_m == 0.0
        or float(np.abs(eigs.imag).max()) < 1e-8 * max_m
    )

    if purely_real:
        # ── 1-D strip plot: Re(λ) along the real axis ────────────────────
        re = eigs.real
        stable   = np.abs(re) < 1.0
        unstable = ~stable

        fig, ax = plt.subplots(figsize=(3.5, 2.0))

        rng_spread = float(np.abs(re).max()) * 1.15
        lim = max(1.3, rng_spread)

        # Jitter in y so overlapping poles are visible
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
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3, axis="x")
        fig.tight_layout()

    else:
        # ── Complex unit-circle diagram ───────────────────────────────────
        fig, ax = plt.subplots(figsize=(3.5, 3.5))

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
        ax.grid(True, alpha=0.3)
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
    """2-D PCA projection of the Lyapunov stability landscape."""
    from sklearn.decomposition import PCA

    n_comp = min(2, states.shape[1], len(states) - 1)
    pca = PCA(n_components=n_comp)
    z = pca.fit_transform(states - states.mean(axis=0))
    z2 = z[:, 1] if z.shape[1] > 1 else np.zeros(len(z))

    fig, ax = plt.subplots(figsize=(3.5, 5.5))

    if P is not None and n_comp == 2:
        V = pca.components_.T[:, :2]   # (D, 2)
        P2 = V.T @ P @ V               # (2, 2)
        lim = np.abs(z).max() * 1.2
        gs = np.linspace(-lim, lim, 120)
        gx, gy = np.meshgrid(gs, gs)
        gpts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        lyap = np.einsum("ni,ij,nj->n", gpts, P2, gpts).reshape(gx.shape)
        cf = ax.contourf(gx, gy, lyap, levels=15, cmap="YlOrRd", alpha=0.65)
        ax.contour(gx,  gy, lyap, levels=15, colors="k", linewidths=0.4, alpha=0.4)
        plt.colorbar(cf, ax=ax, label="V(x) = xᵀPx")

    from matplotlib.colors import TwoSlopeNorm
    lm_vals = np.asarray(lambda_max, dtype=float)
    lm_lo = float(np.percentile(lm_vals, 2))
    lm_hi = float(np.percentile(lm_vals, 98))
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
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "fig2_stability_contours")
    print("  Saved fig2_stability_contours.png")


def plot_stability_vs_uncertainty(
    lambda_max: np.ndarray,
    surrogate_std: np.ndarray,
    model_label: str,
    out_dir: Path,
) -> None:
    """Scatter: |λ_max| vs surrogate posterior std."""
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.scatter(surrogate_std, lambda_max,
               c="C0", s=20, alpha=0.6, linewidths=0, label=model_label)
    ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.6, label="Stability boundary")
    ax.set_xlabel("Surrogate posterior std")
    ax.set_ylabel("|λ_max|")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
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
    """Pareto frontier in 2-D objective space (uncertainty × performance).

    Highlights the non-dominated set of queried points: those achieving the
    best simultaneous trade-off between both objectives.  In an active learning
    context, Pareto-optimal queries have both good objective value *and*
    well-resolved surrogate uncertainty — making them the most trustworthy
    candidates for deployment.

    Parameters
    ----------
    x_vals, y_vals : array-like
        Per-queried-point objective values.
    minimize_x, minimize_y :
        Whether lower is better on each axis (True) or higher is better
        (False).  Use ``minimize_y=False`` when the y-axis is a quantity to
        *maximise* (e.g., battery discharge capacity).
    color_vals :
        If given, colours points by this scalar array (e.g., AL step index).
        A viridis colourbar is added.
    color_label :
        Label for the colourbar.
    """
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) == 0:
        return
    cv = np.asarray(color_vals, dtype=float)[valid] if color_vals is not None else None

    # Convert to minimisation for dominance comparison
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
        # Mark Pareto-optimal points with a ring
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

    # Staircase frontier connecting adjacent Pareto points
    for k in range(len(px) - 1):
        ax.plot([px[k], px[k + 1]], [py[k], py[k]],
                color="C1", lw=1.0, ls="--", alpha=0.8)
        ax.plot([px[k + 1], px[k + 1]], [py[k], py[k + 1]],
                color="C1", lw=1.0, ls="--", alpha=0.8)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_title(model_label)
    fig.tight_layout()
    _save(fig, out_dir, "fig7_pareto_frontier")
    print("  Saved fig7_pareto_frontier.png")


def plot_uncertainty_evolution(
    uncertainties: np.ndarray,
    model_label: str,
    out_dir: Path,
) -> None:
    """Per-step surrogate uncertainty over the AL loop."""
    fig, ax = plt.subplots(figsize=(3.5, 2.625))
    ax.plot(np.arange(len(uncertainties)), uncertainties, color="C0", label=model_label)
    ax.set_xlabel("Step")
    ax.set_ylabel("Surrogate std")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "fig4_uncertainty_evolution")
    print("  Saved fig4_uncertainty_evolution.png")


# ── Time-evolution figures ─────────────────────────────────────────────────────

def plot_lyapunov_evolution(
    lambda_max_seq: np.ndarray,
    uncertainties: np.ndarray,
    model_label: str,
    out_dir: Path,
) -> None:
    """Dual-axis plot: |λ_max| and surrogate std over the AL query trajectory.

    ``lambda_max_seq[i]`` is |λ_max| evaluated at the i-th queried operating
    point (in chronological query order, using the final surrogate).
    ``uncertainties[i]`` is the per-step posterior std logged by the audit hook.
    """
    n = len(lambda_max_seq)
    steps = np.arange(n)

    fig, ax1 = plt.subplots(figsize=(5.5, 3.5))
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

    fig.legend(handles=[l1, l2], loc="upper left",
               bbox_to_anchor=(1.0, 1.0), bbox_transform=ax1.transAxes,
               frameon=False, fontsize=8)
    ax1.grid(True, alpha=0.3)
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
    """6-panel figure: each audit check metric vs AL step.

    Re-runs the pipeline on growing sub-sequences of ``history`` at every
    ``snapshot_every`` steps to show how check values evolve as data
    accumulates.  Requires at least ``snapshot_every`` history entries.
    """
    n_steps = len(history)
    if n_steps < snapshot_every:
        return

    snap_steps = list(range(snapshot_every, n_steps + 1, snapshot_every))
    if snap_steps[-1] != n_steps:
        snap_steps.append(n_steps)

    records: dict[str, tuple[list, list]] = {}  # name → (steps, values)
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
        ax.set_title(name.replace("Check", ""), fontsize=8)
        ax.set_xlabel("Step", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    for j in range(n_checks, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"{model_label} — Audit checks over AL steps", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, "fig6_audit_evolution")
    print("  Saved fig6_audit_evolution.png")


# ── Convenience runner ─────────────────────────────────────────────────────────

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
        ``(state, action=None) → np.ndarray`` same length as state.
        Typically built with :func:`make_gd_predictor`.
    op_states : np.ndarray, shape (N, D)
        Operating points in state space (rows).
    gp_std_fn : callable
        ``(state: np.ndarray) → float`` surrogate posterior std.
    model_label : str
        Used in figure titles and the CSV ``model`` column.
    out_dir : Path
        Directory for figures and ``lyapunov_stability.csv``.
    dx : float
        Finite-difference step for the outer Jacobian computation.

    Returns
    -------
    dict with keys: ``lambda_max``, ``gp_std``, ``eigenvalues``, ``P``,
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

    # Lyapunov matrix at the mean operating point
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

    # Figures
    plot_poles(all_eigs_flat, model_label, out_dir)
    plot_stability_contours(P, op_states, lambda_max_arr, model_label, out_dir)
    plot_stability_vs_uncertainty(lambda_max_arr, gp_std_arr, model_label, out_dir)

    # CSV
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
