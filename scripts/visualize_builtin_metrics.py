#!/usr/bin/env python
"""
Standalone script — NOT part of the traits-audit package.

Generates two-panel (healthy / unhealthy) reference figures for each of the
six built-in AuditCheck classes and saves them as PNG files.

Usage (from repo root):
    python scripts/visualize_builtin_metrics.py
    python scripts/visualize_builtin_metrics.py --out path/to/output
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

# ── Publication style (mirrors src/traits_audit/_viz.py _RCPARAMS) ───────────
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
    "figure.dpi":      300,
})

PASS_COLOR = "#00b4d8"   # cyan
FAIL_COLOR = "#cc00cc"   # magenta
GRAY       = "#888888"


def _badge(ax, passed: bool, x: float = 0.97, y: float = 0.05,
           ha: str = "right", va: str = "bottom") -> None:
    """PASS / FAIL badge, defaulting to the lower-right corner of ax."""
    label = "PASS" if passed else "FAIL"
    color = PASS_COLOR if passed else FAIL_COLOR
    ax.text(
        x, y, label,
        transform=ax.transAxes, fontsize=9, fontweight="bold",
        color="white", ha=ha, va=va,
        bbox=dict(boxstyle="round,pad=0.25", facecolor=color, edgecolor="none"),
    )


def _ann(ax, text: str, x: float = 0.04, y: float = 0.96,
         ha: str = "left", va: str = "top") -> None:
    """Metric annotation box, defaulting to the upper-left corner of ax."""
    ax.text(
        x, y, text, transform=ax.transAxes, fontsize=8.5,
        va=va, ha=ha, zorder=10,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                  edgecolor="#cccccc", alpha=1.0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. CalibrationErrorCheck
# ─────────────────────────────────────────────────────────────────────────────

def plot_calibration_error(out_dir: Path) -> None:
    """Reliability diagram: deviations from the perfect-calibration diagonal.

    CalibrationErrorCheck computes the mean absolute deviation between the
    empirical coverage fraction and the nominal confidence level across 10
    bins.  Points on the diagonal mean the model is perfectly calibrated;
    points below it indicate overconfidence.
    """
    rng       = np.random.default_rng(0)
    n_levels  = 10
    levels    = np.linspace(0, 1, n_levels + 2)[1:-1]
    threshold = 0.10

    # Healthy: near-perfect — observed ≈ expected at every confidence level
    obs_h = np.clip(levels + rng.normal(0, 0.013, n_levels), 0, 1)
    ce_h  = float(np.mean(np.abs(obs_h - levels)))

    # Unhealthy: overconfident — model assigns tight intervals that miss targets
    #   observed coverage = 62 % of expected → systematic shortfall
    obs_u = np.clip(levels * 0.62 + rng.normal(0, 0.013, n_levels), 0, 1)
    ce_u  = float(np.mean(np.abs(obs_u - levels)))

    fig, (ax_h, ax_u) = plt.subplots(1, 2, figsize=(7, 3.5))

    for ax, obs, ce, title in (
        (ax_h, obs_h, ce_h, "Healthy"),
        (ax_u, obs_u, ce_u, "Unhealthy — overconfident model"),
    ):
        passed = ce <= threshold
        c = PASS_COLOR if passed else FAIL_COLOR

        ax.plot([0, 1], [0, 1], color=GRAY, lw=1, ls="--",
                label="Perfect calibration")
        for lv, ob in zip(levels, obs):
            ax.plot([lv, lv], [lv, ob], color=c, lw=0.9, alpha=0.5)
        ax.scatter(levels, obs, color=c, zorder=4, s=36, label="Observed")

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Expected coverage")
        ax.set_ylabel("Observed coverage")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8)
        _ann(ax, f"CE = {ce:.3f}   (threshold ≤ {threshold:.2f})",
             x=0.5, y=-0.28, ha="center", va="top")
        _badge(ax, passed)

    fig.tight_layout()
    fig.savefig(out_dir / "calibration_error.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 2. IntervalCoverageCheck
# ─────────────────────────────────────────────────────────────────────────────

def plot_interval_coverage(out_dir: Path) -> None:
    """Prediction intervals with inside / outside points colour-coded.

    IntervalCoverageCheck measures the fraction of true values that fall
    within ±1σ of the predicted mean.  For a Gaussian model this should be
    ≈ 68.3 %; the check passes when the observed coverage is within ±10 % of
    that target, i.e. in the band [0.583, 0.783].
    """
    rng          = np.random.default_rng(1)
    n            = 40
    expected_cov = 0.683
    tolerance    = 0.10
    band_lo      = expected_cov - tolerance  # 0.583
    band_hi      = expected_cov + tolerance  # 0.783

    x         = np.sort(rng.uniform(0, 1, n))
    mu        = np.sin(2 * np.pi * x)
    residuals = rng.normal(0, 0.30, n)
    y_true    = mu + residuals

    # Use empirical quantiles so coverage is exact regardless of seed.
    abs_res = np.abs(residuals)
    sigma_h = float(np.quantile(abs_res, 0.70))   # exactly 70 % inside → PASS
    sigma_u = float(np.quantile(abs_res, 0.35))   # exactly 35 % inside → FAIL

    def _panel(ax, sigma_val: float, title: str) -> None:
        covered = np.abs(residuals) <= sigma_val
        cov     = float(covered.mean())
        passed  = band_lo <= cov <= band_hi
        c       = PASS_COLOR if passed else FAIL_COLOR

        ax.fill_between(x, mu - sigma_val, mu + sigma_val, alpha=0.18, color=c)
        ax.plot(x, mu, color=c, lw=1.2, label="Predicted mean")
        ax.scatter(x[covered],  y_true[covered],  color=c, s=22, zorder=4)
        ax.scatter(x[~covered], y_true[~covered], color=FAIL_COLOR,
                   s=22, marker="x", zorder=5, label="Outside ±1σ")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8)
        _badge(ax, passed)

    fig, (ax_h, ax_u) = plt.subplots(1, 2, figsize=(7, 3.5))
    _panel(ax_h, sigma_val=sigma_h, title="Healthy")
    _panel(ax_u, sigma_val=sigma_u, title="Unhealthy — intervals too narrow")

    fig.tight_layout()
    fig.savefig(out_dir / "interval_coverage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 3. VarianceAlignmentCheck
# ─────────────────────────────────────────────────────────────────────────────

def plot_variance_alignment(out_dir: Path) -> None:
    """Side-by-side bars: mean predicted variance vs mean actual squared error.

    VarianceAlignmentCheck computes ratio = mean(σ²) / mean((y−ŷ)²).
    A ratio near 1 means the model's uncertainty matches its actual error
    magnitude.  The check passes when |ratio − 1| ≤ 0.5.
    """
    rng       = np.random.default_rng(2)
    n         = 70
    tolerance = 0.5

    residuals = rng.normal(0, 0.25, n)
    error_sq  = residuals ** 2

    # Healthy: sigma closely tracks residual magnitude (ratio ≈ 1)
    sigma_h    = np.clip(
        np.abs(residuals) * np.exp(rng.normal(0, 0.18, n)),
        1e-6, None,
    )
    sigma_sq_h = sigma_h ** 2
    ratio_h    = float(np.mean(sigma_sq_h) / (np.mean(error_sq) + 1e-12))

    # Unhealthy: sigma inflated 2× — model is far too uncertain (ratio ≈ 4)
    sigma_sq_u = sigma_sq_h * 4.0
    ratio_u    = float(np.mean(sigma_sq_u) / (np.mean(error_sq) + 1e-12))

    fig, (ax_h, ax_u) = plt.subplots(1, 2, figsize=(7, 3.5))

    for ax, sigma_sq, ratio, title in (
        (ax_h, sigma_sq_h, ratio_h, "Healthy"),
        (ax_u, sigma_sq_u, ratio_u, "Unhealthy — inflated uncertainty"),
    ):
        passed = abs(ratio - 1.0) <= tolerance
        c      = PASS_COLOR if passed else FAIL_COLOR

        labels = ["Mean (y−ŷ)²\n(actual error var.)", "Mean σ²\n(predicted var.)"]
        means  = [np.mean(error_sq), np.mean(sigma_sq)]
        sems   = [np.std(error_sq) / np.sqrt(n), np.std(sigma_sq) / np.sqrt(n)]
        ax.bar([0, 1], means, yerr=sems, color=[GRAY, c],
               capsize=5, width=0.5, error_kw={"lw": 1.2})
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylabel("Variance")
        # Ideal-ratio annotation line at mean(error_sq) height on the sigma bar
        ideal_y = np.mean(error_sq)
        ax.axhline(ideal_y, xmin=0.55, xmax=0.9, color=GRAY,
                   lw=1.2, ls="--", label="Ideal (ratio = 1)")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, fontsize=8)
        _ann(ax, f"Ratio = {ratio:.2f}\nideal = 1.0 ± {tolerance:.1f}",
             x=0.5, y=-0.28, ha="center", va="top")
        _badge(ax, passed, x=0.03, y=0.97, ha="left", va="top")

    fig.tight_layout()
    fig.savefig(out_dir / "variance_alignment.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 4. UncertaintyEvolutionCheck
# ─────────────────────────────────────────────────────────────────────────────

def plot_uncertainty_evolution(out_dir: Path) -> None:
    """Uncertainty over active-learning steps with linear trend overlay.

    UncertaintyEvolutionCheck fits a line to the uncertainty time-series and
    normalises the slope by the mean uncertainty.  A steep downward relative
    slope (< −0.05 / step) indicates the model is collapsing to overconfidence
    faster than the data acquisition can justify.
    """
    rng       = np.random.default_rng(3)
    n_steps   = 40
    t         = np.arange(n_steps, dtype=float)
    slope_thr = -0.05

    # Healthy: gentle, near-flat decline — normal epistemic reduction
    u_h   = 0.80 * np.exp(-0.008 * t) + rng.normal(0, 0.012, n_steps)
    u_h   = np.clip(u_h, 0.05, None)
    sl_h  = float(np.polyfit(t, u_h, 1)[0])
    rel_h = sl_h / (float(np.mean(u_h)) + 1e-12)
    fit_h = np.polyval(np.polyfit(t, u_h, 1), t)

    # Unhealthy: steep collapse — surrogate overconfidence builds rapidly
    u_u   = 0.80 * np.exp(-0.12 * t) + rng.normal(0, 0.006, n_steps)
    u_u   = np.clip(u_u, 0.005, None)
    sl_u  = float(np.polyfit(t, u_u, 1)[0])
    rel_u = sl_u / (float(np.mean(u_u)) + 1e-12)
    fit_u = np.polyval(np.polyfit(t, u_u, 1), t)

    fig, (ax_h, ax_u) = plt.subplots(1, 2, figsize=(7, 3.5))

    for ax, u, fit, rel, title in (
        (ax_h, u_h, fit_h, rel_h, "Healthy"),
        (ax_u, u_u, fit_u, rel_u, "Unhealthy — collapsing uncertainty"),
    ):
        passed = rel >= slope_thr
        c = PASS_COLOR if passed else FAIL_COLOR

        ax.plot(t, u,   color=c,    lw=1.2, alpha=0.85, label="Uncertainty")
        ax.plot(t, fit, color=GRAY, lw=1.0, ls="--",    label="Linear trend")
        ax.set_xlabel("AL step")
        ax.set_ylabel("Mean predictive uncertainty")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8)
        _ann(ax,
             f"slope = {rel:+.4f}/step\n"
             f"thr   ≥ {slope_thr:+.2f}/step",
             x=0.5, y=-0.28, ha="center", va="top")
        _badge(ax, passed)

    fig.tight_layout()
    fig.savefig(out_dir / "uncertainty_evolution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 5. UncertaintyAnomalyCheck
# ─────────────────────────────────────────────────────────────────────────────

def plot_uncertainty_anomaly(out_dir: Path) -> None:
    """Time series with ±3σ anomaly bounds; flagged steps highlighted.

    UncertaintyAnomalyCheck z-scores the uncertainty series and flags steps
    where |z| > 3.  The check passes when fewer than 5 % of steps are flagged.
    Spikes can indicate numerical instability, data leakage, or sudden
    distribution shifts.
    """
    rng      = np.random.default_rng(4)
    n_steps  = 60
    t        = np.arange(n_steps)
    z_thr    = 3.0
    max_frac = 0.05

    base = 0.50 + 0.04 * np.sin(t * 0.25)

    def _flag(u: np.ndarray):
        std = np.std(u)
        z   = (u - np.mean(u)) / (std + 1e-12)
        return np.abs(z) > z_thr

    fig, (ax_h, ax_u) = plt.subplots(1, 2, figsize=(7, 3.5))

    spike_positions = np.array([10, 22, 36, 50])  # 4 / 60 = 6.7 % → FAIL

    for ax, add_spikes, title in (
        (ax_h, False, "Healthy"),
        (ax_u, True,  "Unhealthy — anomalous spikes"),
    ):
        u = base + rng.normal(0, 0.03, n_steps)
        if add_spikes:
            u[spike_positions] += 1.0      # large positive spikes

        mask   = _flag(u)
        frac   = float(mask.mean())
        passed = frac <= max_frac
        c      = PASS_COLOR if passed else FAIL_COLOR

        mu_u   = np.mean(u)
        sd_u   = np.std(u)
        upper  = mu_u + z_thr * sd_u
        lower  = mu_u - z_thr * sd_u

        ax.fill_between(t, lower, upper, alpha=0.25, color=GRAY,
                        label=f"±{z_thr:.0f}σ band")
        ax.axhline(upper, color=GRAY, lw=1.2, ls="--")
        ax.axhline(lower, color=GRAY, lw=1.2, ls="--")
        ax.plot(t, u, color=c, lw=1.1, alpha=0.85, label="Uncertainty")
        if mask.any():
            ax.scatter(t[mask], u[mask], color=FAIL_COLOR, s=45, zorder=5,
                       label=f"Anomalies ({mask.sum()})")

        ax.set_xlabel("Step")
        ax.set_ylabel("Predictive uncertainty")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8)
        _ann(ax,
             f"Anomaly fraction = {frac:.1%}\n"
             f"Threshold ≤ {max_frac:.0%}  (|z| > {z_thr:.0f})",
             x=0.5, y=-0.28, ha="center", va="top")
        _badge(ax, passed)

    fig.tight_layout()
    fig.savefig(out_dir / "uncertainty_anomaly.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 6. VarianceErrorCorrelationCheck
# ─────────────────────────────────────────────────────────────────────────────

def plot_variance_error_correlation(out_dir: Path) -> None:
    """Scatter of predicted σ vs |error|; Spearman ρ measures alignment.

    VarianceErrorCorrelationCheck computes the Spearman rank correlation
    between the model's predicted standard deviation and the absolute
    prediction error.  A well-calibrated model should be uncertain precisely
    where it makes large errors (ρ > 0).  Negative ρ means the model is most
    confident where it is most wrong.
    """
    rng      = np.random.default_rng(5)
    n        = 80
    min_corr = 0.0

    base  = rng.uniform(0.05, 1.0, n)
    sigma = np.clip(base + rng.normal(0, 0.05, n), 0.01, None)

    # Healthy: |error| grows with sigma — model is cautious where it struggles
    error_h = np.abs(base + rng.normal(0, 0.10, n))
    rho_h, _ = sp_stats.spearmanr(sigma, error_h)

    # Unhealthy: |error| shrinks with sigma — confident exactly where wrong
    error_u = np.abs((1.05 - base) + rng.normal(0, 0.10, n))
    rho_u, _ = sp_stats.spearmanr(sigma, error_u)

    fig, (ax_h, ax_u) = plt.subplots(1, 2, figsize=(7, 3.5))

    for ax, error, rho, title in (
        (ax_h, error_h, rho_h, "Healthy"),
        (ax_u, error_u, rho_u, "Unhealthy — confident where wrong"),
    ):
        passed = rho >= min_corr
        c = PASS_COLOR if passed else FAIL_COLOR

        ax.scatter(sigma, error, color=c, s=20, alpha=0.6, zorder=3)

        # OLS line as a visual guide for trend direction
        m, b = np.polyfit(sigma, error, 1)
        xs   = np.array([sigma.min(), sigma.max()])
        ax.plot(xs, m * xs + b, color=GRAY, lw=1.2, ls="--",
                label="Trend (OLS)")

        ax.set_xlabel("Predicted σ")
        ax.set_ylabel("|error|  =  |y − ŷ|")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, fontsize=8)
        _ann(ax,
             f"Spearman ρ = {rho:+.3f}\n"
             f"Threshold  ≥ {min_corr:.1f}",
             x=0.5, y=-0.28, ha="center", va="top")
        _badge(ax, passed)

    fig.tight_layout()
    fig.savefig(out_dir / "variance_error_correlation.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 7. ConformalCoverageCheck
# ─────────────────────────────────────────────────────────────────────────────

def plot_conformal_coverage(out_dir: Path) -> None:
    """Nonconformity score distributions with conformal q̂ vs. expected Gaussian z.

    ConformalCoverageCheck computes s_i = |y_i − ŷ_i| / σ_i and estimates the
    conformal quantile q̂ at target coverage 90 %.  The ratio q̂ / z_{1-α/2}
    (where z_{1-α/2} ≈ 1.645 for α = 0.10) measures whether σ is already
    scaled correctly for valid coverage.  A calibrated model has q̂ ≈ z and
    ratio ≈ 1; an overconfident model (σ too small) has scores spread beyond z,
    pushing q̂ and the ratio upward.
    """
    rng             = np.random.default_rng(7)
    n               = 300
    target_coverage = 0.9
    alpha           = 1.0 - target_coverage
    max_q_ratio     = 1.5
    z_exp           = float(sp_stats.norm.ppf(1.0 - alpha / 2.0))   # ≈ 1.645

    mu     = rng.standard_normal(n)
    sigma  = np.abs(rng.standard_normal(n)) * 0.4 + 0.4
    y_true = mu + sigma * rng.standard_normal(n)

    level = min(float(np.ceil((n + 1) * (1.0 - alpha)) / n), 1.0)

    def _stats(sig):
        scores = np.abs(y_true - mu) / np.maximum(sig, 1e-12)
        q_hat  = float(np.quantile(scores, level))
        return scores, q_hat, q_hat / z_exp

    # Healthy: σ matches residual scale → scores ~ HalfNormal(1) → ratio ≈ 1
    scores_h, q_hat_h, ratio_h = _stats(sigma)
    # Unhealthy: σ 2.5× too small (overconfident) → scores spread far beyond z → ratio > 1
    scores_u, q_hat_u, ratio_u = _stats(sigma * 0.40)

    fig, (ax_h, ax_u) = plt.subplots(1, 2, figsize=(7, 3.5))

    for ax, scores, q_hat, ratio in (
        (ax_h, scores_h, q_hat_h, ratio_h),
        (ax_u, scores_u, q_hat_u, ratio_u),
    ):
        passed = ratio <= max_q_ratio
        c      = PASS_COLOR if passed else FAIL_COLOR
        x_max  = max(z_exp * 2.5, q_hat * 1.8)
        xs     = np.linspace(0, x_max, 500)

        # KDE of nonconformity scores
        kde = sp_stats.gaussian_kde(
            np.clip(scores, 0, x_max * 1.5), bw_method="scott"
        )
        ys = kde(xs)
        ax.fill_between(xs, ys, alpha=0.20, color=c, lw=0)
        ax.plot(xs, ys, color=c, lw=1.3, label="Score density")

        # Half-normal reference — what a perfectly calibrated model would give
        ax.plot(xs, sp_stats.halfnorm.pdf(xs, scale=1.0),
                color=GRAY, lw=1.0, ls=":", label="Ideal (HalfNormal, σ=1)")

        # Expected Gaussian quantile and actual conformal quantile
        ax.axvline(z_exp,  color=GRAY, lw=1.2, ls="--",
                   label=f"z = {z_exp:.2f}")
        ax.axvline(q_hat,  color=c,    lw=1.5, ls="-",
                   label=f"q̂ = {q_hat:.2f}")

        # Shade the gap between z and q̂ when overconfident
        if q_hat > z_exp:
            ax.axvspan(z_exp, min(q_hat, x_max),
                       alpha=0.08, color=FAIL_COLOR, zorder=0)

        ax.set_xlabel("Nonconformity score  s = |y − ŷ| / σ")
        ax.set_ylabel("Density")
        ax.set_xlim(0, x_max)
        ax.set_ylim(bottom=0)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8)
        _ann(ax,
             f"q-ratio = {ratio:.2f}\n"
             f"threshold ≤ {max_q_ratio:.1f}",
             x=0.5, y=-0.28, ha="center", va="top")
        _badge(ax, passed)

    fig.tight_layout()
    fig.savefig(out_dir / "conformal_coverage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 8. LyapunovStabilityCheck
# ─────────────────────────────────────────────────────────────────────────────

def plot_lyapunov_stability(out_dir: Path) -> None:
    """Eigenvalue scatter on the complex plane (poles plot).

    LyapunovStabilityCheck computes the Jacobian of the surrogate at each
    operating point and checks whether all eigenvalues lie inside the unit
    circle (|λ_max| < 1).  The check passes when the fraction of stable
    operating points is ≥ min_stable_fraction (default 0.5).  Points outside
    the unit circle indicate regions where the surrogate's gradient dynamics
    are divergent.
    """
    rng               = np.random.default_rng(6)
    n_eigs_per_point  = 2        # 2-D surrogate → 2 eigenvalues per Jacobian
    stability_thr     = 1.0
    min_stable_frac   = 0.5

    def _make_eigs(n_stable: int, n_unstable: int):
        n = n_stable + n_unstable
        # Stable cluster: |λ| drawn from U(0.10, 0.88)
        r_s = rng.uniform(0.10, 0.88, n_stable * n_eigs_per_point)
        θ_s = rng.uniform(0, 2 * np.pi, n_stable * n_eigs_per_point)
        eigs_s = (r_s * np.exp(1j * θ_s)).reshape(n_stable, n_eigs_per_point)

        # Unstable cluster: |λ| drawn from U(1.05, 1.50)
        r_u = rng.uniform(1.05, 1.50, n_unstable * n_eigs_per_point)
        θ_u = rng.uniform(0, 2 * np.pi, n_unstable * n_eigs_per_point)
        eigs_u = (r_u * np.exp(1j * θ_u)).reshape(n_unstable, n_eigs_per_point)

        all_eigs   = np.concatenate([eigs_s, eigs_u], axis=0)   # (n, 2)
        lm         = np.abs(all_eigs).max(axis=1)                # λ_max per point
        stable_pts = lm < stability_thr
        frac       = float(stable_pts.mean())
        return all_eigs.ravel(), np.repeat(stable_pts, n_eigs_per_point), frac

    # Healthy: 17 of 20 points stable → fraction = 0.85 ≥ 0.50 → PASS
    eigs_h, stable_h, frac_h = _make_eigs(17, 3)
    # Unhealthy: 5 of 20 points stable → fraction = 0.25 < 0.50 → FAIL
    eigs_u, stable_u, frac_u = _make_eigs(5, 15)

    # Common axis limits that show both clusters
    lim = 2.1
    θ   = np.linspace(0, 2 * np.pi, 300)

    fig, (ax_h, ax_u) = plt.subplots(1, 2, figsize=(7, 3.5))

    for ax, eigs, stable, frac in (
        (ax_h, eigs_h, stable_h, frac_h),
        (ax_u, eigs_u, stable_u, frac_u),
    ):
        passed = frac >= min_stable_frac

        ax.plot(np.cos(θ), np.sin(θ),
                color=GRAY, lw=1.0, ls="--", label="|λ| = 1")
        ax.axhline(0, color=GRAY, lw=0.5, ls=":")
        ax.axvline(0, color=GRAY, lw=0.5, ls=":")
        ax.scatter(eigs[stable].real,  eigs[stable].imag,
                   color=PASS_COLOR, s=28, zorder=4, label="Stable |λ| < 1")
        ax.scatter(eigs[~stable].real, eigs[~stable].imag,
                   color=FAIL_COLOR, s=28, marker="x", zorder=5,
                   label="Unstable |λ| ≥ 1")

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Re(λ)")
        ax.set_ylabel("Im(λ)")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8)
        _ann(ax,
             f"frac stable = {frac:.2f}\n"
             f"threshold  ≥ {min_stable_frac:.1f}",
             x=0.5, y=-0.28, ha="center", va="top")
        _badge(ax, passed)

    fig.tight_layout()
    fig.savefig(out_dir / "lyapunov_stability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 9. CRPSCheck
# ─────────────────────────────────────────────────────────────────────────────

def plot_crps(out_dir: Path) -> None:
    """Histogram of per-sample CRPS with mean and calibrated reference marked.

    CRPSCheck is a proper scoring rule: lower is better.  For a calibrated
    Gaussian the expected CRPS equals mean(σ) / √π.  An overconfident model
    (σ too small) has CRPS ≈ |y − μ| which is systematically higher than the
    calibrated reference.
    """
    rng = np.random.default_rng(10)
    n = 300
    mu = rng.standard_normal(n)
    y_true = mu + rng.standard_normal(n)   # true noise σ = 1

    sigma_h = np.ones(n) * 1.0   # calibrated
    sigma_u = np.ones(n) * 0.25  # 4× underestimate → CRPS ≈ MAE
    threshold = 0.65              # calibrated ≈ 0.564, overconfident ≈ 0.798

    def _crps(sigma):
        s = np.maximum(sigma, 1e-12)
        z = (y_true - mu) / s
        return s * (2.0 * sp_stats.norm.pdf(z)
                    + z * (2.0 * sp_stats.norm.cdf(z) - 1.0)
                    - 1.0 / np.sqrt(np.pi))

    crps_h, crps_u = _crps(sigma_h), _crps(sigma_u)

    fig, (ax_h, ax_u) = plt.subplots(1, 2, figsize=(7, 3.5))

    for ax, crps, sigma in ((ax_h, crps_h, sigma_h), (ax_u, crps_u, sigma_u)):
        mean_crps = float(np.mean(crps))
        crps_ref  = float(np.mean(sigma)) / np.sqrt(np.pi)
        passed    = mean_crps <= threshold
        c = PASS_COLOR if passed else FAIL_COLOR

        ax.hist(crps, bins=30, density=True, color=c, alpha=0.55, edgecolor="none")
        ax.axvline(mean_crps, color=c,    lw=1.5, label="Mean CRPS")
        ax.axvline(crps_ref,  color=GRAY, lw=1.2, ls="--", label="Calibrated ref.")
        ax.axvline(threshold, color="k",  lw=0.9, ls=":", label="Threshold")
        ax.set_xlabel("Per-sample CRPS")
        ax.set_ylabel("Density")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7.5)
        _ann(ax,
             f"Mean CRPS = {mean_crps:.3f}\n"
             f"threshold ≤ {threshold}",
             x=0.5, y=-0.28, ha="center", va="top")
        _badge(ax, passed)

    fig.tight_layout()
    fig.savefig(out_dir / "crps.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 10. NegativeLogLikelihoodCheck
# ─────────────────────────────────────────────────────────────────────────────

def plot_nll(out_dir: Path) -> None:
    """Histogram of per-sample NLL with mean and calibrated reference marked.

    NegativeLogLikelihoodCheck penalises overconfidence more severely than
    CRPS: the log(σ) term drops and the squared z-score soars when σ is too
    small.  A calibrated Gaussian with unit residuals gives NLL ≈ 1.419.
    """
    rng = np.random.default_rng(11)
    n = 300
    mu = rng.standard_normal(n)
    y_true = mu + rng.standard_normal(n)   # true noise σ = 1

    sigma_h = np.ones(n) * 1.0   # calibrated
    sigma_u = np.ones(n) * 0.3   # 3× underestimate → large z² terms
    threshold = 3.0              # calibrated ≈ 1.42, overconfident ≈ 5.3

    def _nll(sigma):
        s = np.maximum(sigma, 1e-12)
        z = (y_true - mu) / s
        return 0.5 * np.log(2.0 * np.pi) + np.log(s) + 0.5 * z ** 2

    nll_h, nll_u = _nll(sigma_h), _nll(sigma_u)

    fig, (ax_h, ax_u) = plt.subplots(1, 2, figsize=(7, 3.5))

    for ax, nll, sigma in ((ax_h, nll_h, sigma_h), (ax_u, nll_u, sigma_u)):
        mean_nll = float(np.mean(nll))
        nll_ref  = (0.5 * np.log(2.0 * np.pi)
                    + float(np.mean(np.log(np.maximum(sigma, 1e-12)))) + 0.5)
        passed   = mean_nll <= threshold
        c = PASS_COLOR if passed else FAIL_COLOR

        clip_hi  = float(np.percentile(nll, 99))
        ax.hist(np.clip(nll, None, clip_hi), bins=30, density=True,
                color=c, alpha=0.55, edgecolor="none")
        ax.axvline(mean_nll,  color=c,    lw=1.5, label="Mean NLL")
        ax.axvline(nll_ref,   color=GRAY, lw=1.2, ls="--", label="Calibrated ref.")
        ax.axvline(threshold, color="k",  lw=0.9, ls=":", label="Threshold")
        ax.set_xlabel("Per-sample NLL")
        ax.set_ylabel("Density")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7.5)
        _ann(ax,
             f"Mean NLL = {mean_nll:.3f}\n"
             f"threshold ≤ {threshold}",
             x=0.5, y=-0.28, ha="center", va="top")
        _badge(ax, passed)

    fig.tight_layout()
    fig.savefig(out_dir / "nll.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 11. PITUniformityCheck
# ─────────────────────────────────────────────────────────────────────────────

def plot_pit_uniformity(out_dir: Path) -> None:
    """PIT histograms: uniform for calibrated model, U-shaped for overconfident.

    PITUniformityCheck computes U_i = Φ((y_i − μ_i) / σ_i) and tests for
    uniformity via a KS test.  A well-calibrated model produces a flat PIT
    histogram; an overconfident model (σ too small) concentrates values near
    0 and 1 because observations fall far from the predicted mean in units of σ.
    """
    from scipy.stats import kstest

    rng = np.random.default_rng(12)
    n = 500     # larger n for cleaner histograms
    mu = rng.standard_normal(n)
    y_true = mu + rng.standard_normal(n)   # true noise σ = 1

    sigma_h = np.ones(n) * 1.0    # calibrated
    sigma_u = np.ones(n) * 0.05   # 20× underestimate → U-shaped PIT
    alpha   = 0.05

    def _pit_stats(sigma):
        s   = np.maximum(sigma, 1e-12)
        pit = sp_stats.norm.cdf((y_true - mu) / s)
        ks, p = kstest(pit, "uniform")
        return pit, float(ks), float(p)

    pit_h, ks_h, p_h = _pit_stats(sigma_h)
    pit_u, ks_u, p_u = _pit_stats(sigma_u)

    fig, (ax_h, ax_u) = plt.subplots(1, 2, figsize=(7, 3.5))

    for ax, pit, p in ((ax_h, pit_h, p_h), (ax_u, pit_u, p_u)):
        passed = p >= alpha
        c = PASS_COLOR if passed else FAIL_COLOR

        ax.hist(pit, bins=20, density=True, color=c, alpha=0.60,
                edgecolor="white", linewidth=0.5)
        ax.axhline(1.0, color=GRAY, lw=1.2, ls="--",
                   label="Uniform density = 1")
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("PIT value  U = Φ((y − μ) / σ)")
        ax.set_ylabel("Density")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, fontsize=8)
        _ann(ax,
             f"KS p-value = {p:.4f}\n"
             f"threshold ≥ {alpha}",
             x=0.5, y=-0.28, ha="center", va="top")
        _badge(ax, passed)

    fig.tight_layout()
    fig.savefig(out_dir / "pit_uniformity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 12. IntervalScoreCheck
# ─────────────────────────────────────────────────────────────────────────────

def plot_interval_score(out_dir: Path) -> None:
    """Histogram of per-sample Winkler interval scores.

    IntervalScoreCheck penalises both unnecessary width and coverage failures.
    A calibrated model produces a compact score distribution around the expected
    reference; an overconfident model (σ too small) incurs heavy per-sample
    penalties for intervals that miss the majority of observations.
    """
    rng = np.random.default_rng(13)
    n = 300
    mu = rng.standard_normal(n)
    y_true = mu + rng.standard_normal(n)   # true noise σ = 1

    alpha_int = 0.1                                                   # 90% intervals
    z_crit    = float(sp_stats.norm.ppf(1.0 - alpha_int / 2.0))      # ≈ 1.645
    sigma_h   = np.ones(n) * 1.0   # calibrated
    sigma_u   = np.ones(n) * 0.2   # 5× underestimate → heavy miss penalty
    threshold = 6.0                # calibrated ≈ 4.1, overconfident >> 6

    def _is(sigma):
        s  = np.maximum(sigma, 1e-12)
        lo = mu - z_crit * s
        hi = mu + z_crit * s
        return (hi - lo
                + (2.0 / alpha_int) * np.maximum(lo - y_true, 0.0)
                + (2.0 / alpha_int) * np.maximum(y_true - hi,  0.0))

    is_h, is_u = _is(sigma_h), _is(sigma_u)

    fig, (ax_h, ax_u) = plt.subplots(1, 2, figsize=(7, 3.5))

    for ax, is_vals, sigma in ((ax_h, is_h, sigma_h), (ax_u, is_u, sigma_u)):
        mean_is = float(np.mean(is_vals))
        is_ref  = (2.0 * z_crit * float(np.mean(sigma))
                   + 2.0 * float(sp_stats.norm.pdf(z_crit)) / alpha_int)
        passed  = mean_is <= threshold
        c = PASS_COLOR if passed else FAIL_COLOR

        clip_hi = float(np.percentile(is_vals, 98))
        ax.hist(np.clip(is_vals, None, clip_hi), bins=30, density=True,
                color=c, alpha=0.55, edgecolor="none")
        ax.axvline(mean_is,   color=c,    lw=1.5, label="Mean IS")
        ax.axvline(is_ref,    color=GRAY, lw=1.2, ls="--", label="Calibrated ref.")
        ax.axvline(threshold, color="k",  lw=0.9, ls=":", label="Threshold")
        ax.set_xlabel("Per-sample interval score")
        ax.set_ylabel("Density")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7.5)
        _ann(ax,
             f"Mean IS = {mean_is:.2f}\n"
             f"threshold ≤ {threshold}",
             x=0.5, y=-0.28, ha="center", va="top")
        _badge(ax, passed)

    fig.tight_layout()
    fig.savefig(out_dir / "interval_score.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

_PLOT_FUNCTIONS = [
    plot_calibration_error,
    plot_interval_coverage,
    plot_variance_alignment,
    plot_conformal_coverage,
    plot_crps,
    plot_nll,
    plot_pit_uniformity,
    plot_interval_score,
    plot_uncertainty_evolution,
    plot_uncertainty_anomaly,
    plot_variance_error_correlation,
    plot_lyapunov_stability,
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate healthy/unhealthy reference figures for each built-in "
            "traits-audit AuditCheck and save them as PNG files."
        ),
    )
    parser.add_argument(
        "--out",
        default="docs/_static/built_in_metrics",
        metavar="DIR",
        help="Output directory (default: docs/_static/built_in_metrics).",
    )
    args    = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fn in _PLOT_FUNCTIONS:
        fn(out_dir)
        stem = fn.__name__.replace("plot_", "")
        print(f"  saved  {stem}.png")

    print(f"\nAll figures written to {out_dir}/")


if __name__ == "__main__":
    main()
