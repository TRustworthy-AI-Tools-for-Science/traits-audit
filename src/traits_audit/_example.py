"""traits_audit demo — four calibration scenarios on a 1-D benchmark.

Runs four active-learning scenarios side-by-side in one MLflow experiment so
you can compare them on the same axes in the dashboard:

  perfectly_calibrated — bootstrap + oracle noise, all checks PASS (gold standard)
  well_calibrated      — 30-estimator bootstrap, healthy baseline
  overconfident        —  5-estimator bootstrap, intervals systematically too narrow
  underconfident       — 30-estimator bootstrap with σ × 4, intervals far too wide

Oracle: Forrester et al. (2008) benchmark with heteroscedastic noise
  f(x) = (6x−2)² sin(12x−4),  σ(x) = 0.1 + 0.4x²

Invoked via the ``ta-demo`` entry point or ``python -m traits_audit._example``.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from traits_audit._viz import (
    _fig_check_grid,
    _fig_state_heatmap,
    _fig_pareto_scenarios,
    _fig_calibration_curves_all,
    _fig_metric_correlations,
    plot_convergence,
)


_METRICS_GUIDE = """\
# Uncertainty Audit — Metrics Guide

Open this alongside the **Metrics** and **Tags** tabs to interpret the run.

---

## CalibrationError  (ECE, lower is better · threshold 0.15)

**What it measures**: Expected Calibration Error (Kuleshov 2018).
For each confidence level p, compute the fraction of true values that fall
inside the predicted p-interval, then average |observed − p| across all levels.

| Value | Interpretation |
|---|---|
| ≈ 0 | Perfect calibration — 50% intervals contain ~50% of points, etc. |
| > 0.15 **FAIL** | Systematic mismatch. Check the calibration curve PNG in Artifacts. |

**Diagnose direction**: if IntervalCoverage is also low → overconfident (intervals
too narrow).  If IntervalCoverage is high → underconfident (intervals too wide).

---

## IntervalCoverage  (target 68.3% · tolerance ±15%)

**What it measures**: Fraction of observations where the truth falls inside
[μ−σ, μ+σ].  A Gaussian with correct σ gives ~68.3%.

| Value | Interpretation |
|---|---|
| < 53% **FAIL** | Overconfident — intervals too narrow, model surprises itself often. |
| 53%–83% PASS | Healthy coverage. |
| > 83% **FAIL** | Underconfident — intervals too wide, model hedges everywhere. |

---

## VarianceAlignment  (target ratio 1.0 · tolerance ±0.5)

**What it measures**: mean(predicted variance) / mean(squared error).
A ratio of 1.0 means stated uncertainty exactly explains actual errors globally.

| Value | Interpretation |
|---|---|
| < 0.5 **FAIL** | Predicted variance too small → overconfident. |
| 0.5–1.5 PASS | Variance and error are in the same ballpark. |
| > 1.5 **FAIL** | Predicted variance too large → underconfident / over-dispersed. |

---

## UncertaintyEvolution  (flagged-channel count · threshold 0)

**What it measures**: Per-channel linear trend in the mean-pool-σ time-series.
A channel is flagged when its least-squares slope is more negative than −1 % of
the channel mean per step (scale-independent).  ``value`` = number of flagged
channels; ``0 ⇒ PASS``.

| Value | Interpretation |
|---|---|
| 0 PASS | All channels non-declining — epistemic uncertainty not collapsing. |
| ≥ 1 **FAIL** | One or more channels show a steep downward trend — surrogate may be collapsing onto a small region, ignoring unexplored space. |

**Tip**: In the Metrics tab, plot `audit/step/pool_sigma_mean` to see how the
mean pool uncertainty evolves.  A cliff-drop in the first 5 steps is a red flag.

---

## UncertaintyAnomalies  (fraction of steps with |z| > 3 · threshold 5%)

**What it measures**: Fraction of steps where the mean-pool σ deviates more than
3 standard deviations from the **first-scenario baseline**.  The first scenario run
supplies the reference mean/std; every subsequent scenario is z-scored against it.
A well-calibrated scenario should cluster near the baseline; an overconfident one
will have anomalously low σ and an underconfident one anomalously high σ.

| Value | Interpretation |
|---|---|
| 0% PASS | Pool uncertainty comparable to the reference scenario. |
| > 5% **FAIL** | Uncertainty level anomalously different from reference — mismatched model variance. |

**Tip**: In the Metrics tab, plot `audit/step/uncertainty` and compare its level
across scenarios.  The first scenario shown will always PASS (no baseline yet);
subsequent runs are scored relative to it.

---

## VarianceErrorCorrelation  (Spearman ρ · threshold 0.1)

**What it measures**: Whether high-σ predictions also have high absolute error
(Spearman rank correlation between σ and |y − μ|).

| Value | Interpretation |
|---|---|
| < 0.1 **FAIL** | Uncertainty provides no signal about where the model is wrong — LCB acquisition will not reliably steer toward informative points. |
| 0.1–0.3 PASS | Weak but statistically meaningful signal. |
| > 0.3 PASS | Strong — model genuinely knows where it's uncertain. |

---

## Reading the Tags tab

After each run, `audit_verdict/*` tags summarise results at a glance:

    audit_verdict/CalibrationError   →  PASS (0.1234)  or  FAIL (0.2345)
    audit_verdict/overall            →  PASS  or  FAIL

`scenario/*` tags describe what the run was designed to demonstrate.

"""


def oracle(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Forrester et al. (2008) 1-D benchmark with heteroscedastic noise."""
    y_clean = (6 * x - 2) ** 2 * np.sin(12 * x - 4)
    noise_std = 0.1 + 0.4 * x ** 2
    return y_clean + rng.normal(0, noise_std, x.shape)


def oracle_calibrated(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simple homoscedastic benchmark: f(x) = sin(2πx), σ = 0.3."""
    return np.sin(2 * np.pi * x) + rng.normal(0, 0.3, x.shape)


class BootstrapSurrogate:
    """Polynomial ridge-regression bootstrap ensemble.

    Parameters
    ----------
    degree : int
        Polynomial feature degree.
    n_estimators : int
        Bootstrap resamples — fewer → underestimates epistemic spread.
    std_scale : float
        Multiply all predicted σ by this factor.
        > 1 → underconfident (over-dispersed); < 1 → overconfident.
    """

    def __init__(
        self,
        degree: int = 5,
        n_estimators: int = 30,
        std_scale: float = 1.0,
        aleatoric_fn=None,
        rng: np.random.Generator | None = None,
    ):
        self.degree = degree
        self.n_estimators = n_estimators
        self.std_scale = std_scale
        self._aleatoric_fn = aleatoric_fn
        self._rng = rng or np.random.default_rng()
        self._coefs: list[np.ndarray] = []

    def _phi(self, x: np.ndarray) -> np.ndarray:
        return np.stack([x ** d for d in range(self.degree + 1)], axis=1)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        phi = self._phi(x)
        ridge = 1e-3 * np.eye(phi.shape[1])
        n = len(x)
        self._coefs = []
        for _ in range(self.n_estimators):
            idx = self._rng.integers(0, n, size=n)
            phi_b, y_b = phi[idx], y[idx]
            coef = np.linalg.solve(phi_b.T @ phi_b + ridge, phi_b.T @ y_b)
            self._coefs.append(coef)

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        phi = self._phi(x)
        preds = np.stack([phi @ c for c in self._coefs])
        sigma_ep = preds.std(0) * self.std_scale
        if self._aleatoric_fn is not None:
            sigma_al = self._aleatoric_fn(x)
            return preds.mean(0), np.sqrt(sigma_ep ** 2 + sigma_al ** 2)
        return preds.mean(0), sigma_ep


def lcb(mu: np.ndarray, sigma: np.ndarray, kappa: float = 2.0) -> int:
    """Lower-confidence bound: argmin(μ − κσ)."""
    return int(np.argmin(mu - kappa * sigma))


@dataclass
class ScenarioConfig:
    name: str
    n_estimators: int
    std_scale: float
    note: str = ""
    tags: dict = field(default_factory=dict)
    oracle_noise_std: float | None = None
    oracle_fn: object = None
    aleatoric_fn: object = None
    acquisition: str = "lcb"


_SCENARIOS = [
    ScenarioConfig(
        name="perfectly_calibrated",
        n_estimators=30,
        std_scale=0.7,
        oracle_fn=oracle_calibrated,
        aleatoric_fn=lambda x: np.full_like(x, 0.3),
        note="",
        tags={
            "scenario/type": "gold_standard",
            "scenario/calibration": "transition_to_calibrated",
            "scenario/oracle": "sin(2pi*x) + N(0, 0.09)",
        },
    ),
    ScenarioConfig(
        name="well_calibrated",
        n_estimators=30,
        std_scale=0.7,
        aleatoric_fn=lambda x: 0.1 + 0.4 * x ** 2,
        note="",
        tags={"scenario/type": "baseline", "scenario/calibration": "healthy"},
    ),
    ScenarioConfig(
        name="overconfident",
        n_estimators=5,
        std_scale=0.3,
        aleatoric_fn=lambda x: np.full_like(x, 0.1),
        note="",
        tags={"scenario/type": "pathological", "scenario/calibration": "overconfident"},
    ),
    ScenarioConfig(
        name="underconfident",
        n_estimators=30,
        std_scale=4.0,
        aleatoric_fn=lambda x: np.full_like(x, 1.0),
        note="",
        tags={"scenario/type": "pathological", "scenario/calibration": "underconfident"},
    ),
]

_SCENARIO_STYLE = {
    "perfectly_calibrated": {"color": "C2", "marker": "o", "label": "Perfectly calib."},
    "well_calibrated":       {"color": "C0", "marker": "s", "label": "Well calib."},
    "overconfident":         {"color": "C1", "marker": "^", "label": "Overconfident"},
    "underconfident":        {"color": "C3", "marker": "D", "label": "Underconfident"},
}


def _ensure_cal_demo_dir() -> Path:
    fig_dir = Path.cwd() / "_results/cal_demo"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def _run_scenario(
    config: ScenarioConfig,
    steps: int,
    check_every: int,
    seed: int,
    mlflow_uri: str,
    experiment_name: str,
    historical_uncertainties: list | None = None,
) -> object:
    import mlflow
    from traits_audit import AuditHook, AuditPipeline
    from traits_audit.mlflow_logger import MLflowLogger
    from traits_audit.checks import (
        CalibrationErrorCheck,
        ConformalCoverageCheck,
        CRPSCheck,
        NegativeLogLikelihoodCheck,
        PITUniformityCheck,
        IntervalScoreCheck,
        IntervalCoverageCheck,
        VarianceAlignmentCheck,
        UncertaintyEvolutionCheck,
        UncertaintyAnomalyCheck,
        VarianceErrorCorrelationCheck,
    )

    oracle_rng = np.random.default_rng(seed)
    surrogate_rng = np.random.default_rng(seed + 2**31)

    pipeline = AuditPipeline(
        checks=[
            CalibrationErrorCheck(threshold=0.15),
            ConformalCoverageCheck(target_coverage=0.9, max_q_ratio=1.5),
            CRPSCheck(),
            NegativeLogLikelihoodCheck(),
            PITUniformityCheck(),
            IntervalScoreCheck(),
            IntervalCoverageCheck(expected_coverage=0.683, tolerance=0.15),
            VarianceAlignmentCheck(tolerance=0.5),
            UncertaintyEvolutionCheck(),
            UncertaintyAnomalyCheck(z_threshold=3.0),
            VarianceErrorCorrelationCheck(min_correlation=0.1),
        ],
        verbose=False,
    )

    oracle_to_use = config.oracle_fn if config.oracle_fn is not None else oracle
    surrogate = BootstrapSurrogate(
        degree=5,
        n_estimators=config.n_estimators,
        std_scale=config.std_scale,
        aleatoric_fn=config.aleatoric_fn,
        rng=surrogate_rng,
    )
    pool = np.linspace(0, 1, 300)

    x_obs = oracle_rng.uniform(0, 1, size=8)
    y_obs = oracle_to_use(x_obs, oracle_rng)

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=config.name):
        mlflow.set_tag("mlflow.note.content", config.note)

        mlflow.set_tags({
            "model": f"bootstrap-poly (degree=5, n_est={config.n_estimators}, std_scale={config.std_scale})",
            "acquisition": "LCB (κ=2.0)",
            "oracle": (
                "sin(2πx) + N(0, 0.09)"
                if config.oracle_fn is not None
                else "Forrester (2008) + heteroscedastic noise σ(x)=0.1+0.4x²"
            ),
            **config.tags,
        })
        mlflow.log_params({
            "steps": steps,
            "seed": seed,
            "check_every": check_every,
            "degree": surrogate.degree,
            "n_estimators": config.n_estimators,
            "std_scale": config.std_scale,
            "aleatoric_floor": config.aleatoric_fn is not None,
            "warm_start_n": 8,
        })

        mlflow.log_text(_METRICS_GUIDE, "audit/METRICS_GUIDE.md")

        logger = MLflowLogger()
        hook = AuditHook(pipeline, check_every=check_every, logger=logger)

        aleat_tag = "  +aleatoric" if config.aleatoric_fn is not None else ""
        print(f"  [{config.name}]  n_est={config.n_estimators}  std_scale={config.std_scale}{aleat_tag}")

        history = []
        for step in range(steps):
            surrogate.fit(x_obs, y_obs)

            mu_pool, sigma_pool = surrogate.predict(pool)
            idx = lcb(mu_pool, sigma_pool)
            x_q = pool[idx]
            y_q = float(oracle_to_use(np.array([x_q]), oracle_rng)[0])
            mu_q = float(mu_pool[idx])
            std_q = float(sigma_pool[idx])

            history.append(np.abs(y_q - mu_q))

            x_obs = np.append(x_obs, x_q)
            y_obs = np.append(y_obs, y_q)

            hook.on_step(
                y_true=y_q,
                y_pred_mean=mu_q,
                y_pred_std=std_q,
                uncertainty=float(sigma_pool.mean()),
                abs_error=abs(y_q - mu_q),
                acquisition_score=float(mu_q - 2.0 * std_q),
                dataset_size=float(len(x_obs)),
                pool_sigma_mean=float(sigma_pool.mean()),
                pool_sigma_max=float(sigma_pool.max()),
            )

        on_end_kwargs = {}
        if historical_uncertainties:
            on_end_kwargs["historical_uncertainties"] = historical_uncertainties
        report = hook.on_end(**on_end_kwargs)

        # Calibration assessment at training locations with fresh oracle draws.
        # A uniform test grid concentrates points in unexplored regions where
        # polynomial extrapolation bias >> predicted sigma, making every scenario
        # appear overconfident regardless of std_scale.  Evaluating at x_obs avoids
        # extrapolation artefacts; fresh oracle draws avoid re-using training labels.
        x_calib = x_obs.copy()
        y_calib = oracle_to_use(x_calib, oracle_rng)
        mu_calib, sigma_calib = surrogate.predict(x_calib)
        calib_check = next(c for c in pipeline.checks if c.name == "CalibrationError")
        test_calib_result = calib_check.run(
            [], y_true=y_calib, y_pred_mean=mu_calib, y_pred_std=sigma_calib
        )

        stage_reports: list[tuple[str, object]] = [
            (f"step {(i + 1) * check_every}", r)
            for i, r in enumerate(hook.intermediate_reports)
        ]
        stage_reports.append(("final", report))

        fig_grid = _fig_check_grid(stage_reports, config.name)
        mlflow.log_figure(fig_grid, "audit/check_grid.html")

        fig_dir = _ensure_cal_demo_dir()
        stem = config.name[:4]
        name_map = {"perf": "perfect", "well": "well", "over": "over", "unde": "under"}
        stem = name_map.get(stem, stem)
        try:
            fig_grid.write_image(
                str(fig_dir / f"check_grid_{stem}.png"),
                width=fig_grid.layout.width, height=fig_grid.layout.height, scale=2,
            )
        except Exception:
            pass

        fig_hmap = _fig_state_heatmap(hook.history, config.name)
        mlflow.log_figure(fig_hmap, "audit/state_heatmap.html")

        # Generate audit check correlations figure
        if hook.intermediate_reports:
            fig_corr = _fig_metric_correlations(hook.intermediate_reports, config.name)
            if fig_corr is not None:
                try:
                    corr_png = fig_dir / f"metric_correlations_{stem}.png"
                    fig_corr.savefig(str(corr_png), dpi=300, bbox_inches="tight")
                    plt.close(fig_corr)
                except Exception:
                    pass

        for r in report.results:
            label = "PASS" if r.passed else "FAIL"
            val = f" ({r.value:.4f})" if r.value is not None else ""
            mlflow.set_tag(f"audit_verdict/{r.name}", f"{label}{val}")
        mlflow.set_tag("audit_verdict/overall", "PASS" if report.passed else "FAIL")

    pareto_pts: list[tuple[float, float, str]] = []
    for stage_label, stage_rep in stage_reports:
        ece = next(
            (r.value for r in stage_rep.results
             if r.name == "CalibrationError" and r.value is not None),
            None,
        )
        if ece is None:
            continue
        if stage_label == "final":
            n_hist = len(hook.history)
        else:
            try:
                n_hist = min(int(stage_label.split()[1]), len(hook.history))
            except (IndexError, ValueError):
                n_hist = len(hook.history)
        mae = float(np.mean([h.get("abs_error", np.nan) for h in hook.history[:n_hist]]))
        if np.isfinite(mae):
            pareto_pts.append((ece, mae, stage_label))

    x_test = np.linspace(0, 1, 500)
    if config.oracle_fn is oracle_calibrated:
        y_clean   = np.sin(2 * np.pi * x_test)
        noise_std = np.full_like(x_test, 0.3)
    else:
        y_clean   = (6 * x_test - 2) ** 2 * np.sin(12 * x_test - 4)
        noise_std = 0.1 + 0.4 * x_test ** 2
    mu_test, sigma_test = surrogate.predict(x_test)

    oracle_plot = dict(
        x_test=x_test, y_clean=y_clean, noise_std=noise_std,
        mu_test=mu_test, sigma_test=sigma_test,
    )

    if len(history) >= 2:
        plot_convergence(
            best_vals=history, 
            query_counts=list(range(1, len(history) + 1)),
            y_label="Mean absolute error (MAE)",
            model_label=config.name,
            out_dir=fig_dir,
            fig_title=f"convergence_{stem}",)

    uncertainty_series = [h["uncertainty"] for h in hook.history if "uncertainty" in h]
    return report, pareto_pts, test_calib_result, oracle_plot, uncertainty_series


def build_parser() -> argparse.ArgumentParser:
    names = [s.name for s in _SCENARIOS]
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--steps",       type=int,  default=250,
                   help="AL iterations per scenario (default: 250)")
    p.add_argument("--seed",        type=int,  default=0,
                   help="RNG seed (default: 0)")
    p.add_argument("--check-every", type=int,  default=10,
                   help="Intermediate audit frequency (default: 10)")
    p.add_argument("--scenarios",   nargs="+", default=None, choices=names,
                   help="Scenarios to run (default: all three)")
    default_uri = "sqlite:///" + str(Path.cwd() / "traits_audit_demo.db")
    p.add_argument("--mlflow-uri",  type=str,  default=default_uri)
    p.add_argument("--ui",          action="store_true",
                   help="Launch the MLflow UI after the run")
    return p


def main() -> None:
    args = build_parser().parse_args()

    import mlflow
    import mlflow.store.db.utils as _db_utils
    if args.mlflow_uri.startswith("sqlite:///"):
        db_path = Path(args.mlflow_uri[len("sqlite:///"):])
        if db_path.exists():
            try:
                import sqlalchemy
                _db_utils._upgrade_db(sqlalchemy.create_engine(args.mlflow_uri))
            except Exception:
                db_path.unlink()
                print(f"  Removed stale MLflow database → {db_path}\n")
    mlflow.set_tracking_uri(args.mlflow_uri)

    selected = (
        [s for s in _SCENARIOS if s.name in args.scenarios]
        if args.scenarios else _SCENARIOS
    )

    experiment_name = "traits_audit_demo"
    print(f"\nRunning {len(selected)} scenario(s) · {args.steps} steps each")
    print(f"Experiment : {experiment_name}")
    print(f"Tracking   : {args.mlflow_uri}\n")

    reports: dict[str, object] = {}
    pareto_data: dict[str, list] = {}
    test_calibs: dict[str, object] = {}
    oracle_data: dict[str, dict] = {}
    baseline_u: list | None = None
    for config in selected:
        (
            reports[config.name],
            pareto_data[config.name],
            test_calibs[config.name],
            oracle_data[config.name],
            unc_series,
        ) = _run_scenario(
            config, args.steps, args.check_every, args.seed,
            args.mlflow_uri, experiment_name,
            historical_uncertainties=baseline_u,
        )
        if baseline_u is None:
            baseline_u = unc_series  # first run becomes the anomaly-detection reference

    check_names = [r.name for r in next(iter(reports.values())).results]
    name_w = max(len(n) for n in check_names) + 2
    col_w = 16

    sep = "─" * name_w + "┼" + "┼".join("─" * col_w for _ in selected)
    print(f"\n{'=' * (name_w + (col_w + 1) * len(selected))}")
    print(" SCENARIO COMPARISON")
    print(f"{'=' * (name_w + (col_w + 1) * len(selected))}")
    header = " " * name_w + "│" + "│".join(f" {s.name[:col_w-1]:<{col_w-1}}" for s in selected)
    print(header)
    print(sep)
    for i, cn in enumerate(check_names):
        row = f" {cn:<{name_w-1}}│"
        for s in selected:
            r = reports[s.name].results[i]
            cell = f"{'PASS' if r.passed else 'FAIL'} {r.value:.4f}" if r.value is not None else ("PASS" if r.passed else "FAIL")
            row += f" {cell:<{col_w-1}}│"
        print(row)
    print(sep)
    overall = f" {'Overall':<{name_w-1}}│"
    for s in selected:
        v = "PASS" if reports[s.name].passed else "FAIL"
        overall += f" {v:<{col_w-1}}│"
    print(overall)
    print(f"{'=' * (name_w + (col_w + 1) * len(selected))}")

    print(f"\nDashboard tips:")
    print(f"  1. Open the '{experiment_name}' experiment in the MLflow UI.")
    print(f"  2. Select all runs → Compare → chart audit/step/pool_sigma_mean")
    print(f"     to see how uncertainty evolves differently across scenarios.")
    print(f"  3. Open any run → Description tab to read scenario context.")
    print(f"  4. Open any run → Tags tab to see audit_verdict/* at a glance.")
    print(f"  5. Open any run → Artifacts → audit/METRICS_GUIDE.md")
    print(f"     for a full explanation of every check.\n")

    if len(pareto_data) >= 2:

        fig_dir = _ensure_cal_demo_dir()

        fig_pareto = _fig_pareto_scenarios(pareto_data, scenario_styles=_SCENARIO_STYLE)
        pareto_png = fig_dir / "pareto_scenarios.png"
        fig_pareto.savefig(str(pareto_png), dpi=300, bbox_inches="tight")
        plt.close(fig_pareto)
        print(f"Saved cross-scenario Pareto frontier → {pareto_png}")

        fig_conv, ax_conv = plt.subplots(figsize=(3.5, 2.625))
        for sname, pts in pareto_data.items():
            style = _SCENARIO_STYLE.get(sname, {"color": "C4", "marker": "x", "label": sname})
            stage_ece = [(p[2], p[0]) for p in pts]
            steps, eces = [], []
            for lbl, ece in stage_ece:
                try:
                    steps.append(int(lbl.split()[-1]) if lbl != "final" else args.steps)
                except (ValueError, IndexError):
                    steps.append(args.steps)
                eces.append(ece)
            ax_conv.plot(steps, eces, color=style["color"], marker=style["marker"],
                         markersize=4, label=style["label"])
        ax_conv.set_xlabel("AL step")
        ax_conv.set_ylabel("Calibration Error (ECE)")
        ax_conv.legend(frameon=False, bbox_to_anchor=(1.05, 0.5), loc='center left')
        ax_conv.grid(False)
        ax_conv.set_box_aspect(1) 
        fig_conv.tight_layout()
        conv_png = fig_dir / "convergence_scenarios.png"
        fig_conv.savefig(str(conv_png), dpi=300, bbox_inches="tight")
        plt.close(fig_conv)
        print(f"Saved cross-scenario convergence → {conv_png}\n")

        calib_results = {
            s.name: test_calibs[s.name]
            for s in selected
            if test_calibs.get(s.name) is not None
        }

        if calib_results:
            fig_calib = _fig_calibration_curves_all(calib_results, _SCENARIO_STYLE)
            if fig_calib is not None:
                calib_png = fig_dir / "calibration_curves.png"
                fig_calib.savefig(str(calib_png), dpi=300, bbox_inches="tight")
                plt.close(fig_calib)
                print(f"Saved calibration curves → {calib_png}\n")

        _scenario_order = [
            "perfectly_calibrated", "well_calibrated", "overconfident", "underconfident"
        ]
        _panel_titles = {
            "perfectly_calibrated": "Perfectly calibrated",
            "well_calibrated":      "Well calibrated",
            "overconfident":        "Overconfident",
            "underconfident":       "Underconfident",
        }
        if oracle_data:
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch
            fig_oracle, axes = plt.subplots(2, 2, figsize=(7, 5.25))
            for ax, sname in zip(axes.flat, _scenario_order):
                if sname not in oracle_data:
                    ax.set_visible(False)
                    continue
                d = oracle_data[sname]
                ax.fill_between(
                    d["x_test"], d["y_clean"] - d["noise_std"], d["y_clean"] + d["noise_std"],
                    color="black", alpha=0.12,
                )
                ax.plot(d["x_test"], d["y_clean"], color="black", linewidth=0.8)
                ax.errorbar(d["x_test"], d["mu_test"], yerr=d["sigma_test"],
                            color="C0", alpha=0.25)
                ax.set_title(_panel_titles.get(sname, sname), fontsize=9)
                ax.set_xlabel("x", fontsize=8)
                ax.set_ylabel("y", fontsize=8)
                ax.tick_params(labelsize=7)
            legend_handles = [
                Patch(facecolor="black", alpha=0.12, label="Oracle ±1σ"),
                Line2D([0], [0], color="black", linewidth=0.8, label="Oracle f(x)"),
                Line2D([0], [0], color="C0", alpha=0.6, label="Surrogate"),
            ]
            fig_oracle.tight_layout()
            fig_oracle.legend(handles=legend_handles, loc="lower center", ncol=3,
                              frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0))
            oracle_png = fig_dir / "oracle_uncertainty_panel.png"
            fig_oracle.savefig(str(oracle_png), dpi=300, bbox_inches="tight")
            plt.close(fig_oracle)
            print(f"Saved oracle uncertainty panel → {oracle_png}\n")

    if args.ui:
        print(f"Launching MLflow UI — open http://127.0.0.1:5000\n")
        subprocess.run(
            [sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", args.mlflow_uri],
        )


if __name__ == "__main__":
    main()
