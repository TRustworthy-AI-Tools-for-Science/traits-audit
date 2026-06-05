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


# ---------------------------------------------------------------------------
# Metrics guide — logged as an artifact into every run so the user can open
# it from the Artifacts tab while inspecting results.
# ---------------------------------------------------------------------------

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

## UncertaintyEvolution  (relative slope · threshold −0.05 / step)

**What it measures**: Linear fit to the per-step σ time-series, normalised by
mean σ so the result is scale-independent (units: fraction / step).

| Value | Interpretation |
|---|---|
| > −0.05 PASS | Uncertainty stable or gently declining — healthy learning. |
| < −0.05 **FAIL** | Steep drop — model may be collapsing onto a small region, ignoring the rest of the space. |

**Tip**: In the Metrics tab, plot `audit/step/pool_sigma_mean` to see how the
mean pool uncertainty evolves.  A cliff-drop in the first 5 steps is a red flag.

---

## UncertaintyAnomalies  (fraction of steps with |z| > 3 · threshold 5%)

**What it measures**: Fraction of steps where per-step σ is more than 3 standard
deviations above the running mean — spikes in the uncertainty time-series.

| Value | Interpretation |
|---|---|
| 0% PASS | Stable — no sudden jumps. |
| > 5% **FAIL** | Frequent spikes — acquisition may be querying out-of-distribution points, or the surrogate occasionally fits poorly. |

**Tip**: In the Metrics tab, plot `audit/step/uncertainty` and look for isolated
large values that correspond to anomalous step indices.

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

---

## Comparing runs

In the MLflow UI, select all three runs and click **Compare** to overlay
`audit/step/uncertainty`, `audit/step/pool_sigma_mean`, and
`audit/step/abs_error` on the same axes.  The three scenarios will diverge
clearly by step 10.
"""


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------

def oracle(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Forrester et al. (2008) 1-D benchmark with heteroscedastic noise.

    f(x) = (6x−2)² sin(12x−4),  x ∈ [0, 1]

    Aleatoric noise std: σ(x) = 0.1 + 0.4x²  — right half of the domain is
    intrinsically noisier.  No homoscedastic surrogate can achieve ECE = 0,
    which gives calibration checks something real to flag.
    """
    y_clean = (6 * x - 2) ** 2 * np.sin(12 * x - 4)
    noise_std = 0.1 + 0.4 * x ** 2
    return y_clean + rng.normal(0, noise_std, x.shape)


def oracle_calibrated(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simple 1-D benchmark with homoscedastic noise, designed to allow calibration.

    f(x) = sin(2πx),  σ = 0.3  (constant)

    A degree-5 bootstrap polynomial can fit this function well after ~15–20
    observations.  The constant noise makes calibration achievable once the
    epistemic component shrinks — allowing the uncertainty audit to show a
    clear transition from under-covered (early steps, surrogate still biased)
    to correctly-calibrated (later steps, surrogate converged).
    """
    return np.sin(2 * np.pi * x) + rng.normal(0, 0.3, x.shape)


# ---------------------------------------------------------------------------
# Surrogate
# ---------------------------------------------------------------------------

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
        self._aleatoric_fn = aleatoric_fn   # callable(x) → σ_aleatoric
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


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------

def lcb(mu: np.ndarray, sigma: np.ndarray, kappa: float = 2.0) -> int:
    """Lower-confidence bound: argmin(μ − κσ)."""
    return int(np.argmin(mu - kappa * sigma))


# ---------------------------------------------------------------------------
# Scenario config
# ---------------------------------------------------------------------------

@dataclass
class ScenarioConfig:
    name: str
    n_estimators: int
    std_scale: float
    note: str                              # shown in MLflow run description
    tags: dict = field(default_factory=dict)
    oracle_noise_std: float | None = None  # None → heteroscedastic σ(x)=0.1+0.4x²
    oracle_fn: object = None               # None → Forrester oracle; callable(x, rng)→y
    aleatoric_fn: object = None            # None → bootstrap-only; callable(x)→σ_aleatoric
    acquisition: str = "lcb"              # "lcb" or "random"


_SCENARIOS = [
    ScenarioConfig(
        name="perfectly_calibrated",
        n_estimators=30,
        std_scale=0.7,
        oracle_fn=oracle_calibrated,
        aleatoric_fn=lambda x: np.full_like(x, 0.3),
        note="""\
## Scenario: perfectly_calibrated — calibration transition gold standard

**Model**: bootstrap polynomial (degree=5, n=30, std_scale=0.7) + aleatoric floor
**Oracle**: sin(2πx) + homoscedastic noise σ = 0.3

### What to expect
This scenario uses a simpler oracle (homoscedastic noise, smooth function)
paired with a properly-sized aleatoric floor to demonstrate the IDEAL
calibration trajectory: the audit starts with one or two failing checks
(due to an under-fitted surrogate and few observations), then converges to
all-PASS as data accumulate.

σ_total = √((0.7 × σ_bootstrap)² + 0.3²)

The 0.3 aleatoric floor prevents σ from collapsing to zero once the surrogate
converges, guaranteeing that the 1-σ bands continue to cover ~68 % of oracle
outputs.

- **CalibrationError** converges from borderline to well below 0.15 by step 20.
- **IntervalCoverage** starts near 60 % (surrogate still biased) and settles
  near 70–80 % once the fit improves.
- **VarianceAlignment** converges to 0.3–0.5 as the epistemic component
  shrinks and σ_total is dominated by the aleatoric floor.
- **UncertaintyEvolution** fails only at step 10 (LCB rapidly reduces
  epistemic uncertainty early) then passes once uncertainty stabilises at the
  aleatoric floor.
- **UncertaintyAnomalies** zero throughout.
- **VarianceErrorCorrelation** passes from step 20 onward.

### The calibration transition
From step 10 (5/6 or fewer PASS) to step 20+ (6/6 PASS), the check-grid
transitions from partially-red to all-green.  Compare this trajectory against
the three Forrester scenarios, which show static or deteriorating calibration,
to understand what a well-designed uncertainty model looks like in practice.

> See Artifacts → `audit/METRICS_GUIDE.md` for a full explanation of each check.
""",
        tags={
            "scenario/type": "gold_standard",
            "scenario/calibration": "transition_to_calibrated",
            "scenario/oracle": "sin(2pi*x) + N(0, 0.09)",
        },
    ),
    ScenarioConfig(
        name="well_calibrated",
        n_estimators=30,
        std_scale=1.0,
        note="""\
## Scenario: well_calibrated — baseline

**Model**: bootstrap polynomial (degree=5, n=30, std_scale=1.0)
**Oracle**: Forrester (2008) + heteroscedastic noise σ(x) = 0.1 + 0.4x²

### What to expect
- **CalibrationError** will not reach 0 — the ensemble is homoscedastic but the
  oracle is heteroscedastic, so some irreducible miscalibration is baked in.
- **IntervalCoverage** should sit near 60–75% — healthy, if a little low because
  the noisy right half of the domain is harder to cover.
- **UncertaintyEvolution** may flag a steep slope as LCB rapidly focuses on the
  global minimum around x ≈ 0.76.
- **VarianceErrorCorrelation** should pass — the ensemble correctly assigns higher
  uncertainty to unexplored regions.

### How to compare
Select all three runs → Compare → chart `audit/step/pool_sigma_mean`.
This run should show a smooth, gradual decline. The overconfident run starts low;
the underconfident run stays high much longer.

> See Artifacts → `audit/METRICS_GUIDE.md` for a full explanation of each check.
""",
        tags={"scenario/type": "baseline", "scenario/calibration": "healthy"},
    ),
    ScenarioConfig(
        name="overconfident",
        n_estimators=5,
        std_scale=1.0,
        note="""\
## Scenario: overconfident

**Model**: bootstrap polynomial (degree=5, n=5, std_scale=1.0)
**Oracle**: Forrester (2008) + heteroscedastic noise σ(x) = 0.1 + 0.4x²

### What to expect
Five bootstrap estimators systematically underestimate the ensemble spread.
Predicted σ is too small everywhere.

- **IntervalCoverage** well below 68% — intervals are too narrow.
- **CalibrationError** high — model claims more confidence than warranted.
- **VarianceAlignment** ratio < 0.5 — predicted variance << actual squared error.
- **Acquisition behaviour**: with small σ, LCB ≈ greedy (κσ ≈ 0). The model
  may still find the minimum quickly but misses uncertainty in other regions.

### Diagnosis tip
In the Metrics tab, `audit/step/pool_sigma_mean` will start low and stay low.
Compare `audit/step/abs_error` — errors are similar to the well_calibrated run
despite much smaller stated σ, confirming overconfidence is the issue.

> See Artifacts → `audit/METRICS_GUIDE.md` for a full explanation of each check.
""",
        tags={"scenario/type": "pathological", "scenario/calibration": "overconfident"},
    ),
    ScenarioConfig(
        name="underconfident",
        n_estimators=30,
        std_scale=4.0,
        note="""\
## Scenario: underconfident

**Model**: bootstrap polynomial (degree=5, n=30, std_scale=4.0)
**Oracle**: Forrester (2008) + heteroscedastic noise σ(x) = 0.1 + 0.4x²

### What to expect
Predicted σ is artificially inflated 4×, simulating a prior-heavy Bayesian model
or a poorly-tuned kernel length-scale that over-spreads uncertainty everywhere.

- **IntervalCoverage** near 100% — intervals are so wide they always contain truth.
- **VarianceAlignment** ratio >> 1.5 — predicted variance far exceeds actual errors.
- **CalibrationError** high in the opposite direction to overconfident.
- **UncertaintyEvolution** slope may be shallower — inflated σ keeps LCB
  exploratory for longer, slowing convergence.

### Diagnosis tip
In the Metrics tab, compare `audit/step/pool_sigma_mean` across runs.
This run's values will be ~4× higher throughout. The absolute error
(`audit/step/abs_error`) will be similar to the other runs, exposing the
mismatch between stated and actual uncertainty.

> See Artifacts → `audit/METRICS_GUIDE.md` for a full explanation of each check.
""",
        tags={"scenario/type": "pathological", "scenario/calibration": "underconfident"},
    ),
]


# ---------------------------------------------------------------------------
# Figure helpers — all viz functions live in _viz.py
# ---------------------------------------------------------------------------

from traits_audit._viz import (
    _fig_check_grid,
    _fig_state_heatmap,
    _fig_pareto_scenarios,
    _fig_calibration_curves_all,
    plot_convergence,
)

#: Visual style per scenario for the cross-scenario Pareto figure.
_SCENARIO_STYLE = {
    "perfectly_calibrated": {"color": "C2", "marker": "o", "label": "Perfectly calib."},
    "well_calibrated":       {"color": "C0", "marker": "s", "label": "Well calib."},
    "overconfident":         {"color": "C1", "marker": "^", "label": "Overconfident"},
    "underconfident":        {"color": "C3", "marker": "D", "label": "Underconfident"},
}


# ---------------------------------------------------------------------------
# Single scenario runner
# ---------------------------------------------------------------------------

def _run_scenario(
    config: ScenarioConfig,
    steps: int,
    check_every: int,
    seed: int,
    mlflow_uri: str,
    experiment_name: str,
) -> object:
    import mlflow
    from traits_audit import AuditHook, AuditPipeline
    from traits_audit.mlflow_logger import MLflowLogger
    from traits_audit.checks import (
        CalibrationErrorCheck,
        IntervalCoverageCheck,
        VarianceAlignmentCheck,
        UncertaintyEvolutionCheck,
        UncertaintyAnomalyCheck,
        VarianceErrorCorrelationCheck,
    )

    rng = np.random.default_rng(seed)

    pipeline = AuditPipeline(
        checks=[
            CalibrationErrorCheck(threshold=0.15),
            IntervalCoverageCheck(expected_coverage=0.683, tolerance=0.15),
            VarianceAlignmentCheck(tolerance=0.5),
            UncertaintyEvolutionCheck(slope_threshold=-0.05),
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
        rng=rng,
    )
    pool = np.linspace(0, 1, 300)

    x_obs = rng.uniform(0, 1, size=8)
    y_obs = oracle_to_use(x_obs, rng)

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=config.name):
        # Description shown in run overview
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

        # Metrics guide artifact — open from Artifacts tab
        mlflow.log_text(_METRICS_GUIDE, "audit/METRICS_GUIDE.md")

        logger = MLflowLogger()
        hook = AuditHook(pipeline, check_every=check_every, logger=logger)

        aleat_tag = "  +aleatoric" if config.aleatoric_fn is not None else ""
        print(f"  [{config.name}]  n_est={config.n_estimators}  std_scale={config.std_scale}{aleat_tag}")

        for step in range(steps):
            surrogate.fit(x_obs, y_obs)

            mu_pool, sigma_pool = surrogate.predict(pool)
            idx = lcb(mu_pool, sigma_pool)
            x_q = pool[idx]
            y_q = float(oracle_to_use(np.array([x_q]), rng)[0])
            mu_q = float(mu_pool[idx])
            std_q = float(sigma_pool[idx])

            x_obs = np.append(x_obs, x_q)
            y_obs = np.append(y_obs, y_q)

            hook.on_step(
                y_true=y_q,
                y_pred_mean=mu_q,
                y_pred_std=std_q,
                uncertainty=std_q,
                abs_error=abs(y_q - mu_q),
                acquisition_score=float(mu_q - 2.0 * std_q),  # LCB value at query
                dataset_size=float(len(x_obs)),
                pool_sigma_mean=float(sigma_pool.mean()),
                pool_sigma_max=float(sigma_pool.max()),
            )

        report = hook.on_end()

        # --- Check-grid (interactive): stages × checks --------------------------
        stage_reports: list[tuple[str, object]] = [
            (f"step {(i + 1) * check_every}", r)
            for i, r in enumerate(hook.intermediate_reports)
        ]
        stage_reports.append(("final", report))

        fig_grid = _fig_check_grid(stage_reports, config.name)
        mlflow.log_figure(fig_grid, "audit/check_grid.html")

        # Save as PNG for docs / static assets
        _grid_png_dir = Path.cwd() / "figures"
        _grid_png_dir.mkdir(exist_ok=True)
        _grid_stem = config.name[:4]  # perfectly→perf, well→well, overc→over, unde→unde
        _name_map = {
            "perf": "perfect", "well": "well", "over": "over", "unde": "under",
        }
        _grid_stem = _name_map.get(_grid_stem, _grid_stem)
        try:
            fig_grid.write_image(
                str(_grid_png_dir / f"check_grid_{_grid_stem}.png"),
                width=1040, height=max(280, len(stage_reports) * 72 + 120), scale=2,
            )
        except Exception:
            pass  # kaleido may not be available in all envs

        # --- State-vector heatmap (interactive): steps × components -------------
        fig_hmap = _fig_state_heatmap(hook.history, config.name)
        mlflow.log_figure(fig_hmap, "audit/state_heatmap.html")

        # --- Verdict tags --------------------------------------------------------
        for r in report.results:
            label = "PASS" if r.passed else "FAIL"
            val = f" ({r.value:.4f})" if r.value is not None else ""
            mlflow.set_tag(f"audit_verdict/{r.name}", f"{label}{val}")
        mlflow.set_tag("audit_verdict/overall", "PASS" if report.passed else "FAIL")

    # --- Pareto data: (ECE, MAE) per stage for the cross-scenario figure -----
    _pareto_pts: list[tuple[float, float, str]] = []
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
            _pareto_pts.append((ece, mae, stage_label))

    return report, _pareto_pts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    names = [s.name for s in _SCENARIOS]
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--steps",       type=int,  default=40,
                   help="AL iterations per scenario (default: 40)")
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
    for config in selected:
        reports[config.name], pareto_data[config.name] = _run_scenario(
            config, args.steps, args.check_every, args.seed,
            args.mlflow_uri, experiment_name,
        )

    # --- Comparison table ---------------------------------------------------
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

    # --- Cross-scenario figures -----------------------------------------------
    if len(pareto_data) >= 2:
        import matplotlib.pyplot as plt
        fig_dir = Path.cwd() / "figures"
        fig_dir.mkdir(exist_ok=True)

        # Pareto frontier (ECE vs MAE per scenario)
        fig_pareto = _fig_pareto_scenarios(pareto_data, scenario_styles=_SCENARIO_STYLE)
        pareto_png = fig_dir / "pareto_scenarios.png"
        fig_pareto.savefig(str(pareto_png), dpi=300, bbox_inches="tight")
        plt.close(fig_pareto)
        print(f"Saved cross-scenario Pareto frontier → {pareto_png}")

        # Convergence: ECE over AL stages per scenario
        fig_conv, ax_conv = plt.subplots(figsize=(3.5, 2.625))
        for sname, pts in pareto_data.items():
            style = _SCENARIO_STYLE.get(sname, {"color": "C4", "marker": "x", "label": sname})
            stage_ece = [(p[2], p[0]) for p in pts]  # (stage_label, ece)
            # Extract numeric step from labels like "step 10", "step 20", "final"
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
        ax_conv.set_ylabel("CalibrationError (ECE)")
        ax_conv.legend(frameon=False)
        ax_conv.grid(False)
        fig_conv.tight_layout()
        conv_png = fig_dir / "convergence_scenarios.png"
        fig_conv.savefig(str(conv_png), dpi=300, bbox_inches="tight")
        plt.close(fig_conv)
        print(f"Saved cross-scenario convergence → {conv_png}\n")

        # Calibration curves: 2×2 reliability-diagram grid, one panel per scenario
        calib_results = {}
        for s in selected:
            match = next(
                (r for r in reports[s.name].results if r.name == "CalibrationError"),
                None,
            )
            if match is not None:
                calib_results[s.name] = match

        if calib_results:
            fig_calib = _fig_calibration_curves_all(calib_results, _SCENARIO_STYLE)
            if fig_calib is not None:
                calib_png = fig_dir / "calibration_curves.png"
                fig_calib.savefig(str(calib_png), dpi=300, bbox_inches="tight")
                plt.close(fig_calib)
                print(f"Saved calibration curves → {calib_png}\n")

    if args.ui:
        print(f"Launching MLflow UI — open http://127.0.0.1:5000\n")
        subprocess.run(
            [sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", args.mlflow_uri],
        )


if __name__ == "__main__":
    main()
