"""traits_audit demo — self-driving-lab-demo LED color-matching.

Runs the default ``SelfDrivingLabDemo`` in simulation mode (no hardware
required) using an Ax Bayesian optimisation loop in ask-tell style so that
the GP posterior mean and std are accessible at every step.  Per-step data
is streamed to :class:`~traits_audit.AuditHook`, and after the loop a
Lyapunov stability analysis is performed on the surrogate landscape.

Install the optional dependency first::

    pip install "traits-audit[sdl]"

Entry point::

    ta-sdl-demo [OPTIONS]
    ta-sdl-demo --n-iter 30 --out-dir _results/sdl_demo --seed 7
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ── Shared audit pipeline ──────────────────────────────────────────────────────

def _make_pipeline(check_every: int, logger=None):
    from traits_audit import AuditHook, AuditPipeline
    from traits_audit.checks import (
        CalibrationErrorCheck,
        IntervalCoverageCheck,
        VarianceAlignmentCheck,
        UncertaintyEvolutionCheck,
        UncertaintyAnomalyCheck,
        VarianceErrorCorrelationCheck,
    )
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
    return AuditHook(pipeline, check_every=check_every, logger=logger)


# ── Ax GP posterior helpers ────────────────────────────────────────────────────

def _ax_predict(ax_client, params: dict, metric: str = "frechet"):
    """Return (mean, std) from the Ax GP at a parameter dict.

    Returns (nan, nan) before the GP model is fitted or on any prediction
    error (e.g. during the initial Sobol phase).
    """
    try:
        from ax.core.observation import ObservationFeatures
        mb = ax_client.generation_strategy.model
        obs = [ObservationFeatures(parameters=params)]
        f, cov = mb.predict(obs)
        mean = float(f[metric][0])
        var  = float(cov[metric][metric][0])
        return mean, float(np.sqrt(max(var, 0.0)))
    except Exception:
        return float("nan"), float("nan")


def _make_ax_scalar(ax_client, bounds: dict, metric: str = "frechet"):
    """Return f(state_3_norm) → float wrapping the Ax GP mean.

    state_3_norm: [r, g, b] ∈ [0, 1] normalised by actual channel bounds.
    """
    from ax.core.observation import ObservationFeatures
    r_max = float(bounds["R"])
    g_max = float(bounds["G"])
    b_max = float(bounds["B"])

    def f(state_3: np.ndarray) -> float:
        params = {
            "R": float(np.clip(state_3[0] * r_max, 0, r_max)),
            "G": float(np.clip(state_3[1] * g_max, 0, g_max)),
            "B": float(np.clip(state_3[2] * b_max, 0, b_max)),
        }
        try:
            mb  = ax_client.generation_strategy.model
            obs = [ObservationFeatures(parameters=params)]
            f_dict, _ = mb.predict(obs)
            return float(f_dict[metric][0])
        except Exception:
            return 0.0
    return f


# ── Observe helper ─────────────────────────────────────────────────────────────

def _observe(sdl, params: dict, metric: str = "frechet") -> float:
    """Query the simulator via evaluate() and return the scalar objective."""
    result = sdl.evaluate(params)
    return float(result[metric])


# ── Main run ───────────────────────────────────────────────────────────────────

def run(
    n_init: int = 6,
    n_iter: int = 25,
    out_dir: Path = Path("_results/sdl_demo"),
    seed: int = 0,
    check_every: int = 10,
    metric: str = "frechet",
    mlflow_uri: str | None = None,
    run_name: str = "sdl_demo",
) -> dict:
    """Run the SDL color-matching demo with uncertainty audit + Lyapunov.

    Parameters
    ----------
    n_init : int
        Sobol warm-start trials (no GP posterior available during these).
    n_iter : int
        Bayesian optimisation iterations where GP posterior is captured.
    out_dir : Path
        Root directory for audit_report.json and figures/.
    seed : int
        RNG seed for Ax and numpy.
    check_every : int
        Intermediate audit frequency (passed to AuditHook).
    metric : str
        Ax metric name used as the optimisation objective.
    """
    try:
        from self_driving_lab_demo import SelfDrivingLabDemoLight
    except ImportError:
        print("ERROR: self-driving-lab-demo is not installed.")
        print("       pip install 'traits-audit[sdl]'")
        sys.exit(1)

    try:
        from ax.service.ax_client import AxClient, ObjectiveProperties
    except ImportError:
        print("ERROR: ax-platform is not installed.")
        print("       pip install ax-platform")
        sys.exit(1)

    out_dir  = Path(out_dir)
    fig_dir  = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("SDL Demo: LED color-matching (simulation=True)")
    print(f"  n_init={n_init}  n_iter={n_iter}  seed={seed}")
    print(f"  Output: {out_dir}")
    print("=" * 60)

    _use_mlflow = mlflow_uri is not None
    if _use_mlflow:
        import mlflow as _mlflow
        from traits_audit.mlflow_logger import MLflowLogger
        _mlflow.set_tracking_uri(mlflow_uri)
        _mlflow.set_experiment("traits_audit_platforms")
        _run_ctx = _mlflow.start_run(run_name=run_name)
        _run_ctx.__enter__()
        _mlflow.log_params({
            "platform":    "self-driving-lab-demo",
            "n_init":      n_init,
            "n_iter":      n_iter,
            "seed":        seed,
            "check_every": check_every,
            "metric":      metric,
        })
        _mlflow.set_tags({
            "platform":    "SDL-Light",
            "model":       "Ax-BoTorch GP",
            "acquisition": "EI (Ax default)",
            "simulation":  "True",
        })
        _mlflow_logger = MLflowLogger()
    else:
        _mlflow_logger = None

    # SelfDrivingLabDemoLight is the concrete subclass with simulation support
    sdl = SelfDrivingLabDemoLight(autoload=True, simulation=True)
    r_max = int(sdl.bounds["R"][1])
    g_max = int(sdl.bounds["G"][1])
    b_max = int(sdl.bounds["B"][1])

    ax_client = AxClient(random_seed=seed, verbose_logging=False)
    ax_client.create_experiment(
        parameters=[
            {"name": "R", "type": "range", "bounds": [0.0, float(r_max)]},
            {"name": "G", "type": "range", "bounds": [0.0, float(g_max)]},
            {"name": "B", "type": "range", "bounds": [0.0, float(b_max)]},
        ],
        objectives={metric: ObjectiveProperties(minimize=True)},
        overwrite_existing_experiment=True,
    )

    hook = _make_pipeline(check_every, logger=_mlflow_logger)

    uncertainties: list[float] = []
    queried_norm: list[list[float]] = []   # (R/255, G/255, B/255)

    # ── Warm-start: Sobol (no GP model yet) ───────────────────────────────────
    print(f"\n[1/3] Warm-start — {n_init} Sobol trials …")
    for _ in range(n_init):
        params, trial_idx = ax_client.get_next_trial()
        y = _observe(sdl, params, metric)
        ax_client.complete_trial(trial_idx, raw_data={metric: (y, 0.0)})

    # ── BO loop: GP model available ───────────────────────────────────────────
    print(f"\n[2/3] Bayesian optimisation — {n_iter} iterations …")
    for step in range(n_iter):
        params, trial_idx = ax_client.get_next_trial()
        y = _observe(sdl, params, metric)

        mu, sigma = _ax_predict(ax_client, params, metric)
        ax_client.complete_trial(trial_idx, raw_data={metric: (y, 0.0)})

        sigma_safe = sigma if not np.isnan(sigma) else 0.0
        uncertainties.append(sigma_safe)
        queried_norm.append([
            params["R"] / r_max,
            params["G"] / g_max,
            params["B"] / b_max,
        ])

        if not np.isnan(mu):
            hook.on_step(
                y_true=y,
                y_pred_mean=mu,
                y_pred_std=sigma,
                uncertainty=sigma_safe,
                abs_error=abs(y - mu),
                acquisition_score=float(mu - 2.0 * sigma_safe),
                dataset_size=float(n_init + step + 1),
            )

    report = hook.on_end()
    print("\n" + report.summary())

    report_path = out_dir / "audit_report.json"
    with open(report_path, "w") as fh:
        json.dump(report.to_dict(), fh, indent=2, default=str)
    print(f"Saved audit report → {report_path}")

    if _use_mlflow:
        for r in report.results:
            label = "PASS" if r.passed else "FAIL"
            val   = f" ({r.value:.4f})" if r.value is not None else ""
            _mlflow.set_tag(f"audit_verdict/{r.name}", f"{label}{val}")
        _mlflow.set_tag("audit_verdict/overall", "PASS" if report.passed else "FAIL")
        _mlflow.log_artifact(str(report_path), "audit")

    # ── Lyapunov analysis ──────────────────────────────────────────────────────
    print("\n[3/3] Lyapunov stability analysis …")
    from traits_audit._viz import (
        make_gd_predictor,
        run_lyapunov_analysis,
        plot_uncertainty_evolution,
        plot_lyapunov_evolution,
        plot_audit_evolution,
        plot_pareto_frontier,
        plot_convergence,
    )

    op_states = np.array(queried_norm)         # (n_iter, 3)
    scalar_fn = _make_ax_scalar(ax_client, {"R": r_max, "G": g_max, "B": b_max}, metric)
    gd_pred   = make_gd_predictor(scalar_fn, alpha=0.05)

    def gp_std_fn(state_3: np.ndarray) -> float:
        p = {"R": float(state_3[0] * r_max),
             "G": float(state_3[1] * g_max),
             "B": float(state_3[2] * b_max)}
        _, s = _ax_predict(ax_client, p, metric)
        return s if not np.isnan(s) else 0.0

    lyap = run_lyapunov_analysis(
        predictor=gd_pred,
        op_states=op_states,
        gp_std_fn=gp_std_fn,
        model_label="Ax-GP (SDL)",
        out_dir=fig_dir,
    )

    plot_uncertainty_evolution(
        np.array(uncertainties),
        model_label="Ax-GP (SDL)",
        out_dir=fig_dir,
    )

    plot_lyapunov_evolution(
        lambda_max_seq=lyap["lambda_max"],
        uncertainties=np.array(uncertainties),
        model_label="Ax-GP (SDL)",
        out_dir=fig_dir,
    )

    plot_audit_evolution(
        pipeline=hook._pipeline,
        history=hook.history,
        model_label="Ax-GP (SDL)",
        out_dir=fig_dir,
        snapshot_every=5,
    )

    frechet_al = [h.get("y_true", float("nan")) for h in hook.history]
    sigma_al   = [h.get("uncertainty", float("nan")) for h in hook.history]
    plot_pareto_frontier(
        x_vals=np.array(sigma_al),
        y_vals=np.array(frechet_al),
        x_label="GP posterior std (Fréchet scale)",
        y_label="Fréchet distance",
        model_label="Ax-GP (SDL)",
        out_dir=fig_dir,
        minimize_x=True,
        minimize_y=True,
        color_vals=np.arange(len(frechet_al)),
        color_label="BO step",
    )

    frechet_arr = np.array([f for f in frechet_al if np.isfinite(f)])
    if len(frechet_arr) > 0:
        best_frechet = np.minimum.accumulate(frechet_arr)
        plot_convergence(
            best_vals=best_frechet,
            query_counts=np.arange(1, len(best_frechet) + 1),
            y_label="Best Fréchet distance",
            model_label="Ax-GP (SDL)",
            out_dir=fig_dir,
            maximise=False,
        )

    if _use_mlflow:
        _mlflow.log_metrics({
            "lyapunov/n_stable":        0,
            "lyapunov/lambda_max_mean": float(lyap["lambda_max"].mean()),
            "lyapunov/lambda_max_max":  float(lyap["lambda_max"].max()),
        })
        _mlflow.log_artifact(str(fig_dir / "lyapunov_stability.csv"), "lyapunov")
        _mlflow.log_artifacts(str(fig_dir), "figures")
        _run_ctx.__exit__(None, None, None)

    print(f"\nDone. All results written to {out_dir}")
    return {"report": report, "lyapunov": lyap}


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--n-init",      type=int,   default=6,
                   help="Sobol warm-start trials (default: 6)")
    p.add_argument("--n-iter",      type=int,   default=25,
                   help="BO iterations (default: 25)")
    p.add_argument("--out-dir",     type=str,   default="_results/sdl_demo")
    p.add_argument("--seed",        type=int,   default=0)
    p.add_argument("--check-every", type=int,   default=10,
                   help="Intermediate audit frequency (default: 10)")
    p.add_argument("--metric",      type=str,   default="frechet",
                   help="Ax metric name (default: frechet)")
    default_uri = "sqlite:///" + str(Path.cwd() / "traits_audit_demo.db")
    p.add_argument("--mlflow-uri",  type=str,   default=default_uri,
                   help="MLflow tracking URI (default: local SQLite DB)")
    p.add_argument("--run-name",    type=str,   default="sdl_demo",
                   help="MLflow run name (default: sdl_demo)")
    p.add_argument("--ui",          action="store_true",
                   help="Launch the MLflow UI after the run")
    return p


def main() -> None:
    args = build_parser().parse_args()
    run(
        n_init=args.n_init,
        n_iter=args.n_iter,
        out_dir=Path(args.out_dir),
        seed=args.seed,
        check_every=args.check_every,
        metric=args.metric,
        mlflow_uri=args.mlflow_uri,
        run_name=args.run_name,
    )
    if args.ui:
        print("Launching MLflow UI — open http://127.0.0.1:5000\n")
        import subprocess
        subprocess.Popen(
            [sys.executable, "-m", "mlflow", "ui",
             "--backend-store-uri", args.mlflow_uri],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )


if __name__ == "__main__":
    main()
