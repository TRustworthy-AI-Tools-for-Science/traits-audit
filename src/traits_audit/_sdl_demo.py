"""traits_audit demo — self-driving-lab-demo LED color-matching.

Uses an Ax ask-tell loop (``AxClient.get_next_trial`` /
``complete_trial``) — the same pattern shown in the original
`self-driving-lab-demo <https://github.com/sparks-baird/self-driving-lab-demo>`_
repository (``scripts/bayesian_optimization_basic.py``).  The GP posterior is
queried *before* each observation is incorporated, so the audit hook receives
genuine pre-observation predictive distributions.

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

# GD step size used for all Lyapunov analysis — both the post-hoc
# run_lyapunov_analysis() call (which generates fig1_poles.png) and the
# LyapunovStabilityCheck in the audit pipeline.
_LYAPUNOV_ALPHA = 0.01


# ── Shared audit pipeline ──────────────────────────────────────────────────────

def _make_pipeline(check_every: int, logger=None):
    from traits_audit import AuditHook, AuditPipeline
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
        LyapunovStabilityCheck,
    )
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
            LyapunovStabilityCheck(stability_threshold=1.0, min_stable_fraction=0.5, alpha=_LYAPUNOV_ALPHA),
        ],
        verbose=False,
    )
    return AuditHook(pipeline, check_every=check_every, logger=logger)


# ── Main run ───────────────────────────────────────────────────────────────────

def run(
    n_init: int = 6,
    n_iter: int = 250,
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
        Sobol warm-start trials.  Passed to ``choose_generation_strategy_kwargs``
        so Ax's internal Sobol budget matches this count exactly; the BoTorch GP
        is then available from the very first BO iteration.
    n_iter : int
        Bayesian optimisation iterations where GP posterior is captured.
    out_dir : Path
        Root directory for audit_report.json and figures/.
    seed : int
        RNG seed for Ax.
    check_every : int
        Intermediate audit frequency (passed to AuditHook).
    metric : str
        Ax metric name used as the optimisation objective.
    """
    import warnings
    # BoTorch retries automatically when scipy hits ABNORMAL status; suppress
    # the resulting RuntimeWarning so it doesn't flood the progress output.
    warnings.filterwarnings(
        "ignore",
        message="Optimization failed in `gen_candidates_scipy`",
        category=RuntimeWarning,
    )

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

    class PredictingAxClient(AxClient):
        """AxClient that captures GP predictions at each generated candidate.

        ``get_next_trial()`` is overridden to call
        ``get_model_predictions_for_parameterizations`` *before* returning,
        so ``last_mu`` / ``last_sigma`` always reflect the pre-observation
        predictive distribution at the most recent candidate point.
        """

        def __init__(self, tracked_metric: str, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._tracked_metric = tracked_metric
            self._last_mu: float = float("nan")
            self._last_sigma: float = float("nan")

        def get_next_trial(self, *args, **kwargs):
            params, trial_idx = super().get_next_trial(*args, **kwargs)
            self._last_mu, self._last_sigma = self.predict(params)
            return params, trial_idx

        def predict(self, params: dict) -> tuple[float, float]:
            """Return (mean, std) from the current GP at *params*.

            Returns (nan, nan) if the model is not yet a GP (Sobol phase)
            or if prediction fails for any reason.
            """
            try:
                preds = self.get_model_predictions_for_parameterizations([params])
                if preds and self._tracked_metric in preds[0]:
                    mu, sigma = preds[0][self._tracked_metric]
                    return float(mu), float(sigma)
            except Exception:
                pass
            return float("nan"), float("nan")

        @property
        def last_mu(self) -> float:
            return self._last_mu

        @property
        def last_sigma(self) -> float:
            return self._last_sigma

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

    # ── SDL setup — access bounds exactly as the original repo does ───────────
    sdl = SelfDrivingLabDemoLight(autoload=True, simulation=True)
    bounds = {k: sdl.bounds[k] for k in ("R", "G", "B")}
    r_max  = float(bounds["R"][1])
    g_max  = float(bounds["G"][1])
    b_max  = float(bounds["B"][1])
    ax_parameters = [
        {"name": nm, "type": "range", "bounds": [float(bnd[0]), float(bnd[1])]}
        for nm, bnd in bounds.items()
    ]

    # ── Ax client — explicit GenerationStrategy ───────────────────────────────
    # Using an explicit GenerationStrategy rather than choose_generation_strategy
    # allows us to set num_restarts / raw_samples on the BoTorch model step.
    # The default num_restarts=8 is too low for a 3-D frechet landscape and
    # causes scipy L-BFGS-B to hit ABNORMAL status regularly; 20 restarts
    # with 512 raw samples reduces that to near-zero.
    from ax.adapter.registry import Generators
    from ax.generation_strategy.generator_spec import GeneratorSpec
    from ax.generation_strategy.generation_node import GenerationNode
    from ax.generation_strategy.generation_strategy import GenerationStrategy
    from ax.generation_strategy.transition_criterion import MinTrials
    from ax.core.base_trial import TrialStatus

    _gen_strategy = GenerationStrategy(nodes=[
        GenerationNode(
            name="sobol",
            generator_specs=[GeneratorSpec(generator_enum=Generators.SOBOL)],
            transition_criteria=[
                MinTrials(
                    threshold=n_init,
                    transition_to="botorch",
                    only_in_statuses=[TrialStatus.COMPLETED],
                    count_only_trials_with_data=True,
                )
            ],
        ),
        GenerationNode(
            name="botorch",
            generator_specs=[
                GeneratorSpec(
                    generator_enum=Generators.BOTORCH_MODULAR,
                    generator_gen_kwargs={
                        "model_gen_options": {
                            "optimizer_kwargs": {
                                "num_restarts": 20,
                                "raw_samples": 512,
                            },
                        }
                    },
                )
            ],
        ),
    ])

    ax_client = PredictingAxClient(
        tracked_metric=metric,
        generation_strategy=_gen_strategy,
        random_seed=seed,
        verbose_logging=False,
    )
    ax_client.create_experiment(
        parameters=ax_parameters,
        objectives={metric: ObjectiveProperties(minimize=True)},
        overwrite_existing_experiment=True,
    )

    hook = _make_pipeline(check_every, logger=_mlflow_logger)

    uncertainties: list[float] = []
    queried_norm:  list[list[float]] = []

    # ── Warm-start: Sobol ─────────────────────────────────────────────────────
    print(f"\n[1/3] Warm-start — {n_init} Sobol trials …")
    for _ in range(n_init):
        params, trial_idx = ax_client.get_next_trial()
        results = sdl.evaluate({"R": params["R"], "G": params["G"], "B": params["B"]})
        # Pass raw float (no SEM=0) so Ax infers observation noise — matches
        # the original repo's evaluation_function pattern.
        ax_client.complete_trial(trial_idx, raw_data={metric: float(results[metric])})

    # ── BO loop: GP posterior queried BEFORE each observation ─────────────────
    # PredictingAxClient.get_next_trial() captures last_mu/last_sigma at the
    # candidate point before complete_trial() is called, giving the genuine
    # pre-observation predictive distribution (sigma > 0 at unseen points).
    from traits_audit._viz import (
        make_gd_predictor,
        numerical_jacobian,
        eigenvalues_and_stability,
    )

    print(f"\n[2/3] Bayesian optimisation — {n_iter} iterations …")
    for step in range(n_iter):
        params, trial_idx = ax_client.get_next_trial()

        mu, sigma = ax_client.last_mu, ax_client.last_sigma
        if np.isnan(mu):
            print(f"  [warn] GP predict returned NaN at step {step}", flush=True)

        results = sdl.evaluate({"R": params["R"], "G": params["G"], "B": params["B"]})
        y = float(results[metric])
        ax_client.complete_trial(trial_idx, raw_data={metric: y})

        sigma_safe = sigma if not np.isnan(sigma) else 0.0
        uncertainties.append(sigma_safe)
        state_norm = [params["R"] / r_max, params["G"] / g_max, params["B"] / b_max]
        queried_norm.append(state_norm)

        # Rolling lambda_max: Jacobian of the GD-predictor at this step's
        # queried point, using the current GP and observed values so far.
        _lm_step = float("nan")
        try:
            _obs = [h["y_true"] for h in hook.history if "y_true" in h] + [y]
            _f_scale = max(float(np.std(_obs)) if len(_obs) > 1 else 1.0, 1e-6)

            def _sfn(s: np.ndarray) -> float:
                p = {
                    "R": float(np.clip(s[0] * r_max, 0, r_max)),
                    "G": float(np.clip(s[1] * g_max, 0, g_max)),
                    "B": float(np.clip(s[2] * b_max, 0, b_max)),
                }
                _mu, _ = ax_client.predict(p)
                return (float(_mu) if not np.isnan(_mu) else 0.0) / _f_scale

            _gd = make_gd_predictor(_sfn, alpha=_LYAPUNOV_ALPHA)
            _J = numerical_jacobian(_gd, np.array(state_norm))
            _lm_step = eigenvalues_and_stability(_J)["lambda_max"]
        except Exception:
            pass

        if not np.isnan(mu):
            _step_kwargs = dict(
                y_true=y,
                y_pred_mean=mu,
                y_pred_std=sigma,
                uncertainty=sigma_safe,
                abs_error=abs(y - mu),
                lcb_score=float(mu - 2.0 * sigma_safe),
                dataset_size=float(n_init + step + 1),
            )
            if np.isfinite(_lm_step):
                _step_kwargs["lambda_max"] = _lm_step
            hook.on_step(**_step_kwargs)

    # ── Lyapunov analysis ──────────────────────────────────────────────────────
    print("\n[3/3] Lyapunov stability analysis + final audit …")
    from traits_audit._viz import (
        make_gd_predictor,
        run_lyapunov_analysis,
        _fig_check_grid,
        plot_uncertainty_evolution,
        plot_lyapunov_evolution,
        plot_audit_evolution,
        plot_pareto_frontier,
        plot_convergence,
    )

    # Use the final fitted GP for Lyapunov landscape analysis.
    def scalar_fn(state_3: np.ndarray) -> float:
        p = {
            "R": float(np.clip(state_3[0] * r_max, 0, r_max)),
            "G": float(np.clip(state_3[1] * g_max, 0, g_max)),
            "B": float(np.clip(state_3[2] * b_max, 0, b_max)),
        }
        mu, _ = ax_client.predict(p)
        return mu if not np.isnan(mu) else 0.0

    def gp_std_fn(state_3: np.ndarray) -> float:
        p = {
            "R": float(state_3[0] * r_max),
            "G": float(state_3[1] * g_max),
            "B": float(state_3[2] * b_max),
        }
        _, sigma = ax_client.predict(p)
        return sigma if not np.isnan(sigma) else 0.0

    op_states = np.array(queried_norm)   # (n_iter, 3)

    # Normalise scalar_fn output by the std of observed frechet values so the
    # function is O(1) in normalised [0,1]³ input space.  Without this,
    # frechet values of O(100–1000) make α·H >> 1 and all poles land far
    # outside the unit circle.
    f_vals  = np.array([h["y_true"] for h in hook.history if "y_true" in h])
    f_scale = float(np.std(f_vals)) if len(f_vals) > 1 else 1.0
    f_scale = max(f_scale, 1e-6)

    def scalar_fn_norm(state_3: np.ndarray) -> float:
        return scalar_fn(state_3) / f_scale

    gd_pred = make_gd_predictor(scalar_fn_norm, alpha=_LYAPUNOV_ALPHA)

    lyap = run_lyapunov_analysis(
        predictor=gd_pred,
        op_states=op_states,
        gp_std_fn=gp_std_fn,
        model_label="Ax-GP (SDL)",
        out_dir=fig_dir,
    )

    # Pass lambda_max so LyapunovStabilityCheck runs as part of the final report.
    report = hook.on_end(lambda_max=lyap["lambda_max"])
    print("\n" + report.summary())

    report_path = out_dir / "audit_report.json"
    with open(report_path, "w") as fh:
        json.dump(report.to_dict(), fh, indent=2, default=str)
    print(f"Saved audit report → {report_path}")

    # Audit check grid: rows = checks, cols = pipeline stages (like cal_demo)
    stage_reports: list[tuple[str, object]] = [
        (f"step {(i + 1) * check_every}", r)
        for i, r in enumerate(hook.intermediate_reports)
    ]
    stage_reports.append(("final", report))
    fig_grid = _fig_check_grid(stage_reports, "Ax-GP (SDL)")
    try:
        fig_grid.write_image(
            str(fig_dir / "check_grid_sdl.png"),
            width=fig_grid.layout.width, height=fig_grid.layout.height, scale=2,
        )
        print("  Saved check_grid_sdl.png")
    except Exception:
        pass

    if _use_mlflow:
        for r in report.results:
            label = "PASS" if r.passed else "FAIL"
            val   = f" ({r.value:.4f})" if r.value is not None else ""
            _mlflow.set_tag(f"audit_verdict/{r.name}", f"{label}{val}")
        _mlflow.set_tag("audit_verdict/overall", "PASS" if report.passed else "FAIL")
        _mlflow.log_artifact(str(report_path), "audit")

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
            "lyapunov/n_stable":        lyap["n_stable"],
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
    p.add_argument("--n-iter",      type=int,   default=250,
                   help="BO iterations (default: 250)")
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
