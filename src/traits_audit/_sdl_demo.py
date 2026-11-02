"""traits_audit demo — self-driving-lab-demo LED colour matching.

Find the (R, G, B) LED settings that minimise the Fréchet distance between
the sensor's measured spectrum and the SDL's internal target spectrum.
Fréchet distance = 0 is a perfect colour match.

Active learning loop
--------------------
1. **Latin Hypercube Sampling** — draw a small, space-filling initial dataset
   from the (R, G, B) domain using the SDL digital twin as the oracle.
2. **Train a BoTorch GP surrogate** (``SingleTaskGP``) on the initial data.
3. **Log Expected Improvement acquisition** — optimise ``LogEI`` continuously
   over the unit cube via ``optimize_acqf`` to select the next point to query.
4. **Adaptive retraining** — after each oracle query, compute the prediction
   z-score ``|y_true - mu| / sigma``.  When the rolling mean z-score exceeds
   ``ood_threshold`` the surrogate is out of distribution; retrain the GP on
   the full accumulated dataset before continuing.

The traits-audit pipeline runs alongside the loop, auditing the surrogate's
predictive uncertainty at every step.

Install the optional dependency first::

    pip install "traits-audit[sdl]"

Entry point::

    ta-sdl-demo [OPTIONS]
    ta-sdl-demo --n-iter 40 --out-dir _results/sdl_demo --seed 7
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_LYAPUNOV_ALPHA = 0.01


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_sdl_importable() -> None:
    """Work around a broken import in self_driving_lab_demo.__init__."""
    if "self_driving_lab_demo" in sys.modules:
        return
    try:
        import self_driving_lab_demo
    except ImportError as exc:
        if not (getattr(exc, "name", None) == "ax" and "optimize" in str(exc)):
            raise
        import types
        stub = types.ModuleType("self_driving_lab_demo.utils.search")
        stub.ax_bayesian_optimization = None
        stub.grid_search = None
        stub.random_search = None
        sys.modules["self_driving_lab_demo.utils.search"] = stub
        import self_driving_lab_demo  # noqa: F401


def _rgb_to_hex(r: float, g: float, b: float,
                r_max: float, g_max: float, b_max: float) -> str:
    ri = min(255, round(r / r_max * 255))
    gi = min(255, round(g / g_max * 255))
    bi = min(255, round(b / b_max * 255))
    return f"#{ri:02X}{gi:02X}{bi:02X}"


# ---------------------------------------------------------------------------
# BoTorch surrogate helpers
# ---------------------------------------------------------------------------

def _fit_gp(train_X: torch.Tensor, train_Y: torch.Tensor):
    """Fit a BoTorch ``SingleTaskGP`` and return the fitted model.

    ``train_Y`` should be the *negated* frechet distances so that BoTorch's
    maximisation objective corresponds to minimising frechet.
    """
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood

    model = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()
    return model


def _predict(model, x: np.ndarray) -> tuple[float, float]:
    """Return (mean, std) of the frechet prediction at a single point ``x``.

    Inverts the negation applied during training so the returned mean is a
    predicted frechet distance (positive, lower is better).
    """
    x_t = torch.tensor(x, dtype=torch.float64).unsqueeze(0)
    with torch.no_grad():
        post = model.posterior(x_t)
    mu    = -float(post.mean[0, 0])
    sigma =  float(post.variance[0, 0].sqrt())
    return mu, sigma


def _next_candidate(model, train_Y: torch.Tensor) -> np.ndarray:
    """Optimise ``LogExpectedImprovement`` and return the next candidate point."""
    from botorch.acquisition import LogExpectedImprovement
    from botorch.optim import optimize_acqf

    acqf = LogExpectedImprovement(model, best_f=train_Y.max())
    bounds = torch.tensor([[0., 0., 0.], [1., 1., 1.]], dtype=torch.float64)
    candidate, _ = optimize_acqf(
        acqf, bounds=bounds, q=1, num_restarts=5, raw_samples=256,
    )
    return candidate.squeeze().detach().numpy()


# ---------------------------------------------------------------------------
# Audit pipeline
# ---------------------------------------------------------------------------

def _make_pipeline(check_every: int, logger=None):
    from traits_audit import AuditHook, AuditPipeline
    from traits_audit.checks import (
        CalibrationErrorCheck,
        ConformalCoverageCheck,
        CRPSCheck,
        DMDcSpectralRadiusCheck,
        IntervalCoverageCheck,
        IntervalScoreCheck,
        LyapunovStabilityCheck,
        NegativeLogLikelihoodCheck,
        PITUniformityCheck,
        ScoreDecompositionCheck,
        SignedBiasCheck,
        TailIndexCheck,
        TypeBMassFractionCheck,
        UncertaintyAnomalyCheck,
        UncertaintyEvolutionCheck,
        VarianceAlignmentCheck,
        VarianceErrorCorrelationCheck,
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
            LyapunovStabilityCheck(
                stability_threshold=1.0,
                min_stable_fraction=0.5,
                alpha=_LYAPUNOV_ALPHA,
                window=10,
            ),
            DMDcSpectralRadiusCheck(stability_threshold=1.0),
            SignedBiasCheck(),
            TailIndexCheck(),
            ScoreDecompositionCheck(),
            TypeBMassFractionCheck(),
        ],
        verbose=False,
    )
    return AuditHook(pipeline, check_every=check_every, logger=logger)


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(
    n_init: int = 10,
    n_iter: int = 40,
    out_dir: Path = Path("_results/sdl_demo"),
    seed: int = 0,
    check_every: int = 10,
    metric: str = "frechet",
    noise_std: float = 200.0,
    ood_threshold: float = 2.5,
    ood_window: int = 5,
    mlflow_uri: str | None = None,
    run_name: str = "sdl_demo",
) -> dict:
    """Run the SDL colour-matching demo with uncertainty audit + Lyapunov.

    Parameters
    ----------
    n_init : int
        Number of Latin Hypercube points in the initial dataset.
    n_iter : int
        Active-learning iterations after initialisation.
    metric : str
        SDL metric to minimise: ``"frechet"`` (default) or ``"mae"``.
    noise_std : float
        Standard deviation of additive Gaussian observation noise added on
        top of the simulator's own (otherwise deterministic) return value,
        representing experimental/sensor noise (default 200.0 -- a few
        percent of typical near-converged Fréchet distances, negligible at
        the ~10-100k scale of an unconverged/exploratory query). Without
        this the simulator is a pure deterministic lookup, which makes a
        replication arm (repeated measurements at one point) meaningless --
        every replicate would be bit-identical.
    ood_threshold : float
        Rolling mean z-score above which the surrogate is considered
        out-of-distribution and is retrained.
    ood_window : int
        Number of recent steps used to compute the rolling z-score.
    out_dir : Path
        Root directory for audit_report.json and figures/.
    seed : int
        RNG seed.
    check_every : int
        Intermediate audit frequency (passed to AuditHook).
    """
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    try:
        _ensure_sdl_importable()
        from self_driving_lab_demo import SelfDrivingLabDemoLight
    except ImportError:
        print("ERROR: self-driving-lab-demo is not installed.")
        print("       pip install 'traits-audit[sdl]'")
        sys.exit(1)

    from scipy.stats.qmc import LatinHypercube
    from scipy.stats.qmc import scale as qmc_scale

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("SDL Demo: LED colour-matching (simulation=True)")
    print(f"  Objective : minimise {metric} distance to target spectrum")
    print(f"  Init      : LHS n={n_init}  BO n_iter={n_iter}  seed={seed}")
    print(f"  OOD retrain: rolling z-score > {ood_threshold} over {ood_window} steps")
    print(f"  Output    : {out_dir}")
    print("=" * 60)

    # ── MLflow ────────────────────────────────────────────────────────────────
    if mlflow_uri is not None:
        import mlflow as _mlflow

        from traits_audit.mlflow_logger import MLflowLogger
        _mlflow.set_tracking_uri(mlflow_uri)
        _mlflow.set_experiment("traits_audit_platforms")
        _run_ctx = _mlflow.start_run(run_name=run_name)
        _run_ctx.__enter__()
        _mlflow.log_params({
            "platform": "sdl", "n_init": n_init, "n_iter": n_iter,
            "seed": seed, "metric": metric, "noise_std": noise_std,
            "ood_threshold": ood_threshold, "ood_window": ood_window,
        })
        mlflow_logger = MLflowLogger()
    else:
        _mlflow = None
        _run_ctx = None
        mlflow_logger = None

    # ── SDL setup ─────────────────────────────────────────────────────────────
    sdl = SelfDrivingLabDemoLight(autoload=True, simulation=True)
    bounds = {k: sdl.bounds[k] for k in ("R", "G", "B")}
    r_min, r_max = float(bounds["R"][0]), float(bounds["R"][1])
    g_min, g_max = float(bounds["G"][0]), float(bounds["G"][1])
    b_min, b_max = float(bounds["B"][0]), float(bounds["B"][1])

    def evaluate(r: float, g: float, b: float) -> float:
        # Additive Gaussian observation noise on top of the simulator's own
        # deterministic return value, representing experimental/sensor
        # noise -- see noise_std's docstring. Without this the simulator is
        # a pure lookup (bit-identical on repeat calls at the same point),
        # which makes any replication-based check meaningless.
        clean = float(sdl.evaluate({"R": r, "G": g, "B": b})[metric])
        return clean + float(rng.normal(0.0, noise_std))

    def normalise(r: float, g: float, b: float) -> np.ndarray:
        return np.array([
            (r - r_min) / (r_max - r_min),
            (g - g_min) / (g_max - g_min),
            (b - b_min) / (b_max - b_min),
        ])

    def denormalise(x: np.ndarray) -> tuple[float, float, float]:
        r = float(x[0]) * (r_max - r_min) + r_min
        g = float(x[1]) * (g_max - g_min) + g_min
        b = float(x[2]) * (b_max - b_min) + b_min
        return r, g, b

    # ── Step 1: Latin Hypercube initial dataset ───────────────────────────────
    print(f"\n[1/3] Latin Hypercube Sampling — {n_init} initial points …")
    lhs_scaled = qmc_scale(
        LatinHypercube(d=3, seed=seed).random(n=n_init),
        l_bounds=[r_min, g_min, b_min],
        u_bounds=[r_max, g_max, b_max],
    )

    X_obs = np.array([normalise(r, g, b) for r, g, b in lhs_scaled])
    y_obs = np.array([evaluate(r, g, b) for r, g, b in lhs_scaled])

    best_init_idx = int(np.argmin(y_obs))
    best_r, best_g, best_b = lhs_scaled[best_init_idx]
    print(f"  Frechet range : [{y_obs.min():.3f}, {y_obs.max():.3f}]")
    print(f"  Best init pt  : R={best_r:.1f} G={best_g:.1f} B={best_b:.1f}"
          f"  → {_rgb_to_hex(best_r, best_g, best_b, r_max, g_max, b_max)}"
          f"  (frechet={y_obs[best_init_idx]:.4f})")

    # ── Step 2: Define the target ─────────────────────────────────────────────
    # The SDL's internal reference spectrum is the target (frechet = 0 at the
    # optimum).  It is not in the initial LHS dataset.
    print("\n  Target: SDL internal reference (frechet → 0 at the optimum)")

    # Held-out set for the final audit report — separate LHS draw, never queried.
    holdout_scaled = qmc_scale(
        LatinHypercube(d=3, seed=seed + 5000).random(n=12),
        l_bounds=[r_min, g_min, b_min],
        u_bounds=[r_max, g_max, b_max],
    )
    holdout_X = np.array([normalise(r, g, b) for r, g, b in holdout_scaled])
    holdout_y = np.array([evaluate(r, g, b) for r, g, b in holdout_scaled])
    print("  Held out 12 points for final audit (never queried in the loop)")

    # Keep the LHS points separate for the CIE trajectory figure.
    lhs_points_norm = X_obs.copy()   # (n_init, 3) normalised
    lhs_y           = y_obs.copy()

    # ── Step 3: Fit initial GP surrogate ─────────────────────────────────────
    # BoTorch maximises, so pass -y_obs (minimising frechet = maximising -frechet).
    train_X = torch.tensor(X_obs, dtype=torch.float64)
    train_Y = torch.tensor(-y_obs.reshape(-1, 1), dtype=torch.float64)

    print(f"\n[2/3] Fitting BoTorch SingleTaskGP on {len(X_obs)} initial points …")
    model = _fit_gp(train_X, train_Y)

    hook = _make_pipeline(check_every, logger=mlflow_logger)

    uncertainties: list[float]       = []
    queried_points: list[np.ndarray] = []
    queried_y: list[float]           = []
    recent_z_scores: list[float]     = []
    retrain_steps: list[int]         = []
    lambda_max_history: list[float]  = []

    # ── Step 4: Active learning loop ──────────────────────────────────────────
    from traits_audit.checks.lyapunov import (
        eigenvalues_and_stability,
        make_gd_predictor,
        numerical_jacobian,
    )

    print(f"\n  Active learning — {n_iter} iterations …")

    for step in range(n_iter):

        # Optimise LogEI to get the next candidate point.
        x_next = _next_candidate(model, train_Y)

        # Query the oracle.
        r_next, g_next, b_next = denormalise(x_next)
        y_true = evaluate(r_next, g_next, b_next)

        # GP prediction at the queried point (before incorporating it).
        mu, sigma = _predict(model, x_next)

        # OOD detection: track rolling prediction z-score.
        z = abs(y_true - mu) / max(sigma, 1e-6)
        recent_z_scores.append(z)

        # Update training tensors.
        x_t = torch.tensor(x_next, dtype=torch.float64).unsqueeze(0)
        y_t = torch.tensor([[-y_true]], dtype=torch.float64)
        train_X = torch.cat([train_X, x_t], dim=0)
        train_Y = torch.cat([train_Y, y_t], dim=0)

        if (len(recent_z_scores) >= ood_window
                and np.mean(recent_z_scores[-ood_window:]) > ood_threshold):
            # Surrogate is out of distribution — full hyperparameter refit.
            model = _fit_gp(train_X, train_Y)
            recent_z_scores.clear()
            retrain_steps.append(step)
            print(f"  [step {step:3d}] OOD detected — GP retrained on {len(train_X)} points")
        else:
            # Fast exact posterior update — same hyperparameters, new observation.
            model = model.condition_on_observations(X=x_t, Y=y_t)

        # Per-step Lyapunov: Jacobian of the GD map at this queried point.
        lm_step = float("nan")
        try:
            obs_so_far = [h["y_true"] for h in hook.history if "y_true" in h] + [y_true]
            f_scale = max(float(np.std(obs_so_far)) if len(obs_so_far) > 1 else 1.0, 1e-6)

            def _sfn(s: np.ndarray, _model=model, _scale=f_scale) -> float:
                m, _ = _predict(_model, s)
                return m / _scale

            gd = make_gd_predictor(_sfn, alpha=_LYAPUNOV_ALPHA)
            J = numerical_jacobian(gd, x_next)
            lm_step = eigenvalues_and_stability(J)["lambda_max"]
        except Exception:
            pass

        lambda_max_history.append(lm_step)
        uncertainties.append(sigma)
        queried_points.append(x_next.copy())
        queried_y.append(y_true)

        step_data = {
            "y_true": y_true,
            "y_pred_mean": mu,
            "y_pred_std": sigma,
            "uncertainty": sigma,
            "abs_error": abs(y_true - mu),
            "dataset_size": float(len(train_X)),
        }
        if np.isfinite(lm_step):
            step_data["lambda_max"] = lm_step
        hook.on_step(**step_data)

    # ── Best match found ──────────────────────────────────────────────────────
    all_y = (-train_Y.squeeze().numpy())  # back to frechet distances
    best_idx = int(np.argmin(all_y))
    best_x   = train_X[best_idx].numpy()
    best_r, best_g, best_b = denormalise(best_x)
    best_hex     = _rgb_to_hex(best_r, best_g, best_b, r_max, g_max, b_max)
    best_frechet = float(all_y[best_idx])

    print(f"\n  Best match: R={best_r:.1f} G={best_g:.1f} B={best_b:.1f}"
          f"  →  {best_hex}  ({metric}={best_frechet:.4f})")
    if retrain_steps:
        print(f"  GP retrained {len(retrain_steps)} time(s) at steps: {retrain_steps}")

    # ── Lyapunov analysis ─────────────────────────────────────────────────────
    print("\n[3/3] Lyapunov stability analysis + final audit …")
    from traits_audit._viz import (
        check_grid_figures,
        plot_audit_evolution,
        plot_cie_trajectory,
        plot_convergence,
        plot_lyapunov_evolution,
        plot_uncertainty_evolution,
        run_dmdc_lyapunov_analysis,
        run_lyapunov_analysis,
    )
    from traits_audit.checks.lyapunov import make_gd_predictor

    op_states = np.array(queried_points)

    f_vals  = np.array([h["y_true"] for h in hook.history if "y_true" in h])
    f_scale = max(float(np.std(f_vals)) if len(f_vals) > 1 else 1.0, 1e-6)

    def surrogate_norm(s: np.ndarray) -> float:
        mu, _ = _predict(model, s)
        return mu / f_scale

    def surrogate_std(s: np.ndarray) -> float:
        _, sigma = _predict(model, s)
        return sigma

    gd_pred = make_gd_predictor(surrogate_norm, alpha=_LYAPUNOV_ALPHA)
    lyap = run_lyapunov_analysis(
        predictor=gd_pred,
        op_states=op_states,
        gp_std_fn=surrogate_std,
        model_label="BoTorch-GP (SDL)",
        out_dir=fig_dir,
    )

    # DMDc: growing-prefix spectral-radius analysis (also saves poles / contour figures).
    aug_states = np.column_stack([op_states, np.array(uncertainties)])
    try:
        dmdc_result = run_dmdc_lyapunov_analysis(
            aug_states=aug_states,
            model_label="BoTorch-GP (SDL)",
            out_dir=fig_dir,
            n_components=min(3, aug_states.shape[1]),
            gp_std_seq=np.array(uncertainties),
        )
        rho_A = float(np.max(np.abs(dmdc_result["eigenvalues"])))
    except Exception as exc:
        print(f"  DMDc fit failed ({exc}); DMDcSpectralRadiusCheck will skip.")
        rho_A = None
        dmdc_result = None

    # TypeBMassFraction: the GP's fitted likelihood noise is a Type-B component
    # (a fixed hyperparameter, not derived from the posterior step by step).
    # After condition_on_observations the model may be a fantasy model; catch failures.
    from traits_audit.provenance import TypeBLedger
    try:
        noise_floor = float(model.likelihood.noise.sqrt())
    except Exception:
        noise_floor = float(np.std(queried_y)) * 0.05 if queried_y else 0.0
    last_sigma  = uncertainties[-1] if uncertainties else 0.0
    total_var   = last_sigma ** 2 + noise_floor ** 2

    type_b_kwargs = {}
    if total_var > 0:
        ledger = TypeBLedger(
            components={
                "noise_level": noise_floor,
                "kernel_variance": total_var - noise_floor ** 2,
            },
            type_b_keys={"noise_level"},
        )

        def variance_without_ablated(ablated_keys):
            if "noise_level" in ablated_keys:
                return max(total_var - noise_floor ** 2, 1e-12)
            return total_var

        type_b_kwargs = {"ledger": ledger, "variance_fn": variance_without_ablated}

    # Evaluate the final GP on the held-out set.
    holdout_X_t = torch.tensor(holdout_X, dtype=torch.float64)
    with torch.no_grad():
        post = model.posterior(holdout_X_t)
    holdout_mu    = -post.mean.squeeze().numpy()
    holdout_sigma =  post.variance.sqrt().squeeze().numpy()
    valid = np.isfinite(holdout_mu) & np.isfinite(holdout_sigma)

    # UncertaintyAnomalyCheck needs a baseline SEPARATE from the current
    # series -- without historical_uncertainties it falls back to z-scoring
    # the series against its own mean/std, which makes crossing 3 sigma of
    # itself structurally near-impossible regardless of seed/n_iter (the
    # point being tested also inflates the std it's measured against).
    # Same convention as ta-camd-demo/ta-pybamm-demo: first ~1/5 of the
    # campaign (or two check-windows) as the baseline.
    n_warmup = max(check_every * 2, len(uncertainties) // 5, 1)

    report = hook.on_end(
        lambda_max=np.array(lambda_max_history),
        rho_A=rho_A,
        y_true=holdout_y[valid] if valid.any() else None,
        y_pred_mean=holdout_mu[valid] if valid.any() else None,
        y_pred_std=holdout_sigma[valid] if valid.any() else None,
        historical_uncertainties=np.array(uncertainties[:n_warmup]),
        **type_b_kwargs,
    )
    print("\n" + report.summary())
    if report.metadata.get("pairing_warnings"):
        print("\n  Pairing warnings:")
        for w in report.metadata["pairing_warnings"]:
            print(f"    - {w}")

    report_path = out_dir / "audit_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2, default=str)
    print(f"Saved audit report → {report_path}")
    history_path = hook.save_history(out_dir / "history.json")
    print(f"Saved history      → {history_path}")

    # ── Figures ───────────────────────────────────────────────────────────────
    stage_reports = [
        (f"step {(i + 1) * check_every}", r)
        for i, r in enumerate(hook.intermediate_reports)
    ]
    stage_reports.append(("final", report))
    fig_grid, fig_grid_final = check_grid_figures(stage_reports, "BoTorch-GP (SDL)")

    for fname, fig in [
        ("check_grid_sdl.png", fig_grid),
        ("check_grid_sdl_final_only.png", fig_grid_final),
    ]:
        if fig is not None:
            try:
                fig.write_image(
                    str(fig_dir / fname),
                    width=fig.layout.width, height=fig.layout.height, scale=2,
                )
                print(f"  Saved {fname}")
            except Exception:
                pass

    # Pairwise Spearman correlations between check values across snapshots
    # (same figure as ta-cal-demo's metric_correlations_*.png).
    import matplotlib.pyplot as _plt_corr

    from traits_audit._viz import _fig_metric_correlations
    fig_corr = _fig_metric_correlations(hook.intermediate_reports, "BoTorch-GP (SDL)")
    if fig_corr is not None:
        try:
            fig_corr.savefig(str(fig_dir / "metric_correlations_sdl.png"), dpi=300, bbox_inches="tight")
            _plt_corr.close(fig_corr)
            print("  Saved metric_correlations_sdl.png")
        except Exception:
            pass

    if _mlflow is not None:
        for r in report.results:
            label = "PASS" if r.passed else "FAIL"
            val = f" ({r.value:.4f})" if r.value is not None else ""
            _mlflow.set_tag(f"audit_verdict/{r.name}", f"{label}{val}")
        _mlflow.set_tag("audit_verdict/overall", "PASS" if report.passed else "FAIL")
        _mlflow.log_artifact(str(report_path), "audit")

    frechet_vals = [h.get("y_true", float("nan")) for h in hook.history]

    plot_uncertainty_evolution(
        np.array(uncertainties),
        model_label="BoTorch-GP (SDL)",
        out_dir=fig_dir,
    )
    plot_audit_evolution(
        pipeline=hook._pipeline,
        history=hook.history,
        model_label="BoTorch-GP (SDL)",
        out_dir=fig_dir,
        snapshot_every=5,
    )
    plot_lyapunov_evolution(
        lambda_max_seq=np.array(lambda_max_history),
        uncertainties=np.array(uncertainties),
        model_label="BoTorch-GP (SDL)",
        out_dir=fig_dir,
    )
    plot_cie_trajectory(
        lhs_points=lhs_points_norm,
        al_points=np.array(queried_points),
        y_lhs=lhs_y,
        y_al=np.array(queried_y),
        out_dir=fig_dir,
        model_label="BoTorch-GP (SDL)",
    )

    best_curve = np.minimum.accumulate(
        [f for f in frechet_vals if np.isfinite(f)]
    )
    if len(best_curve) > 0:
        plot_convergence(
            best_vals=best_curve,
            query_counts=np.arange(1, len(best_curve) + 1),
            y_label="Best Fréchet distance",
            model_label="BoTorch-GP (SDL)",
            out_dir=fig_dir,
            maximise=False,
        )

    if _mlflow is not None:
        lm_arr = np.array([v for v in lambda_max_history if np.isfinite(v)])
        if len(lm_arr) > 0:
            _mlflow.log_metrics({
                "lyapunov/lambda_max_mean": float(lm_arr.mean()),
                "lyapunov/lambda_max_max":  float(lm_arr.max()),
                "lyapunov/n_stable":        int((lm_arr < 1.0).sum()),
            })
        if (fig_dir / "lyapunov_stability.csv").exists():
            _mlflow.log_artifact(str(fig_dir / "lyapunov_stability.csv"), "lyapunov")
        _mlflow.log_artifacts(str(fig_dir), "figures")
        _run_ctx.__exit__(None, None, None)

    print(f"\nDone. All results written to {out_dir}")
    return {
        "report": report,
        "lyapunov": lyap,
        "lambda_max_history": lambda_max_history,
        "best_hex": best_hex,
        "retrain_steps": retrain_steps,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--n-init",        type=int,   default=10,
                   help="LHS initial points (default: 10)")
    p.add_argument("--n-iter",        type=int,   default=40,
                   help="Active-learning iterations (default: 40)")
    p.add_argument("--out-dir",       type=str,   default="_results/sdl_demo")
    p.add_argument("--seed",          type=int,   default=0)
    p.add_argument("--check-every",   type=int,   default=2,
                   help="Intermediate audit frequency (default: 2)")
    p.add_argument("--metric",        type=str,   default="frechet",
                   help="Distance metric: 'frechet' or 'mae' (default: frechet)")
    p.add_argument("--noise-std",     type=float, default=200.0,
                   help="Additive observation noise std, representing "
                        "experimental/sensor noise (default: 200.0)")
    p.add_argument("--ood-threshold", type=float, default=2.5,
                   help="Rolling z-score threshold for OOD retraining (default: 2.5)")
    p.add_argument("--ood-window",    type=int,   default=5,
                   help="Steps in rolling z-score window (default: 5)")
    default_uri = "sqlite:///" + str(Path.cwd() / "traits_audit_demo.db")
    p.add_argument("--mlflow-uri",    type=str,   default=default_uri)
    p.add_argument("--run-name",      type=str,   default="sdl_demo")
    p.add_argument("--ui",            action="store_true",
                   help="Launch MLflow UI after the run")
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
        noise_std=args.noise_std,
        ood_threshold=args.ood_threshold,
        ood_window=args.ood_window,
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
