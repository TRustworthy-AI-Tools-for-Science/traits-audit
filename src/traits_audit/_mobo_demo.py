"""traits_audit demo — multi-objective Bayesian optimisation (BraninCurrin).

Demonstrates TRAITS-AUDIT in a two-objective setting using the BraninCurrin
synthetic test function:

- **Objective 1 (Branin)** — minimise; range ≈ [0.4, 308]
- **Objective 2 (Currin)** — minimise; range ≈ [-13.8, 5.0]
- **Inputs** — x ∈ [0, 1]²

Active learning loop
--------------------
1. **Latin Hypercube Sampling** — small space-filling initial dataset.
2. **ModelListGP** — two independent SingleTaskGP surrogates, one per objective.
3. **qNEHVI acquisition** — q-Noisy Expected Hypervolume Improvement selects
   the next point by maximising expected improvement over the current Pareto
   hypervolume.
4. **Fast update** — ``condition_on_observations`` after each query; full
   hyperparameter refit when rolling prediction error indicates OOD behaviour.

The TRAITS-AUDIT pipeline audits the Branin surrogate (objective 1) as the
primary uncertainty stream.  The key MOBO-specific output is the hypervolume
convergence curve and the discovered vs true Pareto front.

Entry point::

    ta-mobo-demo [OPTIONS]
    ta-mobo-demo --n-init 10 --n-iter 40 --out-dir _results/mobo_demo --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_LYAPUNOV_ALPHA = 0.01
_REF_POINT = torch.tensor([18.0, 6.0], dtype=torch.float64)   # hypervolume reference (minimisation)
# Observation noise std, one per objective -- same values used in BoTorch's
# own BraninCurrin MOBO tutorial. Branin/Currin are noiseless test functions,
# but qNEHVI ("Noisy" EHVI) is built for noisy observations, and fitting a
# GaussianLikelihood to genuinely noiseless data lets its inferred noise
# collapse toward zero, which is what drives the kernel matrix toward
# singularity and triggers GPyTorch's "not positive definite -- added
# jitter of 1.0e-08" warning during Cholesky. Adding this noise is the
# standard fix, not a numerical band-aid: it also makes the aleatoric vs.
# epistemic split the audit pipeline reports on genuinely meaningful,
# rather than auditing a model's noise floor against data that has none.
_NOISE_SE = torch.tensor([15.19, 0.63], dtype=torch.float64)


# ---------------------------------------------------------------------------
# Surrogate helpers
# ---------------------------------------------------------------------------

def _fit_model(train_X: torch.Tensor, train_Y_neg: torch.Tensor):
    """Fit a ModelListGP (two SingleTaskGP, one per objective).

    ``train_Y_neg`` is negated: BoTorch maximises, so we pass –Y so that
    maximising –Y is equivalent to minimising Y.
    """
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import ModelListGP, SingleTaskGP
    from gpytorch.mlls import SumMarginalLogLikelihood

    models = [
        SingleTaskGP(train_X, train_Y_neg[:, i : i + 1])
        for i in range(train_Y_neg.shape[1])
    ]
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()
    return model


def _predict(model, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) for both objectives at a single point.

    Inverts the negation applied during training so means are positive
    objective values (lower = better).

    Returns
    -------
    mean : (2,) array — predicted objective values
    std  : (2,) array — posterior standard deviations
    """
    x_t = torch.tensor(x, dtype=torch.float64).unsqueeze(0)
    with torch.no_grad():
        post = model.posterior(x_t)
    mean = -post.mean[0].numpy()       # undo negation
    std  =  post.variance[0].sqrt().numpy()
    return mean, std


def _next_candidate(model, train_X: torch.Tensor, train_Y_neg: torch.Tensor) -> np.ndarray:
    """Optimise qLogNEHVI and return the next candidate point in [0,1]²."""
    from botorch.acquisition.multi_objective.logei import (
        qLogNoisyExpectedHypervolumeImprovement,
    )
    from botorch.optim import optimize_acqf
    from botorch.sampling.normal import SobolQMCNormalSampler

    # Reference point is negated because Y is negated
    ref_neg = -_REF_POINT.to(dtype=torch.float64)

    acqf = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_neg,
        X_baseline=train_X,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])),
    )
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    # num_restarts=5 hit scipy L-BFGS-B's ABNORMAL_TERMINATION_IN_LNSRCH
    # regularly (too few restarts to reliably avoid a bad initial condition
    # on the qNEHVI surface) -- same root cause and same fix already
    # established for the SDL demo's higher-dimensional acquisition
    # landscape: more restarts/raw samples reduces it to near-zero.
    candidate, _ = optimize_acqf(
        acqf, bounds=bounds, q=1, num_restarts=20, raw_samples=512,
    )
    return candidate.squeeze().detach().numpy()


def _hypervolume(Y_obs: np.ndarray) -> float:
    """Compute hypervolume dominated by the non-dominated set of Y_obs.

    Parameters
    ----------
    Y_obs : (n, 2) array of objective values (to be minimised).
    """
    from botorch.utils.multi_objective.hypervolume import Hypervolume
    from botorch.utils.multi_objective.pareto import is_non_dominated

    Y_t = torch.tensor(Y_obs, dtype=torch.float64)
    # is_non_dominated expects maximisation — negate for minimisation
    mask = is_non_dominated(-Y_t)
    pareto_Y = -Y_t[mask]             # back to maximisation space for Hypervolume
    hv = Hypervolume(ref_point=-_REF_POINT.to(dtype=torch.float64))
    return float(hv.compute(pareto_Y))


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
            CRPSCheck(threshold=0.1),
            NegativeLogLikelihoodCheck(threshold=0.1),
            PITUniformityCheck(alpha=0.1),
            IntervalScoreCheck(threshold=25),
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
            SignedBiasCheck(se_multiplier=2.),
            TailIndexCheck(),
            ScoreDecompositionCheck(cal_threshold=0.1),
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
    out_dir: Path = Path("_results/mobo_demo"),
    seed: int = 0,
    check_every: int = 10,
    ood_threshold: float = 2.5,
    ood_window: int = 5,
    mlflow_uri: str | None = None,
    run_name: str = "mobo_demo",
) -> dict:
    """Run the MOBO demo: BraninCurrin + qNEHVI + uncertainty audit.

    Parameters
    ----------
    n_init : int
        Latin Hypercube initial dataset size.
    n_iter : int
        qNEHVI active-learning iterations.
    out_dir : Path
        Output directory for figures and JSON reports.
    seed : int
        RNG seed.
    check_every : int
        Intermediate audit frequency.
    ood_threshold : float
        Rolling z-score above which the Branin surrogate is retrained.
    ood_window : int
        Window length for the rolling z-score.
    """
    import warnings

    from linear_operator.utils.warnings import NumericalWarning
    warnings.filterwarnings("ignore", category=UserWarning)
    # NumericalWarning is a RuntimeWarning subclass, not a UserWarning, so the
    # filter above doesn't catch GPyTorch's benign-but-noisy "not positive
    # definite -- added jitter" messages during Cholesky; the real fix is the
    # observation noise added in evaluate() below (see _NOISE_SE), this is
    # just cleaning up the rare residual case even with noisy data.
    warnings.filterwarnings("ignore", category=NumericalWarning)

    from botorch.test_functions.multi_objective import BraninCurrin
    from scipy.stats.qmc import LatinHypercube

    torch.manual_seed(seed)

    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("MOBO Demo: BraninCurrin two-objective optimisation")
    print("  Objectives : Branin (min) + Currin (min)")
    print(f"  Init       : LHS n={n_init}  qNEHVI n_iter={n_iter}  seed={seed}")
    print(f"  OOD retrain: rolling z-score > {ood_threshold} over {ood_window} steps")
    print(f"  Output     : {out_dir}")
    print("=" * 60)

    # ── MLflow ─────────────────────────────────────────────────────────────
    if mlflow_uri is not None:
        import mlflow as _mlflow

        from traits_audit.mlflow_logger import MLflowLogger
        _mlflow.set_tracking_uri(mlflow_uri)
        _mlflow.set_experiment("traits_audit_platforms")
        _run_ctx = _mlflow.start_run(run_name=run_name)
        _run_ctx.__enter__()
        _mlflow.log_params({
            "platform": "mobo", "n_init": n_init, "n_iter": n_iter,
            "seed": seed, "ood_threshold": ood_threshold, "ood_window": ood_window,
        })
        mlflow_logger = MLflowLogger()
    else:
        _mlflow = None
        _run_ctx = None
        mlflow_logger = None

    # ── Problem setup ───────────────────────────────────────────────────────
    problem = BraninCurrin(negate=False)   # we handle negation ourselves

    def evaluate(x: np.ndarray) -> np.ndarray:
        """Query both objectives at a single normalised point x ∈ [0,1]², with
        additive Gaussian observation noise (see ``_NOISE_SE``)."""
        with torch.no_grad():
            y = problem(torch.tensor(x, dtype=torch.float64).unsqueeze(0))
            y = y + _NOISE_SE * torch.randn_like(y)
        return y.squeeze().numpy()   # (2,) — [branin, currin]

    # ── Step 1: Latin Hypercube initial dataset ─────────────────────────────
    print(f"\n[1/3] Latin Hypercube Sampling — {n_init} initial points …")
    X_lhs = LatinHypercube(d=2, seed=seed).random(n=n_init)   # already in [0,1]²
    Y_lhs = np.array([evaluate(x) for x in X_lhs])            # (n_init, 2)

    print(f"  Branin range : [{Y_lhs[:,0].min():.2f}, {Y_lhs[:,0].max():.2f}]")
    print(f"  Currin range : [{Y_lhs[:,1].min():.2f}, {Y_lhs[:,1].max():.2f}]")
    print(f"  Initial HV   : {_hypervolume(Y_lhs):.4f}")

    lhs_Y = Y_lhs.copy()

    # ── Step 2: Fit initial surrogates ──────────────────────────────────────
    train_X     = torch.tensor(X_lhs, dtype=torch.float64)
    train_Y     = torch.tensor(Y_lhs, dtype=torch.float64)
    train_Y_neg = -train_Y   # negate for BoTorch maximisation

    print(f"\n[2/3] Fitting ModelListGP on {n_init} initial points …")
    model = _fit_model(train_X, train_Y_neg)

    hook = _make_pipeline(check_every, logger=mlflow_logger)

    uncertainties_obj1: list[float] = []
    queried_points: list[np.ndarray] = []
    queried_Y: list[np.ndarray] = []
    hypervolume_history: list[float] = []
    recent_z_scores: list[float] = []
    retrain_steps: list[int] = []
    lambda_max_history: list[float] = []

    # ── Step 3: Active learning loop ────────────────────────────────────────
    from traits_audit.checks.lyapunov import (
        eigenvalues_and_stability,
        make_gd_predictor,
        numerical_jacobian,
    )

    print(f"\n  qNEHVI active learning — {n_iter} iterations …")

    for step in range(n_iter):

        x_next = _next_candidate(model, train_X, train_Y_neg)
        y_true = evaluate(x_next)   # (2,) — [branin, currin]

        mean, std = _predict(model, x_next)
        mu1, s1 = float(mean[0]), float(std[0])   # Branin (primary audit stream)

        # OOD detection on the Branin surrogate
        z = abs(float(y_true[0]) - mu1) / max(s1, 1e-6)
        recent_z_scores.append(z)

        # Update training tensors
        x_t     = torch.tensor(x_next, dtype=torch.float64).unsqueeze(0)
        y_t_neg = torch.tensor(-y_true, dtype=torch.float64).unsqueeze(0)
        train_X     = torch.cat([train_X, x_t], dim=0)
        train_Y     = torch.cat([train_Y, torch.tensor(y_true, dtype=torch.float64).unsqueeze(0)], dim=0)
        train_Y_neg = -train_Y

        if (len(recent_z_scores) >= ood_window
                and np.mean(recent_z_scores[-ood_window:]) > ood_threshold):
            model = _fit_model(train_X, train_Y_neg)
            recent_z_scores.clear()
            retrain_steps.append(step)
            print(f"  [step {step:3d}] OOD detected — GP retrained on {len(train_X)} points")
        else:
            # Condition each sub-model independently (ModelListGP doesn't support
            # joint condition_on_observations with multi-output Y).
            from botorch.models import ModelListGP
            new_submodels = [
                m.condition_on_observations(X=x_t, Y=y_t_neg[:, i : i + 1])
                for i, m in enumerate(model.models)
            ]
            model = ModelListGP(*new_submodels)

        # Hypervolume of all observed Y so far
        hv = _hypervolume(train_Y.numpy())
        hypervolume_history.append(hv)

        # Per-step Lyapunov (Branin surrogate)
        lm_step = float("nan")
        try:
            obs_so_far = [h["y_true"] for h in hook.history if "y_true" in h] + [float(y_true[0])]
            f_scale = max(float(np.std(obs_so_far)) if len(obs_so_far) > 1 else 1.0, 1e-6)

            def _sfn(s: np.ndarray, _model=model, _scale=f_scale) -> float:
                m, _ = _predict(_model, s)
                return float(m[0]) / _scale

            gd = make_gd_predictor(_sfn, alpha=_LYAPUNOV_ALPHA)
            J = numerical_jacobian(gd, x_next)
            lm_step = eigenvalues_and_stability(J)["lambda_max"]
        except Exception:
            pass

        lambda_max_history.append(lm_step)
        uncertainties_obj1.append(s1)
        queried_points.append(x_next.copy())
        queried_Y.append(y_true.copy())

        step_data = {
            "y_true": float(y_true[0]),
            "y_pred_mean": mu1,
            "y_pred_std": s1,
            "uncertainty": s1,
            "uncertainty_obj2": float(std[1]),
            "abs_error": abs(float(y_true[0]) - mu1),
            "dataset_size": float(len(train_X)),
            "hypervolume": hv,
        }
        if np.isfinite(lm_step):
            step_data["lambda_max"] = lm_step
        hook.on_step(**step_data)

    # ── Summary ─────────────────────────────────────────────────────────────
    from botorch.utils.multi_objective.pareto import is_non_dominated

    pareto_mask = is_non_dominated(-train_Y).numpy()
    n_pareto    = int(pareto_mask.sum())
    best_hv     = hypervolume_history[-1] if hypervolume_history else 0.0

    print(f"\n  Pareto front : {n_pareto} non-dominated points")
    print(f"  Final HV     : {best_hv:.4f}")
    if retrain_steps:
        print(f"  GP retrained {len(retrain_steps)} time(s) at steps: {retrain_steps}")

    # ── Lyapunov analysis ───────────────────────────────────────────────────
    print("\n[3/3] Lyapunov stability analysis + final audit …")
    from traits_audit._viz import (
        check_grid_figures,
        plot_audit_evolution,
        plot_convergence,
        plot_lyapunov_evolution,
        plot_pareto_frontier,
        plot_uncertainty_evolution,
        run_dmdc_lyapunov_analysis,
        run_lyapunov_analysis,
    )
    from traits_audit.checks.lyapunov import make_gd_predictor

    op_states = np.array(queried_points)   # (n_iter, 2)

    f_vals  = np.array([h["y_true"] for h in hook.history if "y_true" in h])
    f_scale = max(float(np.std(f_vals)) if len(f_vals) > 1 else 1.0, 1e-6)

    def surrogate_norm(s: np.ndarray) -> float:
        m, _ = _predict(model, s)
        return float(m[0]) / f_scale

    def surrogate_std(s: np.ndarray) -> float:
        _, sigma = _predict(model, s)
        return float(sigma[0])

    gd_pred = make_gd_predictor(surrogate_norm, alpha=_LYAPUNOV_ALPHA)
    lyap = run_lyapunov_analysis(
        predictor=gd_pred,
        op_states=op_states,
        gp_std_fn=surrogate_std,
        model_label="BoTorch-MOBO (BraninCurrin)",
        out_dir=fig_dir,
    )

    aug_states = np.column_stack([op_states, np.array(uncertainties_obj1)])
    try:
        dmdc_result = run_dmdc_lyapunov_analysis(
            aug_states=aug_states,
            model_label="BoTorch-MOBO (BraninCurrin)",
            out_dir=fig_dir,
            n_components=min(3, aug_states.shape[1]),
            gp_std_seq=np.array(uncertainties_obj1),
        )
        rho_A = float(np.max(np.abs(dmdc_result["eigenvalues"])))
    except Exception as exc:
        print(f"  DMDc fit failed ({exc}); DMDcSpectralRadiusCheck will skip.")
        rho_A = None

    # Held-out set for the final audit (separate LHS draw, never queried)
    X_ho  = LatinHypercube(d=2, seed=seed + 9999).random(n=12)
    Y_ho  = np.array([evaluate(x) for x in X_ho])
    X_ho_t = torch.tensor(X_ho, dtype=torch.float64)
    with torch.no_grad():
        post_ho = model.posterior(X_ho_t)
    ho_mu  = -post_ho.mean[:, 0].numpy()          # Branin (un-negated)
    ho_std =  post_ho.variance[:, 0].sqrt().numpy()
    valid  = np.isfinite(ho_mu) & np.isfinite(ho_std)

    # UncertaintyAnomalyCheck needs a baseline SEPARATE from the current
    # series -- without historical_uncertainties it falls back to z-scoring
    # the series against its own mean/std, which makes crossing 3 sigma of
    # itself structurally near-impossible regardless of seed/n_iter (the
    # point being tested also inflates the std it's measured against).
    # Same convention as ta-camd-demo/ta-pybamm-demo/ta-sdl-demo: first
    # ~1/5 of the campaign (or two check-windows) as the baseline.
    n_warmup = max(check_every * 2, len(uncertainties_obj1) // 5, 1)

    report = hook.on_end(
        lambda_max=np.array(lambda_max_history),
        rho_A=rho_A,
        y_true=Y_ho[valid, 0] if valid.any() else None,
        y_pred_mean=ho_mu[valid] if valid.any() else None,
        y_pred_std=ho_std[valid] if valid.any() else None,
        historical_uncertainties=np.array(uncertainties_obj1[:n_warmup]),
    )
    print("\n" + report.summary())

    # ── Save JSON outputs ───────────────────────────────────────────────────
    report_path = out_dir / "audit_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2, default=str)
    print(f"Saved audit report → {report_path}")
    history_path = hook.save_history(out_dir / "history.json")
    print(f"Saved history      → {history_path}")

    # ── Figures ─────────────────────────────────────────────────────────────
    stage_reports = [
        (f"step {(i + 1) * check_every}", r)
        for i, r in enumerate(hook.intermediate_reports)
    ]
    stage_reports.append(("final", report))
    fig_grid, fig_grid_final = check_grid_figures(stage_reports, "BoTorch-MOBO (BraninCurrin)")
    for fname, fig in [
        ("check_grid_mobo.png", fig_grid),
        ("check_grid_mobo_final_only.png", fig_grid_final),
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
    fig_corr = _fig_metric_correlations(hook.intermediate_reports, "BoTorch-MOBO (BraninCurrin)")
    if fig_corr is not None:
        try:
            fig_corr.savefig(str(fig_dir / "metric_correlations_mobo.png"), dpi=300, bbox_inches="tight")
            _plt_corr.close(fig_corr)
            print("  Saved metric_correlations_mobo.png")
        except Exception:
            pass

    plot_uncertainty_evolution(
        np.array(uncertainties_obj1),
        model_label="BoTorch-MOBO (BraninCurrin)",
        out_dir=fig_dir,
    )
    plot_audit_evolution(
        pipeline=hook._pipeline,
        history=hook.history,
        model_label="BoTorch-MOBO (BraninCurrin)",
        out_dir=fig_dir,
        snapshot_every=5,
    )
    plot_lyapunov_evolution(
        lambda_max_seq=np.array(lambda_max_history),
        uncertainties=np.array(uncertainties_obj1),
        model_label="BoTorch-MOBO (BraninCurrin)",
        out_dir=fig_dir,
    )

    # Dual-objective uncertainty evolution
    import matplotlib.pyplot as _plt

    from traits_audit._viz import _save as _viz_save
    unc_obj2 = np.array([h.get("uncertainty_obj2", float("nan")) for h in hook.history])
    steps_arr = np.arange(len(uncertainties_obj1))
    _fig, _ax = _plt.subplots(figsize=(3.5, 3.5))
    _ax.plot(steps_arr, np.array(uncertainties_obj1), label="Branin (obj 1)", color="C0")
    _ax.plot(steps_arr, unc_obj2, label="Currin (obj 2)", color="C1", ls="--")
    _ax.set_xlabel("AL step")
    _ax.set_ylabel("Posterior std")
    _ax.set_title("BoTorch-MOBO (BraninCurrin)")
    _ax.legend(frameon=False)
    _ax.grid(False)
    _fig.tight_layout()
    _viz_save(_fig, fig_dir, "fig4b_uncertainty_both_objectives")
    print("  Saved fig4b_uncertainty_both_objectives.png")

    # Approximate the true Pareto front via dense grid over [0,1]^2. Uses the
    # CLEAN (noiseless) problem directly, not evaluate() -- the reference
    # front approximation should reflect the true functions, not 2500
    # independently noisy samples of them.
    grid_n = 50
    g = np.linspace(0, 1, grid_n)
    gx, gy = np.meshgrid(g, g)
    grid_pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    with torch.no_grad():
        grid_Y = problem(torch.tensor(grid_pts, dtype=torch.float64)).numpy()
    grid_Y_t = torch.tensor(grid_Y, dtype=torch.float64)
    true_pareto_mask = is_non_dominated(-grid_Y_t).numpy()
    true_front_pts = grid_Y[true_pareto_mask]

    # ── True objective-function panel (Branin, Currin) ──────────────────────
    # Analogous to ta-cal-demo's oracle_uncertainty_panel.png: shows the actual
    # ground-truth functions being optimised, not just the Pareto front in
    # objective space plotted below. Both are defined over x in [0,1]^2, so
    # each gets a filled contour (the 2-D counterpart of ta-cal-demo's 1-D
    # oracle line) rather than a line plot.
    import matplotlib.pyplot as _plt_funcs

    from traits_audit._viz import _save as _save_funcs

    branin_grid = grid_Y[:, 0].reshape(grid_n, grid_n)
    currin_grid = grid_Y[:, 1].reshape(grid_n, grid_n)
    al_pts = np.array(queried_points) if queried_points else np.empty((0, 2))

    fig_funcs, axes_funcs = _plt_funcs.subplots(1, 2, figsize=(9, 4))
    for ax, zgrid, title in (
        (axes_funcs[0], branin_grid, "Branin"),
        (axes_funcs[1], currin_grid, "Currin"),
    ):
        cf = ax.contourf(gx, gy, zgrid, levels=30, cmap="viridis")
        fig_funcs.colorbar(cf, ax=ax, shrink=0.85, label="f(x)")
        ax.scatter(
            X_lhs[:, 0], X_lhs[:, 1], c="white", edgecolors="black",
            marker="o", s=25, linewidths=0.6, label="LHS seed", zorder=3,
        )
        if len(al_pts) > 0:
            ax.scatter(
                al_pts[:, 0], al_pts[:, 1], c=np.arange(len(al_pts)),
                cmap="autumn", marker="^", s=30, edgecolors="black",
                linewidths=0.4, label="qNEHVI-queried", zorder=4,
            )
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title(f"{title} (true surface)", fontsize=10)
    handles, labels = axes_funcs[0].get_legend_handles_labels()
    fig_funcs.legend(
        handles, labels, loc="lower center", ncol=2, frameon=False,
        fontsize=8, bbox_to_anchor=(0.5, -0.05),
    )
    fig_funcs.tight_layout(rect=(0, 0.05, 1, 1))
    _save_funcs(fig_funcs, fig_dir, "fig0_objective_functions")
    print("  Saved fig0_objective_functions.png")

    queried_Y_arr = np.array(queried_Y)
    all_Y = np.vstack([lhs_Y, queried_Y_arr])
    color_vals = np.concatenate([
        np.full(len(lhs_Y), -1.0),
        np.arange(len(queried_Y_arr), dtype=float),
    ])
    plot_pareto_frontier(
        x_vals=all_Y[:, 0],
        y_vals=all_Y[:, 1],
        x_label="Branin",
        y_label="Currin",
        model_label="BoTorch-MOBO (BraninCurrin)",
        out_dir=fig_dir,
        minimize_x=True,
        minimize_y=True,
        color_vals=color_vals,
        color_label="AL step (LHS = −1)",
        true_front_x=true_front_pts[:, 0],
        true_front_y=true_front_pts[:, 1],
    )

    if hypervolume_history:
        plot_convergence(
            best_vals=np.array(hypervolume_history),
            query_counts=np.arange(1, len(hypervolume_history) + 1),
            y_label="Hypervolume",
            model_label="BoTorch-MOBO (BraninCurrin)",
            out_dir=fig_dir,
            maximise=True,
        )

    if _mlflow is not None:
        lm_arr = np.array([v for v in lambda_max_history if np.isfinite(v)])
        if len(lm_arr) > 0:
            _mlflow.log_metrics({
                "mobo/final_hypervolume": best_hv,
                "mobo/n_pareto": n_pareto,
                "lyapunov/lambda_max_mean": float(lm_arr.mean()),
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
        "hypervolume_history": hypervolume_history,
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
    p.add_argument("--n-iter",        type=int,   default=100,
                   help="qNEHVI iterations (default: 40)")
    p.add_argument("--out-dir",       type=str,   default="_results/mobo_demo")
    p.add_argument("--seed",          type=int,   default=0)
    p.add_argument("--check-every",   type=int,   default=4,
                   help="Intermediate audit frequency (default: 4)")
    p.add_argument("--ood-threshold", type=float, default=2.5,
                   help="Rolling z-score threshold for OOD retraining (default: 2.5)")
    p.add_argument("--ood-window",    type=int,   default=5,
                   help="Steps in rolling z-score window (default: 5)")
    default_uri = "sqlite:///" + str(Path.cwd() / "traits_audit_demo.db")
    p.add_argument("--mlflow-uri",    type=str,   default=default_uri)
    p.add_argument("--run-name",      type=str,   default="mobo_demo")
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
