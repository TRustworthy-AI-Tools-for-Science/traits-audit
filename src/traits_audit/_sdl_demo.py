"""traits_audit demo — self-driving-lab-demo LED color-matching.

Uses an Ax ask-tell loop (``AxClient.get_next_trial`` /
``complete_trial``) — the same pattern shown in the original
`self-driving-lab-demo <https://github.com/sparks-baird/self-driving-lab-demo>`_
repository (``scripts/bayesian_optimization_basic.py``).  The GP posterior is
queried *before* each observation is incorporated, so the audit hook receives
genuine pre-observation predictive distributions.

By default this is a genuine Ax **multi-objective** run: ``sdl.evaluate()``
already computes ``frechet`` *and* ``mae`` from the same sensor read, so a
second Ax metric is tracked at no extra oracle cost, giving Ax's own
multi-output BoTorch model (not a hand-rolled one) and two real epistemic
posterior stds. The aleatoric floor is Ax's own **learned** per-metric
observation-noise estimate (introspected from the fitted model, falling back
to a one-time held-out estimate if that introspection ever fails on a future
Ax version) -- a controllability-Gramian mechanism check runs on
``[sigma_ep_frechet, sigma_ep_mae, sigma_al_frechet, sigma_al_mae]``. Pass
``--mechanism-null`` for the two-epistemic falsification test instead (see
``run()``'s docstring).

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


def _ensure_self_driving_lab_demo_importable() -> None:
    """self_driving_lab_demo's __init__.py unconditionally imports
    utils.search, which does ``from ax import optimize`` — an API removed
    from the modern ax-platform this demo needs elsewhere (AxClient service
    API, generation_strategy modules below). We only use
    SelfDrivingLabDemoLight, never utils.search, so stub that submodule out
    and retry rather than pinning ax-platform to a version incompatible with
    the rest of this file.
    """
    if "self_driving_lab_demo" in sys.modules:
        return
    try:
        import self_driving_lab_demo  # noqa: F401
    except ImportError as exc:
        if not (getattr(exc, "name", None) == "ax" and "optimize" in str(exc)):
            raise
        import types
        stub = types.ModuleType("self_driving_lab_demo.utils.search")
        stub.ax_bayesian_optimization = None
        stub.grid_search = None
        stub.random_search = None
        sys.modules["self_driving_lab_demo.utils.search"] = stub
        import self_driving_lab_demo  # noqa: F401  (retry)


# ── Ax-native per-metric noise (mechanism check) ──────────────────────────────

def _extract_ax_noise(ax_client, metrics: list[str]) -> dict[str, float] | None:
    """Best-effort extraction of Ax's LEARNED per-metric observation-noise
    estimate, de-standardized back to each metric's raw scale.

    Introspects the currently-fitted BoTorch model's likelihood noise
    (``fitted_adapter.botorch_model.likelihood.noise``, ordered by
    ``fitted_adapter.outcomes``) and Ax's ``StandardizeY`` transform
    (``Ystd``) to convert the standardized-scale noise variance back to the
    metric's own units: ``sigma_al = sqrt(noise) * Ystd``. This walks
    Ax/BoTorch internals that can change across versions -- if introspection
    fails for ANY reason (wrong node, no fitted model yet, a future Ax
    refactor), returns ``None`` so the caller falls back to a one-time
    held-out estimate instead (see ``run()``'s docstring /
    ``paper1_logical_pitfalls.md`` Category 1: whichever path is taken,
    sigma_al must stay a FIXED floor, re-estimated only this
    infrequently-refreshed way, never the live per-step posterior).
    """
    try:
        node = ax_client.generation_strategy.current_node
        fitted = node.generator_spec_to_gen_from.fitted_adapter
        bm = fitted.botorch_model
        noise = bm.likelihood.noise.detach().cpu().numpy().ravel()
        outcomes = list(fitted.outcomes)
        ystd = fitted.transforms["StandardizeY"].Ystd
        return {
            name: float(np.sqrt(noise[outcomes.index(name)]) * ystd[name])
            for name in metrics
        }
    except Exception:
        return None


def _fit_fallback_floor(X_norm: np.ndarray, y_by_metric: dict[str, list[float]],
                        seed: int) -> dict[str, float]:
    """One-time held-out-residual floor per metric (fallback only).

    Used only if :func:`_extract_ax_noise` isn't introspectable for the Ax
    version installed at runtime. Fit once from the Sobol warm-start
    observations; never refit, so it stays a genuine fixed floor.
    """
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor

    X_norm = np.asarray(X_norm, dtype=float)
    out: dict[str, float] = {}
    for name, y in y_by_metric.items():
        y = np.asarray(y, dtype=float)
        if len(X_norm) < 6:
            out[name] = float(np.std(y)) * 0.1 if len(y) > 1 else 1.0
            continue
        Xtr, Xte, ytr, yte = train_test_split(X_norm, y, test_size=0.3, random_state=seed)
        m = AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=3), n_estimators=20, random_state=seed,
        ).fit(Xtr, ytr)
        out[name] = float(np.std(yte - m.predict(Xte)))
    return out


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
        DMDcSpectralRadiusCheck,
        SignedBiasCheck,
        TailIndexCheck,
        ScoreDecompositionCheck,
        TypeBMassFractionCheck,
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
            # Paired with LyapunovStabilityCheck -- see validation.NAME_PAIRS
            # and run(), where rho_A comes from a single global DMDc fit on
            # the full query trajectory (as opposed to Lyapunov's rolling
            # per-step local Jacobian series).
            DMDcSpectralRadiusCheck(stability_threshold=1.0),
            # Cross-cutting, cheap.
            SignedBiasCheck(),
            TailIndexCheck(),
            ScoreDecompositionCheck(),
            # Ax's learned per-metric observation-noise estimate
            # (_extract_ax_noise, already computed for the mechanism check)
            # is exactly a declared Type-B provenance ledger -- see run().
            TypeBMassFractionCheck(),
        ],
        verbose=False,
    )
    return AuditHook(pipeline, check_every=check_every, logger=logger)


# ── Main run ───────────────────────────────────────────────────────────────────

def run(
    n_init: int = 6,
    n_iter: int = 25,
    out_dir: Path = Path("_results/sdl_demo"),
    seed: int = 0,
    check_every: int = 10,
    metric: str = "frechet",
    second_metric: str = "mae",
    mechanism_null: bool = False,
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
    second_metric : str
        A second Ax metric, computed from the same ``sdl.evaluate()`` call as
        ``metric`` at no extra oracle cost (default ``"mae"`` alongside
        ``"frechet"``). Unless ``mechanism_null``, this turns the experiment
        into a genuine Ax multi-objective problem (both metrics jointly
        optimised) -- an intentional behaviour change from a single-objective
        run, since Ax's own generation strategy now targets both. Used for
        the controllability-Gramian mechanism check's second epistemic
        component.
    mechanism_null : bool
        If True, keep the ORIGINAL single-objective (``metric`` only) Ax
        experiment (so acquisition behaviour is unchanged from a plain run),
        and instead run the E6-style two-epistemic null alongside it: two
        independent sklearn GPs, refit each step on the same growing
        (state, ``metric``) observations with different random seeds, giving
        ``[sigma_run1, sigma_run2]`` -- both reducible, so the eigenvalue
        ratio should collapse toward the null band rather than the real
        multi-objective run's separation. See ``paper1_logical_pitfalls.md``
        Category 5. Run the demo once with this False and once True to
        compare.
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
        _ensure_self_driving_lab_demo_importable()
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

        def __init__(self, tracked_metric: str, second_metric: str | None = None,
                     *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._tracked_metric = tracked_metric
            self._second_metric = second_metric
            self._last_mu: float = float("nan")
            self._last_sigma: float = float("nan")
            self._last_sigma_second: float = float("nan")

        def get_next_trial(self, *args, **kwargs):
            params, trial_idx = super().get_next_trial(*args, **kwargs)
            self._last_mu, self._last_sigma = self.predict(params)
            if self._second_metric is not None:
                _, self._last_sigma_second = self.predict(params, metric=self._second_metric)
            return params, trial_idx

        def predict(self, params: dict, metric: str | None = None) -> tuple[float, float]:
            """Return (mean, std) from the current GP at *params*.

            ``metric`` defaults to the primary tracked metric; pass the
            second metric's name to query its posterior instead (same fitted
            multi-output model, just a different output). Returns (nan, nan)
            if the model is not yet a GP (Sobol phase) or if prediction fails
            for any reason.
            """
            metric = metric or self._tracked_metric
            try:
                preds = self.get_model_predictions_for_parameterizations([params])
                if preds and metric in preds[0]:
                    mu, sigma = preds[0][metric]
                    return float(mu), float(sigma)
            except Exception:
                pass
            return float("nan"), float("nan")

        @property
        def last_sigma_second(self) -> float:
            return self._last_sigma_second

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
            "platform":       "self-driving-lab-demo",
            "n_init":         n_init,
            "n_iter":         n_iter,
            "seed":           seed,
            "check_every":    check_every,
            "metric":         metric,
            "second_metric":  second_metric,
            "mechanism_null": mechanism_null,
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

    # ── Held-out evaluation set ────────────────────────────────────────────────
    # A fixed set of (R, G, B) points drawn uniformly at random from the
    # domain now, before any Ax trial runs, and evaluated once via the real
    # simulator -- but never registered with ax_client.complete_trial(), so
    # they never become training data. The final report's calibration/
    # coverage/scoring checks are evaluated against these points (passed
    # explicitly to hook.on_end() below) instead of falling back to the
    # per-step BO history, which is an acquisition-biased sample.
    _n_holdout = 12
    _hold_rng = np.random.default_rng(int(seed) + 5000)
    _hold_params = [
        {
            "R": float(_hold_rng.uniform(*bounds["R"])),
            "G": float(_hold_rng.uniform(*bounds["G"])),
            "B": float(_hold_rng.uniform(*bounds["B"])),
        }
        for _ in range(_n_holdout)
    ]
    y_hold = np.array([float(sdl.evaluate(p)[metric]) for p in _hold_params])
    print(f"  Held out {_n_holdout} points for final calibration evaluation (never queried)")

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

    # Real run: genuine Ax multi-objective (metric + second_metric), so Ax's
    # own generation strategy fits a real multi-output model and jointly
    # optimises both -- an intentional behaviour change from a plain
    # single-objective run. Null run (--mechanism-null): keep the ORIGINAL
    # single-objective experiment so acquisition is unaffected, and track a
    # separate two-epistemic null (below) via plain sklearn GPs instead.
    ax_client = PredictingAxClient(
        tracked_metric=metric,
        second_metric=None if mechanism_null else second_metric,
        generation_strategy=_gen_strategy,
        random_seed=seed,
        verbose_logging=False,
    )
    if mechanism_null:
        objectives = {metric: ObjectiveProperties(minimize=True)}
    else:
        objectives = {
            metric: ObjectiveProperties(minimize=True),
            second_metric: ObjectiveProperties(minimize=True),
        }
    ax_client.create_experiment(
        parameters=ax_parameters,
        objectives=objectives,
        overwrite_existing_experiment=True,
    )

    hook = _make_pipeline(check_every, logger=_mlflow_logger)

    uncertainties: list[float] = []
    unc_second:    list[float] = []   # 2nd mechanism-check component per step
    unc_null_a:    list[float] = []   # null run's 1st (replacement) component per step
    queried_norm:  list[list[float]] = []

    # ── Two-epistemic null (--mechanism-null only): two independent sklearn
    # GPs on the same `metric` target, refit each step, fully decoupled from
    # Ax's own model (kept single-objective above so acquisition is
    # unaffected). Not used at all in the real (multi-objective) run.
    _null_gpr_a = _null_gpr_b = None
    if mechanism_null:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

        def _make_null_kernel():
            return (
                ConstantKernel(1.0, (1e-3, 1e3)) * RBF([0.3, 0.3, 0.3], (1e-2, 1e2))
                + WhiteKernel(1e-2, (1e-6, 1e2))
            )

        _null_gpr_a = GaussianProcessRegressor(
            kernel=_make_null_kernel(), n_restarts_optimizer=2, normalize_y=True,
            random_state=int(seed),
        )
        _null_gpr_b = GaussianProcessRegressor(
            kernel=_make_null_kernel(), n_restarts_optimizer=2, normalize_y=True,
            random_state=int(seed) + 1000,
        )

    # Sobol warm-start observations, kept for the fallback floor and (in the
    # null run) the two independent GPs.
    _warmstart_state: list[list[float]] = []
    _warmstart_metric: list[float] = []
    _warmstart_second: list[float] = []

    # ── Warm-start: Sobol ─────────────────────────────────────────────────────
    print(f"\n[1/3] Warm-start — {n_init} Sobol trials …")
    for _ in range(n_init):
        params, trial_idx = ax_client.get_next_trial()
        results = sdl.evaluate({"R": params["R"], "G": params["G"], "B": params["B"]})
        # Pass raw float (no SEM=0) so Ax infers observation noise — matches
        # the original repo's evaluation_function pattern.
        raw_data = {metric: float(results[metric])}
        if not mechanism_null:
            raw_data[second_metric] = float(results[second_metric])
        ax_client.complete_trial(trial_idx, raw_data=raw_data)

        _warmstart_state.append([params["R"] / r_max, params["G"] / g_max, params["B"] / b_max])
        _warmstart_metric.append(float(results[metric]))
        _warmstart_second.append(float(results[second_metric if not mechanism_null else metric]))

    # One-time fallback floor (real run only), used if Ax's learned per-metric
    # noise (_extract_ax_noise) isn't introspectable on this Ax version.
    _fallback_floor: dict[str, float] = {}
    if not mechanism_null:
        _fallback_floor = _fit_fallback_floor(
            _warmstart_state,
            {metric: _warmstart_metric, second_metric: _warmstart_second},
            seed=int(seed),
        )
    # Growing training data for the null run's two independent GPs (starts
    # from the Sobol warm-start observations, same as Ax's own model saw).
    _null_X = list(_warmstart_state) if mechanism_null else None
    _null_y = list(_warmstart_metric) if mechanism_null else None

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
        # A NaN sigma is "no posterior std available yet", not "zero
        # uncertainty" -- substituting 0.0 previously asserted infinite
        # confidence for exactly the steps where the model has the LEAST
        # information (and did so inconsistently: y_pred_std below still
        # carried the raw NaN through to hook.on_step while `uncertainty`
        # got the substituted 0.0, so the same step recorded two different
        # values for the same underlying quantity). Skip the step's
        # recording entirely instead -- uncertainties/queried_norm/
        # hook.on_step all stay aligned, one entry per genuinely valid step.
        valid = not (np.isnan(mu) or np.isnan(sigma))
        if not valid:
            print(f"  [warn] GP predict returned NaN at step {step} -- skipping this step's record", flush=True)

        results = sdl.evaluate({"R": params["R"], "G": params["G"], "B": params["B"]})
        y = float(results[metric])
        raw_data = {metric: y}
        if not mechanism_null:
            raw_data[second_metric] = float(results[second_metric])
        ax_client.complete_trial(trial_idx, raw_data=raw_data)

        state_norm = [params["R"] / r_max, params["G"] / g_max, params["B"] / b_max]
        if not valid:
            continue
        uncertainties.append(sigma)
        queried_norm.append(state_norm)

        # Mechanism-check components at this same queried point. `uncertainties`
        # above stays the demo's real audit-facing series (Ax's own primary-
        # metric GP posterior) regardless of mode -- the null run's two
        # independent GPs are tracked separately in `unc_null_a`/`unc_second`
        # and never feed the audit pipeline, Lyapunov analysis, or figures.
        # Guarded by the same `valid` as uncertainties/queried_norm above so
        # unc_second stays index-aligned with them for the zip() below.
        if mechanism_null:
            _Xa = np.array(_null_X)
            _ya = np.array(_null_y)
            _null_gpr_a.fit(_Xa, _ya)
            _null_gpr_b.fit(_Xa, _ya)
            _, _s_a = _null_gpr_a.predict(np.array([state_norm]), return_std=True)
            _, _s_b = _null_gpr_b.predict(np.array([state_norm]), return_std=True)
            unc_null_a.append(float(_s_a[0]))
            unc_second.append(float(_s_b[0]))
            _null_X.append(state_norm)
            _null_y.append(y)
        elif not np.isnan(ax_client.last_sigma_second):
            unc_second.append(float(ax_client.last_sigma_second))
        else:
            # Keep unc_second aligned with uncertainties/queried_norm even
            # when the second metric's posterior isn't available this step.
            unc_second.append(float("nan"))

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

        _step_kwargs = dict(
            y_true=y,
            y_pred_mean=mu,
            y_pred_std=sigma,
            uncertainty=sigma,
            abs_error=abs(y - mu),
            lcb_score=float(mu - 2.0 * sigma),
            dataset_size=float(n_init + step + 1),
        )
        if np.isfinite(_lm_step):
            _step_kwargs["lambda_max"] = _lm_step
        hook.on_step(**_step_kwargs)

    # ── Controllability-Gramian mechanism check ───────────────────────────────
    # Real run: does the Gramian separate the two genuine Ax-fitted epistemic
    # posteriors (frechet, second_metric) from Ax's own LEARNED per-metric
    # noise floor? Null run (--mechanism-null): two independent sklearn GPs
    # on the same target, both reducible -- the ratio should collapse instead
    # of separating. Action is the genuine queried (R,G,B) point
    # (queried_norm), not a placeholder.
    from traits_audit import trajectory as _traj
    from traits_audit._mechanism_check import print_mechanism_check

    # unc_second (and unc_null_a in null mode) can carry a NaN on steps
    # where the SECOND metric's posterior specifically wasn't available
    # even though the primary metric was (see the `valid` gate above, which
    # only guards the primary) -- filter those out here so the DMDc fit
    # below never sees a NaN, rather than propagating one into the Gramian.
    _primary = unc_null_a if mechanism_null else uncertainties
    _finite_mask = [np.isfinite(a) and np.isfinite(b) for a, b in zip(_primary, unc_second)]
    _queried_norm_mech = [q for q, ok in zip(queried_norm, _finite_mask) if ok]
    _primary_f = [v for v, ok in zip(_primary, _finite_mask) if ok]
    _unc_second_f = [v for v, ok in zip(unc_second, _finite_mask) if ok]

    if mechanism_null:
        _unc_vectors = [np.array([a, b]) for a, b in zip(_primary_f, _unc_second_f)]
        _aleatoric_idx = None
        _mech_label = "two-epistemic null (--mechanism-null)"
    else:
        _floor = _extract_ax_noise(ax_client, [metric, second_metric]) or _fallback_floor
        _al_metric = float(_floor[metric])
        _al_second = float(_floor[second_metric])
        _unc_vectors = [
            np.array([ep, ep2, _al_metric, _al_second])
            for ep, ep2 in zip(_primary_f, _unc_second_f)
        ]
        _aleatoric_idx = [2, 3]
        _mech_label = (
            "real split ("
            f"{metric}+{second_metric} posteriors, "
            f"{'Ax-learned' if _floor is not _fallback_floor else 'held-out fallback'} noise floors)"
        )

    mech_rec = _traj.from_sdl(
        {"uncertainties": _unc_vectors, "queried_norm": _queried_norm_mech},
        policy="Ax-BoTorch-GP" if not mechanism_null else "Ax-BoTorch-GP(null)",
    )
    mech_result = _traj.analyze_trajectory(mech_rec, n_components=6)
    print_mechanism_check(mech_result, _mech_label, aleatoric_indices=_aleatoric_idx)

    # ── Lyapunov analysis ──────────────────────────────────────────────────────
    print("\n[3/3] Lyapunov stability analysis + final audit …")
    from traits_audit._viz import (
        make_gd_predictor,
        run_lyapunov_analysis,
        check_grid_figures,
        plot_uncertainty_evolution,
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

    # DMDcSpectralRadius: paired with LyapunovStabilityCheck -- a single
    # global DMDc fit on the query trajectory (augmented with the reported
    # sigma), as opposed to Lyapunov's per-point Jacobian series.
    from traits_audit import dmdc as _dm
    aug_states = np.column_stack([op_states, np.array(uncertainties)])
    try:
        _A_r, _, _ = _dm.fit_dmdc(aug_states, op_states, n_components=min(3, aug_states.shape[1]))
        rho_A = float(np.max(np.abs(np.linalg.eigvals(_A_r))))
    except Exception as exc:
        print(f"  DMDc fit failed ({exc}); DMDcSpectralRadiusCheck will skip.")
        rho_A = None

    # TypeBMassFraction: Ax's learned per-metric observation-noise estimate
    # (already extracted above for the mechanism check) IS a declared
    # Type-B provenance ledger -- the noise floor is a fitted-but-fixed
    # hyperparameter, not derived from the primary metric's own posterior.
    from traits_audit.provenance import TypeBLedger
    _tbmf_floor = _extract_ax_noise(ax_client, [metric]) or _fallback_floor
    _al_primary = float(_tbmf_floor[metric])
    _v_full = float(uncertainties[-1]) ** 2 + _al_primary ** 2 if uncertainties else None
    _tbmf_kwargs = {}
    if _v_full and _v_full > 0:
        _ledger = TypeBLedger(
            components={"noise_level": _al_primary, "kernel_variance": _v_full - _al_primary ** 2},
            type_b_keys={"noise_level"},
        )

        def _tbmf_variance_fn(ablate, _v=_v_full, _noise=_al_primary ** 2):
            return max(_v - _noise, 1e-12) if "noise_level" in ablate else _v

        _tbmf_kwargs = {"ledger": _ledger, "variance_fn": _tbmf_variance_fn}

    # Evaluate the FINAL fitted Ax model on the held-out set carved out
    # before the loop started (never queried, never trained on) -- passed
    # explicitly below so it wins over the per-step history in `_require`
    # (kwargs take priority) for this final report only. Intermediate
    # check_every snapshots are unaffected and keep reading the growing BO
    # history via on_step.
    _mu_hold, _sigma_hold = [], []
    for _p in _hold_params:
        _m, _s = ax_client.predict(_p)
        _mu_hold.append(_m)
        _sigma_hold.append(_s)
    _mu_hold = np.array(_mu_hold)
    _sigma_hold = np.array(_sigma_hold)
    _hold_valid = np.isfinite(_mu_hold) & np.isfinite(_sigma_hold)
    if not _hold_valid.all():
        print(f"  Note: {(~_hold_valid).sum()}/{_n_holdout} held-out points had no "
              "GP prediction available (model not yet fitted?) and were dropped.")

    # Pass lambda_max so LyapunovStabilityCheck runs as part of the final report.
    report = hook.on_end(
        lambda_max=lyap["lambda_max"], rho_A=rho_A,
        y_true=y_hold[_hold_valid] if _hold_valid.any() else None,
        y_pred_mean=_mu_hold[_hold_valid] if _hold_valid.any() else None,
        y_pred_std=_sigma_hold[_hold_valid] if _hold_valid.any() else None,
        **_tbmf_kwargs,
    )
    print("\n" + report.summary())
    if report.metadata.get("pairing_warnings"):
        print("\n  Pairing warnings:")
        for w in report.metadata["pairing_warnings"]:
            print(f"    - {w}")

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
    fig_grid, fig_grid_final = check_grid_figures(stage_reports, "Ax-GP (SDL)")
    if fig_grid is not None:
        try:
            fig_grid.write_image(
                str(fig_dir / "check_grid_sdl.png"),
                width=fig_grid.layout.width, height=fig_grid.layout.height, scale=2,
            )
            print("  Saved check_grid_sdl.png")
        except Exception:
            pass
    if fig_grid_final is not None:
        # Checks with no per-snapshot value (e.g. TypeBMassFraction needs the
        # kernel state only read once, post-loop) get their own compact,
        # single-column grid instead of padding the main one.
        try:
            fig_grid_final.write_image(
                str(fig_dir / "check_grid_sdl_final_only.png"),
                width=fig_grid_final.layout.width, height=fig_grid_final.layout.height, scale=2,
            )
            print("  Saved check_grid_sdl_final_only.png")
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

    # NOT plot_lyapunov_evolution here: lyap["lambda_max"] is the FINAL
    # fitted GP's Jacobian evaluated at every historical operating point --
    # a spatial scan of a static model, not a time series (same issue as
    # ta-pybamm-demo; see that module's matching note). Despite this loop
    # also recording a genuinely rolling per-step _lm_step into
    # hook.history, hook.on_end(lambda_max=lyap["lambda_max"]) above
    # overrides it for the actual check verdict, so the rolling series
    # isn't what gets evaluated either. run_lyapunov_analysis already wrote
    # the honest non-temporal alternative, fig3_stability_vs_unc.png.

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
    return {"report": report, "lyapunov": lyap, "mechanism_check": mech_result}


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--n-init",      type=int,   default=6,
                   help="Sobol warm-start trials (default: 6)")
    p.add_argument("--n-iter",      type=int,   default=25,
                   help="BO iterations (default: 25, ~1 min). Each iteration "
                        "refits an Ax/BoTorch GP (~O(n^3)) plus a rolling "
                        "Jacobian-based Lyapunov step; n_iter=100 takes "
                        "~2-4 min, n_iter=250 (the old default) 7-16 min.")
    p.add_argument("--out-dir",     type=str,   default="_results/sdl_demo")
    p.add_argument("--seed",        type=int,   default=0)
    p.add_argument("--check-every", type=int,   default=10,
                   help="Intermediate audit frequency (default: 10)")
    p.add_argument("--metric",      type=str,   default="frechet",
                   help="Ax metric name (default: frechet)")
    p.add_argument("--second-metric", type=str, default="mae",
                   help="Second Ax metric, tracked from the same evaluate() "
                        "call for the controllability-Gramian mechanism "
                        "check (default: mae)")
    p.add_argument("--mechanism-null", action="store_true",
                   help="Run the two-epistemic null instead of the real "
                        "multi-objective split for the controllability-"
                        "Gramian mechanism check (falsification test; "
                        "compare against a normal run's ratio)")
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
        second_metric=args.second_metric,
        mechanism_null=args.mechanism_null,
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
