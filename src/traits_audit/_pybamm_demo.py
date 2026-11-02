"""traits_audit demo — PyBAMM Li-ion MOBO: capacity × voltage optimisation.

Multi-objective Bayesian optimisation over a **continuous** cell-design space
using the PyBAMM Single Particle Model as the oracle.  Both objectives are
pulled from the same SPM solve at no extra cost:

- **Objective 1**: discharge capacity [Ah] — maximise (preserved capacity)
- **Objective 2**: mean terminal voltage [V] — maximise (proxy for fast-charge
  capability: higher voltage under load means lower internal resistance)

Domain
------
Controlled design inputs, normalised to ∈ [0, 1]³ (continuous — optimised
via ``optimize_acqf``, not a discrete pool):

- **C-rate**                0.5 C – 3.0 C
- **Electrode thickness**   ×0.6 – ×1.4 of the default negative-electrode
                            thickness (thinner → better rate capability,
                            thicker → more capacity at low C-rate)
- **Particle radius**       ×0.5 – ×2.0 of the default negative-particle
                            radius (smaller → faster solid-state diffusion,
                            better high-rate performance)

**Temperature is NOT a design input.**  Every evaluation draws an ambient
temperature uniformly from [10, 40] °C, representing uncontrolled
experimental/environmental variability rather than a variable under the
experimenter's control — the SPM sees it (it genuinely affects the solve),
but the surrogate never does, so its effect on capacity/voltage shows up
purely as unexplained (aleatoric) scatter relative to the three controlled
inputs.

Noise model
-----------
Observations carry two additive components:

1. **i.i.d. Gaussian** sensor noise (``noise_std``, ``noise_std_voltage``).
2. **1/f ("pink"/flicker) noise** — a slowly-wandering, temporally-correlated
   component generated once as a fixed-length sequence (white noise with its
   FFT amplitude spectrum scaled by ``1/sqrt(f)``) and consumed in
   chronological observation order.  This models real instrument drift,
   which is *not* i.i.d. and is exactly the kind of systematic/persistent
   component the replication-arm checks (RSE/DUG) are built to separate
   from pure random noise.

Oracle:      PyBAMM SPM single discharge → (capacity [Ah], mean voltage [V])
Surrogate:   BoTorch ModelListGP (two independent SingleTaskGP, one per
             objective); capacity GP drives the primary uncertainty audit.
Policy:      qLogNoisyExpectedHypervolumeImprovement (qLogNEHVI) via
             continuous ``optimize_acqf`` over the unit cube, in BATCHES of
             ``n_query`` points per iteration (one model fit per batch, not
             per point — the standard batch-AL pattern, matching
             ``ta-camd-demo``).
Stopping:    ``n_iter`` is a MAX batch cap, not a fixed count. The campaign
             auto-terminates once relative hypervolume improvement over the
             trailing ``patience`` batches drops below ``min_improvement``
             ("the experiment quit improving").
Audit:       Full TRAITS-AUDIT pipeline (calibration / coverage / scoring /
             Lyapunov-DMDc / replication arm / mechanism check).
Stability:   Lyapunov (local, per operating point) + DMDc rho(A) (global) on
             the capacity-GP gradient-descent map.
Mechanism:   Controllability-Gramian check on
             [σ_ep_cap, σ_ep_volt, σ_al_cap, σ_al_volt].
Replication: RSE / DUG / AFC on 3 FIXED anchor locations in the continuous
             domain, revisited on a rotating schedule as ONE of the
             ``n_query`` slots in every batch — i.e. real experimental QC
             practice (interleave repeat measurements of a reference
             condition with new exploration, rather than a separate
             all-at-once burst after the campaign ends).

Entry point::

    ta-pybamm-demo [OPTIONS]
    ta-pybamm-demo --n-iter 30 --n-query 4 --out-dir _results/pybamm_demo --seed 7
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# GD-predictor step size for LyapunovStabilityCheck's Jacobian construction --
# matches the check's own documented default and every other GD-predictor-
# based demo in this package (_cal_demo.py, _sdl_demo.py, _mobo_demo.py). A
# larger step size makes J = I - alpha*H_f proportionally more sensitive to
# the surrogate's curvature, so an inconsistent alpha here (previously a
# hardcoded 0.05, 5x steeper) made this demo's instability flags fire far
# more readily than the same surrogate would show elsewhere in the package.
_LYAPUNOV_ALPHA = 0.01

# ── Design-space bounds (controlled inputs) ─────────────────────────────────────
_C_MIN, _C_MAX         = 0.5, 3.0     # C-rate  [C]
_THICK_MIN, _THICK_MAX = 0.6, 1.4     # negative-electrode thickness, × default
_PART_MIN, _PART_MAX   = 0.5, 2.0     # negative-particle radius, × default

# Ambient temperature: NOT a design input -- an uncontrolled noise source.
# Every evaluation draws T ~ Uniform(_T_MIN, _T_MAX) independently; the SPM
# solve sees it, the surrogate never does.
_T_MIN, _T_MAX = 10.0, 40.0   # [°C]

# Hypervolume reference point — must be dominated by all observed Y values.
# The wider design space (thin electrode + large particle + high C-rate) can
# push capacity down to ~0.13 Ah, well below the old 2-D domain's floor, so
# this is set with comfortable margin below the worst physically-reachable
# combination rather than the old domain's typical worst case.
_REF_POINT = torch.tensor([0.1, 3.5], dtype=torch.float64)

_BOUNDS_UNIT_CUBE = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)


def _norm(c_rate: float, thickness_scale: float, particle_scale: float) -> np.ndarray:
    return np.array([
        (c_rate          - _C_MIN)     / (_C_MAX     - _C_MIN),
        (thickness_scale - _THICK_MIN) / (_THICK_MAX - _THICK_MIN),
        (particle_scale  - _PART_MIN)  / (_PART_MAX  - _PART_MIN),
    ])


def _denorm(state_3: np.ndarray) -> tuple[float, float, float]:
    c  = state_3[0] * (_C_MAX     - _C_MIN)     + _C_MIN
    ts = state_3[1] * (_THICK_MAX - _THICK_MIN) + _THICK_MIN
    ps = state_3[2] * (_PART_MAX  - _PART_MIN)  + _PART_MIN
    return float(c), float(ts), float(ps)


# ── PyBAMM oracle ───────────────────────────────────────────────────────────────

def _build_pybam():
    """Initialise the SPM model and base parameter set once."""
    import pybamm
    pybamm.set_logging_level("WARNING")
    model = pybamm.lithium_ion.SPM()
    param = model.default_parameter_values.copy()
    Cn    = float(param["Nominal cell capacity [A.h]"])
    return model, param, Cn


def _simulate_observables(
    c_rate: float, thickness_scale: float, particle_scale: float, T_amb_C: float,
    model, param, Cn: float,
) -> tuple[float, float]:
    """Run one SPM discharge at ``c_rate`` C, with the negative-electrode
    thickness and negative-particle radius scaled relative to the default
    cell design, at ambient temperature ``T_amb_C``.

    ``T_amb_C`` is not a controlled design input (see module docstring) --
    it is drawn per-observation from a uniform ambient-temperature
    distribution in ``run()`` and passed straight through here.

    Returns ``(capacity [Ah], mean_voltage [V])``.  Both are pulled from the
    same solve at no extra oracle cost.  A 2-hour cap prevents solver stalls
    at very low C-rates.
    """
    import pybamm
    p = param.copy()
    p["Current function [A]"]    = c_rate * Cn
    p["Ambient temperature [K]"] = 273.15 + T_amb_C
    p["Initial temperature [K]"] = 273.15 + T_amb_C
    p["Negative electrode thickness [m]"] = float(param["Negative electrode thickness [m]"]) * thickness_scale
    p["Negative particle radius [m]"]     = float(param["Negative particle radius [m]"])     * particle_scale
    sim   = pybamm.Simulation(model, parameter_values=p)
    t_end = min(3600.0 / c_rate, 7200.0)
    sol   = sim.solve([0, t_end])
    capacity     = float(sol["Discharge capacity [A.h]"].entries[-1])
    mean_voltage = float(np.mean(sol["Terminal voltage [V]"].entries))
    return capacity, mean_voltage


# ── Noise model ─────────────────────────────────────────────────────────────────

def _make_pink_noise(n: int, rng: np.random.Generator, target_std: float) -> np.ndarray:
    """Generate a 1/f ("pink"/flicker) noise sequence of length ``n``.

    White noise is generated, its FFT amplitude spectrum scaled by
    ``1/sqrt(f)`` (giving a power spectral density ~ 1/f), then inverse-
    transformed and rescaled to ``target_std``. This models slow, temporally-
    correlated instrument drift -- distinct from i.i.d. Gaussian sensor
    noise, and exactly the kind of persistent/systematic component the
    replication-arm checks (RSE/DUG) are built to separate from pure random
    noise. Consumed in chronological observation order by ``run()``, so
    measurements taken close together in the campaign share correlated
    drift while distant ones do not.
    """
    if n < 8:
        return np.zeros(n)
    white = rng.standard_normal(n)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1]  # avoid the 1/0 divide-by-zero at DC
    spectrum = np.fft.rfft(white) / np.sqrt(freqs)
    pink = np.fft.irfft(spectrum, n)
    pink -= pink.mean()
    std = pink.std()
    if std > 1e-12:
        pink *= target_std / std
    return pink


# ── BoTorch surrogate helpers ──────────────────────────────────────────────────

def _fit_model(train_X: torch.Tensor, train_Y: torch.Tensor):
    """Fit a ModelListGP on two objectives (capacity, voltage) — both maximised.

    Both objectives are positive values (no negation needed since BoTorch
    maximises and we want to maximise both capacity and voltage).
    """
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import ModelListGP, SingleTaskGP
    from gpytorch.mlls import SumMarginalLogLikelihood

    models = [
        SingleTaskGP(train_X, train_Y[:, i : i + 1])
        for i in range(train_Y.shape[1])
    ]
    model = ModelListGP(*models)
    mll   = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()
    return model


def _predict(model, x_n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) for both objectives at a single normalised point."""
    x_t = torch.tensor(x_n, dtype=torch.float64).unsqueeze(0)
    with torch.no_grad():
        post = model.posterior(x_t)
    mean = post.mean[0].numpy()
    std  = post.variance[0].sqrt().numpy()
    return mean, std


def _best_batch_continuous(model, train_X: torch.Tensor, q: int) -> np.ndarray:
    """Select the next batch of ``q`` candidates via qLogNEHVI over the
    continuous unit-cube domain (``optimize_acqf``, not a discrete pool) --
    one joint acquisition-function optimisation over the whole batch, not
    ``q`` independent single-point picks — the standard batch-AL pattern:
    one model state proposes a whole batch, the batch is observed, then the
    model is refit once.

    ``num_restarts=20, raw_samples=512`` (up from BoTorch's tutorial
    defaults of 5/256) avoids the ``ABNORMAL_TERMINATION_IN_LNSRCH``
    L-BFGS-B failures seen at the lower defaults elsewhere in this package
    (ta-mobo-demo, ta-sdl-demo).

    Returns ``(q, D)``.
    """
    from botorch.acquisition.multi_objective.logei import (
        qLogNoisyExpectedHypervolumeImprovement,
    )
    from botorch.optim import optimize_acqf
    from botorch.sampling.normal import SobolQMCNormalSampler

    acqf = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=_REF_POINT.to(dtype=torch.float64),
        X_baseline=train_X,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])),
    )
    best_x, _ = optimize_acqf(
        acqf, bounds=_BOUNDS_UNIT_CUBE, q=q, num_restarts=20, raw_samples=512,
    )
    return best_x.numpy()


def _hypervolume(Y_obs: np.ndarray) -> float:
    """Hypervolume dominated by the non-dominated set of Y_obs (maximisation)."""
    from botorch.utils.multi_objective.hypervolume import Hypervolume
    from botorch.utils.multi_objective.pareto import is_non_dominated

    Y_t  = torch.tensor(Y_obs, dtype=torch.float64)
    mask = is_non_dominated(Y_t)
    hv   = Hypervolume(ref_point=_REF_POINT.to(dtype=torch.float64))
    return float(hv.compute(Y_t[mask]))


# ── Audit pipeline ─────────────────────────────────────────────────────────────

def _make_pipeline(logger=None):
    from traits_audit import AuditHook, AuditPipeline
    from traits_audit.checks import (
        AleatoricFloorConsistencyCheck,
        CalibrationErrorCheck,
        ConformalCoverageCheck,
        CRPSCheck,
        DarkUncertaintyGapCheck,
        DMDcSpectralRadiusCheck,
        IntervalCoverageCheck,
        IntervalScoreCheck,
        LyapunovStabilityCheck,
        NegativeLogLikelihoodCheck,
        PITUniformityCheck,
        ReducibilityRealisationRatioCheck,
        ReplicationShrinkageExponentCheck,
        ScoreDecompositionCheck,
        SignedBiasCheck,
        TailIndexCheck,
        UncertaintyAnomalyCheck,
        UncertaintyEvolutionCheck,
        VarianceAlignmentCheck,
        VarianceErrorCorrelationCheck,
    )
    pipeline = AuditPipeline(
        checks=[
            CalibrationErrorCheck(threshold=0.15),
            ConformalCoverageCheck(target_coverage=0.9, max_q_ratio=1.5),
            # Thresholds below (CRPS / NLL / IntervalScore / ScoreDecomposition)
            # were empirically calibrated against a real run of THIS demo's
            # continuous 3-feature domain + mixed Gaussian/1-f noise model
            # (18 batches, 82 total points, 60-point held-out set):
            #   CRPS measured 0.0313, calibrated reference (perfectly-
            #     calibrated model, same noise scale) ~= 0.0090 -> threshold
            #     = 3x reference = 0.027.
            #   IntervalScore measured 0.4389, reference ~= 0.0849 ->
            #     threshold = 3x reference = 0.25.
            #   NLL measured +1.15, reference ~= -2.76 (very negative
            #     because the noise stds here are small; a well-calibrated
            #     model's NLL scales with -log(sigma) and goes strongly
            #     negative). Reference is negative so a multiplicative
            #     margin doesn't transfer; threshold = 0.0 instead (a
            #     predictive density that on average assigns the true value
            #     density < 1, i.e. NLL > 0, is a meaningful failure mode
            #     independent of the absolute noise scale).
            #   ScoreDecomposition's CAL sits on the same NLL/nats scale as
            #     UNC (measured UNC ~= -1.05 here) -- threshold = 1.0 so
            #     miscalibration is capped at roughly the climatological
            #     base rate's magnitude.
            # These four checks default to threshold=None (report-only)
            # because "good" depends on the problem's noise scale; the
            # values below are this demo's own calibration, not a universal
            # default. RSE deliberately keeps beta_tolerance=None: beta is a
            # property of the replication scheme, not a right/wrong answer
            # (0.5 = pure random, 0 = fully systematic are both legitimate),
            # so no threshold is imposed there.
            CRPSCheck(threshold=0.027),
            NegativeLogLikelihoodCheck(threshold=0.0),
            PITUniformityCheck(),
            IntervalScoreCheck(threshold=0.25),
            IntervalCoverageCheck(expected_coverage=0.683, tolerance=0.15),
            VarianceAlignmentCheck(tolerance=0.5),
            UncertaintyEvolutionCheck(),
            UncertaintyAnomalyCheck(z_threshold=3.0),
            VarianceErrorCorrelationCheck(min_correlation=0.1),
            LyapunovStabilityCheck(
                stability_threshold=1.0, min_stable_fraction=0.5, window=30,
                alpha=_LYAPUNOV_ALPHA,
            ),
            DMDcSpectralRadiusCheck(stability_threshold=1.0),
            SignedBiasCheck(),
            TailIndexCheck(),
            ScoreDecompositionCheck(cal_threshold=1.0),
            # r_values reduced from the check's default (2,4,8,16): the
            # replication arm here is 3 fixed anchors revisited on a
            # rotating schedule (one slot per batch), so by the end of a
            # realistic n_iter/n_query budget each anchor has accumulated
            # only a handful of replicates -- not enough to fit r=8 or r=16
            # subsample windows. (2,4,6) is what this demo's budget can
            # actually support; see the replication-arm setup in run().
            ReplicationShrinkageExponentCheck(r_values=(2, 4, 6)),
            DarkUncertaintyGapCheck(),
            AleatoricFloorConsistencyCheck(),
            ReducibilityRealisationRatioCheck(),
        ],
        verbose=False,
    )
    # check_every=None: intermediate snapshots are built manually in run()
    # instead of via the hook's own auto-trigger, because that auto-trigger
    # never passes kwargs (only accumulated history) -- which is exactly
    # why RSE/DUG/AFC could never have real intermediate values before: the
    # replicate_groups kwarg they need was only ever passed at on_end().
    # Building snapshots manually lets replicate_groups (now growing
    # throughout the campaign, see run()) be passed at every snapshot too.
    return AuditHook(pipeline, check_every=None, logger=logger)


# ── Main run ───────────────────────────────────────────────────────────────────

def run(
    n_seed: int = 10,
    n_iter: int = 30,
    n_query: int = 4,
    patience: int = 4,
    min_improvement: float = 0.01,
    out_dir: Path = Path("_results/pybamm_demo"),
    seed: int = 0,
    check_every: int = 5,
    noise_std: float = 0.003,
    noise_std_voltage: float = 0.005,
    pink_frac: float = 0.8,
    mlflow_uri: str | None = None,
    run_name: str = "pybamm_demo",
) -> dict:
    """Run the PyBAMM MOBO demo (capacity × voltage) over a continuous,
    3-feature design space with mixed Gaussian/1-f observation noise.

    Parameters
    ----------
    n_seed : int
        Random (uniform) seed evaluations before the qNEHVI loop starts.
    n_iter : int
        MAXIMUM number of batch qLogNEHVI iterations. The campaign can stop
        earlier -- see ``patience``/``min_improvement``.
    n_query : int
        Points queried per batch. One slot in every batch goes to a
        rotating replicate anchor (real experimental QC practice: interleave
        repeat measurements with new exploration); the remaining
        ``n_query - 1`` slots are genuine qLogNEHVI-selected new candidates,
        chosen jointly via continuous ``optimize_acqf``. One GP refit per
        batch, not per point.
    patience : int
        Number of trailing batches over which hypervolume improvement is
        measured for the auto-stop rule.
    min_improvement : float
        Relative hypervolume improvement threshold over the trailing
        ``patience`` batches; the campaign stops once improvement falls
        below this ("the experiment quit improving").
    out_dir : Path
        Root directory for results.
    seed : int
        RNG seed.
    check_every : int
        Intermediate audit frequency, in individual point observations
        (matching every other demo's convention) -- NOT in batches.
    noise_std : float
        Additive Gaussian observation noise on discharge capacity [Ah].
    noise_std_voltage : float
        Additive Gaussian observation noise on mean terminal voltage [V].
    pink_frac : float
        Std of the additional 1/f ("pink") noise component, as a fraction
        of ``noise_std``/``noise_std_voltage``. Set to 0 to disable.
    """
    import os
    import warnings
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    warnings.filterwarnings("ignore", category=UserWarning,
                            module="sklearn.gaussian_process")
    warnings.filterwarnings("ignore", category=UserWarning)

    try:
        import pybamm as _pybamm  # noqa: F401
    except ImportError:
        print("ERROR: pybamm is not installed.  pip install pybamm")
        sys.exit(1)

    torch.manual_seed(seed)

    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("PyBAMM Demo: Li-ion MOBO  capacity × voltage optimisation")
    print(f"  n_seed={n_seed}  n_iter(max)={n_iter}  n_query={n_query}  "
          f"patience={patience}  min_improvement={min_improvement:.1%}  seed={seed}")
    print(f"  Controlled: C-rate ∈ [{_C_MIN},{_C_MAX}] C  "
          f"thickness ∈ [{_THICK_MIN},{_THICK_MAX}]×  particle ∈ [{_PART_MIN},{_PART_MAX}]×")
    print(f"  Uncontrolled noise source: ambient T ~ Uniform[{_T_MIN},{_T_MAX}] °C")
    print("  Objectives: max capacity [Ah]  +  max mean voltage [V]")
    print(f"  Output: {out_dir}")
    print("=" * 60)

    # ── MLflow setup ──────────────────────────────────────────────────────────
    _use_mlflow = mlflow_uri is not None
    if _use_mlflow:
        import mlflow as _mlflow

        from traits_audit.mlflow_logger import MLflowLogger
        _mlflow.set_tracking_uri(mlflow_uri)
        _mlflow.set_experiment("traits_audit_platforms")
        _run_ctx = _mlflow.start_run(run_name=run_name)
        _run_ctx.__enter__()
        _mlflow.log_params({
            "platform": "PyBAMM-SPM-MOBO", "n_seed": n_seed, "n_iter": n_iter,
            "n_query": n_query, "patience": patience, "min_improvement": min_improvement,
            "seed": seed, "check_every": check_every,
            "noise_std": noise_std, "noise_std_voltage": noise_std_voltage,
            "pink_frac": pink_frac,
        })
        _mlflow_logger = MLflowLogger()
    else:
        _mlflow_logger = None

    rng = np.random.default_rng(seed)

    def _sample_cube(n: int) -> np.ndarray:
        return rng.uniform(0.0, 1.0, size=(n, 3))

    model_pybam, param_base, Cn = _build_pybam()
    print(f"  Nominal capacity: {Cn:.4f} Ah")

    # ── Noise budget: precompute 1/f sequences long enough for the whole ─────
    # campaign (holdout + seed + max batches), consumed in chronological
    # observation order so temporally-close measurements share correlated
    # drift.
    # 60, not the old discrete-pool demo's 13: TailIndexCheck needs >= 50
    # samples for a Hill estimate and ScoreDecompositionCheck needs >= 30
    # for 10 bins, both evaluated on this held-out set -- with the smaller
    # holdout those two (and PITUniformityCheck, >= 20) always skipped for
    # lack of data rather than reporting anything. Growing the held-out set
    # is cheap (independent oracle evaluations, no effect on the AL/
    # replication budget) so this is the "report everything" fix for those
    # three checks specifically.
    n_holdout = 60
    _noise_budget = n_holdout + n_seed + n_iter * n_query + 16
    _pink_rng  = np.random.default_rng(seed + 5000)
    _pink_cap  = _make_pink_noise(_noise_budget, _pink_rng, target_std=noise_std * pink_frac)
    _pink_volt = _make_pink_noise(_noise_budget, _pink_rng, target_std=noise_std_voltage * pink_frac)
    _noise_i = 0

    def _observe(x_n: np.ndarray) -> tuple[float, float]:
        """Evaluate the oracle at normalised point ``x_n`` = (C-rate,
        thickness scale, particle scale). Draws a random ambient
        temperature (uncontrolled noise source, see module docstring) and
        adds Gaussian + 1/f observation noise. Returns ``(y_cap, y_volt)``.
        """
        nonlocal _noise_i
        c_rate, thick, part = _denorm(x_n)
        T_amb = float(rng.uniform(_T_MIN, _T_MAX))
        cap_true, volt_true = _simulate_observables(
            c_rate, thick, part, T_amb, model_pybam, param_base, Cn,
        )
        i = min(_noise_i, len(_pink_cap) - 1)
        _noise_i += 1
        y_cap  = cap_true  + rng.normal(0, noise_std)         + _pink_cap[i]
        y_volt = volt_true + rng.normal(0, noise_std_voltage) + _pink_volt[i]
        return y_cap, y_volt

    # ── Held-out evaluation set (fixed continuous points, never trained on) ──
    X_hold_n = _sample_cube(n_holdout)
    _hold = [_observe(x_n) for x_n in X_hold_n]
    y_hold_cap  = np.array([c for c, _ in _hold])
    np.array([v for _, v in _hold])
    print(f"  Held out {n_holdout} continuous points for final audit")

    # ── Phase 1: random seed evaluations ─────────────────────────────────────
    print(f"\n[1/3] Seed — {n_seed} random evaluations …")
    X_obs_n = _sample_cube(n_seed)
    _seed_obs = [_observe(x_n) for x_n in X_obs_n]
    y_obs_cap  = np.array([c for c, _ in _seed_obs])
    y_obs_volt = np.array([v for _, v in _seed_obs])

    # ── Phase 2: continuous batch qLogNEHVI, auto-stop on convergence ────────
    print(f"\n[2/3] qLogNEHVI active learning — up to {n_iter} batches of "
          f"{n_query} (auto-stop when hypervolume gain < {min_improvement:.1%} "
          f"over {patience} batches) …")

    train_X = torch.tensor(X_obs_n, dtype=torch.float64)
    train_Y = torch.tensor(
        np.column_stack([y_obs_cap, y_obs_volt]), dtype=torch.float64,
    )
    model = _fit_model(train_X, train_Y)

    hook = _make_pipeline(logger=_mlflow_logger)
    pipeline = hook._pipeline

    uncertainties: list[float]         = []
    unc_volt: list[float]              = []
    epistemic_uncertainties: list[float] = []
    epistemic_unc_volt: list[float]   = []
    queried_n: list[np.ndarray]        = []
    hypervolume_history: list[float]   = []
    batch_end_hv: list[float]          = []

    _rrr_k = 5
    _pending_rrr: list[dict] = []
    _rrr_claimed, _rrr_before, _rrr_after = [], [], []

    # ── Replication arm: 3 fixed anchors, revisited on a rotating schedule ────
    # as one of the n_query slots in every batch (real experimental QC
    # practice: interleave repeat measurements of a reference condition with
    # new exploration, rather than a separate all-at-once burst after the
    # campaign ends), so RSE/DUG/AFC accumulate genuine, growing data
    # throughout the campaign instead of only existing as a single post-hoc
    # number.
    _N_ANCHORS = 3
    anchor_n = _sample_cube(_N_ANCHORS)  # fixed continuous anchor locations

    replicate_groups: dict[str, dict] = {
        f"anchor{k}": {"y_true": [], "y_pred_mean": [], "y_pred_std": []}
        for k in range(_N_ANCHORS)
    }

    n_points_total = 0
    stage_reports: list[tuple[str, object]] = []
    stop_reason: str | None = None
    batch = -1

    for batch in range(n_iter):
        n_new = n_query - 1 if n_query > 1 else 1
        xi_n_batch = _best_batch_continuous(model, train_X, q=n_new)

        anchor_k = batch % _N_ANCHORS
        batch_points = [(xi_n_batch[j], False, None) for j in range(len(xi_n_batch))]
        batch_points.append((anchor_n[anchor_k], True, anchor_k))

        for x_n_raw, is_replicate, anchor_key in batch_points:
            x_n = np.asarray(x_n_raw, dtype=float)
            y_cap, y_volt = _observe(x_n)

            # GP predictions at the queried point (from the batch's shared
            # pre-batch model state — refit happens once per batch, below).
            mean_q, std_q = _predict(model, x_n)
            mu_cap_q,  sigma_cap_q  = float(mean_q[0]), float(std_q[0])
            _mu_volt_q, sigma_volt_q = float(mean_q[1]), float(std_q[1])

            # Epistemic std: subtract the fitted likelihood noise from total variance.
            # With BoTorch SingleTaskGP, the likelihood noise is model.models[i].likelihood.noise
            try:
                noise_cap  = float(model.models[0].likelihood.noise.sqrt())
                noise_volt = float(model.models[1].likelihood.noise.sqrt())
            except Exception:
                noise_cap  = noise_std
                noise_volt = noise_std_voltage
            ep_cap  = float(np.sqrt(max(sigma_cap_q  ** 2 - noise_cap  ** 2, 0.0)))
            ep_volt = float(np.sqrt(max(sigma_volt_q ** 2 - noise_volt ** 2, 0.0)))

            n_points_total += 1
            _pending_rrr.append({
                "due": n_points_total + _rrr_k, "x_q": x_n.copy(),
                "claimed": ep_cap ** 2, "before": sigma_cap_q ** 2,
            })

            if is_replicate:
                key = f"anchor{anchor_key}"
                replicate_groups[key]["y_true"].append(float(y_cap))
                replicate_groups[key]["y_pred_mean"].append(mu_cap_q)
                replicate_groups[key]["y_pred_std"].append(sigma_cap_q)

            # Update training data (model refit once per batch, not per point)
            X_obs_n    = np.vstack([X_obs_n, x_n])
            y_obs_cap  = np.append(y_obs_cap,  y_cap)
            y_obs_volt = np.append(y_obs_volt, y_volt)

            hv = _hypervolume(np.column_stack([y_obs_cap, y_obs_volt]))
            hypervolume_history.append(hv)

            uncertainties.append(sigma_cap_q)
            unc_volt.append(sigma_volt_q)
            epistemic_uncertainties.append(ep_cap)
            epistemic_unc_volt.append(ep_volt)
            queried_n.append(x_n.copy())

            hook.on_step(
                y_true=y_cap,
                y_pred_mean=mu_cap_q,
                y_pred_std=sigma_cap_q,
                uncertainty=sigma_cap_q,
                uncertainty_volt=sigma_volt_q,
                abs_error=abs(y_cap - mu_cap_q),
                dataset_size=float(len(X_obs_n)),
                hypervolume=hv,
                is_replicate=is_replicate,
            )

            # RRR bookkeeping (recompute against the current model; may lag
            # by up to one batch since refits now happen per batch)
            _still_pending = []
            for _rec in _pending_rrr:
                if _rec["due"] <= n_points_total:
                    _after_mean, _after_std = _predict(model, _rec["x_q"])
                    _rrr_claimed.append(_rec["claimed"])
                    _rrr_before.append(_rec["before"])
                    _rrr_after.append(float(_after_std[0]) ** 2)
                else:
                    _still_pending.append(_rec)
            _pending_rrr = _still_pending

            if check_every and n_points_total % check_every == 0:
                snap_groups = {
                    k: v for k, v in replicate_groups.items()
                    if len(v["y_true"]) >= 2
                }
                snap = pipeline.run(hook.history, replicate_groups=snap_groups)
                stage_reports.append((f"point {n_points_total}", snap))

        # Refit the model once per batch — the standard batch-AL pattern:
        # one model state proposes a whole batch, the batch is observed,
        # then the model is refit once (matching ta-camd-demo).
        train_X = torch.tensor(X_obs_n, dtype=torch.float64)
        train_Y = torch.tensor(
            np.column_stack([y_obs_cap, y_obs_volt]), dtype=torch.float64,
        )
        model = _fit_model(train_X, train_Y)

        batch_end_hv.append(hypervolume_history[-1])
        if (batch + 1) % 5 == 0 or batch == n_iter - 1:
            print(f"  Batch {batch + 1}/{n_iter}: {len(batch_points)} pts "
                  f"({len(xi_n_batch)} new + 1 replicate)  "
                  f"HV={hypervolume_history[-1]:.4f}")

        # ── Auto-stop: campaign quit improving ──────────────────────────────
        if len(batch_end_hv) > patience:
            base = batch_end_hv[-(patience + 1)]
            cur  = batch_end_hv[-1]
            rel_impr = (cur - base) / max(abs(base), 1e-9)
            if rel_impr < min_improvement:
                stop_reason = (
                    f"hypervolume improved only {rel_impr:.2%} over the last "
                    f"{patience} batches (< {min_improvement:.1%} target)"
                )
                print(f"  Converged after batch {batch + 1}: {stop_reason}")
                break

    if stop_reason is None:
        stop_reason = f"reached max batches ({n_iter})"
    n_batches_run = batch + 1

    n_al_steps    = len(uncertainties)
    n_seed_actual = n_seed

    best_i = int(np.argmax(y_obs_cap))
    c_best, ts_best, ps_best = _denorm(X_obs_n[best_i])
    print(f"\n  Stopped: {stop_reason}  ({n_batches_run} batches, {n_points_total} queried pts)")
    print(f"  Best capacity: (C={c_best:.2f}, thickness×{ts_best:.2f}, particle×{ps_best:.2f})  "
          f"cap={y_obs_cap[best_i]:.4f} Ah  "
          f"(dataset = {len(y_obs_cap)} pts)")
    print(f"  Final HV: {hypervolume_history[-1]:.4f}" if hypervolume_history else "")

    # ── Controllability-Gramian mechanism check ───────────────────────────────
    from traits_audit import trajectory as _traj
    from traits_audit._mechanism_check import print_mechanism_check

    unc_vectors = [
        np.array([ep_cap, ep_volt, noise_std, noise_std_voltage])
        for ep_cap, ep_volt in zip(epistemic_uncertainties, epistemic_unc_volt, strict=False)
    ]
    mech_rec    = _traj.from_pybamm(
        {"uncertainties": unc_vectors, "queried_n": queried_n},
        policy="qLogNEHVI",
    )
    mech_result = _traj.analyze_trajectory(mech_rec, n_components=6)
    print_mechanism_check(mech_result, "real split (capacity + voltage, known noise floors)",
                          aleatoric_indices=[2, 3])

    # ── Phase 3: Lyapunov stability analysis ──────────────────────────────────
    print("\n[3/3] Lyapunov stability analysis + final audit …")
    from traits_audit._viz import (
        check_grid_figures,
        plot_audit_evolution,
        plot_convergence,
        plot_pareto_frontier,
        plot_uncertainty_evolution,
        run_dmdc_lyapunov_analysis,
        run_lyapunov_analysis,
    )
    from traits_audit.checks.lyapunov import make_gd_predictor

    op_states = np.array(queried_n)

    def _neg_cap(state_3: np.ndarray) -> float:
        mu, _ = _predict(model, state_3)
        return -float(mu[0])   # negate: min(−cap) ≡ max(cap)

    def _cap_std(state_3: np.ndarray) -> float:
        _, std = _predict(model, state_3)
        return float(std[0])

    gd_pred = make_gd_predictor(_neg_cap, alpha=_LYAPUNOV_ALPHA)
    lyap    = run_lyapunov_analysis(
        predictor=gd_pred,
        op_states=op_states,
        gp_std_fn=_cap_std,
        model_label="BoTorch-MOBO (PyBAMM)",
        out_dir=fig_dir,
        dx=1e-3,
    )

    aug_states = np.column_stack([op_states, np.array(uncertainties)])
    try:
        dmdc_result = run_dmdc_lyapunov_analysis(
            aug_states=aug_states,
            model_label="BoTorch-MOBO (PyBAMM)",
            out_dir=fig_dir,
            n_components=min(4, aug_states.shape[1]),
            gp_std_seq=np.array(uncertainties),
        )
        rho_A = float(np.max(np.abs(dmdc_result["eigenvalues"])))
    except Exception as exc:
        print(f"  DMDc fit failed ({exc}); DMDcSpectralRadiusCheck will skip.")
        rho_A = None

    print(f"\n[replication] {len(replicate_groups)} anchors, "
          f"{[len(v['y_true']) for v in replicate_groups.values()]} replicates each")

    # ── Final audit ───────────────────────────────────────────────────────────
    X_hold_t = torch.tensor(X_hold_n, dtype=torch.float64)
    with torch.no_grad():
        post_hold = model.posterior(X_hold_t)
    mu_hold_cap  = post_hold.mean[:, 0].numpy()
    std_hold_cap = post_hold.variance[:, 0].sqrt().numpy()

    # UncertaintyAnomalyCheck compares recent behaviour against an earlier
    # baseline. Without an explicit historical_uncertainties kwarg it falls
    # back to z-scoring the series against its OWN mean/std, which makes
    # >3 sigma-of-itself excursions structurally near-impossible regardless
    # of seed/n_iter (a point that far from its own mean also inflates the
    # std it's being measured against) -- same convention as ta-camd-demo.
    n_warmup = max(check_every * 2, len(uncertainties) // 5, 1)

    report = hook.on_end(
        lambda_max=lyap["lambda_max"],
        rho_A=rho_A,
        replicate_groups=replicate_groups,
        claimed_epistemic_variance=np.array(_rrr_claimed) if _rrr_claimed else None,
        realized_total_variance_before=np.array(_rrr_before) if _rrr_before else None,
        realized_total_variance_after=np.array(_rrr_after) if _rrr_after else None,
        y_true=y_hold_cap, y_pred_mean=mu_hold_cap, y_pred_std=std_hold_cap,
        historical_uncertainties=np.array(uncertainties[:n_warmup]),
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

    if _use_mlflow:
        for r in report.results:
            label = "PASS" if r.passed else "FAIL"
            val   = f" ({r.value:.4f})" if r.value is not None else ""
            _mlflow.set_tag(f"audit_verdict/{r.name}", f"{label}{val}")
        _mlflow.set_tag("audit_verdict/overall", "PASS" if report.passed else "FAIL")
        _mlflow.log_artifact(str(report_path), "audit")

    # ── Figures ───────────────────────────────────────────────────────────────
    # stage_reports was built manually in the loop above (one snapshot every
    # check_every points, with the growing replicate_groups passed each
    # time) rather than via hook.intermediate_reports, since the hook's own
    # auto-trigger never passes kwargs -- exactly why RSE/DUG/AFC could
    # never have real intermediate values before.
    if stage_reports:
        all_stage_reports = [*stage_reports, ("final", report)]
        fig_grid, fig_grid_final = check_grid_figures(all_stage_reports, "BoTorch-MOBO (PyBAMM)")
        for fname, fig in [
            ("check_grid_pybamm.png",            fig_grid),
            ("check_grid_pybamm_final_only.png",  fig_grid_final),
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

        # Pairwise Spearman correlations between check values across
        # snapshots (same figure as ta-cal-demo's metric_correlations_*.png).
        import matplotlib.pyplot as _plt_corr

        from traits_audit._viz import _fig_metric_correlations
        fig_corr = _fig_metric_correlations([r for _, r in all_stage_reports], "BoTorch-MOBO (PyBAMM)")
        if fig_corr is not None:
            try:
                fig_corr.savefig(str(fig_dir / "metric_correlations_pybamm.png"), dpi=300, bbox_inches="tight")
                _plt_corr.close(fig_corr)
                print("  Saved metric_correlations_pybamm.png")
            except Exception:
                pass

    plot_uncertainty_evolution(
        np.array(uncertainties),
        model_label="BoTorch-MOBO (PyBAMM)",
        out_dir=fig_dir,
    )
    plot_audit_evolution(
        pipeline=hook._pipeline,
        history=hook.history,
        model_label="BoTorch-MOBO (PyBAMM)",
        out_dir=fig_dir,
        snapshot_every=4,
    )

    # Pareto frontier: capacity × voltage (both queried and seed points)
    all_cap  = y_obs_cap
    all_volt = y_obs_volt
    plot_pareto_frontier(
        x_vals=all_cap,
        y_vals=all_volt,
        x_label="Discharge capacity (Ah)",
        y_label="Mean terminal voltage (V)",
        model_label="BoTorch-MOBO (PyBAMM)",
        out_dir=fig_dir,
        minimize_x=False,
        minimize_y=False,
        color_vals=np.concatenate([
            np.full(n_seed_actual, -1.0),
            np.arange(n_al_steps, dtype=float),
        ]),
        color_label="qNEHVI step (seed = −1)",
    )

    # Hypervolume convergence
    if hypervolume_history:
        plot_convergence(
            best_vals=np.array(hypervolume_history),
            query_counts=np.arange(1, len(hypervolume_history) + 1),
            y_label="Hypervolume (cap × volt)",
            model_label="BoTorch-MOBO (PyBAMM)",
            out_dir=fig_dir,
            maximise=True,
        )

    # Best-capacity convergence
    best_cap = np.maximum.accumulate(y_obs_cap)
    plot_convergence(
        best_vals=best_cap,
        query_counts=np.arange(1, len(best_cap) + 1),
        y_label="Best discharge capacity (Ah)",
        model_label="BoTorch-MOBO (PyBAMM)",
        out_dir=fig_dir,
        maximise=True,
        fig_title="convergence_capacity",
    )

    if _use_mlflow:
        lm = lyap["lambda_max"]
        _mlflow.log_metrics({
            "mobo/final_hypervolume": hypervolume_history[-1] if hypervolume_history else 0.0,
            "mobo/n_batches_run": n_batches_run,
            "lyapunov/lambda_max_mean": float(lm.mean()),
            "lyapunov/n_stable":        int((lm < 1.0).sum()),
        })
        if (fig_dir / "lyapunov_stability.csv").exists():
            _mlflow.log_artifact(str(fig_dir / "lyapunov_stability.csv"), "lyapunov")
        _mlflow.log_artifacts(str(fig_dir), "figures")
        _run_ctx.__exit__(None, None, None)

    print(f"\nDone. All results written to {out_dir}")
    return {"report": report, "lyapunov": lyap, "mechanism_check": mech_result,
            "hypervolume_history": hypervolume_history, "stop_reason": stop_reason}


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--n-seed",      type=int,   default=10,
                   help="Random seed evaluations (default: 10)")
    p.add_argument("--n-iter",      type=int,   default=30,
                   help="MAX qLogNEHVI batches; may stop earlier on "
                        "convergence (default: 30)")
    p.add_argument("--n-query",     type=int,   default=4,
                   help="Points queried per batch; one slot rotates through "
                        "the replicate anchors (default: 4)")
    p.add_argument("--patience",    type=int,   default=4,
                   help="Trailing batches over which hypervolume gain is "
                        "measured for auto-stop (default: 4)")
    p.add_argument("--min-improvement", type=float, default=0.01,
                   help="Relative hypervolume improvement threshold for "
                        "auto-stop (default: 0.01 = 1%%)")
    p.add_argument("--out-dir",     type=str,   default="_results/pybamm_demo")
    p.add_argument("--seed",        type=int,   default=0)
    p.add_argument("--check-every", type=int,   default=1,
                   help="Intermediate audit frequency (default: 1)")
    p.add_argument("--noise-std",   type=float, default=0.003,
                   help="Observation noise std [Ah] (default: 0.003)")
    p.add_argument("--noise-std-voltage", type=float, default=0.005,
                   help="Observation noise std on voltage [V] (default: 0.005)")
    p.add_argument("--pink-frac",   type=float, default=0.8,
                   help="1/f noise std as a fraction of noise_std/"
                        "noise_std_voltage; 0 disables it (default: 0.8)")
    default_uri = "sqlite:///" + str(Path.cwd() / "traits_audit_demo.db")
    p.add_argument("--mlflow-uri",  type=str,   default=default_uri)
    p.add_argument("--run-name",    type=str,   default="pybamm_demo")
    p.add_argument("--ui",          action="store_true",
                   help="Launch the MLflow UI after the run")
    return p


def main() -> None:
    args = build_parser().parse_args()
    run(
        n_seed=args.n_seed,
        n_iter=args.n_iter,
        n_query=args.n_query,
        patience=args.patience,
        min_improvement=args.min_improvement,
        out_dir=Path(args.out_dir),
        seed=args.seed,
        check_every=args.check_every,
        noise_std=args.noise_std,
        noise_std_voltage=args.noise_std_voltage,
        pink_frac=args.pink_frac,
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
