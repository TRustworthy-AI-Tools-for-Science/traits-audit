"""traits_audit demo — PyBAM Li-ion cell C-rate / temperature optimisation.

Active-learning loop that finds the (charge-rate, temperature) operating point
maximising discharge capacity in a lithium-ion cell.  The oracle is PyBAM's
fast Single Particle Model (SPM) — no hardware required.

Domain
------
State:       2-D normalised  [c_rate_norm, T_norm]  ∈  [0, 1]²
C-rate:      0.5 C – 3.0 C
Temperature: 10 °C – 40 °C
Oracle:      PyBAM SPM single discharge → (capacity [Ah], mean terminal
             voltage [V]) — both pulled from the same solve at no extra cost.
Surrogate:   sklearn GaussianProcessRegressor  (RBF + WhiteKernel), one per
             observable; only the capacity GP drives acquisition, the voltage
             GP is tracked only.
Policy:      UCB  (κ = 2.0)  →  maximise predicted capacity
Audit:       18 uncertainty checks via AuditHook / AuditPipeline (11 total-
             distribution checks + Lyapunov/DMDc pair + 3 cross-cutting +
             3 replication-arm checks -- see _make_pipeline())
Stability:   Lyapunov (local) + DMDc rho(A) (global) on the gradient-descent
             map / query trajectory of the surrogate
Mechanism:   controllability-Gramian check on [σ_ep_cap, σ_ep_volt, σ_al_cap,
             σ_al_volt] — see ``--mechanism-null`` for the E6-style null.
             sigma_ep here is the GP's total predictive std with its OWN
             fitted WhiteKernel noise level subtracted, not the raw total
             (the raw total double-counts against the fixed sigma_al floor
             columns [2, 3] below).
Replication: RSE/DUG/AFC on a small set of dedicated replicate locations
             (see run(): capacity noise is added AFTER the (expensive) SPM
             solve, so replicates only cost cheap extra noise draws, not
             repeated solves).

Dependencies
------------
Required:   pybamm, scikit-learn  (both have no impact on the
            ``battery_forecast`` package — no circular imports).

Entry point::

    ta-pybamm-demo [OPTIONS]
    ta-pybamm-demo --n-iter 25 --out-dir _results/pybamm_demo --seed 7
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ── Physical bounds ────────────────────────────────────────────────────────────
_C_MIN, _C_MAX = 0.5, 3.0     # C-rate  [C]
_T_MIN, _T_MAX = 10.0, 40.0   # Temperature [°C]


def _norm(c_rate: float, T_C: float) -> np.ndarray:
    return np.array([
        (c_rate - _C_MIN) / (_C_MAX - _C_MIN),
        (T_C    - _T_MIN) / (_T_MAX - _T_MIN),
    ])


def _denorm(state_2: np.ndarray) -> tuple[float, float]:
    c = state_2[0] * (_C_MAX - _C_MIN) + _C_MIN
    T = state_2[1] * (_T_MAX - _T_MIN) + _T_MIN
    return float(c), float(T)


# ── PyBAM oracle ───────────────────────────────────────────────────────────────

def _build_pybam():
    """Initialise the SPM model and base parameter set once."""
    import pybamm
    pybamm.set_logging_level("WARNING")
    model = pybamm.lithium_ion.SPM()
    param = model.default_parameter_values.copy()
    Cn    = float(param["Nominal cell capacity [A.h]"])
    return model, param, Cn


def _simulate_observables(c_rate: float, T_C: float, model, param, Cn: float) -> tuple[float, float]:
    """Run one SPM discharge at ``c_rate`` C and ``T_C`` °C.

    Returns ``(capacity, mean_voltage)``: the total discharge capacity [Ah]
    (the optimisation objective) and the mean terminal voltage [V] over the
    discharge — a second, physically distinct observable pulled from the
    *same* solve at no extra oracle cost. ``mean_voltage`` is tracked (for the
    uncertainty Gramian) but never optimised; the UCB acquisition below still
    targets capacity alone.
    A maximum simulation time of 2 h prevents solver stalls at very low rates.
    """
    import pybamm
    p = param.copy()
    p["Current function [A]"]  = c_rate * Cn
    p["Ambient temperature [K]"] = 273.15 + T_C
    p["Initial temperature [K]"] = 273.15 + T_C
    sim   = pybamm.Simulation(model, parameter_values=p)
    t_end = min(3600.0 / c_rate, 7200.0)
    sol   = sim.solve([0, t_end])
    capacity = float(sol["Discharge capacity [A.h]"].entries[-1])
    mean_voltage = float(np.mean(sol["Terminal voltage [V]"].entries))
    return capacity, mean_voltage


# ── Audit pipeline ─────────────────────────────────────────────────────────────

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
        ReplicationShrinkageExponentCheck,
        DarkUncertaintyGapCheck,
        AleatoricFloorConsistencyCheck,
        ReducibilityRealisationRatioCheck,
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
            # window=30: a LOCAL (recent-region) verdict, contrasted with the
            # CAMD/SDL demos' global/cumulative default (window=None) — see
            # docs/checks.rst and LYAPUNOV_ANALYSIS.md for the local/global
            # distinction. Precomputed lambda_max route (see run(), below).
            LyapunovStabilityCheck(stability_threshold=1.0, min_stable_fraction=0.5, window=30),
            # Paired with LyapunovStabilityCheck (ergodic/non-ergodic; local
            # vs global) -- see validation.NAME_PAIRS.
            DMDcSpectralRadiusCheck(stability_threshold=1.0),
            # Cross-cutting, cheap, read from the per-step history like the
            # checks above.
            SignedBiasCheck(),
            TailIndexCheck(),
            ScoreDecompositionCheck(),
            # Replication arm (see run(): capacity noise is added AFTER an
            # expensive SPM solve, so replicates only cost cheap noise
            # draws, not repeated solves -- PyBaMM is the one demo besides
            # ta-demo where these are honestly supportable, since noise_std
            # is a real, exactly-known observation-noise model rather than
            # a synthesised one).
            ReplicationShrinkageExponentCheck(),
            DarkUncertaintyGapCheck(),
            AleatoricFloorConsistencyCheck(),
            # Paired with AleatoricFloorConsistency (aleatoric/epistemic
            # split) -- see validation.NAME_PAIRS.
            ReducibilityRealisationRatioCheck(),
        ],
        verbose=False,
    )
    return AuditHook(pipeline, check_every=check_every, logger=logger)


# ── Main run ───────────────────────────────────────────────────────────────────

def run(
    n_seed: int = 8,
    n_iter: int = 20,
    out_dir: Path = Path("_results/pybamm_demo"),
    seed: int = 0,
    check_every: int = 5,
    noise_std: float = 0.003,
    noise_std_voltage: float = 0.005,
    kappa: float = 2.0,
    mechanism_null: bool = False,
    mlflow_uri: str | None = None,
    run_name: str = "pybamm_demo",
) -> dict:
    """Run the PyBAM C-rate / temperature optimisation demo.

    Parameters
    ----------
    n_seed : int
        Random seed evaluations before the UCB loop starts.
    n_iter : int
        UCB active-learning iterations.
    out_dir : Path
        Root directory for results.
    seed : int
        RNG seed.
    check_every : int
        Intermediate audit frequency.
    noise_std : float
        Additive Gaussian observation noise on discharge capacity [Ah].
        A value of 0.003 Ah ≈ 0.4 % of nominal — realistic for lab variability.
        Also used as the fixed aleatoric floor for capacity in the
        controllability-Gramian mechanism check (see module docstring).
    noise_std_voltage : float
        Additive Gaussian observation noise on the tracked mean terminal
        voltage [V] (default 0.005 V ≈ 5 mV, a realistic voltmeter floor).
        Used as the fixed aleatoric floor for voltage in the mechanism check.
    kappa : float
        UCB exploration-exploitation trade-off (higher → more exploration).
    mechanism_null : bool
        If True, replace the real ``[sigma_ep_cap, sigma_ep_volt, sigma_al_cap,
        sigma_al_volt]`` mechanism check with the E6-style two-epistemic null:
        a second, independently-seeded capacity GP alongside the primary one,
        giving ``[sigma_ep_cap_run1, sigma_ep_cap_run2]`` — both reducible, so
        the eigenvalue ratio should collapse toward the null band rather than
        the real split's separation. See ``paper1_logical_pitfalls.md``
        Category 5. Run the demo once with this False and once True to
        compare.
    mlflow_uri : str | None
        MLflow tracking URI.  ``None`` → no logging.
    run_name : str
        MLflow run name.
    """
    import os, warnings
    os.environ.setdefault("JAX_PLATFORMS", "cpu")  # suppress JAX GPU warning
    warnings.filterwarnings("ignore", category=UserWarning,
                            module="sklearn.gaussian_process")

    try:
        import pybamm as _pybamm  # noqa: F401
    except ImportError:
        print("ERROR: pybamm is not installed.  pip install pybamm")
        sys.exit(1)

    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        RBF, WhiteKernel, ConstantKernel,
    )

    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("PyBAM Demo: Li-ion SPM  C-rate × temperature optimisation")
    print(f"  n_seed={n_seed}  n_iter={n_iter}  seed={seed}  κ={kappa}")
    print(f"  C-rate ∈ [{_C_MIN}, {_C_MAX}] C    T ∈ [{_T_MIN}, {_T_MAX}] °C")
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
            "platform":    "PyBAM-SPM",
            "n_seed":      n_seed,
            "n_iter":      n_iter,
            "seed":        seed,
            "check_every": check_every,
            "noise_std":   noise_std,
            "kappa":       kappa,
            "c_range":     f"[{_C_MIN}, {_C_MAX}]",
            "T_range":     f"[{_T_MIN}, {_T_MAX}]",
        })
        _mlflow.set_tags({
            "platform":    "PyBAM",
            "model":       "sklearn-GPR (RBF+White)",
            "acquisition": f"UCB kappa={kappa}",
            "simulation":  "True",
        })
        _mlflow_logger = MLflowLogger()
    else:
        _mlflow_logger = None

    rng = np.random.default_rng(seed)

    # ── Candidate pool: 10 C-rates × 8 temps = 80 points ─────────────────────
    c_vals   = np.linspace(_C_MIN, _C_MAX, 10)
    T_vals   = np.linspace(_T_MIN, _T_MAX, 8)
    X_pool   = np.array([[c, T] for c in c_vals for T in T_vals])     # (80, 2) raw
    X_pool_n = np.array([_norm(c, T) for c, T in X_pool])             # (80, 2) normed

    model_pybam, param_base, Cn = _build_pybam()
    print(f"  Nominal capacity: {Cn:.4f} Ah")

    # ── Held-out evaluation set ────────────────────────────────────────────────
    # Carved out of the fixed 80-point pool now, before any seed or UCB point
    # is chosen, and excluded from both from here on -- so the final GPR's
    # calibration/coverage/scoring checks are judged on points it never
    # trained on, rather than falling back to the per-step UCB-queried
    # history (an acquisition-biased sample: UCB deliberately over-selects
    # high-mean/high-uncertainty regions, which is not what a calibration
    # verdict should be evaluated against).
    n_holdout = min(15, max(5, len(X_pool) // 6))
    holdout_idx = rng.choice(len(X_pool), size=n_holdout, replace=False)
    holdout_mask = np.zeros(len(X_pool), dtype=bool)
    holdout_mask[holdout_idx] = True
    X_hold_n = X_pool_n[holdout_mask].copy()
    X_hold_r = X_pool[holdout_mask].copy()
    _hold_obs = [
        _simulate_observables(c, T, model_pybam, param_base, Cn)
        for c, T in X_hold_r
    ]
    y_hold = np.array([cap + rng.normal(0, noise_std) for cap, _ in _hold_obs])
    print(f"  Held out {n_holdout} of {len(X_pool)} pool points for final "
          "calibration evaluation (never queried)")

    _eligible_idx = np.where(~holdout_mask)[0]   # pool points available for seed/UCB

    # ── Phase 1: random seed evaluations (no GPR) ─────────────────────────────
    print(f"\n[1/3] Seed — {n_seed} random evaluations …")
    seed_idx = rng.choice(_eligible_idx, size=min(n_seed, len(_eligible_idx) // 4), replace=False)
    X_obs_n  = X_pool_n[seed_idx].copy()
    X_obs_r  = X_pool[seed_idx].copy()    # raw coords for reporting
    # One _simulate_observables call per point gives both observables at no
    # extra oracle cost (module docstring / demo docstring).
    _seed_obs = [
        _simulate_observables(c, T, model_pybam, param_base, Cn)
        for c, T in X_pool[seed_idx]
    ]
    y_obs      = np.array([cap + rng.normal(0, noise_std) for cap, _ in _seed_obs])
    y_obs_volt = np.array([vlt + rng.normal(0, noise_std_voltage) for _, vlt in _seed_obs])

    remaining = np.ones(len(X_pool), dtype=bool)
    remaining[seed_idx] = False
    remaining[holdout_mask] = False    # held-out points are never UCB-eligible either

    # ── GPR definition ─────────────────────────────────────────────────────────
    # Wide bounds avoid ConvergenceWarnings at the edges of the search space.
    # The capacity landscape spans ~3 % (0.025 Ah) over the 2-D space;
    # normalize_y=True maps this to O(1) before kernel fitting.
    def _make_kernel(noise_level: float):
        return (
            ConstantKernel(
                constant_value=0.1,
                constant_value_bounds=(1e-6, 100.0),
            )
            * RBF(
                length_scale=[0.4, 0.4],
                length_scale_bounds=(1e-3, 10.0),
            )
            + WhiteKernel(
                noise_level=noise_level,
                noise_level_bounds=(1e-10, 1.0),
            )
        )

    gpr = GaussianProcessRegressor(
        kernel=_make_kernel(noise_std ** 2), n_restarts_optimizer=3, normalize_y=True,
        random_state=int(seed),
    )

    # ── Second surrogate for the controllability-Gramian mechanism check ──────
    # Real run: an independent GP on the mean terminal voltage (a genuinely
    # distinct observable from the same solve). Null run (--mechanism-null):
    # a second, differently-seeded GP on the SAME capacity target -- both
    # equally reducible, the E6-style falsification test (see run() docstring
    # and paper1_logical_pitfalls.md Category 5). Either way this second model
    # is tracked only; it never influences the UCB acquisition on capacity.
    if mechanism_null:
        gpr_second = GaussianProcessRegressor(
            kernel=_make_kernel(noise_std ** 2), n_restarts_optimizer=3, normalize_y=True,
            random_state=int(seed) + 1000,
        )
        y_obs_second = y_obs
    else:
        gpr_second = GaussianProcessRegressor(
            kernel=_make_kernel(noise_std_voltage ** 2), n_restarts_optimizer=3, normalize_y=True,
            random_state=int(seed),
        )
        y_obs_second = y_obs_volt

    hook = _make_pipeline(check_every, logger=_mlflow_logger)

    uncertainties: list[float]    = []
    unc_second:   list[float]    = []   # second surrogate's posterior std per step
    epistemic_uncertainties: list[float] = []  # sigma_q with fitted WhiteKernel noise removed
    epistemic_unc_second:    list[float] = []
    queried_n:    list[np.ndarray] = []   # AL-queried normalised coords

    # RRR bookkeeping (paired with AleatoricFloorConsistency): claim the
    # epistemic variance at the point about to be queried, then rrr_k steps
    # later -- once the GP has actually been refit on the intervening data
    # -- measure the realized drop in total predictive variance there.
    _rrr_k = 5
    _pending_rrr: list[dict] = []
    _rrr_claimed, _rrr_before, _rrr_after = [], [], []

    # ── Phase 2: UCB active-learning loop ─────────────────────────────────────
    print(f"\n[2/3] UCB active learning — {n_iter} iterations …")
    for step in range(n_iter):
        if not remaining.any():
            print(f"  Pool exhausted — stopping after {step} iteration(s).")
            break

        gpr.fit(X_obs_n, y_obs)
        gpr_second.fit(X_obs_n, y_obs_second)

        X_cand_n = X_pool_n[remaining]
        mu_cand, sigma_cand = gpr.predict(X_cand_n, return_std=True)
        acq = mu_cand + kappa * sigma_cand          # UCB: maximise capacity
        # (gpr_second never enters the acquisition score -- it's tracked only.)

        best_local  = int(np.argmax(acq))
        pool_idx    = np.where(remaining)[0]
        best_global = pool_idx[best_local]

        xi_n = X_pool_n[best_global]
        xi_r = X_pool[best_global]
        c_q, T_q = float(xi_r[0]), float(xi_r[1])

        cap_true, volt_true = _simulate_observables(c_q, T_q, model_pybam, param_base, Cn)
        y_true = cap_true + rng.normal(0, noise_std)
        y_true_second = (
            cap_true + rng.normal(0, noise_std) if mechanism_null
            else volt_true + rng.normal(0, noise_std_voltage)
        )

        # GPR prediction at the queried point (before incorporating it)
        mu_q, sigma_q = gpr.predict(xi_n.reshape(1, -1), return_std=True)
        mu_q    = float(mu_q[0])
        sigma_q = float(sigma_q[0])
        _, sigma_second_q = gpr_second.predict(xi_n.reshape(1, -1), return_std=True)
        sigma_second_q = float(sigma_second_q[0])

        # Epistemic-only channels for the mechanism check: sigma_q above is
        # the GP's TOTAL predictive std, which already includes whatever the
        # kernel's own WhiteKernel component learned as its noise level --
        # using it as unc_vectors' "epistemic" column double-counts against
        # the fixed noise_std/noise_std_voltage aleatoric-floor columns
        # below. Subtract the FITTED (not the nominal) noise level, since
        # MLE is free to move it away from its noise_level_bounds-constrained
        # initial value.
        noise_level_cap = float(gpr.kernel_.k2.noise_level)
        noise_level_second = float(gpr_second.kernel_.k2.noise_level)
        ep_std_q = float(np.sqrt(max(sigma_q ** 2 - noise_level_cap, 0.0)))
        ep_std_second_q = float(np.sqrt(max(sigma_second_q ** 2 - noise_level_second, 0.0)))

        _pending_rrr.append({
            "due": step + _rrr_k, "x_q": xi_n.copy(),
            "claimed": ep_std_q ** 2, "before": sigma_q ** 2,
        })

        X_obs_n = np.vstack([X_obs_n, xi_n])
        X_obs_r = np.vstack([X_obs_r, xi_r])
        y_obs   = np.append(y_obs, y_true)
        y_obs_second = np.append(y_obs_second, y_true_second)
        remaining[best_global] = False

        uncertainties.append(sigma_q)
        unc_second.append(sigma_second_q)
        epistemic_uncertainties.append(ep_std_q)
        epistemic_unc_second.append(ep_std_second_q)
        queried_n.append(xi_n.copy())

        hook.on_step(
            y_true=y_true,
            y_pred_mean=mu_q,
            y_pred_std=sigma_q,
            uncertainty=sigma_q,
            abs_error=abs(y_true - mu_q),
            acquisition_score=float(acq[best_local]),
            dataset_size=float(len(X_obs_n)),
        )

        _still_pending = []
        for _rec in _pending_rrr:
            if _rec["due"] <= step:
                # gpr has already been refit on X_obs_n/y_obs through this
                # step (top of the loop, next iteration) -- refit once more
                # here so "after" reflects the state as of step _rec["due"].
                gpr.fit(X_obs_n, y_obs)
                _, _after_sig = gpr.predict(_rec["x_q"].reshape(1, -1), return_std=True)
                _rrr_claimed.append(_rec["claimed"])
                _rrr_before.append(_rec["before"])
                _rrr_after.append(float(_after_sig[0]) ** 2)
            else:
                _still_pending.append(_rec)
        _pending_rrr = _still_pending

        if (step + 1) % 5 == 0:
            print(f"  Step {step + 1}/{n_iter}: "
                  f"({c_q:.2f} C, {T_q:.0f} °C)  "
                  f"cap={y_true:.4f} Ah  σ={sigma_q:.5f}")

    n_al_steps = len(uncertainties)  # actual AL steps run (may be < n_iter if pool exhausted)
    n_seed_actual = len(seed_idx)    # actual seed count (may be < n_seed, see seed_idx above)

    best_i = int(np.argmax(y_obs))
    c_best, T_best = float(X_obs_r[best_i, 0]), float(X_obs_r[best_i, 1])
    print(f"\n  Best found: ({c_best:.2f} C, {T_best:.0f} °C)  "
          f"cap={y_obs[best_i]:.4f} Ah  "
          f"(dataset = {len(y_obs)} pts)")

    # ── Controllability-Gramian mechanism check ───────────────────────────────
    # Real run: does the Gramian separate the reducible capacity-GP posterior
    # std from the fixed (capacity, voltage) noise floors? Null run
    # (--mechanism-null): two independently-seeded capacity GPs, both
    # reducible -- the ratio should collapse instead of separating. Action is
    # the genuine queried point (queried_n), not a placeholder.
    from traits_audit import trajectory as _traj
    from traits_audit._mechanism_check import print_mechanism_check

    if mechanism_null:
        unc_vectors = [
            np.array([ep, ep2]) for ep, ep2 in zip(epistemic_uncertainties, epistemic_unc_second)
        ]
        aleatoric_idx = None
        mech_label = "two-epistemic null (--mechanism-null)"
    else:
        unc_vectors = [
            np.array([ep_cap, ep_volt, noise_std, noise_std_voltage])
            for ep_cap, ep_volt in zip(epistemic_uncertainties, epistemic_unc_second)
        ]
        aleatoric_idx = [2, 3]
        mech_label = "real split (capacity + voltage, known noise floors)"

    mech_rec = _traj.from_pybamm(
        {"uncertainties": unc_vectors, "queried_n": queried_n},
        policy=f"UCB(kappa={kappa})",
    )
    mech_result = _traj.analyze_trajectory(mech_rec, n_components=6)
    print_mechanism_check(mech_result, mech_label, aleatoric_indices=aleatoric_idx)

    # ── Phase 3: Lyapunov stability analysis ──────────────────────────────────
    # Computed before hook.on_end() so LyapunovStabilityCheck (in the pipeline
    # above) can be given the real lambda_max series via the precomputed route.
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

    op_states = np.array(queried_n)   # (n_iter, 2)

    def _neg_cap(state_2: np.ndarray) -> float:
        mu, _ = gpr.predict(state_2.reshape(1, -1), return_std=True)
        return -float(mu[0])   # negate: min(−cap) ≡ max(cap)

    def _gpr_std(state_2: np.ndarray) -> float:
        _, std = gpr.predict(state_2.reshape(1, -1), return_std=True)
        return float(std[0])

    gd_pred = make_gd_predictor(_neg_cap, alpha=0.05)

    lyap = run_lyapunov_analysis(
        predictor=gd_pred,
        op_states=op_states,
        gp_std_fn=_gpr_std,
        model_label="sklearn-GPR (PyBAM)",
        out_dir=fig_dir,
        dx=1e-3,
    )

    # ── DMDc rho(A): the global counterpart to Lyapunov's local lambda_max,
    # fit on the same query trajectory (augmented with the reported sigma).
    from traits_audit import dmdc as _dm
    aug_states = np.column_stack([op_states, np.array(uncertainties)])
    try:
        _A_r, _, _ = _dm.fit_dmdc(aug_states, op_states, n_components=min(3, aug_states.shape[1]))
        rho_A = float(np.max(np.abs(np.linalg.eigvals(_A_r))))
    except Exception as exc:
        print(f"  DMDc fit failed ({exc}); DMDcSpectralRadiusCheck will skip.")
        rho_A = None

    # ── Replication arm (RSE, DUG, AFC) ───────────────────────────────────
    # noise_std is added AFTER the SPM solve (see module docstring), so each
    # location costs exactly one extra solve; the R noise draws are free.
    print("\n[replication] Building the replicate-measurement arm …")
    _N_REPLICATE_LOCATIONS = 8
    _REPLICATE_R = 64
    _rep_rng = np.random.default_rng(int(seed) + 9000)
    replicate_groups = {}
    for _ in range(_N_REPLICATE_LOCATIONS):
        c_r = float(_rep_rng.uniform(_C_MIN, _C_MAX))
        T_r = float(_rep_rng.uniform(_T_MIN, _T_MAX))
        cap_r, _volt_r = _simulate_observables(c_r, T_r, model_pybam, param_base, Cn)
        y_rep = cap_r + _rep_rng.normal(0, noise_std, _REPLICATE_R)
        xi_rep = _norm(c_r, T_r)
        sigma_rep = float(gpr.predict(xi_rep.reshape(1, -1), return_std=True)[1][0])
        replicate_groups[f"{c_r:.2f}C_{T_r:.0f}C"] = {
            "y_true": y_rep.tolist(),
            "y_pred_mean": [cap_r] * _REPLICATE_R,
            "y_pred_std": [sigma_rep] * _REPLICATE_R,
        }

    # Evaluate the FINAL fitted GPR on the held-out set carved out before the
    # loop started (never queried, never trained on) -- passed explicitly
    # below so it wins over the per-step history in `_require` (kwargs take
    # priority) for this final report only. Intermediate check_every
    # snapshots are unaffected and keep reading the growing UCB history via
    # on_step, since the held-out set is deliberately evaluated once, at the
    # end, with the fully-trained final GPR.
    mu_hold, sigma_hold = gpr.predict(X_hold_n, return_std=True)

    report = hook.on_end(
        lambda_max=lyap["lambda_max"],
        rho_A=rho_A,
        replicate_groups=replicate_groups,
        claimed_epistemic_variance=np.array(_rrr_claimed) if _rrr_claimed else None,
        realized_total_variance_before=np.array(_rrr_before) if _rrr_before else None,
        realized_total_variance_after=np.array(_rrr_after) if _rrr_after else None,
        y_true=y_hold, y_pred_mean=mu_hold, y_pred_std=sigma_hold,
    )

    # DecisionFlipRate needs y_pred_mean/y_pred_std for the CANDIDATE POOL
    # (the UCB decision surface), a different array than the per-step
    # calibration data hook.on_end just consumed -- run as a second, small
    # pipeline and merge results (same pattern as ta-demo).
    from traits_audit import AuditPipeline as _AuditPipeline
    from traits_audit.checks import DecisionFlipRateCheck as _DecisionFlipRateCheck
    mu_pool_final, sigma_pool_final = gpr.predict(X_pool_n, return_std=True)

    def _ucb_argmax(mu_arr):
        return int(np.argmax(mu_arr + kappa * sigma_pool_final))

    _dfr_pipeline = _AuditPipeline(checks=[_DecisionFlipRateCheck(seed=int(seed))])
    _dfr_report = _dfr_pipeline.run(
        [], decision_fn=_ucb_argmax, y_pred_mean=mu_pool_final, y_pred_std=sigma_pool_final,
    )
    report.results.extend(_dfr_report.results)

    print("\n" + report.summary())
    if report.metadata.get("pairing_warnings"):
        print("\n  Pairing warnings:")
        for w in report.metadata["pairing_warnings"]:
            print(f"    - {w}")

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

    plot_uncertainty_evolution(
        np.array(uncertainties),
        model_label="sklearn-GPR (PyBAM)",
        out_dir=fig_dir,
    )

    # Audit check grid: rows = checks, cols = pipeline stages (same pattern
    # as ta-demo/ta-camd-demo/ta-sdl-demo). Split into a dense, step-
    # trackable grid and a compact single-column grid for checks that only
    # ever produce a value in the final report (Lyapunov/DMDc trajectory
    # fits, the replication arm, RRR bookkeeping resolved late, and
    # DecisionFlipRate, which is merged into report.results only after
    # hook.on_end() returns) -- see check_grid_figures()'s docstring.
    from traits_audit._viz import check_grid_figures
    if hook.intermediate_reports:
        stage_reports = [
            (f"step {(i + 1) * check_every}", r)
            for i, r in enumerate(hook.intermediate_reports)
        ]
        stage_reports.append(("final", report))
        fig_grid, fig_grid_final = check_grid_figures(stage_reports, "sklearn-GPR (PyBAM)")
        if fig_grid is not None:
            try:
                fig_grid.write_image(
                    str(fig_dir / "check_grid_pybamm.png"),
                    width=fig_grid.layout.width, height=fig_grid.layout.height, scale=2,
                )
                print("  Saved check_grid_pybamm.png")
            except Exception:
                fig_grid.write_html(str(fig_dir / "check_grid_pybamm.html"))
                print("  Saved check_grid_pybamm.html (install kaleido for PNG export)")
        if fig_grid_final is not None:
            try:
                fig_grid_final.write_image(
                    str(fig_dir / "check_grid_pybamm_final_only.png"),
                    width=fig_grid_final.layout.width, height=fig_grid_final.layout.height, scale=2,
                )
                print("  Saved check_grid_pybamm_final_only.png")
            except Exception:
                fig_grid_final.write_html(str(fig_dir / "check_grid_pybamm_final_only.html"))
                print("  Saved check_grid_pybamm_final_only.html (install kaleido for PNG export)")

    # NOT plot_lyapunov_evolution here: lyap["lambda_max"] is the FINAL
    # fitted GP's Jacobian evaluated at every historical operating point --
    # a spatial scan of a static model, not a time series. Plotting it
    # against "AL step" dual-axed with the genuinely time-varying
    # uncertainties series (as CAMD and SDL correctly do, since their
    # lambda_max comes from a growing-prefix / rolling per-step fit) would
    # imply a temporal correlation that isn't there. run_lyapunov_analysis
    # already writes the honest non-temporal alternative,
    # fig3_stability_vs_unc.png (stability vs. uncertainty across operating
    # points, no step axis).

    plot_audit_evolution(
        pipeline=hook._pipeline,
        history=hook.history,
        model_label="sklearn-GPR (PyBAM)",
        out_dir=fig_dir,
        snapshot_every=4,
    )

    # Sliced by the ACTUAL seed count (n_seed_actual), not the requested
    # n_seed -- seed_idx is capped at min(n_seed, len(X_pool)//4), so passing
    # e.g. --n-seed 30 previously misaligned this against uncertainties
    # (length n_al_steps), corrupting the Pareto-frontier x/y pairing.
    cap_al = y_obs[n_seed_actual:]          # capacities for UCB-queried points only
    assert len(cap_al) == len(uncertainties), (
        f"cap_al/uncertainties length mismatch: {len(cap_al)} vs {len(uncertainties)}"
    )
    plot_pareto_frontier(
        x_vals=np.array(uncertainties),
        y_vals=cap_al,
        x_label="GPR posterior std (Ah)",
        y_label="Discharge capacity (Ah)",
        model_label="sklearn-GPR (PyBAM)",
        out_dir=fig_dir,
        minimize_x=True,
        minimize_y=False,                   # maximise capacity
        color_vals=np.arange(len(uncertainties)),
        color_label="UCB step",
    )

    best_cap = np.maximum.accumulate(y_obs)
    plot_convergence(
        best_vals=best_cap,
        query_counts=np.arange(1, len(best_cap) + 1),
        y_label="Best discharge capacity (Ah)",
        model_label="sklearn-GPR (PyBAM)",
        out_dir=fig_dir,
        maximise=True,
    )

    if _use_mlflow:
        lm = lyap["lambda_max"]
        _mlflow.log_metrics({
            "lyapunov/lambda_max_mean": float(lm.mean()),
            "lyapunov/lambda_max_max":  float(lm.max()),
            "lyapunov/n_stable":        int((lm < 1.0).sum()),
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
    p.add_argument("--n-seed",      type=int,   default=8,
                   help="Random seed evaluations (default: 8)")
    p.add_argument("--n-iter",      type=int,   default=30,
                   help="UCB AL iterations (default: 30). The candidate pool "
                        "is a fixed 10x8=80-point grid (see module docstring), "
                        "of which ~15 points are reserved up front as a "
                        "held-out calibration set and are never eligible for "
                        "seeding or querying; n_seed + n_iter must stay "
                        "comfortably under the remaining ~65, or the loop "
                        "silently stops early when the pool is exhausted "
                        "(a previous default of 250 always hit this).")
    p.add_argument("--out-dir",     type=str,   default="_results/pybamm_demo")
    p.add_argument("--seed",        type=int,   default=0)
    p.add_argument("--check-every", type=int,   default=5,
                   help="Intermediate audit frequency (default: 5)")
    p.add_argument("--noise-std",   type=float, default=0.003,
                   help="Observation noise std [Ah] (default: 0.003)")
    p.add_argument("--noise-std-voltage", type=float, default=0.005,
                   help="Observation noise std on tracked mean terminal "
                        "voltage [V] (default: 0.005)")
    p.add_argument("--kappa",       type=float, default=2.0,
                   help="UCB exploration weight (default: 2.0)")
    p.add_argument("--mechanism-null", action="store_true",
                   help="Run the two-epistemic null instead of the real "
                        "capacity/voltage split for the controllability-"
                        "Gramian mechanism check (falsification test; "
                        "compare against a normal run's ratio)")
    default_uri = "sqlite:///" + str(Path.cwd() / "traits_audit_demo.db")
    p.add_argument("--mlflow-uri",  type=str,   default=default_uri,
                   help="MLflow tracking URI (default: local SQLite DB)")
    p.add_argument("--run-name",    type=str,   default="pybamm_demo",
                   help="MLflow run name (default: pybamm_demo)")
    p.add_argument("--ui",          action="store_true",
                   help="Launch the MLflow UI after the run")
    return p


def main() -> None:
    args = build_parser().parse_args()
    run(
        n_seed=args.n_seed,
        n_iter=args.n_iter,
        out_dir=Path(args.out_dir),
        seed=args.seed,
        check_every=args.check_every,
        noise_std=args.noise_std,
        noise_std_voltage=args.noise_std_voltage,
        kappa=args.kappa,
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
