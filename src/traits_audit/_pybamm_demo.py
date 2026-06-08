"""traits_audit demo — PyBAM Li-ion cell C-rate / temperature optimisation.

Active-learning loop that finds the (charge-rate, temperature) operating point
maximising discharge capacity in a lithium-ion cell.  The oracle is PyBAM's
fast Single Particle Model (SPM) — no hardware required.

Domain
------
State:       2-D normalised  [c_rate_norm, T_norm]  ∈  [0, 1]²
C-rate:      0.5 C – 3.0 C
Temperature: 10 °C – 40 °C
Oracle:      PyBAM SPM single discharge → capacity [Ah]
Surrogate:   sklearn GaussianProcessRegressor  (RBF + WhiteKernel)
Policy:      UCB  (κ = 2.0)  →  maximise predicted capacity
Audit:       6 uncertainty checks via AuditHook / AuditPipeline
Stability:   Lyapunov analysis on the gradient-descent map of the surrogate

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


def _simulate_capacity(c_rate: float, T_C: float, model, param, Cn: float) -> float:
    """Run one SPM discharge at ``c_rate`` C and ``T_C`` °C.

    Returns the total discharge capacity [Ah].
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
    return float(sol["Discharge capacity [A.h]"].entries[-1])


# ── Audit pipeline ─────────────────────────────────────────────────────────────

def _make_pipeline(check_every: int, logger=None):
    from traits_audit import AuditHook, AuditPipeline
    from traits_audit.checks import (
        CalibrationErrorCheck,
        ConformalCoverageCheck,
        IntervalCoverageCheck,
        VarianceAlignmentCheck,
        UncertaintyEvolutionCheck,
        UncertaintyAnomalyCheck,
        VarianceErrorCorrelationCheck,
    )
    pipeline = AuditPipeline(
        checks=[
            CalibrationErrorCheck(threshold=0.15),
            ConformalCoverageCheck(target_coverage=0.9, max_q_ratio=1.5),
            IntervalCoverageCheck(expected_coverage=0.683, tolerance=0.15),
            VarianceAlignmentCheck(tolerance=0.5),
            UncertaintyEvolutionCheck(slope_threshold=-0.05),
            UncertaintyAnomalyCheck(z_threshold=3.0),
            VarianceErrorCorrelationCheck(min_correlation=0.1),
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
    kappa: float = 2.0,
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
        Additive Gaussian observation noise [Ah].
        A value of 0.003 Ah ≈ 0.4 % of nominal — realistic for lab variability.
    kappa : float
        UCB exploration-exploitation trade-off (higher → more exploration).
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

    # ── Phase 1: random seed evaluations (no GPR) ─────────────────────────────
    print(f"\n[1/3] Seed — {n_seed} random evaluations …")
    seed_idx = rng.choice(len(X_pool), size=min(n_seed, len(X_pool) // 4), replace=False)
    X_obs_n  = X_pool_n[seed_idx].copy()
    X_obs_r  = X_pool[seed_idx].copy()    # raw coords for reporting
    y_obs    = np.array([
        _simulate_capacity(c, T, model_pybam, param_base, Cn)
        + rng.normal(0, noise_std)
        for c, T in X_pool[seed_idx]
    ])

    remaining = np.ones(len(X_pool), dtype=bool)
    remaining[seed_idx] = False

    # ── GPR definition ─────────────────────────────────────────────────────────
    # Wide bounds avoid ConvergenceWarnings at the edges of the search space.
    # The capacity landscape spans ~3 % (0.025 Ah) over the 2-D space;
    # normalize_y=True maps this to O(1) before kernel fitting.
    kernel = (
        ConstantKernel(
            constant_value=0.1,
            constant_value_bounds=(1e-6, 100.0),
        )
        * RBF(
            length_scale=[0.4, 0.4],
            length_scale_bounds=(1e-3, 10.0),
        )
        + WhiteKernel(
            noise_level=noise_std ** 2,
            noise_level_bounds=(1e-10, 1.0),
        )
    )
    gpr = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=3, normalize_y=True,
    )

    hook = _make_pipeline(check_every, logger=_mlflow_logger)

    uncertainties: list[float]    = []
    queried_n:    list[np.ndarray] = []   # AL-queried normalised coords

    # ── Phase 2: UCB active-learning loop ─────────────────────────────────────
    print(f"\n[2/3] UCB active learning — {n_iter} iterations …")
    for step in range(n_iter):
        if not remaining.any():
            print(f"  Pool exhausted — stopping after {step} iteration(s).")
            break

        gpr.fit(X_obs_n, y_obs)

        X_cand_n = X_pool_n[remaining]
        mu_cand, sigma_cand = gpr.predict(X_cand_n, return_std=True)
        acq = mu_cand + kappa * sigma_cand          # UCB: maximise capacity

        best_local  = int(np.argmax(acq))
        pool_idx    = np.where(remaining)[0]
        best_global = pool_idx[best_local]

        xi_n = X_pool_n[best_global]
        xi_r = X_pool[best_global]
        c_q, T_q = float(xi_r[0]), float(xi_r[1])

        y_true  = (
            _simulate_capacity(c_q, T_q, model_pybam, param_base, Cn)
            + rng.normal(0, noise_std)
        )

        # GPR prediction at the queried point (before incorporating it)
        mu_q, sigma_q = gpr.predict(xi_n.reshape(1, -1), return_std=True)
        mu_q    = float(mu_q[0])
        sigma_q = float(sigma_q[0])

        X_obs_n = np.vstack([X_obs_n, xi_n])
        X_obs_r = np.vstack([X_obs_r, xi_r])
        y_obs   = np.append(y_obs, y_true)
        remaining[best_global] = False

        uncertainties.append(sigma_q)
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

        if (step + 1) % 5 == 0:
            print(f"  Step {step + 1}/{n_iter}: "
                  f"({c_q:.2f} C, {T_q:.0f} °C)  "
                  f"cap={y_true:.4f} Ah  σ={sigma_q:.5f}")

    best_i = int(np.argmax(y_obs))
    c_best, T_best = float(X_obs_r[best_i, 0]), float(X_obs_r[best_i, 1])
    print(f"\n  Best found: ({c_best:.2f} C, {T_best:.0f} °C)  "
          f"cap={y_obs[best_i]:.4f} Ah  "
          f"(dataset = {len(y_obs)} pts)")

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

    # ── Phase 3: Lyapunov stability analysis ──────────────────────────────────
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

    plot_uncertainty_evolution(
        np.array(uncertainties),
        model_label="sklearn-GPR (PyBAM)",
        out_dir=fig_dir,
    )

    plot_lyapunov_evolution(
        lambda_max_seq=lyap["lambda_max"],
        uncertainties=np.array(uncertainties),
        model_label="sklearn-GPR (PyBAM)",
        out_dir=fig_dir,
    )

    plot_audit_evolution(
        pipeline=hook._pipeline,
        history=hook.history,
        model_label="sklearn-GPR (PyBAM)",
        out_dir=fig_dir,
        snapshot_every=4,
    )

    cap_al = y_obs[n_seed:]                 # capacities for UCB-queried points only
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
    return {"report": report, "lyapunov": lyap}


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--n-seed",      type=int,   default=8,
                   help="Random seed evaluations (default: 8)")
    p.add_argument("--n-iter",      type=int,   default=20,
                   help="UCB AL iterations (default: 20)")
    p.add_argument("--out-dir",     type=str,   default="_results/pybamm_demo")
    p.add_argument("--seed",        type=int,   default=0)
    p.add_argument("--check-every", type=int,   default=5,
                   help="Intermediate audit frequency (default: 5)")
    p.add_argument("--noise-std",   type=float, default=0.003,
                   help="Observation noise std [Ah] (default: 0.003)")
    p.add_argument("--kappa",       type=float, default=2.0,
                   help="UCB exploration weight (default: 2.0)")
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
        kappa=args.kappa,
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
