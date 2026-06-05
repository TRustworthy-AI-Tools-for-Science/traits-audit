"""traits_audit demo — CAMD materials stability screening.

Runs CAMD's ``LocalAgentSimulation`` with ``ATFSampler`` (no real
experiments required) on the built-in test dataset.  The active learning
loop is driven manually step-by-step so that per-step AdaBoost committee
uncertainty can be captured and fed to :class:`~traits_audit.AuditHook`.
After the loop, Lyapunov stability analysis is performed on the learned
surrogate in a PCA-reduced feature space.

Install the optional dependency first::

    pip install "traits-audit[camd]"

Entry point::

    ta-camd-demo [OPTIONS]
    ta-camd-demo --n-iter 20 --out-dir _results/camd_demo --seed 7
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


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_data():
    """Load the CAMD test dataset; fall back to synthetic data."""
    try:
        from camd.utils.data import load_dataframe
        df = load_dataframe("test")
        print(f"  Loaded CAMD test dataset: {df.shape}")
        return df
    except Exception as exc:
        print(f"  CAMD test dataset unavailable ({exc}); using synthetic data")
        import pandas as pd
        rng = np.random.default_rng(42)
        n, d = 300, 12
        X = rng.standard_normal((n, d))
        y = -np.sum(X[:, :3] ** 2, axis=1) + rng.normal(0, 0.3, n)
        cols = [f"feature_{i}" for i in range(d)] + ["stability"]
        return pd.DataFrame(np.column_stack([X, y]), columns=cols)


def _feature_cols(df) -> list[str]:
    """Return numeric feature columns, excluding known target / ID columns."""
    exclude = {
        "stability", "formula", "entry_id", "Composition", "composition",
        "material_id", "icsd_id", "pretty_formula", "hull_distance",
        "delta_e", "target",
    }
    return [c for c in df.columns
            if c not in exclude and df[c].dtype != object]


# ── Committee uncertainty extraction ──────────────────────────────────────────

def _committee_predict(agent, X: np.ndarray):
    """Return (mean, std) from the AdaBoost committee.

    Looks for individual estimators under several attribute paths used
    across CAMD agent versions.
    """
    estimators = (
        getattr(agent, "estimators_", None)
        or getattr(getattr(agent, "regressor", None), "estimators_", None)
        or getattr(getattr(getattr(agent, "cv_result", None), "best_estimator_", None),
                   "estimators_", None)
        or []
    )
    if estimators:
        preds = np.stack([e.predict(X) for e in estimators])  # (n_trees, n_pts)
        return preds.mean(axis=0), preds.std(axis=0)

    # Single-model fallback (no committee uncertainty available)
    for attr in ("model", "regressor"):
        m = getattr(agent, attr, None)
        if m is not None and hasattr(m, "predict"):
            pred = m.predict(X)
            return np.asarray(pred), np.zeros(len(X))

    return np.zeros(len(X)), np.zeros(len(X))


# ── Main run ───────────────────────────────────────────────────────────────────

def run(
    n_seed: int = 25,
    n_iter: int = 50,
    n_query: int = 4,
    out_dir: Path = Path("_results/camd_demo"),
    seed: int = 0,
    check_every: int = 5,
    n_pca: int = 5,
    mlflow_uri: str | None = None,
    run_name: str = "camd_demo",
) -> dict:
    """Run the CAMD stability-screening demo with uncertainty audit + Lyapunov.

    Parameters
    ----------
    n_seed : int
        Initial labelled observations.
    n_iter : int
        Number of active learning iterations.
    n_query : int
        Candidates queried per iteration.
    out_dir : Path
        Root directory for results.
    seed : int
        RNG seed.
    check_every : int
        Intermediate audit frequency.
    n_pca : int
        PCA components for the Lyapunov state space (reduces dimensionality
        to make the Jacobian computation tractable).
    """
    import pandas as pd

    _use_mlflow = mlflow_uri is not None
    if _use_mlflow:
        import mlflow as _mlflow
        from traits_audit.mlflow_logger import MLflowLogger
        _mlflow.set_tracking_uri(mlflow_uri)
        _mlflow.set_experiment("traits_audit_platforms")
        _run_ctx = _mlflow.start_run(run_name=run_name)
        _run_ctx.__enter__()
        _mlflow.log_params({
            "platform":    "CAMD-sklearn-fallback",
            "n_seed":      n_seed,
            "n_iter":      n_iter,
            "n_query":     n_query,
            "seed":        seed,
            "check_every": check_every,
            "n_pca":       n_pca,
        })
        _mlflow.set_tags({
            "platform":    "CAMD",
            "model":       "BaggingRegressor-QBC",
            "acquisition": "max-uncertainty",
            "simulation":  "True",
        })
        _mlflow_logger = MLflowLogger()
    else:
        _mlflow_logger = None

    _camd_available = True
    try:
        from camd.agent.stability import AgentStabilityAdaBoost
        from camd.experiment.base import ATFSampler
    except ImportError:
        _camd_available = False
        print("  camd not installed — using sklearn BaggingRegressor fallback")

    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("CAMD Demo: materials stability screening (simulation)")
    print(f"  n_seed={n_seed}  n_iter={n_iter}  n_query={n_query}  seed={seed}")
    print(f"  Output: {out_dir}")
    print("=" * 60)

    from sklearn.decomposition import PCA as _PCA

    rng     = np.random.default_rng(seed)
    df_full = _load_data()
    feat    = _feature_cols(df_full)
    target  = "stability"

    print(f"  Features: {len(feat)}  Total materials: {len(df_full)}")

    # Seed / candidate split
    seed_idx = rng.choice(len(df_full), size=min(n_seed, len(df_full) // 3), replace=False)
    cand_idx = np.setdiff1d(np.arange(len(df_full)), seed_idx)
    seed_data = df_full.iloc[seed_idx].copy().reset_index(drop=True)
    cand_data = df_full.iloc[cand_idx].copy().reset_index(drop=True)

    # Fit PCA on seed features once, keep it fixed for consistent Lyapunov space
    _n_comp_live = min(n_pca, len(feat), len(seed_data) - 1)
    _pca_live = _PCA(n_components=_n_comp_live)
    _pca_live.fit(df_full[feat].values.astype(float))

    if _camd_available:
        atf   = ATFSampler(dataframe=df_full)
        agent = AgentStabilityAdaBoost(
            n_query=n_query,
            n_estimators=20,
            random_state=int(seed),
        )
        _use_camd_agent = True
    else:
        # sklearn BaggingRegressor as drop-in QBC committee
        from sklearn.ensemble import BaggingRegressor
        from sklearn.tree import DecisionTreeRegressor
        agent = BaggingRegressor(
            estimator=DecisionTreeRegressor(max_depth=5),
            n_estimators=20,
            random_state=int(seed),
        )
        _use_camd_agent = False

    hook = _make_pipeline(check_every, logger=_mlflow_logger)

    uncertainties: list[float] = []
    queried_feature_vecs: list[np.ndarray] = []
    lambda_max_per_step: list[float] = []    # |λ_max| at each step using live model

    # ── AL loop ────────────────────────────────────────────────────────────────
    print(f"\n[1/3] Active learning loop — {n_iter} iterations …")
    for step in range(n_iter):
        if len(cand_data) == 0:
            print("  Candidate pool exhausted — stopping early")
            break

        X_seed = seed_data[feat].values.astype(float)
        y_seed = seed_data[target].values.astype(float)
        X_cand = cand_data[feat].values.astype(float)

        if _use_camd_agent:
            # CAMD path: agent selects hypotheses internally
            hypotheses = agent.get_hypotheses(
                candidate_data=cand_data,
                seed_data=seed_data,
                n_query=min(n_query, len(cand_data)),
                seeded=False,
            )
            X_hyp   = hypotheses[feat].values.astype(float)
            mu_hyp, std_hyp = _committee_predict(agent, X_hyp)
            y_true  = hypotheses[target].values.astype(float)
            drop_idx = cand_data.index.isin(hypotheses.index)
            seed_data = pd.concat([seed_data, hypotheses], ignore_index=True)
            cand_data = cand_data.loc[~drop_idx].reset_index(drop=True)
        else:
            # sklearn fallback: fit BaggingRegressor, select by max uncertainty
            agent.fit(X_seed, y_seed)
            mu_cand, std_cand = _committee_predict(agent, X_cand)
            q = min(n_query, len(cand_data))
            query_loc = np.argsort(std_cand)[-q:]
            hypotheses = cand_data.iloc[query_loc].copy()
            X_hyp   = X_cand[query_loc]
            mu_hyp  = mu_cand[query_loc]
            std_hyp = std_cand[query_loc]
            y_true  = hypotheses[target].values.astype(float)
            seed_data = pd.concat([seed_data, hypotheses], ignore_index=True)
            cand_data = cand_data.drop(cand_data.index[query_loc]).reset_index(drop=True)

        # Track one representative operating point per step (for Lyapunov)
        queried_feature_vecs.append(X_hyp[0])

        # Per-step |λ_max| using the CURRENT live model in PCA space
        try:
            from traits_audit._lyapunov import (
                make_gd_predictor, numerical_jacobian, eigenvalues_and_stability,
            )
            _current_agent = agent  # closure over current fit

            def _live_scalar(state_pca: np.ndarray) -> float:
                s_orig = _pca_live.inverse_transform(state_pca.reshape(1, -1)).flatten()
                mu, _ = _committee_predict(_current_agent, s_orig.reshape(1, -1))
                return float(mu[0])

            # alpha=0.05: balances eigenvalue spread against numerical stability;
            # keeps |λ| near the unit circle across smooth and piecewise-constant
            # surrogates alike.
            # eps=1e-3: large enough to cross decision-tree leaf boundaries so
            # the finite-difference gradient is non-trivial for piecewise-constant
            # tree ensembles.
            _live_pred = make_gd_predictor(_live_scalar, alpha=0.05, eps=1e-3)
            _q_pca = _pca_live.transform(X_hyp[0].reshape(1, -1)).flatten()
            _J = numerical_jacobian(_live_pred, _q_pca, dx=1e-3)
            lambda_max_per_step.append(eigenvalues_and_stability(_J)["lambda_max"])
        except Exception:
            lambda_max_per_step.append(float("nan"))

        mean_mu  = float(mu_hyp.mean())
        mean_std = float(std_hyp.mean())
        mean_y   = float(y_true.mean())

        uncertainties.append(mean_std)

        hook.on_step(
            y_true=mean_y,
            y_pred_mean=mean_mu,
            y_pred_std=mean_std,
            uncertainty=mean_std,
            abs_error=float(np.abs(y_true - mu_hyp).mean()),
            dataset_size=float(len(seed_data)),
        )

        if (step + 1) % 5 == 0:
            n_stable = sum(r["is_stable"] for r in [])  # placeholder
            print(f"  Step {step + 1}/{n_iter}: "
                  f"seed={len(seed_data)}  cand={len(cand_data)}  "
                  f"uncertainty={mean_std:.4f}")

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
    print("\n[2/3] Lyapunov stability analysis …")
    from sklearn.decomposition import PCA
    from traits_audit._viz import (
        make_gd_predictor,
        run_lyapunov_analysis,
        plot_uncertainty_evolution,
        plot_lyapunov_evolution,
        plot_audit_evolution,
        plot_pareto_frontier,
        plot_convergence,
    )

    op_states_raw = np.array(queried_feature_vecs)   # (N, D_feat)
    n_components  = min(n_pca, op_states_raw.shape[1], len(op_states_raw) - 1)
    print(f"  PCA: D={op_states_raw.shape[1]} → {n_components} components")

    pca = PCA(n_components=n_components)
    op_states_pca = pca.fit_transform(op_states_raw)  # (N, n_pca)

    def _camd_mean_pca(state_pca: np.ndarray) -> float:
        state_orig = pca.inverse_transform(state_pca.reshape(1, -1)).flatten()
        mu, _ = _committee_predict(agent, state_orig.reshape(1, -1))
        return float(mu[0])

    def _camd_std_pca(state_pca: np.ndarray) -> float:
        state_orig = pca.inverse_transform(state_pca.reshape(1, -1)).flatten()
        _, std = _committee_predict(agent, state_orig.reshape(1, -1))
        return float(std[0])

    gd_pred = make_gd_predictor(_camd_mean_pca, alpha=0.05, eps=1e-3)

    lyap = run_lyapunov_analysis(
        predictor=gd_pred,
        op_states=op_states_pca,
        gp_std_fn=_camd_std_pca,
        model_label="AdaBoost-QBC (CAMD)",
        out_dir=fig_dir,
    )

    plot_uncertainty_evolution(
        np.array(uncertainties),
        model_label="AdaBoost-QBC (CAMD)",
        out_dir=fig_dir,
    )

    plot_lyapunov_evolution(
        lambda_max_seq=np.array(lambda_max_per_step),
        uncertainties=np.array(uncertainties),
        model_label="AdaBoost-QBC (CAMD)",
        out_dir=fig_dir,
    )

    plot_audit_evolution(
        pipeline=hook._pipeline,
        history=hook.history,
        model_label="AdaBoost-QBC (CAMD)",
        out_dir=fig_dir,
        snapshot_every=4,
    )

    abs_errors_al = [h.get("abs_error", float("nan")) for h in hook.history]
    plot_pareto_frontier(
        x_vals=np.array(uncertainties),
        y_vals=np.array(abs_errors_al),
        x_label="Committee std (stability units)",
        y_label="Mean absolute error",
        model_label="AdaBoost-QBC (CAMD)",
        out_dir=fig_dir,
        minimize_x=True,
        minimize_y=True,
        color_vals=np.arange(len(uncertainties)),
        color_label="AL step",
    )

    y_true_per_step = [h.get("y_true", float("nan")) for h in hook.history]
    best_stability = np.maximum.accumulate(
        np.nan_to_num(np.array(y_true_per_step), nan=-np.inf)
    )
    plot_convergence(
        best_vals=best_stability,
        query_counts=np.arange(1, len(best_stability) + 1) * n_query,
        y_label="Best stability score (per step mean)",
        model_label="AdaBoost-QBC (CAMD)",
        out_dir=fig_dir,
        maximise=True,
    )

    n_stable_final = sum(r["is_stable"] for r in [])  # placeholder (from CSV)
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
    p.add_argument("--n-seed",      type=int, default=25,
                   help="Initial labelled observations (default: 25)")
    p.add_argument("--n-iter",      type=int, default=50,
                   help="AL iterations (default: 50)")
    p.add_argument("--n-query",     type=int, default=4,
                   help="Queries per iteration (default: 4)")
    p.add_argument("--out-dir",     type=str, default="_results/camd_demo")
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--check-every", type=int, default=4,
                   help="Intermediate audit frequency (default: 4)")
    p.add_argument("--n-pca",       type=int, default=5,
                   help="PCA components for Lyapunov state space (default: 5)")
    default_uri = "sqlite:///" + str(Path.cwd() / "traits_audit_demo.db")
    p.add_argument("--mlflow-uri",  type=str, default=default_uri,
                   help="MLflow tracking URI (default: local SQLite DB)")
    p.add_argument("--run-name",    type=str, default="camd_demo",
                   help="MLflow run name (default: camd_demo)")
    p.add_argument("--ui",          action="store_true",
                   help="Launch the MLflow UI after the run")
    return p


def main() -> None:
    args = build_parser().parse_args()
    run(
        n_seed=args.n_seed,
        n_iter=args.n_iter,
        n_query=args.n_query,
        out_dir=Path(args.out_dir),
        seed=args.seed,
        check_every=args.check_every,
        n_pca=args.n_pca,
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
