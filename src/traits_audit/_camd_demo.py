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
            # window=None (default): a GLOBAL/cumulative verdict, consistent
            # with stability_convergence's already-cumulative growing-prefix
            # nature. Contrast with the PyBAMM demo's window=30 (local). See
            # docs/checks.rst and LYAPUNOV_ANALYSIS.md for the distinction.
            LyapunovStabilityCheck(stability_threshold=1.0, min_stable_fraction=0.5),
        ],
        verbose=False,
    )
    return AuditHook(pipeline, check_every=check_every, logger=logger)


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_data():
    """Load OQMD binary-compound data from matr.io cache; fall back to synthetic data.

    Downloads oqmd_1.2_voronoi_magpie_fingerprints.pickle (~150 MB) on first
    run and caches it under ~/.cache/traits_audit/.  Subsequent runs load from
    the local file.  Mirrors camd.utils.data.load_default_atf_data() without
    requiring pymatgen / matminer imports.
    """
    import pandas as pd
    cache_file = Path.home() / ".cache" / "traits_audit" / "oqmd_1.2_voronoi_magpie_fingerprints.pickle"
    url = "https://data.matr.io/3/api/v1/file/5e39ce2cd9f13e075b7dfaaf/download"
    try:
        if not cache_file.exists():
            import urllib.request
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            print("  Downloading OQMD dataset (~150 MB, cached after first run) …")
            urllib.request.urlretrieve(url, cache_file)
        # Shim for pickles created with pandas <2.0: Int64Index etc. were
        # merged into Index in pandas 2.0 and the submodule was removed.
        import sys, types
        if "pandas.core.indexes.numeric" not in sys.modules:
            _m = types.ModuleType("pandas.core.indexes.numeric")
            _m.Int64Index = pd.Index
            _m.Float64Index = pd.Index
            _m.UInt64Index = pd.Index
            sys.modules["pandas.core.indexes.numeric"] = _m
        df = pd.read_pickle(cache_file)
        # Mirror load_default_atf_data: binary compounds, 20 % sample
        if "N_species" in df.columns:
            df = df[df["N_species"] == 2].sample(frac=0.2, random_state=42)
        print(f"  Loaded OQMD dataset: {df.shape}")
        return df
    except Exception as exc:
        print(f"  OQMD dataset unavailable ({exc}); using synthetic data")
        rng = np.random.default_rng(42)
        n, d = 300, 12
        X = rng.standard_normal((n, d))
        y = -np.sum(X[:, :3] ** 2, axis=1) + rng.normal(0, 0.3, n)
        cols = [f"feature_{i}" for i in range(d)] + ["delta_e"]
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

try:
    import json as _json
    import os as _os
    import pandas as _pd
    from sklearn.ensemble import AdaBoostRegressor as _AdaBoostRegressor
    from sklearn.preprocessing import StandardScaler as _StandardScaler
    from sklearn.model_selection import cross_val_score as _cvs, KFold as _KFold
    from sklearn.pipeline import Pipeline as _Pipeline
    from camd.agent.stability import (
        AgentStabilityAdaBoost as _AgentBase,
        diverse_quant as _diverse_quant,
    )

    class _ExposedAdaBoost(_AgentBase):
        """AgentStabilityAdaBoost that stores its fitted AdaBoost and scaler.

        Identical to the parent in every respect except that ``get_hypotheses``
        saves ``self._adaboost`` and ``self._scaler`` so that committee
        predictions and uncertainty are accessible after selection.
        """

        def __init__(self, *args, random_state=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._adaboost:    "_AdaBoostRegressor | None" = None
            self._scaler:      "_StandardScaler | None"   = None
            self._random_state: "int | None"              = random_state

        def get_hypotheses(self, candidate_data, seed_data=None):
            X_cand, X_seed, y_seed = self.update_data(candidate_data, seed_data)

            # Diagnostic cross-validation (unchanged from parent)
            _ada_cv  = _AdaBoostRegressor(
                estimator=self.model, n_estimators=self.n_estimators,
                random_state=self._random_state,
            )
            _pipe = _Pipeline([("scaler", _StandardScaler()), ("ML", _ada_cv)])
            cv = _cvs(
                _pipe, X_seed, y_seed,
                cv=_KFold(3, shuffle=True, random_state=self._random_state),
                scoring="neg_mean_absolute_error",
            )
            self.cv_score = float(np.mean(cv)) * -1

            # Fit production model and STORE it.
            # Convert DataFrames to numpy so the scaler has no feature names —
            # committee_predict always passes numpy arrays and sklearn warns on mismatch.
            self._scaler   = _StandardScaler()
            X_scaled       = self._scaler.fit_transform(np.asarray(X_seed))
            self._adaboost = _AdaBoostRegressor(
                estimator=self.model, n_estimators=self.n_estimators,
                random_state=self._random_state,
            )
            self._adaboost.fit(X_scaled, np.asarray(y_seed))

            # Score candidates (unchanged from parent)
            X_cand_sc = self._scaler.transform(np.asarray(X_cand))
            expected  = self._adaboost.predict(X_cand_sc)
            if self.uncertainty:
                if self.dynamic_alpha and _os.path.exists("iteration.json"):
                    with open("iteration.json") as f:
                        _iter = _json.load(f)
                    expected -= (
                        min(0.1 * _iter, self.alpha)
                        * self._get_unc_ada(self._adaboost, X_cand_sc)
                    )
                else:
                    expected -= self.alpha * self._get_unc_ada(
                        self._adaboost, X_cand_sc
                    )

            self.update_candidate_stabilities(expected, sort=True, floor=-6.0)

            # Exploit / explore selection (unchanged from parent)
            stability_filter = (
                self.candidate_data["pred_stability"] <= self.hull_distance
            )
            within_hull    = self.candidate_data[stability_filter]
            n_exploitation = int(self.n_query * self.exploit_fraction)

            if self.diversify:
                to_compute = _diverse_quant(
                    within_hull.index.tolist(), n_exploitation,
                    self.candidate_data, feature_filter=self.feature_labels,
                )
            else:
                to_compute = within_hull.head(n_exploitation).index.tolist()

            remaining  = within_hull.tail(len(within_hull) - n_exploitation)
            remaining  = _pd.concat(
                [remaining, self.candidate_data[~stability_filter]]
            )
            n_exploration = min(self.n_query - n_exploitation, len(remaining))
            to_compute.extend(remaining.sample(n_exploration, random_state=self._random_state).index.tolist())

            return candidate_data.loc[to_compute]

        def committee_predict(self, X: np.ndarray):
            """Return (mean, std) on raw (unscaled) X using the last fitted model.

            Applies the stored StandardScaler before predicting, so X should
            be in the original feature space.  Returns zero arrays if called
            before the first ``get_hypotheses``.
            """
            if self._adaboost is None or self._scaler is None:
                return np.zeros(len(X)), np.zeros(len(X))
            X_sc = self._scaler.transform(X)
            mean = self._adaboost.predict(X_sc)
            std  = self._get_unc_ada(self._adaboost, X_sc)
            return mean, std

except ImportError:
    _ExposedAdaBoost = None


def _committee_predict(agent, X: np.ndarray):
    """Return (mean, std) from the AdaBoost committee.

    Dispatches to agent.committee_predict(X) when available (e.g.
    _ExposedAdaBoost), otherwise searches for estimators_ under the
    attribute paths used across CAMD agent versions.
    """
    if hasattr(agent, "committee_predict"):
        return agent.committee_predict(X)

    estimators = (
        getattr(agent, "estimators_", None)
        or getattr(getattr(agent, "regressor", None), "estimators_", None)
        or getattr(getattr(getattr(agent, "cv_result", None), "best_estimator_", None),
                   "estimators_", None)
    )

    # Recursive fallback: search all first-level attributes for an object
    # that carries estimators_ (handles any CAMD agent wrapper layout).
    if not estimators:
        for val in vars(agent).values():
            estimators = getattr(val, "estimators_", None)
            if estimators:
                break

    if estimators:
        preds = np.stack([e.predict(X) for e in estimators])  # (n_trees, n_pts)
        return preds.mean(axis=0), preds.std(axis=0)

    # Single-model fallback (no committee uncertainty available)
    for attr in ("model", "regressor"):
        m = getattr(agent, attr, None)
        if m is not None and hasattr(m, "predict"):
            pred = m.predict(X)
            print("  WARNING: no committee estimators found — uncertainty set to zero. "
                  f"(agent type: {type(agent).__name__})")
            return np.asarray(pred), np.zeros(len(X))

    print("  WARNING: agent has no predict method — returning zero predictions and uncertainty. "
          f"(agent type: {type(agent).__name__})")
    return np.zeros(len(X)), np.zeros(len(X))


# ── Main run ───────────────────────────────────────────────────────────────────

def run(
    n_seed: int = 25,
    n_iter: int = 100,
    n_query: int = 4,
    out_dir: Path = Path("_results/camd_demo"),
    seed: int = 0,
    check_every: int = 5,
    n_pca: int = 5,
    max_cand: int = 3000,
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
    max_cand : int
        Maximum candidate pool size.  The CAMD agent's uncertainty loop is
        O(n_candidates), so capping at ~3 000 keeps each step to ~5 s.
        Pass 0 or None to use the full OQMD pool (~14 k entries, ~25 s/step).
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
    target  = "delta_e"

    print(f"  Features: {len(feat)}  Total materials: {len(df_full)}")

    # Seed / candidate split
    seed_idx = rng.choice(len(df_full), size=min(n_seed, len(df_full) // 3), replace=False)
    cand_idx = np.setdiff1d(np.arange(len(df_full)), seed_idx)
    seed_data = df_full.iloc[seed_idx].copy().reset_index(drop=True)
    cand_data = df_full.iloc[cand_idx].copy().reset_index(drop=True)
    if max_cand and len(cand_data) > max_cand:
        cand_data = cand_data.sample(max_cand, random_state=int(seed)).reset_index(drop=True)
        print(f"  Candidate pool capped at {max_cand} (pass --max-cand 0 for full pool)")

    from sklearn.ensemble import BaggingRegressor
    from sklearn.tree import DecisionTreeRegressor
    _surrogate = BaggingRegressor(
        estimator=DecisionTreeRegressor(max_depth=5),
        n_estimators=20,
        random_state=int(seed),
    )

    if _camd_available and _ExposedAdaBoost is not None:
        # alpha=0.5 matches the paper's best-performing "AB-ε0-α0.5" agent
        # (Montoya et al. 2020, https://doi.org/10.1039/D0SC01101K).
        agent = _ExposedAdaBoost(
            n_query=n_query, n_estimators=20, random_state=int(seed), alpha=0.5,
        )
        _use_camd_agent = True
    else:
        agent = None
        _use_camd_agent = False

    # Feature set for PCA / Lyapunov: CAMD agent uses its own feature_labels
    # (which may differ from _feature_cols() by one column), sklearn path uses feat.
    _feat_pca = agent.feature_labels if _use_camd_agent else feat

    # Fit PCA on all data with the correct feature set, keep fixed for Lyapunov space.
    _n_comp_live = min(n_pca, len(_feat_pca), len(seed_data) - 1)
    _pca_live = _PCA(n_components=_n_comp_live)
    _pca_live.fit(df_full[_feat_pca].values.astype(float))

    # Pre-fit a PCA scaler on the full pool so the augmented-state PCA coords
    # are unit-variance and stable across all steps.
    from sklearn.preprocessing import StandardScaler as _KRRScaler
    _krr_scaler = _KRRScaler()
    _krr_scaler.fit(_pca_live.transform(df_full[_feat_pca].values.astype(float)))

    hook = _make_pipeline(check_every, logger=_mlflow_logger)

    initial_seed = seed_data.copy()
    uncertainties: list[float] = []
    queried_batches: list = []
    aug_states_list: list[np.ndarray] = []   # [scaled PCA coords | mean std] per step
    lambda_max_per_step: list[float] = []    # filled post-loop via DMDc stability_convergence

    # ── AL loop ────────────────────────────────────────────────────────────────
    print(f"\n[1/3] Active learning loop — {n_iter} iterations …")
    for step in range(n_iter):
        if len(cand_data) == 0:
            print("  Candidate pool exhausted — stopping early")
            break

        if _use_camd_agent:
            # CAMD path: _ExposedAdaBoost fits internally and stores the model;
            # _committee_predict dispatches to agent.committee_predict().
            hypotheses = agent.get_hypotheses(
                candidate_data=cand_data,
                seed_data=seed_data,
            )
            # Use agent.feature_labels (= _feat_pca) so feature count matches
            # the StandardScaler fitted inside get_hypotheses().
            X_hyp   = hypotheses[_feat_pca].values.astype(float)
            mu_hyp, std_hyp = _committee_predict(agent, X_hyp)
            y_true  = hypotheses[target].values.astype(float)
            drop_idx = cand_data.index.isin(hypotheses.index)
            seed_data = pd.concat([seed_data, hypotheses], ignore_index=True)
            cand_data = cand_data.loc[~drop_idx].reset_index(drop=True)
        else:
            # sklearn fallback: fit surrogate, select by max committee uncertainty
            X_seed = seed_data[feat].values.astype(float)
            y_seed = seed_data[target].values.astype(float)
            X_cand = cand_data[feat].values.astype(float)
            _surrogate.fit(X_seed, y_seed)
            mu_cand, std_cand = _committee_predict(_surrogate, X_cand)
            q = min(n_query, len(cand_data))
            # LCB with α=0.5, matching the paper's best agent (Montoya et al. 2020).
            # Selects the q candidates with the lowest predicted stability under
            # uncertainty — lower delta_e = more stable, so we take the minimum.
            _lcb = mu_cand - 0.5 * std_cand
            query_loc = np.argsort(_lcb)[:q]
            hypotheses = cand_data.iloc[query_loc].copy()
            X_hyp   = X_cand[query_loc]
            mu_hyp  = mu_cand[query_loc]
            std_hyp = std_cand[query_loc]
            y_true  = hypotheses[target].values.astype(float)
            seed_data = pd.concat([seed_data, hypotheses], ignore_index=True)
            cand_data = cand_data.drop(cand_data.index[query_loc]).reset_index(drop=True)

        # Collect batch for exploration map
        _qcols = [c for c in ["Composition", target] + feat if c in hypotheses.columns]
        queried_batches.append(hypotheses[_qcols].copy())

        mean_std = float(std_hyp.mean())
        uncertainties.append(mean_std)

        # Augmented state for DMDc: [scaled PCA coords of queried point | mean std].
        # Analogous to battery-forecast's [ECM means | ECM stds].  A_r fitted on
        # this trajectory is a general (non-symmetric) matrix — complex eigenvalues
        # emerge naturally, unlike the GD-predictor Jacobian J = I − αH_f which is
        # always symmetric.  Per-step lambda_max is computed post-loop via
        # stability_convergence (growing-prefix DMDc fits).
        _q_pca_scaled = _krr_scaler.transform(
            _pca_live.transform(X_hyp[0].reshape(1, -1))
        ).flatten()
        aug_states_list.append(np.append(_q_pca_scaled, mean_std))

        hook.on_step(
            y_true=y_true,
            y_pred_mean=mu_hyp,
            y_pred_std=std_hyp,
            uncertainty=mean_std,
            abs_error=float(np.abs(y_true - mu_hyp).mean()),
            dataset_size=float(len(seed_data)),
        )

        if (step + 1) % 5 == 0:
            print(f"  Step {step + 1}/{n_iter}: "
                  f"seed={len(seed_data)}  cand={len(cand_data)}  "
                  f"uncertainty={mean_std:.4f}")

    # ── Lyapunov analysis ──────────────────────────────────────────────────────
    # Computed before hook.on_end() so LyapunovStabilityCheck (in the pipeline
    # above) can be given the real lambda_max series via the precomputed route.
    print("\n[2/3] Lyapunov stability analysis …")
    from traits_audit._viz import (
        _fig_check_grid,
        run_dmdc_lyapunov_analysis,
        plot_uncertainty_evolution,
        plot_lyapunov_evolution,
        plot_audit_evolution,
        plot_pareto_frontier,
        plot_convergence,
        plot_exploration_campaign,
        plot_discovery_rate,
    )
    from traits_audit import dmdc as dm

    # Build augmented state trajectory and compute per-step lambda_max via
    # growing-prefix DMDc fits (analogous to battery-forecast's stability_convergence).
    aug_traj = np.array(aug_states_list)  # (T, n_pca + 1)
    _dmdc_actions = np.ones((len(aug_traj), 1))
    _min_obs_dmdc = n_pca + 2
    _conv = dm.stability_convergence(
        aug_traj, _dmdc_actions, min_obs=_min_obs_dmdc, n_components=n_pca,
    )
    # Align one |λ_max| value per AL step: the first (len(aug_traj) - len(_conv))
    # steps precede the DMDc warm-up and are NaN.  Padding to the actual step
    # count (not a fixed _min_obs_dmdc) keeps this in sync with `uncertainties`
    # even when n_iter < _min_obs_dmdc leaves _conv empty.  LyapunovStabilityCheck
    # drops the NaN prefix itself rather than counting it as unstable.
    _n_pad = max(0, len(aug_traj) - len(_conv))
    lambda_max_per_step = [float("nan")] * _n_pad + _conv.tolist()

    print(f"  DMDc augmented state: D={aug_traj.shape[1]} "
          f"({n_pca} PCA coords + committee std), T={len(aug_traj)} steps")

    # UncertaintyAnomalyCheck compares recent behaviour against an earlier
    # baseline.  Use the first two check windows as the historical reference so
    # the check detects genuine drift rather than within-series variance.
    n_warmup = max(check_every * 2, len(uncertainties) // 5, 1)
    report = hook.on_end(
        historical_uncertainties=np.array(uncertainties[:n_warmup]),
        lambda_max=np.array(lambda_max_per_step),
    )
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

    lyap = run_dmdc_lyapunov_analysis(
        aug_states=aug_traj,
        model_label="AdaBoost-QBC (CAMD)",
        out_dir=fig_dir,
        n_components=n_pca,
        gp_std_seq=np.array(uncertainties),
        actions=_dmdc_actions,
        min_obs=_min_obs_dmdc,
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

    # Check-grid heatmap: rows = audit checks, cols = snapshot steps.
    # Columns come from hook.intermediate_reports (fired every check_every steps)
    # plus a "final" column from the end-of-run report.
    fig_grid = None
    if hook.intermediate_reports:
        stage_reports = [
            (f"step {(i + 1) * check_every}", r)
            for i, r in enumerate(hook.intermediate_reports)
        ]
        stage_reports.append(("final", report))
        fig_grid = _fig_check_grid(stage_reports, "AdaBoost-QBC (CAMD)")
        if fig_grid is not None:
            _grid_png = fig_dir / "fig10_check_grid.png"
            try:
                fig_grid.write_image(
                    str(_grid_png),
                    width=fig_grid.layout.width,
                    height=fig_grid.layout.height,
                    scale=2,
                )
                print("  Saved fig10_check_grid.png")
            except Exception:
                _grid_html = fig_dir / "fig10_check_grid.html"
                fig_grid.write_html(str(_grid_html))
                print(f"  Saved fig10_check_grid.html (install kaleido for PNG export)")

    abs_errors_al = [h.get("abs_error", float("nan")) for h in hook.history]
    plot_pareto_frontier(
        x_vals=np.array(uncertainties),
        y_vals=np.array(abs_errors_al),
        x_label="Committee std (stability units)",
        y_label="Mean absolute error (MAE)",
        model_label="AdaBoost-QBC (CAMD)",
        out_dir=fig_dir,
        minimize_x=True,
        minimize_y=True,
        color_vals=np.arange(len(uncertainties)),
        color_label="AL step",
    )

    y_true_per_step = [
        float(np.mean(h["y_true"])) if "y_true" in h else float("nan")
        for h in hook.history
    ]
    # Most stable = most negative delta_e; track running minimum.
    best_stability = np.minimum.accumulate(
        np.nan_to_num(np.array(y_true_per_step), nan=np.inf)
    )
    plot_convergence(
        best_vals=best_stability,
        query_counts=np.arange(1, len(best_stability) + 1) * n_query,
        y_label=r"Best $\Delta E$ (eV/atom)",
        model_label="AdaBoost-QBC (CAMD)",
        out_dir=fig_dir,
        maximise=False,
    )

    plot_exploration_campaign(
        df_all=df_full,
        feat=feat,
        target=target,
        seed_df=initial_seed,
        queried_batches=queried_batches,
        model_label="AdaBoost-QBC (CAMD)",
        out_dir=fig_dir,
    )

    # Discovery-rate figure: cumulative stable materials found vs random baseline.
    # Threshold = 25th percentile of the full pool — bottom quartile by delta_e.
    # This is the primary evaluation metric in Montoya et al. (2020): how many
    # stable/near-stable materials does the AL agent find per DFT evaluation,
    # compared to selecting candidates at random?
    _stability_threshold = float(np.percentile(df_full[target].values, 25))
    _y_true_per_batch = [
        np.array(h["y_true"]) for h in hook.history if "y_true" in h
    ]
    plot_discovery_rate(
        y_true_per_batch=_y_true_per_batch,
        df_all_target=df_full[target].values,
        stability_threshold=_stability_threshold,
        model_label="AdaBoost-QBC (CAMD)",
        out_dir=fig_dir,
    )

    if _use_mlflow:
        lm = lyap["lambda_max"]
        _mlflow.log_metrics({
            "lyapunov/lambda_max_mean": float(lm.mean()),
            "lyapunov/lambda_max_max":  float(lm.max()),
            "lyapunov/n_stable":        int((lm < 1.0).sum()),
        })
        _mlflow.log_artifact(str(fig_dir / "lyapunov_stability.csv"), "lyapunov")
        if fig_grid is not None:
            _mlflow.log_figure(fig_grid, "audit/check_grid.html")
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
    p.add_argument("--n-iter",      type=int, default=100,
                   help="AL iterations (default: 100)")
    p.add_argument("--n-query",     type=int, default=4,
                   help="Queries per iteration (default: 4)")
    p.add_argument("--out-dir",     type=str, default="_results/camd_demo")
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--check-every", type=int, default=4,
                   help="Intermediate audit frequency (default: 4)")
    p.add_argument("--n-pca",       type=int, default=5,
                   help="PCA components for Lyapunov state space (default: 5)")
    p.add_argument("--max-cand",    type=int, default=3000,
                   help="Max candidate pool size; 0 = full OQMD pool (default: 3000)")
    default_uri = "sqlite:///" + str(Path.cwd() / "traits_audit_demo.db")
    p.add_argument("--mlflow-uri",  type=str, default=None,
                   help=f"MLflow tracking URI (default: disabled; pass a URI such as "
                        f"sqlite:///traits_audit_demo.db or {default_uri} to enable)")
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
        max_cand=args.max_cand,
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
