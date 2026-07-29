# TRAITS Audit

<p align="center">
  <img src="docs/_static/logo.svg" alt="traits-audit logo" width="200">
</p>

![version](https://img.shields.io/badge/version-0.1.2-blue)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![tests](https://github.com/TRustworthy-AI-Tools-for-Science/traits-audit/actions/workflows/ci.yml/badge.svg)
![docs](https://github.com/TRustworthy-AI-Tools-for-Science/traits-audit/actions/workflows/docs-pages.yml/badge.svg)

A flexible uncertainty audit pipeline that hooks into any pre-existing active learning loop.


## Installation

```bash
# uv workspace (recommended — installs all demos + mlflow, editable, no reinstall on edits)
uv sync

# standalone pip install
pip install "."
pip install ".[mlflow,camd,pybamm,sdl]"   # with all demo extras
```

## Documentation

This repository uses **Sphinx** with docs source files in `docs/`.

- Local preview (build + serve in browser):
  ```bash
  make -C docs html && uv run python -m http.server 8000 --directory docs/_build/html
  # then open http://localhost:8000
  ```
- Production build:
  ```bash
  make -C docs html SPHINXOPTS="-W"
  ```
- Deployment: GitHub Pages is published by `.github/workflows/docs-pages.yml` on pushes to `main`, using `docs/_build/html`.

## Quickstart — run the demo

The package ships with a self-contained demo: a bootstrap-ensemble surrogate
learning a 1-D function via LCB acquisition, fully wired to the audit pipeline.

```bash
ta-demo                          # 100 AL steps, 4 calibration scenarios
ta-demo --steps 60 --seed 7
ta-demo --help
```

The demo entry point is `ta-demo` (source: `src/traits_audit/_example.py`).

## Built-in checks

These 16 checks all score the *total* predictive distribution (or, for the
last three, a trajectory/campaign-level diagnostic) against outcomes. See
[Taxonomy-audit checks](#taxonomy-audit-checks) below for checks that
discriminate *which kind* of uncertainty component is present, rather than
scoring the total.

| Check | Category | What it measures | Required data |
|---|---|---|---|
| `CalibrationErrorCheck` | Aleatoric (model) | [Kuleshov et al. (2018)][kuleshov2018] mean calibration error | `y_true`, `y_pred_mean`, `y_pred_std` |
| `KuleshovCalibrationCheck` | Aleatoric (model) | Same CE metric as its own pipeline row | `y_true`, `y_pred_mean`, `y_pred_std` |
| `ENCECheck` | Aleatoric (model) | Expected Normalized Calibration Error ([Levi et al. (2022)][levi2022]) | `y_true`, `y_pred_mean`, `y_pred_std` |
| `CalibrationError1StdCheck` | Aleatoric (model) | 1-sigma predictive-interval coverage error vs 68.3 % | `y_true`, `y_pred_mean`, `y_pred_std` |
| `ConformalCoverageCheck` | Aleatoric (model) | Distribution-free marginal coverage (Angelopoulos & Bates 2021) | `y_true`, `y_pred_mean`, `y_pred_std` |
| `CRPSCheck` | Aleatoric (model) | Continuous Ranked Probability Score — proper scoring rule (Gneiting & Raftery 2007) | `y_true`, `y_pred_mean`, `y_pred_std` |
| `NegativeLogLikelihoodCheck` | Aleatoric (model) | Gaussian NLL — proper scoring rule (Good 1952) | `y_true`, `y_pred_mean`, `y_pred_std` |
| `PITUniformityCheck` | Aleatoric (model) | KS test for PIT uniformity — distributional calibration (Dawid 1984) | `y_true`, `y_pred_mean`, `y_pred_std` |
| `IntervalScoreCheck` | Aleatoric (model) | Winkler interval score — penalises non-coverage and excessive width jointly; read as Gneiting & Raftery's *sharpness-subject-to-calibration* answer (Winkler 1972; Gneiting & Raftery 2007) | `y_true`, `y_pred_mean`, `y_pred_std` |
| `IntervalCoverageCheck` | Aleatoric (model) | Empirical 1-sigma coverage vs 68.3 % | `y_true`, `y_pred_mean`, `y_pred_std` |
| `VarianceAlignmentCheck` | Aleatoric (model) | Ratio of predicted to empirical variance ([Levi et al. (2022)][levi2022]) | `y_true`, `y_pred_mean`, `y_pred_std` |
| `MahalanobisOODCheck` | Epistemic | Input-support/OOD test with uncertainty-suppression assessment | `op_states` |
| `VarianceErrorCorrelationCheck` | Epistemic | Spearman ρ between std and \|error\| ([Lakshminarayanan et al. (2017)][lakshminarayanan2017]) | `y_true`, `y_pred_mean`, `y_pred_std` |
| `UncertaintyEvolutionCheck` | Reduction under replication | Trend of the *reported* uncertainty over iterations | per-step `uncertainty` |
| `UncertaintyAnomalyCheck` | Reduction under replication | Z-score anomaly detection on the *reported* uncertainty series | per-step `uncertainty` |
| `LyapunovStabilityCheck` | Ergodic/non-ergodic | Local Lyapunov exponent of surrogate dynamics in PCA-reduced feature space, along the campaign trajectory. **Pair with `DMDcSpectralRadiusCheck`** (global) | `op_states` |

## Taxonomy-audit checks

Eighteen checks added following `.claude/METRIC_TAXONOMY_AUDIT.md`'s audit
of `traits-audit` against eight measurement-error/UQ classification schemes
(`.claude/LITERATURE_SUMMARY.md` §3). Each discriminates *which kind* of
uncertainty component is present — random vs. systematic, Type A vs. Type
B, the aleatoric/epistemic split itself (not just its total), ergodic vs.
non-ergodic, variability vs. ignorance, procedural vs. data-driven, locus
in the analysis chain — rather than scoring the total predictive
distribution.

Several of these are only meaningful *in pairs*; a configured
`AuditPipeline` is checked automatically for missing pairings (see
[Pairing validation](#pairing-validation) below).

| Check | Taxonomy class | What it measures | Required data |
|---|---|---|---|
| `SignedBiasCheck` | Random/systematic | Signed mean residual — the constant component absolute-error metrics can't see | `y_true`, `y_pred_mean` |
| `ReplicationShrinkageExponentCheck` | Random/systematic | β in u_obs(r) ∝ r^-β: whether a component averages down under replication (report-only by default — see docstring for why 0.5 isn't a mandated target) | replicate groups |
| `DarkUncertaintyGapCheck` | Random/systematic | Observed replicate dispersion ÷ enumerated uncertainty budget; >1 is Kim et al.'s (2014) underestimation condition | replicate groups (+ `y_pred_std`) |
| `TypeBMassFractionCheck` | Type A/Type B | Declared fraction of predictive variance from non-data-derived components, via ablation | `ledger`, `variance_fn` |
| `ReducibilityRealisationRatioCheck` | Aleatoric/epistemic (split) | Whether claimed epistemic-variance reduction is realized after acquisition | pre-paired claimed/realized series |
| `AleatoricFloorConsistencyCheck` | Aleatoric/epistemic (split) | Ratio of learned aleatoric σ to observed replicate scatter | replicate groups (+ `y_pred_std`) |
| `EnsembleIndependenceDeficitCheck` | Ergodic/non-ergodic | Whether ensemble members are independent or effectively one member | `y_true`, `y_pred_ensemble` |
| `DMDcSpectralRadiusCheck` | Ergodic/non-ergodic | Pipeline wrapper around DMDc's global ρ(A). **Pair with `LyapunovStabilityCheck`** (local) | `rho_A` |
| `ResidualPersistenceHalfLifeCheck` | Ergodic/non-ergodic | Autocorrelation half-life vs. campaign length | `residuals_at_fixed_x` |
| `ImprecisionWidthFractionCheck` | Variability/ignorance | Upper-minus-lower probability of a reference interval from a credal set | credal set (bounds or ensemble) |
| `EnvelopeViolationRateCheck` | Variability/ignorance | Fraction of held-out points outside a bounding set | `y_true` + credal set |
| `ProceduralVarianceShareCheck` | Model/procedural | Fraction of ensemble variance from seed/init variability alone | ensemble + seed-sweep |
| `DataVarianceShareCheck` | Model/procedural | Fraction of ensemble variance from data resampling alone | ensemble + data-bootstrap sweep |
| `MisspecificationResidualFloorCheck` | Model/procedural | Non-vanishing floor ĉ in a·N^-γ+c fit to a learning curve | learning curve |
| `StageVarianceAttributionCheck` | Locus in the chain | Per-stage Sobol indices + interaction gap over an analysis pipeline | `chain_fn`, `stages` |
| `DecisionFlipRateCheck` | Locus in the chain | Fraction of resamples that flip a downstream decision | `decision_fn` + predictive distribution |
| `TailIndexCheck` | Cross-cutting (not a class member) | Hill estimator of the tail index; flags non-finite variance | `y_true`, `y_pred_mean`, `y_pred_std` |
| `ScoreDecompositionCheck` | Cross-cutting (not a class member) | DeGroot-Fienberg calibration/refinement decomposition of Gaussian NLL | `y_true`, `y_pred_mean`, `y_pred_std` |

### Pairing validation

`AuditPipeline.validate_config()` (also run automatically by `.run()`, its
result stored in `report.metadata["pairing_warnings"]`) checks a configured
check list for missing "contextualizing twins" per the pairings above —
e.g. configuring `LyapunovStabilityCheck` without `DMDcSpectralRadiusCheck`
produces an advisory warning. This never fails a run; it only flags that a
metric reported alone is liable to be misread.


### Basic usage — active run

```python
from traits_audit import AuditHook, AuditPipeline
from traits_audit.checks import CalibrationErrorCheck, UncertaintyEvolutionCheck

pipeline = AuditPipeline([
    CalibrationErrorCheck(threshold=0.1),
    UncertaintyEvolutionCheck(),
])

hook = AuditHook(pipeline)

for step in my_loop:
    mu, sigma = model.predict_with_uncertainty(X)
    hook.on_step(
        uncertainty=float(sigma.mean()),
    )

report = hook.on_end(y_true=y_test, y_pred_mean=mu_test, y_pred_std=sigma_test)
```


## Uncertainty taxonomy

`AuditCategory` values. The first three are the aleatoric/epistemic
(reducibility) classification scheme; the rest each correspond to one of
the other seven schemes surveyed in `.claude/LITERATURE_SUMMARY.md` §3
(see `.claude/METRIC_TAXONOMY_AUDIT.md` for the full audit).

| Category | Meaning |
|---|---|
| `ALEATORIC_IRREDUCIBLE` | Cannot be reduced by more data (measurement noise, process stochasticity) |
| `ALEATORIC_MODEL` | Calibration error — the model's stated uncertainty does not match empirical coverage |
| `EPISTEMIC` | Reducible uncertainty — shrinks as more observations are collected |
| `RANDOM_SYSTEMATIC` | Whether a component averages down over an ensemble/replication |
| `TYPE_A_TYPE_B` | Whether a stated value was obtained by statistical analysis of a series of observations, or by other means (declared, not estimated) |
| `ERGODIC_NON_ERGODIC` | Whether components are renewed per realisation (ensemble independence, trajectory persistence) |
| `VARIABILITY_IGNORANCE` | Whether a single precise probability distribution is an adequate representation |
| `MODEL_PROCEDURAL` | Where in modelling a reducible component originates (data vs. procedural/optimizer variability; misspecification) |
| `LOCUS_IN_CHAIN` | At which stage of the analysis pipeline, or point of decision, a component entered |
| `REDUCTION_UNDER_REPLICATION` | Testable shrinkage behaviour under replication |
| `UNKNOWN` | Not a member of any of the eight taxonomy classes (e.g. cross-cutting diagnostics), or not yet characterised |

## Run case studies

Four self-contained case studies ship with the package, each demonstrating the
audit on a different active learning domain.  All run without real hardware.

### Demo 1 — Calibration scenarios (1-D benchmark)

Compares four calibration regimes (perfectly calibrated, well calibrated,
overconfident, underconfident) on the Forrester benchmark function using a
bootstrap-ensemble surrogate and LCB acquisition.

```bash
ta-demo                                        # 100 steps, all 4 scenarios
ta-demo --steps 60 --seed 7
ta-demo --scenarios overconfident underconfident
ta-demo --help
```

### Demo 2 — PyBAMM Li-ion C-rate optimisation

Finds the (charge-rate, temperature) pair that maximises discharge capacity in
a lithium-ion cell using PyBAMM's Single Particle Model as the oracle and a
scikit-learn GPR with UCB acquisition.

```bash
ta-pybamm-demo                                 # 8 seed evals + 20 UCB steps
ta-pybamm-demo --n-iter 30 --kappa 3.0 --seed 7
ta-pybamm-demo --out-dir _results/pybamm
ta-pybamm-demo --help
```

### Demo 3 — Materials stability screening

Applies query-by-committee active learning to an OQMD materials stability
dataset using a BaggingRegressor committee surrogate.  Performs Lyapunov
stability analysis on the surrogate in PCA-reduced feature space.  Falls
back to synthetic data automatically if the OQMD dataset is unavailable.

```bash
ta-camd-demo                                   # 100 iterations, 4 queries/iter
ta-camd-demo --n-iter 30 --n-query 6 --seed 7
ta-camd-demo --out-dir _results/camd
ta-camd-demo --help
```

### Demo 4 — Self-driving lab LED colour matching

Runs Bayesian optimisation over a 3-D RGB LED intensity space to minimise the
Fréchet distance to a target colour, using the Ax/BoTorch GP as the surrogate.
Runs in simulation mode — no hardware required.

```bash
ta-sdl-demo                                    # 6 Sobol warm-start + 25 BO steps
ta-sdl-demo --n-iter 40 --seed 7
ta-sdl-demo --out-dir _results/sdl
ta-sdl-demo --help
```

## References

- **Kuleshov et al. (2018)** — *Accurate Uncertainties for Deep Learning Using Calibrated Regression.* ICML 2018. [arxiv:1807.00263](https://arxiv.org/abs/1807.00263)
- **Lakshminarayanan et al. (2017)** — *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS 2017. [arxiv:1612.01474](https://arxiv.org/abs/1612.01474)
- **Levi et al. (2022)** — *Evaluating and Calibrating Uncertainty Prediction in Regression Tasks.* [arxiv:1905.11659](https://arxiv.org/abs/1905.11659)
- **Angelopoulos & Bates (2021)** — *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.* [arxiv:2107.07511](https://arxiv.org/abs/2107.07511)
- **Gneiting & Raftery (2007)** — *Strictly Proper Scoring Rules, Prediction, and Estimation.* JASA. [doi:10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437)
- **Winkler (1972)** — *A Decision-Theoretic Approach to Interval Estimation.* JASA.

[kuleshov2018]: https://arxiv.org/abs/1807.00263
[lakshminarayanan2017]: https://arxiv.org/abs/1612.01474
[levi2022]: https://arxiv.org/abs/1905.11659
