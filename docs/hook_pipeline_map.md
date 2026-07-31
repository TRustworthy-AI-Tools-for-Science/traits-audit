# Uncertainty Hook — Active Learning Pipeline Map

Each demo runs the same eleven `AuditCheck`s (SDL runs twelve, adding
`LyapunovStabilityCheck`), but feeds them from a different point in its own
active learning pipeline. This document maps every check to the exact AL step
it observes in each demo.

---

## Check reference

| Check | Category | Data consumed | What it asks |
|---|---|---|---|
| `CalibrationError` | `ALEATORIC_MODEL` | `y_true`, `y_pred_mean`, `y_pred_std` | Are predicted confidence intervals statistically correct across all queried points? |
| `ConformalCoverage` | `ALEATORIC_MODEL` | `y_true`, `y_pred_mean`, `y_pred_std` | Is marginal coverage valid in a distribution-free sense? |
| `CRPS` | `ALEATORIC_MODEL` | `y_true`, `y_pred_mean`, `y_pred_std` | Is the Continuous Ranked Probability Score (proper scoring rule) acceptable? |
| `NegativeLogLikelihood` | `ALEATORIC_MODEL` | `y_true`, `y_pred_mean`, `y_pred_std` | Is the Gaussian NLL (proper scoring rule) acceptable? |
| `PITUniformity` | `ALEATORIC_MODEL` | `y_true`, `y_pred_mean`, `y_pred_std` | Are PIT values uniform (distributional calibration gold standard)? |
| `IntervalScore` | `ALEATORIC_MODEL` | `y_true`, `y_pred_mean`, `y_pred_std` | Is the Winkler interval score (coverage + sharpness jointly) acceptable? |
| `IntervalCoverage` | `ALEATORIC_MODEL` | `y_true`, `y_pred_mean`, `y_pred_std` | Does the ±1σ interval contain the oracle outcome ~68% of the time? |
| `VarianceAlignment` | `ALEATORIC_MODEL` | `y_true`, `y_pred_mean`, `y_pred_std` | Is mean predicted variance proportional to mean squared error? |
| `UncertaintyEvolution` | `EPISTEMIC` | `uncertainty` per step | Are any parameter channels showing a declining uncertainty trend (count of flagged channels)? |
| `UncertaintyAnomalies` | `EPISTEMIC` | `uncertainties` + `historical_uncertainties` | Are current uncertainty values anomalously far from a historical baseline (|z| > threshold)? Skipped when no baseline is provided. |
| `VarianceErrorCorrelation` | `EPISTEMIC` | `y_true`, `y_pred_mean`, `y_pred_std` | Is the surrogate most uncertain where it is most wrong (Spearman ρ > 0)? |
| `LyapunovStability` | `EPISTEMIC` | `op_states` | Is the surrogate's gradient-descent dynamics stable in PCA-reduced feature space? (SDL demo only) |

`on_step()` is called once per AL iteration; checks that consume `y_true`/`y_pred_*` operate on the accumulated history at pipeline-run time (either at `check_every` intervals or `on_end()`).

---

## Demo 1 — 1-D Calibration Benchmark (`ta-demo`)

**AL pipeline per step:**

```
Surrogate fit  →  LCB acquisition  →  Oracle query  ←── hook.on_step()
     ↑                                      |
     └──────────── add observation ─────────┘
```

`hook.on_step()` fires immediately after the oracle evaluation, before the surrogate is re-fit. `y_pred_mean` / `y_pred_std` are the bootstrap ensemble predictions at the LCB-selected point **prior** to incorporating that observation.

| Check | AL step monitored | What is being observed |
|---|---|---|
| `CalibrationError` | Oracle query | Whether the bootstrap posterior at each acquired point correctly brackets the oracle outcome, accumulated over all steps |
| `ConformalCoverage` | Oracle query | Whether marginal coverage holds in a distribution-free sense |
| `CRPS` | Oracle query | CRPS as a proper scoring rule on each acquired point |
| `NegativeLogLikelihood` | Oracle query | Gaussian NLL on each acquired point |
| `PITUniformity` | Oracle query | Whether PIT values are uniform across all acquired points |
| `IntervalScore` | Oracle query | Winkler interval score penalising non-coverage and excessive width |
| `IntervalCoverage` | Oracle query | Whether the 1σ bootstrap interval contains the oracle value ~68% of the time |
| `VarianceAlignment` | Oracle query | Whether bootstrap ensemble spread explains prediction error globally |
| `UncertaintyEvolution` | LCB acquisition | Count of channels showing a declining uncertainty trend over iterations |
| `UncertaintyAnomalies` | LCB acquisition | Anomalous uncertainty values relative to a historical baseline (skipped when no baseline provided) |
| `VarianceErrorCorrelation` | Oracle query | Whether the surrogate assigns larger σ to points where it errs most |

---

## Demo 2 — CAMD Materials Screening (`ta-camd-demo`)

**AL pipeline per step:**

```
Committee fit  →  Hypothesis selection  →  Evaluate hypotheses  ←── hook.on_step()
      ↑                                            |
      └──────────── grow seed set ─────────────────┘
```

`hook.on_step()` fires after all hypotheses in a batch are evaluated and added to the seed set, but before the committee is re-fit.

| Check | AL step monitored | What is being observed |
|---|---|---|
| `CalibrationError` | Hypothesis evaluation | Whether committee confidence matches the fraction of correctly predicted hypotheses, accumulated per batch |
| `ConformalCoverage` | Hypothesis evaluation | Distribution-free marginal coverage over the batch |
| `CRPS` | Hypothesis evaluation | CRPS on the evaluated batch |
| `NegativeLogLikelihood` | Hypothesis evaluation | Gaussian NLL on the evaluated batch |
| `PITUniformity` | Hypothesis evaluation | PIT uniformity across all evaluated hypotheses |
| `IntervalScore` | Hypothesis evaluation | Winkler score on the evaluated batch |
| `IntervalCoverage` | Hypothesis evaluation | Whether the batch-mean ±1σ committee interval contains the true stability value |
| `VarianceAlignment` | Hypothesis evaluation | Whether committee variance scales with squared error across the batch |
| `UncertaintyEvolution` | Hypothesis selection | Count of channels with declining committee std across iterations |
| `UncertaintyAnomalies` | Hypothesis selection | Anomalous committee uncertainty relative to a historical baseline (skipped when no baseline provided) |
| `VarianceErrorCorrelation` | Hypothesis evaluation | Whether the committee is more spread out on batches it predicts poorly |

---

## Demo 3 — PyBAMM Battery Optimisation (`ta-pybamm-demo`)

**AL pipeline per step:**

```
GPR fit  →  UCB acquisition  →  PyBAMM oracle (SPM)  ←── hook.on_step()
   ↑                                    |
   └──────────── add observation ───────┘
```

`hook.on_step()` fires after the PyBAMM SPM simulation returns the discharge capacity, before the GPR is re-fit.

| Check | AL step monitored | What is being observed |
|---|---|---|
| `CalibrationError` | PyBAMM oracle call | Whether the GPR posterior at the UCB-selected operating point correctly brackets the true discharge capacity |
| `ConformalCoverage` | PyBAMM oracle call | Distribution-free marginal coverage |
| `CRPS` | PyBAMM oracle call | CRPS on each oracle evaluation |
| `NegativeLogLikelihood` | PyBAMM oracle call | Gaussian NLL on each oracle evaluation |
| `PITUniformity` | PyBAMM oracle call | PIT uniformity across all queried (C-rate, T) points |
| `IntervalScore` | PyBAMM oracle call | Winkler score on each oracle evaluation |
| `IntervalCoverage` | PyBAMM oracle call | Whether the GPR 1σ interval contains the simulated capacity ~68% of the time |
| `VarianceAlignment` | PyBAMM oracle call | Whether GPR posterior variance explains prediction error |
| `UncertaintyEvolution` | UCB acquisition | Count of channels with declining GPR posterior std |
| `UncertaintyAnomalies` | UCB acquisition | Anomalous posterior std relative to a historical baseline (skipped when no baseline provided) |
| `VarianceErrorCorrelation` | PyBAMM oracle call | Whether the GPR is most uncertain at the operating points where its capacity prediction is least accurate |

---

## Demo 4 — SDL LED Colour-Matching (`ta-sdl-demo`)

**AL pipeline per step:**

```
[Sobol init — hook silent]
     ↓
Ax GP model ready
     ↓
Ax proposes (R, G, B)  →  Simulator observation  ←── hook.on_step()
        ↑                          |
        └──── complete_trial() ────┘
```

`hook.on_step()` is **not called during the Sobol initialisation phase** (no GP model, no posterior). It fires during the BO loop only.

| Check | AL step monitored | What is being observed |
|---|---|---|
| `CalibrationError` | Simulator observation (BO phase only) | Whether the Ax GP posterior at each proposed LED setting correctly brackets the true Fréchet distance |
| `ConformalCoverage` | Simulator observation (BO phase only) | Distribution-free marginal coverage |
| `CRPS` | Simulator observation (BO phase only) | CRPS on each simulator evaluation |
| `NegativeLogLikelihood` | Simulator observation (BO phase only) | Gaussian NLL on each simulator evaluation |
| `PITUniformity` | Simulator observation (BO phase only) | PIT uniformity across all BO-phase observations |
| `IntervalScore` | Simulator observation (BO phase only) | Winkler score on each simulator evaluation |
| `IntervalCoverage` | Simulator observation (BO phase only) | Whether the GP 1σ interval contains the simulated Fréchet value ~68% of the time |
| `VarianceAlignment` | Simulator observation (BO phase only) | Whether GP posterior variance scales with prediction error |
| `UncertaintyEvolution` | Ax acquisition (BO phase only) | Count of channels with declining GP posterior std |
| `UncertaintyAnomalies` | Ax acquisition (BO phase only) | Anomalous GP posterior std relative to a historical baseline (skipped when no baseline provided) |
| `VarianceErrorCorrelation` | Simulator observation (BO phase only) | Whether the GP is most uncertain at LED settings where it predicts colour distance poorly |
| `LyapunovStability` | End of run | Whether gradient-descent dynamics of the GP surrogate are stable in PCA-reduced feature space |
