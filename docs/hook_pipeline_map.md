# Uncertainty Hook — Active Learning Pipeline Map

Each demo runs the same six `AuditCheck`s, but feeds them from a different point in its own active learning pipeline. This document maps every check to the exact AL step it observes in each demo.

---

## Check reference

| Check | Category | Data consumed | What it asks |
|---|---|---|---|
| `CalibrationError` | `ALEATORIC_MODEL` | `y_true`, `y_pred_mean`, `y_pred_std` | Are predicted confidence intervals statistically correct across all queried points? |
| `IntervalCoverage` | `ALEATORIC_MODEL` | `y_true`, `y_pred_mean`, `y_pred_std` | Does the ±1σ interval contain the oracle outcome ~68% of the time? |
| `VarianceAlignment` | `ALEATORIC_MODEL` | `y_true`, `y_pred_mean`, `y_pred_std` | Is mean predicted variance proportional to mean squared error? |
| `UncertaintyEvolution` | `EPISTEMIC` | `uncertainty` per step | Is posterior std collapsing too fast (possible model overconfidence)? |
| `UncertaintyAnomalies` | `EPISTEMIC` | `uncertainty` per step | Are any steps anomalously high or low in uncertainty (|z| > 3)? |
| `VarianceErrorCorrelation` | `EPISTEMIC` | `y_true`, `y_pred_mean`, `y_pred_std` | Is the surrogate most uncertain where it is most wrong (Spearman ρ > 0)? |

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
| `IntervalCoverage` | Oracle query | Whether the 1σ bootstrap interval contains the oracle value ~68% of the time |
| `VarianceAlignment` | Oracle query | Whether bootstrap ensemble spread explains prediction error globally |
| `UncertaintyEvolution` | LCB acquisition | Trend in posterior std at the LCB-selected point over iterations — expected to decrease as the surrogate fills in the domain |
| `UncertaintyAnomalies` | LCB acquisition | Anomalous spikes or drops in acquisition-point std across the run |
| `VarianceErrorCorrelation` | Oracle query | Whether the surrogate assigns larger σ to points where it errs most |

---

## Demo 2 — CAMD Materials Screening (`ta-camd-demo`)

**AL pipeline per step:**

```
Committee fit  →  Hypothesis selection  →  Evaluate hypotheses  ←── hook.on_step()
      ↑                                            |
      └──────────── grow seed set ─────────────────┘
```

`hook.on_step()` fires after all hypotheses in a batch are evaluated and added to the seed set, but before the committee is re-fit. Values passed are **means over the batch**: `mean_mu`, `mean_std`, `mean_y`, and `mean |error|`.

| Check | AL step monitored | What is being observed |
|---|---|---|
| `CalibrationError` | Hypothesis evaluation | Whether the committee's mean confidence, averaged over each selected batch, matches the fraction of hypotheses that actually satisfy the stability criterion |
| `IntervalCoverage` | Hypothesis evaluation | Whether the batch-mean ±1σ committee interval contains the batch-mean true stability value |
| `VarianceAlignment` | Hypothesis evaluation | Whether batch-mean committee variance scales with batch-mean squared error |
| `UncertaintyEvolution` | Hypothesis selection | Trend in mean committee std across the selected batch over iterations — should decrease as explored regions are covered |
| `UncertaintyAnomalies` | Hypothesis selection | Batches where the committee's mean uncertainty is anomalously high (candidate pool shrinking into unexplored regions) or low (collapsing committee) |
| `VarianceErrorCorrelation` | Hypothesis evaluation | Whether the committee is more spread out on batches it predicts poorly |

---

## Demo 3 — PyBAM Battery Optimisation (`ta-pybamm-demo`)

**AL pipeline per step:**

```
GPR fit  →  UCB acquisition  →  PyBAM oracle (SPM)  ←── hook.on_step()
   ↑                                    |
   └──────────── add observation ───────┘
```

`hook.on_step()` fires after the PyBAM SPM simulation returns the discharge capacity, before the GPR is re-fit. `y_pred_mean` / `y_pred_std` are the GPR posterior at the UCB-selected (C-rate, T) point **prior** to the update.

| Check | AL step monitored | What is being observed |
|---|---|---|
| `CalibrationError` | PyBAM oracle call | Whether the GPR posterior at the UCB-selected operating point correctly brackets the true discharge capacity |
| `IntervalCoverage` | PyBAM oracle call | Whether the GPR 1σ interval contains the simulated capacity ~68% of the time |
| `VarianceAlignment` | PyBAM oracle call | Whether GPR posterior variance explains prediction error across queried (C-rate, T) points |
| `UncertaintyEvolution` | UCB acquisition | Trend in GPR posterior std at the UCB-selected point — should decrease as the surrogate maps the capacity surface |
| `UncertaintyAnomalies` | UCB acquisition | Anomalous posterior std values indicating the surrogate is exploring far outside its current support |
| `VarianceErrorCorrelation` | PyBAM oracle call | Whether the GPR is most uncertain at the operating points where its capacity prediction is least accurate |

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

`hook.on_step()` is **not called during the Sobol initialisation phase** (no GP model, no posterior). It fires during the BO loop only, after the simulator returns the Fréchet distance and the Ax GP posterior is extracted at the proposed point. `y_pred_mean` / `y_pred_std` are the GP posterior **before** `complete_trial()` incorporates the new observation.

| Check | AL step monitored | What is being observed |
|---|---|---|
| `CalibrationError` | Simulator observation (BO phase only) | Whether the Ax GP posterior at each proposed LED setting correctly brackets the true Fréchet distance |
| `IntervalCoverage` | Simulator observation (BO phase only) | Whether the GP 1σ interval contains the simulated Fréchet value ~68% of the time |
| `VarianceAlignment` | Simulator observation (BO phase only) | Whether GP posterior variance scales with prediction error across proposed colour settings |
| `UncertaintyEvolution` | Ax acquisition (BO phase only) | Trend in GP posterior std at proposed points — expected to decrease as BO concentrates near the optimum |
| `UncertaintyAnomalies` | Ax acquisition (BO phase only) | BO steps where the GP posterior std is anomalously large (exploration excursion) or small (exploitation lock-in) |
| `VarianceErrorCorrelation` | Simulator observation (BO phase only) | Whether the GP is most uncertain at LED settings where it predicts colour distance poorly |
