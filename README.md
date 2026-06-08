# TRAITS Audit

<p align="center">
  <img src="docs/_static/logo.svg" alt="traits-audit logo" width="200">
</p>

![version](https://img.shields.io/badge/version-0.1.0-blue)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![tests](https://github.com/TRustworthy-AI-Tools-for-Science/traits-audit/actions/workflows/ci.yml/badge.svg)
![docs](https://github.com/TRustworthy-AI-Tools-for-Science/traits-audit/actions/workflows/docs-pages.yml/badge.svg)


A flexible uncertainty audit pipeline that hooks into any pre-existing active learning loop.


## Installation

```bash
# as a uv workspace member (recommended — editable, no reinstall needed)
uv sync --all-packages

# standalone into any environment
pip install ./traits-audit

# Install all components at once
pip install "./traits-audit[pybamm,camd,sdl]"
```

## Documentation

This repository uses **Sphinx** with docs source files in `docs/`.

- Local preview:
  ```bash
  pip install -e ".[docs]"
  make -C docs html SPHINXOPTS="-W"
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
ta-demo                          # 40 AL steps
ta-demo --steps 60 --seed 7
ta-demo --help
```

The demo source is at `bin/example_al_pipeline.py` (delegates to
`traits_audit._example`).

## Built-in checks

| Check | Category | What it tests | Required data |
|---|---|---|---|
| `CalibrationErrorCheck` | Aleatoric (model) | [Kuleshov et al. (2018)][kuleshov2018] calibration error | `y_true`, `y_pred_mean`, `y_pred_std` |
| `IntervalCoverageCheck` | Aleatoric (model) | Empirical 1-sigma coverage vs 68.3 % ([Kuleshov et al. (2018)][kuleshov2018]) | `y_true`, `y_pred_mean`, `y_pred_std` |
| `VarianceAlignmentCheck` | Aleatoric (model) | Ratio of predicted to empirical variance ([Levi et al. (2022)][levi2022]) | `y_true`, `y_pred_mean`, `y_pred_std` |
| `UncertaintyEvolutionCheck` | Epistemic | Trend of uncertainty over iterations | `uncertainties` or per-step `uncertainty` |
| `UncertaintyAnomalyCheck` | Epistemic | Z-score anomaly detection on uncertainty | `uncertainties` or per-step `uncertainty` |
| `VarianceErrorCorrelationCheck` | Epistemic | Spearman ρ between std and \|error\| ([Lakshminarayanan et al. (2017)][lakshminarayanan2017]) | `y_true`, `y_pred_mean`, `y_pred_std` |


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
        entropy=float(0.5 * np.log(2 * np.pi * np.e * sigma ** 2).mean()),
    )

report = hook.on_end(y_true=y_test, y_pred_mean=mu_test, y_pred_std=sigma_test)
```


## Uncertainty taxonomy

| Category | Meaning |
|---|---|
| `ALEATORIC_IRREDUCIBLE` | Cannot be reduced by more data (measurement noise, process stochasticity) |
| `ALEATORIC_MODEL` | Calibration error — the model's stated uncertainty does not match empirical coverage |
| `EPISTEMIC` | Reducible uncertainty — shrinks as more observations are collected |
| `UNKNOWN` | Source not yet characterised |

## Run case studies

Four self-contained case studies ship with the package, each demonstrating the
audit on a different active learning domain.  All run without real hardware.

### Demo 1 — Calibration scenarios (1-D benchmark)

> Same as quick start

Compares four calibration regimes (perfectly calibrated, well calibrated,
overconfident, underconfident) on the Forrester benchmark function using a
bootstrap-ensemble surrogate and LCB acquisition.

```bash
ta-demo                                        # 40 steps, all 4 scenarios
ta-demo --steps 100 --seed 7
ta-demo --scenarios overconfident underconfident
ta-demo --help
```

### Demo 2 — PyBAMM Li-ion C-rate optimisation

Finds the (charge-rate, temperature) pair that maximises discharge capacity in
a lithium-ion cell using PyBAMM's Single Particle Model as the oracle and a
scikit-learn GPR with UCB acquisition.

```bash
pip install "./traits-audit[pybamm]"

ta-pybamm-demo                                 # 8 seed evals + 20 UCB steps
ta-pybamm-demo --n-iter 30 --kappa 3.0 --seed 7
ta-pybamm-demo --out-dir _results/pybamm
ta-pybamm-demo --help
```

### Demo 3 — Materials stability screening

Applies query-by-committee active learning to a materials stability dataset
using an AdaBoost committee surrogate.  Performs Lyapunov stability analysis
on the learned surrogate in PCA-reduced feature space after the loop.

```bash
pip install "./traits-audit[camd]"

ta-camd-demo                                   # 50 iterations, 4 queries/iter
ta-camd-demo --n-iter 30 --n-query 6 --seed 7
ta-camd-demo --out-dir _results/camd
ta-camd-demo --help
```

### Demo 4 — Self-driving lab LED colour matching

Runs Bayesian optimisation over a 3-D RGB LED intensity space to minimise the
Fréchet distance to a target colour, using the Ax/BoTorch GP as the surrogate.
Runs in simulation mode — no hardware required.

```bash
pip install "./traits-audit[sdl]"

ta-sdl-demo                                    # 6 Sobol warm-start + 25 BO steps
ta-sdl-demo --n-iter 40 --seed 7
ta-sdl-demo --out-dir _results/sdl
ta-sdl-demo --help
```

## References

- **Kuleshov et al. (2018)** — *Accurate Uncertainties for Deep Learning Using Calibrated Regression.* ICML 2018. [arxiv:1807.00263](https://arxiv.org/abs/1807.00263)
- **Lakshminarayanan et al. (2017)** — *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS 2017. [arxiv:1612.01474](https://arxiv.org/abs/1612.01474)
- **Levi et al. (2022)** — *Evaluating and Calibrating Uncertainty Prediction in Regression Tasks.* [arxiv:1905.11659](https://arxiv.org/abs/1905.11659)

[kuleshov2018]: https://arxiv.org/abs/1807.00263
[lakshminarayanan2017]: https://arxiv.org/abs/1612.01474
[levi2022]: https://arxiv.org/abs/1905.11659
