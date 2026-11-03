# TRAITS Audit

<p align="center">
  <img src="https://raw.githubusercontent.com/TRustworthy-AI-Tools-for-Science/traits-audit/main/docs/_static/logo.svg" alt="traits-audit logo" width="200">
</p>

![version](https://img.shields.io/badge/version-0.2.0-blue)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![tests](https://github.com/TRustworthy-AI-Tools-for-Science/traits-audit/actions/workflows/ci.yml/badge.svg)
![docs](https://github.com/TRustworthy-AI-Tools-for-Science/traits-audit/actions/workflows/docs-pages.yml/badge.svg)

A passive uncertainty audit pipeline that hooks into any pre-existing active
learning loop. The hook only observes: it never drives the loop, selects
actions, or queries an oracle.

## Installation

```bash
pip install traits-audit                 # core: numpy, scipy, scikit-learn, matplotlib
pip install "traits-audit[mlflow]"       # + MLflow logging
pip install "traits-audit[stability]"    # + PyDMD windowed stability curves
```

## Quickstart

```bash
ta-demo                    # 60-step loop with a well-calibrated surrogate
ta-demo --std-scale 0.3    # over-confident surrogate: calibration checks fail
```

Use it in your own loop:

```python
from traits_audit import AuditHook, AuditPipeline
from traits_audit.checks import CalibrationErrorCheck, IntervalCoverageCheck

hook = AuditHook(AuditPipeline([
    CalibrationErrorCheck(threshold=0.1),
    IntervalCoverageCheck(expected_coverage=0.683, tolerance=0.1),
]))

for step in my_existing_loop:
    mu, sigma = model.predict_with_uncertainty(x)
    hook.on_step(y_true=observed, y_pred_mean=mu, y_pred_std=sigma, uncertainty=sigma)

print(hook.on_end().summary())
```

The package ships ~35 built-in checks (calibration, coverage, scoring rules,
conformal, PIT, replication, credal, attribution, and DMDc/Lyapunov
stability of the uncertainty trajectory), and you can write your own by
subclassing `traits_audit.AuditCheck`.

## Documentation

API reference, the built-in checks, writing custom checks, runnable demos and
worked case studies: **https://trustworthy-ai-tools-for-science.github.io/traits-audit/**

## Citation

```bibtex
@software{dale2026traitsaudit,
  author  = {Dale, Ashley},
  title   = {{TRAITS Audit}: A Passive Uncertainty Audit Framework for Active Learning Loops},
  year    = {2026},
  url     = {https://github.com/TRustworthy-AI-Tools-for-Science/traits-audit},
  version = {0.2.0},
}
```
