# TRAITS Audit

<p align="center">
  <img src="docs/_static/logo.svg" alt="traits-audit logo" width="200">
</p>

![version](https://img.shields.io/badge/version-0.1.3-blue)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![tests](https://github.com/TRustworthy-AI-Tools-for-Science/traits-audit/actions/workflows/ci.yml/badge.svg)
![docs](https://github.com/TRustworthy-AI-Tools-for-Science/traits-audit/actions/workflows/docs-pages.yml/badge.svg)

A flexible uncertainty audit pipeline that hooks into any pre-existing active learning loop.

## Installation

```bash
# uv workspace (recommended — installs all demos + mlflow, editable)
uv sync

# standalone pip install
pip install "."
pip install ".[mlflow,camd,pybamm,sdl]"   # with all demo extras
```

## Quickstart

```bash
ta-cal-demo                          # 100 AL steps, 4 calibration scenarios
ta-cal-demo --steps 60 --seed 7
```

The demo runs a bootstrap-ensemble surrogate on the Forrester benchmark with LCB
acquisition, fully wired to the audit pipeline.

## Documentation

Full API reference, built-in checks, custom check examples, and worked case studies:

**https://trustworthy-ai-tools-for-science.github.io/traits-audit/**

## Citation

```bibtex
@software{dale2026traitsaudit,
  author  = {Dale, Ashley},
  title   = {{TRAITS Audit}: A Passive Uncertainty Audit Framework for Active Learning Loops},
  year    = {2026},
  url     = {https://github.com/TRustworthy-AI-Tools-for-Science/traits-audit},
  version = {0.1.3},
}
```
