# Demos

Short, runnable scripts that show how to add `traits-audit` to an experiment.
Each is one file, read top to bottom, with numbered steps. They are also
rendered, with their output and figures, on the
[documentation site](https://trustworthy-ai-tools-for-science.github.io/traits-audit/demos/).

```bash
pip install traits-audit
python demos/01_minimal_loop.py
```

| Demo | What it teaches | Needs |
|---|---|---|
| [`01_minimal_loop.py`](01_minimal_loop.py) | The whole pattern: pipeline, hook, `on_step`, `on_end`. Same example as `ta-demo`. | core |
| [`02_detecting_miscalibration.py`](02_detecting_miscalibration.py) | Which checks catch an over- or under-confident surrogate, and which can't. | core |
| [`03_bayesian_optimization.py`](03_bayesian_optimization.py) | Auditing a BoTorch loop, with an audit every few iterations (`check_every`). | `botorch` |
| [`04_plotting_a_report.py`](04_plotting_a_report.py) | `traits_audit.plotting`: check grid, uncertainty and check evolution, calibration curve. | core |
| [`05_stability_of_the_loop.py`](05_stability_of_the_loop.py) | Whether the loop itself is stable: Lyapunov (local) and DMDc ρ(A) (global). | core |
| [`06_writing_a_custom_check.py`](06_writing_a_custom_check.py) | Turning knowledge about your experiment into a check. | core |
| [`07_logging_to_mlflow.py`](07_logging_to_mlflow.py) | Sending every step, report and figure to MLflow with `MLflowLogger`. | `traits-audit[mlflow]` |
| [`08_existing_framework_callback.py`](08_existing_framework_callback.py) | Auditing a loop you don't own, through the framework's callback. | `scikit-optimize` |

"core" means the library's own dependencies: numpy, scipy, scikit-learn and
matplotlib. Each demo runs in a few seconds.

These demos are for anyone using the library. The case studies from the NMI
2026 paper are a separate package in [`experiments/results`](../experiments/results).

## Rules for adding a demo

- One file, read top to bottom, with `# %%` cells titled `Step N: ...`.
- Public API only: no `_private` imports and nothing from `results`.
- Settings are module-level constants, not command-line flags.
- Under about 150 lines and 10 seconds.
- An optional dependency is named on the first line of the docstring, and the
  script exits with an install hint when it is missing. Add it to
  `_DEMO_EXTRAS` in `docs/conf.py` and to the docs CI job.
