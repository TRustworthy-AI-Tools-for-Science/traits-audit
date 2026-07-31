Changelog
=========

0.1.2 (2026-07-02)
-------------------

- ``uv sync`` now installs all demo extras (mlflow, camd, pybamm, sdl) by
  default via ``[dependency-groups]``; no extra flags required.
- Removed ``camd`` / ``qmpy-tri`` legacy compatibility notes — the demo uses
  a scikit-learn BaggingRegressor surrogate and does not require either package.
- ``requires-python`` raised to ``≥ 3.11`` (required by ax-platform ≥ 1.3.0).
- Fixed ``UncertaintyEvolutionCheck`` being called with a nonexistent
  ``slope_threshold`` argument in all four demos.
- Four-panel ``oracle_uncertainty_panel.png`` replaces individual per-scenario
  oracle figures in the calibration demo.
- Smoke tests added for all four demo entry points; mlflow is stubbed so
  tests run without a real MLflow installation.
- ``fig_dir.mkdir`` calls updated to ``parents=True`` in the calibration demo.


0.1.0 (2026-05-23)
-------------------

- Initial release as a uv-installable package (src layout, hatchling backend).
- :class:`~traits_audit.hook.AuditHook` with manual, callback, and
  context-manager integration patterns.
- :class:`~traits_audit.pipeline.AuditPipeline` with JSON persistence
  and merge support.
- Six built-in checks:
  :class:`~traits_audit.checks.CalibrationErrorCheck`,
  :class:`~traits_audit.checks.IntervalCoverageCheck`,
  :class:`~traits_audit.checks.VarianceAlignmentCheck`,
  :class:`~traits_audit.checks.UncertaintyEvolutionCheck`,
  :class:`~traits_audit.checks.UncertaintyAnomalyCheck`,
  :class:`~traits_audit.checks.VarianceErrorCorrelationCheck`.
- Optional :class:`~traits_audit.mlflow_logger.MLflowLogger` with
  per-step metrics, intermediate reports, and JSON artifact upload.
- ``ta-demo`` CLI entry point demonstrating a full bootstrap-ensemble AL loop
  with MLflow logging.
