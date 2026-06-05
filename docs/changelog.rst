Changelog
=========

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
