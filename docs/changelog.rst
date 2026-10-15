Changelog
=========

0.2.0 (2026-10-09)
-------------------

Stability analysis consolidated into :mod:`traits_audit.stability` (a
behavior-preserving refactor of code that was spread across traits-audit and
battery-forecast's Experiments 3 and 4):

- ``traits_audit.dmdc`` and ``traits_audit.detrend`` moved to
  ``traits_audit.stability.dmdc`` / ``traits_audit.stability.detrend``; the
  old import paths remain as aliases. ``detrend_trajectory`` is now public
  (``_detrend_trajectory`` kept as an alias).
- New ``stability.lyapunov_eq``: the single implementation of the numerical
  Jacobian (scalar and batched), discrete Lyapunov ``P`` / spectral-radius
  rescaling and the gradient-descent predictor, previously copied in
  ``_viz.py``, ``_lyapunov.py``, ``checks/lyapunov.py`` and ``dmdc.py``.
- New ``stability.lle``: Rosenstein largest-Lyapunov-exponent estimation
  (:func:`~traits_audit.stability.lle.rosenstein_divergence`, verified equal to
  ``nolitsa.lyapunov.mle``), rolling-window variant and Eckmann-Ruelle-style
  regime classification. ``nolitsa`` is no longer required.
- New ``stability.windowed`` (windowed DMDc stability curves, detrended-frame
  R² scoring; the PyDMD variant needs the new ``stability`` extra),
  ``stability.regime`` (regime split, background trend) and
  ``stability.action_hygiene`` (drop numerically-constant control columns).
- New ``stability.viz``: stability figure generators with injectable
  :class:`~traits_audit.stability.viz.style.PlotStyle`; no global
  ``rcParams`` mutation on import (use ``style_context()``).
- Removed the unused ``traits_audit._lyapunov`` module (``_viz`` is the live
  copy).

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
