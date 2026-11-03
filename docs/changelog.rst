Changelog
=========

0.2.0 (unreleased)
------------------

Repository split into a library and a paper-reproduction package:

- ``traits-audit`` (``packages/traits-audit``) is now only the generic audit
  library. Core dependencies are numpy, scipy, scikit-learn and matplotlib;
  ``botorch`` and ``kaleido`` are no longer required.
- The case-study demos (``ta-cal-demo``, ``ta-sdl-demo``, ``ta-mobo-demo``,
  ``ta-pybamm-demo``, ``ta-camd-demo``), the committee-RL study and the figure
  code moved to the new ``traits-audit-results`` package
  (``experiments/results``). Imports such as ``traits_audit._viz`` and
  ``traits_audit.committee`` are now ``results.viz`` / ``results.committee``.
- ``ta-demo`` is a new minimal quickstart that needs only core dependencies.
- New checks for complete metric coverage: replication, procedural, ergodic,
  credal, aleatoric/epistemic, tail, decision, provenance and attribution
  (see :doc:`checks`).
- Removed the ``mlflow_logger`` dependency on demo plotting code.
- New :mod:`traits_audit.plotting` (matplotlib, core dependencies only):
  ``plot_check_grid``, ``plot_uncertainty_evolution``,
  ``plot_audit_evolution`` and ``plot_calibration_curve``, plus the shared
  display logic ``result_status``, ``CHECK_ABBREVIATIONS``,
  ``split_trackable_checks`` and ``filter_stage_reports``. Each function
  returns a figure for the caller to save. ``MLflowLogger`` now uses
  ``plot_calibration_curve``.
- New ``demos/`` folder: eight short, runnable scripts (minimal loop,
  miscalibration, BoTorch, plotting, loop stability, custom checks, MLflow,
  framework callbacks), rendered in the docs by sphinx-gallery (added to the
  ``docs`` extra). Demo 01 is the ``ta-demo`` example.
- Fixed check-grid colours: a failing ``UncertaintyEvolution`` was shaded as
  deeply passing (its value is now a count of flagged channels, not a slope),
  and a cell's shade now always agrees with the check's verdict. This also
  corrects the paper's check-grid figures.
- Checks skipped for too few samples (PIT, conformal coverage, CRPS, NLL,
  interval score, uncertainty anomalies) now say ``Skipped — too few …``, so
  plots draw them as skipped rather than report-only. A result with neither a
  value nor a threshold is also treated as skipped.
- Documented when ``UncertaintyEvolutionCheck`` applies. It fails on any
  sustained decline in σ, so a converging active-learning or BO loop fails
  it. The docstring, the Checks page and the quickstart now say so, and the
  Checks page figure caption no longer presents a decline as the expected
  case. ``docs/scripts/visualize_builtin_metrics.py`` now takes that figure's
  verdicts and threshold from the check itself, and ``--only NAME`` regenerates
  a single figure.
- New API page for :mod:`traits_audit.trajectory`.


Stability analysis consolidated into :mod:`traits_audit.stability` (a
behavior-preserving refactor of code that was spread across traits-audit and
battery-forecast's Experiments 3 and 4):

- ``traits_audit.dmdc`` and ``traits_audit.detrend`` moved to
  ``traits_audit.stability.dmdc`` / ``traits_audit.stability.detrend``. The
  old import paths have been removed (no aliases). ``detrend_trajectory`` is
  now public.
- ``traits_audit.checks.lyapunov`` exports only ``LyapunovStabilityCheck``;
  import the Jacobian / Lyapunov helpers from
  ``traits_audit.stability.lyapunov_eq``. The unused ``reference_model``
  argument of ``plot_dmdc_quality`` was removed.
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
