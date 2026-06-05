"""
MLflow logging integration for traits_audit.

``MLflowLogger`` is injected into ``AuditHook`` and called automatically
at each step and at finalisation.  It is the only object in this package
that imports mlflow, so mlflow is not a hard dependency — the rest of the
package works without it.

Usage
-----
::

    import mlflow
    from traits_audit import AuditHook, AuditPipeline
    from traits_audit.mlflow_logger import MLflowLogger

    pipeline = AuditPipeline([...])

    with mlflow.start_run():
        logger = MLflowLogger()                     # attaches to the active run
        hook   = AuditHook(pipeline, logger=logger)

        for step in my_loop:
            hook.on_step(uncertainty=..., ...)

        report = hook.on_end(y_true=..., y_pred_mean=..., y_pred_std=...)

Metric layout in MLflow
-----------------------
Per-step scalars (one point per ``on_step`` call):

    {prefix}/step/{key}                e.g. audit/step/uncertainty

Intermediate audit results (logged every ``check_every`` steps):

    {prefix}/intermediate/{CheckName}          numeric value
    {prefix}/intermediate/{CheckName}/passed   1.0 or 0.0
    {prefix}/intermediate/all_passed           1.0 or 0.0

Final audit results (logged once by ``on_end``):

    {prefix}/final/{CheckName}
    {prefix}/final/{CheckName}/passed
    {prefix}/final/all_passed

Full JSON reports are stored as MLflow artifacts under ``audit/``.

Figure artifacts
----------------
When a check stores visualisation-ready data in ``AuditResult.details``,
``log_report`` automatically generates a PNG figure and logs it under
``audit/figures/{tag}_{safe_name}.png``.

Built-in figure generators
~~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``CalibrationError``  → calibration curve (confidence level vs observed fraction)

Extending figures
~~~~~~~~~~~~~~~~~
Subclass ``MLflowLogger``, update ``self._figure_factories`` in ``__init__``,
and add ``@staticmethod`` factory methods that accept an ``AuditResult`` and
return a ``plt.Figure`` (or ``None`` if required data is absent).

Name matching order
~~~~~~~~~~~~~~~~~~~
1. Exact match on ``result.name``
2. Strip trailing ``"(…)"`` suffix, match on the bare prefix
3. ``startswith`` — e.g. ``"CalibrationError Case 1 (…)"`` → ``"CalibrationError"``
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import AuditReport, AuditResult


FigureFactory = Callable[["AuditResult"], Optional[Any]]  # returns plt.Figure | None


class MLflowLogger:
    """
    Logs per-step metrics, audit reports, and optional figures to an MLflow run.

    All mlflow and matplotlib imports are deferred to method bodies so neither
    is a hard dependency of the package.

    Parameters
    ----------
    run_id : str, optional
        MLflow run ID to log to.  If ``None``, attaches to the currently
        active run (requires an enclosing ``mlflow.start_run()`` context).
    prefix : str
        Root key prefix for all logged metrics (default: ``"audit"``).

    Raises
    ------
    RuntimeError
        If ``run_id`` is ``None`` and no MLflow run is currently active.
    """

    def __init__(self, run_id: Optional[str] = None, prefix: str = "audit"):
        self._explicit_run_id = run_id
        self.prefix = prefix
        self._figure_factories: Dict[str, FigureFactory] = {
            "CalibrationError": self._fig_calibration_curve,
        }

    # ------------------------------------------------------------------
    # Logger protocol — called by AuditHook
    # ------------------------------------------------------------------

    def log_step(self, step_idx: int, **kwargs: Any) -> None:
        """
        Log numeric values from one ``hook.on_step()`` call as MLflow metrics.

        Non-numeric values and keys starting with ``_`` (internal) are
        silently skipped.

        Parameters
        ----------
        step_idx : int
            The zero-indexed loop step (used as the MLflow ``step``).
        **kwargs
            All key-value pairs passed to ``hook.on_step()``.
        """
        metrics = {
            f"{self.prefix}/step/{k}": float(v)
            for k, v in kwargs.items()
            if isinstance(v, (int, float)) and not k.startswith("_")
        }
        if metrics:
            self._log_metrics(metrics, step=step_idx)

    def log_report(
        self,
        report: "AuditReport",
        step: int,
        tag: str = "final",
    ) -> None:
        """
        Log all ``AuditResult`` values and pass/fail flags as MLflow metrics,
        persist the full JSON report as an MLflow artifact, and generate PNG
        figure artifacts for any result whose check name matches a registered
        figure factory.

        Parameters
        ----------
        report : AuditReport
        step : int
            Global step value at the time of logging.
        tag : str
            Sub-key distinguishing intermediate from final reports
            (``"intermediate"`` or ``"final"``).
        """
        # ── metrics ──────────────────────────────────────────────────────
        metrics: Dict[str, float] = {}
        for r in report.results:
            if r.value is not None:
                metrics[f"{self.prefix}/{tag}/{r.name}"] = float(r.value)
            metrics[f"{self.prefix}/{tag}/{r.name}/passed"] = float(r.passed)
        metrics[f"{self.prefix}/{tag}/all_passed"] = float(report.passed)

        self._log_metrics(metrics, step=step)
        self._log_artifact(report.to_dict(), artifact_file=f"audit/{tag}_{step}_report.json")

        # ── figures ───────────────────────────────────────────────────────
        for result in report.results:
            factory = self._match_factory(result.name)
            if factory is None or not result.details:
                continue
            try:
                fig = factory(result)
                if fig is not None:
                    safe = (
                        result.name.lower()
                        .replace(" ", "_").replace("(", "").replace(")", "")
                    )
                    self._log_figure(fig, f"audit/figures/{tag}_{safe}.png")
                    import matplotlib.pyplot as plt
                    plt.close(fig)
            except Exception as exc:
                import warnings
                warnings.warn(
                    f"MLflowLogger: figure skipped for {result.name!r}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    # ------------------------------------------------------------------
    # Figure factory helpers
    # ------------------------------------------------------------------

    def _match_factory(self, name: str) -> Optional[FigureFactory]:
        """Three-level factory lookup: exact → strip-suffix prefix → startswith."""
        # 1. Exact match
        f = self._figure_factories.get(name)
        # 2. Strip trailing "(…)" suffix
        if f is None:
            f = self._figure_factories.get(name.split("(")[0].strip())
        # 3. startswith — handles "CalibrationError Case 1 (…)" → "CalibrationError"
        if f is None:
            for key, fn in self._figure_factories.items():
                if name.startswith(key):
                    f = fn
                    break
        return f

    @staticmethod
    def _fig_calibration_curve(result: "AuditResult") -> Optional[Any]:
        """Calibration reliability diagram for ``CalibrationErrorCheck`` results."""
        from traits_audit._viz import _fig_calibration_curve as _viz_cal_curve
        return _viz_cal_curve(result)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_id(self) -> str:
        if self._explicit_run_id is not None:
            return self._explicit_run_id
        import mlflow
        active = mlflow.active_run()
        if active is None:
            raise RuntimeError(
                "No active MLflow run.  Either pass run_id= to MLflowLogger "
                "or wrap your code in a `with mlflow.start_run():` block."
            )
        return active.info.run_id

    def _log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log a dict of metrics via MlflowClient.log_metric (universally supported)."""
        import mlflow
        client = mlflow.MlflowClient()
        run_id = self._run_id()
        for key, value in metrics.items():
            client.log_metric(run_id, key, value, step=step)

    def _log_artifact(self, data: dict, artifact_file: str) -> None:
        """Serialise ``data`` to JSON and upload as an MLflow artifact."""
        import mlflow
        artifact_dir = os.path.dirname(artifact_file)
        basename = os.path.basename(artifact_file)
        with tempfile.TemporaryDirectory() as tmp:
            local_path = os.path.join(tmp, basename)
            with open(local_path, "w") as f:
                json.dump(data, f, indent=2)
            mlflow.MlflowClient().log_artifact(
                self._run_id(),
                local_path,
                artifact_path=artifact_dir or None,
            )

    def _log_figure(self, fig: Any, artifact_path: str) -> None:
        """Save ``fig`` as a PNG and upload it as an MLflow artifact."""
        import mlflow
        artifact_dir = os.path.dirname(artifact_path)
        basename = os.path.basename(artifact_path)
        with tempfile.TemporaryDirectory() as tmp:
            local_path = os.path.join(tmp, basename)
            fig.savefig(local_path, dpi=300, bbox_inches="tight")
            mlflow.MlflowClient().log_artifact(
                self._run_id(),
                local_path,
                artifact_path=artifact_dir or None,
            )
