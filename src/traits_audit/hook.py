"""
AuditHook — the integration point between any external loop and the pipeline.

The hook is a passive recorder.  It never drives a loop, selects actions, or
queries an oracle.  It only accumulates per-step data that the loop chooses to
push, then triggers the pipeline when the loop signals it is done.

Three integration patterns
--------------------------

**1. Manual (explicit calls after each step)**::

    hook = AuditHook(pipeline)

    for step in my_existing_loop:
        hook.on_step(y_true=y, y_pred_mean=mu, y_pred_std=sigma, uncertainty=std)

    report = hook.on_end()

**2. Callback slot (if the loop exposes an on_step callback)**::

    hook = AuditHook(pipeline)
    my_loop.on_step = hook.on_step       # assign once
    result = my_loop.run(...)
    report = hook.on_end()

**3. Context manager (wrap any existing loop call)**::

    hook = AuditHook(pipeline)
    with hook:
        result = my_existing_loop.run(...)
        # call hook.on_step(...) inside the loop body as normal

    report = hook.report    # available after __exit__

Intermediate checks
-------------------
Pass ``check_every=N`` to run the pipeline after every N steps in addition
to the final run.  Intermediate reports are stored in ``hook.intermediate_reports``.
This is useful for long-running loops where early anomaly detection matters.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import AuditPipeline
    from .base import AuditReport


class AuditHook:
    """
    Thin adapter between any external active learning loop and an AuditPipeline.

    Parameters
    ----------
    pipeline : AuditPipeline
        The pipeline to run when :meth:`on_end` is called (or on context-manager exit).
    check_every : int, optional
        If set, also run the pipeline after every ``check_every`` steps and
        store the result in :attr:`intermediate_reports`.
    logger : MLflowLogger or any object with log_step / log_report, optional
        If provided, metrics are logged after every ``on_step`` call and after
        ``on_end``.  Use :class:`~traits_audit.mlflow_logger.MLflowLogger`
        for MLflow, or supply any object that implements the same two methods.
    """

    def __init__(
        self,
        pipeline: "AuditPipeline",
        check_every: Optional[int] = None,
        logger: Any = None,
    ):
        self._pipeline = pipeline
        self._check_every = check_every
        self._logger = logger
        self._history: List[Dict[str, Any]] = []
        self._report: Optional["AuditReport"] = None
        self.intermediate_reports: List["AuditReport"] = []

    # ------------------------------------------------------------------
    # Core interface — called by the external loop
    # ------------------------------------------------------------------

    def on_step(self, **kwargs: Any) -> None:
        """
        Record data from one loop step.

        Call this inside your loop body after each oracle query.  Pass any
        named values the audit checks will need — predictions, uncertainties,
        ground truth, iteration index, etc.  Keys are unrestricted.

        Examples
        --------
        ::

            hook.on_step(
                y_true=observed_state,
                y_pred_mean=predicted_mean,
                y_pred_std=predicted_std,
                uncertainty=float(predicted_std.mean()),
                iteration=i,
            )
        """
        step_idx = len(self._history)
        self._history.append({"_step": step_idx, **kwargs})

        if self._logger is not None:
            self._logger.log_step(step_idx, **kwargs)

        if self._check_every and len(self._history) % self._check_every == 0:
            report = self._pipeline.run(list(self._history))
            self.intermediate_reports.append(report)
            if self._logger is not None:
                self._logger.log_report(report, step=len(self._history), tag="intermediate")

    def on_end(self, **kwargs: Any) -> "AuditReport":
        """
        Finalise: run the pipeline on the full accumulated history.

        Parameters
        ----------
        **kwargs
            Additional arrays or objects forwarded directly to each check
            (e.g. a held-out test set not available step-by-step).

        Returns
        -------
        AuditReport
            Also stored in :attr:`report` for later access.
        """
        self._report = self._pipeline.run(list(self._history), **kwargs)
        if self._logger is not None:
            self._logger.log_report(self._report, step=len(self._history), tag="final")
        return self._report

    # ------------------------------------------------------------------
    # Context-manager integration
    # ------------------------------------------------------------------

    def __enter__(self) -> "AuditHook":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None:
            try:
                self._report = self._pipeline.run(list(self._history))
                if self._logger is not None:
                    self._logger.log_report(self._report, step=len(self._history), tag="final")
            except KeyError as e:
                import warnings
                warnings.warn(
                    f"AuditHook: pipeline raised KeyError({e!s}) on exit — report not set. "
                    "Call on_end(...) explicitly and pass any batch data the checks require.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def report(self) -> "AuditReport":
        """The most recent final report. Raises if on_end has not been called."""
        if self._report is None:
            raise RuntimeError("No report yet — call on_end() or exit the context manager.")
        return self._report

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Read-only view of the accumulated step data."""
        return list(self._history)

    @property
    def latest_uncertainty_vector(self):
        """Per-parameter variance vector from the most recent on_step that supplied one.

        Returns the ``uncertainty_vector`` kwarg from the latest history entry
        that contains it, or ``None`` if no such entry exists.  This lets an
        active learning policy read the current audit uncertainty without having
        to inspect the full history itself.

        Usage::

            hook = AuditHook(pipeline)

            # Inside your measurement callback:
            hook.on_step(uncertainty_vector=per_param_variances, ...)

            # As the uncertainty_fn for StateSpacePolicy / run_state_space_experiment:
            uncertainty_fn = lambda state: hook.latest_uncertainty_vector
        """
        for step in reversed(self._history):
            if "uncertainty_vector" in step:
                import numpy as np
                return np.asarray(step["uncertainty_vector"])
        return None

    def reset(self) -> None:
        """Clear accumulated history and reports (reuse the hook for a new run)."""
        self._history.clear()
        self._report = None
        self.intermediate_reports.clear()

    # ------------------------------------------------------------------
    # Logger protocol (duck-typed, for reference)
    # ------------------------------------------------------------------
    #
    # Any object passed as `logger` must implement:
    #
    #   log_step(step_idx: int, **kwargs) -> None
    #       Called after every on_step(). Receives the same kwargs.
    #
    #   log_report(report: AuditReport, step: int, tag: str) -> None
    #       Called after intermediate pipeline runs and after on_end().
    #       `tag` is "intermediate" or "final".
