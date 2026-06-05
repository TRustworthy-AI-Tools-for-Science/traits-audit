MLflow integration
==================

Pass a :class:`~traits_audit.mlflow_logger.MLflowLogger` to
:class:`~traits_audit.hook.AuditHook` and metrics are logged
automatically — no changes to the loop or to any check.

``mlflow`` is imported lazily, so the rest of the package works without it.
Install with:

.. code-block:: bash

   pip install "traits-audit[mlflow]"


Launching the UI
----------------

Use a SQLite backend — the file-store backend is deprecated in recent MLflow
versions:

.. code-block:: bash

   mlflow ui --backend-store-uri sqlite:///my_runs.db

Open the **Model training** tab (not the GenAI tab) to see experiment runs.


Basic usage
-----------

.. code-block:: python

   import mlflow
   from traits_audit import AuditHook, AuditPipeline, MLflowLogger
   from traits_audit.checks import CalibrationErrorCheck, UncertaintyEvolutionCheck

   pipeline = AuditPipeline([
       CalibrationErrorCheck(threshold=0.1),
       UncertaintyEvolutionCheck(),
   ])

   mlflow.set_tracking_uri("sqlite:///my_runs.db")

   with mlflow.start_run():
       logger = MLflowLogger()          # attaches to the active run automatically
       hook   = AuditHook(pipeline, logger=logger)

       for step in my_loop:
           mu, sigma = model.predict_with_uncertainty(X)
           hook.on_step(
               uncertainty=float(sigma.mean()),
               y_true=observed, y_pred_mean=mu, y_pred_std=sigma,
           )

       report = hook.on_end()


Explicit run ID
---------------

.. code-block:: python

   logger = MLflowLogger(run_id="your-run-id-here")
   hook   = AuditHook(pipeline, logger=logger)


Intermediate checks + logging
------------------------------

.. code-block:: python

   with mlflow.start_run():
       logger = MLflowLogger()
       hook   = AuditHook(pipeline, check_every=10, logger=logger)
       # pipeline is run (and logged) after every 10 steps

The intermediate reports are logged under the ``intermediate`` tag and
stored in :attr:`~traits_audit.hook.AuditHook.intermediate_reports`.


What gets logged
----------------

**Per step** — one MLflow metric point per :meth:`~traits_audit.hook.AuditHook.on_step`
call, for every numeric kwarg:

.. code-block:: text

   audit/step/uncertainty       (float, step=i)
   audit/step/y_pred_mean       (float, step=i)
   # any other float/int passed to on_step()

**Intermediate reports** — after every ``check_every`` steps:

.. code-block:: text

   audit/intermediate/CalibrationError          (float)
   audit/intermediate/CalibrationError/passed   (1.0 or 0.0)
   audit/intermediate/all_passed                (1.0 or 0.0)

**Final report** — logged once by :meth:`~traits_audit.hook.AuditHook.on_end`:

.. code-block:: text

   audit/final/CalibrationError
   audit/final/CalibrationError/passed
   audit/final/all_passed

**Artifacts** — the full JSON report under ``audit/`` in the artifact store:

.. code-block:: text

   audit/final_report.json
   audit/intermediate_report.json   (if check_every is set)


Custom prefix
-------------

.. code-block:: python

   logger = MLflowLogger(prefix="exp1/audit")
   # logs as: exp1/audit/step/uncertainty, exp1/audit/final/CalibrationError, …


Custom logger
-------------

Any object with ``log_step`` and ``log_report`` methods works as the
``logger`` argument — no inheritance required.

.. code-block:: python

   class PrintLogger:
       """Minimal logger that prints to stdout — useful for debugging."""

       def log_step(self, step_idx: int, **kwargs) -> None:
           nums = {k: v for k, v in kwargs.items() if isinstance(v, (int, float))}
           print(f"  step {step_idx}: {nums}")

       def log_report(self, report, step: int, tag: str = "final") -> None:
           print(f"  [{tag} @ step {step}] {report.n_passed}/{len(report.results)} passed")

   hook = AuditHook(pipeline, logger=PrintLogger())


API
---

.. autoclass:: traits_audit.mlflow_logger.MLflowLogger
   :members:
   :no-index:
