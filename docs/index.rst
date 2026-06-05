traits-audit
=================

A flexible uncertainty audit pipeline that hooks into any pre-existing
active learning loop.  No loop, no acquisition function, no oracle — only
observation. Basic built-in uncertainty metrics, with the option to build 
your own.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   quickstart
   integration
   checks
   custom_checks
   mlflow
   hook_pipeline_map

.. toctree::
   :maxdepth: 2
   :caption: Case studies

   demo_calibration
   demo_camd
   demo_pybamm
   demo_sdl

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: About

   changelog


At a glance
-----------

.. code-block:: python

   from traits_audit import AuditHook, AuditPipeline
   from traits_audit.checks import CalibrationErrorCheck, UncertaintyEvolutionCheck

   pipeline = AuditPipeline([
       CalibrationErrorCheck(threshold=0.1),
       UncertaintyEvolutionCheck(slope_threshold=-0.05),
   ])

   hook = AuditHook(pipeline)

   for step in my_existing_loop:
       mu, sigma = model.predict_with_uncertainty(X)
       hook.on_step(y_true=observed, y_pred_mean=mu, y_pred_std=sigma,
                    uncertainty=float(sigma.mean()))

   report = hook.on_end()
   print(report.summary())


Design principles
-----------------

**Passive observation.**
The hook records data; it never drives the loop, selects actions, or
queries an oracle.

**No coupling to model or loop structure.**
Checks receive a plain ``list[dict]`` (one per step) plus any named
arrays you choose to pass.  Nothing in the package assumes a specific
model family, data shape, or loop framework.

**Non-blocking pipeline.**
A failing check produces an :class:`~traits_audit.base.AuditResult`
with ``passed=False``.  It never raises.  All checks always run.

**Optional MLflow integration.**
:class:`~traits_audit.mlflow_logger.MLflowLogger` is injected as a
dependency; ``mlflow`` itself is an optional extra.  The rest of the
package works without it.
