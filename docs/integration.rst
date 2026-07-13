Integration patterns
====================

:class:`~traits_audit.hook.AuditHook` supports three ways to attach
to an existing loop.  All three produce the same report; pick whichever
requires the fewest changes to the loop.


Pattern 1 — Manual calls
-------------------------

Explicit ``on_step`` / ``on_end`` calls inside the loop body.  Best when
you control the loop directly.

.. code-block:: python

   from traits_audit import AuditHook, AuditPipeline
   from traits_audit.checks import CalibrationErrorCheck, UncertaintyEvolutionCheck

   pipeline = AuditPipeline([
       CalibrationErrorCheck(threshold=0.1),
       UncertaintyEvolutionCheck(),
   ], verbose=True)

   hook = AuditHook(pipeline)

   for step in my_existing_loop:
       mu, sigma = model.predict_with_uncertainty(X)
       hook.on_step(
           y_true=observed,
           y_pred_mean=mu,
           y_pred_std=sigma,
           uncertainty=float(sigma.mean()),
           iteration=step,
       )

   report = hook.on_end()
   print(report.summary())

Keys passed to ``on_step`` are unrestricted.  Each check picks only what
it needs; unused keys are silently ignored.


Pattern 2 — Callback slot
--------------------------

If the loop exposes an ``on_step`` callback, assign the hook once and let
the loop call it automatically.

.. code-block:: python

   hook = AuditHook(pipeline)
   my_loop.on_step = hook.on_step        # assign once

   result = my_loop.run(...)
   report = hook.on_end()


Pattern 3 — Context manager
-----------------------------

Wrap any existing loop call in a ``with`` block.  The pipeline runs
automatically on exit.

.. code-block:: python

   hook = AuditHook(pipeline)
   with hook:
       result = my_loop.run(...)
       # call hook.on_step(...) inside the loop body as usual

   report = hook.report    # available after the with-block

.. note::

   The context manager only runs the pipeline if no exception is raised.
   If ``on_end`` needs batch kwargs that are not in the per-step history
   (e.g. ``y_true``), call it explicitly after the ``with`` block.


Intermediate checks
-------------------

For long-running or irreversible experiments, pass ``check_every=N`` to
run the pipeline periodically and catch anomalies before the loop ends.

.. code-block:: python

   hook = AuditHook(pipeline, check_every=10)

   for step in my_loop:
       hook.on_step(uncertainty=...)

   report = hook.on_end()
   print(f"Intermediate reports: {len(hook.intermediate_reports)}")

Intermediate reports are stored in
:attr:`~traits_audit.hook.AuditHook.intermediate_reports`.


Reusing a hook across runs
--------------------------

Call :meth:`~traits_audit.hook.AuditHook.reset` to clear the
accumulated history and reports, then reuse the same hook object.

.. code-block:: python

   for run_id in experiment_ids:
       hook.reset()
       for step in my_loop.run(run_id):
           hook.on_step(...)
       report = hook.on_end()
       pipeline.save(report, f"audit_{run_id}.json")


Data flow
---------

Data reaches checks via two routes, in priority order:

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Route
     - How to supply
     - When to use
   * - kwargs to ``on_end`` / ``pipeline.run``
     - ``hook.on_end(y_true=y_test, y_pred_mean=mu, y_pred_std=sigma)``
     - Batch arrays assembled after the loop (calibration, coverage)
   * - Per-step history
     - ``hook.on_step(uncertainty=float(sigma.mean()), iteration=i)``
     - Sequences across iterations (trend, anomaly detection)

Both routes are available to every check simultaneously.  A check may use
either or both.  The same ``pipeline.run`` call works for a heterogeneous
mix of batch and sequence checks.
