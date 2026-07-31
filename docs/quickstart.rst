Quickstart
==========

The fastest way to see the package in action is the built-in demo.


Running the demo
----------------

Install the package and run:

.. code-block:: bash

   ta-demo

The demo runs a 100-step active learning loop in which a bootstrap-ensemble
polynomial surrogate learns a 1-D target function via lower-confidence-bound
acquisition.  At each step the loop pushes predictions to
:class:`~traits_audit.hook.AuditHook`, which triggers an intermediate audit
every 10 steps.

Options:

.. code-block:: bash

   ta-demo --steps 60 --seed 7       # more steps, different seed
   ta-demo --check-every 5           # intermediate audit every 5 steps
   ta-demo --help


Minimal example
-------------------------------

.. code-block:: python

   import numpy as np
   from traits_audit import AuditHook, AuditPipeline
   from traits_audit.checks import (
       CalibrationErrorCheck,
       IntervalCoverageCheck,
       UncertaintyEvolutionCheck,
   )

   rng = np.random.default_rng(0)
   n   = 60

   pipeline = AuditPipeline([
       CalibrationErrorCheck(threshold=0.1),
       IntervalCoverageCheck(expected_coverage=0.683, tolerance=0.1),
       UncertaintyEvolutionCheck(),
   ], verbose=True)

   hook = AuditHook(pipeline)

   for i in range(n):
       sigma = float(1.0 / (i + 1) ** 0.5)          # decaying uncertainty
       y_true = float(rng.normal(0, 1))
       mu     = float(rng.normal(0, 0.1))
       hook.on_step(y_true=y_true, y_pred_mean=mu, y_pred_std=sigma,
                    uncertainty=sigma)

   report = hook.on_end()
   print(report.summary())

What each step does:

1. The loop calls :meth:`~traits_audit.hook.AuditHook.on_step` with
   whatever values the checks need — predictions, uncertainties, ground truth.
2. After the loop, :meth:`~traits_audit.hook.AuditHook.on_end` runs the
   full pipeline and returns an :class:`~traits_audit.base.AuditReport`.
3. Each :class:`~traits_audit.base.AuditResult` in the report records
   ``name``, ``passed``, ``value``, ``threshold``, and a one-line ``message``.
