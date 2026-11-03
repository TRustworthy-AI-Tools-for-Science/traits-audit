Quickstart
==========

The fastest way to see the package in action is the built-in demo.


Running the demo
----------------

Install the package and run:

.. code-block:: bash

   ta-demo

The demo runs a 60-step active learning loop in which a bootstrap ensemble of
cubic polynomials learns a noisy 1-D function by uncertainty sampling.  Each
step pushes the prediction for the queried point to
:class:`~traits_audit.hook.AuditHook`; the pipeline audits the history at the
end and prints a report.

Options:

.. code-block:: bash

   ta-demo --std-scale 0.3          # over-confident surrogate: calibration checks fail
   ta-demo --std-scale 3.0          # under-confident surrogate
   ta-demo --steps 100 --seed 7
   ta-demo --help

The same loop, explained step by step, is the first of the :doc:`demos <demos/index>`.
The other demos add one idea each: detecting miscalibration, a BoTorch loop,
plotting, loop stability, custom checks, MLflow and framework callbacks.

The full case studies from the paper (``ta-cal-demo``, ``ta-sdl-demo``, …)
are in the separate ``traits-audit-results`` experiment package; see
:doc:`installation`.


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

All three checks fail here, and they should. The outcomes vary like
:math:`\mathcal{N}(0, 1)`, but the model's σ shrinks toward 0.13, so it is
overconfident (``CalibrationError`` and ``IntervalCoverage``), and its σ
declines steadily (``UncertaintyEvolution``). In a real loop, a steady decline
in σ is often expected as data accumulates; see
:ref:`when UncertaintyEvolutionCheck applies <uncertainty-evolution>`.

What each step does:

1. The loop calls :meth:`~traits_audit.hook.AuditHook.on_step` with
   whatever values the checks need — predictions, uncertainties, ground truth.
2. After the loop, :meth:`~traits_audit.hook.AuditHook.on_end` runs the
   full pipeline and returns an :class:`~traits_audit.base.AuditReport`.
3. Each :class:`~traits_audit.base.AuditResult` in the report records
   ``name``, ``passed``, ``value``, ``threshold``, and a one-line ``message``.
