Built-in checks
===============

All built-in checks are importable from :mod:`traits_audit.checks`.

.. code-block:: python

   from traits_audit.checks import (
       CalibrationErrorCheck,
       IntervalCoverageCheck,
       VarianceAlignmentCheck,
       UncertaintyEvolutionCheck,
       UncertaintyAnomalyCheck,
       VarianceErrorCorrelationCheck,
       LyapunovStabilityCheck,
   )


Summary table
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 25 25

   * - Check
     - Category
     - What it measures
     - Required data
   * - :class:`~traits_audit.checks.CalibrationErrorCheck`
     - ``aleatoric_model``
     - Kuleshov (2018) mean calibration error
     - ``y_true``, ``y_pred_mean``, ``y_pred_std``
   * - :class:`~traits_audit.checks.IntervalCoverageCheck`
     - ``aleatoric_model``
     - Empirical 1-σ coverage vs expected 68.3 %
     - ``y_true``, ``y_pred_mean``, ``y_pred_std``
   * - :class:`~traits_audit.checks.VarianceAlignmentCheck`
     - ``aleatoric_model``
     - Ratio of mean predicted variance to mean empirical squared error
     - ``y_true``, ``y_pred_mean``, ``y_pred_std``
   * - :class:`~traits_audit.checks.UncertaintyEvolutionCheck`
     - ``epistemic``
     - Relative slope of uncertainty over iterations
     - ``uncertainties`` kwarg or per-step ``uncertainty``
   * - :class:`~traits_audit.checks.UncertaintyAnomalyCheck`
     - ``epistemic``
     - Fraction of steps with z-score above threshold
     - ``uncertainties`` kwarg or per-step ``uncertainty``
   * - :class:`~traits_audit.checks.VarianceErrorCorrelationCheck`
     - ``epistemic``
     - Spearman ρ between predicted std and absolute error
     - ``y_true``, ``y_pred_mean``, ``y_pred_std``
   * - :class:`~traits_audit.checks.LyapunovStabilityCheck`
     - ``epistemic``
     - Fraction of operating points with :math:`|\lambda_{\max}|` < stability threshold
     - ``lambda_max`` kwarg / history key, or ``surrogate_fn`` + ``op_states``


Calibration checks
------------------

.. autoclass:: traits_audit.checks.CalibrationErrorCheck
   :members:
   :no-index:

.. autoclass:: traits_audit.checks.IntervalCoverageCheck
   :members:
   :no-index:

.. autoclass:: traits_audit.checks.VarianceAlignmentCheck
   :members:
   :no-index:


Uncertainty evolution checks
-----------------------------

.. autoclass:: traits_audit.checks.UncertaintyEvolutionCheck
   :members:
   :no-index:

.. autoclass:: traits_audit.checks.UncertaintyAnomalyCheck
   :members:
   :no-index:

.. autoclass:: traits_audit.checks.VarianceErrorCorrelationCheck
   :members:
   :no-index:


Lyapunov stability check
------------------------

.. autoclass:: traits_audit.checks.LyapunovStabilityCheck
   :members:
   :no-index:


Uncertainty categories
----------------------

.. autoclass:: traits_audit.base.AuditCategory
   :members:
   :no-index:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Value
     - Meaning
   * - ``aleatoric_irreducible``
     - Cannot be reduced by collecting more data (measurement noise,
       process stochasticity)
   * - ``aleatoric_model``
     - Calibration mismatch — the model's stated uncertainty does not
       match empirical coverage
   * - ``epistemic``
     - Reducible uncertainty — shrinks as more observations are collected
   * - ``unknown``
     - Source not yet characterised
