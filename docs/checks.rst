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

Calibration error
~~~~~~~~~~~~~~~~~

.. figure:: _static/built_in_metrics/calibration_error.png
   :width: 90%
   :alt: Example of good and bad calibration error.

   Calibration error is the area between the collected data scatter plot 
   and the parity line. (Left) Example with minimal calibration error. (Right)
   Example with high calibration error.

.. autoclass:: traits_audit.checks.CalibrationErrorCheck
   :members:
   :no-index:

Interval coverage
~~~~~~~~~~~~~~~~~

.. figure:: _static/built_in_metrics/interval_coverage.png
   :width: 90%
   :alt: Example of interval coverage metric.

   Interval coverage checks that most points are within the variance of the 
   model. (Left) Example of passing interval coverage. (Right) Example of
   failing interval coverage.

.. autoclass:: traits_audit.checks.IntervalCoverageCheck
   :members:
   :no-index:

Variance alignment
~~~~~~~~~~~~~~~~~~

.. figure:: _static/built_in_metrics/variance_alignment.png
   :width: 90%
   :alt: Example of variance alignment.
   
   The variance and error should agree. (Left) Example of agreement between the
   predicted variance and the mean empirical squared error. (Right) Example of
   disagreement between predicted variance and mean empirical squared error.

.. autoclass:: traits_audit.checks.VarianceAlignmentCheck
   :members:
   :no-index:


Uncertainty evolution checks
-----------------------------

Uncertainty evolution
~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/built_in_metrics/uncertainty_evolution.png
   :width: 90%
   :alt: Example of uncertainty evolution.

   Imposes a linear constraint on how uncertainty changes as the system evolves. Example
   shown is for updates during active learning (AL). (Left) Uncertainty decreases linearly
   as more data is acquired. (Right) Uncertainty decreases monotonically but not linearly.

.. autoclass:: traits_audit.checks.UncertaintyEvolutionCheck
   :members:
   :no-index:

Uncertainty anomalies
~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/built_in_metrics/uncertainty_anomaly.png
   :width: 90%
   :alt: Example of uncertainty anomalies.

   Requires that uncertainty estimates stay within 3 ..math:`\sigma` of the mean uncertainty. 
   (Left) Example with no uncertainty anomalies. (Right) Example with multiple anomalies.

.. autoclass:: traits_audit.checks.UncertaintyAnomalyCheck
   :members:
   :no-index:

Variance-error correlation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/built_in_metrics/variance_error_correlation.png
   :width: 90%
   :alt: Example of variance-error correlation.

   Requires that the predicted variance is positively correlated with the empirical error. 
   (Left) Example with positive correlation between predicted variance and absolute error.
   (Right) Example with negative correlation between predicted variance and absolute error.

.. autoclass:: traits_audit.checks.VarianceErrorCorrelationCheck
   :members:
   :no-index:


Lyapunov stability check
------------------------

.. figure:: _static/built_in_metrics/lyapunov_stability.png
   :width: 90%
   :alt: Example of Lyapunov stability check.

   Requires that the maximum eigenvalue of the Jacobian is less than a threshold. (Left) Example
   with all operating points having :math:`|\lambda_{\max}|` < 1. (Right) Example with some operating
   points having :math:`|\lambda_{\max}|` > 1.

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
