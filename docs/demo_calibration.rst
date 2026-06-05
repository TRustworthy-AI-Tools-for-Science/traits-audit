.. _demo-calibration:

Calibration scenarios demo (``ta-demo``)
=========================================

This demo compares four active learning runs side-by-side to show how the
audit checks behave across the full calibration spectrum: ideal calibration, a
well-calibrated baseline, an overconfident system and an underconfident system.

.. code-block:: bash

   ta-demo                                          # all four scenarios, 100 steps
   ta-demo --scenarios perfectly_calibrated         
   ta-demo --scenarios well_calibrated overconfident underconfident
   ta-demo --steps 60 --seed 7


Introduction
------------

Calibration is the alignment between a model's reported uncertainty
and its empirical error rate. Overconfident models produce narrow uncertainty intervals that 
miss observations; underconfident models report wide uncertainty estimates with
little decision-relevant information.

**Questions:** (1) Can the six audit checks reliably distinguish a
well-calibrated ensemble from overconfident and underconfident variants of the
same surrogate on identical data?  (2) What does the ideal calibration
trajectory look like when the surrogate is paired with a correctly-specified
noise model and a simple oracle that allows convergence?

Uncertainty hook placement
~~~~~~~~~~~~~~~~~~~~~~~~~~

``hook.on_step()`` is called **after** each oracle evaluation and **before**
the surrogate is re-fit, so every check receives the surrogate's pre-update
prediction at the LCB-selected point:

.. code-block:: text

   Surrogate fit  →  LCB acquisition  →  Oracle query  ← hook.on_step()
        ↑                                      |
        └──────────── add observation ─────────┘

.. list-table:: Check-to-pipeline-step mapping
   :header-rows: 1
   :widths: 30 25 45

   * - Check
     - AL step monitored
     - What is observed
   * - ``CalibrationError``
     - Oracle query
     - Whether the bootstrap posterior at the acquired point correctly brackets the oracle outcome, accumulated over all steps
   * - ``IntervalCoverage``
     - Oracle query
     - Whether the 1σ bootstrap interval contains the oracle value ~68 % of the time
   * - ``VarianceAlignment``
     - Oracle query
     - Whether bootstrap ensemble spread explains prediction error globally
   * - ``UncertaintyEvolution``
     - LCB acquisition
     - Trend in posterior std at the LCB-selected point — expected to decrease as the surrogate fills in the domain
   * - ``UncertaintyAnomalies``
     - LCB acquisition
     - Anomalous spikes or drops in acquisition-point std across the run
   * - ``VarianceErrorCorrelation``
     - Oracle query
     - Whether the surrogate assigns larger σ to points where it errs most


Methods
-------

Benchmark functions
~~~~~~~~~~~~~~~~~~~

The ``perfectly_calibrated`` scenario uses an oracle with homoscedastic
noise that the bootstrap ensemble can match, allowing calibration to be
achieved in practice:

.. math::

   f_\text{cal}(x) = \sin(2\pi x), \quad \sigma = 0.3 \text{ (constant)}

The function has a single smooth minimum at :math:`x = 0.75` with amplitude 1.
A degree-5 polynomial bootstrap fits it well after 15-20 observations, at which
point the epistemic spread shrinks and the aleatoric floor dominates.

The three remaining scenarios use the Forrester *et al.* (2008) [Forrester2008]_
function with heteroscedastic noise:

.. math::

   f_\text{Forrester}(x) & = (6x - 2)^2 \sin(12x - 4), \quad x \in [0, 1]

   \sigma(x) & = 0.1 + 0.4\,x^2

The non-constant noise level means that a homoscedastic surrogate will
inevitably appear miscalibrated at both ends of the domain — a deliberate
design choice that amplifies calibration differences between scenarios.

Surrogate model
~~~~~~~~~~~~~~~

A **bootstrap-ensemble polynomial surrogate** [Efron1979]_ is used for these examples:

* Each ensemble member fits a degree-5 polynomial with Ridge regularisation on ≤ 60 data points.
  The full 100-step loop with four scenarios completes in under ten seconds.

* Per-step predictive mean and std are estimated from the ensemble mean and
  standard deviation across members.
  
* Lower-confidence-bound (LCB) acquisition [Srinivas2010]_ with :math:`\kappa = 2.0` selects
  the next query point.

* The ensemble is seeded via ``numpy.random.default_rng``
  so every run with the same ``--seed`` flag is byte-for-byte identical.

Four scenario configurations are run sequentially:

.. list-table::
   :header-rows: 1
   :widths: 28 30 42

   * - Scenario tag
     - Configuration
     - Expected behaviour
   * - ``perfectly_calibrated``
     - 30 estimators, std scale 0.7, aleatoric floor 0.3, sin oracle
     - Calibration transition: 5/6 PASS at step 10, 5–6/6 PASS from step 20 onward
   * - ``well_calibrated``
     - 30 estimators, std scale 1.0, Forrester oracle
     - Heteroscedastic oracle limits achievable calibration; several checks fail
   * - ``overconfident``
     - 5 estimators, std scale 1.0, Forrester oracle
     - CalibrationError and IntervalCoverage fail (intervals too narrow)
   * - ``underconfident``
     - 30 estimators, std scale 4.0, Forrester oracle
     - VarianceAlignment and IntervalCoverage fail (intervals far too wide)

Bootstrap coverage underestimates epistemic uncertainty near the boundary but produces 
realistic relative trends that the audit can evaluate.

Results
-------

The results below are from a 100-step run with ``--seed 0``.  Green cells in
the check-grid heatmaps indicate PASS; red cells indicate FAIL.  Check values
are printed in white inside each cell.

Check-grid heatmap
~~~~~~~~~~~~~~~~~~

Rows are the four scenarios; columns are the six
checks.  Green cells indicate PASS; red cells indicate FAIL.  The colour
intensity is proportional to the magnitude of the check value relative to
the threshold, so mild failures appear light red and severe failures appear
deep red.

**Perfectly-calibrated scenario**

.. figure:: _static/demo_calibration/check_grid_perfect.png
   :width: 90%
   :align: center
   :alt: Check-grid heatmap for the perfectly-calibrated scenario

   Check-grid for the ``perfectly_calibrated`` scenario.
   Oracle: :math:`\sin(2\pi x) + \mathcal{N}(0,\, 0.09)`.
   Surrogate: 30-estimator bootstrap, std_scale=0.7, aleatoric floor
   :math:`\sigma_\text{al} = 0.3` added in quadrature.
   The total predicted uncertainty is
   :math:`\sigma_\text{total} = \sqrt{(0.7\,\sigma_\text{ep})^2 + 0.09}`.

This check-grid shows the calibration transition from uncalibrated to calibrated. 
At step 10, ``UncEvolution`` is the only failure (-0.142 per
step): LCB rapidly depletes the bootstrap's epistemic spread early in the
run as it focuses on the minimum of :math:`\sin(2\pi x)`.  All other checks
already PASS — ``CalibError = 0.104``, ``IntCoverage = 0.600`` (within the
53-83 % tolerance), ``VarAlignment = 0.901`` (predicted variance ≈ empirical
variance), ``VarErrCorr = 0.188`` (moderate Spearman correlation). From step 20 onward every check passes.  

* ``UncEvolution`` recovers to -0.045 (within the -0.05 threshold) 
    once the epistemic component stabilises at the 0.3 aleatoric floor.  

* ``VarAlignment`` moves through 0.988 → 1.094 →
    1.253 as the surrogate converges: the predicted variance is now equal to or
    slightly above the empirical squared error because LCB queries near the
    well-explored minimum carry near-zero epistemic std.

* ``VarErrCorr`` rises from 0.188 (step 10) to 0.465 (step 20), reflecting
    that the surrogate correctly assigns higher total uncertainty to the few
    remaining unexplored pool regions.

**Well-calibrated scenario**

.. figure:: _static/demo_calibration/check_grid_well.png
   :width: 90%
   :align: center
   :alt: Check-grid heatmap for the well-calibrated scenario

* ``CalibError`` passes only at step 10 (0.127 < 0.15) before rising to 0.306
    at step 100 — the bootstrap's spread grows proportionally faster than
    empirical error as the LCB concentrates queries.

* ``IntCoverage`` fails throughout at 0.350-0.500: only 35-50 % of observations
    fall within the 1-σ band (target 68.3 %), confirming that the ensemble is
    systematically overconfident in coverage.

* ``VarAlignment`` fails at 4.9-5.4 — the mean predicted variance is roughly
    5× the mean squared error, a consequence of the full-range bootstrap estimating
    high variance at the input boundaries while errors are concentrated near
    observed points.

* ``UncEvolution`` fails at slopes of -0.06 to -0.16 per step — the LCB policy
    rapidly depletes high-uncertainty pool regions, collapsing the epistemic signal
    faster than the threshold allows.

* ``UncAnomalies`` and ``VarErrCorr`` pass cleanly throughout.

**Overconfident scenario**

.. figure:: _static/demo_calibration/check_grid_over.png
   :width: 90%
   :align: center
   :alt: Check-grid heatmap for the overconfident scenario

* ``IntCoverage`` drops further to 0.260 by step 100 — the small committee
    produces narrower intervals, reducing coverage below even the well-calibrated
    case.  

* ``CalibError`` rises to 0.338 (final), more than twice the threshold,
    reflecting the compounded effect of reduced ensemble diversity.

* ``VarAlignment`` is slightly lower (3.8-4.1) because fewer estimators produce
    less aggregate spread, but remains a clear FAIL.  

* ``VarErrCorr`` actually improves relative to the well-calibrated case (0.539-0.740), 
    meaning the small ensemble concentrates residual uncertainty in regions where predictions
    are worst — a useful signal despite the overall miscalibration.

**Underconfident scenario**

.. figure:: _static/demo_calibration/check_grid_under.png
   :width: 90%
   :align: center
   :alt: Check-grid heatmap for the underconfident scenario

Every metric is inverted relative to the overconfident case.
   
* ``IntCoverage`` spikes to 1.000 at step 10 — every observation falls inside
    the inflated 1-σ band — and stays at 0.875 for the final report, far above
    the 83.3 % upper tolerance.  
   
* ``VarAlignment`` reaches 125 at step 10 (predicted
    variance 125× the squared error), settling to 29.0 at step 100, still one-two
    orders of magnitude above the acceptable range.

* ``CalibError`` recovers from
    0.450 to 0.070 by step 100 because the 4× inflation eventually brackets most
    residuals — but coverage and variance checks remain failed.

* ``VarErrCorr`` is near zero or negative throughout: the rank correlation
    between inflated std and absolute error is poor, since the uncertainty
    inflation is uniform rather than heteroscedastic.

Calibration curves (reliability diagrams)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/demo_calibration/calibration_curves.png
   :width: 100%
   :align: center
   :alt: Reliability diagrams for all four calibration scenarios

Each panel is a reliability diagram for one scenario.  The x-axis is the
expected nominal coverage level; the y-axis is the fraction of observations
actually covered at that level.  The dashed diagonal is perfect calibration;
the coloured solid curve is the observed reliability; the shaded region
represents the integral miscalibration (the ECE, [Kuleshov2018]_).

* **Perfectly calibrated (top-left):** The curve tracks the diagonal closely
  across all confidence levels — a small CE confirms the aleatoric floor keeps
  the surrogate from over-contracting its intervals.

* **Well calibrated (top-right):** The curve lies slightly below the diagonal,
  reflecting the irreducible mismatch between the homoscedastic ensemble and the
  heteroscedastic Forrester oracle.

* **Overconfident (bottom-left):** The curve falls sharply below the diagonal —
  50 % nominal intervals cover far fewer than 50 % of observations.  The large
  shaded area reflects the high CE value.

* **Underconfident (bottom-right):** The curve bows above the diagonal —
  intervals are so wide that even low nominal levels already contain most
  observations, and the shaded area extends upward rather than downward.

Cross-scenario Pareto frontier: CalibrationError (ECE) vs Mean Absolute Error (MAE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The Pareto frontier visualization confirms which
scenario-stage combinations are worth deploying: only the Perfectly-calibrated (green) and
Well-calibrated (blue) trajectories touch it. Across all four scenarios and intermediate audit stages
(step 10, 20, …, 100, and final), each scenario traces a trajectory from
earlier stages (lighter, top-right) to later stages (darker, lower-left);
Pareto-optimal (non-dominated) points are shown with heavier markers and
connected by the black staircase frontier. The lower-left corner is optimal: 
low ECE means well-calibrated uncertainty; low MAE means accurate predictions.  

.. figure:: _static/demo_calibration/pareto_scenarios.png
   :width: 80%
   :align: center
   :alt: Cross-scenario Pareto frontier of ECE vs MAE across all four scenarios

* **Perfectly-calibrated (green circles):** The only scenario whose late
   stages (step 30-final) reach the lower-left corner of the plot.  Both
   ECE and MAE decrease monotonically as the surrogate fits the smooth
   sin oracle.  All final-stage points are Pareto-optimal, confirming that
   a correctly-specified noise model allows simultaneous improvement in both
   calibration and accuracy.

* **Well-calibrated (blue squares):** Moderate ECE (0.05-0.15) with MAE
   tracking the perfectly-calibrated trajectory.  Some intermediate stages
   are Pareto-optimal; the heteroscedastic Forrester oracle prevents ECE
   from reaching zero, so the scenario converges to a region above and to
   the right of the perfectly-calibrated final point.

* **Overconfident (orange triangles):** ECE rises above 0.15 at most
   stages (reflecting the narrow ensemble intervals) while MAE is
   comparable to the well-calibrated case.  No stage is Pareto-optimal:
   another scenario always matches or beats this one on both axes
   simultaneously.  This is the defining signature of a pathological
   miscalibration — wasted query budget without commensurate accuracy gain.

* **Underconfident (red diamonds):** Symmetric pathology in the
   opposite direction.  ECE is large (inflated :math:`4\times\sigma`)
   and the inflated intervals make LCB behave more like random sampling,
   slightly increasing MAE relative to the well-calibrated case.  Like
   the overconfident scenario, no stage is Pareto-optimal.


Calibration convergence per scenario
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CalibrationError (ECE) at each intermediate pipeline evaluation (steps 10,
20, …, 100, and final) for all four scenarios.  

.. figure:: _static/demo_calibration/convergence_scenarios.png
   :width: 70%
   :align: center
   :alt: CalibrationError over AL steps for all four scenarios

The ``perfectly_calibrated``
scenario (green) decreases monotonically from ≈ 0.10 to ≈ 0.027, confirming
that the audit correctly tracks the convergence of a well-specified surrogate.
The ``well_calibrated`` (blue) scenario rises slightly due to the
heteroscedastic Forrester oracle: as LCB queries the high-noise boundary
region, the empirical ECE grows even though the surrogate is theoretically
well-calibrated. The overconfident (orange) and underconfident (red) scenarios
both stay above the 0.15 threshold throughout, with no sign of recovery.

This figure complements the Pareto frontier above: the frontier shows *which*
combinations of ECE and MAE are achievable; the convergence plot shows *how
quickly* each scenario reaches them.


Discussion
----------

The final audit report is printed to the console after the loop.

The ``perfectly_calibrated`` scenario passes five of six checks at 100 steps;
``VarianceErrorCorrelation`` fails because the well-explored surrogate has near-zero
epistemic variance everywhere, weakening the rank correlation with absolute error:

.. code-block:: text

   ── Audit report ───────────────────────────────────────────────
   CalibrationError         PASS  value=0.027  threshold=0.150
   IntervalCoverage         PASS  value=0.760  threshold=[0.533, 0.833]
   VarianceAlignment        PASS  value=1.114  threshold=[0.500, 1.500]
   UncertaintyEvolution     PASS  value=-0.002 threshold=-0.050
   UncertaintyAnomalies     PASS  value=0.010  threshold=0.050
   VarianceErrorCorrelation FAIL  value=0.066  threshold=0.100
   ── Overall: FAIL ──────────────────────────────────────────────

The well-calibrated Forrester scenario shows three FAIL checks driven by the
mismatch between the homoscedastic ensemble and the heteroscedastic oracle:

.. code-block:: text

   ── Audit report ───────────────────────────────────────────────
   CalibrationError         FAIL  value=0.306  threshold=0.150
   IntervalCoverage         FAIL  value=0.250  threshold=[0.533, 0.833]
   VarianceAlignment        FAIL  value=2.487  threshold=[0.500, 1.500]
   UncertaintyEvolution     PASS  value=-0.032 threshold=-0.050
   UncertaintyAnomalies     PASS  value=0.030  threshold=0.050
   VarianceErrorCorrelation PASS  value=0.478  threshold=0.100
   ── Overall: FAIL ──────────────────────────────────────────────

For the overconfident scenario, ``IntervalCoverage`` drops well below 53 %
and ``CalibrationError`` rises above 0.15 because the 5-estimator ensemble
underestimates spread.  For the underconfident scenario, ``VarianceAlignment``
exceeds 1.5 and coverage exceeds 83 % because of the artificial 4× noise
inflation.

.. list-table:: Check interpretation guide
   :header-rows: 1
   :widths: 30 20 50

   * - Check
     - Threshold
     - What a FAIL means in this context
   * - CalibrationError
     - ≤ 0.15
     - Ensemble spread is systematically mismatched to empirical residuals [Kuleshov2018]_
   * - IntervalCoverage
     - 53 - 83 %
     - 1-σ intervals contain too few (overconfident) or too many (underconfident) points
   * - VarianceAlignment
     - 0.5 - 1.5
     - Mean predicted variance is not commensurate with mean squared error
   * - UncertaintyEvolution
     - slope ≥ -0.05
     - Uncertainty is collapsing faster than data collection justifies
   * - UncertaintyAnomalies
     - ≤ 5 % steps with \|z\| > 3
     - Sporadic uncertainty spikes indicating a numerically unstable step
   * - VarianceErrorCorrelation
     - Spearman ρ ≥ 0.1
     - Predicted uncertainty is unrelated to where the model actually errs


References
----------

.. [Forrester2008] Forrester, A. I. J., Sóbester, A., & Keane, A. J. (2008).
   *Engineering Design via Surrogate Modelling: A Practical Guide.*
   Wiley. 

.. [Kuleshov2018] Kuleshov, V., Fenner, N., & Ermon, S. (2018).
   Accurate uncertainties for deep learning using calibrated regression.
   *Proceedings of the 35th International Conference on Machine Learning
   (ICML 2018)*, Proceedings of Machine Learning Research, 80, 2796-2804.

.. [Srinivas2010] Srinivas, N., Krause, A., Kakade, S. M., & Seeger, M. (2010).
   Gaussian process optimization in the bandit setting: No regret and
   experimental design.
   *Proceedings of the 27th International Conference on Machine Learning
   (ICML 2010)*, 1015-1022.

.. [Efron1979] Efron, B. (1979). Bootstrap methods: Another look at the
   jackknife. *The Annals of Statistics*, 7(1), 1-26.
