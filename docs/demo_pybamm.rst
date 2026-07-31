.. _demo-pybamm:

Li-ion C-rate optimisation demo (``ta-pybamm-demo``)
======================================================

This demo optimises the charge-rate and temperature operating point of a
lithium-ion cell to maximise discharge capacity.  The oracle is PyBAMM's
Single Particle Model (SPM), which runs in seconds on a CPU.  A Gaussian
process regressor (GPR) fitted to the queried observations acts as the
surrogate, and the uncertainty audit monitors whether the GPR confidence
intervals are trustworthy throughout the optimisation.

.. code-block:: bash

   pip install pybamm scikit-learn   # required dependencies

   ta-pybamm-demo                    # defaults: 8 seed evals, 20 UCB steps
   ta-pybamm-demo --n-iter 30 --kappa 3.0 --seed 7
   ta-pybamm-demo --out-dir _results/pybamm


Introduction
------------

Lithium-ion cell performance is a strong function of operating conditions.
Charging too fast accelerates lithium plating and capacity fade; charging at
too low a temperature increases internal resistance and reduces usable
capacity.  Finding the Pareto-optimal (C-rate, temperature) pair without
exhaustive physical testing is a canonical Bayesian optimisation problem
[Shahriari2016]_.

**Question:** Can a UCB-guided GPR reliably find the capacity-maximising
operating point in a 2-D search space while maintaining calibrated uncertainty
estimates throughout?  If the GPR intervals are overconfident early (before
sufficient coverage), does the audit flag this before the optimiser converges
to a sub-optimal region?

Uncertainty hook placement
~~~~~~~~~~~~~~~~~~~~~~~~~~

``hook.on_step()`` fires after the PyBAMM SPM simulation returns the discharge
capacity, before the GPR is re-fit.  The GPR prediction passed to the hook is
the **pre-update posterior** at the UCB-selected point:

.. code-block:: text

   GPR fit  →  UCB acquisition  →  PyBAMM oracle (SPM)  ← hook.on_step()
      ↑                                    |
      └──────────── add observation ───────┘

.. list-table:: Check-to-pipeline-step mapping
   :header-rows: 1
   :widths: 30 25 45

   * - Check
     - AL step monitored
     - What is observed
   * - ``CalibrationError``
     - PyBAMM oracle call
     - Whether the GPR posterior at the UCB-selected operating point correctly brackets the true discharge capacity
   * - ``ConformalCoverage``
     - PyBAMM oracle call
     - Distribution-free marginal coverage
   * - ``CRPS``
     - PyBAMM oracle call
     - CRPS as a proper scoring rule on each oracle evaluation
   * - ``NegativeLogLikelihood``
     - PyBAMM oracle call
     - Gaussian NLL on each oracle evaluation
   * - ``PITUniformity``
     - PyBAMM oracle call
     - PIT uniformity across all queried (C-rate, T) points
   * - ``IntervalScore``
     - PyBAMM oracle call
     - Winkler score penalising non-coverage and excessive width
   * - ``IntervalCoverage``
     - PyBAMM oracle call
     - Whether the GPR 1σ interval contains the simulated capacity ~68 % of the time
   * - ``VarianceAlignment``
     - PyBAMM oracle call
     - Whether GPR posterior variance explains prediction error across queried (C-rate, T) points
   * - ``UncertaintyEvolution``
     - UCB acquisition
     - Count of channels with a declining uncertainty trend (0 = all stable)
   * - ``UncertaintyAnomalies``
     - UCB acquisition
     - Fraction of current uncertainty values anomalously far from a historical baseline; skipped when no baseline is provided
   * - ``VarianceErrorCorrelation``
     - PyBAMM oracle call
     - Whether the GPR is most uncertain at operating points where its capacity prediction is least accurate
   * - ``LyapunovStability``
     - End of run, last 30 steps
     - Fraction of the *most recent* 30 queried operating points with :math:`|\lambda_{\max}| < 1` — a local/recent-window verdict (``window=30``), in contrast to the CAMD and SDL demos' global/cumulative default (``window=None``, the whole run since step 0)


Methods
-------

Physical domain
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - Variable
     - Symbol
     - Range
     - Steps
   * - C-rate
     - :math:`C`
     - 0.5 – 3.0 C
     - 10 uniformly-spaced values
   * - Temperature
     - :math:`T`
     - 10 – 40 °C
     - 8 uniformly-spaced values
   * - Candidate pool
     - —
     - —
     - 80 points (10 × 8 grid)

The oracle for each candidate is a PyBAMM SPM discharge simulation:

.. code-block:: python

   import pybamm
   model = pybamm.lithium_ion.SPM()
   sim   = pybamm.Simulation(model, parameter_values=p)
   sol   = sim.solve([0, t_end])
   cap   = sol["Discharge capacity [A.h]"].entries[-1]

The SPM represents each electrode as a sphere with diffusion-limited lithium
transport, neglecting electrolyte dynamics [Newman1975]_.  It takes < 1 s per evaluation
on a modern CPU and reproduces realistic capacity trends [Sulzer2021]_.
Additive Gaussian noise (``noise_std = 0.003`` Ah ≈ 0.4 % of nominal)
mimics measurement variability in laboratory conditions.


Surrogate model
~~~~~~~~~~~~~~~

A ``sklearn.gaussian_process.GaussianProcessRegressor`` [Rasmussen2006]_ is
fitted to the growing set of observations after each UCB query:

* **Kernel:** :math:`k(x, x') = \sigma_f^2\, k_\text{RBF}(x, x') + \sigma_n^2\, \delta_{xx'}`
  (constant amplitude × RBF + white noise).  The noise kernel is initialised
  to ``noise_std²`` and allowed to float within ``[1e-10, 1.0]``.
* **Normalisation:** ``normalize_y=True`` maps the capacity observations to
  zero mean and unit variance before kernel fitting, preventing numerical
  issues when the capacity range (≈ 0.025 Ah across the grid) is small
  relative to typical kernel amplitude scales.
* **Acquisition:** Upper-confidence bound (UCB) with :math:`\kappa = 2.0`:

  .. math::

     \alpha(x) = \mu(x) + \kappa\,\sigma(x)

  The next query is :math:`x^* = \arg\max_{x \in \text{pool}} \alpha(x)`.

At each step the GPR posterior at the queried point is recorded *before*
incorporating the new observation, so the audit receives a genuine
out-of-sample prediction rather than a retroactive fit.

The GPR is constructed with ``n_restarts_optimizer=3`` and
``random_state=seed``, so its hyperparameter-optimizer restarts are
reproducible given ``--seed``; without an explicit ``random_state``,
sklearn falls back to the unseeded global NumPy random state for those
restarts, making the fitted kernel hyperparameters — and everything
downstream of them, including the Lyapunov figures below — vary from run
to run even with the same ``--seed``.


Lyapunov stability framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After the AL loop the surrogate landscape is characterised as a discrete
dynamical system via the gradient-descent map [Strogatz2018]_:

.. math::

   F(x) = x - \alpha\,\nabla\hat{f}(x), \quad \alpha = 0.05

The Jacobian :math:`J = I - \alpha H_f` (where :math:`H_f` is the surrogate
Hessian) determines local stability: eigenvalues with :math:`|\lambda| < 1`
are contractive and those with :math:`|\lambda| > 1` are expansive.

The step size :math:`\alpha = 0.05` is chosen to keep eigenvalues near the
unit circle across all demos, balancing curvature visibility against numerical
stability.  The Lyapunov analysis operates in the normalised :math:`[0,1]^2`
(C-rate, temperature) space.

"Local" above is *spatial* — each :math:`\lambda_{\max}` is a per-operating-point
linearization. ``LyapunovStabilityCheck`` is wired into the audit pipeline with
``window=30``, a separate, *temporal* local/global choice: it aggregates the
stable fraction over only the most recent 30 queried points rather than the
whole run, demonstrating the local (recent-window) side of that axis in
contrast to the CAMD and SDL demos' global (cumulative) default — see
:doc:`checks` and ``LYAPUNOV_ANALYSIS.md`` for the full local/global
distinction.


.. Computational trade-offs
.. ~~~~~~~~~~~~~~~~~~~~~~~~

.. * **SPM vs DFN:** The Single Particle Model is 10–50× faster than the
..   Doyle–Fuller–Newman (DFN) model for single-discharge simulations but omits
..   electrolyte concentration gradients.  The capacity landscape it produces
..   is smooth and monotone, which is sufficient for demonstrating the audit
..   framework.

.. * **Discrete grid:** The 80-point grid avoids continuous optimisation over
..   the (C, T) space, eliminating the need for gradient-based or derivative-free
..   inner optimisation of the acquisition function.  This is appropriate for
..   the demo scale but would become a bottleneck for finer grids or
..   higher-dimensional spaces.

.. * **sklearn GPR vs GPyTorch:** ``sklearn.gaussian_process.GaussianProcessRegressor``
..   has cubic training complexity :math:`O(n^3)`.  At 8 seed points and 20 UCB
..   iterations the dataset never exceeds 28 points, making sklearn the right
..   choice.  For larger budgets, GPyTorch with approximate inference
..   (SVGP or KISS-GP) would be preferable [Gardner2018]_.

.. * **No batch querying:** One point is queried per iteration to maximise
..   information value.  Parallel batch strategies (q-EI, q-UCB) could
..   reduce wall-clock time in a real lab setting at the cost of some
..   statistical efficiency.


Results
-------

The figures below were produced by ``ta-pybamm-demo`` with default settings
(``--n-seed 8 --n-iter 20 --kappa 2.0 --noise-std 0.003 --seed 0``).

GPR posterior uncertainty evolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/demo_pybamm/fig4_uncertainty_evolution.png
   :width: 70%
   :align: center
   :alt: GPR posterior std at queried points over 20 UCB steps


GPR posterior standard deviation (in Ah) at the queried point per UCB
step.  The series opens at ≈ 0.0029 Ah and drifts down to ≈ 0.0017–0.0019 Ah
by steps 6–8 as the GPR is conditioned on nearby observations, then spikes
sharply to ≈ 0.0056 Ah at step 9 when UCB jumps to a less-explored corner
of the grid.  It settles back to ≈ 0.0017–0.0019 Ah for steps 10–14, then
produces the largest spike of the run — ≈ 0.0159 Ah, roughly 8× the
resting level — at step 15, before settling back to ≈ 0.0018–0.0024 Ah for
the remainder.  These are isolated single-step spikes rather than a sustained
upward trend: each corresponds to UCB briefly probing an unexplored (C-rate,
T) pair before the GPR is conditioned on it and the posterior std there
collapses again on the next step.

Audit checks over AL steps
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/demo_pybamm/fig6_audit_evolution.png
   :width: 100%
   :align: center
   :alt: Eleven audit check values at snapshot intervals for the PyBAMM demo


Reading across a row (snapshots at steps 5, 10, 15, 20, plus a final
column): ``CalibrationError`` starts PASS (0.135 at step 5) and
*degrades* over the run — 0.139, 0.147, and finally FAIL at 0.157 by
step 20 — the opposite trend from a GPR that calibrates better with more
data.  ``ConformalCoverage`` is unavailable at step 5 (too few points to
form a calibration set) then FAILs at every subsequent snapshot, with a
q-ratio ≈ 9.98 against a target ≤ 1.5 — a large, persistent miscalibration.
``IntervalCoverage`` starts PASS (0.80 at step 5, comfortably inside the
[0.533, 0.833] band) then FAILs for the rest of the run, falling to 0.50,
0.467, and 0.45 — the GPR's 1σ intervals cover fewer and fewer of the true
values as the campaign progresses.  ``VarianceAlignment`` FAILs at every
snapshot (0.017 → 0.029 → 0.036 → 0.128), always far *below* the
[0.5, 1.5] band — the GPR's predicted variance persistently understates
the true squared error, a more severe version of the same under-coverage
problem.  ``UncertaintyEvolution`` and ``UncertaintyAnomalies`` PASS at
steps 5 and 10, both FAIL at step 15, then ``UncertaintyEvolution``
recovers to PASS while ``UncertaintyAnomalies`` stays FAIL through step 20 —
consistent with the single large uncertainty spike at step 15 seen in the
uncertainty-evolution figure above.  ``VarianceErrorCorrelation`` starts
strongly PASS (ρ = 0.60) and steadily weakens — 0.26, 0.14 — before
flipping to FAIL (ρ = −0.048) by the final snapshot: committee disagreement
tracks prediction error well early on but that relationship erodes as the
run progresses.

Lyapunov pole diagram
~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/demo_pybamm/fig1_poles.png
   :width: 65%
   :align: center
   :alt: Real eigenvalue pole diagram (1-D strip) for the PyBAMM sklearn GPR

All 40 eigenvalues (2 per operating point × 20 queried points, since the
gradient-descent Jacobian is a 2×2 matrix in the normalised (C-rate, T)
space) are real (:math:`\mathrm{Im}(\lambda) \equiv 0` exactly — the
Jacobian :math:`J = I - \alpha H_f` is symmetric because the Hessian of any
twice-differentiable scalar function is symmetric, so this demo will never
show complex eigenvalues or a unit-circle diagram, unlike CAMD's
DMDc-fitted operator).  They cluster tightly around Re ≈ 1.0, ranging from
≈ 0.997 to ≈ 1.007 — right at the marginal-stability boundary (the dashed
lines at :math:`\pm 1`) rather than spread across a wide range.  31 of the
40 fall just inside the boundary (stable) and 9 just outside (unstable);
no eigenvalues fall outside the plotted range.  This tight clustering near
:math:`|\lambda| = 1` means the
GPR capacity surface is close to *flat* in the gradient-descent sense
around most queried points — consistent with a well-behaved, smoothly
varying capacity landscape rather than one with sharply diverging
gradients — and also means the stable/unstable classification is sensitive
to small numerical differences right at the boundary, which is reflected
in the roughly 50/50 stable-vs-unstable split.

Queried operating points in PCA space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/demo_pybamm/fig2_stability_contours.png
   :width: 70%
   :align: center
   :alt: Queried points in PCA space coloured by Lyapunov exponent, PyBAMM


:math:`|\lambda_{\max}|`, colour-scaled from ≈ 0.98 (blue) to ≈ 1.02 (red) —
a far narrower range than the CAMD demo's DMDc-fitted operator, since here
:math:`|\lambda_{\max}|` sits right at the marginal-stability boundary
throughout.  The colour structure is nonetheless visible: the handful of
points in the upper-left of the PC plane (PC1 ≈ −0.5 to −0.75, PC2 ≈ 0.2–0.7)
are warm/orange, closer to or above the :math:`|\lambda_{\max}| = 1`
boundary, while the dense cluster near the centre (PC1 ≈ 0–0.5, PC2 ≈ −0.3
to 0.3) is cool/blue, comfortably below it.  The upper-left points correspond
to queries at the extremes of the (C-rate, T) grid, where the capacity
surface curves more sharply; the central cluster corresponds to later UCB
iterations converging near the capacity optimum, where the surface is
flatter.

Lyapunov exponent vs GPR uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/demo_pybamm/fig3_stability_vs_unc.png
   :width: 70%
   :align: center
   :alt: Lyapunov exponent vs GPR posterior std for the PyBAMM demo

The x-axis range is compressed (≈ 0.0018–0.0026 Ah) and :math:`|\lambda_{\max}|`
spans only ≈ 0.997–1.007, but a clear positive relationship is visible
here: points at higher GPR std tend to have higher :math:`|\lambda_{\max}|`,
rising from ≈ 0.997 at std ≈ 0.0018 Ah up to the single highest point
(≈ 1.007) at the highest std (≈ 0.0026 Ah).  Unlike the CAMD demo, where
DMDc stability and committee uncertainty were largely decoupled, here the
two move together — plausibly because both are driven by the same
underlying cause: queries at the edges of the (C-rate, T) grid, where the
GPR has the least training signal *and* the capacity surface curves most
sharply.

Lyapunov evolution
~~~~~~~~~~~~~~~~~~

.. figure:: _static/demo_pybamm/fig5_lyapunov_evolution.png
   :width: 80%
   :align: center
   :alt: Lyapunov exponent and GPR std over UCB steps for the PyBAMM demo

   (orange = :math:`|\lambda_{\max}|`, left axis;
   blue = GPR std, right axis).

:math:`|\lambda_{\max}|` oscillates in a narrow band throughout (≈ 0.997–1.007),
with its single highest peak at step 5 and smaller peaks around steps 2,
7–8, 9, and 14; it never departs far from the marginal-stability boundary.
The GPR-std curve is much flatter for most of the run, punctuated by two
sharp, isolated spikes at steps 9 and 15 (the same spikes visible in the
uncertainty-evolution figure above).  The two series loosely align at step
9, where both are locally elevated, but not elsewhere — the largest std
spike (step 15) does not coincide with the largest :math:`|\lambda_{\max}|`
peak (step 5).  So while the scatter plot above shows a positive overall
association between std and :math:`|\lambda_{\max}|`, the two signals are
not tightly locked together step-by-step — consistent with
:math:`|\lambda_{\max}|` responding to local surface curvature and GPR std
responding to training-data density, two related but distinct properties
of the same landscape.

Pareto frontier: GPR posterior std vs discharge capacity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/demo_pybamm/fig7_pareto_frontier.png
   :width: 70%
   :align: center
   :alt: Pareto frontier of GPR uncertainty vs discharge capacity for the PyBAMM demo

   (coloured
   by UCB step, viridis scale from early = yellow to late = dark purple).
   Each point is one UCB-queried (C-rate, T) pair; the Pareto-optimal
   subset (circled) is the non-dominated set that achieves simultaneously
   low uncertainty *and* high capacity — the most trustworthy operating
   candidates identified by the loop.

For this run, most points already cluster tightly at low std (≈ 0.0015–
0.003 Ah) and high capacity (≈ 0.676–0.685 Ah), so the frontier itself is
compact: three Pareto-optimal points sit right on top of each other at the
lowest std / highest capacity corner, spanning early-to-mid steps (yellow
to dark purple).  Three points are clearly dominated: one mid-run point
(teal, step ≈ 10) at moderate std (≈ 0.0055 Ah) but noticeably lower
capacity (≈ 0.656 Ah); one early point (yellow-green, step ≈ 6) with the
lowest capacity in the whole run (≈ 0.628 Ah) despite low std; and the
step-15 spike point (blue-purple) at by far the highest std (≈ 0.0155 Ah)
without a compensating capacity gain (≈ 0.675 Ah, no better than the
Pareto-optimal cluster).  There is no early/late split in this run's
frontier — the dominated points come from early, middle, and late steps
alike.

Points that lie off the frontier (no circle) are dominated: another
queried point achieved better capacity *and* lower uncertainty
simultaneously.  The frontier therefore serves as a compact summary of
the AL budget: it highlights which fraction of the 20 UCB steps
produced genuinely informative, high-quality observations.

Convergence: running best discharge capacity vs cumulative AL queries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/demo_pybamm/fig8_convergence.png
   :width: 70%
   :align: center
   :alt: Running best discharge capacity vs cumulative AL queries for the PyBAMM demo

The dashed horizontal line marks the best capacity found among the 8 seed
observations (≈ 0.6702 Ah).  The solid curve shows the running maximum as
UCB queries accumulate.  For this run there are two step improvements: the
first at query 2 (to ≈ 0.677 Ah) and a second at query 4 (to ≈ 0.6807 Ah),
after which the curve is completely flat for the next 9 queries; a final
jump at query 14 reaches the run's best value (≈ 0.6847 Ah), which then
holds flat through query 28 — indicating the UCB policy has effectively
converged: the remaining queries refine the GPR posterior rather than
finding a better optimum.  A curve that never exceeds
the seed baseline would indicate that UCB is stuck exploring low-capacity
corners — a failure mode the ``UncertaintyEvolution`` check would flag as a
pathologically rising slope.


Discussion
----------

A run with the documented reproduction command (28 points: 8 seed + 20 UCB)
produces an audit report similar to (showing the six checks present since
the first release; the full pipeline also runs ``ConformalCoverage``,
``CRPS``, ``NegativeLogLikelihood``, ``PITUniformity``, ``IntervalScore``,
and ``LyapunovStability`` — see the check grid above for those):

.. code-block:: text

   ── Audit report ────────────────────────────────────────────────────
   CalibrationError         FAIL  value=0.157  threshold=0.150
   IntervalCoverage         FAIL  value=0.450  threshold=[0.533, 0.833]
   VarianceAlignment        FAIL  value=0.128  threshold=1.0
   UncertaintyEvolution     PASS  value=0     threshold=0.0
   UncertaintyAnomalies     FAIL  value=0.050  threshold=0.050
   VarianceErrorCorrelation FAIL  value=-0.048 threshold=0.100
   ── Overall: FAIL (5 checks failed among these six) ──────────────────

Of the full 12-check pipeline, 6 pass: ``CRPS``, ``NegativeLogLikelihood``,
``PITUniformity``, ``IntervalScore``, ``UncertaintyEvolution``, and
``LyapunovStability`` (0.600 stable fraction).  Scenario-specific guidance:

* **CalibrationError FAIL late in the run**, as seen here (0.135 at step 5
  degrading to 0.157 by step 20): unlike the "too little data yet" failure
  mode below, a calibration error that *worsens* as more observations
  accumulate suggests the GPR's noise model itself is a poor fit — check
  whether ``--noise-std`` matches the oracle's actual noise level, since a
  mismatched fixed noise floor does not improve with more data the way a
  data-scarcity problem would.

* **CalibrationError FAIL early in the run (< 15 steps):** Also expected in
  many runs, for a different reason — with fewer than ~12 observations the
  GPR likelihood surface is poorly constrained.  If calibration does not
  recover by step 20, consider increasing ``--n-seed`` or adding a
  length-scale prior.

* **VarianceAlignment > 1.5:** The GPR is assigning more variance than the
  actual squared errors warrant — often caused by the ``normalize_y`` option
  overestimating the capacity range.  Reduce ``--noise-std`` or fix the
  noise kernel bounds.

* **VarianceAlignment persistently well below 0.5**, as seen here (0.017 →
  0.128 across the run): the opposite failure mode — the GPR's posterior
  variance is *too small* relative to true squared error, i.e. the model is
  overconfident.  This is consistent with the ``IntervalCoverage`` FAILs at
  0.45–0.50 (under-covering the nominal 68.3% band) seen in the same run:
  both point to intervals that are too narrow, not too wide.

* **UncertaintyEvolution slope < −0.05:** UCB with a large :math:`\kappa`
  can lock onto a high-capacity region early, rapidly collapsing the
  uncertainty budget.  Reduce ``--kappa`` or increase ``--n-iter``.

* **VarianceErrorCorrelation FAIL:** The GPR is not more uncertain in regions
  where it is wrong.  In this run it starts strongly positive (ρ = 0.60 at
  step 5) and steadily weakens to FAIL (ρ = −0.048) by the end — worth
  watching for in other runs too, since a correlation that erodes over the
  campaign is a different (and arguably more concerning) pattern than one
  that fails from the start.  This often also occurs when the kernel length
  scale is too long (the model smooths over all variation) or too short (it
  memorises every point with near-zero residual).


References
----------

.. [Sulzer2021] Sulzer, V., Marquis, S. G., Timms, R., Robinson, M., &
   Chapman, S. J. (2021).
   Python Battery Mathematical Modelling (PyBAMM).
   *Journal of Open Research Software*, 9(1), 14.
   https://doi.org/10.5334/jors.309

.. [Shahriari2016] Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., &
   de Freitas, N. (2016).
   Taking the human out of the loop: A review of Bayesian optimization.
   *Proceedings of the IEEE*, 104(1), 148–175.
   https://doi.org/10.1109/JPROC.2015.2494218

.. [Rasmussen2006] Rasmussen, C. E., & Williams, C. K. I. (2006).
   *Gaussian Processes for Machine Learning.*
   MIT Press.

.. [Newman1975] Newman, J., & Tiedemann, W. (1975).
   Porous-electrode theory with battery applications.
   *AIChE Journal*, 21(1), 25–41.
   https://doi.org/10.1002/aic.690210103

