.. _demo-sdl:

Self-driving lab color-matching demo (``ta-sdl-demo``)
=======================================================

This demo applies the uncertainty audit to a self-driving laboratory (SDL)
workflow for LED color matching; `the full implementation is here
<https://github.com/sparks-baird/self-driving-lab-demo>`_.  A Bayesian
optimisation loop built on `Ax <https://ax.dev/>`_ and
`BoTorch <https://botorch.org/>`_ searches the (R, G, B) intensity space for
the settings that minimise the Fréchet distance between the emitted
spectrum and a target color.
The ``SelfDrivingLabDemoLight`` class runs in simulation mode — no physical
hardware is required.

.. code-block:: bash

   pip install "traits-audit[sdl]"   # installs self-driving-lab-demo + ax-platform

   ta-sdl-demo                            # defaults: 6 Sobol + 25 BO iterations
   ta-sdl-demo --n-iter 40 --seed 7
   ta-sdl-demo --out-dir _results/sdl


Introduction
------------

Self-driving laboratories automate the design-build-test-learn cycle for
scientific experiments, using Bayesian optimisation to choose the next
experimental condition from a continuous or combinatorial design space
[Seifrid2022]_.  Monitoring the GP surrogate that drives the BO loop is
critical: a miscalibrated GP selects sub-optimal conditions and may
prematurely declare convergence.

**Question:** Does the uncertainty audit reliably track the quality of the
BoTorch GP throughout a Bayesian optimisation loop on a 3-D color space?
Specifically:

* Do the calibration and coverage checks improve as the GP accumulates
  observations from the Sobol warm-start?
* Does the Lyapunov spectrum of the surrogate objective surface indicate
  convergent gradient dynamics once the BO loop has focused on a promising
  region?

Uncertainty hook placement
~~~~~~~~~~~~~~~~~~~~~~~~~~

The loop has two phases.  During the **Sobol initialisation** phase the GP
model is not yet fitted, so ``on_step`` is not called and the hook accumulates
no history.  ``hook.on_step()`` is called only during the **BO loop**, after
the simulator returns the Fréchet distance and the Ax GP posterior is
extracted at the proposed point:

.. code-block:: text

   [Sobol init — hook silent]
            ↓
   Ax GP model ready
            ↓
   Ax proposes (R, G, B)  →  Simulator observation  ← hook.on_step()
           ↑                          |
           └──── complete_trial() ────┘

.. list-table:: Check-to-pipeline-step mapping (BO phase only)
   :header-rows: 1
   :widths: 30 25 45

   * - Check
     - AL step monitored
     - What is observed
   * - ``CalibrationError``
     - Simulator observation
     - Whether the Ax GP posterior at the proposed LED setting correctly brackets the true Fréchet distance
   * - ``ConformalCoverage``
     - Simulator observation
     - Distribution-free marginal coverage
   * - ``CRPS``
     - Simulator observation
     - CRPS as a proper scoring rule on each simulator evaluation
   * - ``NegativeLogLikelihood``
     - Simulator observation
     - Gaussian NLL on each simulator evaluation
   * - ``PITUniformity``
     - Simulator observation
     - PIT uniformity across all BO-phase observations
   * - ``IntervalScore``
     - Simulator observation
     - Winkler score penalising non-coverage and excessive width
   * - ``IntervalCoverage``
     - Simulator observation
     - Whether the GP 1σ interval contains the simulated Fréchet value ~68 % of the time
   * - ``VarianceAlignment``
     - Simulator observation
     - Whether GP posterior variance scales with prediction error across proposed colour settings
   * - ``UncertaintyEvolution``
     - Ax acquisition
     - Count of channels with a declining uncertainty trend (0 = all stable)
   * - ``UncertaintyAnomalies``
     - Ax acquisition
     - Fraction of current uncertainty values anomalously far from a historical baseline; skipped when no baseline is provided
   * - ``VarianceErrorCorrelation``
     - Simulator observation
     - Whether the GP is most uncertain at LED settings where it predicts colour distance poorly
   * - ``LyapunovStability``
     - End of run
     - Whether gradient-descent dynamics of the GP surrogate are stable in PCA-reduced (R, G, B) space, aggregated **cumulatively since step 0** (``window=None``, the default) — a global verdict, not the model's current/recent state. Contrast with the PyBAMM demo's ``window=30`` local/recent-window verdict.


Methods
-------

Physical domain and objective
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The optimisation space is the 3-channel LED intensity:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Channel
     - Range
     - Physical meaning
   * - R (red)
     - [0, 255]
     - Red LED duty cycle (8-bit)
   * - G (green)
     - [0, 255]
     - Green LED duty cycle
   * - B (blue)
     - [0, 255]
     - Blue LED duty cycle

The objective is the **Fréchet distance** [Frechet1957]_ between the emitted
spectrum and a fixed target, computed by the ``SelfDrivingLabDemoLight``
simulator.  The Fréchet distance generalises the Euclidean distance to
distribution-valued observations and is a standard metric for spectral
dissimilarity.  A lower Fréchet distance means a closer
match to the target color; the BO loop minimises it.

In simulation mode the ``evaluate()`` call returns a pre-computed or
analytically evaluated Fréchet distance, bypassing any physical hardware.
This makes the demo fully reproducible and runnable on any machine.


Surrogate model
~~~~~~~~~~~~~~~

The BO loop is driven by the **Ax platform** (Meta Research) in ask-tell
mode [Bakshy2018]_:

* **GP kernel:** Ax delegates to BoTorch's default kernel stack (Matérn-5/2
  with ARD length scales), fitted via marginal likelihood maximisation.
* **Warm-start:** ``n_init = 6`` Sobol quasi-random trials populate the
  design space before any GP is fitted.  No posterior is available during
  this phase, so ``on_step`` is not called and the hook accumulates no
  history entries.
* **Acquisition:** Expected Improvement (EI, Ax default for minimisation)
  after the Sobol phase.
* **Uncertainty extraction:** After each BO step the GP posterior mean and
  standard deviation at the queried point are read back via
  ``AxClient.get_model_predictions_for_parameterizations()``.  If the GP model is
  not yet fitted (first 1-2 BO steps), the values are ``nan`` and the step
  is skipped.

The hook accumulates only the BO-phase steps (not Sobol), so calibration
checks are evaluated on a history of up to ``n_iter`` entries.


Lyapunov stability framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After the BO loop the surrogate landscape is characterised as a discrete
dynamical system via the gradient-descent map [Strogatz2018]_:

.. math::

   F(x) = x - \alpha\,\nabla\hat{f}(x), \quad \alpha = 0.01

The Jacobian :math:`J = I - \alpha H_f` determines local stability: eigenvalues
with :math:`|\lambda| < 1` are contractive and those with :math:`|\lambda| > 1`
are expansive.

The step size :math:`\alpha = 0.01` is chosen to keep eigenvalues near the
unit circle, balancing curvature visibility against numerical stability.
The Lyapunov analysis operates in the normalised :math:`[0, 1]^3`
(R, G, B) space.

"Local" above is *spatial* — each :math:`\lambda_{\max}` is a per-operating-point
linearization. ``LyapunovStabilityCheck`` is wired into the audit pipeline at
its default ``window=None``, a separate, *temporal* local/global choice: it
aggregates the stable fraction cumulatively over the whole run rather than a
recent window, demonstrating the global side of that axis in contrast to the
PyBAMM demo's ``window=30`` (local/recent) — see :doc:`checks` and
``LYAPUNOV_ANALYSIS.md`` for the full local/global distinction.


.. Computational trade-offs
.. ~~~~~~~~~~~~~~~~~~~~~~~~

.. * **Simulation vs hardware:** The physical SDL hardware (Arduino + NeoPixel)
..   introduces several seconds of latency per evaluation.  In simulation mode
..   each ``evaluate()`` call is nearly instantaneous, so the full demo
..   (6 + 25 steps) completes in under a minute including BoTorch refitting.

.. * **Ax / BoTorch overhead:** Each BoTorch GP refit after a new observation
..   takes 0.5-2 s depending on the number of observations.  For 25 iterations
..   this is negligible, but for longer runs consider reducing
..   ``n_restarts_optimizer`` or switching to a stochastic variational GP.

.. * **3-D vs high-D:** The RGB space is low-dimensional enough that exact
..   BoTorch inference is tractable.  Self-driving labs with more controllable
..   parameters (temperature, pH, solvent ratio, …) require approximate methods
..   (SAASBO [Eriksson2021]_, TuRBO [Eriksson2019]_) or chemistry-aware BO
..   [Shields2021]_ that provide similar posterior outputs for audit purposes.

.. * **Lyapunov in 3-D:** The Lyapunov Jacobian is computed in the normalised
..   (R/255, G/255, B/255) space.  PCA is not applied here because the
..   intrinsic dimension is already 3 — the full Jacobian is
..   :math:`3 \times 3`, fast to compute via finite differences.


Results
-------

The figures below were produced by ``ta-sdl-demo`` with default settings
(``--n-init 6 --n-iter 25 --seed 0``), using the simulated
``SelfDrivingLabDemoLight`` oracle.

GP posterior uncertainty evolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/demo_sdl/fig4_uncertainty_evolution.png
   :width: 70%
   :align: center
   :alt: BoTorch GP posterior std over 25 BO steps for the SDL demo

   (Fréchet-distance scale).
   The y-axis values (0-15 000) reflect the raw Fréchet distance scale,
   not a normalised quantity.

The path is highly oscillatory throughout, without a clean declining
envelope: it opens at ≈ 12 500, dips to ≈ 3 700 at step 1, spikes back up
to the run's maximum (≈ 14 600) at step 2, then falls to a low ≈ 1 300–1 700
band through steps 4–6.  Two more exploration episodes follow — a spike to
≈ 8 500 at step 7 and a sustained rise peaking at ≈ 12 200 around step
13 — each time falling back to the 1 000–2 000 range within a few steps.
Two smaller late spikes (≈ 7 800 at step 17, ≈ 8 500 at step 20) show the
loop is still capable of large exploration excursions well past its
midpoint, before settling to ≈ 2 000 by the final step.  The zig-zag
pattern is the EI acquisition function alternating between exploitation
(queries near the current minimum, where the GP is already certain,
producing low-uncertainty steps) and exploration (queries at the
boundaries of the 3-D RGB space, producing high-uncertainty spikes) —
each exploration episode is resolved within a few steps after the new
observation is added, but the run as a whole does not settle into a
steadily shrinking envelope the way the Lyapunov analysis below eventually
does.

Audit checks over AL steps
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/demo_sdl/fig6_audit_evolution.png
   :width: 100%
   :align: center
   :alt: Twelve audit check values over BO steps for the SDL demo

   (snapshot every 5 steps, seed 0).

``CalibrationError`` FAILs at the first three snapshots (0.175, 0.180,
0.153) then PASSes for the last two (0.115, 0.140) — with only 25 BO-phase
observations in a 3-D continuous space, the BoTorch GP takes a while to
reach calibrated intervals, and even then hovers close to the 0.15
threshold rather than settling comfortably below it.  ``IntervalCoverage``
starts PASS (0.60 at step 5) then FAILs for the rest of the run, oscillating
0.40–0.48 — well short of the acceptable [0.533, 0.833] band — the GP's
1σ bands capture well under half the observations from step 10 onward.
``VarianceAlignment`` PASSes at steps 5 and 10 (0.76, 0.53) then collapses
to FAIL from step 15 onward (0.061, 0.066, 0.071): predicted variance falls
to roughly a fifteenth of actual squared error late in the run, a sharp
swing into severe overconfidence.  ``UncertaintyEvolution`` FAILs at every
snapshot except step 15 (which happens to land right after a sharp drop in
the uncertainty-evolution figure above) — a stricter reading than the
overall oscillatory pattern might suggest, since this check flags a
declining channel over the history seen *so far*, not the run's long-run
average behaviour.  ``UncertaintyAnomalies`` is zero throughout — EI
spikes are large but not outliers relative to the mean, so z-scores stay
below 3.  ``VarianceErrorCorrelation`` PASSes at every snapshot, rising
from +0.30 at step 5 to +0.72 by step 10 and staying in the 0.65–0.69 range
thereafter — the GP consistently assigns higher uncertainty to its worst
predictions from step 10 onward.  Two checks that need a minimum sample
count are undefined until enough steps have accumulated: ``ConformalCoverage``
first reports at step 10 (FAIL throughout, q-ratio 3.1 → 9.2, well above
the 1.5 threshold) and ``PITUniformity`` only at steps 20 and 25 (FAIL,
KS p-value ≈ 0.001 and 0.003 respectively). ``LyapunovStability`` — fed a
*rolling* :math:`\lambda_{\max}` computed live inside the loop at every
step and logged via ``hook.on_step()``, so it has a value at every
snapshot here, unlike the CAMD/PyBAMM demos' end-of-run-only precomputed
route — FAILs at every snapshot (stable fraction 0.0 at step 5, then 0.2
for the rest of the run, threshold 0.5), consistent with the
still-substantial pole magnitudes discussed below.

Lyapunov pole diagram
~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/demo_sdl/fig1_poles.png
   :width: 65%
   :align: center
   :alt: Real eigenvalue pole diagram (1-D strip) for the SDL BoTorch GP

   36 stable (:math:`|\lambda|<1`), 39 unstable (:math:`|\lambda|\geq 1`).

All 75 eigenvalues are real (:math:`\mathrm{Im}(\lambda)\equiv 0`, since the
gradient-descent map differentiates a real scalar surrogate) and range from
≈ −79 to ≈ 143 — no poles fall outside the plotted range for this run.  A
dense cluster of ~40 eigenvalues sits between 0.4 and 1.1, straddling the
stability boundary; the remaining unstable population splits into a
negative group (≈ −79 to −2, about a dozen points) and a positive group
(≈ 7 to 143, about fifteen points, including the single largest-magnitude
pole in the run at ≈ 143).  Just over half the poles are unstable —
consistent with the low ``LyapunovStability`` fraction reported above.

Queried operating points in normalised RGB space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/demo_sdl/fig2_stability_contours.png
   :width: 70%
   :align: center
   :alt: PCA scatter of queried RGB points coloured by Lyapunov exponent, overlaid on the Lyapunov function contour

Background contours show :math:`V(x) = x^T P x`, the discrete Lyapunov
function.  For this run the mean operating point is itself unstable
(:math:`|\lambda_{\max}| \approx 1.02`), so :math:`P` is solved from a
rescaled Jacobian (spectral radius reduced to 0.99) rather than the raw
mean-point Jacobian, purely to give a comparably-scaled contour background
— the per-point :math:`|\lambda_{\max}|` colouring of the scatter itself
uses the unrescaled values.  Each scatter point is one queried point
projected onto the first two principal components, coloured by its own
:math:`|\lambda_{\max}|` (blue ≈ 1 up to dark red ≈ 140).  The lowest
values (grey/light, near the stability boundary) sit in a loose cluster
around PC1 ≈ 0.15–0.3, PC2 ≈ 0; most other points are orange-to-red,
scattered around the ellipse from PC1 ≈ −0.75 to 0.4, with the single
darkest (highest-:math:`|\lambda_{\max}|` ≈ 143) point at the extreme
PC1 ≈ −0.62, PC2 ≈ −0.55 — consistent with the Fréchet landscape having
steeper local curvature away from the small region EI has exploited most.

Lyapunov exponent vs GP uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/demo_sdl/fig3_stability_vs_unc.png
   :width: 70%
   :align: center
   :alt: Lyapunov exponent vs GP posterior std for the SDL demo

A handful of low-std points (≈ 520, 655, 695) sit right at the stability
boundary (:math:`|\lambda_{\max}| \approx 1`).  The rest of the run
clusters at much higher std (≈ 715–830), where :math:`|\lambda_{\max}|`
spans nearly the whole observed range — from single digits up to the
run's maximum (≈ 143, at the single highest std, ≈ 828).  So while the
very lowest-uncertainty points are reliably stable, high std does not
reliably predict high :math:`|\lambda_{\max}|`: several points in that
same 715–830 std band have :math:`|\lambda_{\max}|` under 10, sitting
right alongside points an order of magnitude more unstable.  This
reinforces that Lyapunov analysis captures landscape curvature that GP
uncertainty alone does not.

Lyapunov evolution
~~~~~~~~~~~~~~~~~~

.. figure:: _static/demo_sdl/fig5_lyapunov_evolution.png
   :width: 80%
   :align: center
   :alt: Lyapunov exponent and GP std over BO steps for the SDL demo

   (orange = :math:`|\lambda_{\max}|`, left
   axis; blue = GP std, right axis).

Both signals open elevated — :math:`|\lambda_{\max}|` ≈ 58 and std ≈ 12 200
— and remain volatile for the whole run, without either one settling into
a flat late-run plateau.  The two loosely co-move at some points (both dip
toward their respective floors around steps 8–10) but not others: the
std series' single biggest spike is at step 2 (≈ 14 600), while
:math:`|\lambda_{\max}|`'s biggest spike is at step 20 (≈ 144, the run
maximum) — a point where std is comparatively unremarkable.  Both series
still show large excursions in the run's final third (std spikes at steps
17 and 24; :math:`|\lambda_{\max}|` spikes at step 20 and rises again
toward step 24), consistent with ``LyapunovStability``'s stable fraction
staying flat at 0.2 rather than improving toward the end of the run.  The
partial, inconsistent co-movement suggests surrogate uncertainty and
dynamical sensitivity are tracking
related — though not identical — aspects of how well-resolved the Fréchet
landscape is in each queried region.

Pareto frontier: GP posterior std vs Fréchet distance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/demo_sdl/fig7_pareto_frontier.png
   :width: 70%
   :align: center
   :alt: Pareto frontier of GP uncertainty vs Fréchet distance for the SDL demo

   (coloured by
   BO step, viridis scale from early = dark yellow to late = purple).

Each point is one BO-phase evaluation; the Pareto-optimal set (circled)
achieves simultaneously the lowest Fréchet distance *and* the lowest GP
uncertainty — the most reliable candidate LED settings identified by the
BO loop.  Three points are Pareto-optimal: two sit at low std (≈ 1 200–1 500,
both late steps, one at Fréchet ≈ 15 000 and one at Fréchet ≈ 2 500), and a
third stands well apart at moderate std (≈ 7 000) and by far the highest
Fréchet distance in the run (≈ 130 000, around step 11–13) — worth keeping
on the frontier only because nothing else matches its comparatively low
std at that Fréchet distance.  The remaining points scatter widely: several
early-to-mid steps at std ≈ 1 200–2 000 with Fréchet 15 000–30 000, a
handful at std ≈ 8 000–14 500 (late and early steps alike) with Fréchet
17 000–27 000, and one clear outlier (teal, std ≈ 10 500) at Fréchet
≈ 69 000.

A key distinction from the PyBAMM and CAMD cases: in the SDL demo the GP
does not achieve low Fréchet distance and low std in lockstep — the
Pareto-optimal set spans an enormous range of Fréchet distances
(2 500–130 000) and a meaningful range of std too (1 200–7 000), rather
than std and Fréchet distance falling together as the run progresses.
This reflects EI deliberately querying high-uncertainty regions throughout
the run to build its model, not just in an early exploration phase.

Convergence: running best Fréchet distance vs cumulative BO queries.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/demo_sdl/fig8_convergence.png
   :width: 70%
   :align: center
   :alt: Running best Fréchet distance vs BO queries for the SDL demo

   The dashed horizontal line marks the best Fréchet distance found during
   the 6 Sobol warm-start trials.  The solid curve shows the running minimum
   as BO iterations accumulate.  
   
The dashed baseline (≈ 23 600, best of the 6 Sobol trials) drops sharply to
≈ 16 400 at query 2 — the first BO query already beats the warm-start.
From there the curve declines gradually and in small steps: to ≈ 15 400 by
query 7, ≈ 14 400 by query 11, holding an extended near-flat plateau
(≈ 14 200–14 400) through query 19, then a small step down to ≈ 13 500 by
query 24.  The final query produces by far the largest single improvement
in the whole run — a sharp drop to ≈ 2 700, roughly a fifth of the
previous best — showing that even after a long plateau, EI can still
locate a substantially better region right at the very end of the budget.
A curve that matches the Sobol baseline for the entire BO phase would
indicate EI is still in the exploration regime — more ``--n-iter`` steps
or a higher ``--n-init`` warm-start budget would be needed.
The convergence figure and the Pareto frontier complement each other: the
frontier identifies *which* queries achieved both good performance *and*
low uncertainty; the convergence curve shows *when* the best objective
value was first achieved.


Discussion
----------

A typical output for a 25-iteration BO run (``--n-init 6 --n-iter 25 --seed 0``)
(showing the six checks present since the first release; the full pipeline
also runs ``ConformalCoverage``, ``CRPS``, ``NegativeLogLikelihood``,
``PITUniformity``, ``IntervalScore``, and ``LyapunovStability`` — see the
check evolution figure above for those):

.. code-block:: text

   ── Audit report ────────────────────────────────────────────────────
   CalibrationError         PASS  value=0.140  threshold=0.150
   IntervalCoverage         FAIL  value=0.480  threshold=[0.533, 0.833]
   VarianceAlignment        FAIL  value=0.071  threshold=1.0
   UncertaintyEvolution     FAIL  value=1     threshold=0.0
   UncertaintyAnomalies     PASS  value=0.000  threshold=0.050
   VarianceErrorCorrelation PASS  value=0.654  threshold=0.100
   ── Overall: FAIL (3 checks failed among these six) ─────────────────

Of the full 12-check pipeline, 6 pass: ``CRPS``, ``NegativeLogLikelihood``,
``IntervalScore``, ``CalibrationError``, ``UncertaintyAnomalies``, and
``VarianceErrorCorrelation``; the other 6 (``ConformalCoverage``,
``PITUniformity``, ``IntervalCoverage``, ``VarianceAlignment``,
``UncertaintyEvolution``, and ``LyapunovStability`` at a 0.16 stable
fraction) fail.  Scenario-specific interpretation notes:

* **Small history size:** With ``n_iter = 25`` the hook sees at most 25
  steps (fewer if early BO steps have NaN posteriors).  Calibration checks
  computed on small datasets have high variance; a FAIL by a small margin
  is not necessarily actionable.  Increase ``--n-iter`` or decrease
  ``--n-init`` if you need tighter estimates.

* **CalibrationError PASSing overall despite failing at several of the
  per-snapshot checkpoints above:** EI exploration tends to query points
  where the GP is most uncertain, which means the GP is also least likely
  to be well-calibrated there early on — this is expected behaviour and
  not a failure of the BO loop, it reflects the exploration-exploitation
  tension inherent in EI.  ``IntervalCoverage`` FAILing throughout despite
  this, though, means the GP's intervals are consistently too narrow, not
  just occasionally miscalibrated during exploration bursts.

* **UncertaintyEvolution FAIL (declining channels > 0):** EI rapidly collapses
  uncertainty near the optimum.  If the slope is too steep, increase
  ``--n-init`` so the Sobol phase builds a broader initial model, or
  increase ``--n-iter`` to give the model more exploration steps.

* **VarianceAlignment FAIL, especially a late collapse toward the end of
  the run:** predicted variance falling to a small fraction of actual
  squared error late in the run indicates the GP is increasingly
  overconfident exactly where EI is exploiting most heavily — worth
  watching for in any run where this check degrades sharply rather than
  gradually.

* **VarianceErrorCorrelation FAIL** (not observed in the run above, but
  possible at other seeds/step counts): the BoTorch GP occasionally
  produces near-zero std for points where the Fréchet distance matches the
  GP mean closely.  A low or negative Spearman correlation is only
  meaningful if there is sufficient spread in both sigma and absolute
  error.  Check whether the BO loop has collapsed to a narrow region too
  early.


References
----------

.. [Seifrid2022] Seifrid, M., Pollice, R., Aguilar-Granda, A., Chan, Z.,
   Doyle, K., Gao, T. C., Haberler, S., Ser, C. T., Vestfrid, J.,
   Wu, T. C., & Aspuru-Guzik, A. (2022).
   Autonomous chemical experiments: Challenges and perspectives on
   establishing a self-driving lab.
   *Accounts of Chemical Research*, 55(17), 2454-2466.
   https://doi.org/10.1021/acs.accounts.2c00220

.. [Bakshy2018] Bakshy, E., Dworkin, L., Karrer, B., Kashin, K.,
   Letham, B., Murthy, A., & Singh, S. (2018).
   AE: A domain-agnostic platform for adaptive experimentation.
   *NeurIPS Workshop on Systems for ML*.

.. [Frechet1957] Fréchet, M. (1957).
   Sur la distance de deux lois de probabilité.
   *Comptes Rendus de l'Académie des Sciences*, 244, 689-692.

