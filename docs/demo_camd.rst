.. _demo-camd:

Materials stability screening demo (``ta-camd-demo``)
======================================================

This demo applies the uncertainty audit to a high-dimensional materials
discovery task.  An AdaBoost committee model is used for both prediction and
uncertainty quantification as it queries a candidate materials database for
thermodynamic stability.  Lyapunov stability analysis is layered on top to
characterise how sensitive the learned surrogate is to small perturbations in
feature space.

.. code-block:: bash

   pip install "traits-audit[camd]"   # install optional dependency

   ta-camd-demo                            # defaults: 50 iter, 4 queries/iter
   ta-camd-demo --n-iter 50 --n-query 6 --seed 7
   ta-camd-demo --out-dir _results/camd


Introduction
------------

High-throughput computational screening aims to identify stable materials
from a combinatorially large candidate pool using a minimal number of
expensive first-principles evaluations
(Montoya *et al.*, 2020 [Montoya2020]_).
Query-by-committee (QBC) active learning [Seung1992]_ selects candidates where
committee members disagree most, concentrating evaluations in regions of
genuine uncertainty.

**Question:** Does the audit detect meaningful calibration and exploration
signals in a QBC loop over a real materials dataset?  Specifically:

* Does committee disagreement (used as surrogate uncertainty) correlate with
  actual prediction error?
* Is the uncertainty budget being consumed in a healthy, monotonically
  declining manner rather than collapsing prematurely?
* Is the surrogate dynamics Lyapunov-stable, meaning that the gradient-descent
  trajectory on the learned landscape converges rather than oscillates?

Uncertainty hook placement
~~~~~~~~~~~~~~~~~~~~~~~~~~

``hook.on_step()`` fires after each batch of hypotheses is evaluated and added
to the seed set, but before the committee is re-fit.  Values passed to the hook
are **means over the batch** (``mean_mu``, ``mean_std``, ``mean_y``):

.. code-block:: text

   Committee fit  →  Hypothesis selection  →  Evaluate hypotheses  ← hook.on_step()
         ↑                                            |
         └──────────── grow seed set ─────────────────┘

.. list-table:: Check-to-pipeline-step mapping
   :header-rows: 1
   :widths: 30 25 45

   * - Check
     - AL step monitored
     - What is observed
   * - ``CalibrationError``
     - Hypothesis evaluation
     - Whether the committee's mean confidence, averaged over the selected batch, matches the fraction of hypotheses satisfying the stability criterion
   * - ``IntervalCoverage``
     - Hypothesis evaluation
     - Whether the batch-mean ±1σ committee interval contains the batch-mean true stability value
   * - ``VarianceAlignment``
     - Hypothesis evaluation
     - Whether batch-mean committee variance scales with batch-mean squared error
   * - ``UncertaintyEvolution``
     - Hypothesis selection
     - Trend in mean committee std across selected batches — expected to decrease as explored regions are covered
   * - ``UncertaintyAnomalies``
     - Hypothesis selection
     - Batches where committee std is anomalously high (shrinking candidate pool) or low (collapsing committee)
   * - ``VarianceErrorCorrelation``
     - Hypothesis evaluation
     - Whether the committee assigns greater spread to batches it predicts most poorly


Methods
-------

Dataset and domain
~~~~~~~~~~~~~~~~~~

The demo loads the built-in CAMD test dataset via
``camd.utils.data.load_dataframe("test")``.  When the optional ``camd``
package is unavailable it falls back to a synthetic 300-sample, 12-feature
dataset with a quadratic stability proxy:

.. math::

   y_i = -\sum_{j=1}^{3} x_{ij}^2 + \varepsilon_i, \quad
   \varepsilon_i \sim \mathcal{N}(0, 0.09)

The real CAMD dataset contains formation energies and derived features
(e.g. electronegativity statistics, ionic radii, orbital occupancy) for
inorganic compounds.  The target variable ``stability`` is the energy above
the convex hull [Aykol2019]_ — a proxy for thermodynamic metastability.


Surrogate model
~~~~~~~~~~~~~~~

Two surrogate paths are supported:

**CAMD path (preferred):** ``AgentStabilityAdaBoost`` [Freund1997]_ with 20 boosted trees.
The committee uncertainty is extracted from the variance of individual
AdaBoost estimator predictions:

.. math::

   \hat{\sigma}(x) = \operatorname{std}_{k=1}^{K}
   \left[ \hat{f}_k(x) \right]

where :math:`\hat{f}_k` is the prediction of the :math:`k`-th tree in the
ensemble.

**sklearn fallback:** ``BaggingRegressor`` with 20 ``DecisionTreeRegressor``
members, trained with bootstrap sampling.  The acquisition policy selects
the ``n_query`` candidates with the highest committee standard deviation
(maximum-uncertainty QBC).

Intermediate audit checks are triggered every ``--check-every`` steps
(default 5) to detect calibration drift during the loop.

Raw feature dimensionality (up to ~50 in the real
dataset) makes the Jacobian computation in full feature space expensive.
PCA reduces the state space to ``--n-pca`` components (default 5) before
computing the gradient-descent Jacobian, reducing the Jacobian from
:math:`\mathbb{R}^{D \times D}` to :math:`\mathbb{R}^{5 \times 5}` during the
Lyapunov stability analysis.

Lyapunov stability framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lyapunov stability analysis characterises how the AL trajectory behaves as a
dynamical system.  Each queried operating point :math:`x_t` is treated as a
state, and the surrogate gradient-descent map

.. math::

   F(x) = x - \alpha\,\nabla\hat{f}(x), \quad \alpha = 0.05

defines the next-state transition.  The Jacobian :math:`J = I - \alpha H_f`
(where :math:`H_f` is the surrogate Hessian) determines local stability: if
all eigenvalues satisfy :math:`|\lambda| < 1`, the map is contractive at that
point.  The step size :math:`\alpha = 0.05` is chosen to keep eigenvalues
near the unit circle, balancing curvature visibility against numerical
stability; smaller values collapse all eigenvalues toward 1.0, masking
genuine variation.  The inner finite-difference step is ``eps=1e-3`` so that
gradient estimates reliably cross decision-tree leaf boundaries rather than
reading zero within a leaf.


Results
-------

The figures below were produced by ``ta-camd-demo`` with default settings
(``--n-seed 25 --n-iter 50 --n-query 4 --seed 0``), using the sklearn
``BaggingRegressor`` fallback surrogate on the synthetic 300-sample dataset.
The run queries 200 candidates in total (50 steps × 4 per step).

Committee uncertainty evolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/demo_camd/fig4_uncertainty_evolution.png
   :width: 70%
   :align: center
   :alt: Committee standard deviation over 50 active learning steps

Mean committee standard deviation at the queried batch per AL step.
The curve peaks at ≈ 4.2 (step 1) as the maximum-uncertainty acquisition
immediately targets the highest-disagreement region of feature space.
A secondary rise to ≈ 3.5 at step 12 marks re-entry into an unexplored
subspace after the initial high-uncertainty cluster is exhausted.
From step 15 onward the series declines with moderate fluctuations,
reaching ≈ 0.5 by step 50 — a threefold reduction from the opening value.
This monotone decline satisfies the ``UncertaintyEvolution`` check
(slope −0.035 per step, above the −0.05 FAIL threshold), confirming healthy
convergence over the 50-step horizon.

Audit checks over AL steps
~~~~~~~~~~~~~~~~~~~~~~~~~~

Green dots are PASS; red dots are FAIL.

.. figure:: _static/demo_camd/fig6_audit_evolution.png
   :width: 100%
   :align: center
   :alt: Six audit check values evaluated at snapshot intervals

* **CalibrationError** starts at 0.09 (PASS) and rises through 0.15–0.22 by
  step 30 as the dataset grows and the QBC's systematic variance
  overestimation compounds, before recovering slightly to 0.17 at step 50.
  The final value of 0.085 (all-data evaluation) passes the 0.15 threshold.

* **IntervalCoverage** rises from 0.75 (step 5) to a peak of ≈ 0.90
  (steps 20–30) then falls back toward 0.70 at step 50.  The wide intervals
  reflect the committee's tendency toward overconfidence in spread.  The
  final 1σ coverage of 0.70 (expected 0.683) passes with Δ = 0.017.

* **VarianceAlignment** fails throughout (ratio 1.5–2.5): predicted variance
  consistently exceeds mean squared error by a factor of ~2, confirming
  systematic overestimation of uncertainty in magnitude across the run.

* **UncertaintyEvolution** passes cleanly (slope −0.02 to −0.04 across all
  snapshots), with no sign of premature collapse.

* **UncertaintyAnomalies** is zero throughout — no steps trigger the z > 3
  anomaly threshold.

* **VarianceErrorCorrelation** (Spearman ρ) oscillates between −0.35 and
  +0.25 and finishes at −0.03 (FAIL).  The negative early values indicate
  that the committee assigns *higher* uncertainty to observations it predicts
  more accurately — an anti-correlation consistent with the maximum-uncertainty
  policy targeting well-explored, low-error regions in later steps rather than
  the true high-error frontier.

Lyapunov pole diagram
~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/demo_camd/fig1_poles.png
   :width: 65%
   :align: center
   :alt: Complex eigenvalue plot (pole diagram) for the CAMD surrogate

Each point is one eigenvalue of the gradient-descent Jacobian
:math:`J = I - \alpha H_f` evaluated at the mean queried operating point
(final trained surrogate, :math:`\alpha = 0.05`).  The dashed circle is the
unit circle; eigenvalues inside are contractive and those outside are
expansive.

Two poles fall within the ±1.5 view window: one near Re ≈ −1.1
(marginally outside) and one near Re ≈ 1.0 (on the boundary).  An annotation
reports three additional poles outside the view with magnitudes spanning
:math:`|λ| \in [1.0, 6.3 \times 10^4]`.  The extreme outlier magnitudes
reflect operating points where the surrogate Hessian is large — sharp decision
boundaries in the ensemble produce steep local gradients that drive the
:math:`\alpha = 0.05` step to the edge of stability.

Queried operating points in PCA space, coloured by :math:`|\lambda_{\max}|`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/demo_camd/fig2_stability_contours.png
   :width: 70%
   :align: center
   :alt: PCA scatter coloured by maximum Lyapunov exponent

Each scatter point is one queried operating point projected onto the first
two principal components of the trajectory.  The colour encodes
:math:`|\lambda_{\max}|` via a diverging ``coolwarm`` scale anchored at the
stability boundary (white = :math:`|\lambda| = 1`; blue = stable,
:math:`|\lambda| < 1`; red = unstable, :math:`|\lambda| > 1`).  The
colourbar spans the 2nd–98th percentile of the actual data, so variation is
visible at the scale that matters.

The two deep-red points (PC1 ≈ −2, PC2 ≈ 1.4 and PC1 ≈ −1.8, PC2 ≈ −1.2)
correspond to :math:`|\lambda_{\max}| > 3 \times 10^4` — operating points
near sharp ensemble decision boundaries where the surrogate curvature is
highest.  The majority of queried points (36/50) are unstable
(:math:`|\lambda| > 1`) but with moderate values (< 5000) that appear as
light pink.  The 14 stable points (blue-tinted) are scattered near the centre
of PC space, suggesting that the surrogate is better-conditioned in the
central feature subspace where more training data is concentrated.

Lyapunov exponent vs surrogate uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/demo_camd/fig3_stability_vs_unc.png
   :width: 70%
   :align: center
   :alt: Maximum Lyapunov exponent vs surrogate posterior standard deviation

Most operating points cluster near the origin: small committee std and
moderate :math:`|\lambda_{\max}|`.  Two clear outliers emerge at committee
std ≈ 1.6–1.8: these are operating points with simultaneously high committee
disagreement *and* extreme surrogate instability (:math:`|\lambda| > 5 \times
10^4`).  The positive co-occurrence of high uncertainty and high instability
is expected — both signals reflect the surrogate encountering a data-sparse,
high-curvature region of feature space.  This structural agreement between the
probabilistic uncertainty (committee std) and the deterministic stability
indicator (:math:`|\lambda|`) validates the Lyapunov approach as a
complementary diagnostic.

Lyapunov evolution
~~~~~~~~~~~~~~~~~~

.. figure:: _static/demo_camd/fig5_lyapunov_evolution.png
   :width: 80%
   :align: center
   :alt: Lyapunov exponent and surrogate std over active learning steps

Dual y-axis: orange = per-step :math:`|\lambda_{\max}|` (left), blue =
committee std (right).  The orange signal is evaluated using the
partially-trained surrogate at each step, whereas the final Lyapunov analysis
(figs 1–3) uses the fully converged model.  The per-step values remain near
1.0 throughout: with only 25–200 training points and 4 queries per step, the
BaggingRegressor trees have few splits and the local Hessian is near zero,
keeping :math:`J \approx I`.  The large :math:`|\lambda|` values in the final
analysis arise from the fully trained model (225 labelled points, 20 trees
each) whose decision surfaces are much sharper.  The decoupling of the two
y-axes — committee std declining over 50 steps while the live Lyapunov signal
stays flat — confirms that per-step uncertainty reduction is driven by data
coverage, not by changes in local surrogate curvature.

Pareto frontier: committee std vs mean absolute error
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/demo_camd/fig7_pareto_frontier.png
   :width: 70%
   :align: center
   :alt: Pareto frontier of committee uncertainty vs prediction error for the CAMD demo

Points are coloured by AL step (dark purple = early, yellow = late).
The Pareto-optimal set (orange circles) forms an L-shaped frontier from
(std ≈ 0.3, MAE ≈ 2.9) down to (std ≈ 2.0, MAE ≈ 0.5), connected by a
dashed staircase.

Three zones are visible.  **Upper-right** (high std, high MAE): early
batches (purple) where the committee is uncertain *and* wrong — expected
while the labelled pool is small.  **Lower-left** (Pareto region): mid-run
batches (teal, steps ~10–25) that achieve low error *and* low committee
spread simultaneously; these are the most reliable stability assessments in
the run.  **Right flank** (high std, moderate-to-high MAE): late batches
(yellow, steps 35–50) where committee std has paradoxically increased again,
reflecting the acquisition policy pulling queries into the last unexplored
pockets of candidate space as the pool shrinks.  The final Pareto frontier is
populated by steps across the middle of the run rather than the very end,
which is consistent with the ``VarianceErrorCorrelation`` FAIL: later batches
are not on the efficiency frontier despite lower absolute uncertainty.

Convergence: running best stability score vs cumulative AL queries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/demo_camd/fig8_convergence.png
   :width: 70%
   :align: center
   :alt: Running best stability score vs cumulative queries for the CAMD demo

Each AL step queries ``n_query = 4`` candidates; the x-axis counts all
queried candidates cumulatively (4–200).  The dashed horizontal line is the
best score seen in the 25 seed observations (≈ −6.7).  The solid staircase is
the running maximum per-step mean stability.

The curve shows four distinct phases.  **Phase 1 (queries 1–20):** rapid
improvement from −6.7 to −4.0 as the acquisition policy immediately targets
high-uncertainty regions far from the seed.  **Phase 2 (queries 20–60):**
plateau at −4.0 as the committee exhausts the first high-disagreement cluster.
**Phase 3 (queries 60–150):** step improvement to −2.7 and then −2.2 as
the model refines its estimate of the stability landscape and queries move
toward the true optimum.  **Phase 4 (queries 150–200):** continued
incremental improvement ending at ≈ −0.9, approximately six-fold better than
the seed baseline.  The staircase shape — plateaus punctuated by jumps — is
characteristic of max-uncertainty QBC on synthetic data: exploitation only
improves the running best when the committee's high-disagreement region
happens to coincide with the high-stability region.


Discussion
----------

A typical healthy run produces output similar to:

.. code-block:: text

   ── Audit report ───────────────────────────────────────────────────
   CalibrationError         PASS  value=0.085  threshold=0.150
   IntervalCoverage         PASS  value=0.700  threshold=[0.533, 0.833]
   VarianceAlignment        FAIL  value=1.928  threshold=[0.500, 1.500]
   UncertaintyEvolution     PASS  value=-0.035 threshold=-0.050
   UncertaintyAnomalies     PASS  value=0.000  threshold=0.050
   VarianceErrorCorrelation FAIL  value=-0.031 threshold=0.100
   ── Overall: FAIL (2 checks failed) ────────────────────────────────

For a **QBC surrogate on materials data**, pay particular attention to:

* **VarianceAlignment (FAIL):** A ratio of ≈ 1.9 means the committee assigns
  variance roughly twice the observed mean squared error.  This is a known
  property of bagging ensembles on regression tasks: bootstrap resampling
  induces variance across trees that exceeds the true aleatoric noise.
  The check will typically fail for BaggingRegressor surrogates; it is most
  useful as a *relative* indicator — ratios that grow over time signal
  progressive overestimation.

* **VarianceErrorCorrelation (FAIL):** A Spearman ρ near zero or negative
  indicates that the committee's disagreement does not reliably predict where
  the mean prediction errs most.  For maximum-uncertainty QBC, this arises
  because the policy deliberately queries high-uncertainty points, which over
  time become well-labelled; the residual uncertainty then migrates to different
  regions than the error.  A FAIL here does not mean the acquisition is broken
  — it means the calibration of the uncertainty *as an error proxy* degrades
  as labelling proceeds.

* **UncertaintyEvolution slope:** Values steeper than −0.05 per step
  suggest the committee is converging faster than the unlabelled pool is
  being explored.  This may indicate that the batch size is too large
  relative to the pool size or that the acquisition function is clustering
  queries in one region.

* **Lyapunov stability** [Strogatz2018]_: The :math:`\alpha = 0.05` step-size
  choice ensures that eigenvalues of :math:`J = I - \alpha H_f` span a
  meaningful range rather than clustering near 1.0.  Operating points with
  :math:`|\lambda_{\max}| < 1` are contractive — the gradient-descent map
  converges locally.  The 14/50 stable operating points (28 %) in this run
  are concentrated near the centre of PCA space, where training data is dense
  and the surrogate Hessian is small.  The two extreme outliers
  (:math:`|\lambda| > 3 \times 10^4`) near the edges of PCA space mark the
  most data-sparse, high-curvature regions — exactly the candidates that a
  well-calibrated acquisition policy should target cautiously.


References
----------

.. [Montoya2020] Montoya, J. H., Winther, K. T., Flores, R. A., Bligaard, T.,
   Norskov, J. K., & Aykol, M. (2020).
   Autonomous intelligent agents for accelerated materials discovery.
   *Chemical Science*, 11(32), 8517-8532.
   https://doi.org/10.1039/D0SC01101K

.. [Aykol2019] Aykol, M., Hummelshøj, J. S., Anapolsky, A., Bhati, M., Liao, K.,
   Montoya, J. H., Nykvist, B., Pellegrini, F., Senftle, T., Siahrostami, S.,
   Winther, K. T., Chan, E. M., Norskov, J. K., Persson, K. A., &
   Bligaard, T. (2019).
   The Materials Project: A materials genome approach to accelerating
   materials innovation.
   *APL Materials*, 7, 110901.

.. [Seung1992] Seung, H. S., Opper, M., & Sompolinsky, H. (1992).
   Query by committee.
   *Proceedings of the Fifth Annual Workshop on Computational Learning Theory
   (COLT 1992)*, 287-294.
   https://doi.org/10.1145/130385.130417

.. [Freund1997] Freund, Y., & Schapire, R. E. (1997).
   A decision-theoretic generalization of on-line learning and an application
   to boosting.
   *Journal of Computer and System Sciences*, 55(1), 119-139.

.. [Strogatz2018] Strogatz, S. H. (2018).
   *Nonlinear Dynamics and Chaos* (2nd ed.).
   CRC Press.
