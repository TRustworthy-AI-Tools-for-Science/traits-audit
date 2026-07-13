.. _demo-camd:

Materials stability screening demo (``ta-camd-demo``)
======================================================

This demo applies the uncertainty audit to a high-dimensional materials
discovery task.  An AdaBoost committee model queries a candidate materials
database for thermodynamic stability; Lyapunov stability analysis is then
layered on top, fitting a linear dynamical-systems model to the *observed*
AL query trajectory (via DMDc) to characterise whether it is contracting or
diverging.

.. code-block:: bash

   pip install "traits-audit[camd]"   # install optional dependency

   ta-camd-demo                            # defaults: 100 iter, 4 queries/iter
   ta-camd-demo --n-iter 100 --n-query 4 --seed 0   # used to generate the figures on this page
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

**Questions:** (1) Does the audit detect meaningful calibration and
exploration signals in a QBC loop over a real materials dataset?
(2) Does committee disagreement correlate with actual prediction error?
(3) Is the AL query trajectory Lyapunov-stable — does the DMDc-fitted linear
model of the trajectory contract rather than diverge?

Uncertainty hook placement
~~~~~~~~~~~~~~~~~~~~~~~~~~

``hook.on_step()`` fires after each batch of hypotheses is evaluated and
added to the seed set, but before the committee is re-fit.  Values passed
to the hook are **means over the batch**:

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
     - Whether committee confidence, averaged over the batch, matches the fraction satisfying the stability criterion
   * - ``ConformalCoverage``
     - Hypothesis evaluation
     - Distribution-free marginal coverage over the batch
   * - ``CRPS``
     - Hypothesis evaluation
     - CRPS as a proper scoring rule on the evaluated batch
   * - ``NegativeLogLikelihood``
     - Hypothesis evaluation
     - Gaussian NLL on the evaluated batch
   * - ``PITUniformity``
     - Hypothesis evaluation
     - PIT uniformity across all evaluated hypotheses
   * - ``IntervalScore``
     - Hypothesis evaluation
     - Winkler score penalising non-coverage and excessive width
   * - ``IntervalCoverage``
     - Hypothesis evaluation
     - Whether the batch-mean ±1σ committee interval contains the batch-mean true stability value
   * - ``VarianceAlignment``
     - Hypothesis evaluation
     - Whether batch-mean committee variance scales with batch-mean squared error
   * - ``UncertaintyEvolution``
     - Hypothesis selection
     - Count of channels with a declining uncertainty trend (0 = all stable)
   * - ``UncertaintyAnomalies``
     - Hypothesis selection
     - Fraction of current uncertainty values anomalously far from a historical baseline; skipped when no baseline is provided
   * - ``VarianceErrorCorrelation``
     - Hypothesis evaluation
     - Whether the committee assigns greater spread to batches it predicts most poorly


Methods
-------

Dataset and domain
~~~~~~~~~~~~~~~~~~

The demo downloads the OQMD Voronoi-Magpie fingerprints dataset (~150 MB)
from ``data.matr.io`` on first run, caches it under
``~/.cache/traits_audit/``, and reads it with ``pd.read_pickle()``.  If the
download fails it falls back to a synthetic 300-sample, 12-feature dataset
with a quadratic stability proxy:

.. math::

   y_i = -\sum_{j=1}^{3} x_{ij}^2 + \varepsilon_i, \quad
   \varepsilon_i \sim \mathcal{N}(0, 0.09)

The real CAMD dataset contains formation energies and derived features
(electronegativity statistics, ionic radii, orbital occupancy) for
inorganic compounds.  The target variable is the energy above the convex hull
[Aykol2019]_ — a proxy for thermodynamic metastability.


Surrogate model
~~~~~~~~~~~~~~~

Two surrogate paths are supported:

**CAMD path (preferred):** ``AgentStabilityAdaBoost`` [Freund1997]_ with 20
boosted trees.  The committee uncertainty is the standard deviation of
individual estimator predictions:

.. math::

   \hat{\sigma}(x) = \operatorname{std}_{k=1}^{K}
   \left[ \hat{f}_k(x) \right]

Candidates are ranked by a lower confidence bound (LCB) [Montoya2020]_:

.. math::

   \text{LCB}(x) = \hat{f}(x) - \alpha\,\hat{\sigma}(x), \quad \alpha = 0.5

The :math:`\alpha = 0.5` value matches the best-performing "AB-ε0-α0.5"
agent in [Montoya2020]_.

**sklearn fallback:** ``BaggingRegressor`` with 20 ``DecisionTreeRegressor``
members and the same LCB acquisition (:math:`\alpha = 0.5`).

Intermediate audit checks are triggered every ``--check-every`` steps
(default 4) to detect calibration drift during the loop.

Alignment with Montoya et al. (2020)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The demo directly simulates the "AB-ε0-α0.5" agent described in
[Montoya2020]_.  The key design choices and where they differ from the
full-scale paper are:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Aspect
     - Montoya et al. (2020)
     - This demo
   * - Surrogate model
     - AdaBoost on Voronoi/Magpie features (20 estimators)
     - Same (CAMD path) or BaggingRegressor fallback
   * - Acquisition function
     - LCB, :math:`\alpha = 0.5`
     - LCB, :math:`\alpha = 0.5` (both paths)
   * - Exploration strategy
     - ε-greedy, ε = 0 (fully greedy)
     - Fully greedy
   * - Stability threshold
     - ≤ 0.1 eV/atom above hull (simulation)
     - CAMD default (governed by ``hull_distance``)
   * - Candidate pool
     - ~1,600–2,000 hypothetical binary phases (Fe-X or M-O)
     - CAMD test dataset or 300-sample synthetic fallback
   * - Seed size
     - ~36,000 ICSD-derived OQMD entries
     - 25 randomly sampled seed points
   * - Batch size per step
     - 50 (simulation); 10 DFT (active campaigns)
     - 4 (``--n-query``), adjustable

The reduced seed and batch sizes are deliberate: the demo runs without DFT
in under a minute.  The acquisition strategy, uncertainty model, and LCB
parameterisation are taken directly from the paper so that audit results are
interpretable in its context.

.. _lyapunov-framework:

Lyapunov stability framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rather than differentiating a scalar surrogate at a fixed operating point
(the approach used in the PyBAMM and SDL demos, via the gradient-descent map
:math:`F(x) = x - \alpha\nabla\hat{f}(x)` and its Jacobian
:math:`J = I - \alpha H_f`), CAMD characterises stability from the
*observed* AL trajectory itself, via Dynamic Mode Decomposition with Control
(DMDc, :func:`traits_audit.dmdc.fit_dmdc`; Proctor, Brunton & Kutz, 2016).  Each queried batch
contributes one row to an augmented state

.. math::

   \tilde{s}_t = [\,\text{PC}_1(x_t), \dots, \text{PC}_5(x_t), \bar\sigma_t\,]

— five **StandardScaler-normalised PCA coordinates** of the queried
composition plus that batch's mean committee standard deviation
(:math:`D = 6`) — and DMDc fits a reduced-order linear operator :math:`A_r`
(rank :math:`r = 5`) such that :math:`\tilde{z}_{t+1} \approx A_r\,\tilde{z}_t`
in the PCA-reduced subspace.  Unlike :math:`J = I - \alpha H_f` (always real
and symmetric), :math:`A_r` is a *general* matrix and can have
complex-conjugate eigenvalues representing spiral modes in the joint
state/uncertainty space — though for the run shown below the recovered
eigenvalues came out close to the real axis (see the pole diagram).
:math:`|\lambda_{\max}(A_r)| < 1` means the fitted dynamics are contractive;
:math:`> 1` means they are expansive.

Per-step values come from :func:`traits_audit.dmdc.stability_convergence`,
which refits :math:`A_r` on a **growing prefix** of the trajectory
(:math:`\tilde{s}_0 \dots \tilde{s}_t`) at each step :math:`t`, tracking how
the identified dynamics evolve as evidence accumulates.  The first few steps
(before enough points exist for an overdetermined fit) are undefined and
reported as NaN — ``LyapunovStabilityCheck`` (see below) drops these rather
than counting them as unstable.  Every DMDc fit here — both the per-prefix
fits and the final whole-trajectory fit used for the pole diagram and
Lyapunov-function contour below — centers the augmented-state trajectory
before fitting by default (:mod:`traits_audit.detrend`), rather than fitting
on raw, uncentered coordinates.

``LyapunovStabilityCheck`` is wired into the audit pipeline at its default
``window=None`` — a *global*, cumulative-since-step-0 verdict, in contrast to
the PyBAMM demo's ``window=30`` local/recent-window verdict (see
:doc:`checks` and ``LYAPUNOV_ANALYSIS.md`` for the full local/global
distinction).  This cumulative *aggregation* is a separate concern from the
*spatial* locality of each :math:`\lambda_{\max}(A_r(t))` value itself, which
always reflects a linear operator fit around the trajectory's current
region, not a whole-landscape guarantee.


Results
-------

The figures below were produced by ``ta-camd-demo`` with
``--n-seed 25 --n-iter 100 --n-query 4 --seed 0``, using the sklearn
``BaggingRegressor`` fallback surrogate on the synthetic 300-sample dataset
(400 labelled points total: 25 seed + 100 steps × 4 per step).

Committee uncertainty evolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/demo_camd/fig4_uncertainty_evolution.png
   :width: 70%
   :align: center
   :alt: Committee standard deviation over 100 active learning steps

Mean committee standard deviation at the queried batch per AL step.
The series peaks early as maximum-uncertainty acquisition immediately
targets the highest-disagreement region of feature space, then declines
with moderate fluctuations as explored regions become well-labelled.
The monotone decline satisfies the ``UncertaintyEvolution`` check,
confirming healthy convergence over the 100-step horizon.

Audit check grid
~~~~~~~~~~~~~~~~

.. figure:: _static/demo_camd/fig10_check_grid.png
   :width: 100%
   :align: center
   :alt: Heatmap of all audit check pass/fail margins across AL snapshot steps

Rows are audit checks; columns are snapshot steps (every ``--check-every``
steps, default 4) plus a final all-data evaluation.  Cell colour encodes
distance from the pass/fail threshold: dark green = deeply passing,
white = at the boundary, dark red = deeply failing.

Reading across a row reveals how a single check evolves over the campaign.
``VarianceAlignment`` fails persistently (predicted variance exceeds mean
squared error by a factor of ~2 throughout) — a known property of bagging
ensembles where bootstrap resampling induces excess variance.
``VarianceErrorCorrelation`` oscillates near the pass boundary, reflecting
the maximum-uncertainty policy querying high-uncertainty points that
subsequently become well-labelled, decoupling committee disagreement from
prediction error.  ``LyapunovStability`` shows a blank (``—``) at every
intermediate snapshot and a value only at the final column: its
precomputed ``lambda_max`` series is built from the *complete* growing-prefix
DMDc sweep after the AL loop ends, so — unlike the other rows — it has no
intermediate-snapshot equivalent to report.

Audit checks over AL steps
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/demo_camd/fig6_audit_evolution.png
   :width: 100%
   :align: center
   :alt: Eleven audit check values evaluated at snapshot intervals

Green dots are PASS; red dots are FAIL.  ``VarianceAlignment`` is the
dominant persistent FAIL throughout the run.  ``CalibrationError`` and
``IntervalCoverage`` both pass at most snapshots, confirming that the
committee's uncertainty estimates are directionally correct even if their
magnitude is inflated.  ``UncertaintyAnomalies`` is zero throughout — no
steps trigger the :math:`|z| > 3` anomaly threshold.

Lyapunov pole diagram
~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/demo_camd/fig1_poles.png
   :width: 65%
   :align: center
   :alt: Complex eigenvalue plot (pole diagram) for the DMDc-fitted dynamics matrix

Each point is one eigenvalue of the rank-5 DMDc operator :math:`A_r` fit on
the complete (whole-trajectory) augmented-state history.  The dashed circle
is the unit circle; eigenvalues inside are contractive and those outside are
expansive.  All five eigenvalues fall close to the real axis and well inside
the unit circle for this run.  :math:`A_r` is a general (non-symmetric)
matrix and can in principle produce complex-conjugate pairs — unlike the
always-real, symmetric gradient-descent Jacobian used in the PyBAMM/SDL
demos — but the fitted operator here happens to be close to normal.

Queried operating points coloured by growing-prefix :math:`|\lambda_{\max}|`, with the final Lyapunov function contour
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/demo_camd/fig2_stability_contours.png
   :width: 70%
   :align: center
   :alt: PCA scatter coloured by growing-prefix maximum Lyapunov exponent, overlaid on the final A_r's Lyapunov function contour

Background contours show :math:`V(x) = x^T P x`, the discrete Lyapunov
function solved for the final, whole-trajectory :math:`A_r` (the same
discrete-Lyapunov-equation solver used by the PyBAMM/SDL demos, reused here
on the DMDc-fitted operator rather than a gradient-descent Jacobian).  Each
scatter point is one queried
operating point projected onto the first two principal components, coloured
by its own **growing-prefix** :math:`|\lambda_{\max}(A_r(t))|` at the step it
was queried.  Most points are well inside the stability boundary (blue,
:math:`|\lambda| < 1`); the handful of red/light points are early, small-:math:`t`
steps whose growing-prefix fit was still volatile (see the evolution plot
below), not a persistent instability tied to any region of feature space.

Growing-prefix :math:`|\lambda_{\max}|` vs surrogate uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/demo_camd/fig3_stability_vs_unc.png
   :width: 70%
   :align: center
   :alt: Growing-prefix maximum Lyapunov exponent vs surrogate posterior standard deviation

Each point pairs one AL step's growing-prefix :math:`|\lambda_{\max}(A_r(t))|`
with that step's mean committee standard deviation.  Unlike the tighter
co-occurrence sometimes seen with a gradient-descent-Jacobian approach, the
few unstable points here (above the dashed stability boundary) occur at
moderate — not extreme — committee std, and several of the highest-std
points are comfortably stable.  This is expected: DMDc stability reflects how
*predictable the trajectory's own dynamics* are as a linear system, which is
a different question from how uncertain the committee is at any one queried
point; the two remain complementary, largely independent diagnostics.

Lyapunov evolution over the campaign
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/demo_camd/fig5_lyapunov_evolution.png
   :width: 80%
   :align: center
   :alt: Growing-prefix Lyapunov exponent and surrogate std over active learning steps

Dual y-axis: orange = growing-prefix :math:`|\lambda_{\max}(A_r(t))|` (left),
blue = committee std (right).  :func:`traits_audit.dmdc.stability_convergence`
refits :math:`A_r` on the augmented-state trajectory up to step :math:`t` at
every step; the earliest refits are naturally volatile (few points feeding a
rank-5 fit), producing large early swings up to :math:`|\lambda_{\max}|
\approx 4`.  As the trajectory lengthens the growing-prefix fit becomes
better-determined and :math:`|\lambda_{\max}|` settles below the stability
boundary (dashed) within about 20 steps, ending near 0.19 for this run.  The
committee-std curve stays volatile throughout and is largely decoupled from
:math:`|\lambda_{\max}|` — DMDc stability and committee uncertainty remain
complementary, non-redundant diagnostics.

Pareto frontier: committee std vs mean absolute error
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/demo_camd/fig7_pareto_frontier.png
   :width: 70%
   :align: center
   :alt: Pareto frontier of committee uncertainty vs prediction error

Points are coloured by AL step (dark purple = early, yellow = late).
The Pareto-optimal set (orange circles) forms an L-shaped frontier.
Early batches (purple, upper-right) have high committee std and high MAE;
mid-run batches (teal) achieve low error and low spread simultaneously;
late batches can drift back rightward as acquisition exhausts the
highest-priority candidates and moves into the remaining pool.

Materials exploration campaign
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/demo_camd/fig9_exploration_campaign.png
   :width: 100%
   :align: center
   :alt: Chemical space exploration map and coverage metrics over AL steps

**Left panel** — hexbin background shows candidate density in Pauling
electronegativity (EN) space (real OQMD data) or a PCA projection
(synthetic fallback).  Blue diamonds are the 25 initial seed materials;
coloured circles are the queried batches (plasma colourmap, dark purple =
early, bright yellow = late).  The concentration of queries in the
electropositive EN 0.8–2.0 range reflects LCB targeting the most
predicted-stable region.

**Right panel** — grid-based exploration metrics over a 12 × 12 bin grid
on the 2-D EN/PCA space.  **Coverage** (blue line): cumulative fraction of
non-empty pool grid cells visited.  **Batch novelty** (orange bars):
fraction of each batch landing in cells not yet visited.  Novelty is high
early and declines as grid cells fill up; coverage grows but never saturates,
consistent with a greedy policy concentrating queries on predicted-stable
compositions rather than spreading uniformly.

Discovery rate vs random baseline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/demo_camd/fig11_discovery_rate.png
   :width: 100%
   :align: center
   :alt: Cumulative stable materials discovered vs random baseline over AL queries

This figure mirrors the primary evaluation metric of Montoya et al.
(2020) [Montoya2020]_.  A material is classified as stable if its true
:math:`\Delta E` falls at or below the 25th percentile of the full pool.

**Left panel** — the solid blue line is the AL campaign's cumulative stable
count; the dashed black line is the analytical expected count under uniform
random selection, :math:`\mu_\text{rand}(k) = k \times p_\text{stable}`.
No random policy is actually run — the dashed line and grey band are
computed in closed form.  The grey band is the analytical ±1σ envelope
:math:`\mu_\text{rand}(k) \pm \sqrt{k\,p_\text{stable}(1 - p_\text{stable})}`,
the standard deviation of a :math:`\text{Binomial}(k, p_\text{stable})`
distribution.

**Right panel** — enrichment factor :math:`N_\text{AL}(k) / \mu_\text{rand}(k)`.
Values above 1× mean the agent is finding stable materials faster than
random selection; the annotation reports the terminal enrichment.
Montoya et al. found 383 new stable or nearly-stable materials across 16
campaigns using this identical acquisition strategy, significantly
outperforming random selection.

Running best :math:`\Delta E` vs cumulative AL queries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/demo_camd/fig8_convergence.png
   :width: 70%
   :align: center
   :alt: Running best stability score vs cumulative queries for the CAMD demo

The running minimum :math:`\Delta E` found across all queries.  The dashed
horizontal line is the best value among the 25 seed observations.  The curve
typically drops rapidly in the first 10–20 queries as LCB targets the most
predicted-stable candidates, then plateaus as the committee concentrates on
regions of high uncertainty that are not necessarily the true global minimum.


Discussion
----------

A typical run with ``--n-iter 100`` produces an audit report similar to:

.. code-block:: text

   ── Audit report ───────────────────────────────────────────────────
   CalibrationError         PASS  value=0.085  threshold=0.150
   IntervalCoverage         PASS  value=0.700  threshold=[0.533, 0.833]
   VarianceAlignment        FAIL  value=1.928  threshold=1.0
   UncertaintyEvolution     PASS  value=0     threshold=0.0
   UncertaintyAnomalies     PASS  value=0.000  threshold=0.050
   VarianceErrorCorrelation FAIL  value=-0.031 threshold=0.100
   ── Overall: FAIL (2 checks failed) ────────────────────────────────

.. list-table:: Check interpretation guide
   :header-rows: 1
   :widths: 30 20 50

   * - Check
     - Threshold
     - What a FAIL means for QBC on materials data
   * - CalibrationError
     - ≤ 0.15
     - Committee spread is systematically mismatched to empirical residuals
   * - IntervalCoverage
     - 53–83 %
     - ±1σ committee intervals cover too few or too many true values
   * - VarianceAlignment
     - 0.5–1.5
     - Mean predicted variance is not commensurate with mean squared error; ratios > 1.5 are typical for BaggingRegressor and should be interpreted as relative indicators rather than absolute failures
   * - UncertaintyEvolution
     - slope ≥ −0.05
     - Uncertainty is collapsing faster than data collection justifies
   * - UncertaintyAnomalies
     - ≤ 5 % steps with \|z\| > 3
     - Sporadic uncertainty spikes indicating a numerically unstable step
   * - VarianceErrorCorrelation
     - Spearman ρ ≥ 0.1
     - Committee disagreement does not track where the model errs; common in max-uncertainty QBC as high-uncertainty regions become well-labelled over time

* **VarianceAlignment (persistent FAIL):** A ratio of ≈ 1.9 means the
  committee assigns variance roughly twice the observed MSE.  This is a
  known property of bagging ensembles: bootstrap resampling induces
  inter-tree variance that exceeds the true aleatoric noise.  Monitor
  for *growth* in this ratio over the campaign — a ratio that increases
  from 1.5 to 3.0 over 100 steps signals progressive overestimation that
  warrants investigation.

* **VarianceErrorCorrelation (often FAIL):** Spearman ρ near zero or
  negative indicates that committee disagreement does not reliably predict
  where the mean prediction errs most.  For maximum-uncertainty QBC, this
  arises because the policy deliberately queries high-uncertainty points,
  which become well-labelled; the residual uncertainty then migrates away
  from the current error frontier.

* **Lyapunov stability** [Strogatz2018]_: Operating points with
  :math:`|\lambda_{\max}| < 1` are contractive — the gradient-descent map
  converges locally.  Unstable points (:math:`|\lambda| > 1`) mark
  data-sparse, high-curvature regions of feature space — exactly the
  candidates a well-calibrated acquisition policy should approach cautiously.


References
----------

.. [Montoya2020] Montoya, J. H., Winther, K. T., Flores, R. A., Bligaard, T.,
   Hummelshøj, J. S., & Aykol, M. (2020).
   Autonomous intelligent agents for accelerated materials discovery.
   *Chemical Science*, 11(32), 8517–8532.
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
