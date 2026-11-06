# Derivation: AuditStateSpacePolicy and AuditBayesianDynamicsPolicy

This document derives the audit-augmented state-space policies.  It is a
companion to `state_space_bayes_derivation.md`, which derives the base
`StateSpacePolicy` and `BayesianDynamicsPolicy`.  The reader should be
familiar with those derivations before reading this one.

The central change is the **substitution of the per-parameter ECM variance
vector with the audit health vector** as the belief-state augmentation.
§1 motivates the change.  §2 defines the canonical audit health vector.
§3 re-derives the belief-state construction under the new augmentation.
§4 derives the health-cost term.  §5 gives the full acquisition scores.
§6 discusses dynamics of experiment health.  §7 compares augmentation
strategies.

---

## 1  Motivation: two kinds of uncertainty

The standard belief-state augmentation appends the **per-parameter ECM
variance** $u_t \in \mathbb{R}^n$ to the state:

$$
\tilde{s}_t^\text{param} = \begin{bmatrix} s_t \\ w \, u_t \end{bmatrix},
\qquad
u_t^{(j)} = \operatorname{Var}(\theta^{(j)} \mid \text{EIS data}).
$$

This encodes a single question: *how uncertain is each circuit element?*

The audit system answers five distinct questions:

1. **Is the raw EIS data valid?** (LinKK Kramers-Kronig consistency)
2. **Does the ECM model capture the physics?** (ECM KK self-consistency; DRT structure)
3. **Are the fitted parameters stable?** (MCMC posterior max CV)
4. **How well is the predictive uncertainty characterised?** (calibration, correlation, anomalies)
5. **Is the surrogate a sound guide for the loop?** (local stability of the surrogate
   landscape; conformal coverage)

The audit health vector $v_t \in \mathbb{R}^{14}$ encodes all five classes.
Using $v_t$ instead of $u_t$ shifts the policy from tracking *where* uncertainty
is concentrated (parameter space) to tracking *how healthy* the entire
experiment pipeline is (system level).

The policy can then learn that:
- high linKK RMSE signals a noisy or artefact-contaminated measurement;
- a model self-consistency failure (kkV) means the ECM circuit structure no longer
  describes the physical processes present in the cell;
- DRT peak mismatch warns that the ECM is under- or over-parameterised;
- calibration drift means the GP's uncertainty estimates are becoming unreliable.

---

## 2  The canonical audit health vector

### 2.1  Full 14-dimensional form

$$
v_t =
\begin{pmatrix}
  \alpha_t \\
  \mathrm{CE}_t \\
  \mathrm{ENCE}_t \\
  \mathrm{MCA}_t \\
  \rho_t \\
  d_t \\
  \phi_t \\
  \ell_t \\
  c_t \\
  \kappa_t \\
  \psi_t \\
  \delta_t \\
  \lambda^\text{st}_t \\
  q_t
\end{pmatrix}
\in \mathbb{R}^{14}.
$$

The vector is partitioned into three groups:

#### Part A — Predictive quality (indices 0–6)

These metrics assess how well the downstream GP model characterises its own
uncertainty.  They are produced by the TSModel and Evaluate stages of the
audit pipeline.

| Idx | Symbol | Source check | Formula / source | Direction |
|-----|--------|--------------|-----------------|-----------|
| 0 | $\alpha$ | `VarianceAlignmentCheck` | $\bar{\sigma}^2_\text{pred} / \bar{\sigma}^2_\text{true}$ | ideal = 1.0 |
| 1 | $\mathrm{CE}$ | `KuleshovCalibrationCheck` | $\int_0^1 |F_\text{pred}(p) - p| \, dp$ (Kuleshov 2018) | lower ↓ |
| 2 | $\mathrm{ENCE}$ | `ENCECheck` | $\frac{1}{B} \sum_b |\hat{\sigma}_b - \sigma_b| / \sigma_b$ (Levi 2022) | lower ↓ |
| 3 | $\mathrm{MCA}$ | AuditResult `"TraitsCalibration"` (`details["miscalibration_area"]`), from battery-forecast's checks, not a traits-audit check | Area between calibration curve and diagonal | lower ↓ |
| 4 | $\rho$ | `VarianceErrorCorrelationCheck` | $r_S(\sigma^2_\text{pred}, |y - \hat{y}|)$ (Spearman) | higher ↑ |
| 5 | $d$ | `UncertaintyEvolutionCheck` | Channels with slope $< -1\%$/step | lower ↓ |
| 6 | $\phi$ | `UncertaintyAnomalyCheck` | Fraction $|z\text{-score}| > 3$ | lower ↓ |

#### Part B — EIS inference quality (indices 7–11)

These metrics assess the upstream measurement and parameter-fitting pipeline.
They are produced by the Inference audit stage and the DRT analysis module of
battery-forecast (`battery_forecast.audit`), not by traits-audit checks.

| Idx | Symbol | Source / AuditResult name | Physical question | Direction |
|-----|--------|--------------------------|-------------------|-----------|
| 7 | $\ell$ | `"LinKK Reconstruction Error"` | Is the EIS measurement KK-consistent? | lower ↓ |
| 8 | $c$ | `"ECM Parameter Stability (Max CV)"` | Are the MCMC-fitted parameters well-constrained? | lower ↓ |
| 9 | $\kappa$ | `"Model Self-Consistency — Simulated Mean"` | Does the ECM reproduce the impedance spectrum (kkV)? | lower ↓ |
| 10 | $\psi$ | `"Model Self-Consistency — Posterior Draws"` | Is the ECM posterior KK-consistent? | lower ↓ |
| 11 | $\delta$ | `"DRT Peak Count Mismatch"` | Does the ECM structure match the DRT peak count? | lower ↓ |

#### Part C — Surrogate and coverage quality (indices 12–13)

These metrics assess whether the surrogate that drives acquisition is a sound
guide: whether its landscape is locally stable, and whether its prediction
intervals cover at the nominal rate.  They are produced by the active-learning
audit pipeline.

| Idx | Symbol | Source check | Formula / source | Direction |
|-----|--------|--------------|-----------------|-----------|
| 12 | $\lambda^\text{st}$ | `LyapunovStabilityCheck` | Fraction of operating points with $|\lambda_\text{max}| < 1$ for gradient descent on the surrogate | higher ↑ |
| 13 | $q$ | `ConformalCoverageCheck` | $\hat{q} / z_\text{nom}$: conformal quantile over the nominal Gaussian quantile | ideal = 1.0 |

$q > 1$ means the standardised residuals have heavier tails than the surrogate's
$\sigma$ implies (intervals undercover); $q < 1$ means they are too wide.

**LinKK ($\ell$).**  The linKK test (Boukamp 1995; Schönleber et al. 2014)
checks whether the raw EIS spectrum satisfies the Kramers-Kronig relations, which
hold for any linear, causal, time-invariant system.  A large RMSE ($\ell > 0.05$)
indicates the measurement contains artefacts (drift, nonlinearity, insufficient
settling), making the downstream ECM fit unreliable regardless of the circuit model.

**ECM KK self-consistency ($\kappa$, kkV).**  The ECM-fitted parameters are used to
simulate an impedance spectrum; the KK transform is then applied to the simulated
spectrum (Luo et al. 2021).  A residual $\kappa > 0.05$ means the ECM's functional
form violates KK, implying the circuit topology does not correctly describe the
physical processes in the cell.

**DRT peak mismatch ($\delta$).**  The Distribution of Relaxation Times (Boukamp 1995;
Wan et al. 2015) converts the EIS spectrum into a continuous relaxation time
spectrum without assuming a circuit model.  Significant peaks in the DRT gamma
distribution correspond to distinct physical processes (electrode reactions,
diffusion layers, grain boundaries).  The number of such peaks should match the
number of independent RC/CPE/ZARC elements in the ECM.  A mismatch $\delta \geq 1$
indicates either over-parameterisation (spurious elements) or under-parameterisation
(missing relaxation processes).

### 2.2  Natural ranges and ideal values

| Symbol | Range | Ideal |
|--------|-------|-------|
| $\alpha$ | $(0, \infty)$ | 1.0 |
| $\mathrm{CE}, \mathrm{ENCE}, \mathrm{MCA}, \phi$ | $[0, 1]$ | 0 |
| $\rho$ | $[-1, 1]$ | 1 |
| $d$ | $\{0, \ldots, n\}$ | 0 |
| $\ell$ | $[0, \infty)$ | 0; threshold 0.05 |
| $c$ | $[0, \infty)$ | 0; threshold 0.50 |
| $\kappa$ | $[0, \infty)$ | 0; threshold 0.05 |
| $\psi$ | $[0, 1]$ | 0 |
| $\delta$ | $\{0, 1, 2, \ldots\}$ | 0 |
| $\lambda^\text{st}$ | $[0, 1]$ | 1 |
| $q$ | $(0, \infty)$ | 1.0 |

A **perfectly healthy** experiment has
$v^* = (1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1)^\top$.

### 2.3  Assembly from pipeline outputs

The vector is assembled by `make_audit_health_vector(results)` after any
combination of audit pipeline calls.  The DRT peak mismatch (index 11)
requires an additional call to `drt_peak_mismatch_result(drt_dict, n_ecm_elements)`,
whose `AuditResult` is then included in the `results` list.

```python
inference_results = audit.run_inference_audit(...)
drt_result = drt_peak_mismatch_result(drt_dict, n_ecm_elements=4)
tsmodel_results  = pipeline.run_tsmodel_audit(...)
eval_results     = pipeline.run_evaluate_audit(...)

v = make_audit_health_vector(
    inference_results + [drt_result] + tsmodel_results + eval_results
)
```

Fields whose check did not report (not run, skipped, or a non-finite value) are
filled with a **neutral value**: the value at which the field adds 0 to the
health score $h_\text{score}$ of §4.  That is $\rho = 0$, $\lambda^\text{st} = 0$,
$q = 1$ (so $|q - 1| = 0$), $\alpha = 1$ (`variance_alignment`), and $0$ for
every other field (`NEUTRAL_AUDIT_HEALTH`).  A check that has not reported then
neither rewards nor penalises a protocol.  Filling the ratio fields with 0
instead would charge the full conformal penalty until the conformal check has
enough samples.  A check that starts reporting can still move its field away
from the neutral value in one step, a jump in the state the dynamics model
sees; this is accepted.

A step with no vector at all (`uncertainty_vector=None`) is filled by the
policy (`_missing_uncertainty`): with the policy's `neutral_uncertainty`, which
the audit policies set to `NEUTRAL_AUDIT_HEALTH`, when it has the right length,
and otherwise with zeros (the neutral value for a vector of z-scores, §4.3).

---

## 3  Audit-augmented belief state

### 3.1  Construction

The augmented belief state replaces $u_t$ (per-parameter variance,
dimension $n$) with $v_t$ (audit health vector, dimension 14):

$$
\tilde{s}_t^\text{audit}
= \begin{bmatrix} s_t \\ w \, v_t \end{bmatrix}
\in \mathbb{R}^{n + 14},
$$

where $w$ is `uncertainty_weight`.  For the standard 72-dimensional ECM
feature vector, the augmented state is $\mathbb{R}^{86}$, compared to
$\mathbb{R}^{144}$ for the per-parameter variant — a 58-dimension reduction.

### 3.2  SVD of the health-augmented state matrix

The SVD step (§3 of `state_space_bayes_derivation.md`) is unchanged:

$$
\mathbf{S}^\top \approx U_r \Sigma_r V_r^\top,
\qquad U_r \in \mathbb{R}^{(n+14) \times r}.
$$

For `AuditStateSpacePolicy` the SVD is of the detrended states (§4.1 of
`state_space_bayes_derivation.md`); for `AuditBayesianDynamicsPolicy` it is of
the raw states.  The first $n$ rows of $U_r$ capture ECM parameter directions; the last 14 rows
capture audit health directions.  The dominant SVD modes therefore encode the
**joint co-variation of battery state and experiment health quality**.

Example: a mode with large loading on index 9 ($\kappa$, ECM KK self-consistency)
and index 0 (say, high ohmic resistance $R_1$) reveals that as $R_1$ rises with
degradation, the ECM circuit model progressively fails to reproduce the spectrum
— a warning that the circuit topology requires revision before the model enters
an out-of-distribution regime.

### 3.3  Linear dynamics in the health-augmented space

The DMDc regression identifies $[A_r | B_r]$ from the augmented projected states:

$$
z_{t+1} = A_r z_t + B_r a_t,
\qquad z_t = d_t U_r \in \mathbb{R}^r,
$$

where for `AuditStateSpacePolicy` $d_t$ is the detrended augmented state
(`RegimeDetrender`, single regime; §4.1 of `state_space_bayes_derivation.md`)
and for `AuditBayesianDynamicsPolicy` $d_t = \tilde{s}_t^\text{audit}$ (raw).
The detrender has two phases:

* **Warm-up** (the first 20 updates): the reference is the running global
  mean *including* the current state,
  $d_t = \tilde{s}_t - \frac{1}{t}\sum_{k \le t} \tilde{s}_k$.
  `reconstruct` inverts this for the next state:
  $\tilde{s} = \frac{n+1}{n} d + S/n$, with $S$ and $n$ the sum and count of
  the states so far.
* **After warm-up**: the reference is the regime-group EMA of the *earlier*
  states, $d_t = \tilde{s}_t - b_{t-1}$, and `reconstruct` gives
  $\tilde{s} = b + d$.

In the steering experiment the warm start replays about 20 observations, so
the first steering steps still fall inside the warm-up phase.

The linear model now captures how EIS validity and ECM model quality co-evolve
with the battery state under each charge protocol.  If certain protocols
systematically increase $\ell$ (linKK RMSE) or $\delta$ (DRT mismatch), $B_r$
encodes this, and the health bonus penalises those protocols.

---

## 4  Health cost term

### 4.1  Definition of the health score

Define the **scalar health score** $h_\text{score}: \mathbb{R}^{14} \to [-6, 2]$:

$$
\boxed{
h_\text{score}(v)
= \operatorname{clip}(\rho, -1, 1)
+ \operatorname{clip}(\lambda^\text{st}, 0, 1)
- \operatorname{clip}(\mathrm{CE}, 0, 1)
- \operatorname{clip}(\phi, 0, 1)
- \operatorname{clip}\!\left(\frac{\ell}{0.05}, 0, 1\right)
- \operatorname{clip}\!\left(\frac{\kappa}{0.05}, 0, 1\right)
- \min\!\left(|q - 1|, 1\right)
}
$$

where $\operatorname{clip}(x, a, b) = \min(\max(x, a), b)$.  Every term is
clipped to its range (the lower bound 0 included on both EIS terms), so a
predicted, linearly extrapolated vector cannot earn a reward outside the
range.  A non-finite entry takes its neutral value (§2.3), which adds 0.  The
default health function requires the 14-field vector and raises an error
otherwise; other layouts pass `health_fn` (§4.2).

Each term and its contribution:

| Term | Range | Direction | Physical interpretation |
|------|-------|-----------|------------------------|
| $\operatorname{clip}(\rho, -1, 1)$, $\rho = v[4]$ | $[-1, 1]$ | higher ↑ | Model knows when it is uncertain |
| $\operatorname{clip}(\lambda^\text{st}, 0, 1)$, $\lambda^\text{st} = v[12]$ | $[0, 1]$ | higher ↑ | Surrogate landscape is locally stable |
| $-\operatorname{clip}(\mathrm{CE}, 0, 1)$, $\mathrm{CE} = v[1]$ | $[-1, 0]$ | less negative ↑ | Calibration quality |
| $-\operatorname{clip}(\phi, 0, 1)$, $\phi = v[6]$ | $[-1, 0]$ | less negative ↑ | Absence of anomalous uncertainties |
| $-\operatorname{clip}(\ell/0.05, 0, 1)$ | $[-1, 0]$ | less negative ↑ | EIS data quality (LinKK) |
| $-\operatorname{clip}(\kappa/0.05, 0, 1)$ | $[-1, 0]$ | less negative ↑ | ECM model quality (kkV) |
| $-\min(|q - 1|, 1)$ | $[-1, 0]$ | less negative ↑ | Conformal coverage at the nominal rate |

Perfect health ($\rho = 1$, $\lambda^\text{st} = 1$, every penalty 0) gives
$h_\text{score} = 2$; the worst experiment ($\rho = -1$, $\mathrm{CE} = 1$,
$\phi = 1$, both EIS terms at 1, $|q - 1| \ge 1$) gives $-6$.  A field at its
neutral value (§2.3) contributes 0.

**Why seven terms, not fourteen?**  The explicit health cost uses the seven most
interpretable and independently meaningful metrics.  The remaining seven fields
($\alpha$, ENCE, MCA, $d$, $c$, $\psi$, $\delta$) still contribute to the
**implicit health cost** through the dynamics model: $A_r$ and $B_r$ capture their
co-variation with the battery state, and the linear prediction
$s^\text{aug}_\text{next}$ encodes their predicted future values.

**Why normalise $\ell$ and $\kappa$ by 0.05?**  The thresholds are the engineering
acceptance criteria in the audit checks (`LinKKCheck.QUALITY_THRESHOLD` and
`"Model Self-Consistency — Simulated Mean"`, both 0.05).  Normalising by the
threshold makes the penalty proportional to how far the metric is from passing,
and caps it at 1 (no additional penalty for extreme failures, avoiding score
instability when data is very bad).  The conformal term $|q - 1|$ is
normalised by 1, the deviation at which the conformal quantile is double, or
zero, the nominal one.

### 4.2  Health cost in the acquisition score

Subtracting $\gamma_\text{health} \cdot h_\text{score}(v^\text{pred}_\text{next})$
from the score rewards protocols that are predicted to maintain or improve
experiment health.  The predicted next health vector is:

$$
v^\text{pred}_\text{next} = \tilde{s}^\text{dyn}_\text{next}[n{:}] / w,
$$

with $\tilde{s}^\text{dyn}_\text{next}$ the one-step prediction of the base
policy.  For `AuditStateSpacePolicy`, whose dynamics are fitted to detrended
states (§3.3; §4.1 of `state_space_bayes_derivation.md`), it is
$\operatorname{reconstruct}\bigl(U_r (A_r z + B_r a)\bigr)$ with
$z = d_t U_r$: during the detrender's warm-up
$\frac{n+1}{n}\,U_r(A_r z + B_r a) + S/n$, afterwards
$b_t + U_r (A_r z + B_r a)$ with $b_t$ the EMA including $\tilde{s}_t$.  For
`AuditBayesianDynamicsPolicy` it is $U_r (A_r z + B_r a)$ with
$z = \tilde{s}_t U_r$.

The health bonus is then:

$$
\text{health\_bonus}(a)
= \gamma_\text{health} \cdot h_\text{score}(v^\text{pred}_\text{next}(a)).
$$

This is **prospective**: the policy evaluates the experiment health it
*expects to observe after* applying protocol $a$, not the current health.

**Other health functions.**  The policies accept any `health_fn` in place of
$h_\text{score}$, for health vectors with a different layout.  It should be
bounded for *every* input, as $h_\text{score}$ is because each of its terms is
clipped (§4.1): $v^\text{pred}_\text{next}$ is a linear extrapolation and can
leave the natural ranges of §2.2, but $h_\text{score}$ of it stays in
$[-6, 2]$.  $\gamma_\text{health}$ is then the largest amount of objective the
policy will trade for health, and an extreme candidate cannot dominate the
score.  A smooth bound keeps the term informative beyond any threshold.  For a
vector of z-scored "badness" metrics, as in the steering experiment (§4.3), the
health is

$$
h(v) = -\frac{1}{m} \sum_{i=1}^{m} \tanh\!\left(\frac{v_i}{k}\right) \in [-1, 1],
$$

with $k$ the z-score at which a metric counts as clearly better or worse than
its warm-start level (default $k = 2$).

### 4.3  Layout in the steering experiment

`experiments/results/scripts/steer_state_space.py` does not use the 14-field
layout or $h_\text{score}$.  It feeds the policy
$v_t = (z_1, \ldots, z_m)$, the z-scored *badness* of the metrics the policy
sees: the 4 default steering metrics (`CalibrationError`, `NLL`,
`VarianceErrorCorrelation`, `SignedBias`), or all 13 for
`audit-state-space-all`.  Each metric is z-scored against its values during
the warm start; a metric whose check did not report (NaN) gets $z = 0$, the
neutral value; every $z$ is clipped to $\pm 10$ by default (`--z-clip`).  The
health function is $h(v) = -\frac{1}{m}\sum_i \tanh(z_i / k)$ with $k = 2$
(`--health-scale`), passed as `health_fn`.

---

## 5  Full acquisition scores

### 5.1  AuditStateSpacePolicy

$$
\boxed{
\text{score}(a)
= Q \, \delta(s^\text{dyn}_\text{next})
+ R \, \|\hat{a}\|^2
- \kappa_\text{LCB} \, \bar{\sigma}_\text{GP}
- C_\text{rate} \, \hat{a}_1
- \gamma_\text{health} \, h_\text{score}(v^\text{pred}_\text{next})
}
$$

where $\hat{a} = a \oslash p$ is the protocol normalised component-wise by
`prot_scale`, $p_j = \max\bigl(\max_{a' \in \mathcal{A}} |a'_j|, 1\bigr)$ over
the candidate pool, and $C_\text{rate}\,\hat{a}_1$ (`C_rate_bonus`, default 0)
rewards a higher normalised first protocol component (first-stage charge
current).  Both are inert in the steering experiment, which sets $R = 0$ and
leaves $C_\text{rate} = 0$.

### 5.2  AuditBayesianDynamicsPolicy

$$
\boxed{
\text{score}(a)
= Q \, \delta(s^\text{dyn}_\text{next})
+ R \, \|\hat{a}\|^2
- \kappa_\text{LCB} \, \bar{\sigma}_\text{GP}
- \gamma_\text{info} \, \log\bigl(1 + h(z, a)\bigr)
- \gamma_\text{sub} \, \rho_\text{sub}(\hat{s}_\text{GP})
- C_\text{rate} \, \hat{a}_1
- \gamma_\text{health} \, h_\text{score}(v^\text{pred}_\text{next})
}
$$

with $\hat{a}$ and $C_\text{rate}$ as in §5.1.  Note that $h(z, a)$ is the
leverage score and $h_\text{score}$ (or `health_fn`) the health score; they
are unrelated.

Symbol table:

| Symbol | Definition |
|--------|-----------|
| $z$ | Belief state projected to the rank-$r$ audit subspace: $z = d_t U_r$ (detrended, §3.3) for `AuditStateSpacePolicy`, $z = \tilde{s}_t U_r$ for `AuditBayesianDynamicsPolicy` |
| $\hat{a} = a \oslash p$ | Protocol normalised by `prot_scale` (§5.1) |
| $C_\text{rate}$ | `C_rate_bonus` (default 0) |
| $s^\text{dyn}_\text{next} = \tilde{s}^\text{dyn}_\text{next}[:n]$ | Predicted next ECM state |
| $v^\text{pred}_\text{next} = \tilde{s}^\text{dyn}_\text{next}[n:] / w$ | Predicted next audit health vector (14-dim) |
| $h(z,a) = [z\|a]^\top G_T^{-1} [z\|a]$ | Leverage score in the health-augmented dynamics space |
| $\rho_\text{sub}(\hat{s}_\text{GP})$ | Normalised subspace residual of GP-predicted next state |
| $h_\text{score}(v) \in [-6, 2]$ | Scalar health score (§4.1) |
| $Q, R, \kappa_\text{LCB}, \gamma_\text{info}, \gamma_\text{sub}, \gamma_\text{health}$ | Hyperparameters |

---

## 6  Dynamics of experiment health

### 6.1  EIS measurement validity ($\ell$, linKK)

LinKK RMSE can increase with degradation because:
- **Drift artefacts**: as internal resistance rises, the cell requires longer
  settling times before EIS; insufficient settling produces a non-stationary
  measurement that violates the time-invariance assumption of KK relations.
- **Nonlinearity**: near end-of-life, large impedance values imply the small-signal
  approximation underpinning EIS breaks down at the applied perturbation amplitude.

Protocols that drive the cell to extreme states-of-charge or high currents between
EIS measurements are more likely to induce drift.  The dynamics model captures
this as a non-zero $B_r$ coefficient linking the protocol to $\Delta \ell$.

### 6.2  ECM model validity ($\kappa$, $\delta$)

ECM KK self-consistency failures ($\kappa > 0.05$) and DRT peak mismatches
($\delta \geq 1$) share a common cause: the ECM circuit topology no longer
describes all physical processes present in the cell.

- **New relaxation processes emerge** as degradation creates new resistive
  interfaces (SEI growth, lithium plating, particle cracking), adding DRT peaks
  not covered by the original ECM.
- **Existing elements merge** when two previously separate processes (e.g., two
  diffusion layers with similar time constants) become indistinguishable,
  reducing the effective DRT peak count.

Both types of mismatch will manifest as $\delta \geq 1$ and often as $\kappa > 0.05$
(a bad ECM fit produces KK-inconsistent simulated spectra).

### 6.3  Parameter stability ($c$, ECM max CV)

A high max CV across the MCMC posterior ($c > 0.5$) indicates that the EIS
data are insufficient to constrain one or more circuit elements.  This can occur:
- early in the experiment (few conditioning cycles, wide prior);
- when the cell is in a regime where one element's signature is masked
  (e.g., a CPE whose relaxation frequency is outside the measurement bandwidth);
- when the ECM is over-parameterised relative to the available data.

High $c$ typically precedes a breakdown in ECM KK self-consistency: poorly
constrained posteriors sample unphysical parameter regions.

### 6.4  Predictive quality drift ($\rho$, CE, $\phi$)

These metrics measure how well the GP model characterises its own uncertainty
relative to the observed prediction error.  They degrade through:

* **Distribution shift**: as the battery enters regimes far from the GP training
  data, predicted variances become systematically under- or over-estimated.
* **Calibration shift**: the frequency of GP confidence intervals containing the
  true observation drops below the nominal level.
* **Anomaly onset**: sudden degradation transitions produce GP residuals with
  elevated variance that are flagged as anomalous.

### 6.5  Surrogate and coverage quality ($\lambda^\text{st}$, $q$)

* **Landscape stability** ($\lambda^\text{st}$) falls when the surrogate develops
  regions where a gradient step on its prediction amplifies rather than damps
  perturbations, typically sharp, poorly constrained features fitted to few
  points.  Querying such regions both tests and smooths them.
* **Conformal coverage** ($q$) drifts above 1 when the surrogate's residuals are
  heavier-tailed than its $\sigma$ (overconfidence, unmodelled noise structure)
  and below 1 when $\sigma$ is inflated.  Unlike the calibration terms it makes
  no Gaussian assumption about the residuals.

---

## 7  Comparison of augmentation strategies

| Property | Per-parameter variance ($u_t \in \mathbb{R}^n$) | Audit health vector ($v_t \in \mathbb{R}^{14}$) |
|----------|-------------------------------|----------------------------|
| Dimension | $n$ (matches state dim) | 14 (fixed, pipeline-agnostic) |
| What it captures | Magnitude of per-parameter epistemic uncertainty | Quality of the measurement, model, and prediction pipeline |
| Physical questions | "How uncertain is each ECM element?" | "Is the data valid? Is the model correct? Is the GP calibrated? Is the surrogate a sound guide?" |
| Scale | Variances (same order as state) | Bounded [0,1] scalars + counts |
| Dynamics | MCMC posterior contraction/expansion | KK validity drift, calibration shift, DRT topology changes |
| Protocol sensitivity | Protocols that reduce MCMC uncertainty | Protocols that maintain data quality and model validity |
| Requires | MCMC per cycle | Full inference + audit pipeline per cycle |
| Augmented state size | $2n$ | $n + 14$ |

The two augmentations are **complementary** rather than competing.  A future
extension could combine both:

$$
\tilde{s}_t^\text{full}
= \begin{bmatrix} s_t \\ w_u \, u_t \\ w_v \, v_t \end{bmatrix}
\in \mathbb{R}^{n + n + 14}.
$$

---

## 8  References

Boukamp, B. A. (1995). A linear Kronig-Kramers transform test for immittance data
validation. *Journal of the Electrochemical Society*, 142(6), 1885–1894.

Schönleber, M., Klotz, D., & Ivers-Tiffée, E. (2014). A method for improving the
robustness of linear Kramers-Kronig validity tests.
*Electrochimica Acta*, 131, 20–27.

Wan, T. H., Saccoccio, M., Chen, C., & Ciucci, F. (2015). Influence of the
discretization methods on the distribution of relaxation times deconvolution:
implementing radial basis functions with DRTtools. *Electrochimica Acta*, 184, 483–499.

Luo, J., et al. (2021). ECM-based Kramers-Kronig consistency validation as a
model self-consistency test.
*Journal of the Electrochemical Society*, 168(3), 030507.

Kuleshov, V., Fenner, N., & Ermon, S. (2018). Accurate uncertainties for deep
learning using calibrated regression. *ICML*, pp. 2796–2804.

Levi, D., Gispan, L., Giladi, N., & Fetaya, E. (2022). Evaluating and calibrating
uncertainty prediction in regression tasks. *Sensors*, 22(15), 5540.

All references from `state_space_bayes_derivation.md` apply unchanged.
