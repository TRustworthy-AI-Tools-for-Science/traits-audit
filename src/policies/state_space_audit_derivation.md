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

The audit system answers four distinct questions:

1. **Is the raw EIS data valid?** (LinKK Kramers-Kronig consistency)
2. **Does the ECM model capture the physics?** (ECM KK self-consistency; DRT structure)
3. **Are the fitted parameters stable?** (MCMC posterior max CV)
4. **How well is the predictive uncertainty characterised?** (calibration, correlation, anomalies)

The audit health vector $v_t \in \mathbb{R}^{12}$ encodes all four classes.
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

### 2.1  Full 12-dimensional form

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
  \delta_t
\end{pmatrix}
\in \mathbb{R}^{12}.
$$

The vector is partitioned into two groups:

#### Part A — Predictive quality (indices 0–6)

These metrics assess how well the downstream GP model characterises its own
uncertainty.  They are produced by the TSModel and Evaluate stages of the
audit pipeline.

| Idx | Symbol | Source check | Formula / source | Direction |
|-----|--------|--------------|-----------------|-----------|
| 0 | $\alpha$ | `VarianceAlignmentCheck` | $\bar{\sigma}^2_\text{pred} / \bar{\sigma}^2_\text{true}$ | ideal = 1.0 |
| 1 | $\mathrm{CE}$ | `TraitsCalibrationCheck` | $\int_0^1 |F_\text{pred}(p) - p| \, dp$ (Kuleshov 2018) | lower ↓ |
| 2 | $\mathrm{ENCE}$ | `TraitsCalibrationCheck` | $\frac{1}{B} \sum_b |\hat{\sigma}_b - \sigma_b| / \sigma_b$ (Levi 2022) | lower ↓ |
| 3 | $\mathrm{MCA}$ | `TraitsCalibrationCheck` | Area between calibration curve and diagonal | lower ↓ |
| 4 | $\rho$ | `VarianceErrorCorrelationCheck` | $r_S(\sigma^2_\text{pred}, |y - \hat{y}|)$ (Spearman) | higher ↑ |
| 5 | $d$ | `UncertaintyEvolutionCheck` | Channels with slope $< -1\%$/step | lower ↓ |
| 6 | $\phi$ | `UncertaintyAnomalyCheck` | Fraction $|z\text{-score}| > 3$ | lower ↓ |

#### Part B — EIS inference quality (indices 7–11)

These metrics assess the upstream measurement and parameter-fitting pipeline.
They are produced by the Inference audit stage and the DRT analysis module.

| Idx | Symbol | Source / AuditResult name | Physical question | Direction |
|-----|--------|--------------------------|-------------------|-----------|
| 7 | $\ell$ | `"LinKK Reconstruction Error"` | Is the EIS measurement KK-consistent? | lower ↓ |
| 8 | $c$ | `"ECM Parameter Stability (Max CV)"` | Are the MCMC-fitted parameters well-constrained? | lower ↓ |
| 9 | $\kappa$ | `"Model Self-Consistency — Simulated Mean"` | Does the ECM reproduce the impedance spectrum (kkV)? | lower ↓ |
| 10 | $\psi$ | `"Model Self-Consistency — Posterior Draws"` | Is the ECM posterior KK-consistent? | lower ↓ |
| 11 | $\delta$ | `"DRT Peak Count Mismatch"` | Does the ECM structure match the DRT peak count? | lower ↓ |

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

A **perfectly healthy** experiment has
$v^* = (1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0)^\top$.

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

Fields not present in `results` are filled with 0.0 (neutral, no penalty).

---

## 3  Audit-augmented belief state

### 3.1  Construction

The augmented belief state replaces $u_t$ (per-parameter variance,
dimension $n$) with $v_t$ (audit health vector, dimension 12):

$$
\tilde{s}_t^\text{audit}
= \begin{bmatrix} s_t \\ w \, v_t \end{bmatrix}
\in \mathbb{R}^{n + 12},
$$

where $w$ is `uncertainty_weight`.  For the standard 72-dimensional ECM
feature vector, the augmented state is $\mathbb{R}^{84}$, compared to
$\mathbb{R}^{144}$ for the per-parameter variant — a 60-dimension reduction.

### 3.2  SVD of the health-augmented state matrix

The SVD step (§3 of `state_space_bayes_derivation.md`) is unchanged:

$$
\mathbf{S}^\top \approx U_r \Sigma_r V_r^\top,
\qquad U_r \in \mathbb{R}^{(n+12) \times r}.
$$

The first $n$ rows of $U_r$ capture ECM parameter directions; the last 12 rows
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
\qquad z_t = \tilde{s}_t^\text{audit} U_r \in \mathbb{R}^r.
$$

The linear model now captures how EIS validity and ECM model quality co-evolve
with the battery state under each charge protocol.  If certain protocols
systematically increase $\ell$ (linKK RMSE) or $\delta$ (DRT mismatch), $B_r$
encodes this, and the health bonus penalises those protocols.

---

## 4  Health cost term

### 4.1  Definition of the health score

Define the **scalar health score** $h_\text{score}: \mathbb{R}^{12} \to [-4, 1]$:

$$
\boxed{
h_\text{score}(v)
= \rho
- \mathrm{CE}
- \phi
- \operatorname{clip}\!\left(\frac{\ell}{0.05}, 0, 1\right)
- \operatorname{clip}\!\left(\frac{\kappa}{0.05}, 0, 1\right)
}
$$

where $\operatorname{clip}(x, 0, 1) = \min(\max(x, 0), 1)$.

Each term and its contribution:

| Term | Range | Direction | Physical interpretation |
|------|-------|-----------|------------------------|
| $\rho = v[4]$ | $[-1, 1]$ | higher ↑ | Model knows when it is uncertain |
| $-\mathrm{CE} = -v[1]$ | $[-1, 0]$ | less negative ↑ | Calibration quality |
| $-\phi = -v[6]$ | $[-1, 0]$ | less negative ↑ | Absence of anomalous uncertainties |
| $-\operatorname{clip}(\ell/0.05, 0, 1)$ | $[-1, 0]$ | less negative ↑ | EIS data quality (LinKK) |
| $-\operatorname{clip}(\kappa/0.05, 0, 1)$ | $[-1, 0]$ | less negative ↑ | ECM model quality (kkV) |

**Why five terms, not twelve?**  The explicit health cost uses the five most
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
instability when data is very bad).

**Backward compatibility.**  When $v$ has length 7 (from an older pipeline run),
indices 7–11 are not accessed; the function falls back to the original 3-term
formula $h = \rho - \mathrm{CE} - \phi$.

### 4.2  Health cost in the acquisition score

Subtracting $\gamma_\text{health} \cdot h_\text{score}(v^\text{pred}_\text{next})$
from the score rewards protocols that are predicted to maintain or improve
experiment health.  The predicted next health vector is:

$$
\tilde{s}^\text{dyn}_\text{next} = U_r (A_r z + B_r a),
\qquad
v^\text{pred}_\text{next} = \tilde{s}^\text{dyn}_\text{next}[n{:}] / w.
$$

The health bonus is then:

$$
\text{health\_bonus}(a)
= \gamma_\text{health} \cdot h_\text{score}(v^\text{pred}_\text{next}(a)).
$$

This is **prospective**: the policy evaluates the experiment health it
*expects to observe after* applying protocol $a$, not the current health.

---

## 5  Full acquisition scores

### 5.1  AuditStateSpacePolicy

$$
\boxed{
\text{score}(a)
= Q \, \delta(s^\text{dyn}_\text{next})
+ R \, \|a\|^2
- \kappa_\text{LCB} \, \bar{\sigma}_\text{GP}
- \gamma_\text{health} \, h_\text{score}(v^\text{pred}_\text{next})
}
$$

### 5.2  AuditBayesianDynamicsPolicy

$$
\boxed{
\text{score}(a)
= Q \, \delta(s^\text{dyn}_\text{next})
+ R \, \|a\|^2
- \kappa_\text{LCB} \, \bar{\sigma}_\text{GP}
- \gamma_\text{info} \, \frac{h(z, a)}{1 + h(z, a)}
- \gamma_\text{sub} \, \rho_\text{sub}(\hat{s}_\text{GP})
- \gamma_\text{health} \, h_\text{score}(v^\text{pred}_\text{next})
}
$$

Symbol table:

| Symbol | Definition |
|--------|-----------|
| $z = \tilde{s}^\text{audit} U_r$ | Belief state projected to rank-$r$ audit subspace |
| $s^\text{dyn}_\text{next} = \tilde{s}^\text{dyn}_\text{next}[:n]$ | Predicted next ECM state |
| $v^\text{pred}_\text{next} = \tilde{s}^\text{dyn}_\text{next}[n:] / w$ | Predicted next audit health vector (12-dim) |
| $h(z,a) = [z\|a]^\top G_T^{-1} [z\|a]$ | Leverage score in the health-augmented dynamics space |
| $\rho_\text{sub}(\hat{s}_\text{GP})$ | Normalised subspace residual of GP-predicted next state |
| $h_\text{score}(v) \in [-4, 1]$ | Scalar health score (§4.1) |
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

---

## 7  Comparison of augmentation strategies

| Property | Per-parameter variance ($u_t \in \mathbb{R}^n$) | Audit health vector ($v_t \in \mathbb{R}^{12}$) |
|----------|-------------------------------|----------------------------|
| Dimension | $n$ (matches state dim) | 12 (fixed, pipeline-agnostic) |
| What it captures | Magnitude of per-parameter epistemic uncertainty | Quality of the measurement, model, and prediction pipeline |
| Physical questions | "How uncertain is each ECM element?" | "Is the data valid? Is the model correct? Is the GP calibrated?" |
| Scale | Variances (same order as state) | Bounded [0,1] scalars + counts |
| Dynamics | MCMC posterior contraction/expansion | KK validity drift, calibration shift, DRT topology changes |
| Protocol sensitivity | Protocols that reduce MCMC uncertainty | Protocols that maintain data quality and model validity |
| Requires | MCMC per cycle | Full inference + audit pipeline per cycle |
| Augmented state size | $2n$ | $n + 12$ |

The two augmentations are **complementary** rather than competing.  A future
extension could combine both:

$$
\tilde{s}_t^\text{full}
= \begin{bmatrix} s_t \\ w_u \, u_t \\ w_v \, v_t \end{bmatrix}
\in \mathbb{R}^{n + n + 12}.
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
