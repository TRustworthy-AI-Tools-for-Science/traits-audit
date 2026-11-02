# Derivation: StateSpacePolicy and BayesianDynamicsPolicy

This document gives the full mathematical derivation of both state-space
active learning policies.  §§1–4 establish the shared foundations (belief
state, SVD, linear dynamics identification).  §§5–7 derive `StateSpacePolicy`
(LQR-style one-step cost, DMDc identification, GP exploration).  §§8–13
derive `BayesianDynamicsPolicy`, which adds three information-theoretic
exploration bonuses grounded in Eldredge & Mousavi (2026), Cai et al. (2014),
and Li et al. (2021).  §14 compares the two policies; §15 lists references.

---

# Part I — Shared Foundations

## 1  Problem setup

At each cycle $t$ the battery is in ECM-parameter state
$s_t \in \mathbb{R}^n$.  The agent selects a charge protocol (action)
$a_t \in \mathbb{R}^m$ from a finite candidate set $\mathcal{A}$.
The oracle (physical experiment or PyBAM simulator) returns the next state
$s_{t+1}$.  The goal is to choose the sequence of protocols that
**minimises cumulative degradation** while accumulating enough information
to identify accurate dynamics.

The agent also has a GP surrogate $f : (s, a) \mapsto (\mu, \sigma)$ trained
offline on historical data; it provides predictions and uncertainty estimates
without requiring a new experiment.  The online history grows as
$(s_1, a_1), \ldots, (s_T, a_T)$.

---

## 2  Belief-state augmentation

When the uncertainty audit provides a per-parameter variance vector
$u_t \in \mathbb{R}^n$ (from, e.g., `propagate_fde()` or
`check_parameter_stability()`), the ECM state and variance are concatenated
into an **augmented belief state**:

$$
\tilde{s}_t = \begin{bmatrix} s_t \\ w \, u_t \end{bmatrix} \in \mathbb{R}^{n_\text{aug}},
\qquad n_\text{aug} = n + n_u,
$$

where $w$ is the `uncertainty_weight` hyperparameter.  This augmentation
implements the **belief-state MDP** of Kaelbling et al. (1998): rather than
conditioning the policy on a point estimate of the hidden state, it
conditions on the agent's full posterior — both the estimated parameter
values and their per-parameter uncertainty.  The policy therefore learns how
uncertainty itself evolves under each protocol, enabling it to prefer
protocols that reduce parameter uncertainty as well as degradation.

When no uncertainty vector is available $u_t = 0$ and $\tilde{s}_t = s_t$.

---

## 3  Dimensionality reduction via SVD

Stack the $T$ observed augmented states as rows of
$\mathbf{S} \in \mathbb{R}^{T \times n_\text{aug}}$.  The ECM state lives on a
low-dimensional degradation manifold embedded in $\mathbb{R}^{n_\text{aug}}$;
the SVD finds the best rank-$r$ linear approximation to that manifold.

Compute the rank-$r$ truncated SVD of $\mathbf{S}^\top$:

$$
\mathbf{S}^\top \approx U_r \Sigma_r V_r^\top,
\qquad U_r \in \mathbb{R}^{n_\text{aug} \times r},
\quad \Sigma_r \in \mathbb{R}^{r \times r}.
$$

The columns of $U_r$ are the $r$ **dominant principal axes** of the observed
degradation trajectory.  Project each augmented state to the reduced space:

$$
z_t = \tilde{s}_t^\top U_r \in \mathbb{R}^r.
$$

The singular values $\sigma_1 \geq \ldots \geq \sigma_r$ quantify how much
variance each axis explains; the truncation rank $r$ is chosen so that the
discarded singular values are small relative to the retained ones.

This step is conceptually identical to Dynamic Mode Decomposition
(Schmid 2010; Brunton & Kutz 2022, Ch. 7), which also uses an SVD of the
state matrix as the first step in identifying a reduced-order linear model
from data.

---

## 4  Linear dynamics identification (DMDc)

Assume a **linear time-invariant** (LTI) transition model in the reduced
coordinates:

$$
z_{t+1} = A_r \, z_t + B_r \, a_t + \varepsilon_t,
\qquad \varepsilon_t \sim \mathcal{N}(0, \sigma^2 I_r).
$$

This is the DMD with Control (DMDc) model of Proctor et al. (2016), adapted
to the reduced-order basis $U_r$.  Define the stacked regression matrices:

$$
X_\text{fit} = \bigl[Z_{0:T-2} \;\big|\; A_{0:T-2}\bigr]
\in \mathbb{R}^{(T-1) \times (r+m)},
\qquad
Y_\text{fit} = Z_{1:T-1} \in \mathbb{R}^{(T-1) \times r},
$$

where rows of $Z = \mathbf{S} U_r$ are the projected states and rows of
$A$ are the applied actions.  The joint weight matrix $W = [A_r \;|\; B_r]^\top$
is estimated by ordinary least squares:

$$
\hat{W} = (X_\text{fit}^\top X_\text{fit})^{-1} X_\text{fit}^\top Y_\text{fit}.
$$

Given $\hat{W}$, the **one-step linear prediction** of the next ECM state
for candidate protocol $a$ is:

$$
s^\text{dyn}_\text{next}(a)
= \bigl(U_r (A_r z + B_r a)\bigr)_{[:n]},
$$

taking the first $n$ components of the reconstructed augmented state (the
ECM parameters; the uncertainty dimensions, if present, are discarded for
degradation scoring).

---

# Part II — StateSpacePolicy

## 5  Motivation: LQR control in a learned reduced-order model

The classical **discrete-time Linear Quadratic Regulator** (LQR) minimises
the infinite-horizon quadratic cost

$$
J = \sum_{t=0}^{\infty}
    \bigl[ z_t^\top \mathbf{Q}_z \, z_t + a_t^\top \mathbf{R}_a \, a_t \bigr]
$$

subject to the LTI dynamics $z_{t+1} = A_r z_t + B_r a_t$.  The optimal
control law is $a_t^* = -K_\text{LQR} z_t$, where the gain matrix
$K_\text{LQR}$ is obtained by solving the discrete-time algebraic Riccati
equation (DARE):

$$
P = \mathbf{Q}_z + A_r^\top P A_r
  - A_r^\top P B_r \bigl(\mathbf{R}_a + B_r^\top P B_r\bigr)^{-1}
    B_r^\top P A_r.
$$

`StateSpacePolicy` does **not** solve the DARE, for three reasons:

1. The degradation proxy $\delta : \mathbb{R}^n \to \mathbb{R}$ is
   generally nonlinear (e.g., a sum of resistance-mean features), so the
   cost is not of the quadratic form $z^\top \mathbf{Q}_z z$.
2. $A_r$ and $B_r$ are re-estimated from data after every observation; the
   model is adaptive rather than fixed.
3. The action space $\mathcal{A}$ is discrete and typically small, so
   exhaustive candidate scoring is computationally cheap without needing a
   closed-form gradient.

Instead, `StateSpacePolicy` uses a **one-step greedy approximation** to the
LQR objective (§6) augmented with a GP exploration term (§7).

---

## 6  One-step greedy cost and its LQR interpretation

For each candidate $a \in \mathcal{A}$, score the predicted outcome of
applying $a$ from the current state using:

$$
\text{cost}(a) =
  \underbrace{Q \, \delta\!\bigl(s^\text{dyn}_\text{next}(a)\bigr)}_{\text{state cost}}
+ \underbrace{R \, \|a\|^2}_{\text{control cost}}.
$$

**State cost.**  The scalar $Q \delta(s_\text{next})$ approximates the LQR
term $z_\text{next}^\top \mathbf{Q}_z z_\text{next}$ by replacing the
quadratic with the degradation proxy $\delta$.  This is valid when $\delta$
is a proxy for long-term battery health: if the next state has low
degradation, so do future states (degradation is monotonically accumulating
under mild conditions).  The weight $Q$ scales the relative importance of
degradation vs. control effort.

**Control cost.**  The term $R \|a\|^2$ is the LQR control-effort penalty
$a^\top \mathbf{R}_a a$ with a scalar weight $R$, regularising against
extreme charge rates that could cause thermal runaway or accelerated
side-reactions, independent of the dynamics model.

**One-step optimality.**  The one-step greedy minimiser is:

$$
a^* = \arg\min_{a \in \mathcal{A}}
  \bigl[ Q \, \delta(s^\text{dyn}_\text{next}(a)) + R \, \|a\|^2 \bigr].
$$

For slowly evolving degradation dynamics (as observed in practice for
lithium-ion cells under normal operating conditions), the one-step prediction
$s^\text{dyn}_\text{next}$ is a reliable proxy for multi-step outcomes, and
myopic optimisation of this proxy incurs little regret.

**Connections to DMD/DMDc (Schmid 2010; Brunton & Kutz 2022).**  The
eigenvalues $\lambda_i$ of $A_r$ are the DMD eigenvalues of the reduced
system.  Modes with $|\lambda_i| < 1$ correspond to stable (decaying)
directions of the degradation trajectory; modes with $|\lambda_i| > 1$
correspond to growing directions.  The policy implicitly avoids protocols
that excite growing modes because $s^\text{dyn}_\text{next}$ will be large in
those directions, raising the degradation cost.

---

## 7  GP exploration bonus and the LCB connection

The one-step greedy cost is a pure exploitation policy: it ignores model
uncertainty and can converge to a local minimum of the learned dynamics.  To
encourage exploration, a **GP uncertainty bonus** is subtracted from the
cost:

$$
\text{score}(a) =
  Q \, \delta\!\bigl(s^\text{dyn}_\text{next}(a)\bigr)
+ R \, \|a\|^2
- \kappa \, \bar{\sigma}_\text{GP}(s_t, a),
$$

where $\bar{\sigma}_\text{GP}(s, a) = \frac{1}{n} \sum_{j=1}^n \sigma_j(s, a)$
is the mean GP posterior standard deviation over the $n$ output dimensions.

This term is the **Lower Confidence Bound (LCB)** exploration bonus
of Srinivas et al. (2010): subtracting $\kappa \sigma$ from the score
prefers protocols in regions where the surrogate is uncertain, preventing
the policy from ignoring unexplored parts of the protocol space.

The connection to **PILCO** (Deisenroth & Rasmussen 2011) is that both
frameworks use a learned model of the dynamics and incorporate model
uncertainty into the policy.  In PILCO the uncertainty propagates through
the full trajectory via moment matching; in `StateSpacePolicy` the
uncertainty enters only through the GP bonus on the current step, making
it computationally cheaper at the cost of not propagating uncertainty over
multiple steps ahead.

The resulting full score is (lower = preferred):

$$
\boxed{
\text{score}_\text{SSP}(a) =
\underbrace{Q \, \delta(s^\text{dyn}_\text{next})}_{\text{state cost (LQR)}}
+ \underbrace{R \, \|a\|^2}_{\text{control cost (LQR)}}
- \underbrace{\kappa \, \bar{\sigma}_\text{GP}(s_t, a)}_{\text{LCB exploration}}
}
$$

| Symbol | Definition |
|--------|-----------|
| $z = \tilde{s}_t U_r$ | Current belief state in rank-$r$ subspace |
| $s^\text{dyn}_\text{next} = (U_r(A_r z + B_r a))_{[:n]}$ | DMDc one-step state prediction |
| $\delta(\cdot)$ | Degradation proxy |
| $\bar{\sigma}_\text{GP}$ | Mean GP posterior std (output-dimension average) |
| $Q, R, \kappa$ | Hyperparameters |

**Algorithm (StateSpacePolicy):**

```
history = []

For each cycle t = 1, 2, ...:
  1. [Warm-up] If t < min_fit_obs: select via LCB fallback; go to step 3.
  2. [Fit] SVD of S^T → U_r; OLS of [Z[:-1] | A[:-1]] → A_r, B_r.
  3. [Score] For each candidate a_i ∈ A:
       z           = s̃_t @ U_r
       s_dyn_next  = (U_r @ (A_r @ z + B_r @ a_i))[:n]
       _, σ_gp     = GP(s_t, a_i)
       score(a_i)  = Q δ(s_dyn_next) + R ‖a_i‖² − κ mean(σ_gp)
     Select a* = argmin score(a_i)
  4. [Execute] Apply a*; observe s_{t+1}.
  5. Append (s_t, a*) to history; go to 2.
```

**Complexity (StateSpacePolicy):**

| Operation | Cost | When |
|-----------|------|------|
| SVD refit | $O(T \, n_\text{aug}^2)$ | Every `observe()` |
| OLS solve | $O(T \, (r+m)^2)$ | Every `observe()` |
| Score a single candidate | $O(r + m + n)$ | Every candidate, every step |

The dominant per-step cost is the SVD refit on all $T$ historical states.
For slowly growing $T$ and moderate $n_\text{aug}$, this is practical;
`BayesianDynamicsPolicy` eliminates this cost with online updates (§12).

---

# Part III — BayesianDynamicsPolicy

## 8  Motivation: limitations of the StateSpacePolicy score

`StateSpacePolicy` has two structural gaps:

1. **No model-improvement incentive.**  The LCB bonus rewards
   protocols where the GP surrogate is uncertain, but not protocols that
   maximally improve the *dynamics model* ($A_r$, $B_r$).  A protocol that
   is well-covered by the GP surrogate but poorly covered by the regression
   design $X_\text{fit}$ will not be selected, even though observing it would
   most improve the identified system.

2. **Subspace coverage is implicit.**  The SVD basis $U_r$ is refit from
   all historical data, but the score does not explicitly reward protocols
   that lead to states outside the current SVD manifold.  Unexplored regions
   of ECM-parameter space are avoided rather than probed.

`BayesianDynamicsPolicy` addresses both gaps by adding:

* An **information-gain bonus** that quantifies how much the new
  observation $(z, a) \to z'$ would improve the dynamics model —
  derived from the Bayesian regression posterior (§9–11).
* A **subspace residual bonus** that rewards protocols whose GP-predicted
  next state lies outside the current SVD subspace (§12).

---

## 9  Bayesian linear regression posterior and the Gram matrix

Place a zero-mean Gaussian prior on the weight matrix $W$:

$$
\operatorname{vec}(W) \sim \mathcal{N}(0, \tau^2 I).
$$

With Gaussian noise $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$, the
posterior over $W$ given all $T-1$ transitions is Gaussian with precision

$$
\boxed{G_T = \lambda I + X_\text{fit}^\top X_\text{fit}},
\qquad \lambda = \sigma^2 / \tau^2,
$$

and posterior covariance $G_T^{-1}$.  The posterior **predictive variance**
for a new feature vector $x = [z \;|\; a]^\top$ is

$$
\operatorname{Var}\!\bigl(\hat{y} \mid x, G_T\bigr)
= x^\top G_T^{-1} x
\equiv h(x).
$$

$h(x)$ is the **leverage score** of $x$: it measures how far $x$ is from
the centroid of the existing design in the metric defined by $G_T^{-1}$.
A large leverage score means the candidate $(z, a)$ is poorly covered by
the current regression design — observing it will most reduce the posterior
uncertainty about $W$.

---

## 10  Sequential Bayesian update: Sherman–Morrison (Eldredge & Mousavi 2026)

When a new transition with feature vector $x_\text{new} = [z \;|\; a]^\top$
is observed, the Gram matrix updates by rank-1 addition:

$$
G_{T+1} = G_T + x_\text{new} x_\text{new}^\top.
$$

Recomputing $G_{T+1}^{-1}$ from scratch costs $O((r+m)^3)$.  The
**Sherman–Morrison formula** gives the exact inverse update at cost
$O((r+m)^2)$:

$$
\boxed{
G_{T+1}^{-1} = G_T^{-1}
  - \frac{G_T^{-1} \, x_\text{new} x_\text{new}^\top \, G_T^{-1}}
         {1 + x_\text{new}^\top G_T^{-1} x_\text{new}}
}
$$

**Connection to the Kalman filter (Eldredge & Mousavi 2026, Eq. 33).**
Interpret $W$ as the hidden "state" being estimated, $x_\text{new}^\top$ as
the observation operator $H$, and the new observation $y = z'$ as the
"measurement".  Then:

* **Kalman gain:** $K = G_T^{-1} x_\text{new} / (1 + h)$
* **Mean update:** $\hat{W}_{T+1} = \hat{W}_T + K(y - x_\text{new}^\top \hat{W}_T)$
* **Covariance update:** $G_{T+1}^{-1} = (I - K x_\text{new}^\top) G_T^{-1}$

— which is exactly the Sherman–Morrison formula above.  The SVD of $H =
x_\text{new}^\top$ has one non-zero singular value along the direction
$x_\text{new} / \|x_\text{new}\|$; all other directions are in the null
space of $H$ and receive no update.  This is the observability structure
analysed by Eldredge & Mousavi (2026, §II.B, Eq. 14): only the component of
$W$ aligned with the current observation direction is updated; the
orthogonal complement remains at its prior.

**Initialisation.**  At warm-up (after $T_0$ = `min_fit_obs` observations),
$G_{T_0}^{-1}$ is initialised by direct inversion:

$$
G_{T_0} = \lambda I + X_\text{fit}^\top X_\text{fit},
\qquad G_{T_0}^{-1} = G_{T_0}^{-1}.
$$

Subsequent observations update $G^{-1}$ via Sherman–Morrison only, so the
$O((r+m)^3)$ inversion is paid only once (or once per reanchor).

---

## 11  Information-gain bonus: leverage score as D-optimal criterion

### 11.1  Scalar predictive variance reduction

Before observing $x_\text{new}$, the predictive variance is $h$.
After the update (using the formula from §10):

$$
x_\text{new}^\top G_{T+1}^{-1} x_\text{new}
= h - \frac{h^2}{1 + h}
= \frac{h}{1 + h}.
$$

The variance drops from $h$ to $h/(1+h)$.  The **fractional reduction** is:

$$
\frac{h - h/(1+h)}{h} = \frac{h}{1+h}.
$$

### 11.2  D-optimal (log-determinant) criterion

By the matrix determinant lemma,

$$
\log \det G_{T+1} - \log \det G_T = \log(1 + h).
$$

Maximising $h$ is therefore equivalent to maximising the log-determinant
gain $\log(1+h)$ — the standard **D-optimal experimental design** objective
(Fedorov 1972), which maximises the information content of the new
observation about the parameter vector $W$.

### 11.3  Implementation

Both criteria are monotone in $h$, so in the score we use the normalised
form $h/(1+h) \in [0,1)$, which is bounded and avoids numerical issues
early in training when $G^{-1}$ has large eigenvalues:

$$
\text{info\_bonus}(z, a)
= \gamma_\text{info} \cdot \frac{h(z,a)}{1 + h(z,a)},
\qquad
h(z, a) = [z \;|\; a]^\top G_T^{-1} [z \;|\; a].
$$

---

## 12  Connection to Maximum Model Change (Cai et al. 2014)

Cai et al. (2014) define the **Maximum Model Change (MMC)** criterion for
active learning: select the unlabelled candidate $x_q$ that, once labelled
and added to the training set, produces the **largest change in the learned
model**.

For the linear model $f(x) = w^\top x$ with current OLS estimate $\hat{w}$,
the low-rank update formula gives the exact model change:

$$
\Delta w = \hat{w}_{T+1} - \hat{w}_T
= G_T^{-1} x_q \cdot
  \frac{y_q - x_q^\top \hat{w}_T}{1 + x_q^\top G_T^{-1} x_q}.
$$

The squared magnitude is

$$
\|\Delta w\|^2
= \frac{(y_q - x_q^\top \hat{w}_T)^2}{(1 + h_q)^2}
  \cdot x_q^\top G_T^{-2} x_q.
$$

Since $y_q$ is unknown at query time, treat the residual as fixed at
$\sigma^2$ (the noise level).  The selection criterion becomes:

$$
x_q^* = \arg\max_{x_q \in \mathcal{A}} \frac{x_q^\top G_T^{-2} x_q}{(1+h_q)^2}.
$$

**Relation to the leverage score.**  By the Cauchy–Schwarz inequality applied
to $G_T^{-1}$:

$$
x_q^\top G_T^{-2} x_q
= \|G_T^{-1} x_q\|^2
\leq \lambda_{\max}(G_T^{-1}) \cdot \underbrace{x_q^\top G_T^{-1} x_q}_{= h_q}.
$$

Equality holds in the top eigendirection of $G_T^{-1}$.  In the **margin
region** — high-uncertainty candidates, where the top eigendirection of
$G_T^{-1}$ dominates — maximising $h_q$ ranks candidates identically to
maximising the true MMC.

**RKHS interpretation.**  Cai et al. (2014, Eq. 10) express the MMC for
kernel SVMs as the squared norm in the feature space:
$\|\phi(x_q)\|_\Phi^2$.  In the reproducing kernel Hilbert space view of
Bayesian linear regression, the feature map is the whitened input
$\phi(x) = G_T^{-1/2} x$, so
$\|\phi(x_q)\|_\Phi^2 = x_q^\top G_T^{-1} x_q = h_q$.  **The leverage score
$h$ therefore implements the Cai MMC criterion exactly** when the feature
map is the Bayesian posterior whitening transform.

The normalised form $h/(1+h)$ is monotone in $h$ (same ranking), bounded in
$[0,1)$, and matches the scalar variance-reduction interpretation of §11.1.

---

## 13  Subspace reconstruction residual (Li et al. 2021)

Li et al. (2021) propose **ALSL** (Active Learning via Subspace Learning):
select samples whose representation under the current low-rank subspace is
poorest, because they carry information that expands the basis' coverage of
the data manifold.

**ALSL Formulation II (Li et al. 2021, Eq. 4).** Given data
$X \in \mathbb{R}^{d \times n}$ and a reconstruction coefficient matrix $Q$,
the joint objective is:

$$
\min_{Q, Z} \|X - XQ\|_F^2 + \lambda \|Q\|_* + \mu \|Q - QZ\|_F^2 + \eta \|Z\|_{2,1}.
$$

The nuclear norm $\|Q\|_*$ promotes low rank; the $\ell_{2,1}$ norm
$\|Z\|_{2,1}$ promotes row-sparsity in the selection matrix $Z$, so only
$k$ representative samples are selected.  The optimal $Q$ at convergence
spans the principal subspace of $X$ (i.e., its columns align with $U_r$).
The per-sample contribution to the reconstruction residual is
$\|x_i - U_r U_r^\top x_i\|_F^2$; samples with large residuals are
under-represented in the current subspace and should be queried.

**Sequential adaptation.**  In our online setting, the actual next ECM state
$s_{t+1}$ is unknown at query time.  We substitute the GP posterior mean
$\hat{s}_\text{GP}(a)$ (the surrogate's prediction of $s_{t+1}$) and
normalise by the state magnitude to obtain a scale-invariant residual.  Let
$U_s = U_r[:n, :] \in \mathbb{R}^{n \times r}$ be the ECM-state rows of the
SVD basis (i.e., the rows corresponding to $s$, not $u$):

$$
\boxed{
\rho(a) = \frac
  {\|\hat{s}_\text{GP}(a) - U_s U_s^\top \hat{s}_\text{GP}(a)\|_2}
  {\|\hat{s}_\text{GP}(a)\|_2 + \varepsilon},
\qquad \varepsilon = 10^{-12}
}
$$

$\rho(a) \in [0, 1]$, with $\rho = 0$ if $\hat{s}_\text{GP}$ lies exactly
in the current subspace and $\rho \to 1$ if it is orthogonal to it.

The projection $P_r = U_s U_s^\top$ is the orthogonal projector onto the
column space of $U_s$.  The residual $\hat{s} - P_r \hat{s}$ is the
component of the predicted state that the current basis cannot represent.
Selecting protocols with large $\rho$ expands the manifold coverage in the
ALSL sense: the resulting observations will improve the SVD basis at the
next reanchor.

$$
\text{sub\_bonus}(a) = \gamma_\text{sub} \cdot \rho(a).
$$

---

## 14  Composite score and algorithm summary (BayesianDynamicsPolicy)

Combining the LQR-style one-step cost of `StateSpacePolicy` with the two
information-theoretic exploration bonuses gives the full score:

$$
\boxed{
\text{score}_\text{BDP}(a) =
\underbrace{Q \, \delta(s^\text{dyn}_\text{next})}_{\text{state cost (LQR)}}
+ \underbrace{R \, \|a\|^2}_{\text{control cost (LQR)}}
- \underbrace{\kappa \, \bar{\sigma}_\text{GP}}_{\text{LCB exploration}}
- \underbrace{\gamma_\text{info} \, \dfrac{h(z, a)}{1 + h(z, a)}}_{\text{info gain (Cai / Eldredge)}}
- \underbrace{\gamma_\text{sub} \, \rho\!\bigl(\hat{s}_\text{GP}(a)\bigr)}_{\text{subspace residual (Li)}}
}
$$

| Symbol | Definition |
|--------|-----------|
| $z = \tilde{s}_t U_r$ | Belief state projected to rank-$r$ subspace |
| $s^\text{dyn}_\text{next} = (U_r(A_r z + B_r a))_{[:n]}$ | One-step linear dynamics prediction |
| $\delta(\cdot)$ | Degradation proxy (e.g.\ sum of resistance means) |
| $\bar{\sigma}_\text{GP}$ | Mean GP posterior std over output dimensions |
| $h(z, a) = [z\|a]^\top G_T^{-1} [z\|a]$ | Leverage score |
| $\rho(\hat{s}_\text{GP})$ | Normalised subspace reconstruction residual |
| $Q, R, \kappa, \gamma_\text{info}, \gamma_\text{sub}, \lambda$ | Hyperparameters |

**Algorithm (BayesianDynamicsPolicy):**

```
Initialise:  G_inv = (1/λ) I ∈ ℝ^{(r+m)×(r+m)},  history = []

For each cycle t = 1, 2, ...:
  1. [Warm-up] If t < min_fit_obs: select via LCB fallback; go to step 3.
  2. [Initial fit, once] SVD of S^T → U_r; OLS → A_r, B_r;
                         G = λI + X_fit^T X_fit; G_inv = inv(G).
  3. [Score] For each candidate a_i ∈ A:
       z          = s̃_t @ U_r
       s_dyn_next = (U_r @ (A_r @ z + B_r @ a_i))[:n]
       s_gp, σ_gp = GP(s_t, a_i)
       x          = [z | a_i];  h = x^T G_inv x
       ρ          = ‖s_gp − U_s U_s^T s_gp‖ / (‖s_gp‖ + ε)
       score(a_i) = Q δ(s_dyn_next) + R ‖a_i‖²
                  − κ mean(σ_gp) − γ_info h/(1+h) − γ_sub ρ
     Select a* = argmin score(a_i)
  4. [Execute] Apply a*; observe s_{t+1}.
  5. [Sherman–Morrison update, O((r+m)²)]:
       x = [z | a*];  g = G_inv @ x
       G_inv ← G_inv − outer(g, g) / (1 + x @ g)
  6. [Reanchor, optional] Every reanchor_freq steps:
       Refit SVD → U_r; refit OLS → A_r, B_r;
       Reinit G = λI + X_fit^T X_fit; G_inv = inv(G).
```

---

# Part IV — Policy comparison

| Property | StateSpacePolicy | BayesianDynamicsPolicy |
|----------|-----------------|----------------------|
| SVD refit | Every `observe()` — O($T n_\text{aug}^2$) | Once at warm-up (+ optional reanchor) |
| OLS refit | Every `observe()` — O($T(r+m)^2$) | Once at warm-up (+ optional reanchor) |
| Precision update | Not tracked | Sherman–Morrison — O($(r+m)^2$) per step |
| Exploration | GP uncertainty only | GP + information gain + subspace residual |
| Protocol incentive | Protocols with low degradation *and* high GP σ | Protocols with low degradation, high GP σ, high Gram-leverage, *and* poor subspace coverage |
| Suitable when | Enough compute for full SVD refit each step; exploration via GP is sufficient | Long experiments where online tracking is preferable; explicit dynamics-model improvement needed |

Both policies fall back to `LCBPolicy` during warm-up
(fewer than `min_fit_obs` observations).

---

# Part V — References

Anderson, B. D. O., & Moore, J. B. (1990). *Optimal Control: Linear
Quadratic Methods*. Prentice-Hall.

Brunton, S. L., & Kutz, J. N. (2022). *Data-Driven Science and Engineering:
Machine Learning, Dynamical Systems, and Control* (2nd ed.), Ch. 7–8.
Cambridge University Press. https://doi.org/10.1017/9781009089517

Cai, W., Zhang, Y., Zhou, S., Wang, W., Ding, C., & Gu, X. (2014). Active
learning for support vector machines with maximum model change. In *Joint
European Conference on Machine Learning and Knowledge Discovery in
Databases*, pp. 116–131. https://doi.org/10.1007/978-3-662-44845-8_10

Deisenroth, M. P., & Rasmussen, C. E. (2011). PILCO: A model-based and
data-efficient approach to policy search. In *Proceedings of the 28th
International Conference on Machine Learning (ICML)*, pp. 465–472.
https://proceedings.mlr.press/v15/deisenroth11a.html

Eldredge, J. D., & Mousavi, H. (2026). A practical guide to estimation and
uncertainty quantification of aerodynamic flows. *arXiv preprint*
arXiv:2502.20280. https://doi.org/10.48550/arXiv.2502.20280

Fedorov, V. V. (1972). *Theory of Optimal Experiments*. Academic Press.

Kaelbling, L. P., Littman, M. L., & Cassandra, A. R. (1998). Planning and
acting in partially observable stochastic domains. *Artificial Intelligence*,
101(1–2), 99–134. https://doi.org/10.1016/S0004-3702(98)00023-X

Li, C., Mao, K., Liang, L., Ren, D., Zhang, W., Yuan, Y., & Wang, G.
(2021). Unsupervised active learning via subspace learning. In *Proceedings
of the 35th AAAI Conference on Artificial Intelligence*, pp. 8332–8340.
https://ojs.aaai.org/index.php/AAAI/article/view/17011

Proctor, J. L., Brunton, S. L., & Kutz, J. N. (2016). Dynamic mode
decomposition with control. *SIAM Journal on Applied Dynamical Systems*,
15(1), 142–161. https://doi.org/10.1137/15M1013857

Schmid, P. J. (2010). Dynamic mode decomposition of numerical and
experimental data. *Journal of Fluid Mechanics*, 656, 5–28.
https://doi.org/10.1017/S0022112010001217

Srinivas, N., Krause, A., Kakade, S. M., & Seeger, M. W. (2010). Gaussian
process optimization in the bandit setting: No regret and experimental
design. In *Proceedings of the 27th International Conference on Machine
Learning (ICML)*, pp. 1015–1022. https://arxiv.org/abs/0912.3995
