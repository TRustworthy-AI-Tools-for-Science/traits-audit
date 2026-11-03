"""
Is the loop itself stable?
==========================

*Needs: core dependencies only.*

Calibration asks whether each prediction's uncertainty is honest. Stability
asks a different question about the *loop as a whole*: if something nudged
the campaign, would the disturbance die out or grow? traits-audit answers it
from two angles that should always be reported together:

* **Local, LyapunovStability.** At each point the loop visited, would gradient
  descent on the surrogate settle (a basin) or run away (a ridge or saddle)?
  The check reports the fraction of visited points that are locally stable.
* **Global, DMDcSpectralRadius.** Fit a linear model ``z[t+1] = A z[t] + B a[t]``
  to the whole trajectory of states ``z`` and actions ``a`` (dynamic mode
  decomposition with control). If the spectral radius ρ(A) is below 1,
  disturbances shrink from step to step.
"""

# %%
# Step 1: a 2-D optimisation loop
# -------------------------------
# A scikit-learn Gaussian process and a lower-confidence-bound policy minimise
# a smooth 2-D function. The first ``N_INIT`` points are random.

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from traits_audit import AuditHook, AuditPipeline
from traits_audit.checks import CalibrationErrorCheck, DMDcSpectralRadiusCheck, LyapunovStabilityCheck
from traits_audit.trajectory import TrajectoryRecord, analyze_trajectory

N_INIT = 6
N_STEPS = 40
NOISE = 0.02
SEED = 0

rng = np.random.default_rng(SEED)
probes = np.array([[0.2, 0.2], [0.2, 0.8], [0.8, 0.2], [0.8, 0.8]])  # fixed points to track uncertainty


def objective(x):
    x = np.atleast_2d(x)
    return (x[:, 0] - 0.3) ** 2 + (x[:, 1] - 0.7) ** 2 + 0.1 * np.sin(8 * x[:, 0])


def fit_gp(X, y):
    # The measurement noise is known, so pass it as ``alpha`` instead of fitting it.
    gp = GaussianProcessRegressor(ConstantKernel(1.0) * RBF(0.3), alpha=NOISE**2, random_state=SEED)
    return gp.fit(X, y)


# %%
# Step 2: record two things per step
# ----------------------------------
# The hook records the usual prediction and measurement. A
# :class:`~traits_audit.trajectory.TrajectoryRecord` also records the loop's
# *dynamics*: the state after the step (the new measurement and the best value
# so far), the uncertainty (the GP's std at four fixed probe points), and the
# action (the point the policy chose).

hook = AuditHook(AuditPipeline([
    CalibrationErrorCheck(threshold=0.15),
    LyapunovStabilityCheck(min_stable_fraction=0.5),
    DMDcSpectralRadiusCheck(stability_threshold=1.0),
]))
record = TrajectoryRecord(domain="toy-2d", policy="LCB")

X = rng.uniform(0, 1, (N_INIT, 2))
y = objective(X) + rng.normal(0, NOISE, N_INIT)
for step in range(N_STEPS):
    gp = fit_gp(X, y)
    candidates = rng.uniform(0, 1, (400, 2))
    mu, sd = gp.predict(candidates, return_std=True)
    i = int(np.argmin(mu - 2.0 * sd))  # lower confidence bound
    x_new = candidates[i]
    y_new = float(objective(x_new)[0] + rng.normal(0, NOISE))

    # ``sd`` is the std of the latent function; add the noise to get the std of a measurement.
    hook.on_step(y_true=y_new, y_pred_mean=float(mu[i]), y_pred_std=float(np.hypot(sd[i], NOISE)))
    record.add(
        state=[y_new, min(y_new, y.min())],
        uncertainty=gp.predict(probes, return_std=True)[1],
        action=x_new,
        is_warmup=step < 5,  # let the GP settle before fitting the dynamics
    )
    X, y = np.vstack([X, x_new]), np.append(y, y_new)

# %%
# Step 3: fit the dynamics and run the audit
# ------------------------------------------
# :func:`~traits_audit.trajectory.analyze_trajectory` fits the DMDc model and
# a block-bootstrap confidence interval for ρ(A). The Lyapunov check needs the
# final surrogate (``surrogate_fn``) and the points the loop visited
# (``op_states``); it computes the local Jacobians itself. Both arrive through
# ``on_end``, because they only exist once the loop has finished.

dynamics = analyze_trajectory(record, n_boot=100, seed=SEED)
final_gp = fit_gp(X, y)

report = hook.on_end(
    surrogate_fn=lambda x: final_gp.predict(x[None])[0],
    surrogate_fn_batched=final_gp.predict,
    op_states=X[N_INIT:],
    rho_A=dynamics.rho_A_joint,
    rho_A_ci=dynamics.rho_joint_ci,
)
print(report.summary())
print("Long enough for a trustworthy DMDc fit:", dynamics.min_length_ok)

# %%
# Step 4: what the numbers mean
# -----------------------------
# * A **stable fraction** near 1 means the loop spent its time in basins of the
#   surrogate, where a small step in any direction leads back. Most visited
#   points are near the minimum here, so most are stable.
# * **ρ(A) < 1** means a disturbance to the measurements or to the
#   uncertainties fades over the following steps instead of compounding. Look
#   at the bootstrap interval too: here it straddles 1, so with only about 35
#   fitted steps the loop is *probably* stable, not certainly. A longer
#   campaign narrows the interval.
#
# Neither number alone separates "locally rough" from "globally divergent",
# which is why the pipeline warns (in ``report.metadata["pairing_warnings"]``)
# if you run one check without the other.

fig, (ax_path, ax_unc) = plt.subplots(1, 2, figsize=(9, 3.8))
g = np.linspace(0, 1, 60)
gx, gy = np.meshgrid(g, g)
mean = final_gp.predict(np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
ax_path.contourf(gx, gy, mean, levels=20, cmap="Greys")
sc = ax_path.scatter(X[N_INIT:, 0], X[N_INIT:, 1], c=np.arange(N_STEPS), cmap="viridis", s=18)
fig.colorbar(sc, ax=ax_path, label="step")
ax_path.set(title="Visited points on the final surrogate", xlabel="x0", ylabel="x1")

ax_unc.plot(record.uncertainties)
ax_unc.legend([f"probe ({a}, {b})" for a, b in probes], fontsize=8, frameon=False)
ax_unc.set(title="GP std at the four probe points", xlabel="step", ylabel="std", yscale="log")
fig.tight_layout()
plt.show()
