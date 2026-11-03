"""
Plotting an audit
=================

*Needs: core dependencies only.*

:mod:`traits_audit.plotting` turns a hook's reports and history into
figures. Every function returns a matplotlib figure, so you choose whether to
show it, save it (``fig.savefig("audit.png", dpi=200)``) or log it.

This demo reuses the loop from demo 01 and adds ``check_every=10``, so the
hook also audits the loop every 10 steps while it runs.
"""

# %%
# Step 1: run the loop with intermediate audits
# ---------------------------------------------
# With ``check_every=10`` the hook runs the pipeline on the history so far
# after steps 10, 20, 30, ... and keeps each report in
# ``hook.intermediate_reports``. The final report still comes from ``on_end``.

import matplotlib.pyplot as plt
import numpy as np
from traits_audit import AuditHook, AuditPipeline
from traits_audit.checks import (
    CalibrationErrorCheck,
    CRPSCheck,
    IntervalCoverageCheck,
    PITUniformityCheck,
    VarianceErrorCorrelationCheck,
)
from traits_audit.plotting import (
    plot_audit_evolution,
    plot_calibration_curve,
    plot_check_grid,
    plot_uncertainty_evolution,
)

N_STEPS = 60
CHECK_EVERY = 10
NOISE = 0.1
SEED = 0

rng = np.random.default_rng(SEED)


def objective(x):
    return np.sin(6.0 * x) + 0.5 * x


def ensemble_predict(x_train, y_train, x_eval, n_members=20, degree=3):
    preds = np.empty((n_members, len(x_eval)))
    for k in range(n_members):
        idx = rng.integers(0, len(x_train), len(x_train))
        deg = min(degree, len(np.unique(x_train[idx])) - 1)
        preds[k] = np.polyval(np.polyfit(x_train[idx], y_train[idx], deg=deg), x_eval)
    return preds.mean(axis=0), preds.std(axis=0) + 1e-6


pipeline = AuditPipeline([
    CalibrationErrorCheck(threshold=0.1),
    IntervalCoverageCheck(expected_coverage=0.683, tolerance=0.15),
    PITUniformityCheck(),
    CRPSCheck(),
    VarianceErrorCorrelationCheck(),
])
hook = AuditHook(pipeline, check_every=CHECK_EVERY)

pool = np.linspace(0.0, 1.0, 200)
x_train = rng.uniform(0.0, 1.0, 5)
y_train = objective(x_train) + rng.normal(0.0, NOISE, len(x_train))
for _ in range(N_STEPS):
    mu, epistemic = ensemble_predict(x_train, y_train, pool)
    sigma = np.hypot(epistemic, NOISE)
    i = int(np.argmax(sigma))
    y_new = float(objective(pool[i]) + rng.normal(0.0, NOISE))
    hook.on_step(y_true=y_new, y_pred_mean=float(mu[i]),
                 y_pred_std=float(sigma[i]), uncertainty=float(sigma.mean()))
    x_train = np.append(x_train, pool[i])
    y_train = np.append(y_train, y_new)

report = hook.on_end()

# %%
# Step 2: one grid for the whole run
# ----------------------------------
# :func:`~traits_audit.plotting.plot_check_grid` takes a list of
# ``(label, report)`` pairs and draws one column per stage. Green cells pass,
# red cells fail, and the shade says how far the value is from its threshold.
# Checks with no threshold, such as CRPS, are outlined rather than filled,
# because they report a value but give no verdict.

stages = [(f"step {CHECK_EVERY * (k + 1)}", r) for k, r in enumerate(hook.intermediate_reports)]
stages.append(("final", report))
fig = plot_check_grid(stages, title="Audit at every 10 steps")

# %%
# Step 3: how the uncertainty evolved
# -----------------------------------
# ``hook.history`` holds one dict per ``on_step`` call, so any recorded key can
# be plotted. Here it is the mean predictive std over the candidate pool. The
# early spikes are the ensemble extrapolating at the edges of the domain.

fig = plot_uncertainty_evolution([h["uncertainty"] for h in hook.history], title="Ensemble")
fig.axes[0].set_yscale("log")

# %%
# Step 4: each check's value over the loop
# ----------------------------------------
# :func:`~traits_audit.plotting.plot_audit_evolution` re-runs the pipeline on
# the first 5, 10, 15, ... steps and plots each check's value. It re-runs the
# pipeline once per snapshot, so give it fast checks only.

fig = plot_audit_evolution(pipeline, hook.history, snapshot_every=5)

# %%
# Step 5: a single check in detail
# --------------------------------
# Some checks store enough in ``result.details`` to draw their own figure.
# :func:`~traits_audit.plotting.plot_calibration_curve` draws the reliability
# diagram from a CalibrationError result.

ce = next(r for r in report.results if r.name == "CalibrationError")
fig = plot_calibration_curve(ce)
plt.show()
