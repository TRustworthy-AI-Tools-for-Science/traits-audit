"""
Auditing a loop you don't own
=============================

*Needs: scikit-optimize (``pip install scikit-optimize``).*

Often the loop lives inside a library: you call ``optimizer.run(...)`` and
never see the loop body. Most such libraries accept a **callback** that they
call after every iteration. That callback is where the hook goes.

This demo uses scikit-optimize's ``gp_minimize``, but the pattern is the same
for any framework with a per-iteration callback: write a small function that
translates what the framework gives you into one ``hook.on_step`` call.
"""

# %%
# Step 1: the objective and the hook
# ----------------------------------

import sys

try:
    from skopt import gp_minimize
except ImportError:
    sys.exit("This demo needs scikit-optimize: pip install scikit-optimize")

import matplotlib.pyplot as plt
import numpy as np
from traits_audit import AuditHook, AuditPipeline
from traits_audit.checks import CalibrationErrorCheck, IntervalCoverageCheck, NegativeLogLikelihoodCheck
from traits_audit.plotting import plot_check_grid

N_CALLS = 30
N_INITIAL = 6
CHECK_EVERY = 8  # audit every 8 recorded steps
NOISE = 0.1
SEED = 0

rng = np.random.default_rng(SEED)


def objective(params):
    x = params[0]
    return float(np.sin(3 * x) + 0.3 * x**2 + rng.normal(0, NOISE))


hook = AuditHook(
    AuditPipeline([
        CalibrationErrorCheck(threshold=0.15),
        IntervalCoverageCheck(expected_coverage=0.683, tolerance=0.15),
        NegativeLogLikelihoodCheck(),
    ]),
    check_every=CHECK_EVERY,
)

# %%
# Step 2: the callback adapter
# ----------------------------
# ``gp_minimize`` calls ``callback(result)`` after each evaluation. ``result``
# holds every point tried (``x_iters``), every value (``func_vals``) and every
# fitted GP (``models``). The newest GP has already seen the newest point, so
# the honest prediction for that point comes from the GP *before* it,
# ``models[-2]``. Until two GPs exist there is no prediction to audit.
#
# ``predict(..., return_std=True)`` on scikit-optimize's GP returns the std of
# the latent function, so the measurement noise is added back.


def audit_callback(result):
    if len(result.models) < 2:
        return
    gp = result.models[-2]
    x_scaled = result.space.transform([result.x_iters[-1]])
    mu, sd = gp.predict(x_scaled, return_std=True)
    hook.on_step(
        y_true=float(result.func_vals[-1]),
        y_pred_mean=float(mu[0]),
        y_pred_std=float(np.hypot(sd[0], NOISE)),
    )


# %%
# Step 3: run the framework's loop
# --------------------------------
# The loop itself is untouched; only ``callback=`` is new.

result = gp_minimize(
    objective,
    dimensions=[(-2.0, 2.0)],
    n_calls=N_CALLS,
    n_initial_points=N_INITIAL,
    noise=NOISE**2,
    callback=[audit_callback],
    random_state=SEED,
)
report = hook.on_end()

print(f"Best x = {result.x[0]:.3f}, best value = {result.fun:.3f}")
print(f"Audited {len(hook.history)} of {N_CALLS} evaluations")
print(report.summary())

# %%
# Step 4: plot the audit
# ----------------------
# Steps count recorded predictions, so step 8 is the 14th evaluation: the
# first ``N_INITIAL`` random points have no GP prediction to audit.

stages = [(f"step {CHECK_EVERY * (k + 1)}", r) for k, r in enumerate(hook.intermediate_reports)]
stages.append(("final", report))
plot_check_grid(stages, title="gp_minimize, audited through its callback")
plt.show()
