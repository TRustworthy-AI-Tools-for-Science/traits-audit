"""
A minimal audit loop
====================

*Needs: core dependencies only.*

The smallest complete example: an active-learning loop you already have, plus
four lines of traits-audit. The loop fits a bootstrap ensemble of cubic
polynomials to a noisy 1-D function and measures the point where the ensemble
is least certain. traits-audit only watches. It never changes which point the
loop measures.

This is the same example as the ``ta-demo`` command.
"""

# %%
# Step 1: the experiment you already have
# ---------------------------------------
# A noisy objective and a surrogate that returns a mean and a standard
# deviation for every candidate point. Nothing here knows about traits-audit.

import numpy as np

N_STEPS = 60  # active-learning steps
NOISE = 0.1   # standard deviation of the measurement noise
SEED = 0

rng = np.random.default_rng(SEED)


def objective(x):
    return np.sin(6.0 * x) + 0.5 * x


def ensemble_predict(x_train, y_train, x_eval, n_members=20, degree=3):
    """Mean and standard deviation of a bootstrap ensemble of polynomial fits."""
    preds = np.empty((n_members, len(x_eval)))
    for k in range(n_members):
        idx = rng.integers(0, len(x_train), len(x_train))
        deg = min(degree, len(np.unique(x_train[idx])) - 1)  # resamples can repeat points
        coef = np.polyfit(x_train[idx], y_train[idx], deg=deg)
        preds[k] = np.polyval(coef, x_eval)
    return preds.mean(axis=0), preds.std(axis=0) + 1e-6


# %%
# Step 2: choose the checks
# -------------------------
# An :class:`~traits_audit.pipeline.AuditPipeline` is a list of checks. Each
# check reads what it needs from the recorded steps and returns a verdict.
# Checks without a threshold (here NLL and CRPS) only report a value.

from traits_audit import AuditHook, AuditPipeline
from traits_audit.checks import (
    CalibrationErrorCheck,
    CRPSCheck,
    IntervalCoverageCheck,
    NegativeLogLikelihoodCheck,
    VarianceErrorCorrelationCheck,
)

pipeline = AuditPipeline([
    CalibrationErrorCheck(threshold=0.1),
    IntervalCoverageCheck(expected_coverage=0.683, tolerance=0.15),
    NegativeLogLikelihoodCheck(),
    CRPSCheck(),
    VarianceErrorCorrelationCheck(),
])

# %%
# Step 3: attach a hook
# ---------------------
# The :class:`~traits_audit.hook.AuditHook` is a passive recorder. Your loop
# pushes values to it; it never drives the loop.

hook = AuditHook(pipeline)

# %%
# Step 4: run the loop and record each step
# -----------------------------------------
# The only change to the loop is the ``hook.on_step(...)`` call. Record the
# prediction for the point you measured *before* you see its result, then the
# result itself. The key names are what the checks look for: ``y_true``,
# ``y_pred_mean``, ``y_pred_std`` and ``uncertainty``.

pool = np.linspace(0.0, 1.0, 200)
x_train = rng.uniform(0.0, 1.0, 5)
y_train = objective(x_train) + rng.normal(0.0, NOISE, len(x_train))

for _ in range(N_STEPS):
    mu, epistemic = ensemble_predict(x_train, y_train, pool)
    sigma = np.hypot(epistemic, NOISE)   # ensemble spread plus measurement noise
    i = int(np.argmax(sigma))            # uncertainty sampling
    y_new = float(objective(pool[i]) + rng.normal(0.0, NOISE))

    hook.on_step(
        y_true=y_new,
        y_pred_mean=float(mu[i]),
        y_pred_std=float(sigma[i]),
        uncertainty=float(sigma.mean()),
    )
    x_train = np.append(x_train, pool[i])
    y_train = np.append(y_train, y_new)

# %%
# Step 5: run the audit
# ---------------------
# :meth:`~traits_audit.hook.AuditHook.on_end` runs every check on the recorded
# history and returns an :class:`~traits_audit.base.AuditReport`. A failing
# check never raises, so the audit cannot crash your experiment.

report = hook.on_end()
print(report.summary())

# %%
# Each result is a plain dataclass, so you can act on it in code, for example
# to stop a campaign whose uncertainty can no longer be trusted.

for result in report.results:
    print(f"{result.name:<24} passed={result.passed!s:<5} value={result.value}")

failed = [r.name for r in report.results if not r.passed]
print("Failed checks:", failed or "none")
