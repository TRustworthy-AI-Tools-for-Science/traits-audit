"""
Detecting a miscalibrated surrogate
===================================

*Needs: core dependencies only.*

A surrogate can rank candidates well and still report uncertainties that are
too small (over-confident) or too large (under-confident). This demo runs the
loop from demo 01 three times, with the predicted standard deviation
multiplied by 0.3, 1.0 and 3.0, and shows which checks catch each mistake.
"""

# %%
# Step 1: the loop from demo 01, as a function
# --------------------------------------------
# ``std_scale`` is the only new argument. It mis-scales the uncertainty the
# surrogate reports, while the measurements themselves are unchanged.

import matplotlib.pyplot as plt
import numpy as np
from traits_audit import AuditHook, AuditPipeline
from traits_audit.checks import (
    CalibrationErrorCheck,
    IntervalCoverageCheck,
    NegativeLogLikelihoodCheck,
    VarianceAlignmentCheck,
    VarianceErrorCorrelationCheck,
)
from traits_audit.plotting import plot_calibration_curve

N_STEPS = 60
NOISE = 0.1
SEED = 0
STD_SCALES = {"over-confident (x0.3)": 0.3, "calibrated (x1)": 1.0, "under-confident (x3)": 3.0}


def objective(x):
    return np.sin(6.0 * x) + 0.5 * x


def ensemble_predict(x_train, y_train, x_eval, rng, n_members=20, degree=3):
    preds = np.empty((n_members, len(x_eval)))
    for k in range(n_members):
        idx = rng.integers(0, len(x_train), len(x_train))
        deg = min(degree, len(np.unique(x_train[idx])) - 1)
        preds[k] = np.polyval(np.polyfit(x_train[idx], y_train[idx], deg=deg), x_eval)
    return preds.mean(axis=0), preds.std(axis=0) + 1e-6


def run_loop(std_scale):
    rng = np.random.default_rng(SEED)
    pipeline = AuditPipeline([
        CalibrationErrorCheck(threshold=0.1),
        IntervalCoverageCheck(expected_coverage=0.683, tolerance=0.15),
        VarianceAlignmentCheck(),
        NegativeLogLikelihoodCheck(),
        VarianceErrorCorrelationCheck(),
    ])
    hook = AuditHook(pipeline)

    pool = np.linspace(0.0, 1.0, 200)
    x_train = rng.uniform(0.0, 1.0, 5)
    y_train = objective(x_train) + rng.normal(0.0, NOISE, len(x_train))
    for _ in range(N_STEPS):
        mu, epistemic = ensemble_predict(x_train, y_train, pool, rng)
        sigma = np.hypot(epistemic, NOISE) * std_scale   # the deliberate mistake
        i = int(np.argmax(sigma))
        y_new = float(objective(pool[i]) + rng.normal(0.0, NOISE))
        hook.on_step(y_true=y_new, y_pred_mean=float(mu[i]),
                     y_pred_std=float(sigma[i]), uncertainty=float(sigma.mean()))
        x_train = np.append(x_train, pool[i])
        y_train = np.append(y_train, y_new)
    return hook.on_end()


# %%
# Step 2: run the three surrogates
# --------------------------------
# Because the seed is fixed, the three runs measure exactly the same points.
# Only the reported uncertainty differs.

reports = {label: run_loop(scale) for label, scale in STD_SCALES.items()}

# %%
# Step 3: compare the verdicts
# ----------------------------
# * **CalibrationError** and **IntervalCoverage** fail in both directions:
#   too few observations fall inside an over-confident interval, too many
#   inside an under-confident one.
# * **VarianceAlignment** divides the *mean* predicted variance by the *mean*
#   squared error. It fails even for the calibrated run, because three of the
#   first ten steps sit at the edge of the domain, where the polynomial
#   ensemble spreads to σ ≈ 10–90. Those few points dominate the mean (drop the
#   first ten steps and the ratio is 1.1). CalibrationError and
#   IntervalCoverage count how many errors fall inside each interval, so a few
#   huge intervals barely move them. Read a check's definition before you trust
#   its verdict.
# * **NegativeLogLikelihood** has no threshold. It only reports a value, but
#   that value jumps for the over-confident model.
# * **VarianceErrorCorrelation** passes in all three runs. Scaling every
#   uncertainty by the same factor does not change which points are most
#   uncertain, so a ranking check cannot see this mistake.

names = [r.name for r in next(iter(reports.values())).results]
print(f"{'check':<26}" + "".join(f"{label:>24}" for label in reports))
for k, name in enumerate(names):
    cells = []
    for report in reports.values():
        r = report.results[k]
        verdict = "report" if r.threshold is None else ("PASS" if r.passed else "FAIL")
        cells.append(f"{verdict} {r.value:8.3f}")
    print(f"{name:<26}" + "".join(f"{c:>24}" for c in cells))

# %%
# Step 4: look at the calibration curves
# --------------------------------------
# :func:`~traits_audit.plotting.plot_calibration_curve` draws the observed
# fraction of measurements inside each central interval against the nominal
# confidence level. A calibrated model follows the diagonal; an over-confident
# one falls below it and an under-confident one rises above it.

fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
for ax, (label, report) in zip(axes, reports.items(), strict=True):
    ce = next(r for r in report.results if r.name == "CalibrationError")
    plot_calibration_curve(ce, ax=ax)
    ax.set_title(label, fontsize=9)
    if ax is not axes[1]:
        ax.get_legend().remove()  # one legend, on the middle panel, is enough
fig.tight_layout()
plt.show()
