"""
Auditing a Bayesian-optimisation loop
=====================================

*Needs: botorch (``pip install botorch``).*

A standard BoTorch loop: fit a Gaussian process, maximise an acquisition
function, evaluate the objective, repeat. The audit adds one ``on_step``
call per iteration and, with ``check_every``, a report every few iterations,
so you can see *when* the surrogate's uncertainty became trustworthy, not
only whether it was trustworthy at the end.
"""

# %%
# Step 1: the objective and the BoTorch loop pieces
# -------------------------------------------------
# The Forrester function is a common 1-D test problem on [0, 1]. BoTorch
# maximises, so the loop maximises its negative. Measurements carry Gaussian
# noise with standard deviation ``NOISE``.

import sys

try:
    import torch
    from botorch.acquisition import LogExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood
except ImportError:
    sys.exit("This demo needs botorch: pip install botorch")

import matplotlib.pyplot as plt
import numpy as np
from traits_audit import AuditHook, AuditPipeline
from traits_audit.checks import (
    CalibrationErrorCheck,
    IntervalCoverageCheck,
    NegativeLogLikelihoodCheck,
    PITUniformityCheck,
)
from traits_audit.plotting import plot_check_grid

N_INIT = 4        # random points before the first GP fit
N_ITER = 20       # BO iterations
CHECK_EVERY = 5   # audit every 5 iterations
NOISE = 0.3
SEED = 0

torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)
bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)


def forrester(x):
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)


def measure(x):
    return -forrester(x) + rng.normal(0.0, NOISE)


# %%
# Step 2: the audit
# -----------------
# The same kind of pipeline as in demo 01, and a hook with ``check_every``:
# the hook also audits the loop every ``CHECK_EVERY`` iterations while it runs
# and keeps those reports in ``hook.intermediate_reports``.

pipeline = AuditPipeline([
    CalibrationErrorCheck(threshold=0.15),
    IntervalCoverageCheck(expected_coverage=0.683, tolerance=0.15),
    PITUniformityCheck(),
    NegativeLogLikelihoodCheck(),
])
hook = AuditHook(pipeline, check_every=CHECK_EVERY)

# %%
# Step 3: the BO loop
# -------------------
# Before measuring the chosen point, ask the GP for its prediction there.
# ``observation_noise=True`` gives the predictive distribution of a *noisy
# measurement*, which is what ``y_true`` will be compared with.

train_x = torch.rand(N_INIT, 1, dtype=torch.double)
train_y = torch.tensor([[measure(float(x))] for x in train_x], dtype=torch.double)

for _ in range(N_ITER):
    gp = SingleTaskGP(train_x, train_y)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    acq = LogExpectedImprovement(gp, best_f=train_y.max())
    x_next, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=4, raw_samples=64)

    with torch.no_grad():
        post = gp.posterior(x_next, observation_noise=True)
        mu = float(post.mean)
        sigma = float(post.variance.sqrt())
    y_next = measure(float(x_next))

    hook.on_step(y_true=y_next, y_pred_mean=mu, y_pred_std=sigma, uncertainty=sigma)

    train_x = torch.cat([train_x, x_next])
    train_y = torch.cat([train_y, torch.tensor([[y_next]], dtype=torch.double)])

report = hook.on_end()
print(report.summary())
print(f"Best point found: x = {train_x[train_y.argmax()].item():.3f}  (true minimiser: x = 0.757)")

# %%
# Step 4: when did the audit verdicts settle?
# -------------------------------------------
# One column per audit. Early columns judge only a handful of points, so a
# check can flip between pass and fail before it settles. PIT needs at least
# 20 points and is skipped until then.
#
# IntervalCoverage still fails at the end. Far more than the expected 68 % of
# measurements land inside the ±1σ interval, so the GP is *under-confident*:
# its intervals are wider than its errors. For BO that usually means more
# exploration than needed rather than wrong answers, but it is the kind of
# finding the audit exists to surface.

stages = [(f"iter {CHECK_EVERY * (k + 1)}", r) for k, r in enumerate(hook.intermediate_reports)]
stages.append(("final", report))
plot_check_grid(stages, title="BoTorch loop, audited every 5 iterations")
plt.show()
