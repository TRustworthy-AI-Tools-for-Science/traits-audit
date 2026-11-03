"""
Writing your own check
======================

*Needs: core dependencies only.*

The built-in checks are generic. You usually know something about your own
experiment that they don't, such as the noise floor of your instrument. This
demo turns that knowledge into a check.

The check: **a model must not claim to predict a measurement more precisely
than the instrument can measure it.** If the predicted standard deviation is
below the known measurement noise, the model is over-confident, whatever the
calibration statistics say.
"""

# %%
# Step 1: subclass AuditCheck
# ---------------------------
# A check needs a ``name``, a ``category`` and a ``run`` method. ``run``
# receives the hook's history (one dict per ``on_step`` call) plus any keyword
# arguments passed to ``on_end``, and returns an
# :class:`~traits_audit.base.AuditResult`.
#
# Two rules keep a pipeline robust:
#
# * **Never raise.** If the data you need is missing, return ``passed=True``
#   with a message that starts with ``"Skipped"``. The plotting functions use
#   that prefix to draw the result as skipped rather than passed.
# * **Report a value and a threshold**, so that plots can show how far the
#   result is from the limit, not only pass or fail.

import numpy as np
from traits_audit import AuditHook, AuditPipeline
from traits_audit.base import AuditCategory, AuditCheck, AuditResult
from traits_audit.checks import CalibrationErrorCheck


class NoiseFloorCheck(AuditCheck):
    """Fails if too many predicted stds fall below the instrument's noise."""

    def __init__(self, noise_std, max_fraction=0.1):
        self.noise_std = noise_std
        self.max_fraction = max_fraction

    @property
    def name(self):
        return "NoiseFloor"

    @property
    def category(self):
        return AuditCategory.ALEATORIC_IRREDUCIBLE

    def run(self, history, **kwargs):
        stds = [h["y_pred_std"] for h in history if "y_pred_std" in h]
        if not stds:
            return AuditResult(name=self.name, passed=True, category=self.category,
                               message="Skipped — no y_pred_std recorded.")

        below = float(np.mean(np.asarray(stds) < self.noise_std))
        return AuditResult(
            name=self.name,
            passed=below <= self.max_fraction,
            category=self.category,
            value=below,
            threshold=self.max_fraction,
            message=f"{below:.0%} of predicted stds are below the noise floor {self.noise_std}",
            details={"n_steps": len(stds), "min_pred_std": float(np.min(stds))},
        )


# %%
# Step 2: use it like any built-in check
# --------------------------------------
# Here the "loop" is simulated: the true measurement noise is ``NOISE`` and the
# model's predicted std shrinks as it sees more data. ``STD_SCALE`` below 1
# makes the model over-confident.

NOISE = 0.1
N_STEPS = 50
SEED = 0


def run_loop(std_scale):
    rng = np.random.default_rng(SEED)
    hook = AuditHook(AuditPipeline([
        CalibrationErrorCheck(threshold=0.15),
        NoiseFloorCheck(noise_std=NOISE),
    ]))
    for step in range(N_STEPS):
        model_error = 0.5 / np.sqrt(step + 1)        # shrinks as data accumulates
        honest_std = np.hypot(model_error, NOISE)    # model error plus measurement noise
        y_pred = rng.normal()
        y_true = y_pred + rng.normal(0, honest_std)
        hook.on_step(y_true=y_true, y_pred_mean=y_pred, y_pred_std=honest_std * std_scale)
    return hook.on_end()


print(run_loop(std_scale=1.0).summary())
print()
print(run_loop(std_scale=0.6).summary())

# %%
# With ``std_scale=0.6`` the model ends up predicting stds of about 0.06, below
# the instrument's 0.1, and NoiseFloor fails. It says *why* the model is
# over-confident, in the experiment's own terms.

# %%
# Step 3: test the check on its own
# ---------------------------------
# A check is an ordinary object, so you can unit-test it without a loop. Call
# ``run`` with a hand-made history, including one that lacks the data.

check = NoiseFloorCheck(noise_std=0.1)
print(check.run([{"y_pred_std": 0.05}, {"y_pred_std": 0.2}]))
print(check.run([{"y_true": 1.0}]).message)
