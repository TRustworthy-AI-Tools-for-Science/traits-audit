"""``ta-demo``: a minimal, dependency-light tour of the audit pipeline.

A bootstrap ensemble of cubic polynomials learns a noisy 1-D function by
uncertainty-sampling active learning.  The loop pushes each step's
prediction to :class:`~traits_audit.hook.AuditHook`; the pipeline audits the
history at the end and prints a report.  Pass ``--std-scale`` below 1 to make
the surrogate over-confident and watch the calibration checks fail.

Needs only the core dependencies (numpy, scipy, scikit-learn, matplotlib).
The full paper case studies live in the separate ``results`` experiment
package.
"""
from __future__ import annotations

import argparse

import numpy as np

from .checks import (
    CalibrationErrorCheck,
    CRPSCheck,
    IntervalCoverageCheck,
    NegativeLogLikelihoodCheck,
    VarianceErrorCorrelationCheck,
)
from .hook import AuditHook
from .pipeline import AuditPipeline


def _target(x: np.ndarray) -> np.ndarray:
    return np.sin(6.0 * x) + 0.5 * x


def _ensemble_predict(x_train, y_train, x_eval, rng, n_members=20, degree=3):
    """Mean and std of a bootstrap ensemble of polynomial fits."""
    preds = np.empty((n_members, len(x_eval)))
    for k in range(n_members):
        idx = rng.integers(0, len(x_train), len(x_train))
        deg = min(degree, len(np.unique(x_train[idx])) - 1)  # resamples can repeat points
        coef = np.polyfit(x_train[idx], y_train[idx], deg=deg)
        preds[k] = np.polyval(coef, x_eval)
    return preds.mean(axis=0), preds.std(axis=0) + 1e-6


def run(steps: int = 60, seed: int = 0, std_scale: float = 1.0, noise: float = 0.1, verbose: bool = False):
    """Run the demo loop and return the final :class:`~traits_audit.base.AuditReport`."""
    rng = np.random.default_rng(seed)
    pool = np.linspace(0.0, 1.0, 200)
    x_train = rng.uniform(0.0, 1.0, 5)
    y_train = _target(x_train) + rng.normal(0.0, noise, len(x_train))

    pipeline = AuditPipeline(
        [
            CalibrationErrorCheck(threshold=0.1),
            IntervalCoverageCheck(expected_coverage=0.683, tolerance=0.15),
            NegativeLogLikelihoodCheck(),
            CRPSCheck(),
            VarianceErrorCorrelationCheck(),
        ],
        verbose=verbose,
    )
    hook = AuditHook(pipeline)

    for _ in range(steps):
        mu, epistemic = _ensemble_predict(x_train, y_train, pool, rng)
        # Predictive std = epistemic (ensemble spread) + known aleatoric noise,
        # then deliberately mis-scaled by ``std_scale``.
        sigma = np.hypot(epistemic, noise) * std_scale
        i = int(np.argmax(sigma))  # uncertainty sampling
        x_new = pool[i]
        y_new = float(_target(x_new) + rng.normal(0.0, noise))

        hook.on_step(
            y_true=y_new,
            y_pred_mean=float(mu[i]),
            y_pred_std=float(sigma[i]),
            uncertainty=float(sigma.mean()),
        )
        x_train = np.append(x_train, x_new)
        y_train = np.append(y_train, y_new)

    return hook.on_end()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ta-demo", description=__doc__.splitlines()[0])
    p.add_argument("--steps", type=int, default=60, help="active-learning steps (default 60)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--std-scale", type=float, default=1.0,
                   help="multiply predicted std (<1 over-confident, >1 under-confident)")
    args = p.parse_args(argv)
    report = run(steps=args.steps, seed=args.seed, std_scale=args.std_scale)
    print(report.summary())


if __name__ == "__main__":
    main()
