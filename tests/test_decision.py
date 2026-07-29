import numpy as np
import pytest

from traits_audit.checks.decision import DecisionFlipRateCheck


def _argmax_decision(y):
    return int(np.argmax(y))


def test_dfr_low_flip_rate_for_well_separated_points():
    mu = np.array([0.0, 0.0, 10.0])  # point 2 is a clear winner
    sigma = np.array([0.5, 0.5, 0.5])
    result = DecisionFlipRateCheck(seed=0, max_flip_rate=0.05).run(
        [], decision_fn=_argmax_decision, y_pred_mean=mu, y_pred_std=sigma
    )
    assert result.passed
    assert result.value < 0.05


def test_dfr_high_flip_rate_for_near_tie():
    mu = np.array([5.0, 5.01, 0.0])  # points 0 and 1 are a near-tie
    sigma = np.array([1.0, 1.0, 1.0])
    result = DecisionFlipRateCheck(seed=0, max_flip_rate=0.05).run(
        [], decision_fn=_argmax_decision, y_pred_mean=mu, y_pred_std=sigma
    )
    assert not result.passed
    assert result.value > 0.05


def test_dfr_skips_without_decision_fn():
    result = DecisionFlipRateCheck().run([], y_pred_mean=np.array([1.0, 2.0]))
    assert result.passed
    assert "Skipped" in result.message


def test_dfr_report_only_by_default():
    mu = np.array([5.0, 5.01])
    sigma = np.array([1.0, 1.0])
    result = DecisionFlipRateCheck(seed=0).run([], decision_fn=_argmax_decision, y_pred_mean=mu, y_pred_std=sigma)
    assert result.passed  # max_flip_rate=None -> always pass


def test_dfr_accepts_precomputed_samples():
    mu = np.array([0.0, 10.0])
    samples = np.tile(mu, (50, 1))  # zero-variance samples -> never flips
    result = DecisionFlipRateCheck(max_flip_rate=0.0).run(
        [], decision_fn=_argmax_decision, y_pred_mean=mu, y_pred_samples=samples
    )
    assert result.passed
    assert result.value == 0.0
