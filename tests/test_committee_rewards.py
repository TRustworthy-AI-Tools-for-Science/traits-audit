"""Reward correctness tests for the committee module.

Each agent's reward must equal sign * (score(history+new) - score(history)),
where ``score`` is the underlying audit check's value (or a target-distance
transform for level-valued checks). Catches the "reward drifts from check"
failure mode.

Skips entirely if the committee subpackage isn't importable (e.g. user hasn't
installed the [committee] extra).
"""
from __future__ import annotations

import numpy as np
import pytest


pytest.importorskip(
    "gymnasium",
    reason="committee extra not installed (gymnasium missing)",
)

from traits_audit.committee.rewards import (
    REWARD_REGISTRY,
    RunningZScore,
    _SignalDeltaReward,
    CRPSReward,
    NLLReward,
    IntervalScoreReward,
    CalibrationErrorReward,
    ConformalCoverageReward,
    PITUniformityReward,
    IntervalCoverageReward,
    VarianceAlignmentReward,
    VarErrCorrelationReward,
    KuleshovCalibrationReward,
    ENCEReward,
    CalibrationError1StdReward,
    UncertaintyEvolutionReward,
    UncertaintyAnomalyReward,
    MahalanobisOODReward,
)


def _history(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate (y_true, mu, sigma) with realistic miscalibration."""
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal(n)
    sigma = 0.5 + np.abs(rng.standard_normal(n))
    # Heteroscedastic noise + slight overconfidence so checks return finite values.
    y_true = mu + sigma * 1.2 * rng.standard_normal(n)
    return y_true.astype(float), mu.astype(float), sigma.astype(float)


_YMUSIGMA_REWARDS = [
    cls for cls in REWARD_REGISTRY.values()
    if not issubclass(cls, _SignalDeltaReward)
]


@pytest.mark.parametrize("reward_cls", _YMUSIGMA_REWARDS)
def test_reward_equals_signed_score_delta(reward_cls):
    """Reward = sign * (score(after) - score(before)) for every agent."""
    reward = reward_cls()
    y_before, mu_before, sigma_before = _history(n=40, seed=7)
    # Add a single new observation.
    y_new, mu_new, sigma_new = _history(n=1, seed=23)
    y_after = np.concatenate([y_before, y_new])
    mu_after = np.concatenate([mu_before, mu_new])
    sigma_after = np.concatenate([sigma_before, sigma_new])

    r = reward.reward(
        y_before, mu_before, sigma_before,
        y_after, mu_after, sigma_after,
    )

    # Compute the expected delta from the underlying check directly.
    res_before = reward.check.run(
        history=[], y_true=y_before, y_pred_mean=mu_before, y_pred_std=sigma_before
    )
    res_after = reward.check.run(
        history=[], y_true=y_after, y_pred_mean=mu_after, y_pred_std=sigma_after
    )
    v_before = reward._score(None if res_before.value is None else float(res_before.value))
    v_after = reward._score(None if res_after.value is None else float(res_after.value))
    expected = reward.sign * (v_after - v_before)

    assert r == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_reward_zero_when_below_floor():
    """If either side returns None (e.g. PIT n<20), reward should be 0.0."""
    reward = PITUniformityReward()
    # 10 samples — below PITUniformity's n>=20 floor; check returns value=None.
    y, mu, sigma = _history(n=10, seed=0)
    y2 = np.append(y, 0.5)
    mu2 = np.append(mu, 0.0)
    sigma2 = np.append(sigma, 1.0)
    assert reward.reward(y, mu, sigma, y2, mu2, sigma2) == 0.0


def test_reward_sign_lower_is_better():
    """Scoring-rule rewards: improving the check (lower value) -> positive reward."""
    # Construct: same data before; after = before with a perfectly-calibrated
    # added point so mean score drops.
    reward = CRPSReward()
    y_before, mu_before, sigma_before = _history(n=40, seed=1)
    # New point: residual exactly 0 (best possible CRPS contribution).
    y_after = np.append(y_before, 0.0)
    mu_after = np.append(mu_before, 0.0)
    sigma_after = np.append(sigma_before, 0.5)
    r = reward.reward(
        y_before, mu_before, sigma_before,
        y_after, mu_after, sigma_after,
    )
    # CRPS with z=0 is 2*phi(0)*sigma - sigma/sqrt(pi) -> smaller than typical;
    # adding it should lower the mean, giving positive reward.
    assert r > 0.0


def test_reward_sign_higher_is_better():
    """VarErrCorr/PIT: improving the check (higher value) -> positive reward."""
    reward = VarErrCorrelationReward()
    # Build a history with anti-aligned sigma/error so rho is low; add a
    # point that strongly increases rho (high sigma + high error).
    n = 30
    rng = np.random.default_rng(42)
    sigma = np.linspace(0.5, 1.5, n)
    errors = sigma[::-1] + 0.01 * rng.standard_normal(n)  # anti-aligned
    mu = np.zeros(n)
    y = mu + np.sign(rng.standard_normal(n)) * errors
    # New point: very high sigma + very high |error| (boosts rho).
    y_after = np.append(y, 5.0)
    mu_after = np.append(mu, 0.0)
    sigma_after = np.append(sigma, 3.0)
    r = reward.reward(y, mu, sigma, y_after, mu_after, sigma_after)
    assert r > 0.0


def test_interval_coverage_target_distance():
    """IntervalCoverage scores |observed - 0.683|; closer is better."""
    reward = IntervalCoverageReward()
    # Before: 100 samples with empirical coverage well above target (underconfident).
    rng = np.random.default_rng(0)
    n = 100
    mu = rng.standard_normal(n)
    sigma = np.full(n, 3.0)
    y = mu + 0.1 * rng.standard_normal(n)  # almost no error, sigma huge -> ~100% coverage
    # After: add a point that misses the sigma band -> brings observed closer to 0.683.
    y_after = np.append(y, mu[0] + 10.0)  # large residual, outside +-sigma
    mu_after = np.append(mu, mu[0])
    sigma_after = np.append(sigma, 3.0)
    r = reward.reward(y, mu, sigma, y_after, mu_after, sigma_after)
    assert r > 0.0


def test_variance_alignment_target_distance():
    """VarianceAlignment scores |ratio - 1|; closer is better."""
    reward = VarianceAlignmentReward()
    rng = np.random.default_rng(1)
    n = 50
    mu = rng.standard_normal(n)
    sigma = np.full(n, 0.1)
    # Errors much larger than sigma -> ratio << 1 (overconfident).
    y = mu + rng.standard_normal(n) * 2.0
    ratio_before = np.mean(sigma**2) / np.mean((y - mu) ** 2)
    assert ratio_before < 1.0
    # Add a point with zero error (drives mean error down, ratio toward 1).
    y_after = np.append(y, mu[0])
    mu_after = np.append(mu, mu[0])
    sigma_after = np.append(sigma, 0.1)
    r = reward.reward(y, mu, sigma, y_after, mu_after, sigma_after)
    # ratio moves toward 1 -> |ratio - 1| decreases -> -|ratio - 1| increases -> r > 0
    assert r > 0.0


def test_running_zscore_passes_through_until_min_samples():
    """RunningZScore returns raw values until min_samples reached."""
    z = RunningZScore(min_samples=10)
    for i in range(9):
        assert z.normalize(float(i)) == float(i)
    # 10th observation triggers normalization.
    normalized = z.normalize(9.0)
    assert normalized != 9.0
    # After many samples around mean 4.5, normalized values should be O(1).
    for _ in range(100):
        z.normalize(4.5)
    assert abs(z.normalize(4.5)) < 1.0


def test_running_zscore_handles_constant_stream():
    """A constant stream has std=0; z-score must not produce inf/NaN."""
    z = RunningZScore(min_samples=5, eps=1e-6)
    for _ in range(50):
        out = z.normalize(7.0)
        assert np.isfinite(out)


def test_reward_registry_has_fifteen_agents():
    """Sanity: the registry matches the v1 plan (9 original + 3 calibration
    variants + 3 different-signal personalities)."""
    assert len(REWARD_REGISTRY) == 15
    expected = {
        # Original 9.
        "CRPS", "NLL", "IntervalScore",
        "CalibrationError", "ConformalCoverage", "PITUniformity",
        "IntervalCoverage", "VarianceAlignment", "VarErrCorrelation",
        # Calibration variants.
        "KuleshovCalibration", "ENCE", "CalibrationError1Std",
        # Different-signal personalities.
        "UncertaintyEvolution", "UncertaintyAnomaly", "MahalanobisOOD",
    }
    assert set(REWARD_REGISTRY) == expected


# -- Signal-delta rewards (Phase 2 additions) -------------------------------

def _sigma_series(n: int, seed: int, slope: float = 0.0) -> np.ndarray:
    """Monotone-ish sigma series; slope<0 makes UncertaintyEvolution flag."""
    rng = np.random.default_rng(seed)
    base = 1.0 + slope * np.arange(n) + 0.05 * rng.standard_normal(n)
    return np.clip(base, 0.01, None)


def test_uncertainty_evolution_reward_flips_sign_when_series_worsens():
    """Extending the sigma series with a strongly-decreasing tail flips the
    check from pass (value=0) to fail (value=1). sign=-1 → reward negative."""
    r = UncertaintyEvolutionReward()
    before = _sigma_series(n=20, seed=0, slope=0.0)          # flat, value≈0
    after = np.concatenate([before, np.linspace(0.5, 0.01, 40)])  # steep drop
    val = r.reward(
        None, None, None, None, None, None,
        sigma_series_before=before, sigma_series_after=after,
    )
    assert val < 0.0


def test_uncertainty_anomaly_reward_flips_sign_on_spike():
    """Appending a single sparse spike raises the within-series z-score
    anomalous fraction → sign=-1 → reward negative. (The check falls back to
    within-series z-scoring when no explicit baseline is provided.)"""
    r = UncertaintyAnomalyReward()
    rng = np.random.default_rng(0)
    before = 1.0 + 0.05 * rng.standard_normal(100)
    after = np.append(before, 20.0)  # one sparse outlier vs a tight cloud
    val = r.reward(
        None, None, None, None, None, None,
        sigma_series_before=before, sigma_series_after=after,
    )
    assert val < 0.0


def test_mahalanobis_ood_reward_returns_zero_below_min_history():
    """Below the check's min_history (default 20), value=None → reward=0.0."""
    r = MahalanobisOODReward()
    x_before = np.linspace(0.1, 0.5, 5)
    x_after = np.append(x_before, 0.9)
    sig_before = np.full(5, 1.0)
    sig_after = np.full(6, 1.0)
    val = r.reward(
        None, None, None, None, None, None,
        x_before=x_before, x_after=x_after,
        sigma_series_before=sig_before, sigma_series_after=sig_after,
    )
    assert val == 0.0


def test_mahalanobis_ood_reward_penalizes_far_query():
    """With enough history, appending an x far from the reference cloud raises
    the trailing-window OOD fraction. sign=-1 → reward negative (or zero if
    both windows are already saturated)."""
    r = MahalanobisOODReward()
    rng = np.random.default_rng(1)
    x_before = np.clip(0.5 + 0.05 * rng.standard_normal(40), 0.0, 1.0)
    x_after = np.append(x_before, 10.0)  # blatantly OOD
    sig_before = np.full(40, 1.0)
    sig_after = np.full(41, 1.0)
    val = r.reward(
        None, None, None, None, None, None,
        x_before=x_before, x_after=x_after,
        sigma_series_before=sig_before, sigma_series_after=sig_after,
    )
    assert val <= 0.0
