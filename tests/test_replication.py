import numpy as np
import pytest

from traits_audit.checks.replication import (
    DarkUncertaintyGapCheck,
    ReplicationShrinkageExponentCheck,
    SignedBiasCheck,
)

# ── SignedBiasCheck ──────────────────────────────────────────────────────────

def test_signed_bias_zero_mean_passes():
    rng = np.random.default_rng(0)
    y_true = rng.standard_normal(300)
    mu = y_true + rng.standard_normal(300) * 0.01  # ~zero-mean residual
    result = SignedBiasCheck(threshold=0.05).run([], y_true=y_true, y_pred_mean=mu)
    assert result.passed
    assert abs(result.value) < 0.05


def test_signed_bias_offset_fails_with_threshold():
    rng = np.random.default_rng(0)
    y_true = rng.standard_normal(300)
    mu = y_true + 0.5  # constant offset
    result = SignedBiasCheck(threshold=0.05).run([], y_true=y_true, y_pred_mean=mu)
    assert not result.passed
    assert result.value == pytest.approx(0.5, abs=0.05)


def test_signed_bias_report_only_by_default():
    y_true = np.array([1.0, 2.0, 3.0])
    mu = np.array([10.0, 20.0, 30.0])
    result = SignedBiasCheck().run([], y_true=y_true, y_pred_mean=mu)
    assert result.passed  # threshold=None -> always pass


def test_signed_bias_skips_when_no_data():
    result = SignedBiasCheck().run([])
    assert result.passed
    assert "Skipped" in result.message


def test_signed_bias_se_multiplier_passes_within_noise():
    # Small, genuinely zero-mean offset relative to a large SE at n=10 ->
    # well within a 2xSE band.
    rng = np.random.default_rng(1)
    y_true = rng.standard_normal(10) * 5.0
    mu = y_true + rng.standard_normal(10) * 5.0  # noisy, unbiased residual
    result = SignedBiasCheck(se_multiplier=2.0).run([], y_true=y_true, y_pred_mean=mu)
    assert result.threshold == pytest.approx(2.0 * result.details["bias_std_error"])
    assert result.passed == (abs(result.value) <= result.threshold)


def test_signed_bias_se_multiplier_fails_for_large_persistent_offset():
    rng = np.random.default_rng(0)
    y_true = rng.standard_normal(300)
    mu = y_true + 0.5  # constant offset, tiny SE at n=300 -> easily distinguishable from 0
    result = SignedBiasCheck(se_multiplier=2.0).run([], y_true=y_true, y_pred_mean=mu)
    assert not result.passed
    assert result.threshold == pytest.approx(2.0 * result.details["bias_std_error"])
    assert result.threshold < 0.5


def test_signed_bias_se_multiplier_skipped_for_single_point():
    result = SignedBiasCheck(se_multiplier=2.0).run(
        [], y_true=np.array([1.0]), y_pred_mean=np.array([1.5])
    )
    assert result.passed
    assert "Skipped" in result.message


def test_signed_bias_rel_std_frac_passes_within_fraction_of_sigma():
    y_true = np.zeros(50)
    mu = np.full(50, 0.05)  # bias = 0.05
    sigma = np.full(50, 1.0)  # mean(y_pred_std) = 1.0
    result = SignedBiasCheck(rel_std_frac=0.1).run(
        [], y_true=y_true, y_pred_mean=mu, y_pred_std=sigma
    )
    assert result.threshold == pytest.approx(0.1)
    assert result.passed


def test_signed_bias_rel_std_frac_fails_outside_fraction_of_sigma():
    y_true = np.zeros(50)
    mu = np.full(50, 0.5)  # bias = 0.5
    sigma = np.full(50, 1.0)
    result = SignedBiasCheck(rel_std_frac=0.1).run(
        [], y_true=y_true, y_pred_mean=mu, y_pred_std=sigma
    )
    assert result.threshold == pytest.approx(0.1)
    assert not result.passed


def test_signed_bias_rel_std_frac_skipped_without_y_pred_std():
    result = SignedBiasCheck(rel_std_frac=0.1).run(
        [], y_true=np.zeros(10), y_pred_mean=np.full(10, 0.05)
    )
    assert result.passed
    assert "Skipped" in result.message


def test_signed_bias_absolute_threshold_takes_priority():
    # threshold set alongside se_multiplier/rel_std_frac -> absolute wins.
    y_true = np.zeros(50)
    mu = np.full(50, 0.5)
    sigma = np.full(50, 1.0)
    result = SignedBiasCheck(threshold=1.0, se_multiplier=2.0, rel_std_frac=0.1).run(
        [], y_true=y_true, y_pred_mean=mu, y_pred_std=sigma
    )
    assert result.threshold == pytest.approx(1.0)
    assert result.details["threshold_kind"] == "absolute"
    assert result.passed


# ── ReplicationShrinkageExponentCheck ───────────────────────────────────────
#
# u_obs(r) is a dispersion ACROSS groups (see the check's docstring for why:
# a within-group statistic is blind to any component held constant within a
# group). All fixtures below therefore vary what is held constant PER GROUP
# — this is what "the replication scheme" means for this check.

def _random_groups(n_groups=40, R=128, sigma=1.0, seed=0):
    # No component is shared within a group: every one of the R replicates
    # in every group is an independent draw. Averaging r of them should
    # shrink the across-group dispersion as 1/sqrt(r) -> beta ~= 0.5.
    rng = np.random.default_rng(seed)
    return {
        f"g{i}": {"y_true": (rng.standard_normal(R) * sigma).tolist()}
        for i in range(n_groups)
    }


def _systematic_groups(n_groups=40, R=128, sigma=1.0, frac=0.95, seed=0):
    # Each group draws ONE offset, held constant across all R replicates in
    # that group, plus smaller i.i.d. noise on top. The offset never
    # averages down no matter how many replicates are combined within the
    # group -> beta should sit well below 0.5, approaching 0 as frac -> 1.
    rng = np.random.default_rng(seed)
    groups = {}
    for i in range(n_groups):
        offset = rng.normal(0, sigma * frac)
        noise = rng.normal(0, sigma * np.sqrt(max(1 - frac**2, 0)), R)
        groups[f"g{i}"] = {"y_true": (offset + noise).tolist()}
    return groups


def test_rse_report_only_by_default_regardless_of_value():
    groups = _random_groups()
    result = ReplicationShrinkageExponentCheck(seed=1).run([], replicate_groups=groups)
    assert result.passed  # beta_tolerance=None -> always passes


def test_rse_recovers_beta_near_half_for_pure_random_replication():
    groups = _random_groups(seed=2)
    result = ReplicationShrinkageExponentCheck(seed=2).run([], replicate_groups=groups)
    assert result.value == pytest.approx(0.5, abs=0.2)


def test_rse_recovers_low_beta_for_mostly_systematic_scheme():
    groups = _systematic_groups(frac=0.95, seed=3)
    result = ReplicationShrinkageExponentCheck(seed=3).run([], replicate_groups=groups)
    assert result.value < 0.2


def test_rse_beta_decreases_monotonically_with_systematic_fraction():
    # The whole point of the check: beta is not binary, it tracks how much
    # of the replicate-to-replicate scatter is a per-group constant vs
    # genuinely random. Higher frac -> more systematic -> lower beta.
    betas = []
    for frac in (0.0, 0.7, 0.95):
        groups = _systematic_groups(frac=frac, seed=4)
        result = ReplicationShrinkageExponentCheck(seed=4).run([], replicate_groups=groups)
        betas.append(result.value)
    assert betas[0] > betas[1] > betas[2]


def test_rse_needs_reference_when_group_locations_differ():
    # Without a reference, real between-group location differences look
    # indistinguishable from a per-group systematic component -> beta is
    # driven toward 0 even though each group's own noise is pure random.
    # Supplying y_pred_mean per replicate (the true location) restores the
    # correct beta ~= 0.5 reading.
    rng = np.random.default_rng(5)
    R = 128
    locations = [0.0, 50.0, -30.0, 100.0]
    groups_raw = {}
    groups_residualized = {}
    for i, loc in enumerate(locations):
        noise = rng.standard_normal(R) * 1.0
        y = loc + noise
        groups_raw[f"g{i}"] = {"y_true": y.tolist()}
        groups_residualized[f"g{i}"] = {"y_true": y.tolist(), "y_pred_mean": [loc] * R}

    raw_result = ReplicationShrinkageExponentCheck(seed=5).run([], replicate_groups=groups_raw)
    resid_result = ReplicationShrinkageExponentCheck(seed=5).run([], replicate_groups=groups_residualized)

    assert raw_result.value < 0.2  # location differences masquerade as "systematic"
    assert resid_result.value == pytest.approx(0.5, abs=0.2)  # corrected by residualizing


def test_rse_opt_in_tolerance_can_fail():
    groups = _systematic_groups(frac=0.95, seed=3)
    result = ReplicationShrinkageExponentCheck(seed=3, beta_tolerance=0.15).run(
        [], replicate_groups=groups
    )
    assert not result.passed


def test_rse_skips_without_replicate_data():
    result = ReplicationShrinkageExponentCheck().run([])
    assert result.passed
    assert "Skipped" in result.message


def test_rse_skips_with_too_few_groups():
    groups = {"g0": {"y_true": list(np.arange(20.0))}, "g1": {"y_true": list(np.arange(20.0))}}
    result = ReplicationShrinkageExponentCheck().run([], replicate_groups=groups)
    assert result.passed
    assert "Skipped" in result.message


def test_rse_reads_from_history():
    rng = np.random.default_rng(6)
    n_groups, r_per_group = 8, 40
    history = [
        {"replicate_id": g, "y_true": float(rng.standard_normal())}
        for g in range(n_groups)
        for _ in range(r_per_group)
    ]
    result = ReplicationShrinkageExponentCheck(seed=6).run(history)
    assert result.value is not None
    assert result.value == pytest.approx(0.5, abs=0.3)


# ── DarkUncertaintyGapCheck ──────────────────────────────────────────────────

def test_dug_passes_when_declared_std_matches_scatter():
    rng = np.random.default_rng(0)
    groups = {}
    for i in range(30):
        y = rng.standard_normal(8) * 1.0
        groups[f"g{i}"] = {"y_true": y.tolist(), "y_pred_std": [1.0] * 8}
    result = DarkUncertaintyGapCheck(seed=0).run([], replicate_groups=groups)
    assert result.passed
    assert result.value == pytest.approx(1.0, abs=0.3)


def test_dug_fails_when_declared_std_too_small():
    rng = np.random.default_rng(0)
    groups = {}
    for i in range(30):
        y = rng.standard_normal(8) * 1.0
        groups[f"g{i}"] = {"y_true": y.tolist(), "y_pred_std": [0.1] * 8}
    result = DarkUncertaintyGapCheck(seed=0).run([], replicate_groups=groups)
    assert not result.passed
    assert result.value > 1.0


def test_dug_skips_without_y_pred_std():
    groups = {"g0": {"y_true": [1.0, 2.0, 3.0]}}
    result = DarkUncertaintyGapCheck().run([], replicate_groups=groups)
    assert result.passed
    assert "Skipped" in result.message
