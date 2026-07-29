import numpy as np
import pytest

from traits_audit.checks.aleatoric_epistemic import (
    ReducibilityRealisationRatioCheck,
    AleatoricFloorConsistencyCheck,
)


# ── ReducibilityRealisationRatioCheck ───────────────────────────────────────

def test_rrr_honest_claim_passes():
    rng = np.random.default_rng(0)
    claimed = rng.uniform(0.5, 1.5, size=50)
    before = rng.uniform(5, 10, size=50)
    after = before - claimed  # realized reduction == claimed exactly
    result = ReducibilityRealisationRatioCheck().run(
        [], claimed_epistemic_variance=claimed,
        realized_total_variance_before=before, realized_total_variance_after=after,
    )
    assert result.passed
    assert result.value == pytest.approx(1.0, abs=0.05)


def test_rrr_over_claimed_fails():
    rng = np.random.default_rng(0)
    claimed = rng.uniform(0.5, 1.5, size=50)
    before = rng.uniform(5, 10, size=50)
    after = before - claimed * 0.3  # only 30% of claimed reduction realized
    result = ReducibilityRealisationRatioCheck(rrr_tolerance=0.3).run(
        [], claimed_epistemic_variance=claimed,
        realized_total_variance_before=before, realized_total_variance_after=after,
    )
    assert not result.passed
    assert result.value < 1.0


def test_rrr_under_claimed_fails():
    rng = np.random.default_rng(0)
    claimed = rng.uniform(0.5, 1.5, size=50)
    before = rng.uniform(5, 10, size=50)
    after = before - claimed * 2.0  # more reduction realized than claimed
    result = ReducibilityRealisationRatioCheck(rrr_tolerance=0.3).run(
        [], claimed_epistemic_variance=claimed,
        realized_total_variance_before=before, realized_total_variance_after=after,
    )
    assert not result.passed
    assert result.value > 1.0


def test_rrr_skips_without_data():
    result = ReducibilityRealisationRatioCheck().run([])
    assert result.passed
    assert "Skipped" in result.message


def test_rrr_reads_from_history():
    history = [
        {"claimed_epistemic_variance": 1.0, "realized_total_variance_before": 5.0,
         "realized_total_variance_after": 4.0}
        for _ in range(20)
    ]
    result = ReducibilityRealisationRatioCheck().run(history)
    assert result.value == pytest.approx(1.0)


# ── AleatoricFloorConsistencyCheck ──────────────────────────────────────────

def test_afc_matching_std_passes():
    rng = np.random.default_rng(0)
    groups = {}
    for i in range(30):
        y = rng.standard_normal(8)
        groups[f"g{i}"] = {"y_true": y.tolist(), "y_pred_std": [1.0] * 8}
    result = AleatoricFloorConsistencyCheck().run([], replicate_groups=groups)
    assert result.passed
    assert result.value == pytest.approx(1.0, abs=0.3)


def test_afc_over_declared_fails():
    rng = np.random.default_rng(0)
    groups = {}
    for i in range(30):
        y = rng.standard_normal(8)
        groups[f"g{i}"] = {"y_true": y.tolist(), "y_pred_std": [3.0] * 8}
    result = AleatoricFloorConsistencyCheck(afc_tolerance=0.3).run([], replicate_groups=groups)
    assert not result.passed
    assert result.value > 1.0


def test_afc_under_declared_fails():
    rng = np.random.default_rng(0)
    groups = {}
    for i in range(30):
        y = rng.standard_normal(8)
        groups[f"g{i}"] = {"y_true": y.tolist(), "y_pred_std": [0.2] * 8}
    result = AleatoricFloorConsistencyCheck(afc_tolerance=0.3).run([], replicate_groups=groups)
    assert not result.passed
    assert result.value < 1.0


def test_afc_skips_without_std():
    groups = {"g0": {"y_true": [1.0, 2.0, 3.0]}}
    result = AleatoricFloorConsistencyCheck().run([], replicate_groups=groups)
    assert result.passed
    assert "Skipped" in result.message
