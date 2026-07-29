import numpy as np
import pytest

from traits_audit.pipeline_attribution import StageUncertainty, run_stage_variance_attribution
from traits_audit.checks.attribution import StageVarianceAttributionCheck


def _uniform01(rng):
    return rng.uniform(0.0, 1.0)


# Y = X1 + X2 + X1*X2 (+ a no-effect X3), Xi ~ U(0,1) i.i.d.
# Closed-form Sobol indices (derived from the ANOVA decomposition):
#   V1 = V2 = (3/2)^2 * Var(U) = 2.25/12 = 0.1875
#   V12 (interaction) = Var(U1*U2) = 1/144 (since f12(x1,x2) reduces exactly
#     to (x1-0.5)(x2-0.5))
#   Var(Y) = V1 + V2 + V12 = 0.381944...
#   S1 = S2 = V1/Var(Y) ~= 0.4910;  S12 = V12/Var(Y) ~= 0.01818
#   ST1 = ST2 = S1 + S12 ~= 0.5092
_V1 = 2.25 / 12
_V12 = 1.0 / 144
_VAR_Y = 2 * _V1 + _V12
_S_ANALYTIC = _V1 / _VAR_Y
_ST_ANALYTIC = (_V1 + _V12) / _VAR_Y
_GAP_ANALYTIC = _V12 / _VAR_Y


def _interaction_chain(x1, x2, x3):
    return x1 + x2 + x1 * x2


def test_sva_recovers_known_sobol_indices_with_interaction():
    stages = [
        StageUncertainty("x1", _uniform01),
        StageUncertainty("x2", _uniform01),
        StageUncertainty("x3", _uniform01),
    ]
    result = run_stage_variance_attribution(_interaction_chain, stages, n_mc=50000, seed=0)

    assert result.first_order[0] == pytest.approx(_S_ANALYTIC, abs=0.06)
    assert result.first_order[1] == pytest.approx(_S_ANALYTIC, abs=0.06)
    assert result.total_effect[0] == pytest.approx(_ST_ANALYTIC, abs=0.06)
    assert result.total_effect[1] == pytest.approx(_ST_ANALYTIC, abs=0.06)
    # x3 has no effect on the output at all.
    assert result.first_order[2] == pytest.approx(0.0, abs=0.03)
    assert result.total_effect[2] == pytest.approx(0.0, abs=0.03)
    # The interaction gap is the whole point: must be reported, and must be
    # near the analytic value for x1/x2, near zero for the no-effect x3.
    assert result.interaction_gap[0] == pytest.approx(_GAP_ANALYTIC, abs=0.05)
    assert result.interaction_gap[1] == pytest.approx(_GAP_ANALYTIC, abs=0.05)
    assert result.interaction_gap[2] == pytest.approx(0.0, abs=0.03)


def _additive_chain(x1, x2):
    return x1 + x2


def test_sva_pure_additive_chain_has_near_zero_interaction_gap():
    stages = [StageUncertainty("x1", _uniform01), StageUncertainty("x2", _uniform01)]
    result = run_stage_variance_attribution(_additive_chain, stages, n_mc=20000, seed=1)
    assert np.all(np.abs(result.interaction_gap) < 0.05)
    # A purely additive chain: each stage's first-order and total-effect
    # index should each be close to 0.5 (equal variance contributions).
    assert result.first_order[0] == pytest.approx(0.5, abs=0.05)
    assert result.first_order[1] == pytest.approx(0.5, abs=0.05)


# ── StageVarianceAttributionCheck (thin wrapper) ────────────────────────────

def test_check_wrapper_reports_max_interaction_gap():
    stages = [
        StageUncertainty("x1", _uniform01),
        StageUncertainty("x2", _uniform01),
        StageUncertainty("x3", _uniform01),
    ]
    result = StageVarianceAttributionCheck(n_mc=20000, seed=0).run(
        [], chain_fn=_interaction_chain, stages=stages
    )
    assert result.value == pytest.approx(max(result.details["interaction_gap"]))
    assert "first_order" in result.details
    assert "total_effect" in result.details
    assert "interaction_gap" in result.details


def test_check_wrapper_skips_without_data():
    result = StageVarianceAttributionCheck().run([])
    assert result.passed
    assert "Skipped" in result.message


def test_check_wrapper_report_only_by_default():
    stages = [StageUncertainty("x1", _uniform01), StageUncertainty("x2", _uniform01)]
    result = StageVarianceAttributionCheck(n_mc=2000, seed=0).run(
        [], chain_fn=_additive_chain, stages=stages
    )
    assert result.passed  # max_interaction_gap=None -> always pass
