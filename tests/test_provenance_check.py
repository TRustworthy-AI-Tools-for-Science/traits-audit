import pytest

from traits_audit.checks.provenance import TypeBMassFractionCheck
from traits_audit.provenance import TypeBLedger


def _gp_like_variance_fn(prior_var=0.3, jitter=0.05, noise=0.1, residual_var=0.6):
    def variance_fn(ablate):
        v = residual_var  # Type A always present
        if "prior_var" not in ablate:
            v += prior_var
        if "jitter" not in ablate:
            v += jitter
        if "noise" not in ablate:
            v += noise
        return v

    return variance_fn


def test_tbmf_partial_type_b_contribution():
    ledger = TypeBLedger(
        components={"prior_var": 0.3, "jitter": 0.05, "noise": 0.1, "residual_var": 0.6},
        type_b_keys={"prior_var", "jitter", "noise"},
    )
    variance_fn = _gp_like_variance_fn()
    result = TypeBMassFractionCheck().run([], ledger=ledger, variance_fn=variance_fn)
    v_full = 0.3 + 0.05 + 0.1 + 0.6
    expected = (v_full - 0.6) / v_full
    assert result.value == pytest.approx(expected)


def test_tbmf_zero_when_ablation_changes_nothing():
    ledger = TypeBLedger(components={"a": 1.0}, type_b_keys={"a"})
    variance_fn = lambda ablate: 1.0  # ablation has no effect
    result = TypeBMassFractionCheck().run([], ledger=ledger, variance_fn=variance_fn)
    assert result.value == pytest.approx(0.0)


def test_tbmf_one_when_ablation_zeroes_everything():
    ledger = TypeBLedger(components={"a": 1.0}, type_b_keys={"a"})

    def variance_fn(ablate):
        return 0.0 if "a" in ablate else 1.0

    result = TypeBMassFractionCheck().run([], ledger=ledger, variance_fn=variance_fn)
    assert result.value == pytest.approx(1.0)


def test_tbmf_skips_without_ledger_or_variance_fn():
    result = TypeBMassFractionCheck().run([])
    assert result.passed
    assert "Skipped" in result.message

    ledger = TypeBLedger(components={"a": 1.0}, type_b_keys={"a"})
    result2 = TypeBMassFractionCheck().run([], ledger=ledger)
    assert result2.passed
    assert "Skipped" in result2.message


def test_tbmf_report_only_by_default():
    ledger = TypeBLedger(components={"a": 1.0}, type_b_keys={"a"})
    variance_fn = lambda ablate: 0.0 if "a" in ablate else 1.0
    result = TypeBMassFractionCheck().run([], ledger=ledger, variance_fn=variance_fn)
    assert result.passed  # max_tbmf=None -> always pass, even at TBMF=1


def test_tbmf_opt_in_threshold_can_fail():
    ledger = TypeBLedger(components={"a": 1.0}, type_b_keys={"a"})
    variance_fn = lambda ablate: 0.0 if "a" in ablate else 1.0
    result = TypeBMassFractionCheck(max_tbmf=0.5).run([], ledger=ledger, variance_fn=variance_fn)
    assert not result.passed
