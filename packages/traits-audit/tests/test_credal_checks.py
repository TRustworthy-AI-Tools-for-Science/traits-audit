import numpy as np

from traits_audit.checks.credal import (
    EnvelopeViolationRateCheck,
    ImprecisionWidthFractionCheck,
)

# ── ImprecisionWidthFractionCheck ───────────────────────────────────────────

def test_iwf_tightly_agreeing_ensemble_near_zero():
    rng = np.random.default_rng(0)
    n = 100
    mu = rng.standard_normal(n)
    sigma = np.ones(n)
    ensemble = mu[None, :] + rng.normal(scale=1e-6, size=(5, n))
    std_ensemble = np.full((5, n), 0.01)
    result = ImprecisionWidthFractionCheck().run(
        [], y_pred_mean=mu, y_pred_std=sigma, y_pred_ensemble=ensemble, y_pred_std_ensemble=std_ensemble
    )
    assert result.value < 0.05


def test_iwf_disagreeing_ensemble_large():
    # One member confidently agrees with the reference interval [-1, 1]
    # (mean 0), the other confidently disagrees (mean 4, several std away)
    # -> upper/lower probability of the reference event differ sharply.
    n = 50
    mu = np.zeros(n)
    sigma = np.ones(n)
    ensemble = np.vstack([np.zeros(n), np.full(n, 4.0)])
    std_ensemble = np.ones((2, n))
    result = ImprecisionWidthFractionCheck(ref_z=1.0).run(
        [], y_pred_mean=mu, y_pred_std=sigma, y_pred_ensemble=ensemble, y_pred_std_ensemble=std_ensemble
    )
    assert result.value > 0.3


def test_iwf_skips_without_data():
    result = ImprecisionWidthFractionCheck().run([])
    assert result.passed
    assert "Skipped" in result.message


def test_iwf_report_only_by_default():
    n = 50
    mu = np.zeros(n)
    sigma = np.ones(n)
    ensemble = np.vstack([np.full(n, -5.0), np.full(n, 5.0)])
    result = ImprecisionWidthFractionCheck().run(
        [], y_pred_mean=mu, y_pred_std=sigma, y_pred_ensemble=ensemble
    )
    assert result.passed  # threshold=None


def test_iwf_dirac_mode_detects_disagreement():
    # Regression guard: Dirac-mode (no y_pred_std_ensemble) used to compute
    # P_upper == P_lower == mean(member_in), forcing IWF to 0 always,
    # regardless of how much the ensemble disagreed. With the correct
    # any()/all() sup/inf, a disagreeing point-prediction ensemble must
    # show real imprecision.
    n = 20
    mu = np.zeros(n)
    sigma = np.ones(n)
    ensemble = np.vstack([np.zeros(n), np.zeros(n), np.full(n, 10.0)])
    result = ImprecisionWidthFractionCheck(ref_z=1.0).run(
        [], y_pred_mean=mu, y_pred_std=sigma, y_pred_ensemble=ensemble
    )
    assert result.value > 0.5


# ── EnvelopeViolationRateCheck ──────────────────────────────────────────────

def test_envelope_all_contained_passes():
    n = 100
    y_true = np.zeros(n)
    lower = np.full(n, -1.0)
    upper = np.full(n, 1.0)
    result = EnvelopeViolationRateCheck().run([], y_true=y_true, credal_lower=lower, credal_upper=upper)
    assert result.passed
    assert result.value == 0.0


def test_envelope_systematic_violations_fails():
    n = 100
    y_true = np.full(n, 5.0)
    lower = np.full(n, -1.0)
    upper = np.full(n, 1.0)
    result = EnvelopeViolationRateCheck(max_violation_rate=0.1).run(
        [], y_true=y_true, credal_lower=lower, credal_upper=upper
    )
    assert not result.passed
    assert result.value == 1.0


def test_envelope_skips_without_data():
    result = EnvelopeViolationRateCheck().run([])
    assert result.passed
    assert "Skipped" in result.message


def test_envelope_reads_y_true_from_history():
    history = [{"y_true": 0.0} for _ in range(20)]
    result = EnvelopeViolationRateCheck().run(
        history, credal_lower=np.full(20, -1.0), credal_upper=np.full(20, 1.0)
    )
    assert result.value == 0.0
