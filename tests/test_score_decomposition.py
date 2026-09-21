import numpy as np

from traits_audit.checks.scoring import ScoreDecompositionCheck


def _calibrated(n=600, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal(n)
    sigma = np.abs(rng.standard_normal(n)) + 0.5
    y_true = mu + sigma * rng.standard_normal(n)
    return {"y_true": y_true, "y_pred_mean": mu, "y_pred_std": sigma}


def _overconfident(n=600, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal(n)
    sigma = np.ones(n) * 0.01
    y_true = mu + rng.standard_normal(n)
    return {"y_true": y_true, "y_pred_mean": mu, "y_pred_std": sigma}


def test_calibrated_data_small_cal_and_identity_holds():
    result = ScoreDecompositionCheck().run([], **_calibrated())
    assert abs(result.details["identity_residual"]) < 1e-6
    assert result.value < 0.1


def test_overconfident_data_large_cal():
    calibrated = ScoreDecompositionCheck().run([], **_calibrated())
    overconfident = ScoreDecompositionCheck().run([], **_overconfident())
    assert abs(overconfident.details["identity_residual"]) < 1e-6
    assert overconfident.value > calibrated.value


def test_report_only_by_default():
    result = ScoreDecompositionCheck().run([], **_overconfident())
    assert result.passed  # cal_threshold=None -> always pass


def test_opt_in_threshold_can_fail():
    result = ScoreDecompositionCheck(cal_threshold=0.01).run([], **_overconfident())
    assert not result.passed


def test_skips_when_no_data():
    result = ScoreDecompositionCheck().run([])
    assert result.passed
    assert "Skipped" in result.message


def test_skips_with_too_few_samples():
    data = _calibrated(n=10)
    result = ScoreDecompositionCheck(n_bins=10).run([], **data)
    assert result.passed
    assert "Skipped" in result.message


def test_reads_from_history():
    rng = np.random.default_rng(1)
    n = 600
    mu = rng.standard_normal(n)
    sigma = np.abs(rng.standard_normal(n)) + 0.5
    y_true = mu + sigma * rng.standard_normal(n)
    history = [
        {"y_true": yt, "y_pred_mean": m, "y_pred_std": s}
        for yt, m, s in zip(y_true, mu, sigma, strict=False)
    ]
    result = ScoreDecompositionCheck().run(history)
    assert result.value is not None
