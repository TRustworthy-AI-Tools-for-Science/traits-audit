import numpy as np

from traits_audit.checks.tail import TailIndexCheck


def test_gaussian_residuals_pass_with_large_alpha():
    rng = np.random.default_rng(0)
    n = 2000
    mu = np.zeros(n)
    sigma = np.ones(n)
    y_true = rng.standard_normal(n)
    result = TailIndexCheck().run([], y_true=y_true, y_pred_mean=mu, y_pred_std=sigma)
    assert result.passed
    assert result.value > 2.0


def test_student_t_heavy_tail_fails():
    rng = np.random.default_rng(0)
    n = 2000
    mu = np.zeros(n)
    sigma = np.ones(n)
    y_true = rng.standard_t(df=1.5, size=n)
    result = TailIndexCheck().run([], y_true=y_true, y_pred_mean=mu, y_pred_std=sigma)
    assert not result.passed
    assert result.value < 2.0


def test_skips_below_min_samples():
    y_true = np.zeros(30)
    mu = np.zeros(30)
    sigma = np.ones(30)
    result = TailIndexCheck().run([], y_true=y_true, y_pred_mean=mu, y_pred_std=sigma)
    assert result.passed
    assert "Skipped" in result.message


def test_skips_without_data():
    result = TailIndexCheck().run([])
    assert result.passed
    assert "Skipped" in result.message


def test_reads_from_history():
    rng = np.random.default_rng(0)
    n = 200
    history = [
        {"y_true": v, "y_pred_mean": 0.0, "y_pred_std": 1.0}
        for v in rng.standard_normal(n)
    ]
    result = TailIndexCheck().run(history)
    assert result.value is not None
