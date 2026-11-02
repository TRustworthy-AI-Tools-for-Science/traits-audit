import numpy as np

from traits_audit.bootstrap import moving_block_bootstrap_ci


def test_returns_none_for_short_series():
    assert moving_block_bootstrap_ci(np.arange(4), np.mean, block_len=8) is None


def test_ci_brackets_the_mean_of_a_stationary_series():
    rng = np.random.default_rng(0)
    values = rng.standard_normal(200)
    ci = moving_block_bootstrap_ci(values, np.mean, block_len=8, n_boot=200, seed=1)
    assert ci is not None
    lo, hi = ci
    assert lo < np.mean(values) < hi


def test_deterministic_given_seed():
    rng = np.random.default_rng(0)
    values = rng.standard_normal(100)
    ci1 = moving_block_bootstrap_ci(values, np.std, block_len=5, n_boot=50, seed=7)
    ci2 = moving_block_bootstrap_ci(values, np.std, block_len=5, n_boot=50, seed=7)
    assert ci1 == ci2
