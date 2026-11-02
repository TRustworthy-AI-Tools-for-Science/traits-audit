import numpy as np
import pytest

from traits_audit.credal import CredalSet


def test_bounding_interval_direct():
    cs = CredalSet(lower=np.array([0.0, 1.0]), upper=np.array([1.0, 2.0]))
    lo, hi = cs.bounding_interval()
    np.testing.assert_allclose(lo, [0.0, 1.0])
    np.testing.assert_allclose(hi, [1.0, 2.0])


def test_bounding_interval_from_dirac_ensemble():
    ensemble = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    cs = CredalSet(y_pred_ensemble=ensemble)
    lo, hi = cs.bounding_interval()
    np.testing.assert_allclose(lo, [0.0, 1.0])
    np.testing.assert_allclose(hi, [2.0, 1.0])


def test_no_representation_raises():
    with pytest.raises(ValueError):
        CredalSet().bounding_interval()


def test_contains_direct_interval():
    cs = CredalSet(lower=np.array([0.0]), upper=np.array([1.0]))
    assert cs.contains(np.array([0.5]))[0]
    assert not cs.contains(np.array([2.0]))[0]


def test_reference_probabilities_tight_ensemble_low_imprecision():
    rng = np.random.default_rng(0)
    n = 200
    mean = rng.standard_normal(n)
    ensemble = mean[None, :] + rng.normal(scale=1e-6, size=(5, n))
    cs = CredalSet(y_pred_ensemble=ensemble)
    ref_lo = mean - 3.0
    ref_hi = mean + 3.0
    p_upper, p_lower = cs.reference_probabilities(ref_lo, ref_hi)
    # every member's point prediction sits well inside a wide reference band
    np.testing.assert_allclose(p_upper, p_lower, atol=1e-9)
    assert np.all(p_upper > 0.99)


def test_reference_probabilities_disagreeing_ensemble_high_imprecision():
    n = 50
    ensemble = np.vstack([np.zeros(n), np.full(n, 10.0)])
    std_ensemble = np.ones_like(ensemble)
    cs = CredalSet(y_pred_ensemble=ensemble, y_pred_std_ensemble=std_ensemble)
    ref_lo = np.full(n, -0.5)
    ref_hi = np.full(n, 0.5)
    p_upper, p_lower = cs.reference_probabilities(ref_lo, ref_hi)
    assert np.all(p_upper > p_lower)  # disagreement => imprecision


def test_reference_probabilities_dirac_disagreement_is_visible():
    # Dirac mode (no y_pred_std_ensemble): each member's own probability of
    # the reference event is 0 or 1 (in or out), so the credal set's
    # sup/inf across members is any()/all() of that indicator -- NOT their
    # mean. A regression guard for a real bug where both bounds were
    # computed as mean(member_in), collapsing P_upper == P_lower (and hence
    # IWF) to a single value regardless of how much the ensemble disagreed.
    n = 10
    # 3 of 5 members land inside [-0.5, 0.5], 2 land outside.
    ensemble = np.vstack([
        np.zeros(n), np.zeros(n), np.zeros(n),
        np.full(n, 10.0), np.full(n, 10.0),
    ])
    cs = CredalSet(y_pred_ensemble=ensemble)
    ref_lo = np.full(n, -0.5)
    ref_hi = np.full(n, 0.5)
    p_upper, p_lower = cs.reference_probabilities(ref_lo, ref_hi)
    # any() of 5 members -> at least one inside -> P_upper = 1
    np.testing.assert_allclose(p_upper, 1.0)
    # all() of 5 members -> not all inside -> P_lower = 0
    np.testing.assert_allclose(p_lower, 0.0)


def test_reference_probabilities_dirac_agreement_gives_zero_width():
    # All members agree (all inside, or all outside) -> any() == all(),
    # P_upper == P_lower, genuinely zero imprecision.
    n = 10
    ensemble = np.zeros((4, n))  # every member predicts 0.0
    cs = CredalSet(y_pred_ensemble=ensemble)
    ref_lo = np.full(n, -0.5)
    ref_hi = np.full(n, 0.5)
    p_upper, p_lower = cs.reference_probabilities(ref_lo, ref_hi)
    np.testing.assert_allclose(p_upper, 1.0)
    np.testing.assert_allclose(p_lower, 1.0)
