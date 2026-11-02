import numpy as np

from traits_audit.checks._replicates import build_replicate_groups


def test_build_from_replicate_groups_dict_of_arrays():
    groups = build_replicate_groups([], {"replicate_groups": {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0]}})
    assert len(groups) == 2
    by_key = {g.key: g for g in groups}
    assert list(by_key["a"].y_true) == [1.0, 2.0, 3.0]
    assert by_key["a"].r == 3


def test_build_from_replicate_groups_dict_with_std():
    groups = build_replicate_groups(
        [], {"replicate_groups": {"a": {"y_true": [1.0, 2.0], "y_pred_std": [0.5, 0.5]}}}
    )
    assert len(groups) == 1
    assert groups[0].y_pred_std is not None
    assert list(groups[0].y_pred_std) == [0.5, 0.5]


def test_groups_with_fewer_than_2_replicates_dropped():
    groups = build_replicate_groups([], {"replicate_groups": {"a": [1.0], "b": [1.0, 2.0]}})
    assert len(groups) == 1
    assert groups[0].key == "b"


def test_build_from_replicate_id_kwarg():
    rid = np.array([0, 0, 1, 1, 1])
    y_true = np.array([1.0, 1.1, 2.0, 2.1, 2.2])
    groups = build_replicate_groups([], {"replicate_id": rid, "y_true": y_true})
    assert len(groups) == 2
    sizes = sorted(g.r for g in groups)
    assert sizes == [2, 3]


def test_build_from_history():
    history = [
        {"replicate_id": 0, "y_true": 1.0},
        {"replicate_id": 0, "y_true": 1.2},
        {"replicate_id": 1, "y_true": 2.0},
        {"replicate_id": 1, "y_true": 2.1},
    ]
    groups = build_replicate_groups(history, {})
    assert len(groups) == 2


def test_returns_empty_list_when_nothing_available():
    assert build_replicate_groups([], {}) == []
    assert build_replicate_groups([{"foo": 1}], {}) == []
