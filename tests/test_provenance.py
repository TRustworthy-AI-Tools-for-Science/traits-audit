from traits_audit.provenance import TypeBLedger


def test_type_a_keys_is_complement_of_type_b_keys():
    ledger = TypeBLedger(
        components={"prior_variance": 0.1, "jitter": 0.01, "residual_variance": 0.5},
        type_b_keys={"prior_variance", "jitter"},
    )
    assert ledger.type_a_keys() == {"residual_variance"}


def test_empty_type_b_keys_means_all_type_a():
    ledger = TypeBLedger(components={"residual_variance": 0.5}, type_b_keys=set())
    assert ledger.type_a_keys() == {"residual_variance"}
