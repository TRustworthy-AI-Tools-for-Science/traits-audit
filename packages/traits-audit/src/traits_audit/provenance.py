"""
Type-B provenance ledger for the Type A/Type B taxonomy class
(METRIC_TAXONOMY_AUDIT.md §4.2).

This class is administrative rather than physical (Colclough's category-error
objection, LITERATURE_SUMMARY.md §3.3.2): it records how a number was arrived
at, not a property estimable from data. ``TypeBLedger`` is therefore a
declared record, not a fitted one — the caller states which components are
Type B (not evaluated from a series of observations: priors, jitter, fixed
noise floors, expert bounds) and the check computes the mass fraction by
ablation via a caller-supplied ``variance_fn``.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TypeBLedger:
    """Declared record of a predictive variance's provenance.

    ``components`` is informational bookkeeping (surfaced in
    ``AuditResult.details`` for documentation/audit-trail purposes);
    ``TypeBMassFractionCheck`` computes its value from ``variance_fn``, not
    from these numeric fields directly.
    """
    components: dict[str, float]
    type_b_keys: set[str]

    def type_a_keys(self) -> set[str]:
        return set(self.components) - self.type_b_keys
