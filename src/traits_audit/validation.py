"""
Pairing validation for a configured audit pipeline.

METRIC_TAXONOMY_AUDIT.md repeatedly stresses that several of the proposed
metrics are uninterpretable in isolation — they are only meaningful when
reported alongside a specific "contextualizing twin":

- Lyapunov lambda_max (local) must be paired with DMDc rho(A) (global) — §4.4:
  "neither alone separates a locally rough landscape from a globally
  divergent one."
- EnsembleIndependenceDeficit paired with ResidualPersistenceHalfLife — §4.4.
- ReplicationShrinkageExponent paired with DarkUncertaintyGap — §4.1.
- ReducibilityRealisationRatio paired with AleatoricFloorConsistency — §4.3.
- ImprecisionWidthFraction paired with EnvelopeViolationRate — §4.5.
- ProceduralVarianceShare paired with DataVarianceShare — §4.6 (their ratio
  is the point).
- TypeBMassFraction should be reported alongside any aleatoric/epistemic
  split check — §4.2 ("makes the non-mapping visible in the numbers").

This module checks a *configured* pipeline (the list of ``AuditCheck``
instances passed to ``AuditPipeline``) for these pairings and returns
advisory warnings — it never raises and never fails a run, mirroring the
existing "a failed check never aborts the run" philosophy.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, List, Sequence

from .base import AuditCategory, AuditCheck

# 1:1 name-based twins: at least one name in the frozenset must also be configured.
NAME_PAIRS: Dict[str, FrozenSet[str]] = {
    "LyapunovStability":              frozenset({"DMDcSpectralRadius"}),
    "DMDcSpectralRadius":             frozenset({"LyapunovStability"}),
    "ReplicationShrinkageExponent":   frozenset({"DarkUncertaintyGap"}),
    "DarkUncertaintyGap":             frozenset({"ReplicationShrinkageExponent"}),
    "ReducibilityRealisationRatio":   frozenset({"AleatoricFloorConsistency"}),
    "AleatoricFloorConsistency":      frozenset({"ReducibilityRealisationRatio"}),
    "EnsembleIndependenceDeficit":    frozenset({"ResidualPersistenceHalfLife"}),
    "ResidualPersistenceHalfLife":    frozenset({"EnsembleIndependenceDeficit"}),
    "ImprecisionWidthFraction":       frozenset({"EnvelopeViolationRate"}),
    "EnvelopeViolationRate":          frozenset({"ImprecisionWidthFraction"}),
    "ProceduralVarianceShare":        frozenset({"DataVarianceShare"}),
    "DataVarianceShare":              frozenset({"ProceduralVarianceShare"}),
}

# TBMF's twin is "any check in the aleatoric/epistemic family", not one named
# check, so it is expressed separately from the name->names table above.
_ALEATORIC_EPISTEMIC_FAMILY = frozenset({
    AuditCategory.ALEATORIC_IRREDUCIBLE,
    AuditCategory.ALEATORIC_MODEL,
    AuditCategory.EPISTEMIC,
})


def find_unpaired_checks(checks: Sequence[AuditCheck]) -> List[str]:
    """Given the AuditChecks configured in a pipeline, return one
    human-readable warning per check whose contextualizing twin (per
    METRIC_TAXONOMY_AUDIT.md's explicit pairing requirements) is absent.

    Advisory only: never raises, never blocks a run.
    """
    names = {c.name for c in checks}
    categories_present = {c.category for c in checks}
    warnings: List[str] = []

    for name, twins in NAME_PAIRS.items():
        if name in names and not (twins & names):
            warnings.append(
                f"{name} is configured without its contextualizing twin "
                f"({' or '.join(sorted(twins))}) — reported alone it is "
                f"liable to be misread; see METRIC_TAXONOMY_AUDIT.md."
            )

    if "TypeBMassFraction" in names and not (categories_present & _ALEATORIC_EPISTEMIC_FAMILY):
        warnings.append(
            "TypeBMassFraction is configured without any aleatoric/epistemic-split "
            "check alongside it — the non-mapping between Type A/B and "
            "aleatoric/epistemic is invisible without both reported together."
        )

    return warnings
