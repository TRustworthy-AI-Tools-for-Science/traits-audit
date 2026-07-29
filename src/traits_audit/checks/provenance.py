"""Type A/Type B taxonomy class: Type-B Mass Fraction (TBMF).

See METRIC_TAXONOMY_AUDIT.md §4.2. This class is administrative rather than
physical (Colclough's category-error objection) — it must be *declared*
via a :class:`traits_audit.provenance.TypeBLedger`, not estimated.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..base import AuditCategory, AuditCheck, AuditResult
from ..provenance import TypeBLedger


class TypeBMassFractionCheck(AuditCheck):
    """
    Type-B Mass Fraction (TBMF) — the fraction of a stated predictive
    variance that traces to components not evaluated from a series of
    observations (priors, kernel hyperpriors, fixed noise floors, expert
    bounds), computed by ablation.

    ``TBMF = (variance_fn(no_ablation) - variance_fn(type_b_ablated)) / variance_fn(no_ablation)``

    For a GP surrogate: prior variance, jitter, fixed noise level and
    hyperparameter priors are Type B; the data-fit residual variance is Type
    A. TBMF=0 means the interval is entirely data-derived; TBMF=1 means no
    observation contributed to it.

    LITERATURE_SUMMARY.md records four independent authors establishing that
    Type A/Type B does not map onto aleatoric/epistemic. Reporting TBMF
    alongside any aleatoric/epistemic split check makes that non-mapping
    visible in the numbers — enforced advisorially by configuring this
    check alongside e.g. ``CalibrationErrorCheck`` (see
    :func:`traits_audit.validation.find_unpaired_checks`).

    Kwargs-only: ``variance_fn`` is an ablation callback, not a per-step
    scalar, so there is no history route (compare
    ``ImprecisionWidthFractionCheck``, kwargs-only for the same reason).

    Parameters
    ----------
    max_tbmf : float or None
        Maximum acceptable TBMF. ``None`` (default) disables pass/fail —
        this is a provenance report, not inherently a pass/fail quantity.

    References
    ----------
    JCGM 100:2008, cl. 2.3.2-2.3.3. Kacker, R. & Jones, A. (2003).
    *Metrologia*, 40(5). Bich, W., Cox, M. G. & Michotte, C. (2016).
    *Metrologia*, 53(5), S149-S159.

    Required data (kwargs)
    -----------------------
    ``ledger`` (:class:`traits_audit.provenance.TypeBLedger`),
    ``variance_fn`` (``Callable[[FrozenSet[str]], float]`` — the predictive
    variance when the given set of component keys is ablated to its
    non-informative limit).
    """

    def __init__(self, max_tbmf: Optional[float] = None):
        self.max_tbmf = max_tbmf

    @property
    def name(self) -> str:
        return "TypeBMassFraction"

    @property
    def category(self) -> AuditCategory:
        return AuditCategory.TYPE_A_TYPE_B

    def run(self, history: List[Dict[str, Any]], **kwargs) -> AuditResult:
        ledger: Optional[TypeBLedger] = kwargs.get("ledger")
        variance_fn = kwargs.get("variance_fn")

        if ledger is None or variance_fn is None:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Skipped — ledger and variance_fn not both provided.",
            )

        v_full = float(variance_fn(frozenset()))
        if v_full <= 0:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Skipped — non-positive full variance.",
            )
        v_type_a_only = float(variance_fn(frozenset(ledger.type_b_keys)))
        tbmf = (v_full - v_type_a_only) / v_full

        passed = True if self.max_tbmf is None else tbmf <= self.max_tbmf
        return AuditResult(
            name=self.name,
            passed=passed,
            category=self.category,
            value=tbmf,
            threshold=self.max_tbmf,
            message=f"TBMF = {tbmf:.4f}  (0=fully data-derived, 1=no observation contributed)",
            details={
                "v_full": v_full,
                "v_type_a_only": v_type_a_only,
                "ledger_components": dict(ledger.components),
                "type_b_keys": sorted(ledger.type_b_keys),
            },
        )
