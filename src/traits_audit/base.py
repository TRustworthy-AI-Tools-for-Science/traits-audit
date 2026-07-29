"""
Core primitives: AuditCategory, AuditResult, AuditReport, AuditCheck.

Nothing here assumes an active learning loop, a specific model family, or
a particular data shape.  These types are the only shared contract between
the hook, the pipeline, and individual checks.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class AuditCategory(str, Enum):
    """
    Broad classification of the uncertainty source a check addresses.

    The first three values are the aleatoric/epistemic (reducibility) taxonomy
    class from ``.claude/METRIC_TAXONOMY_AUDIT.md`` §3 — that scheme was
    already exactly what this enum encoded, so it keeps its original three-way
    split rather than gaining a redundant fourth member. The remaining values
    each correspond to one of the other seven classification schemes surveyed
    there (random/systematic, Type A/Type B, ergodic/non-ergodic,
    variability/ignorance, model/approximation/misspecification/procedural,
    locus in the chain, reduction under replication). ``UNKNOWN`` is reserved
    for checks the audit explicitly identifies as NOT belonging to any of the
    eight classes (e.g. cross-cutting diagnostics like the tail index) rather
    than as a generic fallback.
    """
    ALEATORIC_IRREDUCIBLE       = "aleatoric_irreducible"
    ALEATORIC_MODEL             = "aleatoric_model"
    EPISTEMIC                   = "epistemic"
    RANDOM_SYSTEMATIC           = "random_systematic"
    TYPE_A_TYPE_B               = "type_a_type_b"
    ERGODIC_NON_ERGODIC         = "ergodic_non_ergodic"
    VARIABILITY_IGNORANCE       = "variability_ignorance"
    MODEL_PROCEDURAL            = "model_procedural"
    LOCUS_IN_CHAIN              = "locus_in_chain"
    REDUCTION_UNDER_REPLICATION = "reduction_under_replication"
    UNKNOWN                     = "unknown"


@dataclass
class AuditResult:
    """Outcome of a single check."""
    name:      str
    passed:    bool
    category:  AuditCategory
    value:     Optional[float]       = None
    threshold: Optional[Any]         = None
    message:   str                   = ""
    details:   Dict[str, Any]        = field(default_factory=dict)


@dataclass
class AuditReport:
    """Aggregated results from one pipeline run."""
    results:  List[AuditResult]  = field(default_factory=list)
    metadata: Dict[str, Any]     = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def n_passed(self) -> int:
        return sum(r.passed for r in self.results)

    @property
    def n_failed(self) -> int:
        return sum(not r.passed for r in self.results)

    def summary(self) -> str:
        lines = [f"Audit Report  —  {self.n_passed}/{len(self.results)} checks passed"]
        for r in self.results:
            tag = "PASS" if r.passed else "FAIL"
            val = f" ({r.value:.4f})" if r.value is not None else ""
            lines.append(f"  [{tag}] {r.name}{val}: {r.message}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed":   self.passed,
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "metadata": self.metadata,
            "results": [
                {
                    "name":      r.name,
                    "passed":    r.passed,
                    "category":  r.category.value,
                    "value":     r.value,
                    "threshold": r.threshold,
                    "message":   r.message,
                    "details":   r.details,
                }
                for r in self.results
            ],
        }


class AuditCheck(ABC):
    """
    Base class for a single audit check.

    Subclass this, implement :attr:`name`, :attr:`category`, and :meth:`run`.
    Register instances with an :class:`~traits_audit.pipeline.AuditPipeline`.

    ``run`` receives two things:

    ``history``
        A ``list[dict]`` — one dict per loop step, containing whatever the
        loop chose to record via :meth:`~traits_audit.hook.AuditHook.on_step`.
        Checks that operate on step sequences (e.g. uncertainty trend) read from
        here.  May be empty if the loop did not record step data.

    ``**kwargs``
        Named arrays or scalars passed directly to :meth:`~traits_audit.pipeline.AuditPipeline.run`.
        Checks that operate on batch predictions (calibration, coverage) use
        these.  Each check documents exactly which keys it requires.

    Checks may use history, kwargs, or both.  Unused keys are silently ignored,
    so the same call to ``pipeline.run`` works for heterogeneous check lists.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier — must be unique within a pipeline."""
        ...

    @property
    @abstractmethod
    def category(self) -> AuditCategory: ...

    @abstractmethod
    def run(self, history: List[Dict[str, Any]], **kwargs) -> AuditResult: ...
