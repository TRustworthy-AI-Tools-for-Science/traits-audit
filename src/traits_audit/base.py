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
    """Broad classification of the uncertainty source a check addresses."""
    ALEATORIC_IRREDUCIBLE = "aleatoric_irreducible"
    ALEATORIC_MODEL       = "aleatoric_model"
    EPISTEMIC             = "epistemic"
    UNKNOWN               = "unknown"


@dataclass
class AuditResult:
    """Outcome of a single check."""
    name:      str
    passed:    bool
    category:  AuditCategory
    value:     Optional[float]       = None
    threshold: Optional[float]       = None
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
