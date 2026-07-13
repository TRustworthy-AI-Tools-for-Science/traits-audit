"""
traits_audit
=================
A flexible uncertainty audit pipeline that hooks into any pre-existing
active learning loop.  No assumptions about loop structure, model type,
or data representation.

Quick import
------------
>>> from traits_audit import AuditHook, AuditPipeline
>>> from traits_audit.base import AuditCheck, AuditResult, AuditReport, AuditCategory
>>> from traits_audit.checks import (
...     CalibrationErrorCheck,
...     IntervalCoverageCheck,
...     VarianceAlignmentCheck,
...     UncertaintyEvolutionCheck,
...     UncertaintyAnomalyCheck,
...     VarianceErrorCorrelationCheck,
...     LyapunovStabilityCheck,
... )
"""
__version__ = "0.1.2"

from . import dmdc
from .base import AuditCategory, AuditCheck, AuditReport, AuditResult
from .checks.lyapunov import LyapunovStabilityCheck
from .detrend import DetrendResult, RegimeDetrender
from .hook import AuditHook
from .mlflow_logger import MLflowLogger
from .pipeline import AuditPipeline

__all__ = [
    "AuditHook",
    "AuditPipeline",
    "AuditCheck",
    "AuditResult",
    "AuditReport",
    "AuditCategory",
    "MLflowLogger",
    "LyapunovStabilityCheck",
    "DetrendResult",
    "RegimeDetrender",
    "dmdc",
]
