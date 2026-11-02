"""Stability analysis for surrogate/AL dynamics: DMDc, detrending, Lyapunov numerics.

Submodules are pure numpy/scipy (``windowed.stability_convergence_windowed_pydmd``
additionally needs the optional ``pydmd`` dependency, imported lazily).
"""
from . import action_hygiene, detrend, dmdc, lle, lyapunov_eq, regime, windowed
from .detrend import DetrendResult, RegimeDetrender

__all__ = [
    "action_hygiene", "detrend", "dmdc", "lle", "lyapunov_eq", "regime", "windowed",
    "DetrendResult", "RegimeDetrender",
]
