"""Backward-compatible alias for :mod:`traits_audit.stability.detrend`."""
import sys as _sys

from .stability import detrend as _impl

_sys.modules[__name__] = _impl
