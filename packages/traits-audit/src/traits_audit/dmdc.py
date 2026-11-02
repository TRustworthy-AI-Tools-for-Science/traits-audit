"""Backward-compatible alias for :mod:`traits_audit.stability.dmdc`."""
import sys as _sys

from .stability import dmdc as _impl

_sys.modules[__name__] = _impl
