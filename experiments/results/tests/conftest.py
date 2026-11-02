"""Shared pytest fixtures for the traits-audit test suite."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def mlflow_stub(monkeypatch):
    """Stub mlflow so demos run without a real MLflow installation.

    All mlflow imports in demo code are deferred (inside functions), so
    patching sys.modules before calling main() correctly intercepts them.
    """
    if "mlflow" in sys.modules:
        yield sys.modules["mlflow"]
        return

    mock = MagicMock(name="mlflow")
    # start_run() must work as a context manager
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=MagicMock())
    ctx.__exit__ = MagicMock(return_value=False)
    mock.start_run.return_value = ctx

    stubs = {
        "mlflow": mock,
        "mlflow.store": MagicMock(),
        "mlflow.store.db": MagicMock(),
        "mlflow.store.db.utils": MagicMock(),
        "mlflow.tracking": MagicMock(),
    }
    for name, mod in stubs.items():
        monkeypatch.setitem(sys.modules, name, mod)

    yield mock
