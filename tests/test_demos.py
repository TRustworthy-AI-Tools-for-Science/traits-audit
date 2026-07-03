"""Smoke tests for the four demo entry points.

Each test runs the corresponding main() with the minimum viable arguments
(2–3 iterations, smallest seed data) to verify the demo completes without
error and produces the expected output files.

mlflow is stubbed out via the mlflow_stub fixture so the tests run in any
environment — no MLflow installation required.  Demos that need optional
extras (pybamm, sdl/ax) are skipped when those packages are not installed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _mlflow_uri(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'test.db'}"


# ── ta-demo (calibration scenarios) ──────────────────────────────────────────

def test_demo_smoke(tmp_path, monkeypatch, mlflow_stub):
    from traits_audit._example import main

    monkeypatch.setattr(
        sys, "argv",
        [
            "ta-demo",
            "--steps", "3",
            "--check-every", "1",
            "--scenarios", "well_calibrated",
            "--mlflow-uri", _mlflow_uri(tmp_path),
        ],
    )
    monkeypatch.chdir(tmp_path)
    main()

    assert (tmp_path / "_results/cal_demo").exists()


# ── ta-camd-demo (materials discovery) ───────────────────────────────────────

def test_camd_demo_smoke(tmp_path, monkeypatch, mlflow_stub):
    pytest.importorskip("sklearn")
    pytest.importorskip("pandas")

    from traits_audit._camd_demo import main

    monkeypatch.setattr(
        sys, "argv",
        [
            "ta-camd-demo",
            "--n-seed", "10",
            "--n-iter", "2",
            "--n-query", "2",
            "--check-every", "1",
            "--out-dir", str(tmp_path / "camd"),
            "--mlflow-uri", _mlflow_uri(tmp_path),
        ],
    )
    monkeypatch.chdir(tmp_path)
    main()

    assert (tmp_path / "camd").exists()


# ── ta-sdl-demo (self-driving lab) ────────────────────────────────────────────

def test_sdl_demo_smoke(tmp_path, monkeypatch, mlflow_stub):
    pytest.importorskip("self_driving_lab_demo", exc_type=(ModuleNotFoundError, ImportError))
    pytest.importorskip("ax", exc_type=(ModuleNotFoundError, ImportError))

    from traits_audit._sdl_demo import main

    monkeypatch.setattr(
        sys, "argv",
        [
            "ta-sdl-demo",
            "--n-init", "3",
            "--n-iter", "2",
            "--check-every", "1",
            "--out-dir", str(tmp_path / "sdl"),
            "--mlflow-uri", _mlflow_uri(tmp_path),
        ],
    )
    monkeypatch.chdir(tmp_path)
    main()

    assert (tmp_path / "sdl").exists()


# ── ta-pybamm-demo (battery simulation) ──────────────────────────────────────

def test_pybamm_demo_smoke(tmp_path, monkeypatch, mlflow_stub):
    pytest.importorskip("pybamm")
    pytest.importorskip("sklearn")

    from traits_audit._pybamm_demo import main

    monkeypatch.setattr(
        sys, "argv",
        [
            "ta-pybamm-demo",
            "--n-seed", "3",
            "--n-iter", "2",
            "--check-every", "1",
            "--out-dir", str(tmp_path / "pybamm"),
            "--mlflow-uri", _mlflow_uri(tmp_path),
        ],
    )
    monkeypatch.chdir(tmp_path)
    main()

    assert (tmp_path / "pybamm").exists()
