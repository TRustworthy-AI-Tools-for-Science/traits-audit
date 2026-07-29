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
            "--scenarios", "perfectly_calibrated",
            "--mlflow-uri", _mlflow_uri(tmp_path),
        ],
    )
    monkeypatch.chdir(tmp_path)
    main()

    assert (tmp_path / "_results/cal_demo").exists()


def test_demo1_scenario_verdicts(tmp_path, monkeypatch, mlflow_stub):
    """Primary acceptance test for the empirically-derived threshold table
    (see _example.py's module docstring and CHANGELOG for the full table):
    the gold standard passes the core calibration/coverage/scoring checks
    cleanly, and each pathological scenario fails the checks it was
    specifically designed to trip. Regression-guards the threshold
    calibration in _build_pipeline() -- if this starts failing, either the
    surrogate/oracle/acquisition changed in a way that broke the designed
    separation, or a check's default threshold changed underneath it.

    Runs the full default step count (slower than the other smoke tests,
    ~40s) because the separation was validated empirically at this scale;
    fewer steps do not reliably reproduce it.
    """
    from traits_audit._example import _SCENARIOS, _replicate_locations, _run_scenario

    monkeypatch.chdir(tmp_path)
    locations = _replicate_locations()
    reports = {}
    for config in _SCENARIOS:
        report, *_ = _run_scenario(
            config, steps=80, check_every=20, seed=10,
            mlflow_uri=f"sqlite:///{tmp_path / 'demo1_verdicts.db'}",
            experiment_name="test_demo1_verdicts",
            replicate_locations=locations,
        )
        reports[config.name] = report

    def result(scenario: str, check: str):
        by_name = {r.name: r for r in reports[scenario].results}
        assert check in by_name, f"{check} missing from {scenario}'s report"
        return by_name[check]

    # Gold standard: the core calibration/coverage/scoring checks pass cleanly.
    for check in ("CalibrationError", "IntervalCoverage", "VarianceAlignment",
                  "NegativeLogLikelihood", "CRPS", "IntervalScore", "SignedBias"):
        r = result("perfectly_calibrated", check)
        assert r.passed, f"gold standard should pass {check}, got {r.message}"

    # Overconfident: intervals too narrow -> fails magnitude-sensitive checks.
    for check in ("CalibrationError", "IntervalCoverage", "VarianceAlignment",
                  "NegativeLogLikelihood", "SignedBias"):
        r = result("overconfident", check)
        assert not r.passed, f"overconfident should fail {check}, got {r.message}"

    # Underconfident: intervals too wide -> fails calibration/coverage/alignment,
    # but NOT NLL/CRPS (over-wide intervals are not heavily penalised there).
    for check in ("CalibrationError", "IntervalCoverage", "VarianceAlignment"):
        r = result("underconfident", check)
        assert not r.passed, f"underconfident should fail {check}, got {r.message}"

    # Misspecified: wrong model class -> fails calibration/coverage.
    for check in ("CalibrationError", "IntervalCoverage"):
        r = result("misspecified", check)
        assert not r.passed, f"misspecified should fail {check}, got {r.message}"

    # Every scenario's pipeline is internally paired (Lyapunov/DMDc,
    # EID/ResidualPersistenceHalfLife) -- validate_config() should be silent.
    for name, report in reports.items():
        warnings = report.metadata.get("pairing_warnings") or []
        assert warnings == [], f"{name} has unexpected pairing warnings: {warnings}"


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
    from traits_audit._sdl_demo import _ensure_sdl_importable
    try:
        _ensure_sdl_importable()
    except ImportError:
        pass  # let importorskip below report the real skip reason, if any
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
