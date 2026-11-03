"""Golden-output regression tests for the paper case studies.

Each case study runs at smoke size with a fixed seed. The test fingerprints the run:

- every ``AuditReport`` produced by ``AuditPipeline.run`` (check name, verdict,
  value, threshold, details), in order;
- every metric sent to MLflow (``log_metric`` / ``log_metrics`` /
  ``MlflowClient.log_metric``), in order;
- the sorted list of files the run wrote.

It then compares the fingerprint with ``tests/golden/<case>.json``. The
fingerprint does not depend on module layout, so it guards the refactor of
``viz.py`` and the case-study modules: those refactors must not change it.

Record (or deliberately re-record) the golden files with::

    uv run pytest experiments/results/tests/test_golden.py --update-golden

A case is skipped when its optional extras are not installed or when its golden
file has not been recorded yet. Record on a machine that has the extras: torch
has no wheels for Intel macOS, and ``camd`` needs ``pymatgen`` to build.
"""
from __future__ import annotations

import importlib
import json
import math
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import matplotlib
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

matplotlib.use("Agg")

GOLDEN_DIR = Path(__file__).parent / "golden"
# Optimised BLAS kernels are not bit-reproducible run to run (results can depend
# on memory alignment). GP hyperparameter fits amplify that to ~1e-8 relative
# (~2e-7 absolute on near-zero values), so compare with headroom above it. A
# real behaviour change shows up orders of magnitude larger.
RTOL, ATOL = 1e-5, 1e-6
MAX_REPORTED_DIFFS = 25


@dataclass(frozen=True)
class Case:
    module: str                  # module whose main() is the case study's CLI
    argv: tuple[str, ...]        # "{out}" is replaced with the run's output dir
    requires: tuple[str, ...]    # modules that must import for the paper code path


CASES: dict[str, Case] = {
    "cal": Case(
        "results.cal_demo",
        ("--steps", "12", "--check-every", "4", "--seed", "10"),
        (),
    ),
    "sdl": Case(
        "results.sdl_demo",
        ("--n-init", "3", "--n-iter", "4", "--check-every", "2", "--seed", "0", "--out-dir", "{out}"),
        ("torch", "botorch", "ax", "self_driving_lab_demo"),
    ),
    "mobo": Case(
        "results.mobo_demo",
        ("--n-init", "4", "--n-iter", "4", "--check-every", "2", "--seed", "0", "--out-dir", "{out}"),
        ("torch", "botorch"),
    ),
    "pybamm": Case(
        "results.pybamm_demo",
        ("--n-seed", "3", "--n-iter", "2", "--check-every", "1", "--seed", "0", "--out-dir", "{out}"),
        ("torch", "botorch", "pybamm"),
    ),
    "camd": Case(
        "results.camd_demo",
        ("--n-seed", "10", "--n-iter", "2", "--n-query", "2", "--check-every", "1", "--seed", "0",
         "--out-dir", "{out}"),
        ("camd", "pandas"),
    ),
}


# ── Fingerprinting ────────────────────────────────────────────────────────────

def _jsonable(x: Any) -> Any:
    """Convert to plain JSON types. Unknown objects become their type name, so
    reprs carrying memory addresses never leak into the fingerprint."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return x if math.isfinite(x) else repr(x)
    if x is None or isinstance(x, str):
        return x
    if isinstance(x, Enum):
        return _jsonable(x.value)
    return f"<{type(x).__name__}>"


def _report_fingerprint(report) -> dict:
    return {
        "metadata": _jsonable(report.metadata),
        "results": [
            {
                "name": r.name,
                "passed": bool(r.passed),
                "value": _jsonable(r.value),
                "threshold": _jsonable(r.threshold),
                "details": _jsonable(r.details),
            }
            for r in report.results
        ],
    }


def _recording_mlflow(metrics: list) -> MagicMock:
    """An mlflow stand-in that records metrics and otherwise does nothing."""
    mock = MagicMock(name="mlflow")
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=MagicMock())
    ctx.__exit__ = MagicMock(return_value=False)
    mock.start_run.return_value = ctx

    def log_metric(key, value, step=None, **_):
        metrics.append([key, _jsonable(value), step])

    def log_metrics(d, step=None, **_):
        for k in sorted(d):
            metrics.append([k, _jsonable(d[k]), step])

    def client_log_metric(_run_id, key, value, step=None, **_):
        metrics.append([key, _jsonable(value), step])

    mock.log_metric.side_effect = log_metric
    mock.log_metrics.side_effect = log_metrics
    mock.MlflowClient.return_value.log_metric.side_effect = client_log_metric
    return mock


def run_case(name: str, tmp_path: Path, monkeypatch) -> dict:
    case = CASES[name]
    for mod in case.requires:
        pytest.importorskip(mod, exc_type=ImportError)

    from traits_audit.pipeline import AuditPipeline

    reports: list[dict] = []
    metrics: list[list] = []
    original_run = AuditPipeline.run

    def recording_run(self, *args, **kwargs):
        report = original_run(self, *args, **kwargs)
        reports.append(_report_fingerprint(report))
        return report

    monkeypatch.setattr(AuditPipeline, "run", recording_run)
    mlflow = _recording_mlflow(metrics)
    for mod in ("mlflow", "mlflow.store", "mlflow.store.db", "mlflow.store.db.utils", "mlflow.tracking"):
        monkeypatch.setitem(sys.modules, mod, mlflow)

    out = tmp_path / "out"
    argv = [a.replace("{out}", str(out)) for a in case.argv]
    argv += ["--mlflow-uri", f"sqlite:///{tmp_path / 'mlflow.db'}"]
    monkeypatch.setattr(sys, "argv", [name, *argv])
    monkeypatch.chdir(tmp_path)

    # One BLAS/OpenMP thread removes thread-scheduling noise; see RTOL for the rest.
    with threadpool_limits(limits=1):
        importlib.import_module(case.module).main()

    files = sorted(
        str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*")
        if p.is_file() and p.suffix != ".db" and "mlruns" not in p.parts
    )
    return {"case": name, "argv": list(case.argv), "files": files,
            "reports": reports, "metrics": metrics}


# ── Comparison ────────────────────────────────────────────────────────────────

def _diff(expected: Any, actual: Any, path: str, out: list[str]) -> None:
    if len(out) >= MAX_REPORTED_DIFFS:
        return
    if isinstance(expected, dict) and isinstance(actual, dict):
        for k in sorted(set(expected) | set(actual)):
            if k not in actual:
                out.append(f"{path}.{k}: missing")
            elif k not in expected:
                out.append(f"{path}.{k}: unexpected (= {actual[k]!r:.80})")
            else:
                _diff(expected[k], actual[k], f"{path}.{k}", out)
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            out.append(f"{path}: length {len(actual)} != expected {len(expected)}")
        for i, (e, a) in enumerate(zip(expected, actual, strict=False)):
            _diff(e, a, f"{path}[{i}]", out)
    elif (isinstance(expected, float) or isinstance(actual, float)) and \
            isinstance(expected, (int, float)) and isinstance(actual, (int, float)) and \
            not isinstance(expected, bool) and not isinstance(actual, bool):
        if not math.isclose(expected, actual, rel_tol=RTOL, abs_tol=ATOL):
            out.append(f"{path}: {actual!r} != expected {expected!r}")
    elif expected != actual:
        out.append(f"{path}: {actual!r:.80} != expected {expected!r:.80}")


# ── Test ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", list(CASES))
def test_golden(name, tmp_path, monkeypatch, request):
    golden_file = GOLDEN_DIR / f"{name}.json"
    update = request.config.getoption("--update-golden")
    if not update and not golden_file.exists():
        pytest.skip(f"no golden recorded for {name!r}; run with --update-golden "
                    "on a machine with its extras")

    actual = run_case(name, tmp_path, monkeypatch)

    if update:
        GOLDEN_DIR.mkdir(exist_ok=True)
        golden_file.write_text(json.dumps(actual, indent=1) + "\n")
        return

    expected = json.loads(golden_file.read_text())
    diffs: list[str] = []
    _diff(expected, actual, name, diffs)
    assert not diffs, "golden mismatch:\n  " + "\n  ".join(diffs)
