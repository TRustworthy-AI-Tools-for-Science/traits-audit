"""
Tests for the report-only/skipped status channel in _viz.py.

These guard the specific defect an evaluation of the demos surfaced: a
check with threshold=None (report-only, e.g. CRPSCheck()) or a genuinely
skipped check both set AuditResult.passed=True by convention, and the old
_result_intensity collapsed both into the same maximum-saturation "deeply
passing" green as an evaluated, healthy result.
"""
import numpy as np
import pytest

import traits_audit.checks as checks_module
from traits_audit._viz import _CHECK_ABBREV, _fig_check_grid, _result_status
from traits_audit.base import AuditCategory, AuditResult
from traits_audit.checks import __all__ as ALL_CHECK_NAMES


def _result(**kwargs):
    defaults = {"name": "X", "passed": True, "category": AuditCategory.UNKNOWN}
    defaults.update(kwargs)
    return AuditResult(**defaults)


# ── _result_status ───────────────────────────────────────────────────────────

def test_report_only_when_threshold_is_none():
    r = _result(value=13.47, threshold=None, passed=True)
    _intensity, status = _result_status(r)
    assert status == "report_only"


def test_skipped_when_message_starts_with_skipped():
    r = _result(value=None, threshold=None, passed=True, message="Skipped — no data.")
    _intensity, status = _result_status(r)
    assert status == "skipped"


def test_report_only_is_distinct_from_deeply_passing():
    # The core regression: an overconfident model's NLL (report-only, badly
    # failing by any reasonable bar) must not render identically to a
    # genuinely deeply-passing, evaluated check.
    report_only = _result(name="NLL", value=13.47, threshold=None, passed=True)
    deeply_passing = _result(
        name="CalibrationError", value=0.01, threshold=0.15, passed=True,
    )
    _, status_a = _result_status(report_only)
    intensity_b, status_b = _result_status(deeply_passing)
    assert status_a == "report_only"
    assert status_b == "pass"
    assert intensity_b > 0.9  # genuinely deep pass gets a high intensity
    assert status_a != status_b


def test_pass_and_fail_still_produce_graded_intensity():
    deep_pass = _result(name="CalibrationError", value=0.0, threshold=0.15, passed=True)
    boundary = _result(name="CalibrationError", value=0.15, threshold=0.15, passed=True)
    deep_fail = _result(name="CalibrationError", value=0.9, threshold=0.15, passed=False)

    i_pass, s_pass = _result_status(deep_pass)
    i_boundary, s_boundary = _result_status(boundary)
    i_fail, s_fail = _result_status(deep_fail)

    assert s_pass == "pass" and s_boundary == "pass" and s_fail == "fail"
    assert i_pass > i_boundary > i_fail
    assert i_boundary == pytest.approx(0.5, abs=0.05)


def test_variance_alignment_reads_tolerance_from_details_not_hardcoded():
    # tolerance=0.2 is much tighter than the old hardcoded 0.50 -- a value
    # 0.3 away from the threshold should read as failing-side under the
    # check's OWN tolerance, not the previously-hardcoded one.
    r = _result(
        name="VarianceAlignment", value=1.3, threshold=1.0, passed=False,
        details={"tolerance": 0.2},
    )
    intensity, status = _result_status(r)
    assert status == "fail"
    assert intensity < 0.5


def test_interval_coverage_tuple_threshold_still_works():
    r = _result(
        name="IntervalCoverage", value=0.70, threshold=(0.533, 0.833), passed=True,
    )
    intensity, status = _result_status(r)
    assert status == "pass"
    assert intensity > 0.5


def test_unknown_check_name_falls_back_to_binary():
    r = _result(name="SomeFutureCheck", value=1.0, threshold=0.5, passed=True)
    intensity, status = _result_status(r)
    assert status == "pass"
    assert intensity == 1.0


# ── _CHECK_ABBREV coverage ───────────────────────────────────────────────────

def test_every_shipped_check_has_an_abbreviation():
    missing = []
    for cls_name in ALL_CHECK_NAMES:
        cls = getattr(checks_module, cls_name)
        try:
            inst = cls()
        except TypeError:
            continue  # constructor needs required args; not expected here
        if inst.name not in _CHECK_ABBREV:
            missing.append(inst.name)
    assert missing == [], f"Checks missing from _CHECK_ABBREV: {missing}"


# ── _fig_check_grid renders the new states distinctly ───────────────────────

def test_fig_check_grid_distinguishes_report_only_from_pass():
    pytest.importorskip("plotly")
    pass_result = _result(name="CalibrationError", value=0.01, threshold=0.15, passed=True)
    report_only_result = _result(name="CRPS", value=13.47, threshold=None, passed=True)
    skipped_result = _result(
        name="PITUniformity", value=None, threshold=None, passed=True,
        message="Skipped — y_true not available.",
    )

    class _FakeReport:
        def __init__(self, results):
            self.results = results

    stage_reports = [("final", _FakeReport([pass_result, report_only_result, skipped_result]))]
    fig = _fig_check_grid(stage_reports, "test_run")
    assert fig is not None

    heatmap = fig.data[0]
    z = np.array(heatmap.z, dtype=float)
    # row order matches result order: pass, report_only, skipped
    assert np.isfinite(z[0]).all()
    assert np.isnan(z[1]).all()
    assert np.isnan(z[2]).all()
    # The heatmap's own on-cell text is blanked for non-verdict rows (their
    # value is drawn by the overlay traces instead, checked below).
    text = np.array(heatmap.text)
    assert text[0][0] != ""
    assert text[1][0] == ""
    assert text[2][0] == ""

    # Two overlay scatter traces (report_only, skipped) should be present,
    # each with one point carrying the cell's numeric value, plus a legend
    # entry each.
    scatter_traces = [t for t in fig.data if t.type == "scatter"]
    assert len(scatter_traces) == 2
    names = {t.name for t in scatter_traces}
    assert "Report-only (no threshold)" in names
    assert "Skipped (no data)" in names
    for t in scatter_traces:
        assert t.mode == "markers+text"
        if t.name == "Report-only (no threshold)":
            assert t.text == ("13.47",)
