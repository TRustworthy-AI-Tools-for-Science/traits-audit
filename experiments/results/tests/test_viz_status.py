"""
Tests for the paper's Plotly check grid in results.viz.

The status logic it draws from (traits_audit.plotting.result_status) is
tested in the library; this checks the Plotly rendering of the report-only
and skipped states.
"""
import numpy as np
import pytest
from results.viz import _fig_check_grid
from traits_audit.base import AuditCategory, AuditResult


def _result(**kwargs):
    defaults = {"name": "X", "passed": True, "category": AuditCategory.UNKNOWN}
    defaults.update(kwargs)
    return AuditResult(**defaults)


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
