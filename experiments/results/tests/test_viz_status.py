"""
Tests for the paper's check-grid figures in results.viz.

The grid itself is traits_audit.plotting.plot_check_grid, tested in the
library; this checks how results.viz splits and saves it.
"""
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from results.viz import check_grid_figures, save_check_grid_figures
from traits_audit.base import AuditCategory, AuditReport, AuditResult


def _result(name, value, threshold=None, message=""):
    return AuditResult(name=name, passed=True, category=AuditCategory.UNKNOWN,
                       value=value, threshold=threshold, message=message)


def _stage_reports():
    """CalibrationError reports at every stage; ENCE only in the final report."""
    skipped_ence = _result("ENCE", None, message="Skipped — needs the held-out set.")
    stages = [
        (f"step {k}", AuditReport(results=[_result("CalibrationError", 0.1 / k, 0.15), skipped_ence]))
        for k in (1, 2)
    ]
    stages.append(("final", AuditReport(results=[_result("CalibrationError", 0.02, 0.15),
                                                 _result("ENCE", 0.3, 0.5)])))
    return stages


def test_final_only_checks_get_their_own_grid():
    fig_trackable, fig_final_only = check_grid_figures(_stage_reports(), "run")
    assert isinstance(fig_trackable, Figure) and isinstance(fig_final_only, Figure)
    trackable_rows = [t.get_text() for t in fig_trackable.axes[0].get_yticklabels()]
    final_rows = [t.get_text() for t in fig_final_only.axes[0].get_yticklabels()]
    assert trackable_rows == ["CE"] and final_rows == ["ENCE"]
    plt.close("all")


def test_no_final_only_grid_when_every_check_is_trackable():
    stages = [(label, AuditReport(results=[r for r in rep.results if r.name == "CalibrationError"]))
              for label, rep in _stage_reports()]
    fig_trackable, fig_final_only = check_grid_figures(stages, "run")
    assert isinstance(fig_trackable, Figure) and fig_final_only is None
    plt.close("all")


def test_save_writes_png_files(tmp_path):
    paths = save_check_grid_figures(_stage_reports(), "run", tmp_path, "grid.png", "grid_final.png")
    assert [p.name for p in paths] == ["grid.png", "grid_final.png"]
    for path in paths:
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
