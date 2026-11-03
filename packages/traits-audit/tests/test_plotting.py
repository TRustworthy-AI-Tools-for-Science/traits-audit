"""Tests for traits_audit.plotting.

The status tests guard a specific defect: a check with threshold=None
(report-only, e.g. CRPSCheck()) or a skipped check both set
AuditResult.passed=True by convention, and an earlier status function drew
both in the same "deeply passing" green as an evaluated, healthy result.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import traits_audit.checks as checks_module
from traits_audit import AuditPipeline
from traits_audit.base import AuditCategory, AuditReport, AuditResult
from traits_audit.checks import CalibrationErrorCheck, CRPSCheck, IntervalCoverageCheck
from traits_audit.checks import __all__ as ALL_CHECK_NAMES
from traits_audit.plotting import (
    CHECK_ABBREVIATIONS,
    filter_stage_reports,
    plot_audit_evolution,
    plot_calibration_curve,
    plot_check_grid,
    plot_uncertainty_evolution,
    result_status,
    split_trackable_checks,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _result(**kwargs):
    defaults = {"name": "X", "passed": True, "category": AuditCategory.UNKNOWN}
    defaults.update(kwargs)
    return AuditResult(**defaults)


def _history(n=30, seed=0, std_scale=1.0):
    rng = np.random.default_rng(seed)
    history = []
    for _ in range(n):
        sigma = rng.uniform(0.5, 1.5)
        mu = rng.normal()
        y = mu + rng.normal(0, sigma)
        history.append({"y_true": y, "y_pred_mean": mu, "y_pred_std": sigma * std_scale,
                        "uncertainty": sigma * std_scale})
    return history


# ── result_status ─────────────────────────────────────────────────────────────

def test_report_only_when_threshold_is_none():
    _, status = result_status(_result(value=13.47, threshold=None, passed=True))
    assert status == "report_only"


def test_skipped_when_message_starts_with_skipped():
    r = _result(value=None, threshold=None, passed=True, message="Skipped — no data.")
    _, status = result_status(r)
    assert status == "skipped"


def test_report_only_is_distinct_from_deeply_passing():
    # An overconfident model's NLL (report-only, badly failing by any
    # reasonable bar) must not render like a deeply passing, evaluated check.
    report_only = _result(name="NLL", value=13.47, threshold=None, passed=True)
    deeply_passing = _result(name="CalibrationError", value=0.01, threshold=0.15, passed=True)
    _, status_a = result_status(report_only)
    intensity_b, status_b = result_status(deeply_passing)
    assert status_a == "report_only"
    assert status_b == "pass"
    assert intensity_b > 0.9


def test_pass_and_fail_produce_graded_intensity():
    deep_pass = _result(name="CalibrationError", value=0.0, threshold=0.15, passed=True)
    boundary = _result(name="CalibrationError", value=0.15, threshold=0.15, passed=True)
    deep_fail = _result(name="CalibrationError", value=0.9, threshold=0.15, passed=False)

    i_pass, s_pass = result_status(deep_pass)
    i_boundary, s_boundary = result_status(boundary)
    i_fail, s_fail = result_status(deep_fail)

    assert s_pass == "pass" and s_boundary == "pass" and s_fail == "fail"
    assert i_pass > i_boundary > i_fail
    assert i_boundary == pytest.approx(0.5, abs=0.05)


def test_variance_alignment_reads_tolerance_from_details():
    # 0.3 from the threshold is failing-side under the check's own tolerance (0.2).
    r = _result(name="VarianceAlignment", value=1.3, threshold=1.0, passed=False,
                details={"tolerance": 0.2})
    intensity, status = result_status(r)
    assert status == "fail"
    assert intensity < 0.5


def test_interval_coverage_tuple_threshold():
    r = _result(name="IntervalCoverage", value=0.70, threshold=(0.533, 0.833), passed=True)
    intensity, status = result_status(r)
    assert status == "pass"
    assert intensity > 0.5


def test_unknown_check_name_falls_back_to_binary():
    intensity, status = result_status(_result(name="SomeFutureCheck", value=1.0, threshold=0.5))
    assert status == "pass"
    assert intensity == 1.0


def test_every_shipped_check_has_an_abbreviation():
    missing = []
    for cls_name in ALL_CHECK_NAMES:
        cls = getattr(checks_module, cls_name)
        try:
            inst = cls()
        except TypeError:
            continue  # constructor needs required args
        if inst.name not in CHECK_ABBREVIATIONS:
            missing.append(inst.name)
    assert missing == [], f"Checks missing from CHECK_ABBREVIATIONS: {missing}"


# ── split_trackable_checks / filter_stage_reports ────────────────────────────

def _stages():
    skipped = _result(name="Holdout", value=None, threshold=0.1, message="Skipped — no test set.")
    holdout = _result(name="Holdout", value=0.05, threshold=0.1)
    ce = _result(name="CalibrationError", value=0.05, threshold=0.15)
    return [
        ("step 10", AuditReport(results=[ce, skipped])),
        ("step 20", AuditReport(results=[ce, skipped])),
        ("final", AuditReport(results=[ce, holdout])),
    ]


def test_split_trackable_checks_separates_final_only():
    assert split_trackable_checks(_stages()) == (["CalibrationError"], ["Holdout"])


def test_split_with_single_stage_treats_everything_as_trackable():
    assert split_trackable_checks(_stages()[-1:]) == (["CalibrationError", "Holdout"], [])


def test_filter_stage_reports_keeps_only_named_checks_in_order():
    filtered = filter_stage_reports(_stages(), ["Holdout"])
    assert [label for label, _ in filtered] == ["step 10", "step 20", "final"]
    assert all([r.name for r in rep.results] == ["Holdout"] for _, rep in filtered)


# ── Figures ───────────────────────────────────────────────────────────────────

def test_plot_check_grid_draws_verdicts_and_marks_non_verdicts():
    pass_result = _result(name="CalibrationError", value=0.01, threshold=0.15)
    report_only = _result(name="CRPS", value=13.47, threshold=None)
    skipped = _result(name="PITUniformity", value=None, threshold=None,
                      message="Skipped — y_true not available.")
    fig = plot_check_grid([("final", AuditReport(results=[pass_result, report_only, skipped]))],
                          title="run")
    ax = fig.axes[0]
    z = ax.images[0].get_array()
    # rows: pass, report-only, skipped → only the verdict row is filled
    assert not z.mask[0, 0] and z.mask[1, 0] and z.mask[2, 0]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["CE", "CRPS", "PITU"]
    assert "13.47" in [t.get_text() for t in ax.texts]
    legend = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend == ["Report-only (no threshold)", "Skipped (no data)"]


def test_plot_check_grid_marks_checks_absent_from_a_stage():
    final = AuditReport(results=[
        _result(name="CalibrationError", value=0.05, threshold=0.15),
        _result(name="NewOnlyAtEnd", value=1.0, threshold=0.5),
    ])
    fig = plot_check_grid([_stages()[0], ("final", final)])
    z = fig.axes[0].images[0].get_array()
    names = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    row = names.index("NewOnlyAtEnd")
    assert z.mask[row, 0] and not z.mask[row, 1]


def test_plot_check_grid_from_a_real_hook_run():
    from traits_audit import AuditHook
    hook = AuditHook(AuditPipeline([CalibrationErrorCheck(), CRPSCheck()]), check_every=10)
    for step in _history():
        hook.on_step(**step)
    report = hook.on_end()
    stages = [(f"step {10 * (i + 1)}", r) for i, r in enumerate(hook.intermediate_reports)]
    fig = plot_check_grid([*stages, ("final", report)])
    assert fig.axes[0].images[0].get_array().shape == (2, len(stages) + 1)


def test_plot_uncertainty_evolution_draws_one_point_per_step():
    fig = plot_uncertainty_evolution(np.linspace(1, 0.2, 15), title="GP")
    line = fig.axes[0].lines[0]
    assert len(line.get_xdata()) == 15
    assert fig.axes[0].get_title() == "GP"


def test_plot_uncertainty_evolution_draws_into_given_axes():
    fig, ax = plt.subplots()
    assert plot_uncertainty_evolution([1.0, 0.5], ax=ax) is fig
    assert len(ax.lines) == 1


def test_plot_audit_evolution_one_panel_per_check():
    pipeline = AuditPipeline([CalibrationErrorCheck(threshold=0.1),
                              IntervalCoverageCheck(expected_coverage=0.683, tolerance=0.1)])
    fig = plot_audit_evolution(pipeline, _history(), snapshot_every=10)
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert [ax.get_title() for ax in visible] == ["CalibrationError", "IntervalCoverage"]
    # IntervalCoverage draws both band edges
    assert len(visible[1].lines) == 3
    assert fig._suptitle.get_text() == "Audit checks over loop steps"


def test_plot_audit_evolution_returns_none_for_short_history():
    assert plot_audit_evolution(AuditPipeline([CalibrationErrorCheck()]), _history(3)) is None


def test_plot_calibration_curve_from_check_result():
    result = CalibrationErrorCheck().run(_history(60, std_scale=0.3))
    fig = plot_calibration_curve(result)
    assert fig is not None
    assert any(t.get_text().startswith("CE = ") for t in fig.axes[0].texts)


def test_plot_calibration_curve_returns_none_without_details():
    assert plot_calibration_curve(_result(name="CalibrationError", details={})) is None
