import pytest
import numpy as np
from traits_audit.base import AuditCategory, AuditCheck, AuditResult
from traits_audit.hook import AuditHook
from traits_audit.pipeline import AuditPipeline


class _PassCheck(AuditCheck):
    @property
    def name(self): return "Pass"
    @property
    def category(self): return AuditCategory.UNKNOWN
    def run(self, history, **kwargs):
        return AuditResult(name=self.name, passed=True, category=self.category)


class _KwargsCapture(AuditCheck):
    def __init__(self): self.received = {}
    @property
    def name(self): return "Capture"
    @property
    def category(self): return AuditCategory.UNKNOWN
    def run(self, history, **kwargs):
        self.received = kwargs
        return AuditResult(name=self.name, passed=True, category=self.category)


def _make_hook(**kw):
    return AuditHook(AuditPipeline([_PassCheck()]), **kw)


# ── on_step ──────────────────────────────────────────────────────────────────

def test_on_step_accumulates_history():
    hook = _make_hook()
    hook.on_step(x=1)
    hook.on_step(x=2)
    assert len(hook.history) == 2
    assert hook.history[0]["x"] == 1
    assert hook.history[1]["x"] == 2


def test_on_step_injects_step_index():
    hook = _make_hook()
    hook.on_step(a=0)
    hook.on_step(a=1)
    assert hook.history[0]["_step"] == 0
    assert hook.history[1]["_step"] == 1


def test_history_is_a_copy():
    hook = _make_hook()
    hook.on_step(x=1)
    h = hook.history
    hook.on_step(x=2)
    assert len(h) == 1  # snapshot did not grow


# ── on_end ───────────────────────────────────────────────────────────────────

def test_on_end_returns_report():
    hook = _make_hook()
    hook.on_step(x=1)
    report = hook.on_end()
    assert report.passed


def test_report_property_raises_before_on_end():
    with pytest.raises(RuntimeError, match="on_end"):
        _make_hook().report


def test_on_end_kwargs_forwarded_to_checks():
    cap = _KwargsCapture()
    hook = AuditHook(AuditPipeline([cap]))
    hook.on_end(y_true=[1, 2], y_pred_mean=[1, 2])
    assert "y_true" in cap.received
    assert "y_pred_mean" in cap.received


# ── context manager ──────────────────────────────────────────────────────────

def test_context_manager_sets_report():
    hook = _make_hook()
    with hook:
        hook.on_step(x=1)
    assert hook.report.passed


def test_context_manager_skips_pipeline_on_exception():
    hook = _make_hook()
    with pytest.raises(ValueError):
        with hook:
            raise ValueError("boom")
    with pytest.raises(RuntimeError):
        _ = hook.report


# ── check_every ──────────────────────────────────────────────────────────────

def test_check_every_triggers_intermediate_reports():
    hook = _make_hook(check_every=3)
    for i in range(9):
        hook.on_step(x=i)
    assert len(hook.intermediate_reports) == 3


def test_check_every_not_triggered_between_multiples():
    hook = _make_hook(check_every=5)
    for i in range(4):
        hook.on_step(x=i)
    assert len(hook.intermediate_reports) == 0


def test_check_every_reports_are_audit_reports():
    hook = _make_hook(check_every=2)
    for i in range(6):
        hook.on_step(x=i)
    from traits_audit.base import AuditReport
    assert all(isinstance(r, AuditReport) for r in hook.intermediate_reports)


# ── reset ────────────────────────────────────────────────────────────────────

def test_reset_clears_history_and_reports():
    hook = _make_hook()
    hook.on_step(x=1)
    hook.on_end()
    hook.reset()
    assert len(hook.history) == 0
    assert len(hook.intermediate_reports) == 0
    with pytest.raises(RuntimeError):
        _ = hook.report


# ── latest_uncertainty_vector ────────────────────────────────────────────────

def test_latest_uncertainty_vector_is_none_when_empty():
    assert _make_hook().latest_uncertainty_vector is None


def test_latest_uncertainty_vector_returns_last_vector():
    hook = _make_hook()
    hook.on_step(uncertainty_vector=[0.1, 0.2])
    hook.on_step(uncertainty_vector=[0.3, 0.4])
    np.testing.assert_array_equal(hook.latest_uncertainty_vector, [0.3, 0.4])


def test_latest_uncertainty_vector_skips_steps_without_it():
    hook = _make_hook()
    hook.on_step(x=1)                            # no vector
    hook.on_step(uncertainty_vector=[0.5, 0.6])
    hook.on_step(x=2)                            # no vector again
    np.testing.assert_array_equal(hook.latest_uncertainty_vector, [0.5, 0.6])


def test_latest_uncertainty_vector_returns_none_if_never_set():
    hook = _make_hook()
    hook.on_step(x=1)
    hook.on_step(x=2)
    assert hook.latest_uncertainty_vector is None


# ── logger duck-type protocol ────────────────────────────────────────────────

class _MockLogger:
    def __init__(self):
        self.steps = []
        self.reports = []
    def log_step(self, idx, **kwargs):
        self.steps.append((idx, dict(kwargs)))
    def log_report(self, report, step, tag):
        self.reports.append((step, tag))


def test_logger_log_step_called_per_step():
    logger = _MockLogger()
    hook = AuditHook(AuditPipeline([_PassCheck()]), logger=logger)
    hook.on_step(x=1)
    hook.on_step(x=2)
    assert len(logger.steps) == 2
    assert logger.steps[0] == (0, {"x": 1})
    assert logger.steps[1] == (1, {"x": 2})


def test_logger_log_report_called_on_end():
    logger = _MockLogger()
    hook = AuditHook(AuditPipeline([_PassCheck()]), logger=logger)
    hook.on_step(x=1)
    hook.on_end()
    assert len(logger.reports) == 1
    _, tag = logger.reports[0]
    assert tag == "final"


def test_logger_log_report_called_for_intermediate():
    logger = _MockLogger()
    hook = AuditHook(AuditPipeline([_PassCheck()]), check_every=2, logger=logger)
    for i in range(4):
        hook.on_step(x=i)
    intermediate_tags = [tag for _, tag in logger.reports]
    assert intermediate_tags.count("intermediate") == 2
