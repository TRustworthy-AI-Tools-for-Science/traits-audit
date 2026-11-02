import pytest
from traits_audit.base import AuditCategory, AuditCheck, AuditReport, AuditResult


class _PassCheck(AuditCheck):
    @property
    def name(self): return "AlwaysPass"
    @property
    def category(self): return AuditCategory.UNKNOWN
    def run(self, history, **kwargs):
        return AuditResult(name=self.name, passed=True, category=self.category)


class _FailCheck(AuditCheck):
    @property
    def name(self): return "AlwaysFail"
    @property
    def category(self): return AuditCategory.UNKNOWN
    def run(self, history, **kwargs):
        return AuditResult(
            name=self.name, passed=False, category=self.category,
            value=0.9, threshold=0.5,
        )


# ── AuditResult ──────────────────────────────────────────────────────────────

def test_audit_result_defaults():
    r = AuditResult(name="foo", passed=True, category=AuditCategory.EPISTEMIC)
    assert r.value is None
    assert r.threshold is None
    assert r.message == ""
    assert r.details == {}


def test_audit_result_stores_fields():
    r = AuditResult(
        name="bar", passed=False, category=AuditCategory.ALEATORIC_MODEL,
        value=0.3, threshold=0.1, message="too high", details={"k": 1},
    )
    assert r.name == "bar"
    assert not r.passed
    assert r.value == 0.3
    assert r.threshold == 0.1
    assert r.message == "too high"
    assert r.details == {"k": 1}


# ── AuditReport ──────────────────────────────────────────────────────────────

def test_report_all_pass():
    report = AuditReport(results=[
        AuditResult("a", True, AuditCategory.EPISTEMIC),
        AuditResult("b", True, AuditCategory.ALEATORIC_MODEL),
    ])
    assert report.passed
    assert report.n_passed == 2
    assert report.n_failed == 0


def test_report_any_fail():
    report = AuditReport(results=[
        AuditResult("a", True, AuditCategory.EPISTEMIC),
        AuditResult("b", False, AuditCategory.ALEATORIC_MODEL),
    ])
    assert not report.passed
    assert report.n_passed == 1
    assert report.n_failed == 1


def test_report_empty():
    report = AuditReport()
    assert report.passed
    assert report.n_passed == 0
    assert report.n_failed == 0


def test_report_summary_contains_counts():
    report = AuditReport(results=[
        AuditResult("a", True, AuditCategory.EPISTEMIC, value=0.42),
        AuditResult("b", False, AuditCategory.UNKNOWN),
    ])
    summary = report.summary()
    assert "1/2" in summary
    assert "PASS" in summary
    assert "FAIL" in summary
    assert "0.4200" in summary


def test_report_to_dict_structure():
    report = AuditReport(
        results=[AuditResult("x", True, AuditCategory.EPISTEMIC, value=0.1, threshold=0.2)],
        metadata={"run": 1},
    )
    d = report.to_dict()
    assert d["passed"] is True
    assert d["n_passed"] == 1
    assert d["n_failed"] == 0
    assert d["metadata"] == {"run": 1}
    r = d["results"][0]
    assert r["name"] == "x"
    assert r["category"] == "epistemic"
    assert r["value"] == pytest.approx(0.1)
    assert r["threshold"] == pytest.approx(0.2)


# ── AuditCheck ───────────────────────────────────────────────────────────────

def test_check_pass():
    result = _PassCheck().run([], extra="ignored")
    assert result.passed
    assert result.name == "AlwaysPass"
    assert result.category == AuditCategory.UNKNOWN


def test_check_fail():
    result = _FailCheck().run([])
    assert not result.passed
    assert result.value == 0.9


def test_check_receives_history():
    class _HistoryCapture(AuditCheck):
        captured = None
        @property
        def name(self): return "Capture"
        @property
        def category(self): return AuditCategory.UNKNOWN
        def run(self, history, **kwargs):
            _HistoryCapture.captured = history
            return AuditResult(name=self.name, passed=True, category=self.category)

    history = [{"step": 0, "x": 1.0}, {"step": 1, "x": 2.0}]
    _HistoryCapture().run(history)
    assert _HistoryCapture.captured == history


def test_abstract_check_cannot_be_instantiated():
    with pytest.raises(TypeError):
        AuditCheck()  # type: ignore[abstract]
