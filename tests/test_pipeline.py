import json
import pytest
from traits_audit.base import AuditCategory, AuditCheck, AuditResult
from traits_audit.pipeline import AuditPipeline


class _PassCheck(AuditCheck):
    @property
    def name(self): return "Pass"
    @property
    def category(self): return AuditCategory.UNKNOWN
    def run(self, history, **kwargs):
        return AuditResult(name=self.name, passed=True, category=self.category)


class _FailCheck(AuditCheck):
    @property
    def name(self): return "Fail"
    @property
    def category(self): return AuditCategory.UNKNOWN
    def run(self, history, **kwargs):
        return AuditResult(name=self.name, passed=False, category=self.category)


class _KwargsCapture(AuditCheck):
    def __init__(self): self.received = {}
    @property
    def name(self): return "KwargsCapture"
    @property
    def category(self): return AuditCategory.UNKNOWN
    def run(self, history, **kwargs):
        self.received = kwargs
        return AuditResult(name=self.name, passed=True, category=self.category)


class _HistoryCapture(AuditCheck):
    def __init__(self): self.received = []
    @property
    def name(self): return "HistoryCapture"
    @property
    def category(self): return AuditCategory.UNKNOWN
    def run(self, history, **kwargs):
        self.received = history
        return AuditResult(name=self.name, passed=True, category=self.category)


# ── run() ────────────────────────────────────────────────────────────────────

def test_empty_pipeline_returns_empty_passed_report():
    report = AuditPipeline([]).run([])
    assert report.passed
    assert len(report.results) == 0


def test_all_checks_run_even_when_one_fails():
    pipeline = AuditPipeline([_FailCheck(), _PassCheck()])
    report = pipeline.run([])
    assert len(report.results) == 2
    assert report.n_failed == 1
    assert report.n_passed == 1


def test_kwargs_forwarded_to_every_check():
    cap = _KwargsCapture()
    AuditPipeline([cap]).run([], foo=[1, 2], bar="baz")
    assert "foo" in cap.received
    assert "bar" in cap.received


def test_history_forwarded_to_every_check():
    cap = _HistoryCapture()
    h = [{"_step": 0, "x": 1.0}, {"_step": 1, "x": 2.0}]
    AuditPipeline([cap]).run(h)
    assert cap.received == h


def test_metadata_stored_in_report():
    pipeline = AuditPipeline([_PassCheck()])
    report = pipeline.run([], metadata={"experiment": "smoke"})
    assert report.metadata["experiment"] == "smoke"


def test_metadata_defaults_to_empty_dict():
    report = AuditPipeline([_PassCheck()]).run([])
    assert report.metadata == {}


# ── save() ───────────────────────────────────────────────────────────────────

def test_save_creates_valid_json(tmp_path):
    pipeline = AuditPipeline([_PassCheck()])
    report = pipeline.run([])
    out = tmp_path / "report.json"
    pipeline.save(report, out, merge=False)
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["passed"] is True
    assert len(data["results"]) == 1


def test_save_creates_parent_dirs(tmp_path):
    pipeline = AuditPipeline([_PassCheck()])
    report = pipeline.run([])
    out = tmp_path / "nested" / "dir" / "report.json"
    pipeline.save(report, out, merge=False)
    assert out.exists()


def test_save_merge_deduplicates_by_name(tmp_path):
    pipeline = AuditPipeline([_PassCheck(), _FailCheck()])
    out = tmp_path / "report.json"
    pipeline.save(pipeline.run([]), out, merge=True)
    pipeline.save(pipeline.run([]), out, merge=True)  # second save same names
    data = json.loads(out.read_text())
    names = [r["name"] for r in data["results"]]
    assert len(names) == len(set(names))  # no duplicates
    assert set(names) == {"Pass", "Fail"}


def test_save_merge_false_overwrites(tmp_path):
    out = tmp_path / "report.json"
    AuditPipeline([_PassCheck()]).save(AuditPipeline([_PassCheck()]).run([]), out, merge=False)
    AuditPipeline([_FailCheck()]).save(AuditPipeline([_FailCheck()]).run([]), out, merge=False)
    data = json.loads(out.read_text())
    assert data["results"][0]["name"] == "Fail"


def test_save_accepts_string_path(tmp_path):
    pipeline = AuditPipeline([_PassCheck()])
    out = str(tmp_path / "report.json")
    pipeline.save(pipeline.run([]), out, merge=False)
    assert json.loads(open(out).read())["passed"] is True
