"""The ``ta-demo`` quickstart must run on core dependencies and tell a
calibrated surrogate apart from a mis-scaled one."""
from traits_audit._quickstart import main, run


def _passed(report):
    return {r.name: r.passed for r in report.results}


def test_calibrated_surrogate_passes_calibration_checks():
    res = _passed(run(steps=60, seed=0, std_scale=1.0))
    assert res["CalibrationError"] and res["IntervalCoverage"]


def test_overconfident_surrogate_fails_calibration_checks():
    res = _passed(run(steps=60, seed=0, std_scale=0.3))
    assert not res["CalibrationError"] and not res["IntervalCoverage"]


def test_cli_prints_report(capsys):
    main(["--steps", "15", "--seed", "1"])
    assert "Audit Report" in capsys.readouterr().out
