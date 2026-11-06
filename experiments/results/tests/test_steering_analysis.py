"""analyze_steering.py (P1–P7) and aggregate_steering.py."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benjamini_hochberg_matches_a_hand_worked_example():
    aggregate = _load("aggregate_steering")
    q = aggregate.benjamini_hochberg([0.01, 0.04, 0.03, np.nan, 0.20])
    # sorted p: 0.01, 0.03, 0.04, 0.20 (m = 4) -> 0.04, 0.0533, 0.0533, 0.20
    assert np.allclose(q[[0, 2, 1, 4]], [0.04, 0.16 / 3, 0.16 / 3, 0.20])
    assert np.isnan(q[3])


def test_relative_recovery_is_the_share_of_the_gap_closed():
    analyze = _load("analyze_steering")
    target, scale = np.array([1.0]), np.array([1.0])
    raw = np.array([[5.0], [4.0], [3.0], [3.0]])            # distance 4 -> 2: half the gap closed
    assert analyze.relative_recovery(raw, target, scale, window=2) == pytest.approx(0.5)
    worse = np.array([[2.0], [3.0], [4.0], [4.0]])          # distance 1 -> 3: lost ground
    assert analyze.relative_recovery(worse, target, scale, window=2) == pytest.approx(-2.0)
    assert np.isnan(analyze.relative_recovery(np.array([[0.5], [0.4]]), target, scale, window=1))


def test_time_to_target_is_the_first_step_at_or_below_it():
    analyze = _load("analyze_steering")
    steps = np.arange(25, 30)
    score = np.array([0.9, 0.5, 0.2, 0.2, 0.1])
    assert analyze.time_to_target(score, steps, target=0.2) == 27
    assert analyze.time_to_target(score, steps, target=0.01) == 30          # never: one past the end


def test_recovery_needs_three_steps_at_or_below_target():
    analyze = _load("analyze_steering")
    steps = np.arange(10, 20)
    raw = np.array([[3.0], [2.0], [0.5], [2.0], [0.9], [0.8], [0.7], [0.5], [0.4], [0.3]])
    assert analyze.recovery_step(raw, steps, target=np.array([1.0]), scale=np.array([1.0])) == 14
    assert analyze.recovery_step(raw + 5, steps, target=np.array([1.0]), scale=np.array([1.0])) is None


@pytest.mark.filterwarnings("ignore")
def test_analysis_and_aggregation_end_to_end(tmp_path, capsys):
    steer = _load("steer_state_space")
    analyze = _load("analyze_steering")
    aggregate = _load("aggregate_steering")
    policies = ["random", "lcb", "state-space", "audit-state-space"]
    for condition in ("calibrated", "overconfident"):
        steer.main(["--problem", "forrester", "--condition", condition, "--seeds", "5", "--steps", "40",
                    "--n-init", "9", "--policies", *policies, "--out", str(tmp_path / condition)])

    analyze.main(["--in", str(tmp_path / "calibrated"), "--window", "3"])
    analyze.main(["--in", str(tmp_path / "overconfident"), "--reference", str(tmp_path / "calibrated"),
                  "--window", "3"])

    summary = json.loads((tmp_path / "overconfident" / "steering_summary.json").read_text())
    assert set(summary) == set(policies)
    for entry in summary.values():
        assert 0.0 <= entry["failure_rate"] <= 1.0
        assert 0.0 <= entry["stuck_rate"] <= 1.0
        assert 0.0 <= entry["recovered_share"] <= 1.0
        assert 0.0 <= entry["settled_share"] <= 1.0
    assert summary["state-space"]["dH_minus_state_space"] == 0.0
    for entry in summary.values():
        assert entry["cumulative_regret_median"] >= entry["final_score_median"]
    run = json.loads((tmp_path / "overconfident" / "steering_run.json").read_text())
    assert run["condition"] == "overconfident"

    aggregate.main([str(tmp_path)])
    with open(tmp_path / "steering_aggregate.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert {row["cell"] for row in rows} == {"calibrated", "overconfident"}
    assert "P6 gap closed vs state-space" in {row["prediction"] for row in rows}
    assert "with q < 0.05" in capsys.readouterr().out
