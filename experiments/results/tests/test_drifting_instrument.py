"""The forrester-drift benchmark: an instrument with its own state."""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from results.committee.problems import DriftingInstrumentForrester


def test_queries_heat_the_instrument_and_it_cools_between_them():
    problem = DriftingInstrumentForrester()
    rng = np.random.default_rng(0)
    for _ in range(20):
        problem.observe(np.array([[0.76]]), rng)
    hot = problem.temperature
    assert hot == pytest.approx(0.3 * 0.76 / (1 - 0.8), rel=0.05)    # steady state
    problem.observe(np.array([[0.0]]), rng)
    assert problem.temperature == pytest.approx(0.8 * hot)


def test_measuring_at_rest_leaves_the_instrument_cold():
    problem = DriftingInstrumentForrester()
    problem.observe_at_rest(np.linspace(0, 1, 50)[:, None], np.random.default_rng(1))
    assert problem.temperature == 0.0


def test_a_hot_instrument_reads_higher_and_noisier():
    x = np.full((4000, 1), 0.5)
    cold = DriftingInstrumentForrester(heating=0.0, cooling=1.0)
    hot = DriftingInstrumentForrester(heating=0.0, cooling=1.0)
    hot.temperature = 1.0
    cold_readings = cold.observe(x, np.random.default_rng(2))[:, 0]
    hot_readings = hot.observe(x, np.random.default_rng(2))[:, 0]
    assert hot_readings.mean() - cold_readings.mean() == pytest.approx(1.0, abs=0.1)
    assert hot_readings.std() / cold_readings.std() == pytest.approx(5.0, rel=0.1)
    assert np.allclose(cold.clean(x), hot.clean(x))


@pytest.mark.filterwarnings("ignore")
def test_steering_runs_with_the_temperature_in_the_state(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "steer_state_space.py"
    spec = importlib.util.spec_from_file_location("steer_state_space", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(["--problem", "forrester-drift", "--seeds", "1", "--steps", "16", "--n-init", "9",
                 "--policies", "lcb", "state-space", "audit-state-space", "--out", str(tmp_path)])
    with open(tmp_path / "steps.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    temperatures = [float(row["instrument"]) for row in rows]
    assert min(temperatures) >= 0.0 and max(temperatures) > 0.0
    assert module.state_size(module.make_problem("forrester-drift")) == 3


def test_wear_accumulates_with_use_and_never_recovers():
    from results.committee.problems import WearingInstrumentForrester

    light, heavy = WearingInstrumentForrester(), WearingInstrumentForrester()
    rng = np.random.default_rng(3)
    for _ in range(100):
        light.observe(np.array([[0.2]]), rng)
        heavy.observe(np.array([[0.76]]), rng)
    assert heavy.temperature == pytest.approx(100 * 0.015 * 0.76)
    assert heavy.temperature > light.temperature
    damage = heavy.temperature
    heavy.observe(np.array([[0.0]]), rng)
    assert heavy.temperature == damage                                  # a zero-dose query adds nothing, removes nothing
