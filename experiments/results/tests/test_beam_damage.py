"""The branin-currin-beam benchmark: a 2-D sample damaged by the beam that measures it."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from results.committee.problems import BeamDamageBraninCurrin

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _noise_free(problem):
    problem.__dict__["noise_std"] = np.zeros(2)       # overrides the cached noise level
    return problem


def test_dose_follows_a_gaussian_beam_footprint():
    problem = BeamDamageBraninCurrin(beam_fwhm=0.1)
    centre = np.array([0.5, 0.5])
    problem.exposures.append(centre)
    half_fwhm_away = centre + np.array([0.05, 0.0])
    far_away = centre + np.array([0.5, 0.0])
    assert problem.dose_at(centre)[0] == pytest.approx(problem.dose_per_exposure)
    assert problem.dose_at(half_fwhm_away)[0] == pytest.approx(problem.dose_per_exposure / 2)
    assert problem.dose_at(far_away)[0] < 1e-6 * problem.dose_per_exposure


def test_signal_falls_linearly_and_halves_at_the_half_dose():
    problem = BeamDamageBraninCurrin()
    doses = np.array([0.0, problem.half_dose, 2 * problem.half_dose, 3 * problem.half_dose])
    assert np.allclose(problem.intensity(doses), [1.0, 0.5, 0.0, 0.0])


def test_repeated_exposure_makes_a_spot_read_lower_a_false_minimum():
    problem = _noise_free(BeamDamageBraninCurrin())
    spot = np.array([[0.3, 0.7]])
    pristine = problem.clean(spot)[0]
    rng = np.random.default_rng(0)
    readings = [problem.observe(spot, rng)[0] for _ in range(5)]
    for k, reading in enumerate(readings):
        expected = (1 - k * problem.dose_per_exposure / (2 * problem.half_dose)) * pristine
        assert np.allclose(reading, expected)
    assert np.all(readings[-1] < pristine)                 # both objectives minimised: looks better
    assert problem.instrument_reading == pytest.approx(5 * problem.dose_per_exposure / problem.half_dose)


def test_reading_the_pristine_sample_deposits_no_dose():
    problem = _noise_free(BeamDamageBraninCurrin())
    grid = problem.grid
    assert np.allclose(problem.observe_at_rest(grid, np.random.default_rng(1)), problem.clean(grid))
    assert problem.exposures == [] and problem.instrument_reading == 0.0


def test_dose_aware_lcb_prefers_a_fresh_spot():
    steer = _load("steer_state_space")
    candidates = [np.array([0.1, 0.1, 0.0, 0.5, 1.5]), np.array([0.9, 0.9, 0.0, 0.5, 0.0])]
    campaign = SimpleNamespace(problem=BeamDamageBraninCurrin(), policy_name="dose-aware-lcb",
                               args=SimpleNamespace(kappa=1.0, dose_penalty=1.0))
    assert steer.Campaign.classic_choice(campaign, candidates, np.array([0.0, 0.0, 0.0]), step=0) == 1


@pytest.mark.filterwarnings("ignore")
def test_beam_steering_and_analysis_end_to_end(tmp_path):
    steer = _load("steer_state_space")
    analyze = _load("analyze_steering")
    steer.main(["--problem", "branin-currin-beam", "--seeds", "5", "--steps", "34", "--n-init", "24",
                "--policies", "random", "lcb", "dose-aware-lcb", "state-space", "--out", str(tmp_path)])
    with open(tmp_path / "steps.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert all(float(row["dose_at_query"]) >= 0.0 for row in rows)
    assert {"instrument", "dose_at_query"} <= set(rows[0])

    analyze.main(["--in", str(tmp_path), "--window", "3"])
    summary = json.loads((tmp_path / "steering_summary.json").read_text())
    for entry in summary.values():
        assert 0.0 <= entry["damaged_query_share"] <= 1.0
        assert np.isfinite(entry["dose_at_query_median"])
        assert np.isfinite(entry["cumulative_regret_median"]) and entry["cumulative_regret_median"] >= 0.0


def test_query_regret_is_zero_on_the_front_and_grows_away_from_it():
    steer = _load("steer_state_space")
    problem = BeamDamageBraninCurrin()
    front = problem.pareto_front
    assert np.all(np.diff(front[:, 0]) > 0) and np.all(np.diff(front[:, 1]) < 0)   # non-dominated
    assert steer.query_regret(problem, front[len(front) // 2]) == 0.0
    corner = front[len(front) // 2] + 0.1 * problem.y_scales
    farther = front[len(front) // 2] + 0.5 * problem.y_scales
    assert 0.0 < steer.query_regret(problem, corner) < steer.query_regret(problem, farther)
    assert steer.query_regret(problem, front.min(axis=0) - 1.0) == 0.0               # beyond the front
