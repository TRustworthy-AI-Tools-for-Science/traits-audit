"""The classic Bayesian-optimisation arms of the steering experiment (minimisation)."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "steer_state_space.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("steer_state_space", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _choose(module, policy_name, mu, sigma, best=0.0):
    """classic_choice on 1-D candidates [x, mu, sigma], without a full campaign."""
    candidates = [np.array([i / 10, m, s]) for i, (m, s) in enumerate(zip(mu, sigma, strict=True))]
    campaign = SimpleNamespace(problem=SimpleNamespace(dim=1), policy_name=policy_name,
                               args=SimpleNamespace(pi_margin=0.01))
    return module.Campaign.classic_choice(campaign, candidates, np.array([best, 0.0]), step=0)


def test_classic_rules_rank_as_their_formulas_imply():
    module = _load_script()
    same_sigma = [0.5, 0.5, 0.5]
    assert _choose(module, "max-sigma", [0.0, 0.0, 0.0], [0.1, 0.9, 0.3]) == 1
    assert _choose(module, "expected-improvement", [0.4, -0.2, 0.1], same_sigma) == 1
    assert _choose(module, "probability-of-improvement", [0.4, -0.2, 0.1], same_sigma) == 1
    assert _choose(module, "expected-improvement", [0.3, 0.3, 0.3], [0.1, 0.9, 0.3]) == 1


def test_polynomial_members_average_to_the_prediction():
    from results.committee.problems import PolyBootstrapSurrogate

    rng = np.random.default_rng(0)
    x = rng.uniform(size=(30, 1))
    model = PolyBootstrapSurrogate(dim=1, degree=3, n_estimators=8, rng=rng).fit(x, np.sin(6 * x[:, 0]))
    grid = np.linspace(0, 1, 11)[:, None]
    members = model.predict_members(grid)
    assert members.shape == (11, 8)
    assert np.allclose(members.mean(axis=1), model.predict(grid)[0])


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("condition", ["calibrated", "misspecified"])
def test_classic_arms_run_end_to_end(tmp_path, condition):
    module = _load_script()
    out = tmp_path / condition
    module.main(["--problem", "forrester", "--condition", condition, "--seeds", "1", "--steps", "14",
                 "--n-init", "9", "--policies", *module.CLASSIC_POLICIES, "--out", str(out)])
    import csv

    with open(out / "steps.csv", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["phase"] == "steer"]
    assert {r["policy"] for r in rows} == set(module.CLASSIC_POLICIES)
    assert all(0.0 <= float(r["x0"]) <= 1.0 for r in rows)


def test_gp_joint_draw_survives_a_near_singular_posterior():
    module = _load_script()
    gp = module.ScaledGP(sigma_scale=0.3, seed=0)
    x = np.repeat(np.linspace(0, 1, 6), 4)[:, None]                 # repeated points
    gp.fit(x, np.sin(6 * x[:, 0]))
    grid = np.linspace(0, 1, 400)[:, None]                         # dense: covariance near-singular
    draw = gp.sample(grid, np.random.default_rng(0))
    assert draw.shape == (400,) and np.all(np.isfinite(draw))
    mu, _ = gp.predict(grid)
    assert np.max(np.abs(draw - mu)) < 10 * np.max(np.abs(mu) + 1)
