"""results.policies must work stand-alone (numpy + traits_audit only), and the
steering scripts built on them must run end to end."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from results.policies import (
    AuditStateSpacePolicy,
    OracleFailure,
    RandomPolicy,
    StateSpacePolicy,
    drt_peak_mismatch_result,
)
from traits_audit.base import AuditResult

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


class _Model:
    """[state | action] -> next state = state + action, std = |action|."""

    def predict_with_uncertainty(self, X):
        X = np.atleast_2d(X)
        return {"mean": X[:, :2] + X[:, 2:], "std": np.abs(X[:, 2:])}


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_oracle_failure_ends_run_loop():
    calls = []

    def oracle(a):
        calls.append(a)
        if len(calls) == 2:
            raise OracleFailure("instrument fault")
        return np.zeros(2)

    pol = RandomPolicy(num_action_features=2, rng=np.random.default_rng(0))
    hist = pol.run(np.zeros(2), [np.ones(2), -np.ones(2)], oracle, n_iterations=5)
    assert hist["terminated_early"] and hist["n_cycles_completed"] == 1


def test_state_space_policy_learns_dynamics_and_uses_health_fn():
    rng = np.random.default_rng(0)
    cands = [np.array([a, b]) for a in (-1.0, 0.0, 1.0) for b in (-1.0, 0.0, 1.0)]

    def make(health_fn, gamma):
        return AuditStateSpacePolicy(_Model(), n_components=3, kappa=0.0, R_weight=0.0,
                                     gamma_health=gamma, health_fn=health_fn, min_fit_obs=4,
                                     num_action_features=2, degradation_fn=lambda s: 0.0)

    # Health vector v_{t+1} = v_t + a[0]: higher a[0] is "healthier" under health_fn=+mean(v).
    pols = [make(lambda v: float(np.mean(v)), 5.0), make(None, 0.0)]
    for pol in pols:
        s, v = np.zeros(2), np.zeros(1)
        for _ in range(12):
            a = cands[rng.integers(len(cands))]
            pol.observe(s, a, uncertainty_vector=v)
            s, v = s + 0.1 * rng.normal(size=2), v + a[:1]
        assert pol._A_r is not None
    best_health, _, _ = pols[0].select_next(np.zeros(2), cands, uncertainty_vector=np.zeros(1))
    best_plain, _, _ = pols[1].select_next(np.zeros(2), cands, uncertainty_vector=np.zeros(1))
    assert best_health[0] == 1.0          # health term picks the action that improves v
    assert best_plain[0] != 1.0           # without it, every candidate ties
    assert isinstance(pols[1], StateSpacePolicy)


def test_drt_peak_mismatch_returns_traits_audit_result():
    tau = np.logspace(-4, 0, 200)
    gamma = np.exp(-((np.log10(tau) + 3) ** 2) / 0.02) + np.exp(-((np.log10(tau) + 1) ** 2) / 0.02)
    r = drt_peak_mismatch_result({0: np.column_stack([tau, gamma])}, n_ecm_elements=2)
    assert isinstance(r, AuditResult) and r.passed and r.value == 0.0


@pytest.mark.filterwarnings("ignore")
def test_steering_scripts_end_to_end(tmp_path, capsys):
    out = tmp_path / "steer"
    _load("steer_state_space").main(["--policies", "random", "audit-state-space", "--seeds", "5",
                                     "--steps", "20", "--n-init", "9", "--out", str(out)])
    cfg = json.loads((out / "config.json").read_text())
    assert set(cfg["steering"]).isdisjoint(cfg["audit_only"])
    _load("analyze_steering").main(["--in", str(out), "--window", "3"])
    summary = json.loads((out / "steering_summary.json").read_text())
    assert set(summary) == {"random", "audit-state-space"}
    assert summary["random"]["dS_minus_null"] == 0.0
    assert "Audit-only" in capsys.readouterr().out
