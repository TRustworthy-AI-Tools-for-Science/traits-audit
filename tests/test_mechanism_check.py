"""Smoke tests for the internal demo mechanism-check printer."""
import numpy as np

from traits_audit._mechanism_check import print_mechanism_check
from traits_audit.trajectory import TrajectoryRecord, analyze_trajectory


def _make_record(T=80, seed=0):
    rng = np.random.default_rng(seed)
    rec = TrajectoryRecord(domain="unit_test", policy="LCB")
    z = rng.standard_normal(2) * 0.1
    A = 0.85 * np.eye(2)
    for t in range(T):
        a = rng.standard_normal(2) * 0.1
        z = A @ z + a + rng.standard_normal(2) * 1e-3
        sigma_ep = 1.0 * np.exp(-t / 20.0) + 0.01 * rng.standard_normal()
        sigma_al = 0.3 + 0.01 * rng.standard_normal()
        rec.add(state=z, uncertainty=[sigma_ep, sigma_al], action=a)
    return rec.finalize()


def test_prints_headline_ratio_and_ci(capsys):
    rec = _make_record()
    result = analyze_trajectory(rec, n_components=4, n_boot=20)
    print_mechanism_check(result, "real split", aleatoric_indices=[1])
    out = capsys.readouterr().out
    assert "HEADLINE statistic" in out
    assert "secondary, non-discriminating" in out
    assert "UNCONTROLLABLE" in out


def test_flags_short_trajectory(capsys):
    rec = _make_record(T=10)
    result = analyze_trajectory(rec, n_components=4, n_boot=5)
    print_mechanism_check(result, "short run")
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "4d+1" in out


def test_no_alignment_line_without_aleatoric_indices(capsys):
    rec = _make_record()
    result = analyze_trajectory(rec, n_components=4, n_boot=5)
    print_mechanism_check(result, "null run", aleatoric_indices=None)
    out = capsys.readouterr().out
    assert "secondary, non-discriminating" not in out
