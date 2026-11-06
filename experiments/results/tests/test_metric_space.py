"""The metric-space analysis: participation ratio, orthogonal residuals, separability."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_metric_space.py"


def _load():
    spec = importlib.util.spec_from_file_location("analyze_metric_space", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_participation_ratio_counts_independent_directions():
    space = _load()
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(500)
    copies = np.column_stack([signal * k for k in (1.0, 2.0, -3.0)])
    assert abs(space.participation_ratio(copies) - 1.0) < 1e-9
    independent = rng.standard_normal((5000, 4))
    assert 3.8 < space.participation_ratio(independent) <= 4.0


def test_participation_ratio_squares_the_singular_values():
    space = _load()
    rng = np.random.default_rng(1)
    matrix = rng.standard_normal((300, 3)) @ np.diag([1.0, 1.0, 1.0])
    matrix[:, 1] += 2.0 * matrix[:, 0]
    s = np.linalg.svd(space.zscore_columns(matrix), compute_uv=False)
    expected = (s ** 2).sum() ** 2 / np.sum(s ** 4)
    assert abs(space.participation_ratio(matrix) - expected) < 1e-9


def test_orthogonal_residual_is_zero_inside_the_basis_and_one_outside():
    space = _load()
    rng = np.random.default_rng(2)
    basis = rng.standard_normal((400, 3))
    inside = basis @ np.array([1.0, -2.0, 0.5])
    assert space.orthogonal_residual(inside, basis) < 1e-9
    outside = rng.standard_normal(400)
    assert space.orthogonal_residual(outside, basis) > 0.95


def test_separable_conditions_are_classified_and_noise_is_at_chance():
    space = _load()
    rng = np.random.default_rng(3)
    n_columns = len(space.ALL_METRICS)
    campaigns = {}
    for c, condition in enumerate(space.CONDITIONS):
        for seed in range(8):
            metrics = rng.standard_normal((20, n_columns)) * 0.1
            metrics[:, 0] += 3.0 * c                         # only the first metric knows the condition
            campaigns[(condition, "random", seed)] = {"metrics": metrics, "sigma": rng.standard_normal(20)}
    result = space.separability(campaigns)
    assert result[space.ALL_METRICS[0]] == 1.0
    assert result["mean sigma"] < 0.6


@pytest.mark.filterwarnings("ignore")
def test_metric_space_runs_end_to_end_on_a_small_grid(tmp_path):
    space = _load()
    for condition in ("calibrated", "overconfident"):
        space.steer.main(["--problem", "forrester", "--condition", condition, "--seeds", "3", "--steps", "30",
                          "--n-init", "9", "--policies", "random", "--out", str(tmp_path / "forrester" / condition)])
    space.main(["--runs", str(tmp_path), "--problem", "forrester", "--arms", "random",
                "--conditions", "calibrated", "overconfident", "--partition-seeds", "2", "--partition-steps", "20"])
    summary = json.loads((tmp_path / "forrester" / "metric_space.json").read_text())
    assert 0.0 <= summary["separability"]["all metrics"] <= 1.0
    for condition in ("calibrated", "overconfident"):
        partition = summary["partition_test"][condition]
        assert all(partition[name]["fiber only"] for name in space.FIBER_METRICS)
        assert not partition["MisspecificationResidualFloor"]["fiber only"]
