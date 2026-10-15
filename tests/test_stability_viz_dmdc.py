"""Smoke tests for traits_audit.stability.viz.dmdc_plots."""
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from traits_audit.stability.viz import dmdc_plots as dp
from traits_audit.stability.viz.style import PlotStyle

POLS = ["alpha", "beta", "gamma"]
MODELS = ["m1", "m2"]
NF = 6
FN = [f"f{i}" for i in range(NF)]


def _cplx(rng, n=5):
    return rng.normal(0, 0.5, n) + 1j * rng.normal(0, 0.5, n)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    T = 8
    gx, gy = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
    re_, im_ = np.meshgrid(np.linspace(-1.5, 1.5, 20), np.linspace(-1.5, 1.5, 20))
    modes = {p: [{"magnitude": float(rng.uniform(0.3, 1.3)), "eigenvalue": complex(rng.normal(), 0.3),
                  "dominant_feature": "f0"} for _ in range(2)] for p in POLS}
    return {
        "eig": {m: {p: {"posthoc": _cplx(rng), "raw_state": _cplx(rng), "native": _cplx(rng, 3)}
                    for p in POLS} for m in MODELS},
        "series": {m: {p: {"mean": rng.random(T) + 0.5, "std_epistemic": rng.random(T) * .1,
                           "std_aleatoric": rng.random(T) * .1} for p in POLS} for m in MODELS},
        "plane": {"ds": {"gx": gx, "gy": gy, "lyap": gx ** 2 + gy ** 2, "v_safe": 2.0,
                         "trajectories": {p: {"pc1": rng.normal(size=T), "pc2": rng.normal(size=T),
                                              "lyap_deriv": rng.uniform(-1, 1, T)} for p in POLS}}},
        "gram": {m: {p: {"cond_c": 10 ** rng.uniform(0, 4), "cond_o": 10 ** rng.uniform(0, 4)}
                     for p in POLS} for m in MODELS},
        "conv": {m: {p: {"convergence": rng.random(T) + .3, "convergence_std": rng.random(T) * .1,
                         "perturbation_mean": np.exp(-np.arange(T) * .3), "perturbation_std": rng.random(T) * .1}
                     for p in POLS} for m in MODELS},
        "A": rng.normal(size=(NF, NF)), "B": rng.normal(size=(NF, 2)), "modes": modes,
        "pafull": {p: rng.normal(size=(NF, NF)) for p in POLS},
        "rank": {p: {2: 0.5, 4: 0.8, 6: 0.9} for p in POLS},
        "cross": {p: {"a": {"mean": 0.3, "std": 0.05}, "b": {"mean": 0.4, "std": 0.02}} for p in POLS},
        "r2": {m: {p: float(rng.random()) for p in POLS} for m in MODELS},
        "pseudo": {m: {p: {"RE": re_, "IM": im_, "sigma_min": np.abs(re_ + 1j * im_ - 0.5) + 0.1,
                           "eigenvalues": _cplx(rng)} for p in POLS} for m in MODELS},
    }


def _calls(d):
    return {
        "plot_poles": lambda o, **k: dp.plot_poles(d["eig"], o, **k),
        "plot_poles_combined": lambda o, **k: dp.plot_poles_combined(d["eig"], o, **k),
        "plot_stability_plane": lambda o, **k: dp.plot_stability_plane(d["plane"], o, **k),
        "plot_lyapunov_timeseries": lambda o, **k: dp.plot_lyapunov_timeseries(d["series"], 1.0, o, **k),
        "plot_lyapunov_all_models": lambda o, **k: dp.plot_lyapunov_all_models(d["series"], o, **k),
        "plot_gramians": lambda o, **k: dp.plot_gramians(d["gram"], o, stability_margin={"alpha": -0.1}, **k),
        "plot_convergence_robustness": lambda o, **k: dp.plot_convergence_robustness(d["conv"], o, **k),
        "plot_matrix_heatmaps": lambda o, **k: dp.plot_matrix_heatmaps(
            d["A"], d["B"], FN, ["u1", "u2"], d["modes"]["alpha"], o, policy_label="alpha", **k),
        "plot_coupling_network_combined": lambda o, **k: dp.plot_coupling_network_combined(d["pafull"], FN, o, **k),
        "plot_modal_equations": lambda o, **k: dp.plot_modal_equations(d["modes"], o, **k),
        "plot_rank_dataset": lambda o, **k: dp.plot_rank_dataset(d["rank"], 4, d["cross"], o, **k),
        "plot_dmdc_quality": lambda o, **k: dp.plot_dmdc_quality(d["r2"], d["pseudo"], o, **k),
    }

NAMES = [
    "plot_poles", "plot_poles_combined", "plot_stability_plane", "plot_lyapunov_timeseries",
    "plot_lyapunov_all_models", "plot_gramians", "plot_convergence_robustness", "plot_matrix_heatmaps",
    "plot_coupling_network_combined", "plot_modal_equations", "plot_rank_dataset", "plot_dmdc_quality"]


def _check(res):
    paths = res if isinstance(res, list) else [res]
    assert paths
    for p in paths:
        assert p.suffix == ".png" and p.stat().st_size > 0


@pytest.mark.parametrize("name", NAMES)
def test_default_style(name, data, tmp_path):
    _check(_calls(data)[name](tmp_path))


@pytest.mark.parametrize("name", NAMES)
def test_custom_style(name, data, tmp_path):
    style = PlotStyle(policy_order=["gamma", "alpha", "not-present"], model_order=["m2", "m1"],
                      policy_colors={"alpha": "red", "gamma": "green"},
                      policy_markers={"alpha": "s"}, model_linestyles={"m1": "-.", "m2": ":"},
                      blue="navy")
    _check(_calls(data)[name](tmp_path, style=style))


def test_import_does_not_change_rcparams():
    code = (
        "import matplotlib; before = dict(matplotlib.rcParams)\n"
        "import traits_audit.stability.viz.dmdc_plots\n"
        "assert dict(matplotlib.rcParams) == before\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_call_does_not_leak_rcparams(data, tmp_path):
    before = dict(matplotlib.rcParams)
    dp.plot_poles_combined(data["eig"], tmp_path)
    assert dict(matplotlib.rcParams) == before
