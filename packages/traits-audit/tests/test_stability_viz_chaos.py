"""Smoke tests for traits_audit.stability.viz.chaos_plots."""
import subprocess
import sys

import matplotlib
import numpy as np
import pytest

from traits_audit.stability.viz import chaos_plots as cp
from traits_audit.stability.viz.style import PlotStyle

POLS = ["alpha", "beta", "gamma"]


def _curves(base=1.0, n=20):
    rng = np.random.default_rng(0)
    out = {}
    for p in POLS:
        m = base + rng.normal(size=n).cumsum() * 0.05
        out[p] = (np.arange(n), m, m - 0.2, m + 0.2)
    return out


def _cases():
    rng = np.random.default_rng(1)
    dist = {p: rng.normal(size=40) for p in POLS}
    return {
        "plot_early_vs_late_lambda_max": (
            ({p: {"early": list(rng.random(5)), "late": list(rng.random(5))} for p in POLS},)),
        "plot_lle_distribution": (dist,),
        "plot_local_linearity_evolution": (_curves(1.0), _curves(0.5)),
        "plot_rolling_lyapunov_evolution": (_curves(0.0),),
        "plot_trend_evolution": (_curves(),),
        "plot_trend_rolling_lyapunov_evolution": (_curves(0.0),),
        "plot_local_linear_stability_spectrum": (dist,),
        "plot_effective_rank_evolution": (_curves(3.0),),
        "plot_poles": ({p: (rng.normal(size=50) + 1j * rng.normal(size=50)) * 0.6 for p in POLS},),
    }


CASES = _cases()


def test_all_public_functions_covered():
    assert set(CASES) == set(cp.__all__)


@pytest.mark.parametrize("name", sorted(CASES))
def test_default_style(name, tmp_path):
    path = getattr(cp, name)(*CASES[name], tmp_path)
    assert path.exists() and path.stat().st_size > 0 and path.suffix == ".png"


@pytest.mark.parametrize("name", sorted(CASES))
def test_custom_style_filters_and_orders(name, tmp_path):
    style = PlotStyle(policy_order=["gamma", "alpha", "absent"],
                      policy_colors={"gamma": "red", "alpha": "blue"},
                      policy_markers={"gamma": "s", "alpha": "^"})
    path = getattr(cp, name)(*CASES[name], tmp_path, style=style)
    assert path.exists() and path.stat().st_size > 0


def test_label_kwargs(tmp_path):
    p = cp.plot_trend_evolution(CASES["plot_trend_evolution"][0], tmp_path,
                                xlabel="step", ylabel="drift")
    assert p.exists()


def test_import_does_not_change_rcparams():
    code = (
        "import matplotlib; before = dict(matplotlib.rcParams);"
        "import traits_audit.stability.viz.chaos_plots;"
        "assert dict(matplotlib.rcParams) == before"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_plotting_restores_rcparams(tmp_path):
    before = dict(matplotlib.rcParams)
    cp.plot_lle_distribution(CASES["plot_lle_distribution"][0], tmp_path)
    assert dict(matplotlib.rcParams) == before
