"""Run every script in the repository's ``demos/`` folder as a user would.

The docs build also executes the demos (sphinx-gallery); this test catches a
broken demo without building the docs. A demo whose optional dependency is
not installed is skipped. Skipped entirely when ``demos/`` is absent (e.g.
when testing an unpacked sdist).
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

DEMOS = Path(__file__).resolve().parents[3] / "demos"
# Optional dependency per demo; keep in step with _DEMO_EXTRAS in docs/conf.py.
EXTRAS = {
    "03_bayesian_optimization": "botorch",
    "07_logging_to_mlflow": "mlflow",
    "08_existing_framework_callback": "skopt",
}

scripts = sorted(DEMOS.glob("[0-9][0-9]_*.py")) if DEMOS.is_dir() else []


@pytest.mark.skipif(not scripts, reason="demos/ not found")
@pytest.mark.parametrize("script", scripts, ids=lambda p: p.stem)
def test_demo_runs(script, tmp_path):
    extra = EXTRAS.get(script.stem)
    if extra and importlib.util.find_spec(extra) is None:
        pytest.skip(f"needs {extra}")
    proc = subprocess.run(
        [sys.executable, str(script)],
        check=False,
        cwd=tmp_path,
        env={**os.environ, "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]


@pytest.mark.skipif(not scripts, reason="demos/ not found")
def test_every_demo_with_an_extra_is_listed_in_the_docs_config():
    conf = (DEMOS.parent / "docs" / "conf.py").read_text()
    for stem, module in EXTRAS.items():
        assert f'"{stem}": "{module}"' in conf
