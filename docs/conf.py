import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

_DOCS = Path(__file__).parent
sys.path.insert(0, str(_DOCS.parent / "packages" / "traits-audit" / "src"))

# ── Generated figures ─────────────────────────────────────────────────────────
# The built-in-metric reference figures are regenerated on every build (~2 s,
# numpy/scipy/matplotlib only) rather than committed to git.
subprocess.run(
    [sys.executable, str(_DOCS / "scripts" / "visualize_builtin_metrics.py"),
     "--out", str(_DOCS / "_static" / "built_in_metrics")],
    check=True,
    env={**os.environ, "MPLBACKEND": "Agg"},
)

# ── Project metadata ─────────────────────────────────────────────────────────
project   = "traits-audit"
author    = "Ashley Dale"
copyright = "2026, Ashley Dale"
release   = "0.2.0"

# ── Extensions ────────────────────────────────────────────────────────────────
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_gallery.gen_gallery",
]

# ── Demo gallery (sphinx-gallery) ─────────────────────────────────────────────
# Every script in ../demos is executed on each build and rendered, with its
# output and figures, under docs/demos/. A demo whose optional dependency is
# missing is still rendered but not executed. CI sets
# TRAITS_AUDIT_DEMOS_STRICT=1 so that a missing dependency fails the build.
_DEMO_EXTRAS = {
    "03_bayesian_optimization": "botorch",
    "07_logging_to_mlflow": "mlflow",
    "08_existing_framework_callback": "skopt",
}
_not_executed = [demo for demo, module in _DEMO_EXTRAS.items() if importlib.util.find_spec(module) is None]
if _not_executed and os.environ.get("TRAITS_AUDIT_DEMOS_STRICT"):
    raise RuntimeError(f"Demos missing their optional dependency: {_not_executed}")
if _not_executed:
    print(f"sphinx-gallery: not executing {_not_executed} (optional dependency missing)")
_skip = f"(?!{'|'.join(_not_executed)})" if _not_executed else ""

sphinx_gallery_conf = {
    "examples_dirs": "../demos",
    "gallery_dirs": "demos",
    "filename_pattern": re.escape(os.sep) + _skip + r"\d\d_",
    "within_subsection_order": "FileNameSortKey",
    "write_computation_times": False,
    "download_all_examples": False,
}

# ── Napoleon (NumPy docstrings) ───────────────────────────────────────────────
napoleon_numpy_docstring        = True
napoleon_google_docstring       = False
napoleon_use_param              = True
napoleon_use_rtype              = True
napoleon_preprocess_types       = True
napoleon_use_ivar               = True   # avoid duplicate attribute entries for dataclasses

# ── Autodoc ───────────────────────────────────────────────────────────────────
autodoc_member_order            = "bysource"
autodoc_typehints               = "description"
autodoc_typehints_format        = "short"
autoclass_content               = "both"
autodoc_default_options = {
    "members":          True,
    "undoc-members":    False,
    "show-inheritance": True,
    "special-members":  "__init__",
    "exclude-members":  "__weakref__",
}

# ── Intersphinx ───────────────────────────────────────────────────────────────
intersphinx_mapping = {
    "python": ("https://docs.python.org/3",         None),
    "numpy":  ("https://numpy.org/doc/stable",      None),
    "scipy":  ("https://docs.scipy.org/doc/scipy",  None),
}

# ── MyST (Markdown) ───────────────────────────────────────────────────────────
myst_enable_extensions = ["colon_fence", "deflist"]

# ── HTML output ───────────────────────────────────────────────────────────────
html_theme         = "furo"
html_logo          = "_static/logo.svg"
html_static_path   = ["_static"]
html_title         = "traits-audit"
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}
