import os
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
]

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
