import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ── Project metadata ─────────────────────────────────────────────────────────
project   = "traits-audit"
author    = "Ashley Dale"
copyright = "2026, Ashley Dale"
release   = "0.1.0"

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
