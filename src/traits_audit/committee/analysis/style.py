"""Shared figure style for the committee analysis subpackage.

Local to ``traits_audit.committee.analysis`` — we do NOT touch the
project-wide rcParams in :mod:`traits_audit._viz`. Existing MLflow
dashboards continue to render with their default tab10 palette.

Colour map
----------
Okabe–Ito: an 8-colour qualitative palette designed for colour-blind
accessibility, distinguishable in grayscale, and standard for scientific
publication. Reference:
    Okabe, M. & Ito, K. (2008). "Color Universal Design (CUD): How to make
    figures and presentations that are friendly to colorblind people."

Headline policy / winning curve uses ``VERMILLION``; baselines are muted
grey + dashed via :data:`BASELINE_STYLE`.

Font sizes
----------
Targets the density-figure scale (title 15, label 13, ticks 11) so the
whole committee figure family reads consistently.
"""
from __future__ import annotations

from typing import Optional

# Okabe-Ito 8-colour palette ------------------------------------------------

BLACK = "#000000"
ORANGE = "#E69F00"
SKY_BLUE = "#56B4E9"
BLUISH_GREEN = "#009E73"
YELLOW = "#F0E442"
BLUE = "#0072B2"
VERMILLION = "#D55E00"
REDDISH_PURPLE = "#CC79A7"

OKABE_ITO = [
    BLACK, ORANGE, SKY_BLUE, BLUISH_GREEN,
    YELLOW, BLUE, VERMILLION, REDDISH_PURPLE,
]

# Semantic roles ------------------------------------------------------------

HEADLINE = VERMILLION
NEUTRAL_DARK = BLACK
NEUTRAL_LIGHT = "0.40"          # mid-grey for baselines (darker than before)
NEUTRAL_FAINT = "0.55"          # grey for random floor (still distinguishable)


# Font sizes (match density.py: title 15, label 13, tick 11) ---------------

TITLE_FS = 15
SUBTITLE_FS = 12
LABEL_FS = 13
TICK_FS = 11
LEGEND_FS = 11
ANNOT_FS = 10


# Line styles ---------------------------------------------------------------

BASELINE_STYLE = dict(color=NEUTRAL_LIGHT, lw=2.0, ls="--", alpha=1.0)
RANDOM_STYLE = dict(color=NEUTRAL_FAINT, lw=2.0, ls=":", alpha=1.0)
HEADLINE_STYLE = dict(color=HEADLINE, lw=2.6)


# Per-policy colour map (used by regret / B1 / A1 / A3) --------------------
#
# Stable assignment so a given policy gets the same colour across all
# committee figures. Add new policies here when introduced.

POLICY_COLOR: dict[str, str] = {
    # Baselines (rendered with BASELINE_STYLE / RANDOM_STYLE)
    "random": NEUTRAL_FAINT,
    "max-sigma": NEUTRAL_LIGHT,
    "LCB": BLACK,
    # v0 solo agents
    "solo:CRPS":              SKY_BLUE,
    "solo:NLL":               ORANGE,
    "solo:IntervalScore":     BLUISH_GREEN,
    "solo:CalibrationError":  YELLOW,
    "solo:ConformalCoverage": BLUE,
    "solo:PITUniformity":     VERMILLION,
    "solo:IntervalCoverage":  REDDISH_PURPLE,
    "solo:VarianceAlignment": "#777777",
    "solo:VarErrCorrelation": "#117733",
    # v0 best-solo (also alias for the line in Thread A)
    "best-solo:PITUniformity": VERMILLION,
    # v0 committee + v1 aggregators
    "committee": REDDISH_PURPLE,
    "committee:uniform":          REDDISH_PURPLE,
    "committee:agree":            ORANGE,
    "committee:disagree":         SKY_BLUE,
    "committee:weighted-indep":   BLUISH_GREEN,
    "committee:weighted-invreg":  VERMILLION,   # headline winner
    # Thread B vote-augmented variants
    "max-sigma+votes": VERMILLION,              # headline winner in B1
    "LCB+votes":       BLUE,
}


def policy_color(name: str) -> str:
    """Stable colour lookup for a policy. Falls back to BLUE if unknown."""
    return POLICY_COLOR.get(name, BLUE)


# Convenience: style a matplotlib Axes consistently ------------------------

def style_axes(ax, *, xlim: Optional[tuple[float, float]] = None,
               ylog: bool = False) -> None:
    """Apply tick font sizes and an optional xlim / log y-axis in one call."""
    ax.tick_params(axis="both", labelsize=TICK_FS)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylog:
        ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both" if ylog else "major")
