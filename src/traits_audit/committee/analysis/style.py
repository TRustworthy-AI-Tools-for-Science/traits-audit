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


# --- Solo agents: colour AND linestyle ------------------------------------
#
# The registry has 15 agents and Okabe-Ito has 8 hues, so hue alone cannot
# separate them -- the previous dict covered only the original 9 and let
# `policy_color`'s fallback paint the other six the same blue, which made
# seven curves in the regret figure indistinguishable.
#
# Each agent therefore gets (hue, linestyle). Agents sharing a hue are
# deliberately from different metric families, so a hue collision is never
# also a semantic collision, and the two members of a pair are always
# solid vs dashed. 8 hues x 2 styles = 16 slots for 15 agents.
_AGENT_STYLE: dict[str, tuple[str, str]] = {
    # Proper scoring rules
    "CRPS":                 (SKY_BLUE,       "-"),
    "NLL":                  (ORANGE,         "-"),
    "IntervalScore":        (BLUISH_GREEN,   "-"),
    # Calibration diagnostics. Okabe-Ito's YELLOW (#F0E442) is designed for
    # fills and is too faint as a thin line on white, so these use a darker
    # gold of the same hue family.
    "CalibrationError":     ("#B8860B",      "-"),
    "CalibrationError1Std": ("#B8860B",      "--"),
    "KuleshovCalibration":  (BLUE,           "-"),
    "ENCE":                 (BLUE,           "--"),
    "PITUniformity":        ("#AA4499",      "-"),   # magenta
    # Coverage
    "ConformalCoverage":    (REDDISH_PURPLE, "-"),
    "IntervalCoverage":     (REDDISH_PURPLE, "--"),
    # Variance alignment
    "VarianceAlignment":    (SKY_BLUE,       "--"),
    "VarErrCorrelation":    (BLUISH_GREEN,   "--"),
    # Signal-based. Deliberately NOT black: LCB is drawn in black and the
    # grey baselines are dashed, so an agent in black/grey would collide
    # with a comparator rather than with another agent.
    "UncertaintyEvolution": ("#117733",      "-"),   # dark green
    "UncertaintyAnomaly":   ("#882255",      "-"),   # wine
    "MahalanobisOOD":       (VERMILLION,     "--"),
}


def agent_style(name: str) -> tuple[str, str]:
    """(colour, linestyle) for an agent, with or without a ``solo:`` prefix.

    Raises for an unknown agent rather than silently returning a default:
    a fallback colour is how fifteen agents ended up sharing one blue.
    """
    key = name[5:] if name.startswith("solo:") else name
    if key not in _AGENT_STYLE:
        raise KeyError(
            f"No style for agent {key!r}. Add it to _AGENT_STYLE in "
            f"style.py -- known: {sorted(_AGENT_STYLE)}"
        )
    return _AGENT_STYLE[key]


def policy_color(name: str) -> str:
    """Stable colour lookup for a policy.

    Solo agents resolve through :data:`_AGENT_STYLE`, so every agent has a
    distinct (colour, linestyle) pair; see :func:`agent_style` for the
    linestyle. Non-agent policies come from :data:`POLICY_COLOR`.
    """
    if name.startswith("solo:") or name in _AGENT_STYLE:
        return agent_style(name)[0]
    if name.startswith("best-solo:"):
        return agent_style(name.split(":", 1)[1])[0]
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
