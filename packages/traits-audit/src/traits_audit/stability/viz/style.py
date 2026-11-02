"""Injectable plot styling for the stability figures.

The figure functions group their data by two labels, ``policy`` (the inner
grouping, e.g. an acquisition policy) and ``model`` (the outer grouping, e.g.
a surrogate model). A :class:`PlotStyle` decides which labels are drawn, in
which order, and with which color / marker / linestyle. Callers that want a
fixed visual identity (a paper's figure conventions) build their own
``PlotStyle`` and pass it to every function; the default style assigns
colors and markers automatically and draws every label present in the data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

__all__ = ["PlotStyle", "DEFAULT_STYLE", "resolve_style", "TAB10"]

#: matplotlib's ``tab10`` palette, spelled out so the default style needs no matplotlib import.
TAB10 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
_MARKERS = ["o", "s", "^", "v", "D", "P", "X", "<", ">", "*"]
_LINESTYLES = ["-", "--", ":", "-."]


@dataclass
class PlotStyle:
    """Label order and visual identity for the stability figures.

    ``policy_order`` / ``model_order`` empty (the default) means "draw every
    label present in the data, in order of first appearance". When non-empty,
    only the listed labels are drawn, in that order (labels absent from the
    data are skipped, labels absent from the list are not drawn).

    ``policy_colors`` / ``policy_markers`` / ``model_colors`` /
    ``model_linestyles`` map labels to explicit values; labels without an
    entry get automatic ones (cycling through ``tab10`` / a marker list /
    a linestyle list) that stay stable for the lifetime of the style object.
    """

    policy_order: List[str] = field(default_factory=list)
    model_order: List[str] = field(default_factory=list)
    policy_colors: Dict[str, str] = field(default_factory=dict)
    policy_markers: Dict[str, str] = field(default_factory=dict)
    model_colors: Dict[str, str] = field(default_factory=dict)
    model_linestyles: Dict[str, str] = field(default_factory=dict)
    # semantic accent colors used by single-series bars/lines
    blue: str = TAB10[0]
    green: str = TAB10[2]
    orange: str = TAB10[1]
    red: str = TAB10[3]
    neutral: str = "0.8"
    #: fallback for labels with no color and ``auto_colors=False``
    fallback_color: str = "0.3"
    auto_colors: bool = True

    # ── label selection ────────────────────────────────────────────────────
    def policies(self, present: Iterable[str]) -> List[str]:
        """Policies to draw, given those present in the data."""
        present = list(dict.fromkeys(present))
        if self.policy_order:
            have = set(present)
            return [p for p in self.policy_order if p in have]
        return present

    def models(self, present: Iterable[str]) -> List[str]:
        """Models to draw, given those present in the data."""
        present = list(dict.fromkeys(present))
        if self.model_order:
            have = set(present)
            return [m for m in self.model_order if m in have]
        return present

    # ── visual identity ────────────────────────────────────────────────────
    def _auto(self, table: Dict[str, str], key: str, palette: Sequence[str]) -> str:
        if key not in table:
            table[key] = palette[len(table) % len(palette)]
        return table[key]

    def color(self, policy: str, default: str | None = None) -> str:
        if policy in self.policy_colors:
            return self.policy_colors[policy]
        if default is not None:
            return default
        if not self.auto_colors:
            return self.fallback_color
        return self._auto(self.policy_colors, policy, TAB10)

    def marker(self, policy: str, default: str | None = None) -> str:
        if policy in self.policy_markers:
            return self.policy_markers[policy]
        if default is not None:
            return default
        return self._auto(self.policy_markers, policy, _MARKERS)

    def model_color(self, model: str, default: str | None = None) -> str:
        if model in self.model_colors:
            return self.model_colors[model]
        return default if default is not None else self._auto(self.model_colors, model, TAB10)

    def linestyle(self, model: str) -> str:
        return self._auto(self.model_linestyles, model, _LINESTYLES)


def DEFAULT_STYLE() -> PlotStyle:
    """A fresh automatic style (a new object per call: automatic assignments are
    remembered on the instance, so it should not be shared between unrelated figures)."""
    return PlotStyle()


def resolve_style(style: "PlotStyle | None") -> PlotStyle:
    """``style`` if given, else a fresh default style."""
    return style if style is not None else PlotStyle()
