"""Shared figure helpers and the publication ``rcParams`` block.

Importing this module (or anything in :mod:`traits_audit.stability.viz`) does
**not** change global matplotlib state. Use :func:`style_context` to render
under the publication style, or :func:`apply_publication_style` to opt in
process-wide (e.g. at the top of a standalone figure script).
"""
from __future__ import annotations

import functools
from contextlib import contextmanager
from pathlib import Path
from string import ascii_lowercase
from typing import Any, Dict, Iterator

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "PUBLICATION_RCPARAMS",
    "SINGLE_COL",
    "DOUBLE_COL",
    "style_context",
    "apply_publication_style",
    "publication_style",
    "letters",
    "save_fig",
    "draw_unit_circle",
    "set_square",
    "box",
    "signed_log_colorbar",
    "maybe_log",
]

SINGLE_COL = 3.5
DOUBLE_COL = 7.0

PUBLICATION_RCPARAMS: Dict[str, Any] = {
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "lines.linewidth":   1.5,
    "axes.linewidth":    0.8,
    "grid.alpha":        0.3,
    "figure.dpi":        300,
}


@contextmanager
def style_context() -> Iterator[None]:
    """Render under :data:`PUBLICATION_RCPARAMS`, restoring the previous
    ``rcParams`` on exit."""
    with plt.rc_context(PUBLICATION_RCPARAMS):
        yield


def publication_style(func):
    """Decorator: run a plot function under :func:`style_context`, so figures
    get the publication ``rcParams`` without any import-time global change."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with style_context():
            return func(*args, **kwargs)
    return wrapper


def apply_publication_style() -> None:
    """Apply :data:`PUBLICATION_RCPARAMS` process-wide (opt-in)."""
    plt.rcParams.update(PUBLICATION_RCPARAMS)


def letters(n: int):
    return ascii_lowercase[:n]


def save_fig(fig, out_dir: Path, stem: str, transparent: bool = False, extra_artists=None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", transparent=transparent,
                bbox_extra_artists=extra_artists)
    plt.close(fig)
    return path


def draw_unit_circle(ax) -> None:
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), "k--", lw=0.8, alpha=0.5)


def set_square(ax) -> None:
    """Force the axes bounding box to be square (physical 1:1), independent of data scale."""
    ax.set_box_aspect(1)


def box(ax) -> None:
    """Draw a complete black box (all four spines), regardless of any global
    rcParams style applied elsewhere (e.g. a process-wide ``plt.style.use``
    that turns off the top/right spines)."""
    for side in ("top", "right", "bottom", "left"):
        spine = ax.spines[side]
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)


def signed_log_colorbar(im, ax, vmax: float, linthresh: float, label: str = "signed log"):
    """Colorbar for a SymLogNorm-mapped image with a small, readable tick set.

    SymLogNorm's default tick locator produces dense major+minor log ticks on
    both the positive and negative side, which overlap and become unreadable
    at this figure size. Replace them with exactly 5 explicit ticks
    (-vmax, -linthresh, 0, linthresh, vmax) labelled in compact ``±10^k`` form.
    """
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, label=label)
    cbar.ax.minorticks_off()
    ticks = sorted({-vmax, -linthresh, 0.0, linthresh, vmax})
    cbar.set_ticks(ticks)

    def _fmt(v, _pos=None):
        if abs(v) < 1e-15:
            return "0"
        exp = int(np.floor(np.log10(abs(v))))
        sign = "-" if v < 0 else ""
        return f"{sign}$10^{{{exp}}}$"

    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(_fmt))
    cbar.ax.tick_params(labelsize=7)
    return cbar


def maybe_log(ax, data, axis: str = "y") -> None:
    """Switch axis to log scale when all-positive data spans > 2 orders of magnitude."""
    flat = np.asarray(data, dtype=float).ravel()
    flat = flat[np.isfinite(flat) & (flat > 0)]
    if len(flat) > 1 and flat.max() / flat.min() > 100:
        if axis == "y":
            ax.set_yscale("log")
        else:
            ax.set_xscale("log")
