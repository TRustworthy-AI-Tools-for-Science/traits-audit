"""
Plots of audit results, built only on the hook history and the reports.

Every function returns a matplotlib ``Figure`` and leaves saving to the caller
(``fig.savefig("grid.png")``). Font sizes come from the active
``matplotlib.rcParams``, so a project-wide style applies automatically.

>>> from traits_audit.plotting import plot_check_grid
>>> stages = [(f"step {k}", r) for k, r in zip(steps, hook.intermediate_reports)]
>>> stages.append(("final", hook.report))
>>> plot_check_grid(stages).savefig("check_grid.png")

The module also exposes the logic the figures share, for callers drawing their
own: :func:`result_status` (how far a result sits from its threshold, and
whether it was judged at all), :data:`CHECK_ABBREVIATIONS`, and
:func:`split_trackable_checks` / :func:`filter_stage_reports`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .base import AuditReport, AuditResult
    from .pipeline import AuditPipeline

__all__ = [
    "CHECK_ABBREVIATIONS",
    "filter_stage_reports",
    "plot_audit_evolution",
    "plot_calibration_curve",
    "plot_check_grid",
    "plot_uncertainty_evolution",
    "result_status",
    "split_trackable_checks",
]

#: Short labels for check names, used where full names would crowd an axis.
#: Every built-in ``AuditCheck.name`` has an entry; unknown names are shown in full.
CHECK_ABBREVIATIONS: dict[str, str] = {
    # Total-predictive-distribution checks.
    "CalibrationError":            "CE",
    "KuleshovCalibrationError":    "KCE",
    "ENCE":                        "ENCE",
    "CalibrationError1Std":        "CE1Sig",
    "ConformalCoverage":           "CC",
    "CRPS":                        "CRPS",
    "IntervalCoverage":            "IC",
    "IntervalScore":               "IS",
    "NegativeLogLikelihood":       "NLL",
    "PITUniformity":               "PITU",
    "UncertaintyEvolution":        "UE",
    "UncertaintyAnomalies":        "UA",
    "VarianceAlignment":           "VA",
    "VarianceErrorCorrelation":    "VEC",
    "LyapunovStability":           "LYS",
    "MahalanobisOOD":              "MOOD",
    # Taxonomy-audit additions (METRIC_TAXONOMY_AUDIT.md §4).
    "SignedBias":                  "SB",
    "ReplicationShrinkageExponent": "RSE",
    "DarkUncertaintyGap":          "DUG",
    "TypeBMassFraction":           "TBMF",
    "ReducibilityRealisationRatio": "RRR",
    "AleatoricFloorConsistency":   "AFC",
    "EnsembleIndependenceDeficit": "EID",
    "DMDcSpectralRadius":          "Rho",
    "ResidualPersistenceHalfLife": "RHL",
    "ImprecisionWidthFraction":    "IWF",
    "EnvelopeViolationRate":       "EV",
    "ProceduralVarianceShare":     "PVS",
    "DataVarianceShare":           "DVS",
    "MisspecificationResidualFloor": "MRF",
    "StageVarianceAttribution":    "SVA",
    "DecisionFlipRate":            "DFR",
    "TailIndex":                   "TI",
    "ScoreDecomposition":          "SD",
}

#: Check-grid colour stops: dark red (deeply failing) → white (at threshold) → dark green.
_GRID_STOPS = [
    (0.00, "#7b0000"),
    (0.25, "#c0392b"),
    (0.45, "#f0a09a"),
    (0.50, "#f5f5f5"),
    (0.55, "#9fd49f"),
    (0.75, "#27ae60"),
    (1.00, "#1a5e35"),
]
_STATUS_COLOR = {"pass": "#27ae60", "fail": "#c0392b", "report_only": "#9a9a9a", "skipped": "#9a9a9a"}
#: Outline colour and line style for cells that are not verdicts.
_NON_VERDICT_STYLE = {"report_only": ("#7a7a7a", "-"), "skipped": ("#8888c0", "--")}


# ── Shared logic ──────────────────────────────────────────────────────────────

def result_status(result: AuditResult) -> tuple[float, str]:
    """Classify an :class:`~traits_audit.base.AuditResult` for display.

    A check with no threshold (``CRPSCheck()`` by default, ``TailIndexCheck``, …)
    always reports ``passed=True``, and so does a check skipped for lack of
    data. Neither was judged against a criterion, so neither should be drawn as
    "deeply passing".

    Returns
    -------
    (intensity, status)
        ``status`` is ``"pass"``, ``"fail"``, ``"report_only"`` (no threshold
        configured) or ``"skipped"`` (required data unavailable).
        ``intensity`` is meaningful only for ``"pass"`` / ``"fail"``:
        1.0 = deeply passing, 0.5 = at the threshold, 0.0 = deeply failing.
    """
    if isinstance(result.message, str) and result.message.startswith("Skipped"):
        return 0.5, "skipped"

    v = result.value
    t = result.threshold
    if t is None:
        return 0.5, "report_only"
    if v is None:
        return (1.0 if result.passed else 0.0), ("pass" if result.passed else "fail")

    name = result.name
    details = result.details or {}
    if name in ("CalibrationError", "UncertaintyAnomalies"):
        # Lower is better; PASS if v ≤ t
        signed = (t - v) / max(abs(t), 1e-6)
    elif name == "UncertaintyEvolution":
        # Higher (less negative) is better; PASS if v ≥ t. The check always
        # reports threshold=0.0, which carries no natural scale, so use a
        # display span instead.
        span = max(abs(t) * 3, 0.10)
        signed = (v - t) / span
    elif name == "VarianceErrorCorrelation":
        # Higher is better; PASS if v ≥ t.
        span = max(1.0 - t, 0.30)
        signed = (v - t) / span
    elif name == "IntervalCoverage":
        # t is the (lo, hi) acceptance band; the scalar branch is a fallback.
        if isinstance(t, tuple):
            target = (t[0] + t[1]) / 2
            tol = max((t[1] - t[0]) / 2, 1e-6)
        else:
            tol = details.get("tolerance", 0.1)
            target = t
        signed = (tol - abs(v - target)) / tol
    elif name == "VarianceAlignment":
        # Toward-target (ideal ratio = t); tolerance from the check's own details.
        tol = details.get("tolerance", 0.5)
        signed = (tol - abs(v - t)) / tol
    else:
        return (1.0 if result.passed else 0.0), ("pass" if result.passed else "fail")

    intensity = float(np.clip(0.5 + 0.5 * np.clip(signed, -1.0, 1.0), 0.0, 1.0))
    return intensity, ("pass" if result.passed else "fail")


def split_trackable_checks(
    stage_reports: list[tuple[str, AuditReport]],
) -> tuple[list[str], list[str]]:
    """Split check names into step-trackable and final-report-only.

    A check is final-only when it is skipped (or absent) at every intermediate
    stage and produces a value only in the last report. That is normal for
    checks that need post-loop data, such as a held-out test set or a refit
    sweep. Plotting the two groups separately keeps the grid readable.

    Parameters
    ----------
    stage_reports : list of (label, AuditReport)
        Reports in stage order; the last one is the final report.

    Returns
    -------
    (trackable, final_only)
        Check names in the order they appear in the final report.
    """
    if len(stage_reports) < 2:
        return [r.name for r in stage_reports[-1][1].results], []
    *intermediate, (_, final_report) = stage_reports
    trackable = set()
    for _, rep in intermediate:
        for r in rep.results:
            _, status = result_status(r)
            if status != "skipped":
                trackable.add(r.name)
    trackable_names, final_only_names = [], []
    for r in final_report.results:
        (trackable_names if r.name in trackable else final_only_names).append(r.name)
    return trackable_names, final_only_names


def filter_stage_reports(
    stage_reports: list[tuple[str, AuditReport]],
    names: list[str],
) -> list[tuple[str, AuditReport]]:
    """Copy ``stage_reports`` keeping only the checks in ``names``, in that order.

    A stage that lacks one of the names simply omits it.
    """
    from .base import AuditReport

    name_set = set(names)
    order = {n: i for i, n in enumerate(names)}
    out = []
    for label, rep in stage_reports:
        by_name = {r.name: r for r in rep.results if r.name in name_set}
        filtered = sorted(by_name.values(), key=lambda r: order[r.name])
        out.append((label, AuditReport(results=filtered, metadata=rep.metadata)))
    return out


def _format_cell(v: float | None) -> str:
    """Format a value in at most five characters."""
    if v is None:
        return "—"
    for fmt in (".3f", ".2f", ".1f", ".0f"):
        s = format(v, fmt)
        if len(s) <= 5:
            return s
    mantissa, exp_part = f"{v:.0e}".split("e")
    exp_digits = exp_part[1:].lstrip("0") or "0"
    return f"{mantissa}e{exp_digits}" if exp_part[0] == "+" else f"{mantissa}e-{exp_digits}"


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_check_grid(
    stage_reports: list[tuple[str, AuditReport]],
    title: str | None = None,
) -> Figure:
    """Heatmap of every check (rows) at every audit stage (columns).

    Cell colour shows how far the value sits from its threshold: dark green =
    deeply passing, white = at the threshold, dark red = deeply failing. Checks
    with no threshold ("report-only") and checks skipped for lack of data are
    not verdicts, so they get no fill: they are drawn as an outlined cell (solid
    grey for report-only, dashed blue-grey for skipped) with their value in grey.

    Parameters
    ----------
    stage_reports : list of (label, AuditReport)
        One entry per stage, e.g. ``hook.intermediate_reports`` labelled by
        step, followed by the final report.
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Patch, Rectangle

    check_names: list[str] = []
    for _, rep in stage_reports:
        for r in rep.results:
            if r.name not in check_names:
                check_names.append(r.name)
    stage_labels = [label for label, _ in stage_reports]
    n_checks, n_stages = len(check_names), len(stage_labels)

    z = np.full((n_checks, n_stages), np.nan)
    status = [["skipped"] * n_stages for _ in range(n_checks)]
    cells = [["—"] * n_stages for _ in range(n_checks)]
    for si, (_, rep) in enumerate(stage_reports):
        by_name = {r.name: r for r in rep.results}
        for ci, name in enumerate(check_names):
            result = by_name.get(name)
            if result is None:
                continue  # not evaluated at this stage: drawn like a skipped check
            intensity, st = result_status(result)
            status[ci][si] = st
            cells[ci][si] = _format_cell(result.value)
            if st in ("pass", "fail"):
                z[ci, si] = intensity

    fig, ax = plt.subplots(figsize=(max(4.0, 1.6 + 0.55 * n_stages), max(2.0, 0.8 + 0.32 * n_checks)))
    cmap = LinearSegmentedColormap.from_list("check_grid", _GRID_STOPS)
    ax.set_facecolor("#dcdcdc")
    ax.imshow(np.ma.masked_invalid(z), cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")

    small = plt.rcParams["xtick.labelsize"]
    for ci in range(n_checks):
        for si in range(n_stages):
            st = status[ci][si]
            if st in ("pass", "fail"):
                dark = abs(z[ci, si] - 0.5) > 0.3
                ax.text(si, ci, cells[ci][si], ha="center", va="center", fontsize=small,
                        fontweight="bold", color="white" if dark else "black")
            else:
                color, ls = _NON_VERDICT_STYLE[st]
                ax.add_patch(Rectangle((si - 0.38, ci - 0.34), 0.76, 0.68, fill=False,
                                       edgecolor=color, linestyle=ls, linewidth=1.2))
                ax.text(si, ci, cells[ci][si], ha="center", va="center", fontsize=small, color=color)

    ax.set_xticks(range(n_stages), stage_labels)
    ax.set_yticks(range(n_checks), [CHECK_ABBREVIATIONS.get(n, n) for n in check_names])
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(-0.5, n_stages), minor=True)
    ax.set_yticks(np.arange(-0.5, n_checks), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_ylabel("Audit check")

    present = {s for row in status for s in row}
    handles = [
        Patch(fill=False, edgecolor=_NON_VERDICT_STYLE[st][0], linestyle=_NON_VERDICT_STYLE[st][1], label=label)
        for st, label in (("report_only", "Report-only (no threshold)"), ("skipped", "Skipped (no data)"))
        if st in present
    ]
    if handles:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.02),
                  ncols=len(handles), frameon=False, fontsize=small)
    if title:
        ax.set_title(title, pad=24)
    fig.tight_layout()
    return fig


def plot_uncertainty_evolution(
    uncertainties: np.ndarray,
    title: str | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Line plot of the surrogate's uncertainty at each loop step.

    Parameters
    ----------
    uncertainties : array-like
        One value per step, e.g. ``[h["uncertainty"] for h in hook.history]``.
    title : str, optional
        Axes title, e.g. the surrogate's name.
    ax : matplotlib Axes, optional
        Draw into this axes instead of a new figure.
    """
    import matplotlib.pyplot as plt

    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
    else:
        fig = ax.figure
    ax.plot(np.arange(len(uncertainties)), uncertainties, color="C0")
    ax.set_xlabel("Step")
    ax.set_ylabel("Surrogate std")
    if title:
        ax.set_title(title, fontsize=9)
    ax.grid(False)
    if own_figure:
        fig.tight_layout()
    return fig


def plot_audit_evolution(
    pipeline: AuditPipeline,
    history: list[dict[str, Any]],
    snapshot_every: int = 5,
    title: str | None = "Audit checks over loop steps",
) -> Figure | None:
    """Each check's value over the loop, one panel per check.

    The pipeline is re-run on the first ``k`` steps of ``history`` for every
    ``k`` that is a multiple of ``snapshot_every`` (and on the full history).
    Dots are green (pass), red (fail) or grey (report-only or skipped). A
    dashed line marks the threshold; for checks with a tolerance band
    (``IntervalCoverage``, ``VarianceAlignment``) the band edges are drawn.

    Because the pipeline runs once per snapshot, pass a lightweight pipeline
    (fast checks only) rather than one with refit sweeps or bootstrap CIs.

    Returns
    -------
    matplotlib.figure.Figure or None
        ``None`` when the history is shorter than ``snapshot_every`` or no
        check produced a value.
    """
    import matplotlib.pyplot as plt

    n_steps = len(history)
    if n_steps < snapshot_every:
        return None

    snap_steps = list(range(snapshot_every, n_steps + 1, snapshot_every))
    if snap_steps[-1] != n_steps:
        snap_steps.append(n_steps)

    records: dict[str, tuple[list, list]] = {}
    status_at: dict[str, list[str]] = {}
    thresholds: dict[str, Any] = {}
    tolerances: dict[str, float] = {}

    for k in snap_steps:
        try:
            report = pipeline.run(history[:k])
        except Exception:
            continue
        for r in report.results:
            if r.value is None:
                continue
            _, status = result_status(r)
            records.setdefault(r.name, ([], []))[0].append(k)
            records[r.name][1].append(r.value)
            status_at.setdefault(r.name, []).append(status)
            if r.name not in thresholds and r.threshold is not None:
                thresholds[r.name] = r.threshold
            if r.name not in tolerances and r.details:
                tol = r.details.get("tolerance")
                if tol is not None:
                    tolerances[r.name] = tol

    if not records:
        return None

    check_names = list(records.keys())
    n_checks = len(check_names)
    ncols = 3
    nrows = (n_checks + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0, nrows * 1.9))
    axes_flat = np.array(axes).flatten()

    for i, name in enumerate(check_names):
        ax = axes_flat[i]
        xs, ys = records[name]
        statuses = status_at.get(name, ["pass"] * len(xs))
        colors = [_STATUS_COLOR[s] for s in statuses]
        ax.plot(xs, ys, color="C0", lw=1.2)
        ax.scatter(xs, ys, c=colors, s=18, zorder=3)

        t = thresholds.get(name)
        if t is not None:
            if isinstance(t, tuple):
                ax.axhline(t[0], color="k", lw=0.8, ls="--", alpha=0.5)
                ax.axhline(t[1], color="k", lw=0.8, ls="--", alpha=0.5)
            elif name in tolerances:
                tol = tolerances[name]
                ax.axhline(float(t) - tol, color="k", lw=0.8, ls="--", alpha=0.5)
                ax.axhline(float(t) + tol, color="k", lw=0.8, ls="--", alpha=0.5)
            else:
                ax.axhline(float(t), color="k", lw=0.8, ls="--", alpha=0.5)

        panel_title = name.replace("Check", "")
        if all(s in ("report_only", "skipped") for s in statuses):
            panel_title += " (report-only)"
        ax.set_title(panel_title, fontsize=plt.rcParams["legend.fontsize"])
        ax.set_xlabel("Step")
        ax.tick_params(labelsize=plt.rcParams["xtick.labelsize"])
        ax.grid(False)

    for j in range(n_checks, len(axes_flat)):
        axes_flat[j].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=plt.rcParams["font.size"])
    fig.tight_layout()
    return fig


def plot_calibration_curve(result: AuditResult, ax: Axes | None = None) -> Figure | None:
    """Reliability diagram for a ``CalibrationErrorCheck`` result.

    Plots observed against expected interval coverage; the shaded gap is the
    miscalibration. Reads ``confidence_levels`` and ``observed_fractions`` from
    ``result.details`` and returns ``None`` when they are absent.
    """
    import matplotlib.pyplot as plt

    d = result.details
    expected = d.get("confidence_levels")
    observed = d.get("observed_fractions")
    if expected is None or observed is None:
        return None

    expected = np.asarray(expected)
    observed = np.asarray(observed)
    ce = d.get("calibration_error", float("nan"))

    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
    else:
        fig = ax.figure
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Perfect calibration")
    ax.plot(expected, observed, color="C0", label="Observed")
    ax.fill_between(expected, expected, observed, alpha=0.15, color="C1", label="Miscalibration")
    if not np.isnan(ce):
        ax.text(
            0.05, 0.95, f"CE = {ce:.4f}",
            transform=ax.transAxes, va="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
            fontsize=9,
        )
    ax.set_xlabel("Expected coverage")
    ax.set_ylabel("Observed coverage")
    ax.legend(frameon=False)
    ax.grid(False)
    if own_figure:
        fig.tight_layout()
    return fig
