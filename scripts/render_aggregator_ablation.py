"""Dense leave-one-out figure: 5 aggregator rules x 15 committee members.

One figure per problem. Each panel is one aggregator rule; each bar is the
change in terminal score when that member is dropped from the committee.
Positive = dropping the member made the rule worse.

Bars are drawn pale unless the paired Wilcoxon against the full committee
clears 0.05, so a panel of large-but-noisy deltas cannot be misread as
member importance. The 95% CI whisker is on every bar for the same reason.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, false_discovery_control, wilcoxon

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from traits_audit.committee.analysis import style as st

PROBLEMS = {
    "forrester": ("_results/committee_v1", "thread_a_regret_all.csv",
                  "simple regret", "Forrester"),
    "branin-currin-mo": ("_results/committee_branin-currin-mo",
                         "thread_a_regret.csv",
                         r"$\log_{10}(HV_{\max}-HV)$", "Branin--Currin (MO)"),
    "color": ("_results/committee_color", "thread_a_regret_all.csv",
              "simple regret", "Colour matching"),
}

RULES = ["uniform", "agree", "disagree", "weighted-indep", "weighted-invreg"]


def baseline_terminal(regret_csv: Path, rule: str) -> np.ndarray:
    df = pd.read_csv(regret_csv)
    f = df[df.policy == f"committee:{rule}"]
    f = f[f.step == f.step.max()].sort_values("episode_seed")
    return f["simple_regret"].to_numpy()


def render(problem: str) -> None:
    out_dir, regret_name, xlabel, title = PROBLEMS[problem]
    base_dir = REPO / out_dir / "threadA"

    fig, axes = plt.subplots(1, len(RULES), figsize=(23, 7.2), sharey=True)
    any_data = False

    for ax, rule in zip(axes, RULES):
        path = base_dir / f"thread_a_ablation_{rule}.csv"
        if not path.exists():
            ax.set_title(f"committee:{rule}\n(missing)", fontsize=st.TITLE_FS - 3)
            ax.axis("off")
            continue
        any_data = True
        abl = pd.read_csv(path)
        base = baseline_terminal(base_dir / regret_name, rule)

        stats = []
        for agent, g in abl.groupby("dropped_agent"):
            v = g.sort_values("episode_seed")["terminal_regret"].to_numpy()
            n = min(len(v), len(base))
            d = v[:n] - base[:n]
            # All-zero differences: older scipy raised, current scipy returns
            # NaN (0/0 in the z statistic). Both mean "no detectable effect",
            # and a NaN would blow up false_discovery_control downstream.
            try:
                p = float(wilcoxon(v[:n], base[:n]).pvalue)
            except ValueError:            # all-zero differences
                p = 1.0
            if not np.isfinite(p):
                p = 1.0
            sem = d.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
            stats.append((agent, float(d.mean()), 1.96 * sem, p))

        stats.sort(key=lambda r: r[1], reverse=True)
        names = [s[0] for s in stats]
        vals = [s[1] for s in stats]
        errs = [s[2] for s in stats]
        # Benjamini-Hochberg across the 15 members of this panel. Per-member
        # significance is only claimed after correction; with 15 tests at
        # alpha=0.05 the uncorrected hits are not interpretable on their own.
        ps = list(false_discovery_control(np.array([s[3] for s in stats]),
                                          method="bh"))

        # The robust statement is directional and pre-specified: does dropping
        # an arbitrary member tend to hurt? One test per panel, no multiplicity.
        #
        # Guard against degenerate panels before testing direction. Some rules
        # reach the same terminal value whichever member is dropped (colour
        # `agree` is bit-identical across all 15; MO `agree` has 12 exact zeros
        # and three deltas ~1e-3 against a baseline of 1.68). The sign test
        # then reports p ~ 1e-5 off what is numerically a tie, so require the
        # between-drop spread to be material against seed noise before
        # claiming a direction.
        spread = max(vals) - min(vals)
        noise = float(np.std(base, ddof=1)) if len(base) > 1 else 0.0
        n_zero = sum(abs(v) <= 1e-12 for v in vals)
        # Degenerate = dropping a member does not change the score: either the
        # drops return identical values (colour and MO `agree`), or the whole
        # between-drop spread is a rounding error next to one seed's spread.
        #
        # Deliberately NOT gating on effect size relative to seed noise: on
        # colour the seed std is ~3900 because the problem saturates, so a real
        # directional effect is only ~0.14x noise there and such a rule would
        # suppress it. Effect size is reported next to the test instead, as a
        # percentage of the rule's own baseline, and left to the reader.
        largest = max(abs(v) for v in vals)
        baseline = abs(float(np.mean(base))) or 1.0
        effect_pct = 100.0 * largest / baseline
        degenerate = (spread <= 0.05 * noise) or (n_zero >= len(vals) // 2)
        n_pos = int(sum(v > 0 for v in vals))
        sign_p = (float("nan") if degenerate
                  else float(binomtest(n_pos, len(vals), 0.5).pvalue))

        y = np.arange(len(names))
        for i, (v, e, p) in enumerate(zip(vals, errs, ps)):
            sig = p < 0.05
            col = st.VERMILLION if v > 0 else st.BLUISH_GREEN
            ax.barh(i, v, color=col, alpha=0.95 if sig else 0.30,
                    edgecolor=col if sig else "none",
                    linewidth=1.2 if sig else 0)
            ax.errorbar(v, i, xerr=e, fmt="none", ecolor=st.BLACK,
                        elinewidth=0.8, capsize=2, alpha=0.8)
            if sig:
                ax.text(v + (e if v > 0 else -e), i, "  *" if v > 0 else "*  ",
                        va="center", ha="left" if v > 0 else "right",
                        fontsize=st.TICK_FS)

        ax.axvline(0, color=st.BLACK, lw=1.0)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=st.TICK_FS - 1)
        ax.invert_yaxis()
        # Two short lines rather than one long one: at five panels across, a
        # single-line subtitle overruns into its neighbours.
        subtitle = ("degenerate\nscore unchanged by any drop"
                    if degenerate else
                    f"{n_pos}/{len(vals)} positive, $p={sign_p:.3g}$\n"
                    f"largest drop {effect_pct:.1f}% of baseline")
        ax.set_title(f"committee:{rule}\n{subtitle}", fontsize=st.TITLE_FS - 4)
        ax.tick_params(axis="x", labelsize=st.TICK_FS - 1)
        ax.grid(axis="x", alpha=0.3)
        ax.set_xlabel(rf"$\Delta$ {xlabel}", fontsize=st.LABEL_FS - 2)

    if not any_data:
        plt.close(fig)
        print(f"[{problem}] no ablation CSVs yet; skipped")
        return

    fig.suptitle(
        f"{title}: leave-one-out over committee members, per aggregation rule\n"
        "positive = dropping that member made the rule worse;  "
        "solid bar + * = Benjamini-Hochberg q < 0.05 across the 15 members;  "
        "whiskers are 95% CIs",
        fontsize=st.TITLE_FS - 2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out = base_dir / "a4_member_ablation.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[{problem}] wrote {out}")


if __name__ == "__main__":
    for p in (sys.argv[1:] or list(PROBLEMS)):
        render(p)
