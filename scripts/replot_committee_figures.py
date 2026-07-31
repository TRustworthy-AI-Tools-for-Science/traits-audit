"""Re-render committee figures from CSVs without re-running rollouts.

Loads `_results/committee_v0/regret.csv`, `query_density.csv`, and the
Thread A/B CSVs, reconstructs the result dataclasses, and writes new PNGs
with the updated style. Used after stylistic changes to avoid re-running
the full bake-off.

Does NOT touch:
    - correlation_*.png (no style update requested)
    - b2_ablation_maxsigma.png (needs a fresh ablation run)
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# Density.
# ---------------------------------------------------------------------------

def replot_density() -> None:
    from traits_audit.committee.analysis.density import (
        AGENT_NAMES, DensityResult, render_headline_figure,
    )

    csv = Path("_results/committee_v0/query_density.csv")
    by_agent_seed: dict[str, dict[int, list[float]]] = {
        n: defaultdict(list) for n in AGENT_NAMES
    }
    with csv.open() as fh:
        header = fh.readline()
        for line in fh:
            agent, seed, x = line.strip().split(",")
            by_agent_seed[agent][int(seed)].append(float(x))

    queries_by_agent_seed = {
        n: {s: np.asarray(xs, dtype=float) for s, xs in by_seed.items()}
        for n, by_seed in by_agent_seed.items()
    }
    queries_by_agent = {
        n: np.concatenate(list(d.values())) for n, d in queries_by_agent_seed.items()
    }
    # n_episodes_per_seed: total queries / (n_seeds * 100). Not strictly
    # needed for the plot; we just pass the canonical value.
    n_eps = 50
    result = DensityResult(
        queries_by_agent=queries_by_agent,
        queries_by_agent_seed=queries_by_agent_seed,
        n_episodes_per_seed=n_eps,
    )
    out = Path("_results/committee_v0/query_density_headline.png")
    render_headline_figure(result, out)
    print(f"[replot] wrote {out}")


# ---------------------------------------------------------------------------
# v0 regret.
# ---------------------------------------------------------------------------

def replot_regret() -> None:
    from traits_audit.committee.analysis.regret import (
        RegretResult, render_regret_figure,
    )

    csv = Path("_results/committee_v0/regret.csv")
    per_policy: dict[str, dict[int, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    with csv.open() as fh:
        fh.readline()
        for line in fh:
            policy, es, step, sr = line.strip().split(",")
            per_policy[policy][int(es)][int(step)] = float(sr)

    # Re-shape to (n_seeds, episode_length) per policy.
    any_policy = next(iter(per_policy))
    seeds = sorted(per_policy[any_policy].keys())
    episode_length = max(per_policy[any_policy][seeds[0]].keys()) + 1
    arrs: dict[str, np.ndarray] = {}
    for policy, by_seed in per_policy.items():
        arr = np.zeros((len(seeds), episode_length), dtype=float)
        for i, s in enumerate(seeds):
            for t, v in by_seed[s].items():
                arr[i, t] = v
        arrs[policy] = arr
    result = RegretResult(
        per_policy=arrs,
        seeds=seeds,
        episode_length=episode_length,
        warmstart_n=20,
    )
    out = Path("_results/committee_v0/regret.png")
    render_regret_figure(result, out)
    print(f"[replot] wrote {out}")


# ---------------------------------------------------------------------------
# Thread B / Thread A: shared CSV loader.
# ---------------------------------------------------------------------------

def _load_thread_csv(csv_path: Path):
    from traits_audit.committee.analysis.thread_regret import ThreadResult

    rows = csv_path.read_text().splitlines()
    header = rows[0]
    per_policy_regret: dict[str, dict[int, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    per_policy_std: dict[str, dict[int, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for line in rows[1:]:
        parts = line.split(",")
        policy, es, step, sr = parts[0], int(parts[1]), int(parts[2]), float(parts[3])
        std_str = parts[4] if len(parts) > 4 else ""
        per_policy_regret[policy][es][step] = sr
        per_policy_std[policy][es][step] = (
            float(std_str) if std_str else float("nan")
        )

    any_policy = next(iter(per_policy_regret))
    seeds = sorted(per_policy_regret[any_policy].keys())
    episode_length = max(per_policy_regret[any_policy][seeds[0]].keys()) + 1
    reg_arrs: dict[str, np.ndarray] = {}
    std_arrs: dict[str, np.ndarray] = {}
    for policy, by_seed in per_policy_regret.items():
        arr = np.zeros((len(seeds), episode_length), dtype=float)
        sarr = np.zeros((len(seeds), episode_length), dtype=float)
        for i, s in enumerate(seeds):
            for t, v in by_seed[s].items():
                arr[i, t] = v
            for t, v in per_policy_std[policy][s].items():
                sarr[i, t] = v
        reg_arrs[policy] = arr
        std_arrs[policy] = sarr
    return ThreadResult(
        per_policy_regret=reg_arrs,
        per_policy_action_std=std_arrs,
        seeds=seeds,
        episode_length=episode_length,
        warmstart_n=20,
    )


def _load_lcb_ablation(csv_path: Path) -> dict[str, np.ndarray]:
    rows = csv_path.read_text().splitlines()[1:]
    by_agent: dict[str, list[float]] = defaultdict(list)
    for line in rows:
        agent, _es, v = line.split(",")
        by_agent[agent].append(float(v))
    return {a: np.asarray(vs, dtype=float) for a, vs in by_agent.items()}


# ---------------------------------------------------------------------------
# Thread B figures.
# ---------------------------------------------------------------------------

def replot_thread_b() -> None:
    from traits_audit.committee.analysis.thread_figures import (
        render_ablation, render_b1,
    )

    out_dir = Path("_results/committee_v1_threadB")
    result = _load_thread_csv(out_dir / "thread_b_regret.csv")
    tests = render_b1(result, out_dir / "b1_regret_paired.png")
    print(f"[replot] wrote {out_dir/'b1_regret_paired.png'}")
    (out_dir / "thread_b_tests.json").write_text(json.dumps(tests, indent=2) + "\n")

    # B2 — LCB+votes ablation re-render from old CSV (renamed in-place if needed).
    old_csv = out_dir / "thread_b_ablation.csv"
    new_csv = out_dir / "thread_b_ablation_lcb.csv"
    if old_csv.exists() and not new_csv.exists():
        old_csv.rename(new_csv)
    if new_csv.exists():
        ablation = _load_lcb_ablation(new_csv)
        baseline_terminal = result.per_policy_regret["LCB+votes"][:, -1]
        from traits_audit.committee.analysis.density import AGENT_NAMES
        render_ablation(
            ablation, baseline_terminal,
            agent_names=AGENT_NAMES,
            output_path=out_dir / "b2_ablation_lcb.png",
            headline="LCB+votes",
        )
        print(f"[replot] wrote {out_dir/'b2_ablation_lcb.png'}")

    # B2 max-σ+votes ablation re-render.
    maxsigma_csv = out_dir / "thread_b_ablation_maxsigma.csv"
    if maxsigma_csv.exists():
        ablation = _load_lcb_ablation(maxsigma_csv)
        baseline_terminal = result.per_policy_regret["max-sigma+votes"][:, -1]
        from traits_audit.committee.analysis.density import AGENT_NAMES
        render_ablation(
            ablation, baseline_terminal,
            agent_names=AGENT_NAMES,
            output_path=out_dir / "b2_ablation_maxsigma.png",
            headline="max-sigma+votes",
        )
        print(f"[replot] wrote {out_dir/'b2_ablation_maxsigma.png'}")


# ---------------------------------------------------------------------------
# Thread A figures.
# ---------------------------------------------------------------------------

def replot_thread_a() -> None:
    from traits_audit.committee.analysis.thread_figures import (
        render_a1, render_a2, render_a3,
    )

    out_dir = Path("_results/committee_v1_threadA")
    result = _load_thread_csv(out_dir / "thread_a_regret.csv")
    tests = render_a1(result, out_dir / "a1_aggregator_bakeoff.png")
    (out_dir / "thread_a_tests.json").write_text(json.dumps(tests, indent=2) + "\n")
    print(f"[replot] wrote {out_dir/'a1_aggregator_bakeoff.png'}")

    weights = json.loads((out_dir / "thread_a_weights.json").read_text())
    indep_w = weights["independence"]
    invreg_w = weights["inverse_regret"]
    regret_test = json.loads(
        Path("_results/committee_v0/regret_test.json").read_text()
    )
    from traits_audit.committee.analysis.density import AGENT_NAMES
    solo_terminal = {
        a: float(regret_test["solo_means"][f"solo:{a}"]) for a in AGENT_NAMES
    }
    render_a2(indep_w, invreg_w, solo_terminal,
              out_dir / "a2_weight_vs_regret.png")
    print(f"[replot] wrote {out_dir/'a2_weight_vs_regret.png'}")
    render_a3(result, out_dir / "a3_disagreement.png")
    print(f"[replot] wrote {out_dir/'a3_disagreement.png'}")


def main() -> None:
    replot_density()
    replot_regret()
    replot_thread_b()
    replot_thread_a()


if __name__ == "__main__":
    main()
