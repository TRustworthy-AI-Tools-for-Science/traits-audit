"""Replot the headline density figure with the Forrester oracle overlaid.

Loads the existing _results/committee_v0/query_density.csv (the pooled
queries already on disk — no rollouts re-run) and re-renders
query_density_headline.png via the updated render_headline_figure().
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from traits_audit.committee.analysis.density import (
    AGENT_NAMES,
    DensityResult,
    render_headline_figure,
)


def load_density_csv(csv_path: Path) -> DensityResult:
    by_agent_seed: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with csv_path.open() as f:
        header = f.readline().strip().split(",")
        assert header == ["agent", "seed", "x"], f"unexpected header: {header}"
        for line in f:
            agent, seed, x = line.strip().split(",")
            by_agent_seed[agent][int(seed)].append(float(x))

    queries_by_agent_seed: dict[str, dict[int, np.ndarray]] = {
        a: {s: np.asarray(xs) for s, xs in seeds.items()}
        for a, seeds in by_agent_seed.items()
    }
    queries_by_agent: dict[str, np.ndarray] = {
        a: np.concatenate(list(seeds.values()))
        for a, seeds in queries_by_agent_seed.items()
    }

    missing = set(AGENT_NAMES) - set(queries_by_agent)
    if missing:
        raise ValueError(f"CSV missing agents: {sorted(missing)}")

    any_agent = next(iter(queries_by_agent_seed.values()))
    n_seeds = len(any_agent)
    n_queries_one_seed = len(next(iter(any_agent.values())))
    n_episodes_per_seed = n_queries_one_seed // 100

    print(f"[replot] loaded {sum(len(v) for v in queries_by_agent.values())} "
          f"queries across {len(queries_by_agent)} agents x {n_seeds} seeds, "
          f"inferred {n_episodes_per_seed} eps/seed")

    return DensityResult(
        queries_by_agent=queries_by_agent,
        queries_by_agent_seed=queries_by_agent_seed,
        n_episodes_per_seed=n_episodes_per_seed,
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    csv_path = root / "_results" / "committee_v0" / "query_density.csv"
    png_path = root / "_results" / "committee_v0" / "query_density_headline.png"

    result = load_density_csv(csv_path)
    render_headline_figure(result, png_path)
    print(f"[replot] wrote {png_path}")


if __name__ == "__main__":
    main()
