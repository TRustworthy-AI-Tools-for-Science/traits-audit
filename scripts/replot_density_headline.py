"""Re-render a headline query-density figure from the queries already on disk.

Loads ``<results-dir>/query_density.csv`` (the pooled queries written by
``ta-committee-analyze density``) and re-runs ``render_headline_figure()``
on it. No rollouts are re-run and no models are loaded, so this is the
cheap path for cosmetic changes to the figure -- alpha, z-order, colours.

Works for any problem: the input dimension is read off the CSV header
(``x`` for Forrester, ``x0,x1`` for Branin-Currin, ``x0,x1,x2`` for colour).

Usage:
    python scripts/replot_density_headline.py forrester
    python scripts/replot_density_headline.py branin-currin
    python scripts/replot_density_headline.py color

    # or point it anywhere:
    python scripts/replot_density_headline.py color --results-dir _results/foo
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from traits_audit.committee.analysis.density import (
    AGENT_NAMES,
    DensityResult,
    render_headline_figure,
)
from traits_audit.committee.problems import get_problem

# Where each problem's figures live by default, matching the layout that
# scripts/make_problem_figures.sh writes.
DEFAULT_RESULTS_DIR: dict[str, str] = {
    "forrester": "_results/committee_v0",
    "branin-currin": "_results/committee_branin-currin",
    "color": "_results/committee_color",
}


def load_density_csv(csv_path: Path, episode_length: int = 100) -> DensityResult:
    """Read a tidy query-density CSV back into a :class:`DensityResult`.

    Mirrors ``density.write_density_csv``: one row per query, columns
    ``agent, seed, x`` (1-D) or ``agent, seed, x0..x{dim-1}``.
    """
    by_agent_seed: dict[str, dict[int, list[list[float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with csv_path.open() as f:
        header = f.readline().strip().split(",")
        if header[:2] != ["agent", "seed"]:
            raise ValueError(f"unexpected header: {header}")
        x_cols = header[2:]
        if not x_cols:
            raise ValueError(f"no coordinate columns in header: {header}")
        dim = len(x_cols)
        for line in f:
            agent, seed, *coords = line.strip().split(",")
            by_agent_seed[agent][int(seed)].append([float(c) for c in coords])

    # Keep the 1-D shape convention of CommitteeEnv.x_obs: flat for dim==1,
    # (n, dim) otherwise -- render_headline_figure reshapes either way, but
    # the per-seed step overlay on the Forrester panels wants 1-D.
    def _pack(rows: list[list[float]]) -> np.ndarray:
        arr = np.asarray(rows, dtype=float)
        return arr[:, 0] if dim == 1 else arr

    queries_by_agent_seed = {
        a: {s: _pack(rows) for s, rows in seeds.items()}
        for a, seeds in by_agent_seed.items()
    }
    queries_by_agent = {
        a: np.concatenate(list(seeds.values()), axis=0)
        for a, seeds in queries_by_agent_seed.items()
    }

    missing = set(AGENT_NAMES) - set(queries_by_agent)
    if missing:
        raise ValueError(f"CSV missing agents: {sorted(missing)}")

    any_agent = next(iter(queries_by_agent_seed.values()))
    n_seeds = len(any_agent)
    n_episodes_per_seed = len(next(iter(any_agent.values()))) // episode_length

    total = sum(len(v) for v in queries_by_agent.values())
    print(f"[replot] loaded {total} queries ({dim}-D) across "
          f"{len(queries_by_agent)} agents x {n_seeds} seeds, "
          f"inferred {n_episodes_per_seed} eps/seed")

    return DensityResult(
        queries_by_agent=queries_by_agent,
        queries_by_agent_seed=queries_by_agent_seed,
        n_episodes_per_seed=n_episodes_per_seed,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("problem", choices=list(DEFAULT_RESULTS_DIR))
    ap.add_argument("--results-dir", type=Path, default=None,
                    help="Override the default results directory.")
    ap.add_argument("--n-bins", type=int, default=30)
    ap.add_argument("--episode-length", type=int, default=100)
    ap.add_argument("--channels", action="store_true",
                    help="Also write query_channels.png, the per-channel "
                         "marginals (shows bound-saturation the CIE "
                         "projection hides). Multi-dimensional problems only.")
    ap.add_argument("--surface3d", action="store_true",
                    help="Also write query_surface3d.png, queries drawn on "
                         "the true surface in 3-D. 2-D problems only.")
    ap.add_argument("--max-points", type=int, default=1500,
                    help="Markers per panel for --surface3d.")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    results_dir = args.results_dir or (root / DEFAULT_RESULTS_DIR[args.problem])
    csv_path = results_dir / "query_density.csv"
    png_path = results_dir / "query_density_headline.png"

    problem = get_problem(args.problem)
    result = load_density_csv(csv_path, episode_length=args.episode_length)
    render_headline_figure(
        result, png_path, n_bins=args.n_bins, problem=problem,
    )
    print(f"[replot] wrote {png_path}")

    if args.channels:
        if problem.dim == 1:
            raise SystemExit(
                "--channels needs a multi-dimensional problem; the 1-D "
                "headline figure is already a marginal."
            )
        from traits_audit.committee.analysis.channel_marginals import (
            render_channel_marginals,
        )
        ch_path = results_dir / "query_channels.png"
        render_channel_marginals(result, ch_path, problem=problem)
        print(f"[replot] wrote {ch_path}")

    if args.surface3d:
        if problem.dim != 2:
            raise SystemExit(
                f"--surface3d needs a 2-D problem; {args.problem} is "
                f"{problem.dim}-D."
            )
        from traits_audit.committee.analysis.surface3d import render_surface3d
        s3_path = results_dir / "query_surface3d.png"
        render_surface3d(result, s3_path, problem=problem,
                         max_points=args.max_points)
        print(f"[replot] wrote {s3_path}")


if __name__ == "__main__":
    main()
