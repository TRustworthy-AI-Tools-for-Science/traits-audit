"""Uncertainty state space: separability, orthogonality and participation ratio.

Reads the steering grid (one ``<problem>/<condition>/steps.csv`` per cell) and
analyses the metric trajectories of the reference arms (``random`` and ``lcb``
by default), as in Methods, "Uncertainty State Space Construction":

1. Separability: can the starting condition (calibrated, overconfident,
   underconfident, misspecified) be told apart from a campaign's metrics?
   Leave-one-seed-out nearest-centroid classification with all metrics, with
   each metric alone, with the fiber or system block, and with the mean sigma
   alone.  Chance is one over the number of conditions.
2. Orthogonality: the residual r_i of each metric's trajectory against the
   column basis of the fiber block (system metrics) and against the mean sigma
   (every metric).  r = 1 means orthogonal, r = 0 means fully explained.
3. Participation ratio of the eigenvalues lambda_i = s_i^2 / (T - 1):
   PR(fiber block), PR(all metrics) and their difference, per campaign.
4. Partition test: re-pair the inputs and observations of the acquisition
   history at random, keep the reference-set fibers (y, mu, sigma) fixed, and
   recompute every metric.  A metric that does not change is a function of the
   fibers only and belongs to the fiber block.

    python analyze_metric_space.py --runs /path/to/grid --problem forrester
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
CONDITIONS = ["calibrated", "overconfident", "underconfident", "misspecified"]


def load_steer_script():
    spec = importlib.util.spec_from_file_location("steer_state_space", SCRIPTS / "steer_state_space.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


steer = load_steer_script()
FIBER_METRICS = list(steer.METRICS)
SYSTEM_METRICS = list(steer.SYSTEM_METRICS)
ALL_METRICS = FIBER_METRICS + SYSTEM_METRICS


# ── Loading ───────────────────────────────────────────────────────────────────

def load_campaigns(runs, problem, arms, conditions):
    """{(condition, arm, seed): {"metrics": T x m raw badness, "sigma": T mean sigma}}, steering phase only."""
    campaigns = {}
    for condition in conditions:
        path = Path(runs) / problem / condition / "steps.csv"
        if not path.exists():
            print(f"  missing {path}, skipped")
            continue
        rows_by_campaign = {}
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                if row["phase"] != "steer" or row["policy"] not in arms:
                    continue
                key = (condition, row["policy"], int(row["seed"]))
                rows_by_campaign.setdefault(key, []).append(row)
        for key, rows in rows_by_campaign.items():
            rows.sort(key=lambda row: int(row["step"]))
            metrics = np.array([[float(row[f"raw:{name}"]) for name in ALL_METRICS] for row in rows])
            sigma = np.array([float(row["mean_sigma"]) for row in rows])
            campaigns[key] = {"metrics": metrics, "sigma": sigma}
    return campaigns


# ── Linear algebra ────────────────────────────────────────────────────────────

def zscore_columns(matrix):
    """Centre and scale each column; columns that never change are dropped."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    spread = matrix.std(axis=0)
    keep = spread > 1e-12
    return (matrix[:, keep] - matrix[:, keep].mean(axis=0)) / spread[keep]


def participation_ratio(matrix):
    """(sum lambda)^2 / sum lambda^2 for the eigenvalues lambda of the column covariance (NaN if empty)."""
    z = zscore_columns(matrix)
    if z.shape[1] == 0:
        return np.nan
    singular_values = np.linalg.svd(z, compute_uv=False)
    eigenvalues = singular_values ** 2 / (len(z) - 1)
    return float(eigenvalues.sum() ** 2 / np.sum(eigenvalues ** 2))


def orthogonal_residual(column, basis_columns):
    """||v - U U^T v|| / ||v|| for z-scored v and an orthonormal basis U of the basis columns."""
    v = zscore_columns(column)
    basis = zscore_columns(basis_columns)
    if v.shape[1] == 0:
        return np.nan
    if basis.shape[1] == 0:
        return 1.0
    v = v[:, 0]
    U, s, _ = np.linalg.svd(basis, full_matrices=False)
    U = U[:, s > 1e-10 * s.max()]
    leftover = v - U @ (U.T @ v)
    return float(np.linalg.norm(leftover) / np.linalg.norm(v))


# ── 1. Separability ───────────────────────────────────────────────────────────

def campaign_features(campaign, columns):
    """One row per campaign: the mean of each chosen column over the steering phase."""
    table = np.column_stack([campaign["metrics"], campaign["sigma"]])
    return table[:, columns].mean(axis=0)


def leave_one_seed_out_accuracy(campaigns, columns):
    """Share of campaigns whose condition the nearest training-set centroid gets right."""
    keys = sorted(campaigns)
    labels = np.array([key[0] for key in keys])
    seeds = np.array([key[2] for key in keys])
    features = np.array([campaign_features(campaigns[key], columns) for key in keys])
    correct = 0
    for held_out in np.unique(seeds):
        train = seeds != held_out
        test = seeds == held_out
        centre = features[train].mean(axis=0)
        spread = features[train].std(axis=0)
        spread[spread < 1e-12] = 1.0
        scaled_train = (features[train] - centre) / spread
        scaled_test = (features[test] - centre) / spread
        classes = np.unique(labels[train])
        centroids = np.array([scaled_train[labels[train] == c].mean(axis=0) for c in classes])
        for row, truth in zip(scaled_test, labels[test], strict=True):
            distances = np.linalg.norm(centroids - row, axis=1)
            correct += int(classes[np.argmin(distances)] == truth)
    return correct / len(keys)


def separability(campaigns):
    sigma_column = len(ALL_METRICS)
    fiber = [ALL_METRICS.index(name) for name in FIBER_METRICS]
    system = [ALL_METRICS.index(name) for name in SYSTEM_METRICS]
    feature_sets = {"all metrics": fiber + system, "fiber block": fiber, "system block": system,
                    "mean sigma": [sigma_column]}
    for i, name in enumerate(ALL_METRICS):
        feature_sets[name] = [i]
    n_conditions = len({key[0] for key in campaigns})
    result = {"chance": 1.0 / n_conditions}
    for label, columns in feature_sets.items():
        result[label] = leave_one_seed_out_accuracy(campaigns, columns)
    return result


# ── 2 and 3. Orthogonality and participation ratio, per campaign ──────────────

def per_campaign_geometry(campaign):
    metrics = campaign["metrics"]
    fiber = metrics[:, :len(FIBER_METRICS)]
    geometry = {"PR fiber": participation_ratio(fiber), "PR all": participation_ratio(metrics)}
    geometry["PR increase"] = geometry["PR all"] - geometry["PR fiber"]
    for i, name in enumerate(ALL_METRICS):
        geometry[f"r vs sigma: {name}"] = orthogonal_residual(metrics[:, i], campaign["sigma"])
    for i, name in enumerate(SYSTEM_METRICS):
        column = metrics[:, len(FIBER_METRICS) + i]
        geometry[f"r vs fiber: {name}"] = orthogonal_residual(column, fiber)
    return geometry


def summarise(values):
    """Median and a 95 % bootstrap interval of the median, ignoring NaN."""
    values = np.asarray([v for v in values if np.isfinite(v)])
    if len(values) == 0:
        return {"median": np.nan, "low": np.nan, "high": np.nan, "n": 0}
    rng = np.random.default_rng(0)
    medians = [np.median(rng.choice(values, size=len(values))) for _ in range(2000)]
    return {"median": float(np.median(values)), "low": float(np.percentile(medians, 2.5)),
            "high": float(np.percentile(medians, 97.5)), "n": len(values)}


def geometry_by_condition(campaigns):
    result = {}
    for condition in sorted({key[0] for key in campaigns}):
        per_campaign = [per_campaign_geometry(c) for key, c in campaigns.items() if key[0] == condition]
        result[condition] = {name: summarise([g[name] for g in per_campaign]) for name in per_campaign[0]}
    return result


# ── 4. Partition test ─────────────────────────────────────────────────────────

def partition_test(problem_name, condition, seeds, steps):
    """Largest change of each metric when the history's (x, y) pairing is shuffled, fibers fixed.

    Changes are in units of the metric's spread over the seeds' unshuffled values.
    """
    original_values = {name: [] for name in ALL_METRICS}
    changes = {name: [] for name in ALL_METRICS}
    for seed in range(seeds):
        args = steer.parse_args(["--problem", problem_name, "--condition", condition, "--steps", str(steps)])
        problem = steer.make_problem(problem_name)
        campaign = steer.Campaign(problem, condition, "random", seed, args, steer.DEFAULT_STEERING)
        query_rng = np.random.default_rng([seed, 42])
        while len(campaign.x_observed) < steps:
            campaign.query(query_rng.uniform(0.0, 1.0, problem.dim))
        X = np.array(campaign.x_observed)
        Y = np.array(campaign.y_observed)
        campaign.surrogate.fit(X, Y, refit_number=len(X))
        MU_ref, SD_ref = campaign.surrogate.predict(campaign.grid)
        kernels = campaign.surrogate.fitted_kernels()
        shuffled_Y = Y[np.random.default_rng([seed, 43]).permutation(len(Y))]

        def all_metrics(history_Y):
            values, _ = steer.score_metrics(campaign.Y_ref, MU_ref, SD_ref)
            system, _ = steer.score_system_metrics(problem, condition, X, history_Y, campaign.grid,
                                                   campaign.Y_ref, MU_ref, SD_ref, kernels, args,
                                                   seed=[seed, 6, len(X)])
            values.update(system)
            return values

        original = all_metrics(Y)
        shuffled = all_metrics(shuffled_Y)
        for name in ALL_METRICS:
            original_values[name].append(original[name])
            changes[name].append(abs(shuffled[name] - original[name]))
    result = {}
    for name in ALL_METRICS:
        spread = max(float(np.std(original_values[name])), 1e-12)
        largest = float(np.max(changes[name]))
        result[name] = {"largest change": largest, "in spreads": largest / spread,
                        "fiber only": bool(largest < 1e-9)}
    return result


# ── Command line ──────────────────────────────────────────────────────────────

def print_report(problem, separation, geometry, partition):
    print(f"\n######## {problem}")
    print(f"\n1. Separability (leave-one-seed-out accuracy; chance {separation['chance']:.2f})")
    for label, accuracy in separation.items():
        if label != "chance":
            print(f"  {label:32s} {accuracy:.2f}")

    print("\n2-3. Geometry per condition: median [95 % interval]")
    for condition, entries in geometry.items():
        print(f"  {condition}")
        for name, s in entries.items():
            print(f"    {name:48s} {s['median']:6.3f} [{s['low']:6.3f}, {s['high']:6.3f}]  n={s['n']}")

    if partition:
        print("\n4. Partition test (shuffled history pairing, fibers fixed)")
        for condition, entries in partition.items():
            print(f"  {condition}")
            for name, entry in entries.items():
                verdict = "fiber only" if entry["fiber only"] else "reads the history"
                print(f"    {name:32s} change {entry['in spreads']:8.3f} spreads   {verdict}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs", type=Path, required=True, help="grid root holding <problem>/<condition>/steps.csv")
    p.add_argument("--problem", required=True)
    p.add_argument("--arms", nargs="+", default=["random", "lcb"])
    p.add_argument("--conditions", nargs="+", default=CONDITIONS)
    p.add_argument("--partition-seeds", type=int, default=5, help="seeds for the partition test (0 skips it)")
    p.add_argument("--partition-steps", type=int, default=60, help="history length for the partition test")
    p.add_argument("--out", type=Path, default=None, help="output JSON (default: <runs>/<problem>/metric_space.json)")
    args = p.parse_args(argv)

    campaigns = load_campaigns(args.runs, args.problem, args.arms, args.conditions)
    if not campaigns:
        raise SystemExit("no campaigns found")
    separation = separability(campaigns)
    geometry = geometry_by_condition(campaigns)
    partition = {}
    if args.partition_seeds > 0:
        for condition in sorted({key[0] for key in campaigns}):
            partition[condition] = partition_test(args.problem, condition, args.partition_seeds,
                                                  args.partition_steps)

    print_report(args.problem, separation, geometry, partition)
    out = args.out or args.runs / args.problem / "metric_space.json"
    out.write_text(json.dumps({"problem": args.problem, "arms": args.arms, "separability": separation,
                               "geometry": geometry, "partition_test": partition}, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
