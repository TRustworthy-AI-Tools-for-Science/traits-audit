#!/usr/bin/env python
"""Test the pre-registered predictions on one steering run (one problem, one condition).

Reads ``steps.csv`` + ``config.json`` written by ``steer_state_space.py``.  The
predictions are those of ``paper/steering_experiment_spec.md``:

P1  Steered metrics improve:          ΔS(policy) − ΔS(reference) < 0
P2  Audit-only metrics do not worsen: ΔH(policy) − ΔH(reference) ≥ 0
P3  Co-movement of S and H breaks:    Spearman ρ of first differences, late vs early half
P4  Closing the loop changes A:       ‖A_late − A_early‖ / ‖A_early‖
P5  Outcome: final score, failure rate, stuck rate, link to the warm start
P6  Relative recovery (needs ``--reference``, the calibrated run of the same
    problem): the share of the starting distance to the calibrated run's end
    level that a campaign closes (1 = reached it, 0 = no progress, < 0 = worse).
    The pathological conditions are fixed properties of the surrogate, so full
    recovery is not expected; the share closed is comparable across arms.
P7  Stability: windowed spectral radius ρ(A) of the metric state, when it
    settles below 1, and whether healthier end states go with smaller ρ

S_t / H_t are the mean z-scored badness of the steering / audit-only metrics
(higher = worse); Δ is the mean over the last ``--window`` steering steps minus
the mean over the first ``--window``.  Comparisons are paired by seed, with the
Wilcoxon signed-rank test; P1, P2 and P4 are reported against ``random`` (the
metrics drift under any policy as data accumulates) and P2 also against
``state-space`` (the same dynamics without the health term), which separates
gaming from the effect of concentrating queries.

Writes ``steering_summary.json`` (per policy) and ``steering_run.json`` (per run),
both read by ``aggregate_steering.py``, and ``steering_summary.png``.

Usage::

    python experiments/results/scripts/analyze_steering.py --in RUN_DIR [--reference CALIBRATED_RUN_DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats
from traits_audit.stability.dmdc import transition_step, windowed_rho

PERSIST_STEPS = 3


# ── Loading ───────────────────────────────────────────────────────────────────

def load_run(path):
    """Campaigns of one run: {(policy, seed): campaign dict} and the run's config."""
    config = json.loads((Path(path) / "config.json").read_text())
    with open(Path(path) / "steps.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))

    metrics = config["steering"] + config["audit_only"]
    rows_by_campaign = {}
    for row in rows:
        key = (row["policy"], int(row["seed"]))
        rows_by_campaign.setdefault(key, []).append(row)

    campaigns = {}
    for key, campaign_rows in rows_by_campaign.items():
        campaign_rows.sort(key=lambda row: int(row["step"]))
        warm = [row for row in campaign_rows if row["phase"] == "warm"]
        steer = [row for row in campaign_rows if row["phase"] == "steer"]
        Z = np.array([[float(row[f"z:{m}"]) for m in metrics] for row in steer])
        n_steering = len(config["steering"])
        campaigns[key] = {
            "step": np.array([int(row["step"]) for row in steer]),
            "x": np.array([[float(row[f"x{i}"]) for i in range(config["dim"])] for row in steer]),
            "score": np.array([float(row["score"]) for row in steer]),
            "warm_score": float(warm[-1]["score"]),
            "Z": Z,
            "S": Z[:, :n_steering].mean(axis=1),
            "H": Z[:, n_steering:].mean(axis=1) if Z.shape[1] > n_steering else np.zeros(len(steer)),
            "raw_steering": np.array([[float(row[f"raw:{m}"]) for m in config["steering"]] for row in steer]),
        }
    return campaigns, config


# ── Per-campaign measures ─────────────────────────────────────────────────────

def change(series, window):
    return float(series[-window:].mean() - series[:window].mean())


def comovement(S, H):
    """Spearman ρ of the first differences of S and H, early half and late half."""
    dS = np.diff(S)
    dH = np.diff(H)
    half = len(dS) // 2
    result = []
    for a, b in ((dS[:half], dH[:half]), (dS[half:], dH[half:])):
        if np.std(a) > 0 and np.std(b) > 0:
            result.append(float(stats.spearmanr(a, b).statistic))
        else:
            result.append(np.nan)
    return result


def fit_A(Z, x, ridge=1e-2):
    """Ridge least squares for z_{t+1} = A z_t + B u_t, with u_t the query that produced z_{t+1}."""
    inputs = np.hstack([Z[:-1], x[1:]])
    targets = Z[1:]
    W = np.linalg.solve(inputs.T @ inputs + ridge * np.eye(inputs.shape[1]), inputs.T @ targets)
    return W[: Z.shape[1]].T


def spectral_radius(A):
    return float(np.abs(np.linalg.eigvals(A)).max())


def a_drift(Z, x):
    half = len(Z) // 2
    A_early = fit_A(Z[:half], x[:half])
    A_late = fit_A(Z[half:], x[half:])
    drift = float(np.linalg.norm(A_late - A_early) / (np.linalg.norm(A_early) + 1e-12))
    return drift, spectral_radius(A_early), spectral_radius(A_late)


def stability(Z, x, window):
    """P7: windowed ρ(A) of the metric state; when it settles below 1, and its last value."""
    n_components = min(8, Z.shape[1])
    centers, rho = windowed_rho(Z, x, window=window, stride=5, n_components=n_components)
    if len(rho) == 0:
        return None, np.nan
    settle = transition_step(centers, rho, threshold=1.0, persist=2)
    return settle, float(rho[-1])


def distance_to_target(raw_steering, target, scale):
    """Per step: the steering metrics' mean distance above the target level, in units of its spread."""
    return ((raw_steering - target) / scale).mean(axis=1)


def relative_recovery(raw_steering, target, scale, window):
    """P6: share of the starting distance to the target closed by the end (NaN if it starts at the target)."""
    distance = distance_to_target(raw_steering, target, scale)
    start = distance[0]
    if start <= 0:
        return np.nan
    end = distance[-window:].mean()
    return float((start - end) / start)


def recovery_step(raw_steering, steps, target, scale):
    """First step after which the steering metrics stay at or below the target level."""
    distance = distance_to_target(raw_steering, target, scale)
    run = 0
    for i, value in enumerate(distance):
        run = run + 1 if value <= 0 else 0
        if run >= PERSIST_STEPS:
            return int(steps[i - PERSIST_STEPS + 1])
    return None


def measure(campaign, window, rho_window):
    early, late = comovement(campaign["S"], campaign["H"])
    drift, _, _ = a_drift(campaign["Z"], campaign["x"])
    settle, last_rho = stability(campaign["Z"], campaign["x"], rho_window)
    return {
        "dS": change(campaign["S"], window),
        "dH": change(campaign["H"], window),
        "corr_early": early,
        "corr_late": late,
        "drift": drift,
        "score": float(campaign["score"][-1]),
        "warm_score": campaign["warm_score"],
        "stuck": bool(campaign["score"][-1] >= campaign["warm_score"]),
        "settle_step": settle,
        "last_rho": last_rho,
        "final_S": float(campaign["S"][-window:].mean()),
    }


# ── Statistics ────────────────────────────────────────────────────────────────

def paired_difference(values, reference):
    """Mean of policy − reference over shared seeds, Wilcoxon p, number of seeds."""
    seeds = sorted(seed for seed in values if seed in reference
                   and np.isfinite(values[seed]) and np.isfinite(reference[seed]))
    differences = np.array([values[seed] - reference[seed] for seed in seeds])
    if len(differences) >= 5 and np.any(differences != 0):
        p = float(stats.wilcoxon(differences).pvalue)
    else:
        p = np.nan
    mean = float(differences.mean()) if len(differences) else np.nan
    return mean, p, len(differences)


def per_seed(per_campaign, policy, key):
    return {seed: value[key] for (q, seed), value in per_campaign.items() if q == policy}


def recovery_targets(reference_campaigns, policy, window):
    """Each steering metric's end level (median) and spread (IQR) in the calibrated run, for one arm."""
    ends = [c["raw_steering"][-window:] for (q, _), c in reference_campaigns.items() if q == policy]
    if not ends:
        return None, None
    pooled = np.vstack(ends)
    target = np.median(pooled, axis=0)
    q75, q25 = np.percentile(pooled, [75, 25], axis=0)
    return target, np.maximum(q75 - q25, 1e-12)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--in", dest="path", type=Path, default=Path("_results/steering"))
    p.add_argument("--reference", type=Path, default=None,
                   help="calibrated run of the same problem, for P6 (recovery)")
    p.add_argument("--window", type=int, default=10, help="steps averaged at each end for Δ")
    p.add_argument("--rho-window", type=int, default=20, help="window of the P7 ρ(A) fits")
    p.add_argument("--null", default="random", help="reference policy for P1, P2 and P4")
    return p.parse_args(argv)


def fmt(value, digits=3):
    return "     —" if value is None or not np.isfinite(value) else f"{value:+.{digits}f}"


def fmt_p(value):
    return "     —" if value is None or not np.isfinite(value) else f"{value:6.3f}"


def main(argv=None):
    args = parse_args(argv)
    campaigns, config = load_run(args.path)
    policies = [q for q in config["policies"] if any(key[0] == q for key in campaigns)]

    per_campaign = {key: measure(c, args.window, args.rho_window) for key, c in campaigns.items()}

    reference_campaigns = None
    if args.reference is not None:
        reference_campaigns, _ = load_run(args.reference)
        for (policy, seed), campaign in campaigns.items():
            target, scale = recovery_targets(reference_campaigns, policy, args.window)
            step = None
            closed = np.nan
            if target is not None:
                step = recovery_step(campaign["raw_steering"], campaign["step"], target, scale)
                closed = relative_recovery(campaign["raw_steering"], target, scale, args.window)
            per_campaign[(policy, seed)]["recovery_step"] = step
            per_campaign[(policy, seed)]["recovery_closed"] = closed

    random_finals = [v["score"] for (q, _), v in per_campaign.items() if q == "random"]
    failure_level = float(np.median(random_finals)) if random_finals else np.nan

    print(f"Steering: {', '.join(config['steering'])}")
    print(f"Audit-only: {', '.join(config['audit_only'])}\n")

    summary = {}
    for policy in policies:
        mine = {seed: v for (q, seed), v in per_campaign.items() if q == policy}
        finals = np.array([v["score"] for v in mine.values()])
        warm = np.array([v["warm_score"] for v in mine.values()])
        entry = {"n_seeds": len(mine)}

        # P1, P2, P4 against the null; P2 also against state-space
        for name, key in (("dS", "dS"), ("dH", "dH"), ("A_drift", "drift")):
            mean, p, _ = paired_difference(per_seed(per_campaign, policy, key), per_seed(per_campaign, args.null, key))
            entry[f"{name}_minus_null"] = mean
            entry[f"{name}_p"] = p
        mean, p, _ = paired_difference(per_seed(per_campaign, policy, "dH"),
                                       per_seed(per_campaign, "state-space", "dH"))
        entry["dH_minus_state_space"] = mean
        entry["dH_vs_state_space_p"] = p

        # P3
        entry["comovement_early"] = float(np.nanmean([v["corr_early"] for v in mine.values()]))
        entry["comovement_late"] = float(np.nanmean([v["corr_late"] for v in mine.values()]))
        entry["A_drift"] = float(np.mean([v["drift"] for v in mine.values()]))

        # P5
        entry["final_score_mean"] = float(finals.mean())
        entry["final_score_sem"] = float(finals.std() / np.sqrt(len(finals)))
        entry["final_score_median"] = float(np.median(finals))
        entry["failure_rate"] = float(np.mean(finals > failure_level))
        entry["stuck_rate"] = float(np.mean([v["stuck"] for v in mine.values()]))
        if np.std(warm) > 0 and np.std(finals) > 0:
            entry["warm_vs_final_spearman"] = float(stats.spearmanr(warm, finals).statistic)
        else:
            entry["warm_vs_final_spearman"] = np.nan
        for reference in ("random", "lcb", "state-space"):
            mean, p, _ = paired_difference(per_seed(per_campaign, policy, "score"),
                                           per_seed(per_campaign, reference, "score"))
            entry[f"score_minus_{reference}"] = mean
            entry[f"score_vs_{reference}_p"] = p

        # P6
        if reference_campaigns is not None:
            steps = [v["recovery_step"] for v in mine.values()]
            entry["recovered_share"] = sum(s is not None for s in steps) / len(steps)
            entry["recovery_closed_median"] = float(np.nanmedian([v["recovery_closed"] for v in mine.values()]))
            mean, p, _ = paired_difference(per_seed(per_campaign, policy, "recovery_closed"),
                                           per_seed(per_campaign, "state-space", "recovery_closed"))
            entry["recovery_closed_minus_state_space"] = mean
            entry["recovery_closed_vs_state_space_p"] = p

        # P7
        settles = [v["settle_step"] for v in mine.values()]
        settled = [s for s in settles if s is not None]
        entry["settled_share"] = len(settled) / len(settles)
        entry["settle_step_median"] = float(np.median(settled)) if settled else np.nan
        entry["last_rho_median"] = float(np.nanmedian([v["last_rho"] for v in mine.values()]))

        summary[policy] = entry

    pooled = [(v["final_S"], v["last_rho"]) for v in per_campaign.values() if np.isfinite(v["last_rho"])]
    health_vs_rho = np.nan
    if len(pooled) >= 5:
        health_vs_rho = float(stats.spearmanr([a for a, _ in pooled], [b for _, b in pooled]).statistic)

    print_tables(summary, policies, args, health_vs_rho, reference_campaigns is not None)
    run = {"health_vs_rho_spearman": health_vs_rho, "failure_level": failure_level,
           "problem": config.get("problem"), "condition": config.get("condition"),
           "score": config.get("score"), "reference": str(args.reference) if args.reference else None}
    (args.path / "steering_summary.json").write_text(json.dumps(summary, indent=2))
    (args.path / "steering_run.json").write_text(json.dumps(run, indent=2))
    try:
        draw_figure(campaigns, policies, config, args.path / "steering_summary.png")
        print(f"\nWrote {args.path / 'steering_summary.json'} and steering_summary.png")
    except ImportError:
        print(f"\nWrote {args.path / 'steering_summary.json'} (matplotlib missing: no figure)")


def print_tables(summary, policies, args, health_vs_rho, have_recovery):
    print(f"P1–P4 (paired against '{args.null}'; P2 also against 'state-space')")
    header = (f"{'policy':24s} {'ΔS−null':>8s} {'p':>6s} {'ΔH−null':>8s} {'p':>6s} {'ΔH−ss':>8s} {'p':>6s}"
              f" {'ρ(dS,dH) early→late':>20s} {'A drift':>8s}")
    print(header + "\n" + "-" * len(header))
    for policy in policies:
        e = summary[policy]
        print(f"{policy:24s} {fmt(e['dS_minus_null'])} {fmt_p(e['dS_p'])} {fmt(e['dH_minus_null'])} {fmt_p(e['dH_p'])}"
              f" {fmt(e['dH_minus_state_space'])} {fmt_p(e['dH_vs_state_space_p'])}"
              f"   {e['comovement_early']:+.2f} → {e['comovement_late']:+.2f}   {e['A_drift']:8.3f}")
    print("Δ < 0 means the group got better (lower badness).")
    print("Goodharting shows as ΔS < 0 with ΔH ≥ 0 against state-space, plus weaker late co-movement.")

    print("\nP5 outcome (failure = final score worse than the random arm's median; stuck = no gain after warm start)")
    header = (f"{'policy':24s} {'median':>9s} {'mean':>9s} {'fail':>5s} {'stuck':>5s} {'ρ(warm,final)':>13s}"
              f" {'p vs lcb':>8s} {'p vs ss':>8s}")
    print(header + "\n" + "-" * len(header))
    for policy in policies:
        e = summary[policy]
        print(f"{policy:24s} {e['final_score_median']:9.4f} {e['final_score_mean']:9.4f} {e['failure_rate']:5.2f}"
              f" {e['stuck_rate']:5.2f} {fmt(e['warm_vs_final_spearman'], 2):>13s}"
              f" {fmt_p(e['score_vs_lcb_p']):>8s} {fmt_p(e['score_vs_state-space_p']):>8s}")

    print("\nP6 relative recovery (share of the gap to the calibrated end level closed) and P7 stability")
    header = (f"{'policy':24s} {'gap closed':>10s} {'−ss':>7s} {'p vs ss':>8s} {'reached':>7s}"
              f" {'settled':>8s} {'step':>6s} {'last ρ':>7s}")
    print(header + "\n" + "-" * len(header))
    for policy in policies:
        e = summary[policy]
        if have_recovery:
            recovery = (f"{fmt(e['recovery_closed_median'], 2):>10s} {fmt(e['recovery_closed_minus_state_space'], 2):>7s}"
                        f" {fmt_p(e['recovery_closed_vs_state_space_p']):>8s} {e['recovered_share']:7.2f}")
        else:
            recovery = f"{'—':>10s} {'—':>7s} {'—':>8s} {'—':>7s}"
        print(f"{policy:24s} {recovery} {e['settled_share']:8.2f} {fmt(e['settle_step_median'], 0):>6s}"
              f" {e['last_rho_median']:7.3f}")
    print(f"Health vs stability: Spearman(final S, last ρ) over all campaigns = {fmt(health_vs_rho, 2)}"
          " (> 0: healthier end states have smaller ρ)")


def draw_figure(campaigns, policies, config, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for policy in policies:
        mine = [c for (q, _), c in campaigns.items() if q == policy]
        n = min(len(c["step"]) for c in mine)
        step = mine[0]["step"][:n]
        for ax, key in zip(axes, ("score", "S", "H"), strict=True):
            ax.plot(step, np.median([c[key][:n] for c in mine], axis=0), label=policy,
                    lw=1.0 if policy == "random" else 1.4)
    if config["n_objectives"] == 1:
        axes[0].set_yscale("log")
    axes[0].set_title(f"{config['score']} (median)")
    axes[1].set_title("Steering metrics (median z, ↓ better)")
    axes[2].set_title("Audit-only metrics (median z, ↓ better)")
    for ax in axes:
        ax.set_xlabel("oracle queries")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
