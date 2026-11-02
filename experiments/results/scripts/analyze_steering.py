#!/usr/bin/env python
"""Test the steered / audit-only (Goodhart) predictions on a steering run.

Reads ``steps.csv`` + ``config.json`` written by ``steer_state_space.py`` and
evaluates the four predictions pre-registered in the experimental plan (§6.1),
each against the random policy on the same seed (paired; the metrics drift
under *any* policy as data accumulates, so the random campaign is the null):

P1  Steered metrics improve:      ΔS(policy) − ΔS(random) < 0
P2  Audit-only metrics do not:    ΔH(policy) − ΔH(random) ≥ 0
P3  Cross-group co-movement breaks: Spearman ρ between the first differences
    of S_t and H_t (plan §3.2), late half of the steering phase vs early half
P4  A drifts once the loop is closed: ‖A_late − A_early‖_F / ‖A_early‖_F for
    x_{t+1} = A x_t + B u_t fitted (ridge) on all metrics, policy vs random

S_t / H_t are the mean z-scored badness of the steering / audit-only metrics
(higher = worse); Δ is the mean over the last ``--window`` steering steps
minus the mean over the first ``--window``.  Paired Wilcoxon signed-rank
tests across seeds.

Usage::

    python experiments/results/scripts/analyze_steering.py --in _results/steering
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats


def load(path: Path):
    with open(path / "steps.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    cfg = json.loads((path / "config.json").read_text())
    runs: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for r in rows:
        if r["phase"] != "steer":
            continue
        d = runs.setdefault((r["policy"], int(r["seed"])), {"_rows": []})
        d["_rows"].append(r)
    metrics = cfg["steering"] + cfg["audit_only"]
    for d in runs.values():
        rs = sorted(d.pop("_rows"), key=lambda r: int(r["step"]))
        d["step"] = np.array([int(r["step"]) for r in rs])
        d["x"] = np.array([float(r["x"]) for r in rs])
        d["regret"] = np.array([float(r["regret"]) for r in rs])
        d["Z"] = np.array([[float(r[f"z:{m}"]) for m in metrics] for r in rs])
        d["S"] = d["Z"][:, : len(cfg["steering"])].mean(axis=1)
        d["H"] = d["Z"][:, len(cfg["steering"]):].mean(axis=1)
    return runs, cfg


def delta(series, w):
    return float(series[-w:].mean() - series[:w].mean())


def comovement(S, H):
    """Spearman ρ of first differences, early half and late half."""
    dS, dH = np.diff(S), np.diff(H)
    h = len(dS) // 2
    out = []
    for a, b in ((dS[:h], dH[:h]), (dS[h:], dH[h:])):
        rho = stats.spearmanr(a, b).statistic if np.std(a) > 0 and np.std(b) > 0 else np.nan
        out.append(float(rho))
    return out


def fit_A(Z, x, lam=1e-2):
    """Ridge least squares for x_{t+1} = A x_t + B u_t, u_t = query location."""
    X = np.hstack([Z[:-1], x[1:, None]])  # u_t is the query that produced Z[t+1]
    Y = Z[1:]
    W = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)
    return W[: Z.shape[1]].T


def a_drift(Z, x):
    h = len(Z) // 2
    A_e, A_l = fit_A(Z[:h], x[:h]), fit_A(Z[h:], x[h:])
    rho = lambda A: float(np.abs(np.linalg.eigvals(A)).max())  # noqa: E731
    return float(np.linalg.norm(A_l - A_e) / (np.linalg.norm(A_e) + 1e-12)), rho(A_e), rho(A_l)


def paired(vals: dict[int, float], null: dict[int, float]):
    seeds = sorted(set(vals) & set(null))
    diff = np.array([vals[s] - null[s] for s in seeds])
    p = stats.wilcoxon(diff).pvalue if len(diff) >= 5 and np.any(diff != 0) else np.nan
    return float(np.mean(diff)), float(p), len(diff)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--in", dest="path", type=Path, default=Path("_results/steering"))
    p.add_argument("--window", type=int, default=10, help="steps averaged at each end for Δ")
    p.add_argument("--null", default="random", help="reference policy for paired tests")
    args = p.parse_args(argv)

    runs, cfg = load(args.path)
    policies = [q for q in cfg["policies"] if any(k[0] == q for k in runs)]
    per = {q: {} for q in policies}
    for (q, seed), d in runs.items():
        e, l_ = comovement(d["S"], d["H"])
        drift, rho_e, rho_l = a_drift(d["Z"], d["x"])
        per[q][seed] = dict(dS=delta(d["S"], args.window), dH=delta(d["H"], args.window),
                            corr_early=e, corr_late=l_, drift=drift, rho_early=rho_e,
                            rho_late=rho_l, regret=float(d["regret"][-1]))

    pick = lambda q, k: {s: v[k] for s, v in per[q].items() if np.isfinite(v[k])}  # noqa: E731
    summary = {}
    hdr = (f"{'policy':18s} {'regret':>8s} {'ΔS−null':>9s} {'p':>6s} {'ΔH−null':>9s} {'p':>6s} "
           f"{'ρ(dS,dH) early→late':>20s} {'A drift':>8s} {'−null':>7s} {'p':>6s}")
    print(f"Steering: {', '.join(cfg['steering'])}\nAudit-only: {', '.join(cfg['audit_only'])}\n")
    print(hdr + "\n" + "-" * len(hdr))
    for q in policies:
        reg = np.array([v["regret"] for v in per[q].values()])
        dS = paired(pick(q, "dS"), pick(args.null, "dS"))
        dH = paired(pick(q, "dH"), pick(args.null, "dH"))
        dr = paired(pick(q, "drift"), pick(args.null, "drift"))
        ce = np.nanmean([v["corr_early"] for v in per[q].values()])
        cl = np.nanmean([v["corr_late"] for v in per[q].values()])
        drift = np.mean([v["drift"] for v in per[q].values()])
        summary[q] = dict(final_regret_mean=float(reg.mean()), final_regret_sem=float(reg.std() / np.sqrt(len(reg))),
                          dS_minus_null=dS[0], dS_p=dS[1], dH_minus_null=dH[0], dH_p=dH[1],
                          comovement_early=float(ce), comovement_late=float(cl),
                          A_drift=float(drift), A_drift_minus_null=dr[0], A_drift_p=dr[1], n_seeds=dS[2])
        fmt = lambda x: "   —" if not np.isfinite(x) else f"{x:6.3f}"  # noqa: E731
        print(f"{q:18s} {reg.mean():8.4f} {dS[0]:+9.3f} {fmt(dS[1])} {dH[0]:+9.3f} {fmt(dH[1])} "
              f"{ce:+9.2f} → {cl:+6.2f}   {drift:8.3f} {dr[0]:+7.3f} {fmt(dr[1])}")
    print(f"\nΔ < 0 means the group got better (lower badness) than under '{args.null}'.")
    print("Goodharting shows as ΔS−null < 0 with ΔH−null ≥ 0, plus weaker late co-movement.")

    (args.path / "steering_summary.json").write_text(json.dumps(summary, indent=2))
    try:
        _figure(runs, policies, cfg, args.path / "steering_summary.png")
        print(f"\nWrote {args.path / 'steering_summary.json'} and steering_summary.png")
    except ImportError:
        print(f"\nWrote {args.path / 'steering_summary.json'} (matplotlib missing: no figure)")


def _figure(runs, policies, cfg, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for q in policies:
        ds = [d for (pq, _), d in runs.items() if pq == q]
        n = min(len(d["step"]) for d in ds)
        step = ds[0]["step"][:n]
        for ax, key in zip(axes, ("regret", "S", "H"), strict=False):
            m = np.mean([d[key][:n] for d in ds], axis=0)
            ax.plot(step, m, label=q, lw=1.4 if q != cfg.get("null", "random") else 1.0)
    axes[0].set_yscale("log")
    axes[0].set_title("Simple regret")
    axes[1].set_title("Steering metrics (mean z, ↓ better)")
    axes[2].set_title("Audit-only metrics (mean z, ↓ better)")
    for ax in axes:
        ax.set_xlabel("oracle queries")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
