#!/usr/bin/env python
"""Steer active learning on the uncertainty state vector (Forrester, 1-D).

Implements the paper's "Novel Active Learning Policies" idea with the
state-space policies in :mod:`results.policies`:

* Every step, a suite of traits-audit checks is evaluated on a **fixed
  reference grid** (experimental plan §3.3), never on the acquired points.
* Each metric is mapped to a "badness" score (higher = worse) and z-scored
  against its own warm-start baseline (§3.1), giving the uncertainty state
  vector ``x_t``.
* The metrics are partitioned (§6.1, Goodhart's law) into a **steering**
  subset, which the policy sees, and an **audit-only** subset, which it never
  sees and which is recorded only to check whether steering fools the
  instruments it optimises.
* The state-space policies learn ``x_{t+1} = A x_t + B u_t`` online
  (SVD + least squares; Bayesian/Sherman-Morrison for the Bayesian variant),
  with ``u_t = [x, mu(x), sigma(x)]`` the candidate query, and score
  candidates by predicted progress, exploration bonus and (for the ``audit-*``
  policies) the predicted health ``-mean(x_{t+1})`` of the steering metrics.
* A random probe fraction of the budget is reserved for every policy (§3.4).

Writes one row per (policy, seed, step) to ``<out>/steps.csv`` plus
``<out>/config.json``; analyse with ``analyze_steering.py``.

Usage::

    python experiments/results/scripts/steer_state_space.py            # defaults
    python experiments/results/scripts/steer_state_space.py --seeds 3 --steps 60
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from pathlib import Path

import numpy as np

from results.policies import (
    AuditBayesianDynamicsPolicy,
    AuditStateSpacePolicy,
    BayesianDynamicsPolicy,
    LCBPolicy,
    StateSpacePolicy,
)
from traits_audit import checks as C

# ── Oracle: Forrester et al. (2008) with heteroscedastic noise ────────────────


def forrester(x):
    x = np.asarray(x, dtype=float)
    return (6.0 * x - 2.0) ** 2 * np.sin(12.0 * x - 4.0)


def noise_sd(x):
    return 0.1 + 0.4 * np.asarray(x, dtype=float) ** 2


_DENSE = np.linspace(0.0, 1.0, 200_001)
F_STAR = float(forrester(_DENSE).min())


# ── Surrogate: bootstrap ensemble of polynomials (paper, Methods) ─────────────


class BootstrapPoly:
    """Bootstrap ensemble of degree-``degree`` polynomial fits.

    Predictive std combines the ensemble spread (epistemic) with a fixed
    aleatoric floor; the floor is never learned from the ensemble.
    """

    def __init__(self, degree: int = 5, n_members: int = 20, noise_floor: float = 0.25):
        self.degree, self.n_members, self.noise_floor = degree, n_members, noise_floor
        self._coefs: np.ndarray | None = None

    def fit(self, x, y, rng):
        x, y = np.asarray(x, float), np.asarray(y, float)
        coefs = []
        for _ in range(self.n_members):
            idx = rng.integers(0, len(x), len(x))
            deg = min(self.degree, len(np.unique(x[idx])) - 1)
            c = np.zeros(self.degree + 1)
            c[self.degree - deg:] = np.polyfit(x[idx], y[idx], deg)
            coefs.append(c)
        self._coefs = np.array(coefs)
        return self

    def predict(self, x):
        preds = np.array([np.polyval(c, x) for c in self._coefs])
        return preds.mean(axis=0), np.sqrt(preds.var(axis=0) + self.noise_floor ** 2)


# ── Uncertainty state: metrics -> badness (higher = worse) ────────────────────

METRICS = {
    "CalibrationError":         (C.CalibrationErrorCheck,         lambda r: r.value),
    "NLL":                      (C.NegativeLogLikelihoodCheck,    lambda r: r.value),
    "VarianceErrorCorrelation": (C.VarianceErrorCorrelationCheck, lambda r: -r.value),
    "SignedBias":               (C.SignedBiasCheck,               lambda r: abs(r.value)),
    "IntervalCoverage":         (C.IntervalCoverageCheck,         lambda r: abs(r.value - 0.683)),
    "IntervalScore":            (C.IntervalScoreCheck,            lambda r: r.value),
    "CRPS":                     (C.CRPSCheck,                     lambda r: r.value),
    "ENCE":                     (C.ENCECheck,                     lambda r: r.value),
    "PITUniformity":            (C.PITUniformityCheck,            lambda r: r.details["ks_statistic"]),
    "VarianceAlignment":        (C.VarianceAlignmentCheck,        lambda r: abs(np.log(r.value))),
}
DEFAULT_STEERING = ["CalibrationError", "NLL", "VarianceErrorCorrelation", "SignedBias"]


def metric_snapshot(y_ref, mu, sd) -> tuple[dict[str, float], dict[str, str]]:
    """Badness of every metric on the reference grid, plus each check's category."""
    hist = [{"y_true": a, "y_pred_mean": b, "y_pred_std": c, "uncertainty": c}
            for a, b, c in zip(y_ref.tolist(), mu.tolist(), sd.tolist())]
    out, cats = {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, (cls, badness) in METRICS.items():
            r = cls().run(hist)
            v = badness(r) if r.value is not None else np.nan
            out[name] = float(v) if np.isfinite(v) else 0.0
            cats[name] = r.category.value
    return out, cats


class Baseline:
    """Per-metric z-score against the warm-start snapshots (plan §3.1)."""

    def __init__(self, snapshots: list[dict[str, float]]):
        arr = {k: np.array([s[k] for s in snapshots]) for k in METRICS}
        self.mu = {k: float(v.mean()) for k, v in arr.items()}
        self.sd = {k: max(float(v.std()), 1e-3 * abs(float(v.mean())) + 1e-9) for k, v in arr.items()}

    def z(self, snap: dict[str, float]) -> dict[str, float]:
        return {k: (snap[k] - self.mu[k]) / self.sd[k] for k in METRICS}


# ── Policy adapter ────────────────────────────────────────────────────────────
# results.policies score candidates through ``model.predict_with_uncertainty``
# on ``[state | action]`` rows.  Here the policy state is
# s = [best observed y, mean predictive sigma on the reference grid] (both
# standardised) and the action is u = [x, mu(x), sigma(x)]; the "model"
# predicts the next best-so-far as min(best, mu(x)) with uncertainty sigma(x),
# which makes the LCB fallback an ordinary LCB on the surrogate.

N_STATE, N_ACTION = 2, 3


class SurrogateView:
    def predict_with_uncertainty(self, X):
        X = np.atleast_2d(X)
        s, u = X[:, :N_STATE], X[:, N_STATE:]
        mean = s.copy()
        mean[:, 0] = np.minimum(s[:, 0], u[:, 1])
        return {"mean": mean, "std": np.repeat(u[:, 2:3], N_STATE, axis=1)}


def progress(s) -> float:
    return float(s[0])


def make_policy(name: str, args, seed: int):
    common = dict(model=SurrogateView(), num_action_features=N_ACTION, degradation_fn=progress)
    dyn = dict(common, n_components=args.rank, kappa=args.kappa, Q_weight=1.0, R_weight=0.0,
               min_fit_obs=args.min_fit_obs, uncertainty_weight=1.0)
    health = lambda v: -float(np.mean(v))  # noqa: E731  (steering metrics are badness z-scores)
    if name == "random":
        return None
    if name == "lcb":
        return LCBPolicy(kappa=args.kappa, **common)
    if name == "state-space":
        return StateSpacePolicy(**dyn)
    if name == "bayes-dynamics":
        return BayesianDynamicsPolicy(gamma_info=args.gamma_info, gamma_sub=0.0, **dyn)
    if name == "audit-state-space":
        return AuditStateSpacePolicy(gamma_health=args.gamma_health, health_fn=health, **dyn)
    if name == "audit-bayes":
        return AuditBayesianDynamicsPolicy(gamma_info=args.gamma_info, gamma_sub=0.0,
                                           gamma_health=args.gamma_health, health_fn=health, **dyn)
    raise ValueError(name)


# ── One campaign ──────────────────────────────────────────────────────────────


def run_campaign(policy_name: str, seed: int, args, steering: list[str]) -> list[dict]:
    rng_seed = np.random.default_rng([seed, 0])
    grid = np.linspace(0.0, 1.0, args.grid)
    x_ref = np.linspace(0.0, 1.0, args.n_ref)
    y_ref = forrester(x_ref) + rng_seed.normal(0.0, noise_sd(x_ref))  # fixed reference observations
    x_obs = list(rng_seed.uniform(0.0, 1.0, args.n_init))

    def observe_y(x, t):
        return float(forrester(x) + np.random.default_rng([seed, 1, t]).normal(0.0, noise_sd(x)))

    y_obs = [observe_y(x, t) for t, x in enumerate(x_obs)]
    model = BootstrapPoly(args.degree, args.members, args.noise_floor)
    policy = make_policy(policy_name, args, seed)
    stateful = isinstance(policy, (StateSpacePolicy, BayesianDynamicsPolicy))

    # Standardisation of y is frozen after the warm start so the state is stationary.
    y_mu, y_sd = float(np.mean(y_obs)), float(np.std(y_obs)) or 1.0

    def snapshot(t):
        """Refit on the first ``t`` observations; metrics + policy state."""
        model.fit(x_obs[:t], y_obs[:t], np.random.default_rng([seed, 2, t]))
        mu_ref, sd_ref = model.predict(x_ref)
        snap, cats = metric_snapshot(y_ref, mu_ref, sd_ref)
        s = np.array([(min(y_obs[:t]) - y_mu) / y_sd, float(sd_ref.mean()) / y_sd])
        return snap, cats, s

    rows, warm = [], []
    for t in range(args.degree + 2, args.n_init + 1):  # warm start: random queries
        snap, cats, s = snapshot(t)
        warm.append((t, snap, s))
    base = Baseline([w[1] for w in warm])

    def vec(snap, names):
        z = base.z(snap)
        return np.array([z[k] for k in names])

    # Replay warm-start transitions into the policy (random actions = good excitation).
    if stateful:
        for (t, snap, s), x_next in zip(warm[:-1], x_obs[warm[0][0]:]):
            model.fit(x_obs[:t], y_obs[:t], np.random.default_rng([seed, 2, t]))
            mu_x, sd_x = model.predict(np.array([x_next]))
            u = np.array([x_next, (mu_x[0] - y_mu) / y_sd, sd_x[0] / y_sd])
            policy.observe(s, u, uncertainty_vector=vec(snap, steering))

    def record(t, phase, probe, snap):
        """Row for the state after ``t`` observations (simple regret on clean f)."""
        z = base.z(snap)
        row = {"policy": policy_name, "seed": seed, "step": t, "phase": phase, "probe": int(probe),
               "x": x_obs[t - 1], "y": y_obs[t - 1],
               "regret": float(forrester(np.array(x_obs[:t])).min() - F_STAR)}
        for k in METRICS:
            row[f"raw:{k}"], row[f"z:{k}"] = snap[k], z[k]
        return row

    for t, snap, s in warm:
        rows.append(record(t, "warm", False, snap))

    snap, cats, s = snapshot(len(x_obs))
    for t in range(len(x_obs), args.steps):
        mu_g, sd_g = model.predict(grid)
        cands = [np.array([g, (m - y_mu) / y_sd, d / y_sd]) for g, m, d in zip(grid, mu_g, sd_g)]
        v = vec(snap, steering)
        probe = np.random.default_rng([seed, 3, t]).random() < args.probe
        if policy is None or probe:
            u = cands[int(np.random.default_rng([seed, 4, t]).integers(len(cands)))]
        elif stateful:
            u, _, _ = policy.select_next(s, cands, uncertainty_vector=v)
        else:
            u, _, _ = policy.select_next(s, cands)
        x_new = float(u[0])
        x_obs.append(x_new)
        y_obs.append(observe_y(x_new, t))
        if stateful:
            policy.observe(s, u, uncertainty_vector=v)
        snap, cats, s = snapshot(len(x_obs))
        rows.append(record(t + 1, "steer", probe, snap))
    return rows, cats


POLICIES = ["random", "lcb", "state-space", "bayes-dynamics", "audit-state-space", "audit-bayes"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--policies", nargs="+", default=POLICIES, choices=POLICIES)
    p.add_argument("--steering", nargs="+", default=DEFAULT_STEERING, choices=list(METRICS),
                   help="metrics the policy may see; the rest are audit-only")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--steps", type=int, default=100, help="total oracle queries per campaign")
    p.add_argument("--n-init", type=int, default=12, help="random warm-start queries (>= degree + 3)")
    p.add_argument("--probe", type=float, default=0.1, help="random probe fraction (plan §3.4)")
    p.add_argument("--grid", type=int, default=200, help="candidate grid size")
    p.add_argument("--n-ref", type=int, default=100, help="fixed reference grid size (plan §3.3)")
    p.add_argument("--degree", type=int, default=5)
    p.add_argument("--members", type=int, default=20)
    p.add_argument("--noise-floor", type=float, default=0.25)
    p.add_argument("--kappa", type=float, default=1.0)
    p.add_argument("--rank", type=int, default=4, help="SVD rank of the learned dynamics")
    p.add_argument("--min-fit-obs", type=int, default=5)
    p.add_argument("--gamma-health", type=float, default=0.5)
    p.add_argument("--gamma-info", type=float, default=0.5)
    p.add_argument("--out", type=Path, default=Path("_results/steering"))
    args = p.parse_args(argv)

    audit_only = [m for m in METRICS if m not in args.steering]
    args.out.mkdir(parents=True, exist_ok=True)
    rows, cats, t0 = [], {}, time.time()
    for name in args.policies:
        for seed in range(args.seeds):
            r, cats = run_campaign(name, seed, args, args.steering)
            rows += r
        print(f"  {name:18s} done ({time.time() - t0:.0f}s)")

    with open(args.out / "steps.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config.update(audit_only=audit_only, categories=cats, f_star=F_STAR)
    (args.out / "config.json").write_text(json.dumps(config, indent=2))

    print("\nGroup coverage (check categories):")
    for label, names in (("steering", args.steering), ("audit-only", audit_only)):
        print(f"  {label:10s} " + ", ".join(f"{n} [{cats[n]}]" for n in names))
    print(f"\nFinal simple regret (mean over {args.seeds} seeds):")
    for name in args.policies:
        final = [r["regret"] for r in rows if r["policy"] == name and r["step"] == args.steps]
        print(f"  {name:18s} {np.mean(final):.4f} ± {np.std(final) / np.sqrt(len(final)):.4f}")
    print(f"\nWrote {args.out / 'steps.csv'}  — next: analyze_steering.py --in {args.out}")


if __name__ == "__main__":
    main()
