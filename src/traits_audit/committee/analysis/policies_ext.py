"""Vote-augmented baselines (Thread B) and committee aggregators (Thread A).

All policies share the :class:`~.votes.CommitteeVoter` primitive so a single
loaded committee is reused across the regret bake-off.

Thread B (votes as features):
    lcb_with_votes_policy        LCB minimization with a soft action-distance
                                 penalty toward the 9-policy centroid.
    max_sigma_with_votes_policy  max-sigma with the same penalty.

Thread A (vote-based aggregation):
    committee_uniform_policy     v0 baseline (reproduced here for symmetry).
    committee_agree_policy       pick the candidate closest to the mean of
                                 the 9 preferred actions.
    committee_disagree_policy    pick the candidate that maximises action-std
                                 of the committee in its grid neighborhood
                                 (interpretation: the candidate the committee
                                 is most divided about — the QBC `disagree`
                                 mode in [src/policies/qbc.py]).
    committee_weighted_policy    weighted-mean preferred action; weights from
                                 a user-supplied dict (independence or
                                 inverse-regret weighting both go through
                                 this same policy).

All policies expose the same callable signature as the rest of the rollout
engine: ``(obs, env) -> np.ndarray shape (1,)``.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from traits_audit.committee.analysis.rollouts import (
    _acquisition_grid,
    _reduce_objectives,
)
from traits_audit.committee.analysis.votes import CommitteeVoter


Policy = Callable[[np.ndarray, "object"], np.ndarray]


# ---------------------------------------------------------------------------
# Thread B: vote-augmented baselines.
# ---------------------------------------------------------------------------

def _mean_vote_distance(grid: np.ndarray, prefs: np.ndarray) -> np.ndarray:
    """Mean ||grid - prefs[k]|| across the K committee members.

    grid: shape (G, dim). prefs: shape (K, dim). Returns shape (G,).
    In 1-D this is the original ``mean_k |x - pi_k|``.
    """
    return np.linalg.norm(grid[:, None, :] - prefs[None, :, :], axis=2).mean(axis=1)


def lcb_with_votes_policy(
    voter: CommitteeVoter,
    kappa: float = 2.0,
    vote_weight: float = 1.0,
    grid_size: int = 300,
) -> Policy:
    """LCB acquisition with a soft committee-distance penalty.

    score(x) = mu(x) - kappa * sigma(x) + vote_weight * mean_k ||x - pi_k(s)||

    Defaults: ``vote_weight=1.0`` makes the committee term comparable in
    magnitude to the LCB term on Forrester (mu in roughly [-6, 15]; mean
    distance bounded by 1). Higher weight => more committee influence.

    The candidate set comes from :func:`~.rollouts._acquisition_grid`, so
    this matches whatever the plain LCB baseline searches over on the same
    problem (a 300-point line in 1-D, the problem's own state-grid in 2-D/3-D).
    """
    def _pi(obs: np.ndarray, env) -> np.ndarray:
        surrogate = env.unwrapped.surrogate
        grid = _acquisition_grid(env, grid_size)
        mu, sigma = surrogate.predict(grid)
        prefs = voter.preferred_actions(obs)
        dist = _mean_vote_distance(grid, prefs)
        score = (_reduce_objectives(mu) - kappa * _reduce_objectives(sigma)
                 + vote_weight * dist)
        return grid[int(np.argmin(score))].astype(np.float32)

    return _pi


def max_sigma_with_votes_policy(
    voter: CommitteeVoter,
    vote_weight: float = 1.0,
    grid_size: int = 300,
) -> Policy:
    """max-sigma with a soft committee-distance penalty.

    score(x) = -sigma(x) + vote_weight * mean_k ||x - pi_k(s)||
    """
    def _pi(obs: np.ndarray, env) -> np.ndarray:
        surrogate = env.unwrapped.surrogate
        grid = _acquisition_grid(env, grid_size)
        _mu, sigma = surrogate.predict(grid)
        prefs = voter.preferred_actions(obs)
        dist = _mean_vote_distance(grid, prefs)
        score = -_reduce_objectives(sigma) + vote_weight * dist
        return grid[int(np.argmin(score))].astype(np.float32)

    return _pi


# ---------------------------------------------------------------------------
# Thread A: aggregation rules over the 9 committee members.
# ---------------------------------------------------------------------------

def committee_uniform_policy(
    voter: CommitteeVoter,
    rng: np.random.Generator,
) -> Policy:
    """v0 baseline: pick one of the 9 preferred actions uniformly each step."""
    n = voter.n_agents

    def _pi(obs: np.ndarray, env) -> np.ndarray:
        prefs = voter.preferred_actions(obs)
        idx = int(rng.integers(0, n))
        return prefs[idx].astype(np.float32)

    return _pi


def committee_agree_policy(voter: CommitteeVoter) -> Policy:
    """Pick the centroid (mean) of the K preferred actions.

    Mirrors `agree=True` mode in [src/policies/qbc.py]: when the committee
    converges on a region, exploit there. When they disagree, this falls back
    to the average which may be a no-man's-land between modes — exactly the
    failure mode worth flagging. In d dimensions the centroid can land
    further from every member than it can on a line, so this failure mode
    gets *worse* with dimension, not better.
    """
    def _pi(obs: np.ndarray, env) -> np.ndarray:
        prefs = voter.preferred_actions(obs)
        return prefs.mean(axis=0).astype(np.float32)

    return _pi


def committee_disagree_policy(
    voter: CommitteeVoter,
    grid_size: int = 300,
    bandwidth: float = 0.05,
) -> Policy:
    """Pick the grid point with maximum local committee disagreement.

    For each candidate x, weight each agent by a Gaussian on its distance to
    x and compute the weighted spread of the preferred actions around their
    local weighted centroid. The candidate with the highest local spread is
    the most-contested neighborhood.

    In d dimensions "spread" is the trace of the weighted covariance, i.e.
    the sum of the per-axis weighted variances. That is the direct
    generalization of the 1-D weighted variance and reduces to it exactly
    when dim == 1, so Forrester numbers are unchanged. It is isotropic:
    a committee split along any single axis scores the same as one split
    along another.

    ponytail: trace, not the top covariance eigenvalue -- the leading
    eigenvalue would flag anisotropic splits specifically, but it is noisy
    at K=15 and breaks 1-D bit-compatibility. Swap it in if the isotropic
    score turns out to hide a directional split worth acting on.
    """
    def _pi(obs: np.ndarray, env) -> np.ndarray:
        grid = _acquisition_grid(env, grid_size)               # (G, dim)
        prefs = voter.preferred_actions(obs)                   # (K, dim)
        d2 = ((grid[:, None, :] - prefs[None, :, :]) ** 2).sum(axis=2)   # (G, K)
        w = np.exp(-d2 / (2.0 * bandwidth ** 2))               # (G, K)
        w_sum = w.sum(axis=1, keepdims=True)
        w_sum = np.where(w_sum < 1e-12, 1.0, w_sum)
        # Weighted centroid per candidate, then trace of the weighted
        # covariance about it -- summed over axes, so (G,).
        mean = (w[:, :, None] * prefs[None, :, :]).sum(axis=1) / w_sum   # (G, dim)
        dev2 = ((prefs[None, :, :] - mean[:, None, :]) ** 2).sum(axis=2)  # (G, K)
        spread = (w * dev2).sum(axis=1) / w_sum.squeeze(1)
        return grid[int(np.argmax(spread))].astype(np.float32)

    return _pi


def committee_weighted_policy(
    voter: CommitteeVoter,
    weights: dict[str, float],
) -> Policy:
    """Weighted-mean preferred action.

    ``weights`` keys must match :attr:`voter.agent_names`; missing keys
    default to 0 (the agent does not vote). Weights are renormalised to
    sum to 1.
    """
    names = voter.agent_names
    w = np.array([float(weights.get(n, 0.0)) for n in names], dtype=np.float64)
    total = w.sum()
    if total <= 0:
        raise ValueError("All weights are zero; nothing to aggregate.")
    w = w / total

    def _pi(obs: np.ndarray, env) -> np.ndarray:
        prefs = voter.preferred_actions(obs)                   # (K, dim)
        return (w @ prefs).astype(np.float32)

    return _pi


# ---------------------------------------------------------------------------
# Convenience: weight schemes from v0 results.
# ---------------------------------------------------------------------------

def independence_weights(
    correlation_csv: "Path",
    agent_names: list[str],
) -> dict[str, float]:
    """w_k proportional to 1 - mean |rho_{k, j}| over j != k.

    Reads the v0 trained-rollouts correlation CSV produced by
    ``ta-committee-analyze corr-trained``. More independent agents
    (lower mean |rho|) get higher weight.
    """
    import csv
    from pathlib import Path

    p = Path(correlation_csv)
    with p.open() as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [r for r in reader]
    # Expect square matrix with first column the row label.
    col_names = header[1:]
    matrix = {}
    for row in rows:
        rname = row[0]
        for cname, v in zip(col_names, row[1:]):
            matrix[(rname, cname)] = float(v)
    # Mean |rho| off-diagonal per agent.
    indep = {}
    for k in agent_names:
        offdiag = [abs(matrix[(k, j)]) for j in agent_names if j != k]
        indep[k] = 1.0 - (sum(offdiag) / len(offdiag))
    return indep


def inverse_regret_weights(
    regret_json: "Path",
    agent_names: list[str],
    floor: float = 0.001,
) -> dict[str, float]:
    """w_k proportional to 1 / max(solo_SR_k, floor).

    Reads the v0 terminal-SR `solo_means` from ``regret_test.json``. Better
    solo agents (lower terminal SR) get higher weight; ``floor`` prevents
    divide-by-zero when a solo has saturated regret near 0.
    """
    import json
    from pathlib import Path

    p = Path(regret_json)
    data = json.loads(p.read_text())
    solo_means = data["solo_means"]
    out = {}
    for k in agent_names:
        sr = float(solo_means.get(f"solo:{k}", float("inf")))
        out[k] = 1.0 / max(sr, floor)
    return out
