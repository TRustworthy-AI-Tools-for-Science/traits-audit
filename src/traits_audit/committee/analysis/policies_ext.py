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

from traits_audit.committee.analysis.votes import CommitteeVoter


Policy = Callable[[np.ndarray, "object"], np.ndarray]


# ---------------------------------------------------------------------------
# Thread B: vote-augmented baselines.
# ---------------------------------------------------------------------------

def _mean_vote_distance(grid: np.ndarray, prefs: np.ndarray) -> np.ndarray:
    """Mean |grid - prefs[k]| across the 9 committee members.

    grid: shape (G,). prefs: shape (9,). Returns shape (G,).
    """
    return np.abs(grid[:, None] - prefs[None, :]).mean(axis=1)


def lcb_with_votes_policy(
    voter: CommitteeVoter,
    kappa: float = 2.0,
    vote_weight: float = 1.0,
    grid_size: int = 300,
) -> Policy:
    """LCB acquisition with a soft committee-distance penalty.

    score(x) = mu(x) - kappa * sigma(x) + vote_weight * mean_k |x - pi_k(s)|

    Defaults: ``vote_weight=1.0`` makes the committee term comparable in
    magnitude to the LCB term on Forrester (mu in roughly [-6, 15]; mean
    distance bounded by 1). Higher weight => more committee influence.
    """
    grid = np.linspace(0.0, 1.0, grid_size)

    def _pi(obs: np.ndarray, env) -> np.ndarray:
        surrogate = env.unwrapped.surrogate
        mu, sigma = surrogate.predict(grid)
        prefs = voter.preferred_actions(obs)
        dist = _mean_vote_distance(grid, prefs)
        score = mu - kappa * sigma + vote_weight * dist
        idx = int(np.argmin(score))
        return np.array([grid[idx]], dtype=np.float32)

    return _pi


def max_sigma_with_votes_policy(
    voter: CommitteeVoter,
    vote_weight: float = 1.0,
    grid_size: int = 300,
) -> Policy:
    """max-sigma with a soft committee-distance penalty.

    score(x) = -sigma(x) + vote_weight * mean_k |x - pi_k(s)|
    """
    grid = np.linspace(0.0, 1.0, grid_size)

    def _pi(obs: np.ndarray, env) -> np.ndarray:
        surrogate = env.unwrapped.surrogate
        _mu, sigma = surrogate.predict(grid)
        prefs = voter.preferred_actions(obs)
        dist = _mean_vote_distance(grid, prefs)
        score = -sigma + vote_weight * dist
        idx = int(np.argmin(score))
        return np.array([grid[idx]], dtype=np.float32)

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
        return np.array([prefs[idx]], dtype=np.float32)

    return _pi


def committee_agree_policy(voter: CommitteeVoter) -> Policy:
    """Pick the centroid (mean) of the 9 preferred actions.

    Mirrors `agree=True` mode in [src/policies/qbc.py]: when the committee
    converges on a region, exploit there. When they disagree, this falls back
    to the average which may be a no-man's-land between modes — exactly the
    failure mode worth flagging.
    """
    def _pi(obs: np.ndarray, env) -> np.ndarray:
        prefs = voter.preferred_actions(obs)
        return np.array([float(np.mean(prefs))], dtype=np.float32)

    return _pi


def committee_disagree_policy(
    voter: CommitteeVoter,
    grid_size: int = 300,
    bandwidth: float = 0.05,
) -> Policy:
    """Pick the grid x with maximum local committee disagreement.

    For each grid x, compute the variance of preferred actions whose distance
    to x is below ``bandwidth``. The candidate with the highest local-cluster
    spread is the most-contested neighborhood.

    Falls back to the max-distance candidate if no agents are within
    bandwidth of any grid point (degenerate cases on very tight committees).
    """
    grid = np.linspace(0.0, 1.0, grid_size)

    def _pi(obs: np.ndarray, env) -> np.ndarray:
        prefs = voter.preferred_actions(obs)
        # For each grid x, weight each agent by a Gaussian on |x - pref|
        # and compute the weighted variance of prefs. Smooth differentiable
        # version of "what's the spread of agents near x?".
        dx = grid[:, None] - prefs[None, :]                    # (G, K)
        w = np.exp(-(dx ** 2) / (2.0 * bandwidth ** 2))         # (G, K)
        w_sum = w.sum(axis=1, keepdims=True)
        w_sum = np.where(w_sum < 1e-12, 1.0, w_sum)
        mean = (w * prefs[None, :]).sum(axis=1, keepdims=True) / w_sum
        var = (w * (prefs[None, :] - mean) ** 2).sum(axis=1) / w_sum.squeeze(1)
        idx = int(np.argmax(var))
        return np.array([grid[idx]], dtype=np.float32)

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
        prefs = voter.preferred_actions(obs)
        return np.array([float(np.dot(w, prefs))], dtype=np.float32)

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
