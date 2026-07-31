"""Rollout engine shared by all analyses.

A single rollout walks the CommitteeEnv for ``episode_length`` steps and
records, per step, the data needed to score every reward computer offline:

    x_q, y_q, mu_q, sigma_q  (per-step query, observed y, predicted mu/sigma)

Plus the warm-start (x, y) so the cumulative-mean checks see the full
history. From this we reconstruct the per-step y/mu/sigma arrays after the
fact and call every RewardComputer on the same trajectory — the rewards on
trajectory T scored by agent k's reward function are exactly what we'd need
to compute the cross-agent correlation matrix.

Why one env instance per rollout: BootstrapSurrogate is stateful (it refits
on every step). Sharing across rollouts would leak; we re-instantiate via
``env.reset(seed=...)``.

The policy is a callable ``obs -> action (np.ndarray shape (1,))``:
  * ``random_policy(rng)``  — uniform on [0, 1]
  * ``sac_policy(model)``   — model.predict(obs, deterministic=True)
  * ``lcb_policy()``        — argmin(mu - kappa*sigma) over a fixed grid
  * ``max_sigma_policy()``  — argmax(sigma) over a fixed grid
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from traits_audit.committee.env import (
    CommitteeEnv,
    DEFAULT_EPISODE_LENGTH,
    DEFAULT_WARMSTART,
)
from traits_audit.committee.rewards import REWARD_REGISTRY


Policy = Callable[[np.ndarray, "CommitteeEnv"], np.ndarray]


@dataclass
class RolloutTrace:
    """One episode of (warm-start + acquisition) data.

    Attributes
    ----------
    x_obs : np.ndarray, shape (warmstart_n + episode_length,)
        Full history of queried x's.
    y_obs : np.ndarray, shape (warmstart_n + episode_length,)
        Full history of observed y's.
    mu_hist : np.ndarray, shape (warmstart_n + episode_length,)
        Surrogate mean at each queried point, *as seen at query time*.
    sigma_hist : np.ndarray, shape (warmstart_n + episode_length,)
        Surrogate std at each queried point, as seen at query time.
    x_queries : np.ndarray, shape (episode_length,)
        The acquisition queries (warm-start excluded). For density plots.
    warmstart_n : int
        How many initial points are warm-start vs acquisition.
    """

    x_obs: np.ndarray
    y_obs: np.ndarray
    mu_hist: np.ndarray
    sigma_hist: np.ndarray
    x_queries: np.ndarray
    warmstart_n: int

    @property
    def n_steps(self) -> int:
        return len(self.x_queries)


# -- Policies ---------------------------------------------------------------

def random_policy(rng: np.random.Generator) -> Policy:
    def _pi(obs: np.ndarray, env: CommitteeEnv) -> np.ndarray:
        return np.array([rng.uniform(0.0, 1.0)], dtype=np.float32)
    return _pi


def sac_policy(model) -> Policy:
    def _pi(obs: np.ndarray, env: CommitteeEnv) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(1)
    return _pi


def lcb_policy(kappa: float = 2.0, grid_size: int = 300) -> Policy:
    grid = np.linspace(0.0, 1.0, grid_size)
    def _pi(obs: np.ndarray, env: CommitteeEnv) -> np.ndarray:
        surrogate = env.unwrapped.surrogate
        mu, sigma = surrogate.predict(grid)
        idx = int(np.argmin(mu - kappa * sigma))
        return np.array([grid[idx]], dtype=np.float32)
    return _pi


def max_sigma_policy(grid_size: int = 300) -> Policy:
    grid = np.linspace(0.0, 1.0, grid_size)
    def _pi(obs: np.ndarray, env: CommitteeEnv) -> np.ndarray:
        surrogate = env.unwrapped.surrogate
        _mu, sigma = surrogate.predict(grid)
        idx = int(np.argmax(sigma))
        return np.array([grid[idx]], dtype=np.float32)
    return _pi


# -- Rollout driver --------------------------------------------------------

def run_rollout(
    policy: Policy,
    seed: int,
    episode_length: int = DEFAULT_EPISODE_LENGTH,
    warmstart_n: int = DEFAULT_WARMSTART,
) -> RolloutTrace:
    """Walk the env one episode under ``policy`` and return the trace.

    The reward computer attached to the env is irrelevant — we discard the
    scalar reward and score the trajectory offline with every reward
    computer in :func:`score_trace`.
    """
    # Reward computer here is a dummy; we don't use the scalar reward path.
    dummy_reward = REWARD_REGISTRY["CRPS"]()
    env = CommitteeEnv(
        reward_computer=dummy_reward,
        episode_length=episode_length,
        warmstart_n=warmstart_n,
    )
    obs, _ = env.reset(seed=seed)
    x_q_list: list[float] = []
    for _ in range(episode_length):
        action = policy(obs, env)
        obs, _reward, terminated, truncated, info = env.step(action)
        x_q_list.append(info["x_q"])
        if terminated or truncated:
            break

    x_obs = np.asarray(env.x_obs, dtype=float)
    y_obs = np.asarray(env.y_obs, dtype=float)
    mu_hist = np.asarray(env._mu_history, dtype=float)
    sigma_hist = np.asarray(env._sigma_history, dtype=float)
    return RolloutTrace(
        x_obs=x_obs,
        y_obs=y_obs,
        mu_hist=mu_hist,
        sigma_hist=sigma_hist,
        x_queries=np.asarray(x_q_list, dtype=float),
        warmstart_n=warmstart_n,
    )


# -- Offline scoring -------------------------------------------------------

def score_trace(trace: RolloutTrace) -> dict[str, np.ndarray]:
    """Score every reward computer against the trace.

    For each agent k in REWARD_REGISTRY, returns a length-``n_steps`` array
    of per-step raw rewards on this trajectory. This is what the cross-agent
    correlation matrix is computed over.

    The cumulative-mean delta at step t uses history[:warmstart_n + t] vs
    history[:warmstart_n + t + 1] — matching what the env returned at
    training time.
    """
    out: dict[str, np.ndarray] = {}
    w = trace.warmstart_n
    y = trace.y_obs
    mu = trace.mu_hist
    sigma = trace.sigma_hist
    n_steps = trace.n_steps

    for name, reward_cls in REWARD_REGISTRY.items():
        rc = reward_cls()
        rewards = np.zeros(n_steps, dtype=float)
        for t in range(n_steps):
            i = w + t
            rewards[t] = rc.reward(
                y[:i], mu[:i], sigma[:i],
                y[:i + 1], mu[:i + 1], sigma[:i + 1],
            )
        out[name] = rewards
    return out
