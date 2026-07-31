"""Gymnasium env wrapping BootstrapSurrogate + Forrester oracle for committee training.

Single env, single agent at a time. State is shared across all agents (the
surrogate's posterior + observed-history summary). Different agents use
different RewardComputers — the env exposes a ``reward_for(agent_name)`` API
so trainers can attach their agent's reward function at construction.

State vector (length 200 + 200 + 14 = 414):
    mu_grid[200]         surrogate mean on fixed grid x in [0, 1]
    sigma_grid[200]      surrogate std on same grid
    step_count_norm      current step / max_steps in [0, 1]
    mean_y_obs           mean of observed y
    var_y_obs            variance of observed y
    best_y               minimum observed y
    density_hist[10]     histogram of observed x's over 10 bins of [0, 1]

Action: Box(low=0, high=1, shape=(1,)) — a single x to query.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from traits_audit._example import BootstrapSurrogate, oracle as forrester_oracle
from traits_audit.committee.rewards import RewardComputer


GRID_SIZE = 200
DENSITY_BINS = 10
SUMMARY_FEATURES = 4  # step_count_norm, mean_y, var_y, best_y
STATE_DIM = 2 * GRID_SIZE + SUMMARY_FEATURES + DENSITY_BINS
DEFAULT_WARMSTART = 20
DEFAULT_EPISODE_LENGTH = 100


class CommitteeEnv(gym.Env):
    """Single-agent acquisition env for one committee member at a time.

    The reward_computer determines which audit objective this env instance
    optimizes. Swap reward_computer to train a different agent on the same
    state/action/dynamics.

    Parameters
    ----------
    reward_computer : RewardComputer
        Per-step reward; cumulative-mean delta of an audit check.
    oracle_fn : callable, optional
        Oracle f(x, rng) -> y. Defaults to noisy Forrester from _example.py.
    warmstart_n : int
        Number of initial random queries before step 0 (clean oracle, no noise).
    episode_length : int
        Maximum acquisition steps per episode.
    n_estimators : int
        BootstrapSurrogate ensemble size.
    std_scale : float
        BootstrapSurrogate predictive-sigma scaling.
    degree : int
        Polynomial feature degree.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        reward_computer: RewardComputer,
        oracle_fn: Optional[Callable[[np.ndarray, np.random.Generator], np.ndarray]] = None,
        warmstart_n: int = DEFAULT_WARMSTART,
        episode_length: int = DEFAULT_EPISODE_LENGTH,
        n_estimators: int = 30,
        std_scale: float = 0.7,
        degree: int = 5,
    ):
        super().__init__()
        self.reward_computer = reward_computer
        self.oracle_fn = oracle_fn if oracle_fn is not None else forrester_oracle
        self.warmstart_n = warmstart_n
        self.episode_length = episode_length
        self.n_estimators = n_estimators
        self.std_scale = std_scale
        self.degree = degree

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32
        )

        self._grid = np.linspace(0.0, 1.0, GRID_SIZE)
        self._rng: Optional[np.random.Generator] = None
        self._surrogate: Optional[BootstrapSurrogate] = None
        self._x_obs: Optional[np.ndarray] = None
        self._y_obs: Optional[np.ndarray] = None
        self._mu_history: Optional[list[float]] = None
        self._sigma_history: Optional[list[float]] = None
        self._step_count = 0

    # -- gym interface ----------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        surrogate_rng = np.random.default_rng(
            None if seed is None else seed + 2**31
        )
        self._surrogate = BootstrapSurrogate(
            degree=self.degree,
            n_estimators=self.n_estimators,
            std_scale=self.std_scale,
            rng=surrogate_rng,
        )

        x_init = self._rng.uniform(0.0, 1.0, size=self.warmstart_n)
        y_init = self.oracle_fn(x_init, self._rng)
        self._x_obs = x_init.astype(np.float64)
        self._y_obs = y_init.astype(np.float64)
        self._surrogate.fit(self._x_obs, self._y_obs)

        # Backfill per-observation (mu, sigma) at the warm-start x's so the
        # reward computers can see the surrogate's view of every observation.
        mu_init, sigma_init = self._surrogate.predict(self._x_obs)
        self._mu_history = list(mu_init.astype(float))
        self._sigma_history = list(sigma_init.astype(float))

        self._step_count = 0
        return self._observation(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._surrogate is None or self._x_obs is None or self._y_obs is None:
            raise RuntimeError("Call reset() before step().")

        x_q = float(np.clip(action[0], 0.0, 1.0))
        # Query the oracle and refit surrogate. (mu, sigma) at x_q from
        # the *pre-update* surrogate represent the agent's prediction at
        # query time, which is what the audit checks expect.
        mu_q_pred, sigma_q_pred = self._surrogate.predict(np.array([x_q]))
        mu_q = float(mu_q_pred[0])
        sigma_q = float(sigma_q_pred[0])
        y_q = float(self.oracle_fn(np.array([x_q]), self._rng)[0])

        y_before = np.asarray(self._y_obs, dtype=float)
        mu_before = np.asarray(self._mu_history, dtype=float)
        sigma_before = np.asarray(self._sigma_history, dtype=float)
        x_before = np.asarray(self._x_obs, dtype=float)

        self._x_obs = np.append(self._x_obs, x_q)
        self._y_obs = np.append(self._y_obs, y_q)
        self._mu_history.append(mu_q)
        self._sigma_history.append(sigma_q)
        self._surrogate.fit(self._x_obs, self._y_obs)

        y_after = np.asarray(self._y_obs, dtype=float)
        mu_after = np.asarray(self._mu_history, dtype=float)
        sigma_after = np.asarray(self._sigma_history, dtype=float)
        x_after = np.asarray(self._x_obs, dtype=float)

        reward = float(self.reward_computer.reward(
            y_before, mu_before, sigma_before,
            y_after, mu_after, sigma_after,
            x_before=x_before,
            x_after=x_after,
            sigma_series_before=sigma_before,
            sigma_series_after=sigma_after,
        ))

        self._step_count += 1
        terminated = False
        truncated = self._step_count >= self.episode_length
        info = {
            "x_q": x_q,
            "y_q": y_q,
            "mu_q": mu_q,
            "sigma_q": sigma_q,
            "step": self._step_count,
        }
        return self._observation(), reward, terminated, truncated, info

    # -- state assembly ---------------------------------------------------

    def _observation(self) -> np.ndarray:
        assert self._surrogate is not None and self._x_obs is not None
        mu_grid, sigma_grid = self._surrogate.predict(self._grid)
        step_norm = self._step_count / max(self.episode_length, 1)
        mean_y = float(np.mean(self._y_obs))
        var_y = float(np.var(self._y_obs))
        best_y = float(np.min(self._y_obs))
        density, _ = np.histogram(self._x_obs, bins=DENSITY_BINS, range=(0.0, 1.0))
        density = density.astype(np.float32) / max(len(self._x_obs), 1)
        state = np.concatenate([
            mu_grid.astype(np.float32),
            sigma_grid.astype(np.float32),
            np.array([step_norm, mean_y, var_y, best_y], dtype=np.float32),
            density,
        ])
        return state

    # -- convenience for downstream code ----------------------------------

    @property
    def x_obs(self) -> np.ndarray:
        return None if self._x_obs is None else np.asarray(self._x_obs, dtype=float)

    @property
    def y_obs(self) -> np.ndarray:
        return None if self._y_obs is None else np.asarray(self._y_obs, dtype=float)

    @property
    def surrogate(self) -> Optional[BootstrapSurrogate]:
        return self._surrogate
