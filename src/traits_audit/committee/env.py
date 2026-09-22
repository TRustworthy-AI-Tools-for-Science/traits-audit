"""Gymnasium env wrapping a bootstrap surrogate + benchmark oracle for committee training.

Single env, single agent at a time. State is shared across all agents (the
surrogate's posterior + observed-history summary). Different agents use
different RewardComputers — the env exposes a ``reward_for(agent_name)`` API
so trainers can attach their agent's reward function at construction.

The benchmark is a :class:`~traits_audit.committee.problems.Problem`
(default: Forrester). For input dimension d, a G-point state grid and B
density bins per dimension, the state vector has length 2G + 4 + B**d:

    mu_grid[G]           surrogate mean on the problem's fixed grid in [0, 1]^d
    sigma_grid[G]        surrogate std on same grid
    step_count_norm      current step / max_steps in [0, 1]
    mean_y_obs           mean of observed y
    var_y_obs            variance of observed y
    best_y               minimum observed y
    density_hist[B**d]   histogram of observed x's, B bins per dimension

y is the problem's audited objective divided by its ``y_scale``.

    forrester      d=1  G=200      B=10  -> 414
    branin-currin  d=2  G=15x15    B=5   -> 479
    color          d=3  G=8x8x8    B=4   -> 1092

Action: Box(low=0, high=1, shape=(d,)) — the next x to query.
"""
from __future__ import annotations

from typing import Callable, Optional, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from traits_audit.committee.problems import ForresterProblem, Problem, get_problem
from traits_audit.committee.rewards import RewardComputer


# Forrester layout, kept for callers that import these directly.
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
        Oracle f(x, rng) -> y for the Forrester problem. Defaults to noisy
        Forrester from _cal_demo.py. Not valid with other problems.
    warmstart_n : int
        Number of initial random queries before step 0, fit by a fresh
        surrogate each episode (same noisy oracle as the acquisition steps).
    episode_length : int
        Maximum acquisition steps per episode.
    n_estimators : int
        Bootstrap surrogate ensemble size.
    std_scale : float
        Bootstrap surrogate predictive-sigma scaling.
    degree : int, optional
        Polynomial feature degree. Defaults to the problem's (5 for Forrester).
    problem : Problem or str, optional
        Benchmark to query: a Problem instance or a name from
        ``problems.PROBLEMS``. Defaults to Forrester.
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
        degree: Optional[int] = None,
        problem: Union[Problem, str, None] = None,
    ):
        super().__init__()
        if isinstance(problem, str):
            problem = get_problem(problem)
        if problem is None:
            problem = ForresterProblem(oracle_fn)
        elif oracle_fn is not None:
            raise ValueError("oracle_fn only applies to the default Forrester problem.")
        self.problem = problem
        self.reward_computer = reward_computer
        self.warmstart_n = warmstart_n
        self.episode_length = episode_length
        self.n_estimators = n_estimators
        self.std_scale = std_scale
        self.degree = problem.degree if degree is None else degree

        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(problem.dim,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(problem.state_dim,), dtype=np.float32
        )

        self._rng: Optional[np.random.Generator] = None
        self._surrogate = None
        self._x: Optional[np.ndarray] = None      # (n, d) queried inputs
        self._y_raw: Optional[np.ndarray] = None  # (n, n_objectives), original units
        self._y_obs: Optional[np.ndarray] = None  # (n,) audited objective / y_scale
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
        self._surrogate = self.problem.make_surrogate(
            degree=self.degree,
            n_estimators=self.n_estimators,
            std_scale=self.std_scale,
            rng=surrogate_rng,
        )

        x_init = self._rng.uniform(0.0, 1.0, size=(self.warmstart_n, self.problem.dim))
        self._x = x_init.astype(np.float64)
        self._y_raw = np.asarray(self.problem.observe(self._x, self._rng), dtype=np.float64)
        self._y_obs = self._y_raw[:, 0] / self.problem.y_scale
        self._surrogate.fit(self._x, self._y_obs)

        # Backfill per-observation (mu, sigma) at the warm-start x's so the
        # reward computers can see the surrogate's view of every observation.
        mu_init, sigma_init = self._surrogate.predict(self._x)
        self._mu_history = list(mu_init.astype(float))
        self._sigma_history = list(sigma_init.astype(float))

        self._step_count = 0
        return self._observation(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._surrogate is None or self._x is None or self._y_obs is None:
            raise RuntimeError("Call reset() before step().")

        x_q = np.clip(np.asarray(action, dtype=np.float64).reshape(self.problem.dim), 0.0, 1.0)
        # Query the oracle and refit surrogate. (mu, sigma) at x_q from
        # the *pre-update* surrogate represent the agent's prediction at
        # query time, which is what the audit checks expect.
        mu_q_pred, sigma_q_pred = self._surrogate.predict(x_q[None, :])
        mu_q = float(mu_q_pred[0])
        sigma_q = float(sigma_q_pred[0])
        y_raw_q = np.asarray(self.problem.observe(x_q[None, :], self._rng), dtype=np.float64)[0]
        y_q = float(y_raw_q[0] / self.problem.y_scale)

        y_before = np.asarray(self._y_obs, dtype=float)
        mu_before = np.asarray(self._mu_history, dtype=float)
        sigma_before = np.asarray(self._sigma_history, dtype=float)
        x_before = self.x_obs

        self._x = np.vstack([self._x, x_q[None, :]])
        self._y_raw = np.vstack([self._y_raw, y_raw_q[None, :]])
        self._y_obs = np.append(self._y_obs, y_q)
        self._mu_history.append(mu_q)
        self._sigma_history.append(sigma_q)
        self._surrogate.fit(self._x, self._y_obs)

        y_after = np.asarray(self._y_obs, dtype=float)
        mu_after = np.asarray(self._mu_history, dtype=float)
        sigma_after = np.asarray(self._sigma_history, dtype=float)
        x_after = self.x_obs

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
            "x_q": float(x_q[0]) if self.problem.dim == 1 else x_q.copy(),
            "y_q": y_q,
            "y_raw_q": y_raw_q,
            "mu_q": mu_q,
            "sigma_q": sigma_q,
            "step": self._step_count,
        }
        return self._observation(), reward, terminated, truncated, info

    # -- state assembly ---------------------------------------------------

    def _observation(self) -> np.ndarray:
        assert self._surrogate is not None and self._x is not None
        p = self.problem
        mu_grid, sigma_grid = self._surrogate.predict(p.grid)
        step_norm = self._step_count / max(self.episode_length, 1)
        mean_y = float(np.mean(self._y_obs))
        var_y = float(np.var(self._y_obs))
        best_y = float(np.min(self._y_obs))
        if p.dim == 1:
            density, _ = np.histogram(self._x[:, 0], bins=p.density_bins, range=(0.0, 1.0))
        else:
            density, _ = np.histogramdd(
                self._x, bins=p.density_bins, range=[(0.0, 1.0)] * p.dim
            )
            density = density.ravel()
        density = density.astype(np.float32) / max(len(self._x), 1)
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
        """Queried inputs: shape (n,) for 1-D problems, (n, d) otherwise."""
        if self._x is None:
            return None
        x = np.asarray(self._x, dtype=float)
        return x[:, 0] if self.problem.dim == 1 else x

    @property
    def y_obs(self) -> np.ndarray:
        """Audited objective (column 0), divided by the problem's y_scale."""
        return None if self._y_obs is None else np.asarray(self._y_obs, dtype=float)

    @property
    def y_raw_obs(self) -> np.ndarray:
        """All objectives in original units, shape (n, n_objectives)."""
        return None if self._y_raw is None else np.asarray(self._y_raw, dtype=float)

    @property
    def surrogate(self):
        return self._surrogate
