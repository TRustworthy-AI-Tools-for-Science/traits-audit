"""SAC trainer wrapper + per-agent reward normalization.

One CommitteeAgent instance per audit-check reward. Each owns an env, an SAC
policy, a replay buffer, and a RunningZScore. Solo training: each agent runs
independent episodes on its own env instance — no shared trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import Wrapper

from traits_audit.committee.env import CommitteeEnv
from traits_audit.committee.rewards import REWARD_REGISTRY, RewardComputer, RunningZScore


class ZScoreRewardWrapper(Wrapper):
    """Apply per-agent running z-score normalization to rewards.

    Wraps a CommitteeEnv (or anything gym-like). The underlying env still
    produces raw cumulative-mean deltas; this wrapper updates the running
    statistics and returns the normalized scalar to whatever sees env.step.

    The raw reward is preserved in info["raw_reward"] for logging.
    """

    def __init__(self, env: gym.Env, normalizer: Optional[RunningZScore] = None):
        super().__init__(env)
        self.normalizer = normalizer if normalizer is not None else RunningZScore()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        raw = float(reward)
        normalized = float(self.normalizer.normalize(raw))
        info = {**info, "raw_reward": raw}
        return obs, normalized, terminated, truncated, info


@dataclass
class CommitteeAgent:
    """A single committee member: reward + env + SAC policy + normalizer.

    Attributes
    ----------
    name : str
        Agent identifier (matches REWARD_REGISTRY key).
    reward_computer : RewardComputer
    env : gym.Env
        ZScore-wrapped CommitteeEnv.
    normalizer : RunningZScore
        Per-agent reward statistics. Same object as env.normalizer.
    model : Any
        stable-baselines3 SAC instance (lazy-initialized via build_model).
    """

    name: str
    reward_computer: RewardComputer
    env: gym.Env
    normalizer: RunningZScore
    model: Any = None
    seed: int = 0
    tensorboard_log: Optional[str] = None

    def build_model(
        self,
        policy_kwargs: Optional[dict] = None,
        learning_rate: float = 3e-4,
        buffer_size: int = 100_000,
        batch_size: int = 256,
        learning_starts: int = 200,
        gamma: float = 0.99,
        verbose: int = 0,
    ) -> None:
        """Lazy-import SB3 here so the rest of the repo works without it."""
        from stable_baselines3 import SAC

        self.model = SAC(
            "MlpPolicy",
            self.env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            learning_starts=learning_starts,
            gamma=gamma,
            policy_kwargs=policy_kwargs or dict(net_arch=[128, 128]),
            verbose=verbose,
            seed=self.seed,
            tensorboard_log=self.tensorboard_log,
        )

    def learn(self, total_timesteps: int, **kwargs) -> None:
        if self.model is None:
            self.build_model()
        self.model.learn(total_timesteps=total_timesteps, **kwargs)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(f"Agent {self.name} has no model — call build_model().")
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    def save(self, dirpath: Path | str) -> None:
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        if self.model is None:
            raise RuntimeError(f"Agent {self.name} has no model to save.")
        self.model.save(str(dirpath / f"{self.name}.zip"))
        # Persist normalizer stats for reproducible deployment.
        np.savez(
            dirpath / f"{self.name}.normalizer.npz",
            n=self.normalizer.n,
            mean=self.normalizer.mean,
            M2=self.normalizer.M2,
        )

    @classmethod
    def load(cls, dirpath: Path | str, name: str, env: gym.Env) -> "CommitteeAgent":
        from stable_baselines3 import SAC

        dirpath = Path(dirpath)
        reward_cls = REWARD_REGISTRY[name]
        reward = reward_cls()
        normalizer_data = np.load(dirpath / f"{name}.normalizer.npz")
        normalizer = RunningZScore()
        normalizer.n = int(normalizer_data["n"])
        normalizer.mean = float(normalizer_data["mean"])
        normalizer.M2 = float(normalizer_data["M2"])
        agent = cls(
            name=name,
            reward_computer=reward,
            env=env,
            normalizer=normalizer,
        )
        agent.model = SAC.load(str(dirpath / f"{name}.zip"), env=env)
        return agent


def make_agent(
    name: str,
    seed: int = 0,
    tensorboard_log: Optional[str] = None,
    env_kwargs: Optional[dict] = None,
) -> CommitteeAgent:
    """Construct one committee member from its registry name."""
    if name not in REWARD_REGISTRY:
        raise KeyError(f"Unknown reward {name!r}; registry: {list(REWARD_REGISTRY)}")
    reward = REWARD_REGISTRY[name]()
    base_env = CommitteeEnv(reward_computer=reward, **(env_kwargs or {}))
    normalizer = RunningZScore()
    wrapped = ZScoreRewardWrapper(base_env, normalizer=normalizer)
    return CommitteeAgent(
        name=name,
        reward_computer=reward,
        env=wrapped,
        normalizer=normalizer,
        seed=seed,
        tensorboard_log=tensorboard_log,
    )
