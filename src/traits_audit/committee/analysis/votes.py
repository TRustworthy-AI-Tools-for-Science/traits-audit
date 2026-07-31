"""Committee-vote primitive shared by Thread B (votes-as-features) and Thread A (aggregators).

A `CommitteeVoter` wraps the 9 frozen SAC seed-0 policies and exposes two views:

- ``preferred_actions(obs)`` — length-9 vector of each agent's deterministic
  preferred x at the current acquisition state. Cheap (one SB3 predict per
  policy, batched is not supported across distinct policies).
- ``votes_for(obs, x_candidates)`` — (K, 9) matrix where entry [i, k] is
  ``|x_candidates[i] - pi_k(obs)|``. Small distance = "agent k endorses this
  candidate." This is the vote shape Thread B feeds into augmented surrogates
  and Thread A's QBC aggregators consume.

Action-distance was chosen over Q-value votes because:
  1. SAC critics on this env give scores in z-scored-reward units that differ
     per agent (each agent's reward was z-scored on its own stream during
     training) — comparing across agents would require a normalization choice
     that we'd rather not bake in here.
  2. The actor's deterministic action is the natural "what would you query?"
     readout from an SAC trained with the reparameterization trick.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from traits_audit.committee.rewards import REWARD_REGISTRY


AGENT_NAMES: list[str] = list(REWARD_REGISTRY.keys())


@dataclass
class CommitteeVoter:
    """Frozen 9-policy committee that votes on candidate x's.

    Parameters
    ----------
    models_dir : Path
        Directory containing ``{agent}_seed{committee_solo_seed}.zip`` files.
    committee_solo_seed : int
        Which training seed to load (defaults to 0 to match the v0 committee).
    agents : iterable of str, optional
        Subset of agents to include. Defaults to all 9.
    """

    models_dir: Path
    committee_solo_seed: int = 0
    agents: Optional[Iterable[str]] = None
    _models: list = field(default_factory=list, init=False)
    _agent_names: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        from stable_baselines3 import SAC

        names = list(self.agents) if self.agents is not None else AGENT_NAMES
        self._agent_names = names
        # Skip allocating the full SAC replay buffer at load time. We only
        # need policy.predict() — the buffer is dead weight at eval and
        # 9 x 500_000 x 414 floats blows out RAM in this env.
        custom_objects = {"buffer_size": 1}
        for agent in names:
            p = self.models_dir / f"{agent}_seed{self.committee_solo_seed}.zip"
            if not p.exists():
                raise FileNotFoundError(f"Missing model: {p}")
            self._models.append(SAC.load(str(p), custom_objects=custom_objects))

    @property
    def agent_names(self) -> list[str]:
        return list(self._agent_names)

    @property
    def n_agents(self) -> int:
        return len(self._models)

    def preferred_actions(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic preferred x for each committee member at obs.

        Returns a length-``n_agents`` vector of floats in [0, 1].
        """
        prefs = np.empty(self.n_agents, dtype=np.float64)
        for k, model in enumerate(self._models):
            action, _ = model.predict(obs, deterministic=True)
            prefs[k] = float(np.clip(np.asarray(action).reshape(-1)[0], 0.0, 1.0))
        return prefs

    def votes_for(
        self, obs: np.ndarray, x_candidates: np.ndarray
    ) -> np.ndarray:
        """Action-distance vote vector per candidate.

        Returns a (K, n_agents) matrix where entry [i, k] is
        ``|x_candidates[i] - pi_k(obs)|``. Smaller = stronger endorsement.
        The preferred-action call is made once per agent and broadcast across
        candidates — votes for K candidates cost the same K=1 SB3 predicts.
        """
        prefs = self.preferred_actions(obs)
        x = np.asarray(x_candidates, dtype=np.float64).reshape(-1)
        return np.abs(x[:, None] - prefs[None, :])
