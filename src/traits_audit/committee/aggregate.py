"""Committee aggregation rules — v0 only ships uniform random pick.

Disagreement-weighted picking is a v1 question (see plan).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from traits_audit.committee.agents import CommitteeAgent


def uniform_random_pick(
    agents: Sequence[CommitteeAgent],
    obs: np.ndarray,
    rng: np.random.Generator,
    deterministic: bool = True,
) -> tuple[float, int, np.ndarray]:
    """Each agent proposes an x; one is selected uniformly at random.

    Returns
    -------
    x_chosen : float
        The selected proposal.
    selected_idx : int
        Which agent was selected (index into ``agents``).
    all_proposals : np.ndarray, shape (n_agents,)
        Every agent's proposal (for logging/diagnostics).
    """
    proposals = np.array([
        float(agent.predict(obs, deterministic=deterministic)[0])
        for agent in agents
    ])
    selected_idx = int(rng.integers(0, len(agents)))
    return float(proposals[selected_idx]), selected_idx, proposals
