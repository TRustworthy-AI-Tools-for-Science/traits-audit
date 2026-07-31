"""Constant charge-rate baseline policy.

Always charges at a fixed first-step C-rate regardless of the surrogate
model's predictions or the current battery state.  Provides a controlled
baseline: every iteration applies the same charge current so any degradation
difference vs. an informed policy is attributable to protocol choice alone.

Reference
---------
Brochu, E., Cora, V. M., & de Freitas, N. (2010). A tutorial on Bayesian
optimization of expensive cost functions, with application to active user
modeling and hierarchical reinforcement learning. arXiv:1012.2599.
https://arxiv.org/abs/1012.2599
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .base import BasePolicy


class ConstantPolicy(BasePolicy):
    """Always applies the same fixed charge current (1C by default).

    At every iteration the policy returns a protocol whose first charge step
    is set to *c_rate_mA*, regardless of the surrogate model or current state.
    All other protocol fields (durations, discharge current) are taken from the
    median of the candidate set so they remain physically reasonable.

    This is the constant-rate control condition: it answers the question
    "what happens if you always charge at the same rate?"

    Parameters
    ----------
    model :
        Kept for API consistency; not consulted for candidate selection.
    num_action_features : int
        Length of each candidate protocol vector (default 6).
    degradation_fn : callable, optional
        ``f(state) -> float``.  Used only to record observed degradation in
        :meth:`~base.BasePolicy.run`; not used for protocol selection.
    scaler : optional
        Not used; kept for API consistency.
    c_rate_mA : float
        First-step charge current in mA, in the *dataset's own current scale*
        (e.g. jones2022 mA, not the oracle's scaled 5 Ah mA).
        Default 200 mA = 1C on the jones2022 CR2032 / 5 Ah oracle cell
        (1C = 5000 mA oracle scale; _CAP_SCALE = 25; 5000 / 25 = 200 mA).
    """

    def __init__(
        self,
        model=None,
        num_action_features: int = 6,
        degradation_fn: Optional[Callable[[np.ndarray], float]] = None,
        scaler=None,
        c_rate_mA: float = 200.0,
    ) -> None:
        super().__init__(
            model=model,
            num_action_features=num_action_features,
            degradation_fn=degradation_fn,
            scaler=scaler,
        )
        self.c_rate_mA = c_rate_mA

    def select_next(
        self,
        current_state: np.ndarray,
        candidate_protocols: list[np.ndarray],
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Return a protocol with the fixed charge current every time.

        The median candidate is used as the template; only the first element
        (C_rate_1) is replaced with *c_rate_mA*.

        Parameters
        ----------
        current_state : np.ndarray
            Ignored; included for interface compatibility.
        candidate_protocols : list of np.ndarray
            Pool of candidate protocols; used to derive the median template.

        Returns
        -------
        protocol : np.ndarray
            Fixed-rate protocol.
        score : float
            Always 0.0 (not meaningful for this policy).
        scores : np.ndarray
            All-zeros array of length ``len(candidate_protocols)``.
        """
        median = np.median(np.stack(candidate_protocols), axis=0)
        protocol = median.copy()
        protocol[0] = self.c_rate_mA
        scores = np.zeros(len(candidate_protocols))
        return protocol, 0.0, scores
