"""Greedy (uncertainty-blind) policy.

At each iteration selects the candidate protocol predicted by the surrogate
model to produce the least battery degradation, ignoring model uncertainty
entirely.  This is a pure exploitation strategy that typically converges
quickly to a local minimum but may miss better protocols in unexplored
regions.

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


class GreedyPolicy(BasePolicy):
    """Greedy exploitation: selects the candidate with the lowest predicted degradation.

    Scores each candidate protocol with::

        score_i = degradation_fn(unscale(μ_i))

    where ``μ_i`` is the surrogate's mean prediction for the next ECM state
    given the current state and candidate protocol ``i``.  The candidate with
    the lowest score is selected.

    Model uncertainty (posterior standard deviation) is not consulted.  This
    is equivalent to setting ``κ = 0`` in the LCB acquisition function and
    makes the policy a pure exploitation strategy.

    Parameters
    ----------
    model :
        Trained surrogate with a ``predict(X)`` method returning mean
        predictions.
    num_action_features : int
        Length of each candidate protocol vector.
    degradation_fn : callable, optional
        ``f(state: np.ndarray) -> float``.  Higher = more degraded.
        Defaults to mean absolute value of the state vector.
    scaler : sklearn-compatible scaler, optional
        If provided, inputs are scaled before prediction and outputs are
        inverse-transformed before computing degradation.
    """

    def __init__(
        self,
        model,
        num_action_features: int = 6,
        degradation_fn: Optional[Callable[[np.ndarray], float]] = None,
        scaler=None,
    ) -> None:
        super().__init__(
            model=model,
            num_action_features=num_action_features,
            degradation_fn=degradation_fn,
            scaler=scaler,
        )

    def select_next(
        self,
        current_state: np.ndarray,
        candidate_protocols: list[np.ndarray],
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Select the candidate with the minimum predicted degradation.

        Parameters
        ----------
        current_state : np.ndarray
            Flattened, unscaled ECM parameter vector, shape
            ``(num_param_features,)``.
        candidate_protocols : list of np.ndarray
            Candidate charge-protocol vectors, each shape
            ``(num_action_features,)``.

        Returns
        -------
        best_protocol : np.ndarray
            Protocol with the lowest predicted degradation.
        best_score : float
            ``degradation_fn`` value for the selected candidate.
        scores : np.ndarray
            ``degradation_fn`` values for every candidate.
        """
        scores = np.empty(len(candidate_protocols))
        for i, protocol in enumerate(candidate_protocols):
            x_scaled = self._build_input(current_state, protocol)
            mean, _ = self._predict(x_scaled)
            state_pred = self._unscale_state(mean[0])
            scores[i] = self.degradation_fn(state_pred)

        best = int(np.argmin(scores))
        return np.asarray(candidate_protocols[best]), float(scores[best]), scores
