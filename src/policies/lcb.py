"""Lower Confidence Bound (LCB) policy.

Balances exploitation of low predicted degradation with exploration of
high-uncertainty regions of the protocol space.  For a GP surrogate with
posterior mean ``μ`` and standard deviation ``σ`` over the next ECM state::

    score_i = degradation_fn(unscale(μ_i)) − κ · mean(σ_i)

The candidate with the lowest score is selected.  Setting ``κ = 0`` recovers
the :class:`~battery_forecast.policies.GreedyPolicy`.

Reference
---------
Srinivas, N., Krause, A., Kakade, S. M., & Seeger, M. W. (2010).
Gaussian process optimization in the bandit setting: No regret and
experimental design. In *Proceedings of the 27th International Conference
on Machine Learning (ICML)*, pp. 1015–1022.
https://arxiv.org/abs/0912.3995
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .base import BasePolicy


class LCBPolicy(BasePolicy):
    """Lower Confidence Bound acquisition for battery charge-rate optimisation.

    Scores each candidate protocol ``i`` with::

        score_i = degradation_fn(unscale(μ_i)) − κ · mean(σ_i)

    where ``μ_i`` and ``σ_i`` are the GP posterior mean and standard deviation
    for the predicted next ECM state given the current state and protocol ``i``.
    The candidate with the *lowest* score is selected.

    The ``−κ · mean(σ)`` term encourages the policy to also probe protocols
    where the surrogate is uncertain (high ``σ``), preventing premature
    convergence to a local minimum.  Higher ``κ`` emphasises exploration;
    ``κ = 0`` gives pure exploitation (equivalent to
    :class:`~battery_forecast.policies.GreedyPolicy`).

    Parameters
    ----------
    model :
        Trained surrogate.  Must support ``predict_with_uncertainty(X)``
        (returning ``{'mean': ..., 'std': ...}``) or
        ``predict(X, return_std=True)``.  ``MyGPRModel`` from
        ``battery_forecast.model`` satisfies this.
    kappa : float
        Exploration weight.  Typical range: 1–3.  Higher values encourage
        the policy to explore uncertain regions.
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
        kappa: float = 2.0,
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
        self.kappa = kappa

    def select_next(
        self,
        current_state: np.ndarray,
        candidate_protocols: list[np.ndarray],
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Select the candidate with the minimum LCB acquisition score.

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
            Protocol with the lowest LCB score.
        best_score : float
            LCB score of the selected candidate.
        scores : np.ndarray
            LCB scores for every candidate.
        """
        scores = np.empty(len(candidate_protocols))
        for i, protocol in enumerate(candidate_protocols):
            x_scaled = self._build_input(current_state, protocol)
            mean, std = self._predict(x_scaled)
            state_pred = self._unscale_state(mean[0])
            degradation = self.degradation_fn(state_pred)
            exploration = self.kappa * float(np.mean(std[0])) if std is not None else 0.0
            scores[i] = degradation - exploration

        best = int(np.argmin(scores))
        return np.asarray(candidate_protocols[best]), float(scores[best]), scores
