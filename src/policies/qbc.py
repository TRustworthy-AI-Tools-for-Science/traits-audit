"""Query-by-committee (QBC) policy.

Uses a committee of models (e.g. VarFN, VarGPR, HSGP) to select the next
protocol based on their agreement or disagreement over predicted degradation.

When ``agree=True`` (exploitation mode), the candidate with the lowest *mean*
committee degradation is selected — favouring protocols that all models
consistently predict to be good.

When ``agree=False`` (exploration mode), the candidate with the highest
committee *entropy* over predicted degradation is selected — favouring the
most contested protocol, where the committee disagrees most.

Reference
---------
Seung, H. S., Opper, M., & Sompolinsky, H. (1992). Query by committee.
In *Proceedings of the Fifth Annual Workshop on Computational Learning
Theory (COLT)*, pp. 287–294.
https://doi.org/10.1145/130385.130417
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .base import BasePolicy


class QBCPolicy(BasePolicy):
    """Query-by-committee active learning policy.

    Parameters
    ----------
    model_list : list
        Committee of trained surrogates.  Each must support
        ``predict_with_uncertainty(X)`` (returning ``{'mean': ..., 'std': ...}``)
        or ``predict(X, return_std=True)``.
    agree : bool
        If ``True`` (default), select the candidate where the committee agrees
        on the lowest degradation (exploitation).  If ``False``, select the
        most contested candidate, i.e. maximum committee entropy (exploration).
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
        model_list: list,
        agree: bool = True,
        num_action_features: int = 6,
        degradation_fn: Optional[Callable[[np.ndarray], float]] = None,
        scaler=None,
    ) -> None:
        if not model_list:
            raise ValueError("model_list must contain at least one model.")
        super().__init__(
            model=model_list[0],
            num_action_features=num_action_features,
            degradation_fn=degradation_fn,
            scaler=scaler,
        )
        self.model_list = model_list
        self.agree = agree

    def _predict(
        self, X_scaled: np.ndarray
    ) -> list[tuple[np.ndarray, Optional[np.ndarray]]]:
        """Return a list of ``(mean, std_or_None)`` — one entry per committee member."""
        results = []
        for m in self.model_list:
            if hasattr(m, "predict_with_uncertainty"):
                res = m.predict_with_uncertainty(X_scaled)
                results.append((res["mean"], res.get("std")))
            else:
                try:
                    mean, std = m.predict(X_scaled, return_std=True)
                    results.append((mean, std))
                except TypeError:
                    results.append((m.predict(X_scaled), None))
        return results

    def _entropy(self, X: np.ndarray) -> float:
        """Shannon entropy of X treated as an unnormalised probability distribution.

        Absolute values of X are normalised to sum to 1, then
        H = -Σ p_i log p_i is returned.  A uniform committee (maximum
        disagreement) maximises entropy; a unanimous committee minimises it.
        """
        x = np.abs(np.asarray(X, dtype=float))
        total = x.sum()
        if total == 0.0:
            return 0.0
        p = x / total
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    def select_next(
        self,
        current_state: np.ndarray,
        candidate_protocols: list[np.ndarray],
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Select the candidate with the [min|max] committee disagreement.

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
            Selected protocol.
        best_score : float
            Acquisition score of the selected candidate (lower = better).
        scores : np.ndarray
            Acquisition scores for every candidate.
        """
        scores = np.empty(len(candidate_protocols))

        for i, protocol in enumerate(candidate_protocols):
            x_scaled = self._build_input(current_state, protocol)
            committee_preds = self._predict(x_scaled)

            degradation_scores = []
            for mean, _ in committee_preds:
                state_pred = self._unscale_state(mean[0])
                degradation_scores.append(self.degradation_fn(state_pred))

            if self.agree:
                # Exploitation: lowest mean predicted degradation across the committee.
                scores[i] = float(np.mean(degradation_scores))
            else:
                # Exploration: negate entropy so argmin recovers the most contested candidate.
                scores[i] = -self._entropy(degradation_scores)

        best = int(np.argmin(scores))
        return np.asarray(candidate_protocols[best]), float(scores[best]), scores
