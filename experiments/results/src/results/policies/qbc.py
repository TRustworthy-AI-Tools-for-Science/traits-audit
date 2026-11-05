"""Query-by-committee (QBC) policy.

Uses a committee of models (e.g. VarFN, VarGPR, HSGP) to select the next
protocol based on their agreement or disagreement over predicted degradation.

Both modes share the same underlying statistic — the standard deviation of
the committee's per-candidate degradation predictions — and differ only in
which extreme is selected.

When ``agree=True`` (consensus mode), the candidate with the *lowest*
committee standard deviation is selected — favouring protocols the committee
is most confident and consistent about.

When ``agree=False`` (disagreement mode), the candidate with the *highest*
committee standard deviation is selected — favouring the most contested
protocol, where the committee disagrees most.

Reference
---------
Burbidge, R., Rowland, J. J., & King, R. D. (2007). Active learning for
regression based on query by committee. In *Intelligent Data Engineering
and Automated Learning (IDEAL 2007)*, LNCS vol. 4881, pp. 209–218.
https://doi.org/10.1007/978-3-540-77226-2_22
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
        If ``True`` (default), select the candidate the committee most agrees
        on, i.e. minimum committee standard deviation over predicted
        degradation (consensus).  If ``False``, select the most contested
        candidate, i.e. maximum committee standard deviation (disagreement).
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

            spread = float(np.std(degradation_scores))
            # Consensus (agree=True) and disagreement (agree=False) are the
            # same statistic, just inverted: argmin picks lowest spread for
            # consensus, or (via negation) highest spread for disagreement.
            scores[i] = spread if self.agree else -spread

        best = int(np.argmin(scores))
        return np.asarray(candidate_protocols[best]), float(scores[best]), scores
