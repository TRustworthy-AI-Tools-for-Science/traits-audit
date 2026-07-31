"""Random baseline policy.

Selects a charge protocol uniformly at random from the candidate set at
each iteration.  Provides a lower-bound performance reference: any informed
policy should outperform random selection over enough iterations.

Reference
---------
Settles, B. (2009). Active learning literature survey. Computer Sciences
Technical Report 1648, University of Wisconsin–Madison.
https://burrsettles.com/pub/settles.activelearning.pdf
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .base import BasePolicy


class RandomPolicy(BasePolicy):
    """Uniform-random baseline: selects a candidate protocol uniformly at random.

    At each iteration a candidate is drawn uniformly at random from
    *candidate_protocols*, without consulting the surrogate model.  All
    candidates receive a score of 0.0 (scores are meaningless for this policy).

    This policy requires no surrogate model and can therefore be used as a
    model-free baseline or during early warm-up phases where no trained
    model is yet available.

    Parameters
    ----------
    model : optional
        Not used for candidate selection; kept for API consistency with other
        policies so that all policies can be swapped without changing call sites.
    num_action_features : int
        Length of each candidate protocol vector.
    degradation_fn : callable, optional
        ``f(state) -> float``.  Used only in :meth:`~base.BasePolicy.run`
        to record observed degradation.  Defaults to mean absolute value.
    scaler : optional
        Not used; kept for API consistency.
    rng : np.random.Generator, optional
        Random number generator.  Pass an explicit seed for reproducibility::

            policy = RandomPolicy(rng=np.random.default_rng(42))
    """

    def __init__(
        self,
        model=None,
        num_action_features: int = 6,
        degradation_fn: Optional[Callable[[np.ndarray], float]] = None,
        scaler=None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(
            model=model,
            num_action_features=num_action_features,
            degradation_fn=degradation_fn,
            scaler=scaler,
        )
        self.rng = rng or np.random.default_rng()

    def select_next(
        self,
        current_state: np.ndarray,
        candidate_protocols: list[np.ndarray],
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Select a candidate uniformly at random.

        Parameters
        ----------
        current_state : np.ndarray
            Ignored; included for interface compatibility.
        candidate_protocols : list of np.ndarray
            Pool of candidate protocols to draw from.

        Returns
        -------
        best_protocol : np.ndarray
            A randomly selected protocol.
        best_score : float
            Always 0.0 (scores are not meaningful for this policy).
        scores : np.ndarray
            All-zeros array of length ``len(candidate_protocols)``.
        """
        n = len(candidate_protocols)
        idx = int(self.rng.integers(0, n))
        scores = np.zeros(n)
        return np.asarray(candidate_protocols[idx]), 0.0, scores
