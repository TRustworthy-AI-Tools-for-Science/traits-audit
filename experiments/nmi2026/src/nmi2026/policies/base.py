"""Abstract base class for battery active learning policies."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np

from .. import utils
from ..oracle import OracleFailure

log = utils.get_logger(__name__, level=logging.INFO)


def _default_degradation(state: np.ndarray) -> float:
    """Mean absolute value of the state vector."""
    return float(np.mean(np.abs(state)))


def resistance_degradation(
    state: np.ndarray,
    resistance_indices: Optional[list[int]] = None,
) -> float:
    """Degradation proxy based on resistance parameter means.

    For the default circuit ``R1-P2-[R3,P4]-[R5,P6]`` with the ``mzscore``
    featurisation (4 moments per parameter, charge + discharge), resistance
    means appear at indices 0, 12, 24 (charge) and 36, 48, 60 (discharge)
    in the 72-dimensional state vector.

    Parameters
    ----------
    state : np.ndarray
        ECM parameter feature vector (unscaled).
    resistance_indices : list of int, optional
        Indices of resistance-mean features.  Defaults to R1, R3, R5 in the
        standard 72-dim state.

    Returns
    -------
    float
        Sum of resistance-mean values (higher = more degraded).
    """
    if resistance_indices is None:
        resistance_indices = [0, 12, 24, 36, 48, 60]
    return float(np.sum(state[resistance_indices]))


class BasePolicy(ABC):
    """Abstract base for battery charge-rate active learning policies.

    All policies share a common interface:

    - :meth:`select_next` — abstract; implemented by each concrete policy.
    - :meth:`observe` — hook for stateful policies; no-op by default.
    - :meth:`run` — shared active learning loop.

    Subclasses must implement :meth:`select_next` and may override
    :meth:`observe` to accumulate data between iterations.

    Parameters
    ----------
    model :
        Trained surrogate.  Must support ``predict(X)``; uncertainty-aware
        policies additionally require ``predict_with_uncertainty(X)``
        (returning ``{'mean': ..., 'std': ...}``) or
        ``predict(X, return_std=True)``.
    num_action_features : int
        Number of features in each candidate protocol vector (appended at
        the end of the combined ``[state | protocol]`` input vector).
    degradation_fn : callable, optional
        ``f(state: np.ndarray) -> float``.  Higher return value = more
        degraded.  Defaults to :func:`_default_degradation`.
    scaler : sklearn-compatible scaler, optional
        If provided, inputs are scaled before model prediction and model
        outputs are inverse-transformed before computing degradation.
    """

    def __init__(
        self,
        model,
        num_action_features: int = 6,
        degradation_fn: Optional[Callable[[np.ndarray], float]] = None,
        scaler=None,
    ) -> None:
        self.model = model
        self.num_action_features = num_action_features
        self.degradation_fn = degradation_fn or _default_degradation
        self.scaler = scaler

    # ── Shared helpers ────────────────────────────────────────────────────

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X) if self.scaler is not None else X

    def _unscale_state(self, state_scaled: np.ndarray) -> np.ndarray:
        """Inverse-transform the state portion of a scaled output vector."""
        if self.scaler is None:
            return state_scaled
        n_action = self.num_action_features
        padded = np.concatenate([state_scaled, np.zeros(n_action)])
        unscaled = self.scaler.inverse_transform(padded.reshape(1, -1))[0]
        return unscaled[:-n_action]

    def _predict(
        self, X_scaled: np.ndarray
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Return ``(mean, std_or_None)`` in the model's output space."""
        if hasattr(self.model, "predict_with_uncertainty"):
            res = self.model.predict_with_uncertainty(X_scaled)
            return res["mean"], res.get("std")
        try:
            return self.model.predict(X_scaled, return_std=True)
        except TypeError:
            pass
        return self.model.predict(X_scaled), None

    def _build_input(
        self, state: np.ndarray, protocol: np.ndarray
    ) -> np.ndarray:
        """Concatenate state and protocol, scale, and return shape ``(1, n)``."""
        x_raw = np.concatenate([state, np.asarray(protocol)]).reshape(1, -1)
        return self._scale(x_raw)

    # ── Hook for stateful policies ────────────────────────────────────────

    def observe(self, state: np.ndarray, action: np.ndarray) -> None:
        """Record an observed ``(state, action)`` transition.

        Called automatically by :meth:`run` after each iteration.
        Stateless policies leave this as a no-op.  Stateful policies
        (e.g. :class:`~battery_forecast.policies.StateSpacePolicy`)
        override it to update their internal models.
        """

    # ── Abstract interface ────────────────────────────────────────────────

    @abstractmethod
    def select_next(
        self,
        current_state: np.ndarray,
        candidate_protocols: list[np.ndarray],
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Select the next charge protocol.

        Parameters
        ----------
        current_state : np.ndarray
            Flattened, unscaled ECM parameter vector for the current cycle,
            shape ``(num_param_features,)``.
        candidate_protocols : list of np.ndarray
            Candidate charge-protocol vectors, each shape
            ``(num_action_features,)``.

        Returns
        -------
        best_protocol : np.ndarray
            The selected charge protocol.
        best_score : float
            Acquisition score of the selected candidate (lower = better).
        scores : np.ndarray
            Score for every candidate in the same order as the input list.
        """

    # ── Shared run loop ───────────────────────────────────────────────────

    def run(
        self,
        initial_state: np.ndarray,
        candidate_protocols: list[np.ndarray],
        oracle_fn: Callable[[np.ndarray], np.ndarray],
        n_iterations: int = 10,
        on_step: Optional[Callable] = None,
    ) -> dict:
        """Run the full active learning loop.

        At each iteration:
        1. Call :meth:`select_next` to choose a protocol.
        2. Call :meth:`observe` with the current state and chosen protocol.
        3. Call *oracle_fn* to obtain the next state.
        4. Record the transition.

        Parameters
        ----------
        initial_state : np.ndarray
            Starting ECM parameter state (unscaled).
        candidate_protocols : list of np.ndarray
            Fixed candidate set to choose from at every iteration.
        oracle_fn : callable
            ``oracle_fn(protocol) -> next_state`` (unscaled ``np.ndarray``).
            Represents the physical battery experiment: apply *protocol*,
            measure EIS, infer ECM parameters, return the resulting state.
        n_iterations : int
            Number of charge cycles to optimise.

        Returns
        -------
        dict with keys:

        - ``selected_protocols`` - protocols chosen at each step
        - ``observed_states``    - ECM states (length ``n_iterations + 1``)
        - ``scores``             - acquisition scores at each step
        - ``degradation``        - degradation values of observed states
        - ``policy``             - class name of this policy
        """
        history: dict = {
            "selected_protocols": [],
            "observed_states": [initial_state],
            "scores": [],
            "degradation": [],
            "policy": type(self).__name__,
            "n_cycles_completed": 0,
            "terminated_early": False,
        }
        state = initial_state
        for i in range(n_iterations):
            log.info(f"[{type(self).__name__}] iteration {i + 1}/{n_iterations}")
            protocol, score, _ = self.select_next(state, candidate_protocols)
            self.observe(state, protocol)

            try:
                next_state = oracle_fn(protocol)
                succeeded = True
            except OracleFailure as exc:
                log.warning(
                    "[%s] oracle failure at iteration %d — protocol %s: %s; "
                    "ending loop.",
                    type(self).__name__, i + 1, protocol, exc,
                )
                next_state = np.full_like(state, np.nan)
                succeeded = False

            # Always record — the killing protocol and score are diagnostic data.
            history["selected_protocols"].append(protocol)
            history["observed_states"].append(next_state)
            history["scores"].append(float(score))
            history["degradation"].append(self.degradation_fn(next_state))

            if on_step is not None:
                step_kw: dict = {
                    "iteration":   i,
                    "protocol":    protocol,
                    "y_true":      next_state,
                    "degradation": history["degradation"][-1],
                    "score":       float(score),
                }
                if self.model is not None:
                    try:
                        x_sc = self._build_input(state, protocol)
                        pred_mean_sc, pred_std_sc = self._predict(x_sc)
                        step_kw["y_pred_mean"] = self._unscale_state(pred_mean_sc[0])
                        if pred_std_sc is not None:
                            step_kw["y_pred_std"] = pred_std_sc[0]
                            step_kw["uncertainty"] = float(
                                np.mean(np.abs(pred_std_sc[0]))
                            )
                        # Store model reference + current input so IntegratedGradients
                        # can run at each step without needing on_end kwargs.
                        step_kw["model"] = self.model
                        step_kw["input_features"] = x_sc
                    except Exception:
                        pass
                on_step(**step_kw)

            if succeeded:
                log.info(f"  degradation = {history['degradation'][-1]:.4f}")
                state = next_state
                history["n_cycles_completed"] = i + 1
            else:
                history["terminated_early"] = True
                break

        return history
