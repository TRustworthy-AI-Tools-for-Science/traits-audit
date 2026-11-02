"""State-space policy with SVD dimensionality reduction and linear dynamics.

Inspired by autopilot control systems (LQR), this policy learns a
low-dimensional linear dynamical model from the observed state-action history
and uses it to predict future battery states.  The high-dimensional ECM
parameter vector is projected onto a principal subspace via Singular Value
Decomposition (SVD); a linear transition model is then identified in that
subspace by least squares.  Candidate protocols are scored by a composite cost
function that combines:

1. **Degradation cost** — predicted degradation of the next state under the
   learned linear dynamics (exploitation of the identified system model).
2. **Control-effort cost** — L2 penalty on the protocol vector (regularisation
   against extreme charge rates, analogous to the ``R`` matrix in LQR).
3. **Uncertainty bonus** — GP posterior standard deviation encourages
   exploration of regions where the surrogate is uncertain.

When an audit uncertainty vector is available (from
``ParameterInferenceAudit.check_parameter_stability()`` or ``propagate_fde()``), it is
appended to the ECM state to form a *belief state* that encodes both the
estimated parameter values and their per-parameter posterior variance.  This
augmented representation allows the linear dynamics to learn how uncertainty
itself evolves under each protocol — following the belief-state MDP formulation
in which the agent conditions on its full posterior over hidden variables rather
than a point estimate.

Before enough transitions have been observed to fit the dynamics model, the
policy falls back to :class:`~battery_forecast.policies.LCBPolicy` (GP-only).

References
----------
Brunton, S. L., & Kutz, J. N. (2022). *Data-Driven Science and Engineering:
Machine Learning, Dynamical Systems, and Control* (2nd ed.), Ch. 7-8.
Cambridge University Press. https://doi.org/10.1017/9781009089517

Deisenroth, M. P., & Rasmussen, C. E. (2011). PILCO: A model-based and
data-efficient approach to policy search. In *Proceedings of the 28th
International Conference on Machine Learning (ICML)*, pp. 465-472.
https://proceedings.mlr.press/v15/deisenroth11a.html

Kaelbling, L. P., Littman, M. L., & Cassandra, A. R. (1998). Planning and
acting in partially observable stochastic domains. *Artificial Intelligence*,
101(1-2), 99-134. https://doi.org/10.1016/S0004-3702(98)00023-X
(Belief-state MDP: augmenting state with posterior uncertainty so the agent
conditions on its full information state, not just a point estimate.)

Schmid, P. J. (2010). Dynamic mode decomposition of numerical and
experimental data. *Journal of Fluid Mechanics*, 656, 5-28.
https://doi.org/10.1017/S0022112010001217
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

from .. import utils
from ..oracle import OracleFailure
from .base import BasePolicy
from .lcb import LCBPolicy

log = utils.get_logger(__name__, level=logging.INFO)


class StateSpacePolicy(BasePolicy):
    """SVD projection + linear dynamics + LQR-style composite cost.

    **Algorithm**

    At each call to :meth:`observe`:

    1. Append the ``(state, action)`` pair to history.  If an audit
       *uncertainty vector* is provided, it is concatenated with the state to
       form an augmented belief state ``s̃ = [state | uncertainty_weight * σ²]``,
       following the belief-state MDP of Kaelbling et al. (1998).
    2. Once ``min_fit_obs`` transitions are available, fit the model:

       * **SVD** of the ``(T × n_aug)`` augmented-state matrix identifies the
         ``n_components`` principal axes ``U_r ∈ ℝ^{n_aug × r}`` (similar to
         Dynamic Mode Decomposition).
       * **Linear dynamics** in the reduced space via least squares::

               z_{t+1} = A_r z_t + B_r a_t,   z_t = s̃_t @ U_r

    At each call to :meth:`select_next`:

    3. For each candidate protocol ``a``:

       * Project current augmented state: ``z = current_aug @ U_r``
       * Predict next reduced state: ``z_next = A_r @ z + B_r @ a``
       * Reconstruct: ``s̃_next ≈ U_r @ z_next``; take first ``_n_state``
         elements for degradation.
       * Query GP surrogate for uncertainty: ``σ = GP_std(current_state, a)``
       * Score::

               score = Q · degradation(s̃_next[:n_state])
                     + R · ‖a‖²
                     − κ · mean(σ)

    4. Return the candidate with the lowest score.

    Parameters
    ----------
    model :
        Trained surrogate with uncertainty support.  Used for the LCB
        fallback during warm-up and for uncertainty queries thereafter.
    n_components : int
        Rank of the SVD approximation (number of principal state axes).
    kappa : float
        Exploration weight on GP uncertainty (same role as in LCB).
    Q_weight : float
        Cost weight on predicted degradation.
    R_weight : float
        Cost weight on control effort ``‖a‖²``.
    min_fit_obs : int
        Minimum observations before the dynamics model is fitted.  Falls
        back to LCB until this threshold is reached.
    num_action_features : int
        Length of each candidate protocol vector.
    uncertainty_weight : float
        Scaling factor applied to the audit uncertainty vector before
        concatenation with the ECM state.  Increase to give uncertainty
        magnitudes more influence on the learned dynamics.
    degradation_fn : callable, optional
        ``f(state: np.ndarray) -> float``.  Higher = more degraded.
    scaler : sklearn-compatible scaler, optional
    """

    def __init__(
        self,
        model,
        n_components: int = 10,
        kappa: float = 1.0,
        Q_weight: float = 1.0,
        R_weight: float = 0.01,
        C_rate_bonus: float = 0.0,
        min_fit_obs: int = 5,
        num_action_features: int = 6,
        uncertainty_weight: float = 1.0,
        degradation_fn: Optional[Callable[[np.ndarray], float]] = None,
        scaler=None,
    ) -> None:
        super().__init__(
            model=model,
            num_action_features=num_action_features,
            degradation_fn=degradation_fn,
            scaler=scaler,
        )
        self.n_components = n_components
        self.kappa = kappa
        self.Q_weight = Q_weight
        self.R_weight = R_weight
        self.C_rate_bonus = C_rate_bonus
        self.min_fit_obs = min_fit_obs
        self.uncertainty_weight = uncertainty_weight

        # Warm-up fallback (GP-only LCB) used before dynamics are fitted
        self._fallback = LCBPolicy(
            model=model,
            kappa=kappa,
            num_action_features=num_action_features,
            degradation_fn=degradation_fn,
            scaler=scaler,
        )

        # Accumulated history
        self._state_history: list[np.ndarray] = []
        self._action_history: list[np.ndarray] = []
        self._uncertainty_history: list[Optional[np.ndarray]] = []

        # Length of the raw ECM state (set on first observe); used to split
        # the reconstructed augmented vector back into state + uncertainty.
        self._n_state: Optional[int] = None

        # Fitted reduced-order model (None until min_fit_obs reached)
        self._U_r: Optional[np.ndarray] = None  # (n_aug, r)
        self._A_r: Optional[np.ndarray] = None  # (r, r)
        self._B_r: Optional[np.ndarray] = None  # (r, m)

    # ── Belief-state construction ─────────────────────────────────────────

    def _build_augmented(
        self, state: np.ndarray, uncertainty: Optional[np.ndarray]
    ) -> np.ndarray:
        """Concatenate ECM state with scaled uncertainty vector.

        Returns the state unchanged when no uncertainty vector is available.
        Scaling by ``uncertainty_weight`` lets the caller control how much
        influence the uncertainty dimensions have on the SVD basis relative
        to the ECM parameter dimensions.

        This augmentation implements the belief-state MDP idea of Kaelbling
        et al. (1998): the agent conditions on its full posterior over hidden
        variables (here, ECM parameters) rather than a point estimate.
        """
        if uncertainty is None:
            return state
        return np.concatenate(
            [state, self.uncertainty_weight * np.asarray(uncertainty, dtype=float)]
        )

    # ── Stateful history ──────────────────────────────────────────────────

    def observe(
        self,
        state: np.ndarray,
        action: np.ndarray,
        uncertainty_vector: Optional[np.ndarray] = None,
    ) -> None:
        """Record an observed ``(state, action)`` pair and refit the model.

        Parameters
        ----------
        state : np.ndarray
            Flattened, unscaled ECM parameter vector.
        action : np.ndarray
            Protocol vector applied at this step.
        uncertainty_vector : np.ndarray, optional
            Per-parameter variance vector from the uncertainty audit
            (e.g., from ``ParameterInferenceAudit.check_parameter_stability()`` or
            ``propagate_fde()``), shape ``(n_params,)``.  When provided,
            it is appended to the state to form an augmented belief state.
        """
        state = np.asarray(state, dtype=float)
        if self._n_state is None:
            self._n_state = len(state)

        self._state_history.append(state)
        self._action_history.append(np.asarray(action, dtype=float))
        self._uncertainty_history.append(
            np.asarray(uncertainty_vector, dtype=float)
            if uncertainty_vector is not None
            else None
        )
        if len(self._state_history) >= self.min_fit_obs:
            self._fit_dynamics()

    # ── Dynamics identification ───────────────────────────────────────────

    def _fit_dynamics(self) -> None:
        """Fit SVD basis and linear transition model on augmented belief states.

        The augmented state ``s̃ = [state | uncertainty_weight * σ²]`` is used
        as the observation so the SVD and regression capture how both ECM
        parameters *and* their uncertainties co-evolve under each protocol.

        **SVD step** — decomposes the ``(n_aug × T)`` augmented-state matrix
        to find the ``r`` dominant directions of variation.

        **Regression step** — fits ``A_r``, ``B_r`` by solving::

            [Z_{1:T}] ≈ [Z_{0:T-1} | A_{0:T-1}] @ W^T,   W ∈ ℝ^{(r+m) × r}
        """
        # Infer uncertainty dimension from the first non-None entry so that
        # None entries (no audit data available yet) are replaced with zeros
        # of the same length, keeping the augmented state shape homogeneous.
        u_dim = next(
            (len(u) for u in self._uncertainty_history if u is not None), 0
        )
        aug_states = np.array([
            self._build_augmented(s, u if u is not None else np.zeros(u_dim))
            for s, u in zip(self._state_history, self._uncertainty_history)
        ])  # (T, n_aug)
        A = np.array(self._action_history[:-1])  # (T-1, n_action)
        T = len(A)

        if T < 2:
            return

        r = min(self.n_components, aug_states.shape[1], T)

        if not np.isfinite(aug_states).all() or not np.isfinite(A).all():
            log.warning("[StateSpacePolicy] aug_states/actions contain NaN/inf — skipping dynamics fit")
            return

        # SVD of augmented state matrix
        U, _, _ = np.linalg.svd(aug_states.T, full_matrices=False)
        self._U_r = U[:, :r]                       # (n_aug, r)

        # Project augmented states to reduced space
        Z = aug_states @ self._U_r                 # (T, r)

        # Least-squares regression: Z[1:] = [Z[:-1], A] @ W
        X_fit = np.hstack([Z[:T], A])              # (T-1, r+m)
        Y_fit = Z[1 : T + 1]                       # (T-1, r)
        if not (np.isfinite(X_fit).all() and np.isfinite(Y_fit).all()):
            log.warning("[StateSpacePolicy] projected X_fit/Y_fit contain NaN/inf — skipping dynamics fit")
            return
        W, _, _, _ = np.linalg.lstsq(X_fit, Y_fit, rcond=None)

        self._A_r = W[:r].T                        # (r, r)
        self._B_r = W[r:].T                        # (r, m)

        log.debug(
            f"[StateSpacePolicy] refitted dynamics: "
            f"r={r}, T={T}, n_aug={aug_states.shape[1]}, "
            f"A_r={self._A_r.shape}, B_r={self._B_r.shape}"
        )

    # ── Candidate scoring ─────────────────────────────────────────────────

    def _score_candidate(
        self,
        current_aug: np.ndarray,
        current_state: np.ndarray,
        protocol: np.ndarray,
        prot_scale: Optional[np.ndarray] = None,
    ) -> float:
        """Compute the composite LQR-style score for one candidate.

        Uses the augmented belief state for dynamics projection and splits the
        reconstructed vector at ``_n_state`` to extract only the ECM parameter
        portion for degradation scoring.

        ``prot_scale`` (per-component max absolute value across the candidate
        pool) normalises the control cost so that ``R_weight * ‖a/scale‖²``
        stays in [0, n_action * R_weight] regardless of the physical units of
        the protocol vector (mA, hours, etc.).  Without normalisation the mA-
        scale currents (~100) dominate the Ω-scale degradation signal (~0.1)
        by three orders of magnitude.
        """
        # GP uncertainty for this (state, protocol) pair
        x_scaled = self._build_input(current_state, protocol)
        _, std = self._predict(x_scaled)
        uncertainty = float(np.mean(std[0])) if std is not None else 0.0

        # Linear-dynamics prediction in reduced space
        z = current_aug @ self._U_r                                # (r,)
        z_next = self._A_r @ z + self._B_r @ np.asarray(protocol)
        s_aug_next = self._U_r @ z_next                            # (n_aug,)

        # Slice off the ECM state portion; ignore reconstructed uncertainty dims
        n = self._n_state if self._n_state is not None else s_aug_next.shape[0]
        s_next = s_aug_next[:n]

        state_cost = self.Q_weight * self.degradation_fn(s_next)
        # Normalise protocol to dimensionless units before computing R cost
        p_norm = (
            np.asarray(protocol) / prot_scale
            if prot_scale is not None
            else np.asarray(protocol)
        )
        control_cost = self.R_weight * float(np.dot(p_norm, p_norm))
        uncertainty_bonus = self.kappa * uncertainty
        # Explicit charge-rate reward: subtract bonus proportional to the
        # normalised first-stage charge current so higher rates are preferred
        # when degradation is comparable.  Disabled when C_rate_bonus = 0.
        c_rate_bonus = (
            self.C_rate_bonus * float(p_norm[0])
            if self.C_rate_bonus != 0.0
            else 0.0
        )

        return state_cost + control_cost - uncertainty_bonus - c_rate_bonus

    # ── Public interface ──────────────────────────────────────────────────

    def select_next(
        self,
        current_state: np.ndarray,
        candidate_protocols: list[np.ndarray],
        uncertainty_vector: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Select the candidate that minimises the composite cost.

        Falls back to :class:`~battery_forecast.policies.LCBPolicy` (GP-only)
        until ``min_fit_obs`` transitions have been observed.

        Parameters
        ----------
        current_state : np.ndarray
            Flattened, unscaled ECM parameter vector, shape
            ``(num_param_features,)``.
        candidate_protocols : list of np.ndarray
            Candidate charge-protocol vectors, each shape
            ``(num_action_features,)``.
        uncertainty_vector : np.ndarray, optional
            Per-parameter variance vector from the uncertainty audit for the
            current state.  Concatenated with ``current_state`` to form the
            belief state used for dynamics projection.

        Returns
        -------
        best_protocol : np.ndarray
            Protocol with the lowest composite score.
        best_score : float
            Composite score of the selected candidate.
        scores : np.ndarray
            Composite scores for every candidate.
        """
        if self._A_r is None:
            n_obs = len(self._state_history)
            log.info(
                f"[StateSpacePolicy] warm-up ({n_obs}/{self.min_fit_obs} obs): "
                "using LCB fallback"
            )
            return self._fallback.select_next(current_state, candidate_protocols)

        current_aug = self._build_augmented(
            np.asarray(current_state, dtype=float), uncertainty_vector
        )
        # Normalise protocol components by their max absolute value across the
        # candidate pool so the control cost is dimensionless and comparable to
        # the Ω-scale degradation signal (fixes raw-mA vs Ω scale mismatch).
        cand_arr = np.array([np.asarray(p) for p in candidate_protocols])
        prot_scale = np.maximum(np.abs(cand_arr).max(axis=0), 1.0)
        scores = np.array(
            [
                self._score_candidate(current_aug, current_state, p, prot_scale)
                for p in candidate_protocols
            ]
        )
        best = int(np.argmin(scores))
        return np.asarray(candidate_protocols[best]), float(scores[best]), scores

    def run(
        self,
        initial_state: np.ndarray,
        candidate_protocols: list[np.ndarray],
        oracle_fn,
        n_iterations: int = 10,
        uncertainty_oracle_fn=None,
        on_step: Optional[Callable] = None,
    ) -> dict:
        """Active learning loop with optional uncertainty oracle.

        Parameters
        ----------
        initial_state : np.ndarray
            Starting ECM state vector.
        candidate_protocols : list of np.ndarray
            Pool of candidate protocols to select from at each step.
        oracle_fn : callable
            ``oracle_fn(protocol) -> next_state``.  Runs the experiment and
            returns the resulting ECM state.
        n_iterations : int
            Number of selection-observe cycles.
        uncertainty_oracle_fn : callable, optional
            ``uncertainty_oracle_fn(protocol) -> np.ndarray``.  Returns the
            per-parameter variance vector from the uncertainty audit for the
            experiment associated with ``oracle_fn``.  When provided, the
            uncertainty vector is passed to both :meth:`observe` and the next
            :meth:`select_next` call so the policy conditions on the full
            belief state at every step.

            For audit-hook integration, pass
            ``uncertainty_oracle_fn=lambda _p: hook.latest_uncertainty_vector``
            — the hook's :attr:`latest_uncertainty_vector` property reads the
            most recent ``uncertainty_vector`` kwarg supplied to
            ``hook.on_step(...)``.
        on_step : callable, optional
            Called after each iteration with keyword arguments:
            ``iteration``, ``protocol``, ``state``, ``next_state``,
            ``degradation``, ``score``, ``uncertainty_vector``.  Compatible
            with :meth:`~traits_audit.hook.AuditHook.on_step`.

        Returns
        -------
        dict with keys ``selected_protocols``, ``observed_states``,
        ``scores``, ``degradation``, ``policy``.
        """
        current_state = np.asarray(initial_state, dtype=float)
        current_uncertainty: Optional[np.ndarray] = None

        selected_protocols = []
        observed_states = [current_state]
        scores_history = []
        degradation_history = [self.degradation_fn(current_state)]

        n_cycles_completed = 0
        terminated_early   = False

        for i in range(n_iterations):
            protocol, score, all_scores = self.select_next(
                current_state,
                candidate_protocols,
                uncertainty_vector=current_uncertainty,
            )

            try:
                next_state = np.asarray(oracle_fn(protocol), dtype=float)
                succeeded = True
            except OracleFailure as exc:
                log.warning(
                    "[StateSpacePolicy] oracle failure at iter %d — %s; "
                    "ending loop.",
                    i + 1, exc,
                )
                next_state = np.full_like(current_state, np.nan)
                succeeded = False

            # Always record — killing protocol is diagnostic data.
            selected_protocols.append(protocol)
            observed_states.append(next_state)
            scores_history.append(all_scores)
            degradation_history.append(self.degradation_fn(next_state))

            if not succeeded:
                terminated_early = True
                break

            next_uncertainty = (
                np.asarray(uncertainty_oracle_fn(protocol), dtype=float)
                if uncertainty_oracle_fn is not None
                else None
            )

            # observe() only on success — NaN state would corrupt dynamics model.
            self.observe(current_state, protocol, uncertainty_vector=current_uncertainty)

            log.info(
                f"[StateSpacePolicy] iter {i + 1}/{n_iterations}: "
                f"score={score:.4f}, degradation={degradation_history[-1]:.4f}"
            )

            if on_step is not None:
                on_step(
                    iteration=i,
                    protocol=protocol,
                    state=current_state,
                    next_state=next_state,
                    degradation=degradation_history[-1],
                    score=float(score),
                    uncertainty_vector=current_uncertainty,
                )

            current_state = next_state
            current_uncertainty = next_uncertainty
            n_cycles_completed = i + 1

        return {
            "selected_protocols": selected_protocols,
            "observed_states":    observed_states,
            "scores":             scores_history,
            "degradation":        degradation_history,
            "policy":             self,
            "n_cycles_completed": n_cycles_completed,
            "terminated_early":   terminated_early,
        }
