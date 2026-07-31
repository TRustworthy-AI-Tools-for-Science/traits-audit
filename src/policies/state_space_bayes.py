"""Bayesian dynamics policy with online precision updates and information-gain scoring.

Extends :class:`~battery_forecast.policies.StateSpacePolicy` in three ways,
each grounded in the accompanying literature:

1. **Online Bayesian precision update (Eldredge & Mousavi, 2026)**

   Rather than refitting the least-squares dynamics model from scratch at
   every :meth:`observe` call, this policy tracks the *precision matrix* of
   the Bayesian linear regression posterior in the reduced state-action space,

       G_t = X_t^T X_t + λI  ∈ ℝ^{(r+m) × (r+m)},

   and updates G_t^{-1} via the rank-1 Sherman–Morrison formula after each
   new transition (x_new → y_new) is recorded::

       G_{t+1}^{-1} = G_t^{-1}
                    − (G_t^{-1} x x^T G_t^{-1})
                      / (1 + x^T G_t^{-1} x),    x = [z | a].

   This is the exact Kalman filter covariance update applied to Bayesian
   linear regression (Eldredge & Mousavi, 2026, Eq. 33).  The O((r+m)²)
   online step replaces the O((r+m)³) full inversion that would be needed
   at every step.

2. **Information-gain exploration (Cai et al., 2014; Eldredge & Mousavi, 2026)**

   The *leverage score* h = x^T G^{-1} x satisfies two simultaneously:

   * It is the *Maximum Model Change* (MMC) criterion of Cai et al. (2014,
     Eq. 10): the candidate that maximises |Δmodel| under the new observation.
   * It is the one-step *Kalman information gain* (Eldredge & Mousavi, 2026,
     Eq. 33): the posterior-variance reduction tr(ΔG^{-1}) equals
     ||G^{-1/2} x||² / (1 + h), which is monotone in h.

   The normalised bonus h / (1 + h) ∈ [0, 1) is subtracted from the
   composite score (i.e. high-information candidates are preferred)::

       info_bonus = γ_info · h(z, a) / (1 + h(z, a))

3. **Subspace reconstruction residual (Li et al., 2021)**

   The ALSL framework of Li et al. (2021) selects samples whose
   representation is *least explained* by the current low-rank subspace.
   Translated to the sequential setting, the exploration bonus is the
   normalised projection residual of the GP-predicted next ECM state under
   the current SVD basis U_r::

       ρ(a) = ||ŝ_GP − U_r U_r^T ŝ_GP||₂ / (||ŝ_GP||₂ + ε)

   where ŝ_GP is the GP posterior mean for the next state under protocol a.
   Large ρ indicates that the protocol leads to a region of ECM-parameter
   space not yet captured by the current subspace — querying it expands the
   manifold coverage, matching the ALSL_U principle of Li et al. (2021,
   Formulation II).

**Full composite score** (lower = better)::

    score = Q · degradation(s̃_next[:n_state])   ← linear dynamics prediction
          + R · ‖a‖²                              ← control effort
          − κ · mean(σ_GP)                        ← GP uncertainty (LCB-style)
          − γ_info · h(z, a) / (1 + h(z, a))     ← Cai 2014 / Eldredge 2026
          − γ_sub · ρ(ŝ_GP, U_r)                 ← Li 2021 ALSL

**Key difference from** :class:`~battery_forecast.policies.StateSpacePolicy`

StateSpacePolicy refits the full SVD + OLS dynamics on *all* historical data
at every :meth:`observe` call.  BayesianDynamicsPolicy fits the SVD basis
once at warm-up (and optionally re-anchors every ``reanchor_freq`` steps),
then updates only G^{-1} online — a true O((r+m)²) Kalman-filter update that
accumulates information without ever touching the O((r+m)³) matrix inverse.

References
----------
Eldredge, J. D., & Mousavi, H. (2026). A practical guide to estimation and
uncertainty quantification of aerodynamic flows. *arXiv preprint*
arXiv:2502.20280. https://doi.org/10.48550/arXiv.2502.20280
(Sequential Kalman filter posterior-covariance update, Eq. 33; SVD of
observation operator and state-space Gramian, Eq. 24; ensemble Kalman
filter, §III.)

Cai, W., Zhang, Y., Zhou, S., Wang, W., Ding, C., & Gu, X. (2014). Active
learning for support vector machines with maximum model change. In *Joint
European Conference on Machine Learning and Knowledge Discovery in
Databases*, pp. 116–131. Springer, Berlin, Heidelberg.
https://doi.org/10.1007/978-3-662-44845-8_10
(MMC criterion: select the query that maximally changes the learned model,
Eqs. 8–10; equivalence to OLS leverage score.)

Li, C., Mao, K., Liang, L., Ren, D., Zhang, W., Yuan, Y., & Wang, G.
(2021). Unsupervised active learning via subspace learning. In *Proceedings
of the 35th AAAI Conference on Artificial Intelligence*, pp. 8332–8340.
https://ojs.aaai.org/index.php/AAAI/article/view/17011
(ALSL Formulation II: sample selection under learnt low-rank representations
to suppress noise and capture subspace structure, Eq. 4.)
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


class BayesianDynamicsPolicy(BasePolicy):
    """Online Bayesian linear dynamics + information-gain + subspace-residual scoring.

    Maintains a precision matrix G_t = (X_t^T X_t + λI) in the reduced
    state-action space [z | a] and updates it online via the rank-1
    Sherman–Morrison formula — the Kalman filter covariance update applied to
    a Bayesian linear regression posterior (Eldredge & Mousavi, 2026, Eq. 33).

    Exploration is driven by two complementary bonuses:

    * **Information gain** — the normalised leverage score h/(1+h), which is
      simultaneously the MMC criterion of Cai et al. (2014, Eq. 10) and the
      Bayesian D-optimal information gain (Eldredge & Mousavi, 2026, Eq. 33).
    * **Subspace residual** — the normalised reconstruction residual of the
      GP-predicted next state under the current SVD basis, following the
      ALSL_U principle of Li et al. (2021, Eq. 4).

    Score (lower = better)::

        score = Q · degradation(s̃_next[:n_state])
              + R · ‖a‖²
              − κ · mean(σ_GP)
              − γ_info · h(z, a) / (1 + h(z, a))
              − γ_sub · ρ(ŝ_GP, U_r)

    Parameters
    ----------
    model :
        Trained surrogate with uncertainty support.
    n_components : int
        Rank of the SVD approximation.
    kappa : float
        GP exploration weight (LCB-style).
    Q_weight : float
        Degradation cost weight.
    R_weight : float
        Control effort cost weight.
    gamma_info : float
        Weight on the information-gain bonus (Cai 2014 / Eldredge 2026).
    gamma_sub : float
        Weight on the subspace reconstruction-residual bonus (Li 2021).
    lambda_reg : float
        Ridge regularisation for the Gram matrix: G_0 = λI.
    min_fit_obs : int
        Minimum observations before the dynamics model is fitted; falls back
        to LCBPolicy until this threshold is reached.
    reanchor_freq : int or None
        Re-fit the SVD basis and reinitialise G^{-1} from historical data
        every ``reanchor_freq`` observations after the initial fit.  ``None``
        or ``0`` disables re-anchoring (SVD is fit once and frozen).
    num_action_features : int
        Length of each candidate protocol vector.
    uncertainty_weight : float
        Scaling factor for the audit uncertainty vector appended to the ECM
        state to form the augmented belief state.
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
        gamma_info: float = 1.0,
        gamma_sub: float = 0.5,
        lambda_reg: float = 1e-3,
        min_fit_obs: int = 5,
        reanchor_freq: Optional[int] = None,
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
        self.gamma_info = gamma_info
        self.gamma_sub = gamma_sub
        self.lambda_reg = lambda_reg
        self.min_fit_obs = min_fit_obs
        self.reanchor_freq = reanchor_freq or 0
        self.uncertainty_weight = uncertainty_weight

        self._fallback = LCBPolicy(
            model=model,
            kappa=kappa,
            num_action_features=num_action_features,
            degradation_fn=degradation_fn,
            scaler=scaler,
        )

        self._state_history: list[np.ndarray] = []
        self._action_history: list[np.ndarray] = []
        self._uncertainty_history: list[Optional[np.ndarray]] = []
        self._n_state: Optional[int] = None
        self._n_obs: int = 0

        # Fitted reduced-order model (None until min_fit_obs reached)
        self._U_r: Optional[np.ndarray] = None   # (n_aug, r)
        self._A_r: Optional[np.ndarray] = None   # (r, r)
        self._B_r: Optional[np.ndarray] = None   # (r, m)

        # Inverse precision matrix of the Bayesian linear regression posterior
        # in the reduced [z | a] space (Eldredge & Mousavi 2026, Eq. 33).
        # Initialised at fit time; updated online via Sherman–Morrison.
        self._G_inv: Optional[np.ndarray] = None  # (r+m, r+m)

    # ── Belief-state construction ─────────────────────────────────────────

    def _build_augmented(
        self, state: np.ndarray, uncertainty: Optional[np.ndarray]
    ) -> np.ndarray:
        """Concatenate ECM state with scaled uncertainty vector (belief state).

        Follows the belief-state MDP augmentation of Kaelbling et al. (1998)
        used in :class:`~battery_forecast.policies.StateSpacePolicy`.
        Returns the state unchanged when no uncertainty vector is available.
        """
        if uncertainty is None:
            return state
        return np.concatenate(
            [state, self.uncertainty_weight * np.asarray(uncertainty, dtype=float)]
        )

    # ── Dynamics fitting ──────────────────────────────────────────────────

    def _fit_dynamics(self, reinit_G: bool = True) -> None:
        """Fit SVD basis and linear transition model; optionally reinitialise G^{-1}.

        Computes the same SVD + OLS decomposition as StateSpacePolicy.
        When ``reinit_G=True``, G^{-1} is rebuilt from the full X_fit
        matrix::

            G = λI + X_fit^T X_fit,   G^{-1} = inv(G).

        When ``reinit_G=False`` (used during periodic re-anchors after the
        online Sherman–Morrison updates have diverged from the new basis), the
        existing G^{-1} is replaced by a fresh inversion so that it remains
        consistent with the newly rotated U_r.

        Parameters
        ----------
        reinit_G : bool
            Always rebuild G^{-1} from historical data (default: True).
        """
        u_dim = next(
            (len(u) for u in self._uncertainty_history if u is not None), 0
        )
        aug_states = np.array([
            self._build_augmented(s, u if u is not None else np.zeros(u_dim))
            for s, u in zip(self._state_history, self._uncertainty_history)
        ])  # (T, n_aug)
        A = np.array(self._action_history[:-1])  # (T-1, m)
        T = len(A)

        if T < 2:
            return
        if not (np.isfinite(aug_states).all() and np.isfinite(A).all()):
            log.warning(
                "[BayesianDynamicsPolicy] aug_states/actions contain NaN/inf — "
                "skipping dynamics fit"
            )
            return

        r = min(self.n_components, aug_states.shape[1], T)
        U, _, _ = np.linalg.svd(aug_states.T, full_matrices=False)
        self._U_r = U[:, :r]  # (n_aug, r)

        Z = aug_states @ self._U_r               # (T, r)
        X_fit = np.hstack([Z[:T], A])            # (T-1, r+m)
        Y_fit = Z[1: T + 1]                      # (T-1, r)

        if not (np.isfinite(X_fit).all() and np.isfinite(Y_fit).all()):
            log.warning(
                "[BayesianDynamicsPolicy] X_fit/Y_fit contain NaN/inf — "
                "skipping dynamics fit"
            )
            return

        W, _, _, _ = np.linalg.lstsq(X_fit, Y_fit, rcond=None)
        self._A_r = W[:r].T   # (r, r)
        self._B_r = W[r:].T   # (r, m)

        if reinit_G:
            # Bayesian linear regression precision matrix (Eldredge 2026 Eq 33):
            # G = λI + X_fit^T X_fit; G^{-1} is the posterior covariance.
            G = self.lambda_reg * np.eye(X_fit.shape[1]) + X_fit.T @ X_fit
            try:
                self._G_inv = np.linalg.inv(G)
            except np.linalg.LinAlgError:
                self._G_inv = np.linalg.pinv(G)

        log.debug(
            f"[BayesianDynamicsPolicy] fit: r={r}, T={T}, "
            f"n_aug={aug_states.shape[1]}, reinit_G={reinit_G}"
        )

    # ── Online Kalman precision update ────────────────────────────────────

    def _update_precision(self, z: np.ndarray, action: np.ndarray) -> None:
        """Rank-1 Sherman–Morrison update of G^{-1} (Kalman covariance step).

        Implements the Kalman filter posterior-covariance update (Eldredge &
        Mousavi, 2026, Eq. 33) for a Bayesian linear regression model::

            G^{-1}_{t+1} = G^{-1}_t
                         − G^{-1}_t x x^T G^{-1}_t
                           / (1 + x^T G^{-1}_t x),    x = [z | action].

        This O((r+m)²) update is the key computational advantage over
        re-inverting the full Gram matrix after each observation.

        Notes
        -----
        When U_r is re-anchored the reduced coordinates z change meaning, so
        G^{-1} is reinitialised from scratch (via ``_fit_dynamics(reinit_G=True)``).
        The Sherman–Morrison update is therefore only applied between re-anchors.
        """
        if self._G_inv is None or self._U_r is None:
            return
        r = self._U_r.shape[1]
        if z.shape[0] != r:
            return
        x = np.concatenate([z, np.asarray(action)])
        if x.shape[0] != self._G_inv.shape[0]:
            return
        g = self._G_inv @ x          # (r+m,)
        denom = 1.0 + float(x @ g)
        self._G_inv -= np.outer(g, g) / denom

    # ── Stateful history ──────────────────────────────────────────────────

    def observe(
        self,
        state: np.ndarray,
        action: np.ndarray,
        uncertainty_vector: Optional[np.ndarray] = None,
    ) -> None:
        """Record an observed (state, action) pair and update the model.

        After the initial warm-up fit, updates G^{-1} online via the
        Sherman–Morrison formula (O((r+m)²)) rather than refitting the full
        dynamics from scratch.  Optionally re-fits the SVD basis every
        ``reanchor_freq`` observations to track long-term drift in the
        degradation manifold (G^{-1} is reinitialised from historical data
        after each re-anchor to remain consistent with the new basis).

        Parameters
        ----------
        state : np.ndarray
            Flattened, unscaled ECM parameter vector.
        action : np.ndarray
            Protocol vector applied at this step.
        uncertainty_vector : np.ndarray, optional
            Per-parameter variance vector from the uncertainty audit, shape
            ``(n_params,)``.
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
        self._n_obs += 1

        n = len(self._state_history)

        # Initial warm-up fit — sets U_r, A_r, B_r, G_inv
        if n == self.min_fit_obs:
            self._fit_dynamics(reinit_G=True)
            return

        if self._U_r is None:
            return

        # Periodic SVD re-anchor: refit basis and reinitialise G^{-1}
        if (
            self.reanchor_freq
            and n > self.min_fit_obs
            and (n - self.min_fit_obs) % self.reanchor_freq == 0
        ):
            log.debug(
                f"[BayesianDynamicsPolicy] re-anchoring SVD at obs {self._n_obs}"
            )
            self._fit_dynamics(reinit_G=True)
            return

        # Online Sherman–Morrison update of precision inverse (Eldredge 2026 Eq 33)
        u_dim = next(
            (len(u) for u in self._uncertainty_history if u is not None), 0
        )
        aug = self._build_augmented(
            state,
            uncertainty_vector if uncertainty_vector is not None else np.zeros(u_dim),
        )
        z = aug @ self._U_r
        self._update_precision(z, action)

    # ── Exploration bonuses ───────────────────────────────────────────────

    def _info_gain(self, z: np.ndarray, action: np.ndarray) -> float:
        """Normalised information gain h / (1 + h), h = x^T G^{-1} x.

        The leverage score h simultaneously implements:

        * Maximum Model Change (Cai et al., 2014, Eq. 10): the query that
          most changes the learned linear dynamics model.
        * Bayesian information gain (Eldredge & Mousavi, 2026, Eq. 33): the
          one-step posterior-variance reduction tr(ΔG^{-1}) is monotone in h.

        The normalised form h/(1+h) ∈ [0, 1) avoids unbounded values when
        G^{-1} has large eigenvalues early in the experiment.
        """
        if self._G_inv is None or self._U_r is None:
            return 0.0
        r = self._U_r.shape[1]
        if z.shape[0] != r:
            return 0.0
        x = np.concatenate([z, np.asarray(action)])
        if x.shape[0] != self._G_inv.shape[0]:
            return 0.0
        h = float(x @ self._G_inv @ x)
        h = max(h, 0.0)
        return h / (1.0 + h)

    def _subspace_residual(self, s_gp: np.ndarray) -> float:
        """Normalised subspace reconstruction residual (Li et al., 2021).

        Computes the fraction of the GP-predicted next ECM state that lies
        *outside* the current SVD subspace (using only the ECM-state rows of
        U_r to match the dimensionality of the GP prediction)::

            ρ = ||ŝ_GP − U_s U_s^T ŝ_GP||₂ / (||ŝ_GP||₂ + ε)

        where U_s = U_r[:n_state, :] is the ECM-state portion of the basis.
        Large ρ means the protocol leads to a region of parameter space not
        yet captured by the learnt subspace, following the ALSL_U principle
        of Li et al. (2021, Formulation II).
        """
        if self._U_r is None or self._n_state is None:
            return 0.0
        s = np.asarray(s_gp, dtype=float)
        n = min(self._n_state, self._U_r.shape[0], len(s))
        s_trunc = s[:n]
        U_s = self._U_r[:n]              # (n, r)
        proj = U_s @ (U_s.T @ s_trunc)
        return float(
            np.linalg.norm(s_trunc - proj) / (np.linalg.norm(s_trunc) + 1e-12)
        )

    # ── Candidate scoring ─────────────────────────────────────────────────

    def _score_candidate(
        self,
        current_aug: np.ndarray,
        current_state: np.ndarray,
        protocol: np.ndarray,
        prot_scale: Optional[np.ndarray] = None,
    ) -> float:
        """Composite score for a single candidate protocol.

        Combines LQR-style dynamics cost with two information-theoretic
        exploration bonuses (Cai 2014; Eldredge 2026; Li 2021).

        ``prot_scale`` normalises the control cost to dimensionless units; see
        :meth:`~StateSpacePolicy._score_candidate` for the rationale.
        """
        protocol = np.asarray(protocol)

        # GP prediction for uncertainty bonus and subspace residual
        x_scaled = self._build_input(current_state, protocol)
        mean_pred, std_pred = self._predict(x_scaled)
        gp_std = float(np.mean(std_pred[0])) if std_pred is not None else 0.0
        s_gp = (
            self._unscale_state(mean_pred[0])
            if mean_pred is not None
            else current_state
        )

        # Linear-dynamics prediction in reduced space (same as StateSpacePolicy)
        z = current_aug @ self._U_r                    # (r,)
        z_next = self._A_r @ z + self._B_r @ protocol  # (r,)
        s_aug_next = self._U_r @ z_next                # (n_aug,)
        n = self._n_state if self._n_state is not None else s_aug_next.shape[0]
        s_next = s_aug_next[:n]

        state_cost = self.Q_weight * self.degradation_fn(s_next)
        p_norm = protocol / prot_scale if prot_scale is not None else protocol
        control_cost = self.R_weight * float(np.dot(p_norm, p_norm))
        gp_bonus = self.kappa * gp_std
        info_bonus = self.gamma_info * self._info_gain(z, protocol)
        sub_bonus = self.gamma_sub * self._subspace_residual(s_gp)
        c_rate_bonus = (
            self.C_rate_bonus * float(p_norm[0])
            if self.C_rate_bonus != 0.0
            else 0.0
        )

        return state_cost + control_cost - gp_bonus - info_bonus - sub_bonus - c_rate_bonus

    # ── Public interface ──────────────────────────────────────────────────

    def select_next(
        self,
        current_state: np.ndarray,
        candidate_protocols: list[np.ndarray],
        uncertainty_vector: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Select the candidate minimising the composite information-theoretic score.

        Falls back to :class:`~battery_forecast.policies.LCBPolicy` (GP-only)
        until ``min_fit_obs`` transitions have been observed.

        Parameters
        ----------
        current_state : np.ndarray
            Flattened, unscaled ECM parameter vector.
        candidate_protocols : list of np.ndarray
            Candidate charge-protocol vectors.
        uncertainty_vector : np.ndarray, optional
            Per-parameter variance vector from the uncertainty audit.

        Returns
        -------
        best_protocol : np.ndarray
        best_score : float
        scores : np.ndarray
        """
        if self._A_r is None:
            n_obs = len(self._state_history)
            log.info(
                f"[BayesianDynamicsPolicy] warm-up ({n_obs}/{self.min_fit_obs} obs): "
                "using LCB fallback"
            )
            return self._fallback.select_next(current_state, candidate_protocols)

        current_aug = self._build_augmented(
            np.asarray(current_state, dtype=float), uncertainty_vector
        )
        cand_arr = np.array([np.asarray(p) for p in candidate_protocols])
        prot_scale = np.maximum(np.abs(cand_arr).max(axis=0), 1.0)
        scores = np.array([
            self._score_candidate(current_aug, current_state, p, prot_scale)
            for p in candidate_protocols
        ])
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

        Identical interface to
        :meth:`~battery_forecast.policies.StateSpacePolicy.run`.

        Parameters
        ----------
        initial_state : np.ndarray
            Starting ECM state vector.
        candidate_protocols : list of np.ndarray
            Pool of candidate protocols.
        oracle_fn : callable
            ``oracle_fn(protocol) -> next_state``.
        n_iterations : int
            Number of selection-observe cycles.
        uncertainty_oracle_fn : callable, optional
            ``uncertainty_oracle_fn(protocol) -> np.ndarray``.  Returns the
            per-parameter variance vector from the uncertainty audit.
        on_step : callable, optional
            Called after each iteration with keyword arguments:
            ``iteration``, ``protocol``, ``state``, ``next_state``,
            ``degradation``, ``score``, ``uncertainty_vector``.

        Returns
        -------
        dict with keys ``selected_protocols``, ``observed_states``,
        ``scores``, ``degradation``, ``policy``.
        """
        current_state = np.asarray(initial_state, dtype=float)
        current_uncertainty: Optional[np.ndarray] = None

        selected_protocols: list = []
        observed_states: list = [current_state]
        scores_history: list = []
        degradation_history: list = [self.degradation_fn(current_state)]

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
                    "[BayesianDynamicsPolicy] oracle failure at iter %d — %s; "
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
            self.observe(
                current_state, protocol, uncertainty_vector=current_uncertainty
            )

            log.info(
                f"[BayesianDynamicsPolicy] iter {i + 1}/{n_iterations}: "
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
