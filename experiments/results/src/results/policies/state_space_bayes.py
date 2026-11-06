"""Bayesian dynamics policy with online precision updates and information-gain scoring.

Extends ``results.policies.StateSpacePolicy`` in three ways,
each grounded in the accompanying literature:

1. **Online Bayesian precision update (Eldredge & Mousavi, 2026)**

   Rather than refitting the least-squares dynamics model from scratch at
   every :meth:`observe` call, this policy tracks the *precision matrix* of
   the Bayesian linear regression posterior in the reduced state-action space,

       G_t = X_t^T X_t + λI  ∈ ℝ^{(r+m) × (r+m)},

   and updates G_t^{-1} via the rank-1 Sherman–Morrison formula as soon as an
   action is chosen (the precision does not depend on the outcome); the mean
   is updated when the outcome arrives::

       G_{t+1}^{-1} = G_t^{-1}
                    − (G_t^{-1} x x^T G_t^{-1})
                      / (1 + x^T G_t^{-1} x),    x = [z | a].

   This is the exact Kalman filter covariance update applied to Bayesian
   linear regression (Eldredge & Mousavi, 2026, Eq. 33).  The O((r+m)²)
   online step replaces the O((r+m)³) full inversion that would be needed
   at every step.

2. **Information-gain exploration (Cai et al., 2014; Eldredge & Mousavi, 2026)**

   The *leverage score* h = x^T G^{-1} x is the predictive variance of the
   candidate in units of the noise variance σ² (the posterior covariance of
   each output is σ² G^{-1}).  Its log is the one-step D-optimal *Kalman
   information gain* (Eldredge & Mousavi, 2026, Eq. 33).  It is related to,
   but not the same as, the *Maximum Model Change* criterion of Cai et al.
   (2014): the expected model change is ||G^{-1} x||² / (1 + h), which is not
   monotone in h.

   The bonus is the D-optimal gain log(1 + h) = log det G_{t+1} − log det G_t
   (the matrix determinant lemma), subtracted from the composite score (i.e.
   high-information candidates are preferred)::

       info_bonus = γ_info · log(1 + h(z, a))

   Unlike a bounded transform such as h / (1 + h), log(1 + h) keeps ranking
   candidates while the prior is still vague and h is large for all of them.

3. **Subspace reconstruction residual (Li et al., 2021)**

   The ALSL framework of Li et al. (2021) selects samples whose
   representation is *least explained* by the current low-rank subspace.
   Translated to the sequential setting, the exploration bonus is the
   normalised projection residual of the GP-predicted next ECM state under
   the current SVD basis U_r::

       ρ(a) = ||ŝ_GP − P_s ŝ_GP||₂ / (||ŝ_GP||₂ + ε)

   where ŝ_GP is the GP posterior mean for the next state under protocol a
   and P_s the orthogonal projector onto the span of the state rows of U_r.
   Large ρ indicates that the protocol leads to a region of ECM-parameter
   space not yet captured by the current subspace — querying it expands the
   manifold coverage, matching the ALSL_U principle of Li et al. (2021,
   Formulation II).

**Full composite score** (lower = better)::

    score = Q · degradation(s̃_next[:n_state])   ← linear dynamics prediction
          + R · ‖a‖²                              ← control effort
          − κ · mean(σ_GP)                        ← GP uncertainty (LCB-style)
          − γ_info · log(1 + h(z, a))            ← Cai 2014 / Eldredge 2026
          − γ_sub · ρ(ŝ_GP, U_r)                 ← Li 2021 ALSL

**Key difference from** ``results.policies.StateSpacePolicy``

StateSpacePolicy refits the full SVD + OLS dynamics on *all* historical data
at every :meth:`observe` call.  BayesianDynamicsPolicy fits the SVD basis
once at warm-up (and optionally re-anchors every ``reanchor_freq`` steps),
then updates the Bayesian posterior of [A_r | B_r] online — the O((r+m)²)
Kalman-filter update of both the posterior mean and the precision inverse
G^{-1}, without re-inverting the O((r+m)³) Gram matrix (except in ``window``
mode, which refits the posterior on the window at every :meth:`observe`).

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
import time
from typing import Callable, Optional

import numpy as np

from . import base as _base
from .base import BasePolicy
from .lcb import LCBPolicy

log = logging.getLogger(__name__)


class BayesianDynamicsPolicy(BasePolicy):
    """Online Bayesian linear dynamics + information-gain + subspace-residual scoring.

    Maintains the Bayesian linear regression posterior of the reduced
    dynamics W = [A_r | B_r]^T in the state-action space [z | a]: precision
    G_t = (X_t^T X_t + λI) and posterior mean W_t = G_t^{-1} X_t^T Y_t.  Both
    are updated online by the Kalman filter for this model (Eldredge &
    Mousavi, 2026, Eq. 33): G^{-1} by the rank-1 Sherman–Morrison formula as
    soon as an action is chosen (the precision does not depend on the
    outcome), and W by the Kalman gain once the outcome of that action, the
    next state, is observed.

    Exploration is driven by two complementary bonuses:

    * **Information gain** — log(1 + h) of the leverage score h, the Bayesian
      D-optimal information gain (Eldredge & Mousavi, 2026, Eq. 33) (related to, but
      not the same as, the MMC criterion of Cai et al. 2014).
    * **Subspace residual** — the normalised reconstruction residual of the
      GP-predicted next state under the current SVD basis, following the
      ALSL_U principle of Li et al. (2021, Eq. 4).

    Score (lower = better)::

        score = Q · degradation(s̃_next[:n_state])
              + R · ‖a‖²
              − κ · mean(σ_GP)
              − γ_info · log(1 + h(z, a))
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
        Re-fit the SVD basis, and the posterior in the new basis, every
        ``reanchor_freq`` observations after the initial fit.  ``None`` or
        ``0`` disables re-anchoring: the basis is fit once and frozen, while
        the posterior of [A_r | B_r] keeps learning online.
    window : int or None
        If set, the posterior is recomputed after every observation from only
        the last ``window`` transitions (in the current basis) instead of
        being updated recursively over the whole history.
    num_action_features : int
        Length of each candidate protocol vector.
    uncertainty_weight : float
        Scaling factor for the audit uncertainty vector appended to the ECM
        state to form the augmented belief state.
    degradation_fn : callable, optional
        ``f(state: np.ndarray) -> float``.  Higher = more degraded.
    scaler : sklearn-compatible scaler, optional
    """

    # Fill for a step with no uncertainty vector; None means zeros (see _missing_uncertainty).
    neutral_uncertainty: Optional[np.ndarray] = None

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
        window: Optional[int] = None,
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
        # Lookback window for _fit_dynamics -- see StateSpacePolicy.window for
        # the full rationale. None (default) refits on the entire accumulated
        # history, as before.
        self.window = window

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

        # Bayesian linear regression posterior in the reduced [z | a] space
        # (Eldredge & Mousavi 2026, Eq. 33): posterior mean W = [A_r | B_r]^T
        # and inverse precision G^{-1}.  Initialised at fit time; updated
        # online by the Kalman filter.
        self._W: Optional[np.ndarray] = None      # (r+m, r)
        self._G_inv: Optional[np.ndarray] = None  # (r+m, r+m)
        # Design point [z_t | a_t] of the latest action: already in G^{-1},
        # waiting for its outcome z_{t+1} to update W.
        self._pending_x: Optional[np.ndarray] = None

    # ── Belief-state construction ─────────────────────────────────────────

    def _build_augmented(
        self, state: np.ndarray, uncertainty: Optional[np.ndarray]
    ) -> np.ndarray:
        """Concatenate ECM state with scaled uncertainty vector (belief state).

        Follows the belief-state MDP augmentation of Kaelbling et al. (1998)
        used in ``results.policies.StateSpacePolicy``.
        Returns the state unchanged when no uncertainty vector is available.
        A step with no vector (``None``) uses :meth:`_missing_uncertainty` once
        vectors have been seen, so every augmented state has the same length.
        """
        if uncertainty is None:
            uncertainty = self._missing_uncertainty()
        if uncertainty is None:
            return state
        return np.concatenate(
            [state, self.uncertainty_weight * np.asarray(uncertainty, dtype=float)]
        )

    def _missing_uncertainty(self) -> Optional[np.ndarray]:
        """Stand-in for a step whose uncertainty vector is missing (None if none seen yet).

        The policy's ``neutral_uncertainty`` when it has the right length (the
        audit policies set it to ``NEUTRAL_AUDIT_HEALTH``), else zeros, which is
        neutral for z-scored vectors.
        """
        u_dim = next((len(u) for u in self._uncertainty_history if u is not None), 0)
        if u_dim == 0:
            return None
        if self.neutral_uncertainty is not None and len(self.neutral_uncertainty) == u_dim:
            return np.array(self.neutral_uncertainty, dtype=float)
        return np.zeros(u_dim)

    # ── Dynamics fitting ──────────────────────────────────────────────────

    def _augmented_history(self) -> tuple[np.ndarray, np.ndarray]:
        """Augmented states and the actions between them, window-limited.

        Returns ``(aug_states, actions)`` of shapes ``(T, n_aug)`` and
        ``(T-1, m)``: ``actions[i]`` took ``aug_states[i]`` to
        ``aug_states[i + 1]``.  When ``self.window`` is set, only the last
        ``window + 1`` states (``window`` transitions) are used.
        """
        state_hist = self._state_history
        uncertainty_hist = self._uncertainty_history
        action_hist = self._action_history
        if self.window is not None:
            state_hist = state_hist[-(self.window + 1):]
            uncertainty_hist = uncertainty_hist[-(self.window + 1):]
            action_hist = action_hist[-(self.window + 1):]

        aug_states = np.array([
            self._build_augmented(s, u)
            for s, u in zip(state_hist, uncertainty_hist, strict=True)
        ])  # (T, n_aug)
        actions = np.array(action_hist[:-1])  # (T-1, m)
        return aug_states, actions

    def _fit_dynamics(self) -> None:
        """Fit the SVD basis, then the dynamics posterior in that basis.

        The basis ``U_r`` comes from the SVD of the augmented states, as in
        StateSpacePolicy.  The posterior of ``W = [A_r | B_r]^T`` is then
        computed in the new basis by :meth:`_fit_posterior`; a re-anchor
        changes the meaning of ``z``, so the posterior is rebuilt rather than
        carried over.
        """
        aug_states, actions = self._augmented_history()
        T = len(actions)
        if T < 2:
            return
        if not (np.isfinite(aug_states).all() and np.isfinite(actions).all()):
            log.warning(
                "[BayesianDynamicsPolicy] aug_states/actions contain NaN/inf — "
                "skipping dynamics fit"
            )
            return

        r = min(self.n_components, aug_states.shape[1], T)
        U, _, _ = np.linalg.svd(aug_states.T, full_matrices=False)
        self._U_r = U[:, :r]  # (n_aug, r)
        self._fit_posterior()

        log.debug(
            f"[BayesianDynamicsPolicy] fit: r={r}, T={T}, n_aug={aug_states.shape[1]}"
        )

    def _fit_posterior(self) -> None:
        """Batch Bayesian posterior of ``W = [A_r | B_r]^T`` in the current basis.

        With prior ``vec(W) ~ N(0, τ²I)`` and noise variance ``σ²``
        (``λ = σ²/τ²``), the posterior over the observed transitions has::

            G = λI + X_fit^T X_fit,     W = G^{-1} X_fit^T Y_fit,

        with ``X_fit = [Z_{0:T-2} | actions]`` and ``Y_fit = Z_{1:T-1}``.  The
        prior keeps the posterior defined even before there are as many
        transitions as unknowns.  Afterwards the design point of the latest
        action is added to ``G^{-1}``; its outcome arrives with the next
        observation.
        """
        aug_states, actions = self._augmented_history()
        if len(actions) < 1:
            return
        if not (np.isfinite(aug_states).all() and np.isfinite(actions).all()):
            log.warning(
                "[BayesianDynamicsPolicy] aug_states/actions contain NaN/inf — "
                "skipping posterior fit"
            )
            return

        Z = aug_states @ self._U_r                 # (T, r)
        X_fit = np.hstack([Z[:-1], actions])       # (T-1, r+m)
        Y_fit = Z[1:]                              # (T-1, r)
        G = self.lambda_reg * np.eye(X_fit.shape[1]) + X_fit.T @ X_fit
        try:
            self._G_inv = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            self._G_inv = np.linalg.pinv(G)
        self._set_mean(self._G_inv @ X_fit.T @ Y_fit)

        self._pending_x = None
        self._add_design_point(Z[-1], self._action_history[-1])

    def _set_mean(self, W: np.ndarray) -> None:
        r = self._U_r.shape[1]
        self._W = W
        self._A_r = W[:r].T   # (r, r)
        self._B_r = W[r:].T   # (r, m)

    # ── Online Kalman update ──────────────────────────────────────────────

    def _add_design_point(self, z: np.ndarray, action: np.ndarray) -> None:
        """Kalman covariance step for the action just chosen from state ``z``.

        Sherman–Morrison rank-1 update (Eldredge & Mousavi, 2026, Eq. 33)::

            G^{-1} ← G^{-1} − G^{-1} x x^T G^{-1} / (1 + x^T G^{-1} x),
            x = [z | action].

        The posterior precision does not depend on the outcome, so it is
        updated as soon as the action is chosen; ``x`` is kept until its
        outcome arrives (:meth:`_learn_from_outcome`).
        """
        x = np.concatenate([np.asarray(z, dtype=float), np.asarray(action, dtype=float)])
        if not np.isfinite(x).all() or x.shape[0] != self._G_inv.shape[0]:
            self._pending_x = None
            return
        g = self._G_inv @ x          # (r+m,)
        self._G_inv -= np.outer(g, g) / (1.0 + float(x @ g))
        self._G_inv = 0.5 * (self._G_inv + self._G_inv.T)    # keep it symmetric against round-off
        self._pending_x = x

    def _learn_from_outcome(self, z_next: np.ndarray) -> None:
        """Kalman mean step once the outcome ``z_next`` of the pending action is known.

        With ``G^{-1}`` already including the pending design point ``x``, the
        Kalman gain is ``K = G^{-1} x`` and::

            W ← W + K (z_next − W^T x)^T.

        This is the recursive form of the batch posterior mean, so after any
        number of steps ``W`` equals ``G^{-1} X^T Y`` over all transitions seen.
        """
        if self._pending_x is None:
            return
        if not np.isfinite(z_next).all():
            # No outcome to learn from: take the pending point back out of the
            # precision (Sherman–Morrison with the opposite sign), so that
            # G^{-1} and W keep describing the same transitions.
            x = self._pending_x
            g = self._G_inv @ x
            self._G_inv += np.outer(g, g) / (1.0 - float(x @ g))
            self._G_inv = 0.5 * (self._G_inv + self._G_inv.T)
            self._pending_x = None
            return
        x = self._pending_x
        gain = self._G_inv @ x                                   # (r+m,)
        innovation = np.asarray(z_next, dtype=float) - self._W.T @ x   # (r,)
        self._set_mean(self._W + np.outer(gain, innovation))
        self._pending_x = None

    # ── Stateful history ──────────────────────────────────────────────────

    def observe(
        self,
        state: np.ndarray,
        action: np.ndarray,
        uncertainty_vector: Optional[np.ndarray] = None,
    ) -> None:
        """Record an observed (state, action) pair and update the model.

        ``state`` is the outcome of the previous action and the state from
        which ``action`` was chosen.  After the initial warm-up fit each call
        is one Kalman step in O((r+m)²): the posterior mean learns from the
        previous action's outcome, then the precision takes the new action's
        design point.  Optionally re-fits the SVD basis every
        ``reanchor_freq`` observations to track long-term drift in the
        degradation manifold (the posterior is rebuilt in the new basis).
        With ``window`` set, the posterior is instead recomputed from the last
        ``window`` transitions.

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

        # Initial warm-up fit — sets U_r and the posterior (W, G_inv); retried
        # on later observations if it could not be made at min_fit_obs
        if self._U_r is None and n >= self.min_fit_obs:
            self._fit_dynamics()
            return

        if self._U_r is None:
            return

        # Periodic SVD re-anchor: refit the basis and the posterior in it
        if (
            self.reanchor_freq
            and n > self.min_fit_obs
            and (n - self.min_fit_obs) % self.reanchor_freq == 0
        ):
            log.debug(
                f"[BayesianDynamicsPolicy] re-anchoring SVD at obs {self._n_obs}"
            )
            self._fit_dynamics()
            return

        if self.window is not None:
            self._fit_posterior()
            return

        # Online Kalman update (Eldredge 2026 Eq 33): mean, then precision
        aug = self._build_augmented(state, uncertainty_vector)
        z = aug @ self._U_r
        self._learn_from_outcome(z)
        self._add_design_point(z, action)

    # ── Exploration bonuses ───────────────────────────────────────────────

    def _info_gain(self, z: np.ndarray, action: np.ndarray) -> float:
        """D-optimal information gain log(1 + h), h = x^T G^{-1} x.

        h is the candidate's predictive variance in units of the noise variance.
        It is related to, but not the same as, the Maximum Model Change
        criterion of Cai et al. (2014), whose expected change ||G^{-1} x||² / (1 + h)
        is not monotone in h.

        log(1 + h) is the gain in log det G from observing x (matrix
        determinant lemma).  It grows without bound but slowly, so it keeps
        distinguishing candidates while G^{-1} still has large eigenvalues
        early in the experiment, where h / (1 + h) is close to 1 for all of
        them.
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
        return float(np.log1p(h))

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

        When the state is augmented, U_s is a block of rows of U_r and its
        columns are not orthonormal, so U_s U_s^T is not a projector.  The
        projection uses an orthonormal basis of the column space of U_s
        instead, so that ρ = 0 exactly for a state inside the subspace.
        """
        if self._U_r is None or self._n_state is None:
            return 0.0
        s = np.asarray(s_gp, dtype=float)
        n = min(self._n_state, self._U_r.shape[0], len(s))
        s_trunc = s[:n]
        U_s = self._U_r[:n]              # (n, r)
        basis, singular_values, _ = np.linalg.svd(U_s, full_matrices=False)
        basis = basis[:, singular_values > 1e-10]
        proj = basis @ (basis.T @ s_trunc)
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

        Falls back to ``results.policies.LCBPolicy`` (GP-only)
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
        ``results.policies.StateSpacePolicy.run``.

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
            ``degradation``, ``score``, ``uncertainty_vector``, plus (when
            ``self.model`` is available) ``y_true``, ``y_pred_mean``,
            ``y_pred_std``, ``uncertainty``, ``model``, ``input_features`` —
            the same prediction/uncertainty data :class:`BasePolicy` supplies,
            so traits-audit checks that need live predictions run for this
            policy too.

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
        degradation_history: list = []
        # Per-iteration wall-clock breakdown -- see BasePolicy.run's identical
        # instrumentation (TODO_ISSUES.md B5: H2 was untestable from any
        # artifact since nothing computed this anywhere).
        select_next_wall_time_s: list = []
        observe_wall_time_s: list = []
        oracle_wall_time_s: list = []

        n_cycles_completed = 0
        terminated_early   = False

        for i in range(n_iterations):
            _t0 = time.perf_counter()
            protocol, score, all_scores = self.select_next(
                current_state,
                candidate_protocols,
                uncertainty_vector=current_uncertainty,
            )
            select_next_wall_time_s.append(time.perf_counter() - _t0)

            _t_oracle0 = time.perf_counter()
            try:
                next_state = np.asarray(oracle_fn(protocol), dtype=float)
                succeeded = True
            except _base.ORACLE_FAILURES as exc:
                log.warning(
                    "[BayesianDynamicsPolicy] oracle failure at iter %d — %s; "
                    "ending loop.",
                    i + 1, exc,
                )
                next_state = np.full_like(current_state, np.nan)
                succeeded = False
            oracle_wall_time_s.append(time.perf_counter() - _t_oracle0)

            # Always record — killing protocol is diagnostic data.
            selected_protocols.append(protocol)
            observed_states.append(next_state)
            scores_history.append(all_scores)
            # Degradation of the observed state, as in BasePolicy.run and the
            # state cost the policy minimises: delta is a function of a state.
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
            _t_observe0 = time.perf_counter()
            self.observe(
                current_state, protocol, uncertainty_vector=current_uncertainty
            )
            observe_wall_time_s.append(time.perf_counter() - _t_observe0)

            log.info(
                f"[BayesianDynamicsPolicy] iter {i + 1}/{n_iterations}: "
                f"score={score:.4f}, degradation={degradation_history[-1]:.4f}"
            )

            if on_step is not None:
                step_kw: dict = {
                    "iteration":         i,
                    "protocol":          protocol,
                    "state":             current_state,
                    "next_state":        next_state,
                    "y_true":            next_state,
                    "degradation":       degradation_history[-1],
                    "score":             float(score),
                    "uncertainty_vector": current_uncertainty,
                }
                if self.model is not None:
                    try:
                        x_sc = self._build_input(current_state, protocol)
                        pred_mean_sc, pred_std_sc = self._predict(x_sc)
                        step_kw["y_pred_mean"] = self._unscale_state(pred_mean_sc[0])
                        if pred_std_sc is not None:
                            step_kw["y_pred_std"] = pred_std_sc[0]
                            step_kw["uncertainty"] = float(
                                np.mean(np.abs(pred_std_sc[0]))
                            )
                        step_kw["model"] = self.model
                        step_kw["input_features"] = x_sc
                    except Exception:
                        pass
                on_step(**step_kw)

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
            "select_next_wall_time_s": select_next_wall_time_s,
            "observe_wall_time_s":     observe_wall_time_s,
            "oracle_wall_time_s":      oracle_wall_time_s,
        }
