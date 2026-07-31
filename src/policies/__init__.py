"""Battery charge-rate active learning policies.

Each policy inherits from :class:`BasePolicy` and implements
:meth:`~BasePolicy.select_next`.  The shared :meth:`~BasePolicy.run` loop
and helper methods (scaling, GP prediction) live on the base class.

Policies
--------
RandomPolicy
    Uniform-random baseline.  Selects a candidate protocol uniformly at
    random — no surrogate model required.
    *Settles (2009)*

GreedyPolicy
    Pure exploitation.  Selects the candidate with the lowest predicted
    degradation; model uncertainty is ignored.
    *Brochu, Cora & de Freitas (2010)*

LCBPolicy
    Lower Confidence Bound.  Balances exploitation of low predicted
    degradation with exploration of high-uncertainty regions via
    ``score = degradation(μ) − κ · mean(σ)``.
    *Srinivas, Krause, Kakade & Seeger (2010)*

StateSpacePolicy
    SVD + linear dynamics + LQR-style cost.  Learns a reduced-order linear
    dynamical model from state-action history (inspired by Dynamic Mode
    Decomposition and autopilot LQR design) and scores candidates via a
    composite cost that incorporates predicted dynamics, control effort, and
    GP uncertainty.  Falls back to LCBPolicy during warm-up.
    *Brunton & Kutz (2022); Deisenroth & Rasmussen (2011)*

BayesianDynamicsPolicy
    Online Bayesian linear dynamics + information-gain + subspace-residual
    scoring.  Fits the SVD basis once at warm-up, then maintains the
    Bayesian regression precision matrix G^{-1} = (X^T X + λI)^{-1} via
    online rank-1 Sherman–Morrison updates (the Kalman filter covariance
    step).  Candidates are scored by a composite function that adds two
    information-theoretic exploration bonuses to the LQR-style cost of
    StateSpacePolicy:

    * **Information gain** h/(1+h) where h = x^T G^{-1} x is the leverage
      score, simultaneously the Maximum Model Change criterion of Cai et al.
      and the Bayesian D-optimal gain of Eldredge & Mousavi.
    * **Subspace residual** — normalised reconstruction residual of the
      GP-predicted next state under the current SVD basis, following the
      ALSL_U principle of Li et al.

    Falls back to LCBPolicy during warm-up.
    *Cai et al. (2014); Li et al. (2021); Eldredge & Mousavi (2026)*

Usage
-----
::

    from battery_forecast.policies import LCBPolicy, StateSpacePolicy
    from battery_forecast.policies import BayesianDynamicsPolicy
    from battery_forecast.policies.base import resistance_degradation

    policy = LCBPolicy(model, kappa=2.0, scaler=scaler,
                       degradation_fn=resistance_degradation)
    protocol, score, _ = policy.select_next(current_state, candidates)
    history = policy.run(initial_state, candidates, oracle_fn, n_iterations=20)

    policy = BayesianDynamicsPolicy(model, gamma_info=1.0, gamma_sub=0.5,
                                    scaler=scaler,
                                    degradation_fn=resistance_degradation)
    history = policy.run(initial_state, candidates, oracle_fn, n_iterations=20)
"""

from .base import BasePolicy, _default_degradation, resistance_degradation
from .random import RandomPolicy
from .greedy import GreedyPolicy
from .constant import ConstantPolicy
from .lcb import LCBPolicy
from .state_space import StateSpacePolicy
from .state_space_bayes import BayesianDynamicsPolicy
from .qbc import QBCPolicy
from .state_space_audit import (
    AuditStateSpacePolicy,
    AuditBayesianDynamicsPolicy,
    make_audit_health_vector,
    drt_peak_mismatch_result,
    AUDIT_HEALTH_FIELDS,
)

__all__ = [
    "BasePolicy",
    "ConstantPolicy",
    "RandomPolicy",
    "GreedyPolicy",
    "LCBPolicy",
    "StateSpacePolicy",
    "BayesianDynamicsPolicy",
    "QBCPolicy",
    "AuditStateSpacePolicy",
    "AuditBayesianDynamicsPolicy",
    "make_audit_health_vector",
    "drt_peak_mismatch_result",
    "AUDIT_HEALTH_FIELDS",
    "resistance_degradation",
]
