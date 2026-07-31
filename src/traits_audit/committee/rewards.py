"""Per-agent reward computers — one per non-Lyapunov, non-anti-exploratory audit check.

Each computer takes (y_before, mu_before, sigma_before) and the new observation
(y_t, mu_t, sigma_t), assembles the full y/mu/sigma arrays before vs after, and
calls the underlying AuditCheck.run() to produce cumulative-mean values. Reward
is delta * sign, where sign matches the check's improvement direction.

For checks whose "value" is a level rather than a loss (IntervalCoverage observed
fraction, VarianceAlignment ratio), the reward uses delta of distance-to-target.

The underlying check implementations are imported directly so the reward math
never diverges from the audit math. See:
- src/traits_audit/checks/scoring.py
- src/traits_audit/checks/calibration.py
- src/traits_audit/checks/conformal.py
- src/traits_audit/checks/coverage.py
- src/traits_audit/checks/pit.py
- src/traits_audit/checks/uncertainty.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from traits_audit.checks import (
    CalibrationError1StdCheck,
    CalibrationErrorCheck,
    ConformalCoverageCheck,
    CRPSCheck,
    ENCECheck,
    IntervalCoverageCheck,
    IntervalScoreCheck,
    KuleshovCalibrationCheck,
    MahalanobisOODCheck,
    NegativeLogLikelihoodCheck,
    PITUniformityCheck,
    UncertaintyAnomalyCheck,
    UncertaintyEvolutionCheck,
    VarianceAlignmentCheck,
    VarianceErrorCorrelationCheck,
)


@dataclass
class RewardComputer:
    """Base class for per-agent reward computation.

    Subclasses configure ``check`` (an AuditCheck instance), ``sign`` (+1 if
    higher check value is better, -1 if lower is better), and override
    ``_score`` if they need a non-default mapping from check.value to a scalar
    score (used when ``value`` is a level like coverage fraction or variance
    ratio rather than a loss).
    """

    name: str
    check: Any
    sign: float = -1.0

    def _score(self, value: Optional[float]) -> Optional[float]:
        """Map AuditResult.value to the scalar whose delta becomes reward."""
        return value

    def _run_check(
        self,
        y_true: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
    ) -> Optional[float]:
        """Run the underlying check on (y, mu, sigma) and return the scored value."""
        if len(y_true) == 0:
            return None
        result = self.check.run(
            history=[],
            y_true=y_true,
            y_pred_mean=mu,
            y_pred_std=sigma,
        )
        if result.value is None:
            return None
        return self._score(float(result.value))

    def reward(
        self,
        y_before: np.ndarray,
        mu_before: np.ndarray,
        sigma_before: np.ndarray,
        y_after: np.ndarray,
        mu_after: np.ndarray,
        sigma_after: np.ndarray,
        **extras: Any,
    ) -> float:
        """Compute the cumulative-mean delta reward.

        Both before/after triples represent full histories at step t and t+1.
        Returns 0.0 if either check returns None (e.g. n_samples below the
        check's floor); the running z-score normalizer downstream will leave
        these untouched.

        ``**extras`` is forwarded from ``env.step()`` so subclasses that need
        extra signals (query x's, uncertainty series, op_states) can read them
        without changing every existing subclass's signature. Base ignores.
        """
        v_before = self._run_check(y_before, mu_before, sigma_before)
        v_after = self._run_check(y_after, mu_after, sigma_after)
        if v_before is None or v_after is None:
            return 0.0
        return self.sign * (v_after - v_before)


# -- Proper scoring rules ---------------------------------------------------
# CRPS, NLL, IntervalScore: lower check.value is better -> sign = -1.

class CRPSReward(RewardComputer):
    def __init__(self):
        super().__init__(name="CRPS", check=CRPSCheck(), sign=-1.0)


class NLLReward(RewardComputer):
    def __init__(self):
        super().__init__(name="NLL", check=NegativeLogLikelihoodCheck(), sign=-1.0)


class IntervalScoreReward(RewardComputer):
    def __init__(self, alpha: float = 0.1):
        super().__init__(
            name="IntervalScore", check=IntervalScoreCheck(alpha=alpha), sign=-1.0
        )


# -- Calibration / coverage / conformal ------------------------------------
# Lower is better for CalibrationError and ConformalCoverage's q_ratio.

class CalibrationErrorReward(RewardComputer):
    def __init__(self, n_bins: int = 10):
        super().__init__(
            name="CalibrationError",
            check=CalibrationErrorCheck(n_bins=n_bins),
            sign=-1.0,
        )


class KuleshovCalibrationReward(RewardComputer):
    def __init__(self):
        super().__init__(
            name="KuleshovCalibration",
            check=KuleshovCalibrationCheck(),
            sign=-1.0,
        )


class ENCEReward(RewardComputer):
    def __init__(self):
        super().__init__(name="ENCE", check=ENCECheck(), sign=-1.0)


class CalibrationError1StdReward(RewardComputer):
    def __init__(self):
        super().__init__(
            name="CalibrationError1Std",
            check=CalibrationError1StdCheck(),
            sign=-1.0,
        )


class ConformalCoverageReward(RewardComputer):
    def __init__(self, target_coverage: float = 0.9):
        super().__init__(
            name="ConformalCoverage",
            check=ConformalCoverageCheck(target_coverage=target_coverage),
            sign=-1.0,
        )


# -- PIT uniformity: higher p_value better -> sign = +1, no floor delta when
#    sample count below 20 (check returns value=None then; reward()->0.0).

class PITUniformityReward(RewardComputer):
    def __init__(self):
        super().__init__(name="PITUniformity", check=PITUniformityCheck(), sign=+1.0)


# -- IntervalCoverage: value is observed fraction; reward = -delta(|observed - target|).
#    Implementing as score = -|observed - target|, then sign = +1 so positive
#    delta = improvement.

class IntervalCoverageReward(RewardComputer):
    def __init__(self, expected_coverage: float = 0.683):
        super().__init__(
            name="IntervalCoverage",
            check=IntervalCoverageCheck(expected_coverage=expected_coverage),
            sign=+1.0,
        )
        self._target = expected_coverage

    def _score(self, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        return -abs(value - self._target)


# -- VarianceAlignment: value is ratio; reward = -delta(|ratio - 1|).

class VarianceAlignmentReward(RewardComputer):
    def __init__(self):
        super().__init__(
            name="VarianceAlignment",
            check=VarianceAlignmentCheck(),
            sign=+1.0,
        )

    def _score(self, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        return -abs(value - 1.0)


# -- VarErrCorrelation: higher Spearman rho better -> sign = +1.

class VarErrCorrelationReward(RewardComputer):
    def __init__(self):
        super().__init__(
            name="VarErrCorrelation",
            check=VarianceErrorCorrelationCheck(),
            sign=+1.0,
        )


# -- Different-signal rewards: consume sigma-series and/or op_states rather
#    than (y, mu, sigma). Each overrides reward() because the base class's
#    (y, mu, sigma) delta shape doesn't apply.

class _SignalDeltaReward(RewardComputer):
    """Base for rewards whose check consumes signals in ``**extras`` instead
    of (y, mu, sigma) triples. Subclasses implement ``_value_from_extras`` to
    return the scalar to delta between before/after; None means "skip"."""

    def _value_from_extras(self, extras: dict, when: str) -> Optional[float]:
        raise NotImplementedError

    def reward(
        self,
        y_before, mu_before, sigma_before,
        y_after, mu_after, sigma_after,
        **extras: Any,
    ) -> float:
        v_before = self._value_from_extras(extras, "before")
        v_after = self._value_from_extras(extras, "after")
        if v_before is None or v_after is None:
            return 0.0
        return self.sign * (v_after - v_before)


class UncertaintyEvolutionReward(_SignalDeltaReward):
    """Personality: reward query patterns whose forward-uncertainty series
    does NOT trend implausibly downward. value = flagged-channel count; the
    scalar sigma-series has n_channels=1, so value ∈ {0, 1}. Delta is often
    zero step-to-step — SAC learns from the rare flips.
    """
    def __init__(self, slope_threshold: float = -0.01):
        super().__init__(
            name="UncertaintyEvolution",
            check=UncertaintyEvolutionCheck(slope_threshold=slope_threshold),
            sign=-1.0,
        )

    def _value_from_extras(self, extras: dict, when: str) -> Optional[float]:
        series = extras.get(f"sigma_series_{when}")
        if series is None or len(series) < 2:
            return None
        res = self.check.run(history=[], uncertainties=np.asarray(series, dtype=float))
        return None if res.value is None else float(res.value)


class UncertaintyAnomalyReward(_SignalDeltaReward):
    """Personality: reward query patterns whose forward uncertainty stays
    within-distribution vs its own accumulated history. value = anomalous
    fraction ∈ [0, 1]; lower is better.
    """
    def __init__(self, z_threshold: float = 3.0):
        super().__init__(
            name="UncertaintyAnomaly",
            check=UncertaintyAnomalyCheck(z_threshold=z_threshold),
            sign=-1.0,
        )

    def _value_from_extras(self, extras: dict, when: str) -> Optional[float]:
        series = extras.get(f"sigma_series_{when}")
        if series is None or len(series) < 3:
            return None
        res = self.check.run(history=[], uncertainties=np.asarray(series, dtype=float))
        return None if res.value is None else float(res.value)


class MahalanobisOODReward(_SignalDeltaReward):
    """Personality: reward keeping queries in-distribution wrt the queried-x
    manifold. value = OOD fraction over the trailing window; lower is better.
    Below MahalanobisOODCheck's min_history (default 20) the check returns
    value=None → reward 0.0 until the buffer fills.
    """
    def __init__(self):
        super().__init__(
            name="MahalanobisOOD",
            check=MahalanobisOODCheck(),
            sign=-1.0,
        )

    def _value_from_extras(self, extras: dict, when: str) -> Optional[float]:
        x = extras.get(f"x_{when}")
        sigma_series = extras.get(f"sigma_series_{when}")
        if x is None or len(x) < 2:
            return None
        op_states = np.asarray(x, dtype=float).reshape(-1, 1)
        kwargs = {"op_states": op_states}
        if sigma_series is not None and len(sigma_series) == op_states.shape[0]:
            kwargs["uncertainties"] = np.asarray(sigma_series, dtype=float)
        res = self.check.run(history=[], **kwargs)
        return None if res.value is None else float(res.value)


# Public registry for trainers / env / aggregator.
REWARD_REGISTRY: dict[str, type[RewardComputer]] = {
    "CRPS": CRPSReward,
    "NLL": NLLReward,
    "IntervalScore": IntervalScoreReward,
    "CalibrationError": CalibrationErrorReward,
    "KuleshovCalibration": KuleshovCalibrationReward,
    "ENCE": ENCEReward,
    "CalibrationError1Std": CalibrationError1StdReward,
    "ConformalCoverage": ConformalCoverageReward,
    "PITUniformity": PITUniformityReward,
    "IntervalCoverage": IntervalCoverageReward,
    "VarianceAlignment": VarianceAlignmentReward,
    "VarErrCorrelation": VarErrCorrelationReward,
    "UncertaintyEvolution": UncertaintyEvolutionReward,
    "UncertaintyAnomaly": UncertaintyAnomalyReward,
    "MahalanobisOOD": MahalanobisOODReward,
}


class RunningZScore:
    """Welford-style running mean/std for per-agent reward normalization.

    Returns (raw - mean) / (std + eps). After fewer than ``min_samples``
    observations, returns the raw reward (insufficient stats).
    """

    def __init__(self, eps: float = 1e-6, min_samples: int = 32):
        self.eps = eps
        self.min_samples = min_samples
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def std(self) -> float:
        if self.n < 2:
            return 1.0
        return float(np.sqrt(self.M2 / (self.n - 1)))

    def normalize(self, x: float) -> float:
        self.update(x)
        if self.n < self.min_samples:
            return x
        return (x - self.mean) / (self.std + self.eps)
