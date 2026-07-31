"""Committee-RL acquisition v0 — stylistic differentiation via audit rewards.

See _results/committee_v0/predicted_styles.md for the pre-registered
stylistic predictions and the design rationale.
"""
from traits_audit.committee.rewards import (
    REWARD_REGISTRY,
    RewardComputer,
    CRPSReward,
    NLLReward,
    IntervalScoreReward,
    CalibrationErrorReward,
    ConformalCoverageReward,
    PITUniformityReward,
    IntervalCoverageReward,
    VarianceAlignmentReward,
    VarErrCorrelationReward,
)

__all__ = [
    "REWARD_REGISTRY",
    "RewardComputer",
    "CRPSReward",
    "NLLReward",
    "IntervalScoreReward",
    "CalibrationErrorReward",
    "ConformalCoverageReward",
    "PITUniformityReward",
    "IntervalCoverageReward",
    "VarianceAlignmentReward",
    "VarErrCorrelationReward",
]
