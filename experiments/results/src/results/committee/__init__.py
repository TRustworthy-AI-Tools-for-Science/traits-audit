"""Committee-RL acquisition v0 — stylistic differentiation via audit rewards.

See _results/committee_v0/predicted_styles.md for the pre-registered
stylistic predictions and the design rationale.
"""
from results.committee.rewards import (
    REWARD_REGISTRY,
    CalibrationErrorReward,
    ConformalCoverageReward,
    CRPSReward,
    IntervalCoverageReward,
    IntervalScoreReward,
    NLLReward,
    PITUniformityReward,
    RewardComputer,
    VarErrCorrelationReward,
    VarianceAlignmentReward,
)

__all__ = [
    "REWARD_REGISTRY",
    "CRPSReward",
    "CalibrationErrorReward",
    "ConformalCoverageReward",
    "IntervalCoverageReward",
    "IntervalScoreReward",
    "NLLReward",
    "PITUniformityReward",
    "RewardComputer",
    "VarErrCorrelationReward",
    "VarianceAlignmentReward",
]
