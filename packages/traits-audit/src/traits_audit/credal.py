"""
Minimal credal-set representation for the variability/ignorance taxonomy class
(METRIC_TAXONOMY_AUDIT.md §4.5).

A single distribution conflates variability (genuine dispersion) with
ignorance (imprecision about which distribution applies) by construction.
``CredalSet`` is the smallest representation that can tell them apart: either
a direct bounding interval (Dubois/Swinburne-Perez style), or an ensemble of
predictive distributions whose envelope defines the bounding interval.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CredalSet:
    """
    Exactly one representation must be supplied:

    - ``lower``, ``upper`` — direct bounding interval, ``(n_points,)`` each.
    - ``y_pred_ensemble`` — ``(n_models, n_points)``, optionally with
      ``y_pred_std_ensemble`` of the same shape.

    When ``y_pred_std_ensemble`` is given, the upper/lower probability of a
    reference interval is the sup/inf over members of each member's own
    Gaussian CDF over that interval — a genuine imprecise-probability
    sup/inf. When omitted, each ensemble member is treated as a Dirac point
    mass, so its own probability of the reference event is 0 or 1 (in or
    out); the upper/lower probability of the credal set spanned by the
    members is then ``any``/``all`` of that indicator across members — the
    standard "extreme point" sup/inf for a finite set of vertex measures,
    matching the Gaussian branch above rather than reducing to it as a
    special case with zero width.
    """

    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    y_pred_ensemble: np.ndarray | None = None
    y_pred_std_ensemble: np.ndarray | None = None

    def __post_init__(self):
        if self.lower is not None:
            self.lower = np.asarray(self.lower, dtype=float).ravel()
        if self.upper is not None:
            self.upper = np.asarray(self.upper, dtype=float).ravel()
        if self.y_pred_ensemble is not None:
            self.y_pred_ensemble = np.asarray(self.y_pred_ensemble, dtype=float)
        if self.y_pred_std_ensemble is not None:
            self.y_pred_std_ensemble = np.asarray(self.y_pred_std_ensemble, dtype=float)

    def bounding_interval(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns ``(lower, upper)``, ``(n_points,)`` each, regardless of
        which representation was supplied."""
        if self.lower is not None and self.upper is not None:
            return self.lower, self.upper
        if self.y_pred_ensemble is not None:
            if self.y_pred_std_ensemble is not None:
                lo = np.min(self.y_pred_ensemble - 3.0 * self.y_pred_std_ensemble, axis=0)
                hi = np.max(self.y_pred_ensemble + 3.0 * self.y_pred_std_ensemble, axis=0)
            else:
                lo = np.min(self.y_pred_ensemble, axis=0)
                hi = np.max(self.y_pred_ensemble, axis=0)
            return lo, hi
        raise ValueError("CredalSet needs either (lower, upper) or y_pred_ensemble.")

    def reference_probabilities(
        self, ref_lower: np.ndarray, ref_upper: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-point ``(P_upper, P_lower)`` of the event
        ``{y in [ref_lower, ref_upper]}``."""
        ref_lower = np.asarray(ref_lower, dtype=float).ravel()
        ref_upper = np.asarray(ref_upper, dtype=float).ravel()

        if self.y_pred_ensemble is not None:
            if self.y_pred_std_ensemble is not None:
                from scipy.stats import norm

                probs = norm.cdf(ref_upper[None, :], self.y_pred_ensemble, self.y_pred_std_ensemble) - \
                    norm.cdf(ref_lower[None, :], self.y_pred_ensemble, self.y_pred_std_ensemble)
                return probs.max(axis=0), probs.min(axis=0)
            member_in = (self.y_pred_ensemble >= ref_lower[None, :]) & (
                self.y_pred_ensemble <= ref_upper[None, :]
            )
            # Each Dirac member m has P_m(A) in {0, 1} (in or out of the
            # reference interval). The credal set is the convex hull of
            # these vertex measures, and for a fixed event A the probability
            # functional P(A) = sum_i w_i P_i(A) is linear in the mixture
            # weights w_i, so its sup/inf over the hull is attained at the
            # vertices themselves: P_upper = max_i P_i(A) = "any member in",
            # P_lower = min_i P_i(A) = "all members in". A plain mean here
            # (the previous implementation) computes the fraction of members
            # inside instead, which returns the SAME value for both bounds
            # by construction -- collapsing IWF to 0 even when the ensemble
            # members genuinely disagree about the event.
            p_upper = member_in.any(axis=0).astype(float)
            p_lower = member_in.all(axis=0).astype(float)
            return p_upper, p_lower

        lo, hi = self.bounding_interval()
        # A bare bounding interval carries no interior distributional shape,
        # so the reference event's upper/lower probability collapses to
        # whether [ref_lower, ref_upper] could plausibly overlap [lo, hi]:
        # P_upper = 1 if the intervals overlap at all, else 0; P_lower = 1
        # only if [lo, hi] is entirely contained in the reference interval.
        overlap = (hi >= ref_lower) & (lo <= ref_upper)
        contained = (lo >= ref_lower) & (hi <= ref_upper)
        p_upper = overlap.astype(float)
        p_lower = contained.astype(float)
        return p_upper, p_lower

    def contains(self, y_true: np.ndarray) -> np.ndarray:
        """Boolean ``(n_points,)``: whether ``y_true`` falls within
        ``[lower, upper]``."""
        y_true = np.asarray(y_true, dtype=float).ravel()
        lo, hi = self.bounding_interval()
        return (y_true >= lo) & (y_true <= hi)
