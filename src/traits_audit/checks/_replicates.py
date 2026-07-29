"""Shared replicate-group extraction for RSE, DUG, and AFC.

All three of ``ReplicationShrinkageExponentCheck``, ``DarkUncertaintyGapCheck``,
and ``AleatoricFloorConsistencyCheck`` need the same underlying data shape:
repeated measurements of ``y_true`` (and, for DUG/AFC, the declared
``y_pred_std``) at the same nominal input, grouped by that input's identity.
This module extracts that shape once so the three checks share one contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ReplicateGroup:
    """One nominal input's repeated measurements."""
    key: Any
    y_true: np.ndarray
    y_pred_mean: Optional[np.ndarray] = None
    y_pred_std: Optional[np.ndarray] = None

    @property
    def r(self) -> int:
        """Number of replicates in this group."""
        return len(self.y_true)


def _coerce_group(key: Any, raw: Any) -> Optional[ReplicateGroup]:
    if isinstance(raw, dict):
        if "y_true" not in raw or raw["y_true"] is None:
            return None
        y_true = np.asarray(raw["y_true"], dtype=float).ravel()
        y_pred_mean = (
            np.asarray(raw["y_pred_mean"], dtype=float).ravel()
            if raw.get("y_pred_mean") is not None else None
        )
        y_pred_std = (
            np.asarray(raw["y_pred_std"], dtype=float).ravel()
            if raw.get("y_pred_std") is not None else None
        )
    else:
        y_true = np.asarray(raw, dtype=float).ravel()
        y_pred_mean = None
        y_pred_std = None
    if y_true.size < 2:
        return None
    return ReplicateGroup(key=key, y_true=y_true, y_pred_mean=y_pred_mean, y_pred_std=y_pred_std)


def build_replicate_groups(history: List[Dict[str, Any]], kwargs: dict) -> List[ReplicateGroup]:
    """
    Pull repeated-measurement groups from kwargs, falling back to history.

    Supported inputs (kwarg wins over history, mirroring ``_require``'s
    convention):

    1. ``kwargs['replicate_groups']``: a dict of
       ``key -> {"y_true": array_like, "y_pred_mean": array_like?, "y_pred_std": array_like?}``,
       or ``key -> array_like`` as shorthand for ``y_true`` only.
    2. ``kwargs['replicate_id']`` (n,) aligned with ``kwargs['y_true']``
       (and optional ``kwargs['y_pred_mean']`` / ``kwargs['y_pred_std']``),
       grouped by distinct ``replicate_id`` values.
    3. The same three keys (``replicate_id``, ``y_true``, ``y_pred_mean``,
       ``y_pred_std``) read per-step from ``history`` dicts.

    Groups with fewer than 2 replicates are dropped — dispersion is undefined
    for a single observation. Returns ``[]`` (never ``None``/raises) when
    nothing usable is found, so callers can treat "no data" as a Skip.
    """
    raw_groups = kwargs.get("replicate_groups")
    if raw_groups is not None:
        groups = []
        for key, raw in raw_groups.items():
            g = _coerce_group(key, raw)
            if g is not None:
                groups.append(g)
        return groups

    if "replicate_id" in kwargs and kwargs["replicate_id"] is not None and "y_true" in kwargs:
        rid = np.asarray(kwargs["replicate_id"]).ravel()
        y_true = np.asarray(kwargs["y_true"], dtype=float).ravel()
        y_pred_mean = (
            np.asarray(kwargs["y_pred_mean"], dtype=float).ravel()
            if kwargs.get("y_pred_mean") is not None else None
        )
        y_pred_std = (
            np.asarray(kwargs["y_pred_std"], dtype=float).ravel()
            if kwargs.get("y_pred_std") is not None else None
        )
        return _group_by_id(rid, y_true, y_pred_mean, y_pred_std)

    rid_vals = [h["replicate_id"] for h in history if "replicate_id" in h and "y_true" in h]
    if rid_vals:
        rid = np.asarray(rid_vals)
        y_true = np.asarray(
            [h["y_true"] for h in history if "replicate_id" in h and "y_true" in h], dtype=float
        )
        y_pred_mean_vals = [
            h.get("y_pred_mean") for h in history if "replicate_id" in h and "y_true" in h
        ]
        y_pred_std_vals = [
            h.get("y_pred_std") for h in history if "replicate_id" in h and "y_true" in h
        ]
        y_pred_mean = (
            np.asarray(y_pred_mean_vals, dtype=float)
            if all(v is not None for v in y_pred_mean_vals) else None
        )
        y_pred_std = (
            np.asarray(y_pred_std_vals, dtype=float)
            if all(v is not None for v in y_pred_std_vals) else None
        )
        return _group_by_id(rid, y_true, y_pred_mean, y_pred_std)

    return []


def _group_by_id(
    rid: np.ndarray,
    y_true: np.ndarray,
    y_pred_mean: Optional[np.ndarray],
    y_pred_std: Optional[np.ndarray],
) -> List[ReplicateGroup]:
    groups: List[ReplicateGroup] = []
    for key in np.unique(rid):
        mask = rid == key
        if mask.sum() < 2:
            continue
        groups.append(
            ReplicateGroup(
                key=key,
                y_true=y_true[mask],
                y_pred_mean=y_pred_mean[mask] if y_pred_mean is not None else None,
                y_pred_std=y_pred_std[mask] if y_pred_std is not None else None,
            )
        )
    return groups
