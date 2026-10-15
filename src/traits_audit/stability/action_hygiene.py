"""DMDc action-column hygiene: drop numerically-constant control columns.

Feeding constant action columns (std as low as 1e-16) into a DMDc fit
produces a severely ill-conditioned lstsq/SVD design matrix. For
``pydmd.DMDc`` (whose default ``svd_rank_omega=-1`` means no truncation of the
stacked ``[state; control]`` matrix, and whose ``np.reciprocal``-based
inversion has no floor) this was confirmed to corrupt fitted control-coupling
coefficients to astronomically large (up to ~1e15), physically meaningless
values that silently corrupted downstream R²/stability comparisons.
:func:`~traits_audit.stability.dmdc.fit_dmdc`'s ``numpy.linalg.lstsq(rcond=None)``
backend does not reproduce that blowup (its SVD-based pseudoinverse floors
near-zero singular values) but still wastes fitted rank on columns that carry
no information.

The mask must be computed GLOBALLY across a whole experiment (every
dataset/policy/seed), not per trajectory: a per-trajectory mask gives each
policy/model a different action-space width/definition, which breaks direct
comparability of lambda_max / R² / effective rank across them.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

__all__ = ["compute_global_action_kept_mask", "drop_degenerate_action_columns"]


def compute_global_action_kept_mask(
    action_arrays: Iterable[np.ndarray | None],
    std_floor: float = 1e-6,
    feature_names: Sequence[str] | None = None,
) -> np.ndarray:
    """Determine which action columns carry real (non-degenerate) variance
    ANYWHERE in a whole experiment.

    ORs each array's own std-based degeneracy check together -- a column is
    kept globally if it is non-degenerate in AT LEAST ONE array.

    Parameters
    ----------
    action_arrays : iterable of (T_i, n_action) arrays
        One per trajectory (pooled across every dataset/policy/seed). ``None``
        entries and arrays with fewer than 2 rows are skipped.
    std_floor : float
        A column is non-degenerate in an array iff its std is ``>= std_floor``.
    feature_names : sequence of str, optional
        Column names, used only for the progress message.

    Returns
    -------
    kept_mask : boolean array of length ``n_action``

    Raises
    ------
    RuntimeError
        If no array has usable actions.
    """
    any_real = None
    for actions in action_arrays:
        if actions is None or len(actions) < 2:
            continue
        actions = np.asarray(actions, dtype=np.float64)
        col_real = actions.std(axis=0) >= std_floor
        any_real = col_real if any_real is None else (any_real | col_real)
    if any_real is None:
        raise RuntimeError("No trajectories with usable actions found -- cannot "
                            "determine the global action-column mask.")
    names = (list(feature_names) if feature_names is not None
             else list(range(len(any_real))))[:len(any_real)]
    kept_names = [n for n, k in zip(names, any_real) if k]
    dropped_names = [n for n, k in zip(names, any_real) if not k]
    print(f"  Global action-column mask (std >= {std_floor} in at least one "
          f"trajectory): kept {kept_names}, dropped {dropped_names}")
    return any_real


def drop_degenerate_action_columns(
    actions: np.ndarray,
    kept_mask: np.ndarray | None = None,
    std_floor: float = 1e-6,
    feature_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce an action array to its non-degenerate columns.

    IMPORTANT -- callers should almost always pass ``kept_mask`` explicitly
    (from :func:`compute_global_action_kept_mask`, computed ONCE across the
    whole experiment), not rely on this function's own per-array fallback
    (``kept_mask=None``, which recomputes std over just the array passed in):
    a per-trajectory mask gives different policies/models different
    action-space widths/definitions, breaking direct cross-policy/
    cross-model comparability. The ``kept_mask=None`` fallback exists only for
    callers that genuinely have no broader context.

    If EVERY globally-kept column happens to be locally constant for this
    particular trajectory, returns a single all-zero column rather than a
    0-width array: DMDc fitters generally require ``actions.shape[1] >= 1``.
    Callers must still expect NaN/failed fits downstream for this case --
    an all-zero control input carries no estimable control response, which is
    the mathematically honest answer, not a bug to work around.

    Returns
    -------
    (reduced_actions, kept_mask) : ``reduced_actions`` has shape
    ``(T, kept_mask.sum())`` (or ``(T, 1)`` all-zero if ``kept_mask`` is all
    False); ``kept_mask`` is the boolean array actually used (either the one
    passed in, or a freshly computed per-array fallback).
    """
    actions = np.asarray(actions, dtype=np.float64)
    if kept_mask is None:
        kept_mask = actions.std(axis=0) >= std_floor
    kept_mask = np.asarray(kept_mask, dtype=bool)
    if not kept_mask.any():
        names = (list(feature_names) if feature_names is not None
                 else list(range(actions.shape[1])))
        dropped_names = names[:actions.shape[1]]
        print(f"    All {actions.shape[1]} action column(s) degenerate for this "
              f"trajectory: {dropped_names} -- using a single all-zero column "
              f"(honest 'no real control input').")
        return np.zeros((len(actions), 1)), kept_mask
    return actions[:, kept_mask], kept_mask
