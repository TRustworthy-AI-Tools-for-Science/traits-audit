"""Random/systematic taxonomy class: SignedBias, RSE, DUG.

See METRIC_TAXONOMY_AUDIT.md §4.1. All three read repeated-measurement data
via ``build_replicate_groups`` (``checks/_replicates.py``) except
``SignedBiasCheck``, which only needs plain ``y_true``/``y_pred_mean``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..base import AuditCategory, AuditCheck, AuditResult
from ._replicates import build_replicate_groups
from .calibration import _require


class SignedBiasCheck(AuditCheck):
    """
    Signed mean residual (bias) — the class-1 constant component that
    absolute-error metrics (MAE, RMSE, MdAE) cannot express.

    ``value = mean(y_pred_mean - y_true)``. A replication scheme would
    classify a persistent non-zero value here as systematic (Class 1,
    Colclough 1987) rather than random.

    Bias is in the same units as the target, so a single hardcoded
    absolute threshold cannot generalize across datasets. Three ways to
    judge it are supported; when more than one is configured, the first
    that applies wins, in this order:

    1. ``threshold`` — an absolute cutoff you supply yourself, in y's units.
    2. ``se_multiplier`` — a statistical, scale-free cutoff:
       ``passed = abs(bias) <= se_multiplier * bias_std_error``. Comparing
       the bias to its own standard error asks whether it is
       distinguishable from zero at all, adapting automatically to sample
       size and noise level. ``se_multiplier=2`` is roughly a 95 % band.
    3. ``rel_std_frac`` — a domain-relative cutoff:
       ``passed = abs(bias) <= rel_std_frac * mean(y_pred_std)``, expressed
       as a fraction of the model's own typical predictive spread rather
       than an absolute number. Needs ``y_pred_std``.

    If none of the three is set (the default), this reports for
    monitoring only, like ``CRPSCheck``.

    Parameters
    ----------
    threshold : float or None
        Absolute ``abs(bias)`` cutoff, in y's units. Takes priority over
        the other two if set.
    se_multiplier : float or None
        ``abs(bias) <= se_multiplier * bias_std_error``. Used when
        ``threshold`` is ``None``.
    rel_std_frac : float or None
        ``abs(bias) <= rel_std_frac * mean(y_pred_std)``. Used when both
        ``threshold`` and ``se_multiplier`` are ``None``. Requires
        ``y_pred_std``.

    References
    ----------
    Colclough, A. R. (1987). Two theories of experimental error. *J. Res.
    NBS*, 92(3), 167-185.

    Required data (kwargs or history keys)
    ----------------------------------------
    ``y_true``, ``y_pred_mean``; ``y_pred_std`` additionally required when
    ``rel_std_frac`` is the active mode.
    """

    def __init__(
        self,
        threshold: float | None = None,
        se_multiplier: float | None = None,
        rel_std_frac: float | None = None,
    ):
        self.threshold = threshold
        self.se_multiplier = se_multiplier
        self.rel_std_frac = rel_std_frac

    @property
    def name(self) -> str:
        return "SignedBias"

    @property
    def category(self) -> AuditCategory:
        return AuditCategory.RANDOM_SYSTEMATIC

    def run(self, history: list[dict[str, Any]], **kwargs) -> AuditResult:
        y_true = _require("y_true", history, kwargs)
        mu = _require("y_pred_mean", history, kwargs)
        if any(v is None for v in (y_true, mu)):
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Skipped — y_true / y_pred_mean not available.",
            )

        residual = mu - y_true
        n = len(residual)
        bias = float(np.mean(residual))
        bias_se = float(np.std(residual, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")

        effective_threshold = self.threshold
        threshold_kind = "absolute" if effective_threshold is not None else None

        if effective_threshold is None and self.se_multiplier is not None:
            if not np.isfinite(bias_se):
                return AuditResult(
                    name=self.name, passed=True, category=self.category,
                    message="Skipped — need n >= 2 to compute bias_std_error for se_multiplier.",
                )
            effective_threshold = self.se_multiplier * bias_se
            threshold_kind = f"{self.se_multiplier:g}x SE"

        elif effective_threshold is None and self.rel_std_frac is not None:
            sigma = _require("y_pred_std", history, kwargs)
            if sigma is None:
                return AuditResult(
                    name=self.name, passed=True, category=self.category,
                    message="Skipped — y_pred_std not available for rel_std_frac.",
                )
            effective_threshold = self.rel_std_frac * float(np.mean(sigma))
            threshold_kind = f"{self.rel_std_frac:g}x mean(y_pred_std)"

        passed = True if effective_threshold is None else abs(bias) <= effective_threshold
        message = f"Signed mean residual = {bias:.4f} (SE={bias_se:.4f}, n={n})"
        if effective_threshold is not None:
            message += f"  [threshold: {threshold_kind} = {effective_threshold:.4f}]"

        return AuditResult(
            name=self.name,
            passed=passed,
            category=self.category,
            value=bias,
            threshold=effective_threshold,
            message=message,
            details={
                "n_samples": n,
                "bias_std_error": bias_se,
                "threshold_kind": threshold_kind,
            },
        )


class ReplicationShrinkageExponentCheck(AuditCheck):
    """
    Replication Shrinkage Exponent (RSE) — whether observed dispersion at a
    repeatedly-measured input averages down with the number of replicates.

    Fits ``u_obs(r) ~ A * r^(-beta)`` over ``r_values`` by subsampling within
    each replicate group, and reports beta-hat with a bootstrap CI.
    beta=0.5 is the fully-random limit (averaging as 1/sqrt(r)); beta=0 is
    the fully-systematic limit (no averaging at all).

    ``u_obs(r)`` is the standard uncertainty of the *r-replicate averaged
    estimate*, and it is measured **across replicate groups**, not within
    one. A component that is constant *within* a group but varies *across*
    groups (a per-run calibration offset, say) is invisible to any
    within-group statistic by construction — it shifts every replicate in
    that group by the same amount, so it cancels out of anything computed
    from that group alone, no matter how the group is sliced. It only shows
    up as dispersion *between* groups, and critically that dispersion does
    **not** shrink as r grows, while the random component's contribution to
    it does (as 1/sqrt(r)) — which is exactly the r^(-beta) law this check
    fits.

    Concretely: for each r, draw one window of r replicates per group,
    average each window, and take the standard deviation of those
    group-level averages *across groups*. Repeat over many random window
    draws and average the resulting std for precision. A purely random
    replication scheme (no shared per-group component) drives this to zero
    as 1/sqrt(r); a scheme with a per-group offset does not.

    .. important::
       Because the statistic is a dispersion **across groups**, groups at
       genuinely different nominal inputs must be residualised against a
       reference before comparison, or real between-input differences (not
       replication behaviour) will dominate the estimate. Supply
       ``y_pred_mean`` per replicate (via ``replicate_groups`` — see
       ``checks/_replicates.py``); this check then operates on
       ``y_true - y_pred_mean``. If ``y_pred_mean`` is omitted, groups are
       compared in raw units — only sensible when every group targets the
       same true value (e.g. repeated runs of one calibration standard).

    .. important::
       **Why this defaults to report-only (``beta_tolerance=None``) rather
       than judging beta against 0.5**: beta=0.5 and beta=0 are both
       *legitimate* values, not a healthy pole and a defective one — for two
       independent reasons documented in the literature this package is
       built from:

       1. The classification is definitionally circular. VIM3 (JCGM
          200:2012) defines each term only in terms of the other — entry
          2.17 gives systematic error as "measurement error minus random
          error," entry 2.19 gives random error as "measurement error minus
          systematic error." Neither pole has an independent definition, so
          treating one as the target smuggles in a value judgement the
          classification itself cannot ground non-circularly.
       2. A component can be legitimately, unfixably systematic without that
          being a defect. Colclough (1987) gives a fixed calibration offset
          as the paradigm "Class 1: constant" case — it is systematic by
          nature, and no amount of replication will or should make it look
          random. A fixed kernel-hyperparameter bias, or measurements
          sharing one instrument's constant offset, correctly drives
          ``beta`` toward 0, and there is nothing here to "fix".

       So, like ``CRPSCheck``/``NegativeLogLikelihoodCheck``/
       ``IntervalScoreCheck``, this check reports beta-hat and its CI for the
       user to interpret in context. A user with an independent reason to
       expect a specific regime (e.g. "this should be dominated by counting
       statistics, so beta should sit near 0.5") can opt in via
       ``beta_tolerance``.

    .. note::
       RSE is *also* a reduction-under-replication metric per
       METRIC_TAXONOMY_AUDIT.md's dual listing — a single ``AuditCategory``
       value can't carry both, so this note records it. Pair with
       ``DarkUncertaintyGapCheck`` (enforced advisorially by
       :func:`traits_audit.validation.find_unpaired_checks`).

    Parameters
    ----------
    r_values : sequence of int
        Replicate-count subsample sizes to fit over (default ``(2,4,8,16)``).
        Only sizes ``<=`` a group's replicate count are used for that group.
    n_subsample : int
        Random size-r subsamples drawn per group per r (default 50).
    beta_tolerance : float or None
        If set, ``passed = abs(beta_hat - 0.5) <= beta_tolerance``. ``None``
        (default) disables pass/fail.
    seed : int
        RNG seed for subsampling and the group-resampling bootstrap CI.

    References
    ----------
    Kim, S. H., Kim, C.-S. & Hwang, S. (2014). Bull. Korean Chem. Soc.,
    35(4), 1057-1064.

    Required data
    -------------
    Replicate groups via ``kwargs['replicate_groups']`` or
    ``kwargs['replicate_id']`` + ``kwargs['y_true']`` (or the same read from
    history) — see ``checks/_replicates.py``. Populate ``y_pred_mean`` per
    replicate too, per the note above, unless every group shares one true
    value.

    .. note::
       At least ``_MIN_GROUPS`` groups are needed for a given r (the
       statistic is a standard deviation *across* groups); groups with fewer
       than r replicates are excluded from that r's estimate. Precision
       (not correctness) improves with more groups and with
       ``n_subsample``: each subsample trial draws one window of r
       replicates per group and the resulting across-group std is averaged
       over trials.
    """

    _MIN_GROUPS = 3

    def __init__(
        self,
        r_values=(2, 4, 8, 16),
        n_subsample: int = 50,
        beta_tolerance: float | None = None,
        seed: int = 0,
    ):
        self.r_values = tuple(r_values)
        self.n_subsample = n_subsample
        self.beta_tolerance = beta_tolerance
        self.seed = seed

    @property
    def name(self) -> str:
        return "ReplicationShrinkageExponent"

    @property
    def category(self) -> AuditCategory:
        return AuditCategory.RANDOM_SYSTEMATIC

    @staticmethod
    def _residuals(g) -> np.ndarray:
        if g.y_pred_mean is not None:
            return g.y_true - g.y_pred_mean
        return g.y_true

    def _window_mean_draws(self, groups, rng: np.random.Generator) -> dict[int, dict[Any, np.ndarray]]:
        """Precompute, once, each group's r-window sample means at every r.

        For each r and group with >= r replicates, draws ``n_subsample``
        random windows of length r and records their means as a
        ``(n_subsample,)`` array keyed by group. Computed a single time up
        front — not inside the bootstrap loop below — because the
        group-level bootstrap resamples *which groups* are included, not
        the window-drawing procedure itself, so each group's draws can be
        reused across all bootstrap resamples.
        """
        out: dict[int, dict[Any, np.ndarray]] = {}
        for r in self.r_values:
            per_group: dict[Any, np.ndarray] = {}
            for g in groups:
                if g.r < r:
                    continue
                y = self._residuals(g)
                n = len(y)
                starts = rng.integers(0, n - r + 1, size=self.n_subsample)
                per_group[g.key] = np.array([np.mean(y[s:s + r]) for s in starts])
            out[r] = per_group
        return out

    def _fit_beta(self, group_keys, draws: dict[int, dict[Any, np.ndarray]]):
        rs_used, u_obs = [], []
        for r in self.r_values:
            per_group = draws.get(r, {})
            rows = [per_group[k] for k in group_keys if k in per_group]
            if len(rows) < self._MIN_GROUPS:
                continue
            # (n_groups, n_subsample): across-group std per trial, averaged over trials.
            M = np.array(rows)
            u = float(np.mean(np.std(M, axis=0, ddof=1)))
            if u > 0:
                rs_used.append(r)
                u_obs.append(u)
        if len(rs_used) < 2:
            return None, rs_used, u_obs
        slope, _ = np.polyfit(np.log(rs_used), np.log(u_obs), 1)
        return float(-slope), rs_used, u_obs

    def run(self, history: list[dict[str, Any]], **kwargs) -> AuditResult:
        groups = build_replicate_groups(history, kwargs)
        if not groups:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Skipped — no replicate groups available (need replicate_groups or replicate_id + y_true).",
            )

        rng = np.random.default_rng(self.seed)
        draws = self._window_mean_draws(groups, rng)
        group_keys = [g.key for g in groups]
        beta_hat, rs_used, u_obs = self._fit_beta(group_keys, draws)
        if beta_hat is None:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message=(
                    f"Skipped — fewer than 2 r_values with >= {self._MIN_GROUPS} "
                    "usable groups (need more groups or more replicates per group)."
                ),
            )

        n_boot = 200
        boot_betas = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(group_keys), size=len(group_keys))
            resampled_keys = [group_keys[i] for i in idx]
            b, _, _ = self._fit_beta(resampled_keys, draws)
            if b is not None:
                boot_betas.append(b)
        beta_ci = (
            tuple(float(x) for x in np.percentile(boot_betas, [2.5, 97.5]))
            if len(boot_betas) >= 10 else None
        )

        passed = True if self.beta_tolerance is None else abs(beta_hat - 0.5) <= self.beta_tolerance

        return AuditResult(
            name=self.name,
            passed=passed,
            category=self.category,
            value=beta_hat,
            threshold=self.beta_tolerance,
            message=f"beta_hat = {beta_hat:.4f} (CI={beta_ci}) — 0.5=random, 0=systematic; both legitimate.",
            details={
                "beta_ci": beta_ci,
                "u_obs_by_r": dict(zip(rs_used, u_obs)),
                "r_values_used": rs_used,
                "n_groups": len(groups),
                "taxonomy_note": "Also a reduction-under-replication metric (dual-listed); pair with DarkUncertaintyGap.",
            },
        )


class DarkUncertaintyGapCheck(AuditCheck):
    """
    Dark Uncertainty Gap (DUG) — observed replicate dispersion divided by the
    declared/enumerated combined standard uncertainty.

    ``DUG_g = std(y_true_g, ddof=1) / mean(y_pred_std_g)`` per replicate
    group; ``value = median(DUG_g)`` across groups.

    **DUG > 1 is Kim et al.'s (2014) underestimation condition** — the
    enumerated uncertainty budget is smaller than the observed scatter, i.e.
    unrecognized ("dark") sources contribute more dispersion than the
    declared budget accounts for. Unlike RSE's contested reference point
    (see that class's docstring), DUG's boundary of 1.0 IS the
    literature-mandated one, so ``dug_threshold`` defaults to exactly 1.0
    rather than a padded value — only the *aggregation across groups*
    (median, chosen for robustness to outlier groups) is a free
    implementation choice.

    Parameters
    ----------
    dug_threshold : float
        Maximum acceptable DUG (default 1.0 — Kim et al.'s own boundary).
    n_boot : int
        Bootstrap resamples over groups for the CI (default 200).
    seed : int
        RNG seed.

    References
    ----------
    Kim, S. H., Kim, C.-S. & Hwang, S. (2014). Bull. Korean Chem. Soc.,
    35(4), 1057-1064.

    Required data
    -------------
    Replicate groups (see ``ReplicationShrinkageExponentCheck``), each with
    both ``y_true`` and ``y_pred_std`` populated.
    """

    def __init__(self, dug_threshold: float = 1.0, n_boot: int = 200, seed: int = 0):
        self.dug_threshold = dug_threshold
        self.n_boot = n_boot
        self.seed = seed

    @property
    def name(self) -> str:
        return "DarkUncertaintyGap"

    @property
    def category(self) -> AuditCategory:
        return AuditCategory.RANDOM_SYSTEMATIC

    def run(self, history: list[dict[str, Any]], **kwargs) -> AuditResult:
        groups = build_replicate_groups(history, kwargs)
        groups = [g for g in groups if g.y_pred_std is not None]
        if not groups:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Skipped — no replicate groups with y_pred_std available.",
            )

        per_group = {}
        dugs = []
        for g in groups:
            u_enum = float(np.mean(g.y_pred_std))
            if u_enum <= 0:
                continue
            dug_g = float(np.std(g.y_true, ddof=1) / u_enum)
            per_group[g.key] = dug_g
            dugs.append(dug_g)

        if not dugs:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Skipped — no groups with a positive enumerated uncertainty.",
            )

        value = float(np.median(dugs))
        rng = np.random.default_rng(self.seed)
        boot = [float(np.median(rng.choice(dugs, size=len(dugs), replace=True))) for _ in range(self.n_boot)]
        dug_ci = tuple(float(x) for x in np.percentile(boot, [2.5, 97.5]))

        return AuditResult(
            name=self.name,
            passed=value <= self.dug_threshold,
            category=self.category,
            value=value,
            threshold=self.dug_threshold,
            message=f"DUG (median) = {value:.4f} (CI={dug_ci}) — >1 is underestimation of the enumerated budget.",
            details={
                "dug_ci": dug_ci,
                "per_group_dug": per_group,
                "n_groups": len(dugs),
                "taxonomy_note": "Also a reduction-under-replication metric (dual-listed); pair with ReplicationShrinkageExponent.",
            },
        )
