"""Ergodic/non-ergodic taxonomy class: EID, DMDcSpectralRadius, ResidualPersistenceHalfLife.

See METRIC_TAXONOMY_AUDIT.md §4.4. ``LyapunovStabilityCheck`` (local
lambda_max, checks/lyapunov.py) and ``DMDcSpectralRadiusCheck`` (global
rho(A), this module) are the already-present pairing this class discusses;
``EnsembleIndependenceDeficitCheck`` and
``ResidualPersistenceHalfLifeCheck`` add the ensemble-renewal and
autocorrelation-persistence sides of the same question.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..base import AuditCategory, AuditCheck, AuditResult
from ..bootstrap import moving_block_bootstrap_ci
from .calibration import _require


class EnsembleIndependenceDeficitCheck(AuditCheck):
    """
    Ensemble Independence Deficit (EID) — whether ensemble members are
    renewed per realisation, or are effectively one member wearing many
    hats.

    Der Kiureghian & Ditlevsen (2009): shared epistemic uncertainty induces
    statistical dependence between ensemble members; treating them as
    independent misestimates failure probability by orders of magnitude in
    redundant systems.

    With rho_bar the mean pairwise correlation of ensemble-member residuals
    across the test set::

        n_eff = n / (1 + (n - 1) * rho_bar)
        EID   = 1 - n_eff / n

    EID=0: members are independent, a variance-over-members means what it's
    normally taken to mean. EID -> 1: the ensemble is effectively one member;
    its reported spread is a persistent, non-averaging component.

    Parameters
    ----------
    eid_threshold : float
        Maximum acceptable EID (default 0.5).

    References
    ----------
    Kish, L. (1965). *Survey Sampling*. Wiley. (source of the ``n_eff``
    design-effect formula this check reuses; no DOI, predates registration).
    Der Kiureghian, A. & Ditlevsen, O. (2009). Aleatory or epistemic? Does it
    matter? *Structural Safety*, 31(2), 105-112.

    Required data (kwargs or history keys)
    ----------------------------------------
    ``y_true``, ``y_pred_ensemble`` (n_models, n_points) — per-member
    predictions, needs nothing beyond what a deep ensemble already computes.
    """

    def __init__(self, eid_threshold: float = 0.5):
        self.eid_threshold = eid_threshold

    @property
    def name(self) -> str:
        return "EnsembleIndependenceDeficit"

    @property
    def category(self) -> AuditCategory:
        return AuditCategory.ERGODIC_NON_ERGODIC

    def run(self, history: list[dict[str, Any]], **kwargs) -> AuditResult:
        y_true = _require("y_true", history, kwargs)
        ensemble = kwargs.get("y_pred_ensemble")

        if y_true is None or ensemble is None:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Skipped — y_true / y_pred_ensemble not available.",
            )

        ensemble = np.asarray(ensemble, dtype=float)
        if ensemble.ndim != 2 or ensemble.shape[0] < 2:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Skipped — y_pred_ensemble needs >= 2 members (n_models, n_points).",
            )

        residuals = ensemble - y_true[None, :]
        n_models = residuals.shape[0]
        corr = np.corrcoef(residuals)
        iu = np.triu_indices(n_models, k=1)
        rho_bar = float(np.mean(corr[iu]))

        n_eff = n_models / (1.0 + (n_models - 1) * rho_bar)
        eid = 1.0 - n_eff / n_models

        return AuditResult(
            name=self.name,
            passed=eid <= self.eid_threshold,
            category=self.category,
            value=eid,
            threshold=self.eid_threshold,
            message=f"EID = {eid:.4f}  (rho_bar={rho_bar:.4f}, n_eff={n_eff:.2f}/{n_models})",
            details={"rho_bar": rho_bar, "n_eff": n_eff, "n_models": n_models},
        )


class DMDcSpectralRadiusCheck(AuditCheck):
    """
    Thin pipeline-configurable wrapper around DMDc's spectral radius rho(A),
    so it can appear in a pipeline's check list and be validated for pairing
    with ``LyapunovStabilityCheck`` (see
    :func:`traits_audit.validation.find_unpaired_checks`).

    lambda_max (Lyapunov, local) is evaluated at the operating points the
    acquisition policy actually visited — the *local* dynamics of the
    uncertainty landscape. rho(A) (this check) is the *global*, spectral
    view of the same trajectory's fitted linear operator. Neither alone
    separates a locally rough landscape from a globally divergent one — the
    two must always be reported together (METRIC_TAXONOMY_AUDIT.md §4.4).

    The actual DMDc fit (``fit_dmdc`` / ``trajectory.analyze_trajectory``)
    is not re-implemented here; this check only wraps an already-computed
    ``rho_A`` value for pipeline/report integration.

    Parameters
    ----------
    stability_threshold : float
        Maximum acceptable rho(A) (default 1.0 — the discrete-time
        unit-circle boundary, matching ``LyapunovStabilityCheck``'s
        convention).

    References
    ----------
    Proctor, J. L., Brunton, S. L. & Kutz, J. N. (2016). Dynamic mode
    decomposition with control. *SIAM J. Appl. Dyn. Syst.*, 15(1), 142-161.
    Base method: Schmid, P. J. (2010). Dynamic mode decomposition of
    numerical and experimental data. *J. Fluid Mech.*, 656, 5-28.

    Required data (kwargs)
    -----------------------
    ``rho_A`` (float) — precomputed via ``dmdc.py``/``trajectory.py``.
    Optional: ``rho_A_ci`` (tuple).
    """

    def __init__(self, stability_threshold: float = 1.0):
        self.stability_threshold = stability_threshold

    @property
    def name(self) -> str:
        return "DMDcSpectralRadius"

    @property
    def category(self) -> AuditCategory:
        return AuditCategory.ERGODIC_NON_ERGODIC

    def run(self, history: list[dict[str, Any]], **kwargs) -> AuditResult:
        rho_A = kwargs.get("rho_A")
        if rho_A is None:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Skipped — rho_A not provided.",
            )
        rho_A = float(rho_A)
        rho_A_ci = kwargs.get("rho_A_ci")

        return AuditResult(
            name=self.name,
            passed=rho_A <= self.stability_threshold,
            category=self.category,
            value=rho_A,
            threshold=self.stability_threshold,
            message=f"rho(A) = {rho_A:.4f}" + (f"  (CI={rho_A_ci})" if rho_A_ci else ""),
            details={"rho_A_ci": rho_A_ci},
        )


class ResidualPersistenceHalfLifeCheck(AuditCheck):
    """
    Residual Persistence Half-Life — the lag at which residual
    autocorrelation (at a fixed input) crosses 1/e, normalized by campaign
    length.

    Half-life exceeding the campaign length means no amount of replication
    *within* the campaign averages the component down — the trajectory-side
    form of the same question ``EnsembleIndependenceDeficitCheck`` asks of
    ensemble members.

    Parameters
    ----------
    max_lag : int or None
        Maximum lag to search (default ``min(T // 4, 40)``).
    block_len : int
        Moving-block-bootstrap block length for the CI (default 8).
    n_boot : int
        Bootstrap resamples (default 200).
    seed : int
        RNG seed.
    half_life_fraction_threshold : float
        Maximum acceptable ``half_life / campaign_length`` (default 1.0 — the
        literature's own stated failure condition: half-life exceeding
        campaign length).

    References
    ----------
    Der Kiureghian, A. & Ditlevsen, O. (2009). *Structural Safety*, 31(2),
    105-112.

    Required data (kwargs or history keys)
    ----------------------------------------
    ``residuals_at_fixed_x`` — ``(T,)`` residual series at a single
    (approximately fixed) input, ordered by campaign step.
    """

    def __init__(
        self,
        max_lag: int | None = None,
        block_len: int = 8,
        n_boot: int = 200,
        seed: int = 42,
        half_life_fraction_threshold: float = 1.0,
    ):
        self.max_lag = max_lag
        self.block_len = block_len
        self.n_boot = n_boot
        self.seed = seed
        self.half_life_fraction_threshold = half_life_fraction_threshold

    @property
    def name(self) -> str:
        return "ResidualPersistenceHalfLife"

    @property
    def category(self) -> AuditCategory:
        return AuditCategory.ERGODIC_NON_ERGODIC

    @staticmethod
    def _acf(x: np.ndarray, max_lag: int) -> np.ndarray:
        x = x - np.mean(x)
        n = len(x)
        var = np.dot(x, x) / n
        if var == 0:
            return np.zeros(max_lag)
        return np.array([
            (np.dot(x[:n - lag], x[lag:]) / n) / var for lag in range(1, max_lag + 1)
        ])

    def _half_life(self, x: np.ndarray) -> float:
        n = len(x)
        max_lag = self.max_lag or max(1, min(n // 4, 40))
        max_lag = min(max_lag, n - 1)
        if max_lag < 1:
            return float(n)
        acf = self._acf(x, max_lag)
        threshold = 1.0 / np.e
        below = np.where(acf <= threshold)[0]
        if below.size == 0:
            # Never crosses within max_lag: report a lower-bound sentinel
            # strictly above the campaign length so normalized_half_life
            # comes out > 1 (a genuine fail against the default threshold),
            # rather than landing exactly on the pass/fail boundary — "we
            # could not observe this component averaging down at all within
            # the available data" is the failure condition this metric
            # exists to catch, not a borderline pass.
            return float(n) + 1.0
        k = below[0]
        if k == 0:
            return 1.0
        # linear-interpolate between lag k (acf<=1/e) and lag k-1 (acf>1/e)
        a_hi, a_lo = acf[k - 1], acf[k]
        if a_hi == a_lo:
            return float(k + 1)
        frac = (a_hi - threshold) / (a_hi - a_lo)
        return float(k) + frac  # lag k-1 is index k-1 -> "lag k" in 1-indexed terms

    def run(self, history: list[dict[str, Any]], **kwargs) -> AuditResult:
        series = kwargs.get("residuals_at_fixed_x")
        if series is None:
            vals = [h["residuals_at_fixed_x"] for h in history if "residuals_at_fixed_x" in h]
            series = vals if vals else None
        if series is None:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message="Skipped — residuals_at_fixed_x not available.",
            )

        x = np.asarray(series, dtype=float).ravel()
        n = len(x)
        if n < 3 * self.block_len:
            return AuditResult(
                name=self.name, passed=True, category=self.category,
                message=f"Skipped — series too short ({n} < {3 * self.block_len}).",
            )

        half_life = self._half_life(x)
        normalized = half_life / n
        ci = moving_block_bootstrap_ci(x, self._half_life, self.block_len, self.n_boot, self.seed)
        normalized_ci = tuple(c / n for c in ci) if ci is not None else None

        return AuditResult(
            name=self.name,
            passed=normalized <= self.half_life_fraction_threshold,
            category=self.category,
            value=normalized,
            threshold=self.half_life_fraction_threshold,
            message=(
                f"Normalized half-life = {normalized:.4f} "
                f"({half_life:.2f} steps / {n} step campaign)"
            ),
            details={
                "half_life_steps": half_life,
                "campaign_length": n,
                "normalized_ci": normalized_ci,
            },
        )
