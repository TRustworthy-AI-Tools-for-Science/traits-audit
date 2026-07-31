"""State-space policies that use the audit system health vector as belief state.

Standard usage of :class:`~battery_forecast.policies.StateSpacePolicy` and
:class:`~battery_forecast.policies.BayesianDynamicsPolicy` appends the
**per-parameter ECM variance vector** (shape ``(n_params,)``) to the battery
state when augmenting the belief state.  This represents the *magnitude* of
uncertainty for each individual circuit element.

The policies in this module instead append the **audit health vector**
(shape ``(12,)``) produced by the full uncertainty audit pipeline.
Rather than asking "how uncertain is each parameter?", this vector asks
"how well is the experiment's uncertainty characterisation functioning?"

It encodes two classes of experiment health:

**Predictive-quality fields** (from TSModel + Evaluate audit):

.. code-block:: text

    Index  Field                  Source check              Direction
    ─────  ─────────────────────  ────────────────────────  ─────────
      0    variance_alignment     VarianceAlignmentCheck     ≈ 1.0
      1    calibration_error      TraitsCalibrationCheck     lower ↓
      2    ence                   TraitsCalibrationCheck     lower ↓
      3    miscalibration_area    TraitsCalibrationCheck     lower ↓
      4    variance_error_corr    VarianceErrorCorrelation   higher ↑
      5    n_decreasing_channels  UncertaintyEvolution       lower ↓
      6    anomalous_fraction     UncertaintyAnomalies       lower ↓

**EIS-specific inference quality** (from Inference audit):

.. code-block:: text

    Index  Field                    Source check / name                    Direction
    ─────  ───────────────────────  ─────────────────────────────────────  ─────────
      7    linkk_rmse               "LinKK Reconstruction Error"           lower ↓
      8    ecm_max_cv               "ECM Parameter Stability (Max CV)"     lower ↓
      9    ecm_kk_self_consistency  "Model Self-Consistency — Simulated Mean" lower ↓
     10    ecm_kk_posterior_frac    "Model Self-Consistency — Posterior Draws" lower ↓
     11    drt_peak_mismatch        "DRT Peak Count Mismatch"              lower ↓

* ``linkk_rmse``: linKK reconstruction RMSE — is the EIS measurement
  Kramers-Kronig consistent (i.e. causal, linear, stable)?  Threshold 0.05.
* ``ecm_max_cv``: maximum coefficient of variation across the MCMC posterior
  — are the ECM parameters well-constrained by the data?  Threshold 0.5.
* ``ecm_kk_self_consistency`` (kkV): KK residual on the ECM-predicted mean
  impedance spectrum — does the fitted circuit model reproduce the physics?
  Threshold 0.05.
* ``ecm_kk_posterior_frac``: fraction of posterior draws whose simulated
  spectra exceed 10% KK residual — posterior uncertainty quality.
* ``drt_peak_mismatch``: absolute difference between the number of peaks in
  the DRT gamma spectrum and the number of RC/CPE elements in the ECM circuit.
  Zero means the model structure captures all physical relaxation processes.
  Compute via :func:`drt_peak_mismatch_result`.

Assembling this vector from a run of :class:`~battery_forecast.audit.AuditPipeline`
is handled by :func:`make_audit_health_vector`.

Both policies add an **explicit health cost** (weight ``gamma_health``) to
the acquisition score.  The health score is::

    h(v) = v[4] − v[1] − v[6]
             − clip(v[7] / 0.05, 0, 1)   # linKK data validity
             − clip(v[9] / 0.05, 0, 1)   # ECM KK self-consistency (kkV)
         = variance_error_corr
             − calibration_error
             − anomalous_fraction
             − normalised_linkk_rmse
             − normalised_ecm_kk_error

Range: [−4, 1].  Perfect health gives ``h = 1``; all-bad experiment gives ``h = −4``.

Subtracting ``γ_health · h(v_predicted_next)`` from the acquisition score
rewards protocols whose linear-dynamics prediction yields a healthier next
experiment state — a direction not captured by the GP uncertainty bonus or
the subspace residual.

See ``state_space_audit_derivation.md`` for the full derivation.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

import numpy as np

from .. import utils
from ..audit.base import AuditCategory, AuditResult
from .state_space import StateSpacePolicy
from .state_space_bayes import BayesianDynamicsPolicy

log = utils.get_logger(__name__, level=logging.INFO)

# ── Canonical audit health vector ─────────────────────────────────────────────

AUDIT_HEALTH_FIELDS: List[str] = [
    # ── Predictive-quality (TSModel + Evaluate audit) ─────────────────────
    "variance_alignment",        # 0: pred_var / true_var ratio; ideal = 1.0
    "calibration_error",         # 1: Kuleshov CE ∈ [0, 1]; lower is better
    "ence",                      # 2: ENCE ≥ 0; lower is better
    "miscalibration_area",       # 3: area under |predicted_pi − observed_pi|; lower is better
    "variance_error_corr",       # 4: Spearman ρ(true_var, |error|) ∈ [-1, 1]; higher is better
    "n_decreasing_channels",     # 5: channels whose uncertainty slope < −1%/step; lower is better
    "anomalous_fraction",        # 6: fraction of steps beyond 3 σ from baseline; lower is better
    # ── EIS-specific inference quality (Inference audit) ──────────────────
    "linkk_rmse",                # 7: linKK RMSE — data KK-consistency; threshold 0.05; lower is better
    "ecm_max_cv",                # 8: MCMC posterior max CV — parameter stability; threshold 0.5; lower is better
    "ecm_kk_self_consistency",   # 9: KK residual on ECM mean spectrum (kkV); threshold 0.05; lower is better
    "ecm_kk_posterior_fraction", # 10: fraction of posterior draws with >10% KK residual; lower is better
    "drt_peak_mismatch",         # 11: |DRT peak count − ECM element count|; 0 = exact match; lower is better
]

AUDIT_HEALTH_VECTOR_DIM: int = len(AUDIT_HEALTH_FIELDS)


def make_audit_health_vector(results: list, fill_value: float = 0.0) -> np.ndarray:
    """Assemble a canonical 12-dim audit health vector from a list of AuditResult objects.

    Covers two audit stages:

    * **Predictive-quality** (indices 0–6): extracted from TSModel and Evaluate
      audit checks (VarianceAlignment, TraitsCalibration, VarianceErrorCorrelation,
      UncertaintyEvolution, UncertaintyAnomalies).
    * **EIS inference quality** (indices 7–11): extracted from Inference audit
      checks (LinKKCheck, ECMStabilityCheck, model self-consistency checks, and the
      DRT peak-count mismatch produced by :func:`drt_peak_mismatch_result`).

    Fields not present in ``results`` are filled with ``fill_value``.

    Parameters
    ----------
    results : list of AuditResult
        Output of any combination of
        :meth:`~battery_forecast.audit.AuditPipeline.run_tsmodel_audit`,
        :meth:`~battery_forecast.audit.AuditPipeline.run_evaluate_audit`, and
        :meth:`~battery_forecast.audit.AuditPipeline.run_inference_audit`,
        plus any :class:`~battery_forecast.audit.base.AuditResult` objects
        returned by :func:`drt_peak_mismatch_result`.
    fill_value : float
        Value to use for checks that did not run or returned ``None``.
        Defaults to 0.0 (neutral / "no information" for all fields).

    Returns
    -------
    np.ndarray, shape ``(12,)``
        Values in the order defined by :data:`AUDIT_HEALTH_FIELDS`.
    """
    v = np.full(AUDIT_HEALTH_VECTOR_DIM, fill_value, dtype=float)
    for r in results:
        if r.value is None and not r.details:
            continue
        name = r.name
        # ── Predictive-quality fields (indices 0–6) ──────────────────────────
        if name == "VarianceAlignment" and r.value is not None:
            v[0] = float(r.value)
        elif name == "TraitsCalibration":
            if r.value is not None:
                v[1] = float(r.value)               # CE
            d = r.details or {}
            if "ence" in d and d["ence"] is not None:
                v[2] = float(d["ence"])
            if "miscalibration_area" in d and d["miscalibration_area"] is not None:
                v[3] = float(d["miscalibration_area"])
        elif name == "VarianceErrorCorrelation" and r.value is not None:
            v[4] = float(r.value)
        elif name == "UncertaintyEvolution" and r.value is not None:
            v[5] = float(r.value)
        elif name == "UncertaintyAnomalies" and r.value is not None:
            v[6] = float(r.value)
        # ── EIS inference quality fields (indices 7–11) ──────────────────────
        elif name == "LinKK Reconstruction Error" and r.value is not None:
            v[7] = float(r.value)
        elif name == "ECM Parameter Stability (Max CV)" and r.value is not None:
            v[8] = float(r.value)
        elif name == "Model Self-Consistency — Simulated Mean" and r.value is not None:
            v[9] = float(r.value)
        elif name == "Model Self-Consistency — Posterior Draws" and r.value is not None:
            v[10] = float(r.value)
        elif name == "DRT Peak Count Mismatch" and r.value is not None:
            v[11] = float(r.value)
    return v


# ── Composite uncertainty vector ───────────────────────────────────────────────

COMPOSITE_UNCERTAINTY_DIM: int = 28
"""Dimension of the composite uncertainty vector returned by
:func:`build_composite_uncertainty_vector`.

Layout
------
* ``[0:18]``  — surrogate predictive std per ECM parameter (18-D)
* ``[18:23]`` — traits-audit check metrics: variance_alignment, calibration_error,
                ence, miscalibration_area, variance_error_corr
* ``[23:28]`` — battery-forecast custom check metrics: conformal_coverage_ratio,
                pit_pvalue, crps_mean, n_decreasing_channels, anomalous_fraction
"""


def build_composite_uncertainty_vector(
    pred_std: Optional[np.ndarray],
    audit_report,
) -> np.ndarray:
    """Assemble a 28-D composite uncertainty vector from three sources.

    Parameters
    ----------
    pred_std : np.ndarray of shape ``(18,)`` or ``None``
        Per-output predictive standard deviation from the surrogate model.
        Zeros when the policy has no surrogate (e.g. Random).
    audit_report : AuditReport or ``None``
        Most recent report from ``_build_al_audit_pipeline()``.  Zeros when
        the pipeline has not yet produced an intermediate report.

    Returns
    -------
    np.ndarray, shape ``(28,)``
        All missing / unavailable values filled with 0.0.
    """
    v = np.zeros(COMPOSITE_UNCERTAINTY_DIM, dtype=float)

    if pred_std is not None:
        arr = np.asarray(pred_std, dtype=float)
        v[:min(18, len(arr))] = arr[:18]

    if audit_report is None:
        return v

    for r in audit_report.results:
        d = r.details or {}
        name = r.name
        if name == "VarianceAlignment" and r.value is not None:
            v[18] = float(r.value)
        elif name == "TraitsCalibration":
            if r.value is not None:
                v[19] = float(r.value)
            if "ence" in d and d["ence"] is not None:
                v[20] = float(d["ence"])
            if "miscalibration_area" in d and d["miscalibration_area"] is not None:
                v[21] = float(d["miscalibration_area"])
        elif name == "VarianceErrorCorrelation" and r.value is not None:
            v[22] = float(r.value)
        elif name == "ConformalCoverage" and r.value is not None:
            v[23] = float(r.value)
        elif name == "PITUniformity" and r.value is not None:
            v[24] = float(r.value)
        elif name == "CRPS" and r.value is not None:
            v[25] = float(r.value)
        elif name == "UncertaintyEvolution" and r.value is not None:
            v[26] = float(r.value)
        elif name == "IntegratedGradients" and r.value is not None:
            v[27] = float(r.value)
    return v


def _health_score(v: np.ndarray) -> float:
    """Scalar health score (higher = better experiment quality).

    Defined as::

        h(v) = variance_error_corr
               − calibration_error
               − anomalous_fraction
               − clip(linkk_rmse / 0.05, 0, 1)
               − clip(ecm_kk_self_consistency / 0.05, 0, 1)

    The five terms and their ranges:

    * ``variance_error_corr`` ∈ [-1, 1]: Spearman ρ between predicted variance
      and absolute error.  High ρ means the model recognises when it is uncertain.
    * ``calibration_error`` ∈ [0, 1]: Kuleshov CE — fraction-coverage error.
    * ``anomalous_fraction`` ∈ [0, 1]: fraction of steps with |z-score| > 3.
    * ``linkk_rmse`` (index 7): linKK reconstruction RMSE, normalised by the
      0.05 acceptance threshold.  Zero penalty for valid data; full penalty (1)
      at the threshold or above.  Tests whether the raw EIS measurement is
      Kramers-Kronig consistent.
    * ``ecm_kk_self_consistency`` (index 9, kkV): KK residual of the ECM-fitted
      mean spectrum, normalised by 0.05.  Tests whether the circuit model
      reproduces the physics of the measurement.

    ``h`` ∈ [-4, 1].  Perfect health gives h = 1; all-bad experiment gives h = -4.

    Indices 7–11 gracefully degrade to 0 (no penalty) when not present in ``v``
    or when the value is ``np.nan``, so the function is backward-compatible with
    7-dimensional vectors from earlier pipeline runs.
    """
    n = len(v)
    rho  = float(v[4]) if n > 4 and np.isfinite(v[4]) else 0.0
    ce   = float(v[1]) if n > 1 and np.isfinite(v[1]) else 0.0
    frac = float(v[6]) if n > 6 and np.isfinite(v[6]) else 0.0

    # EIS validity: normalised by acceptance threshold, capped at 1
    linkk_norm = (
        min(float(v[7]) / 0.05, 1.0)
        if n > 7 and np.isfinite(v[7]) else 0.0
    )
    kk_sc_norm = (
        min(float(v[9]) / 0.05, 1.0)
        if n > 9 and np.isfinite(v[9]) else 0.0
    )

    return rho - ce - frac - linkk_norm - kk_sc_norm


def drt_peak_mismatch_result(
    drt_dict: dict,
    n_ecm_elements: int,
    min_gamma_rel: float = 0.05,
    spectrum_idx: Optional[int] = None,
) -> AuditResult:
    """Create an AuditResult encoding the DRT peak count vs ECM element mismatch.

    Counts significant local maxima in the DRT gamma distribution and compares
    them to the expected number of relaxation processes (RC/CPE/ZARC elements
    in the ECM circuit).  A mismatch of 0 means the circuit structure accounts
    for all physical processes visible in the spectrum.

    The returned ``AuditResult`` has ``name="DRT Peak Count Mismatch"`` and can
    be included in the ``results`` list passed to :func:`make_audit_health_vector`
    to populate index 11 of the health vector.

    Parameters
    ----------
    drt_dict : dict
        ``DRTdict`` returned by
        :func:`~battery_forecast.analysis.drt.get_drt_impedance` — a dict of
        ``np.ndarray([tau, gamma]).T`` per spectrum (keyed by integer index).
    n_ecm_elements : int
        Number of RC/CPE/ZARC elements in the ECM circuit string.  This is the
        expected number of DRT peaks (one peak per independent relaxation process).
    min_gamma_rel : float
        Fraction of the maximum gamma value below which a local maximum is not
        counted as a peak.  Default 0.05 (peaks must be ≥ 5% of the tallest peak).
    spectrum_idx : int, optional
        If given, compute the mismatch only for this index in ``drt_dict``.
        If ``None`` (default), compute the rounded mean mismatch across all spectra.

    Returns
    -------
    AuditResult
        ``name="DRT Peak Count Mismatch"``, ``value=|n_peaks − n_ecm_elements|``,
        ``passed=(value == 0)``.
    """
    try:
        from scipy.signal import find_peaks
    except ImportError as exc:
        raise ImportError("scipy is required for DRT peak counting") from exc

    indices = [spectrum_idx] if spectrum_idx is not None else sorted(drt_dict.keys())
    mismatches = []
    for idx in indices:
        arr = np.asarray(drt_dict[idx])       # shape (N, 2): [tau, gamma]
        gamma = arr[:, 1]
        threshold = float(gamma.max()) * min_gamma_rel
        peaks, _ = find_peaks(gamma, height=threshold)
        mismatches.append(abs(len(peaks) - n_ecm_elements))

    mismatch = int(round(float(np.mean(mismatches)))) if mismatches else 0
    passed = mismatch == 0
    label = "single" if spectrum_idx is not None else f"mean over {len(indices)} spectra"
    return AuditResult(
        name="DRT Peak Count Mismatch",
        passed=passed,
        value=float(mismatch),
        threshold=0.0,
        category=AuditCategory.ALEATORIC_MODEL,
        message=(
            f"|DRT peaks − ECM elements| = {mismatch}  ({label})"
            if not passed
            else f"DRT peaks match ECM elements ({n_ecm_elements})  ({label})"
        ),
        details={
            "n_ecm_elements": n_ecm_elements,
            "mismatches_per_spectrum": mismatches,
            "min_gamma_rel": min_gamma_rel,
        },
    )


# ── AuditStateSpacePolicy ─────────────────────────────────────────────────────

class AuditStateSpacePolicy(StateSpacePolicy):
    """StateSpacePolicy augmented with the audit system health vector.

    Replaces the per-parameter ECM variance augmentation of
    :class:`~battery_forecast.policies.StateSpacePolicy` with the
    12-dimensional **audit health vector** produced by
    :class:`~battery_forecast.audit.AuditPipeline`.

    The augmented belief state is::

        s̃_t = [s_t | uncertainty_weight · v_audit_t]  ∈ ℝ^{n + 12}

    where ``v_audit_t`` encodes variance alignment, calibration error, ENCE,
    miscalibration area, variance-error correlation, decreasing-channel count,
    anomaly fraction, linKK RMSE, ECM max CV, ECM KK self-consistency,
    posterior KK fraction, and DRT peak mismatch at cycle ``t``.

    An additional **health cost** term is added to the acquisition score::

        score = Q · δ(s̃_next[:n])
              + R · ‖a‖²
              − κ · σ̄_GP
              − γ_health · h(v_audit_next)

    where ``v_audit_next`` is the predicted next audit health vector extracted
    from the linear-dynamics prediction of the augmented state, and
    ``h(v) = rho − CE − anomaly_frac`` (see :func:`_health_score`).

    The :meth:`make_audit_health_vector` class method assembles ``v_audit``
    from a list of :class:`~battery_forecast.audit.AuditResult` objects.

    Parameters
    ----------
    gamma_health : float
        Weight on the audit health bonus.  Positive values encourage the
        policy to select protocols that are predicted to maintain or improve
        experiment health.  Set to 0.0 to disable (equivalent to the base
        :class:`~battery_forecast.policies.StateSpacePolicy` except for the
        different augmentation vector).
    All other parameters are inherited from StateSpacePolicy.
    """

    def __init__(
        self,
        model,
        n_components: int = 10,
        kappa: float = 1.0,
        Q_weight: float = 1.0,
        R_weight: float = 0.01,
        C_rate_bonus: float = 0.0,
        gamma_health: float = 0.5,
        min_fit_obs: int = 5,
        num_action_features: int = 6,
        uncertainty_weight: float = 1.0,
        degradation_fn: Optional[Callable[[np.ndarray], float]] = None,
        scaler=None,
    ) -> None:
        super().__init__(
            model=model,
            n_components=n_components,
            kappa=kappa,
            Q_weight=Q_weight,
            R_weight=R_weight,
            C_rate_bonus=C_rate_bonus,
            min_fit_obs=min_fit_obs,
            num_action_features=num_action_features,
            uncertainty_weight=uncertainty_weight,
            degradation_fn=degradation_fn,
            scaler=scaler,
        )
        self.gamma_health = gamma_health

    # ── Health vector interface ───────────────────────────────────────────

    @staticmethod
    def make_audit_health_vector(results: list, fill_value: float = 0.0) -> np.ndarray:
        """Assemble the canonical 12-dim audit health vector from AuditResult objects.

        Convenience wrapper around the module-level :func:`make_audit_health_vector`.
        Typical usage::

            results = pipeline.run_tsmodel_audit(...)
            results += pipeline.run_evaluate_audit(...)
            v_audit = AuditStateSpacePolicy.make_audit_health_vector(results)
            policy.observe(state, action, uncertainty_vector=v_audit)
        """
        return make_audit_health_vector(results, fill_value=fill_value)

    # ── Scoring override ──────────────────────────────────────────────────

    def _score_candidate(
        self,
        current_aug: np.ndarray,
        current_state: np.ndarray,
        protocol: np.ndarray,
        prot_scale: Optional[np.ndarray] = None,
    ) -> float:
        """LQR score + explicit health cost on the predicted next audit state."""
        protocol = np.asarray(protocol)

        x_scaled = self._build_input(current_state, protocol)
        _, std = self._predict(x_scaled)
        uncertainty = float(np.mean(std[0])) if std is not None else 0.0

        z = current_aug @ self._U_r
        z_next = self._A_r @ z + self._B_r @ protocol
        s_aug_next = self._U_r @ z_next

        n = self._n_state if self._n_state is not None else s_aug_next.shape[0]
        s_next = s_aug_next[:n]

        state_cost = self.Q_weight * self.degradation_fn(s_next)
        p_norm = protocol / prot_scale if prot_scale is not None else protocol
        control_cost = self.R_weight * float(np.dot(p_norm, p_norm))
        uncertainty_bonus = self.kappa * uncertainty
        c_rate_bonus = (
            self.C_rate_bonus * float(p_norm[0])
            if self.C_rate_bonus != 0.0
            else 0.0
        )

        # Health cost: penalise poor predicted next audit health
        health_bonus = 0.0
        if self.gamma_health != 0.0 and len(s_aug_next) > n:
            v_pred = s_aug_next[n:].copy()
            if self.uncertainty_weight > 0:
                v_pred /= self.uncertainty_weight
            health_bonus = self.gamma_health * _health_score(v_pred)

        return state_cost + control_cost - uncertainty_bonus - c_rate_bonus - health_bonus


# ── AuditBayesianDynamicsPolicy ───────────────────────────────────────────────

class AuditBayesianDynamicsPolicy(BayesianDynamicsPolicy):
    """BayesianDynamicsPolicy augmented with the audit system health vector.

    Replaces the per-parameter ECM variance augmentation of
    :class:`~battery_forecast.policies.BayesianDynamicsPolicy` with the
    12-dimensional **audit health vector** produced by
    :class:`~battery_forecast.audit.AuditPipeline`.

    The augmented belief state is::

        s̃_t = [s_t | uncertainty_weight · v_audit_t]  ∈ ℝ^{n + 12}

    An additional **health cost** term is added to the acquisition score::

        score = Q · δ(s̃_next[:n])
              + R · ‖a‖²
              − κ · σ̄_GP
              − γ_info · h(z, a) / (1 + h(z, a))
              − γ_sub · ρ(ŝ_GP)
              − γ_health · h_score(v_audit_next)

    where ``v_audit_next`` is extracted from the linear-dynamics prediction
    of the augmented state and
    ``h_score(v) = rho − CE − anomaly_frac`` (see :func:`_health_score`).

    Parameters
    ----------
    gamma_health : float
        Weight on the audit health bonus.  Set to 0.0 to fall back to the
        behaviour of :class:`~battery_forecast.policies.BayesianDynamicsPolicy`.
    All other parameters are inherited from BayesianDynamicsPolicy.
    """

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
        gamma_health: float = 0.5,
        lambda_reg: float = 1e-3,
        min_fit_obs: int = 5,
        reanchor_freq: Optional[int] = None,
        num_action_features: int = 6,
        uncertainty_weight: float = 1.0,
        degradation_fn: Optional[Callable[[np.ndarray], float]] = None,
        scaler=None,
    ) -> None:
        super().__init__(
            model=model,
            n_components=n_components,
            kappa=kappa,
            Q_weight=Q_weight,
            R_weight=R_weight,
            C_rate_bonus=C_rate_bonus,
            gamma_info=gamma_info,
            gamma_sub=gamma_sub,
            lambda_reg=lambda_reg,
            min_fit_obs=min_fit_obs,
            reanchor_freq=reanchor_freq,
            num_action_features=num_action_features,
            uncertainty_weight=uncertainty_weight,
            degradation_fn=degradation_fn,
            scaler=scaler,
        )
        self.gamma_health = gamma_health

    # ── Health vector interface ───────────────────────────────────────────

    @staticmethod
    def make_audit_health_vector(results: list, fill_value: float = 0.0) -> np.ndarray:
        """Assemble the canonical 12-dim audit health vector from AuditResult objects.

        See :meth:`~AuditStateSpacePolicy.make_audit_health_vector` for usage.
        """
        return make_audit_health_vector(results, fill_value=fill_value)

    # ── Scoring override ──────────────────────────────────────────────────

    def _score_candidate(
        self,
        current_aug: np.ndarray,
        current_state: np.ndarray,
        protocol: np.ndarray,
        prot_scale: Optional[np.ndarray] = None,
    ) -> float:
        """Full composite score including audit health bonus on the predicted next state."""
        protocol = np.asarray(protocol)

        x_scaled = self._build_input(current_state, protocol)
        mean_pred, std_pred = self._predict(x_scaled)
        gp_std = float(np.mean(std_pred[0])) if std_pred is not None else 0.0
        s_gp = (
            self._unscale_state(mean_pred[0])
            if mean_pred is not None
            else current_state
        )

        z = current_aug @ self._U_r
        z_next = self._A_r @ z + self._B_r @ protocol
        s_aug_next = self._U_r @ z_next

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

        health_bonus = 0.0
        if self.gamma_health != 0.0 and len(s_aug_next) > n:
            v_pred = s_aug_next[n:].copy()
            if self.uncertainty_weight > 0:
                v_pred /= self.uncertainty_weight
            health_bonus = self.gamma_health * _health_score(v_pred)

        return (
            state_cost + control_cost
            - gp_bonus - info_bonus - sub_bonus - c_rate_bonus - health_bonus
        )
