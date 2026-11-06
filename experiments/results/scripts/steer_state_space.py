#!/usr/bin/env python
"""Steer active learning on the uncertainty state vector.

Runs the campaigns of ``paper/steering_experiment_spec.md`` for one problem and
one starting condition:

* problems come from ``results.committee.problems`` (forrester,
  branin-currin-mo, color, and forrester-drift: Forrester measured by an
  instrument that heats up with use, forrester-wear: one damaged by use that
  never recovers, and branin-currin-beam: a 2-D sample mapped by an X-ray beam
  that damages the spots it measures; the instrument's state is logged and
  added to the policy state, and on the beam problem the dose already absorbed
  at each candidate is part of the action);
* the starting condition decides the surrogate: a GP whose reported sigma is
  scaled (calibrated x1, overconfident x0.3, underconfident x2.5) or a
  bootstrap polynomial ensemble (misspecified);
* every step, thirteen traits-audit checks are scored on a fixed reference
  grid and z-scored against the warm start, giving the uncertainty state
  vector: ten that use only (y, mu, sigma), and three system metrics (DFR,
  MRF, IWF) that need refits of the surrogate or a downstream decision;
* the policy sees only the *steering* metrics; the *audit-only* metrics are
  recorded to detect Goodharting.

Writes one row per (policy, seed, step) to ``<out>/steps.csv`` plus
``<out>/config.json``; analyse with ``analyze_steering.py``.

Usage::

    python experiments/results/scripts/steer_state_space.py --problem forrester --condition calibrated
    python experiments/results/scripts/steer_state_space.py --problem color --condition overconfident \\
        --seeds 5 --seed-start 10 --out _results/steering/color/overconfident/seeds10
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from pathlib import Path

import numpy as np
from results.cal_demo import _make_gp_kernel
from results.committee.problems import (
    BeamDamageBraninCurrin,
    DriftingInstrumentForrester,
    PolyBootstrapSurrogate,
    WearingInstrumentForrester,
    get_problem,
)
from results.policies import (
    AuditBayesianDynamicsPolicy,
    AuditStateSpacePolicy,
    BayesianDynamicsPolicy,
    LCBPolicy,
    StateSpacePolicy,
)
from scipy.stats import norm
from traits_audit import checks as C

PROBLEMS = ["forrester", "branin-currin-mo", "color", "forrester-drift", "forrester-wear", "branin-currin-beam"]
CONDITIONS = ["calibrated", "overconfident", "underconfident", "misspecified"]
SIGMA_SCALE = {"calibrated": 1.0, "overconfident": 0.3, "underconfident": 2.5, "misspecified": 1.0}
# Classic Bayesian-optimisation rules: they score the surrogate directly and
# learn nothing about the experiment's dynamics.
CLASSIC_POLICIES = ["expected-improvement", "probability-of-improvement", "thompson", "max-sigma"]
# Needs a problem with a dose map; not in the default arm list.
DOSE_POLICIES = ["dose-aware-lcb"]
POLICIES = ["random", "lcb", *CLASSIC_POLICIES, "state-space", "bayes-dynamics",
            "audit-state-space", "audit-bayes", "audit-state-space-all"]


# ── Uncertainty metrics ───────────────────────────────────────────────────────

METRICS = {
    "CalibrationError": C.CalibrationErrorCheck,
    "NLL": C.NegativeLogLikelihoodCheck,
    "VarianceErrorCorrelation": C.VarianceErrorCorrelationCheck,
    "SignedBias": C.SignedBiasCheck,
    "IntervalCoverage": C.IntervalCoverageCheck,
    "IntervalScore": C.IntervalScoreCheck,
    "CRPS": C.CRPSCheck,
    "ENCE": C.ENCECheck,
    "PITUniformity": C.PITUniformityCheck,
    "VarianceAlignment": C.VarianceAlignmentCheck,
}
# System metrics need more than (y, mu, sigma): refits of the surrogate (MRF, IWF)
# or a downstream decision (DFR).  They are computed by score_system_metrics.
SYSTEM_METRICS = ["DecisionFlipRate", "MisspecificationResidualFloor", "ImprecisionWidthFraction"]
ALL_METRICS = list(METRICS) + SYSTEM_METRICS
DEFAULT_STEERING = ["CalibrationError", "NLL", "VarianceErrorCorrelation", "SignedBias"]


def badness(name, result):
    """Turn a check result into one number where higher always means worse."""
    if result.value is None:
        return np.nan
    if name == "VarianceErrorCorrelation":
        return -result.value
    if name == "SignedBias":
        return abs(result.value)
    if name == "IntervalCoverage":
        return abs(result.value - 0.683)
    if name == "PITUniformity":
        return result.details["ks_statistic"]
    if name == "VarianceAlignment":
        return abs(np.log(result.value))
    return result.value


def score_metrics_one_objective(y_ref, mu_ref, sd_ref):
    history = []
    for y, mu, sd in zip(y_ref, mu_ref, sd_ref, strict=True):
        history.append({"y_true": float(y), "y_pred_mean": float(mu),
                        "y_pred_std": float(sd), "uncertainty": float(sd)})
    values = {}
    categories = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, check_class in METRICS.items():
            result = check_class().run(history)
            value = badness(name, result)
            if not np.isfinite(value):
                value = 0.0
            values[name] = float(value)
            categories[name] = result.category.value
    return values, categories


def score_metrics(Y_ref, MU_ref, SD_ref):
    """Badness of every metric on the reference grid, averaged over objectives."""
    n_objectives = Y_ref.shape[1]
    totals = dict.fromkeys(METRICS, 0.0)
    categories = {}
    for j in range(n_objectives):
        values, categories = score_metrics_one_objective(Y_ref[:, j], MU_ref[:, j], SD_ref[:, j])
        for name in METRICS:
            totals[name] += values[name] / n_objectives
    return totals, categories


def predicted_minimiser(prediction):
    """The decision DFR resamples: which reference point the surrogate says is lowest."""
    return int(np.argmin(prediction))


def fit_one_model(problem, condition, X, y, n_members, seed, kernel):
    """Refit a single-objective surrogate of the condition's kind; return its predict function.

    GP refits reuse ``kernel``, the main surrogate's fitted kernel, without
    re-optimising it: the refits then measure how the prediction varies with
    the data, not with re-tuning the hyperparameters.
    """
    if condition == "misspecified":
        model = PolynomialEnsemble(problem.dim, problem.degree, n_members, seed)
        model.fit(X, y, refit_number=0)
    else:
        model = ScaledGP(SIGMA_SCALE[condition], seed, fixed_kernel=kernel)
        model.fit(X, y)
    return model.predict


def score_system_metrics_one_objective(problem, condition, X, y, grid, y_ref, mu_ref, sd_ref, kernel,
                                       args, seed):
    rng = np.random.default_rng(seed)

    def fit_fn(X_fit, y_fit, seed=0):
        return fit_one_model(problem, condition, X_fit, y_fit, args.members, seed, kernel)

    ensemble_means = []
    ensemble_stds = []
    for _ in range(args.ensemble_size):
        rows = rng.integers(0, len(y), size=len(y))
        predict = fit_fn(X[rows], y[rows], seed=int(rng.integers(2**31 - 1)))
        mu, sd = predict(grid)
        ensemble_means.append(mu)
        ensemble_stds.append(sd)

    check_seed = int(rng.integers(2**31 - 1))
    results = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results["DecisionFlipRate"] = C.DecisionFlipRateCheck(n_resamples=100, seed=check_seed).run(
            [], decision_fn=predicted_minimiser, y_pred_mean=mu_ref, y_pred_std=sd_ref)
        results["MisspecificationResidualFloor"] = C.MisspecificationResidualFloorCheck(
            subset_fracs=(0.25, 0.5, 0.75, 1.0), reps=args.mrf_reps, seed=check_seed,
            n_bootstrap=0).run(
            [], fit_fn=fit_fn, X_train=X, y_train=y, X_eval=grid, y_eval=y_ref)
        results["ImprecisionWidthFraction"] = C.ImprecisionWidthFractionCheck().run(
            [], y_pred_ensemble=np.array(ensemble_means), y_pred_std_ensemble=np.array(ensemble_stds),
            y_pred_mean=mu_ref, y_pred_std=sd_ref)

    values = {}
    categories = {}
    for name, result in results.items():
        value = badness(name, result)
        if not np.isfinite(value):
            value = 0.0
        values[name] = float(value)
        categories[name] = result.category.value
    return values, categories


def score_system_metrics(problem, condition, X, Y, grid, Y_ref, MU_ref, SD_ref, kernels, args, seed):
    """Badness of every system metric, averaged over objectives (higher is worse).

    ``kernels``: the main surrogate's fitted kernel for each objective (None for
    the polynomial surrogate).
    """
    n_objectives = Y_ref.shape[1]
    totals = dict.fromkeys(SYSTEM_METRICS, 0.0)
    categories = {}
    for j in range(n_objectives):
        values, categories = score_system_metrics_one_objective(
            problem, condition, X, Y[:, j], grid, Y_ref[:, j], MU_ref[:, j], SD_ref[:, j],
            kernels[j], args, seed=[*seed, j])
        for name in SYSTEM_METRICS:
            totals[name] += values[name] / n_objectives
    return totals, categories


def choose_steering(split, split_index, explicit_list):
    if explicit_list:
        return list(explicit_list)
    if split == "default":
        return list(DEFAULT_STEERING)
    if split == "all":
        return list(ALL_METRICS)
    if split == "random":
        names = sorted(ALL_METRICS)
        order = np.random.default_rng([split_index, 99]).permutation(len(names))
        half = len(names) // 2
        return [names[i] for i in sorted(order[:half])]
    raise ValueError(f"unknown split {split!r}")


class WarmStartBaseline:
    """Z-scores each metric against its own values during the warm start.

    ``z_floor``: the std is never smaller than ``z_floor * |warm-start mean|``.
    ``z_clip``: if given, every z-score is limited to [-z_clip, +z_clip].
    """

    def __init__(self, warm_snapshots, z_floor, z_clip):
        self.z_clip = z_clip
        self.mean = {}
        self.std = {}
        for name in ALL_METRICS:
            values = np.array([snapshot[name] for snapshot in warm_snapshots])
            self.mean[name] = float(values.mean())
            self.std[name] = max(float(values.std()), z_floor * abs(float(values.mean())) + 1e-9)

    def z_scores(self, snapshot):
        scores = {}
        for name in ALL_METRICS:
            z = (snapshot[name] - self.mean[name]) / self.std[name]
            if self.z_clip is not None:
                z = min(max(z, -self.z_clip), self.z_clip)
            scores[name] = z
        return scores


# ── Surrogates for the four starting conditions ───────────────────────────────

class ScaledGP:
    """GP with the cal_demo kernel; the reported sigma is multiplied by ``sigma_scale``.

    With ``fixed_kernel`` the kernel hyperparameters are not optimised: the GP
    uses that (already fitted) kernel as it is.
    """

    def __init__(self, sigma_scale, seed, fixed_kernel=None):
        self.sigma_scale = sigma_scale
        self.seed = seed
        self.fixed_kernel = fixed_kernel
        self.gp = None

    def fit(self, x, y):
        from sklearn.gaussian_process import GaussianProcessRegressor
        if self.fixed_kernel is None:
            self.gp = GaussianProcessRegressor(kernel=_make_gp_kernel(), n_restarts_optimizer=1,
                                               normalize_y=True, random_state=self.seed)
        else:
            self.gp = GaussianProcessRegressor(kernel=self.fixed_kernel, optimizer=None,
                                               normalize_y=True, random_state=self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gp.fit(x, y)

    def predict(self, x):
        mu, sd = self.gp.predict(x, return_std=True)
        return mu, sd * self.sigma_scale

    def sample(self, x, rng):
        """One joint draw from the predictive distribution, with sigma scaled like predict."""
        draw = self.gp.sample_y(x, random_state=int(rng.integers(2**31 - 1)))[:, 0]
        mu = self.gp.predict(x)
        return mu + self.sigma_scale * (draw - mu)


class PolynomialEnsemble:
    """Bootstrap polynomial ensemble: the misspecified surrogate."""

    def __init__(self, dim, degree, n_members, seed):
        self.dim = dim
        self.degree = degree
        self.n_members = n_members
        self.seed = seed

    def fit(self, x, y, refit_number):
        rng = np.random.default_rng([self.seed, 2, refit_number])
        self.model = PolyBootstrapSurrogate(dim=self.dim, degree=self.degree,
                                            n_estimators=self.n_members, rng=rng)
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def sample(self, x, rng):
        """One joint draw: the prediction of one randomly chosen bootstrap member."""
        members = self.model.predict_members(x)
        return members[:, rng.integers(members.shape[1])]


class Surrogate:
    """One model per objective. ``fit`` takes Y of shape (n, n_objectives)."""

    def __init__(self, problem, condition, n_members, seed):
        self.condition = condition
        self.models = []
        for j in range(problem.n_objectives):
            if condition == "misspecified":
                self.models.append(PolynomialEnsemble(problem.dim, problem.degree, n_members, seed * 10 + j))
            else:
                self.models.append(ScaledGP(SIGMA_SCALE[condition], seed * 10 + j))

    def fit(self, X, Y, refit_number):
        for j, model in enumerate(self.models):
            if self.condition == "misspecified":
                model.fit(X, Y[:, j], refit_number)
            else:
                model.fit(X, Y[:, j])

    def sample(self, X, rng):
        """One joint draw from the surrogate on X, shape (n_points, n_objectives)."""
        return np.column_stack([model.sample(X, rng) for model in self.models])

    def fitted_kernels(self):
        """Each objective's fitted GP kernel, or None for the polynomial surrogate."""
        if self.condition == "misspecified":
            return [None] * len(self.models)
        return [model.gp.kernel_ for model in self.models]

    def predict(self, X):
        means = []
        stds = []
        for model in self.models:
            mu, sd = model.predict(X)
            means.append(mu)
            stds.append(sd)
        return np.column_stack(means), np.column_stack(stds)


def make_problem(name):
    if name == "forrester-drift":
        return DriftingInstrumentForrester()
    if name == "forrester-wear":
        return WearingInstrumentForrester()
    if name == "branin-currin-beam":
        return BeamDamageBraninCurrin()
    return get_problem(name)


def has_instrument(problem):
    """True for problems whose instrument state is logged (temperature, wear or dose)."""
    return hasattr(problem, "instrument_reading")


def has_dose_map(problem):
    """True for problems that know the dose already absorbed at every position."""
    return hasattr(problem, "dose_at")


def dose_feature(problem, x):
    """Dose already absorbed at x, in units of the half dose (0 for problems without a dose map)."""
    if not has_dose_map(problem):
        return 0.0
    return float(problem.dose_at(np.asarray(x).reshape(1, -1))[0] / problem.half_dose)


def action_size(problem):
    """Action u = [x, mu(x), sigma(x)], plus the dose at x on problems with a dose map."""
    return problem.dim + 3 if has_dose_map(problem) else problem.dim + 2


# ── Campaign score: simple regret, or log hypervolume difference ──────────────

def query_regret(problem, clean_value):
    """Regret of one query: its clean objective minus the optimum (NaN for multi-objective)."""
    if problem.n_objectives != 1:
        return np.nan
    return float(clean_value[0] - problem.true_min)


def campaign_score(problem, clean_values):
    """``clean_values``: noise-free objectives of every point queried so far."""
    clean_values = np.asarray(clean_values)
    if problem.n_objectives == 1:
        return float(clean_values[:, 0].min() - problem.true_min)
    hypervolume = problem.hypervolume(clean_values)
    return float(np.log10(max(problem.hv_max - hypervolume, 1e-12)))


# ── Policies ──────────────────────────────────────────────────────────────────
# The policies score candidates through ``model.predict_with_uncertainty`` on
# [state | action] rows.  Policy state s = [best scalarised y so far, mean sigma
# on the reference grid]; action u = [x_1..x_d, mu(x), sigma(x)], with mu and
# sigma averaged over objectives.  The "model" predicts the next best-so-far
# as min(best, mu(x)) with uncertainty sigma(x), so the LCB fallback is an
# ordinary LCB on the surrogate.  On a problem with a logged instrument the
# state also holds the instrument reading, which the model passes through.


def state_size(problem):
    return 3 if has_instrument(problem) else 2


class SurrogateView:
    def __init__(self, dim, n_state):
        self.dim = dim
        self.n_state = n_state

    def predict_with_uncertainty(self, X):
        X = np.atleast_2d(X)
        state = X[:, :self.n_state]
        mu_of_query = X[:, self.n_state + self.dim]
        sigma_of_query = X[:, self.n_state + self.dim + 1]
        mean = state.copy()
        mean[:, 0] = np.minimum(state[:, 0], mu_of_query)
        std = np.column_stack([sigma_of_query] * self.n_state)
        return {"mean": mean, "std": std}


def progress(state):
    return float(state[0])


def make_health_function(scale):
    """Health of a vector of badness z-scores: higher is healthier, always in [-1, 1].

    Each z-score passes through tanh(z / scale), so a metric counts as fully
    bad or fully good once it is a few ``scale`` units from its warm-start
    level, while smaller changes still move the score smoothly.
    """
    def health(badness_z):
        return -float(np.mean(np.tanh(np.asarray(badness_z) / scale)))
    return health


def make_policy(name, problem, args):
    common = dict(model=SurrogateView(problem.dim, state_size(problem)), num_action_features=action_size(problem),
                  degradation_fn=progress)
    dynamics = dict(common, n_components=args.rank, kappa=args.kappa, Q_weight=1.0, R_weight=0.0,
                    min_fit_obs=args.min_fit_obs, uncertainty_weight=1.0)
    if name == "random" or name in CLASSIC_POLICIES or name in DOSE_POLICIES:
        return None
    if name == "lcb":
        return LCBPolicy(kappa=args.kappa, **common)
    if name == "state-space":
        return StateSpacePolicy(**dynamics)
    if name == "bayes-dynamics":
        return BayesianDynamicsPolicy(gamma_info=args.gamma_info, gamma_sub=0.0, **dynamics)
    if name in ("audit-state-space", "audit-state-space-all"):
        return AuditStateSpacePolicy(gamma_health=args.gamma_health,
                                     health_fn=make_health_function(args.health_scale),
                                     **dynamics)
    if name == "audit-bayes":
        return AuditBayesianDynamicsPolicy(gamma_info=args.gamma_info, gamma_sub=0.0,
                                           gamma_health=args.gamma_health,
                                           health_fn=make_health_function(args.health_scale),
                                           **dynamics)
    raise ValueError(name)


# ── One campaign ──────────────────────────────────────────────────────────────

class Campaign:
    """Everything one (problem, condition, policy, seed) run needs."""

    def __init__(self, problem, condition, policy_name, seed, args, steering):
        self.problem = problem
        self.policy_name = policy_name
        self.seed = seed
        self.args = args
        if policy_name == "audit-state-space-all":
            self.policy_sees = list(ALL_METRICS)
        else:
            self.policy_sees = steering
        self.grid = problem.grid
        self.y_scales = problem.y_scales
        self.surrogate = Surrogate(problem, condition, args.members, seed)
        self.policy = make_policy(policy_name, problem, args)
        self.stateful = isinstance(self.policy, (StateSpacePolicy, BayesianDynamicsPolicy))

        reference_rng = np.random.default_rng([seed, 0])
        if hasattr(problem, "observe_at_rest"):
            self.Y_ref = problem.observe_at_rest(self.grid, reference_rng) / self.y_scales
        else:
            self.Y_ref = problem.observe(self.grid, reference_rng) / self.y_scales
        self.x_observed = []
        self.y_observed = []
        self.clean_observed = []
        self.instrument_history = []
        self.dose_at_query = []
        warm_start_rng = np.random.default_rng([seed, 5])
        for _ in range(args.n_init):
            self.query(warm_start_rng.uniform(0.0, 1.0, problem.dim))

        warm_Y = np.array(self.y_observed)
        self.y_mean = warm_Y.mean(axis=0)
        self.y_std = warm_Y.std(axis=0)
        self.y_std[self.y_std == 0] = 1.0

    def query(self, x):
        step = len(self.x_observed)
        noise_rng = np.random.default_rng([self.seed, 1, step])
        x_row = np.asarray(x, dtype=float).reshape(1, -1)
        self.x_observed.append(x_row[0])
        self.dose_at_query.append(dose_feature(self.problem, x_row[0]))
        self.y_observed.append(self.problem.observe(x_row, noise_rng)[0] / self.y_scales)
        self.clean_observed.append(self.problem.clean(x_row)[0])
        if has_instrument(self.problem):
            self.instrument_history.append(self.problem.instrument_reading)

    def scalarise(self, values_per_objective):
        return float(np.mean((values_per_objective - self.y_mean) / self.y_std))

    def refit_and_measure(self, n_points):
        """Refit on the first ``n_points`` observations; return metrics and policy state."""
        X = np.array(self.x_observed[:n_points])
        Y = np.array(self.y_observed[:n_points])
        self.surrogate.fit(X, Y, refit_number=n_points)
        MU_ref, SD_ref = self.surrogate.predict(self.grid)
        snapshot, categories = score_metrics(self.Y_ref, MU_ref, SD_ref)
        system_snapshot, system_categories = score_system_metrics(
            self.problem, self.surrogate.condition, X, Y, self.grid, self.Y_ref, MU_ref, SD_ref,
            self.surrogate.fitted_kernels(), self.args, seed=[self.seed, 6, n_points])
        snapshot.update(system_snapshot)
        categories.update(system_categories)
        best_so_far = min(self.scalarise(y) for y in Y)
        mean_sigma = float(np.mean(SD_ref / self.y_std))
        state = [best_so_far, mean_sigma]
        if has_instrument(self.problem):
            state.append(self.instrument_history[n_points - 1])
        state = np.array(state)
        return snapshot, categories, state

    def candidate_actions(self):
        MU, SD = self.surrogate.predict(self.grid)
        actions = []
        for x, mu, sd in zip(self.grid, MU, SD, strict=True):
            mu_scalar = self.scalarise(mu)
            sigma_scalar = float(np.mean(sd / self.y_std))
            action = [mu_scalar, sigma_scalar]
            if has_dose_map(self.problem):
                action.append(dose_feature(self.problem, x))
            actions.append(np.concatenate([x, action]))
        return actions

    def classic_choice(self, candidates, state, step):
        """Index of the candidate a classic Bayesian-optimisation rule picks (minimisation).

        Uses the same scalarised, standardised mu and sigma of each candidate as
        the other policies; ``state[0]`` is the best scalarised value seen so far.
        """
        mu = np.array([candidate[self.problem.dim] for candidate in candidates])
        sigma = np.array([candidate[self.problem.dim + 1] for candidate in candidates])
        sigma = np.maximum(sigma, 1e-12)
        best = state[0]
        if self.policy_name == "max-sigma":
            return int(np.argmax(sigma))
        if self.policy_name == "expected-improvement":
            z = (best - mu) / sigma
            improvement = (best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return int(np.argmax(improvement))
        if self.policy_name == "probability-of-improvement":
            z = (best - mu - self.args.pi_margin) / sigma
            return int(np.argmax(norm.cdf(z)))
        if self.policy_name == "dose-aware-lcb":
            if not has_dose_map(self.problem):
                raise ValueError("dose-aware-lcb needs a problem with a dose map")
            dose = np.array([candidate[self.problem.dim + 2] for candidate in candidates])
            return int(np.argmin(mu - self.args.kappa * sigma + self.args.dose_penalty * dose))
        if self.policy_name == "thompson":
            rng = np.random.default_rng([self.seed, 7, step])
            draw = self.surrogate.sample(self.grid, rng)
            return int(np.argmin([self.scalarise(row) for row in draw]))
        raise ValueError(self.policy_name)

    def action_for(self, x):
        MU, SD = self.surrogate.predict(np.asarray(x).reshape(1, -1))
        action = [self.scalarise(MU[0]), float(np.mean(SD[0] / self.y_std))]
        if has_dose_map(self.problem):
            action.append(dose_feature(self.problem, x))
        return np.concatenate([x, action])

    def row(self, n_points, phase, probe, snapshot, baseline):
        z = baseline.z_scores(snapshot)
        row = {"problem": self.problem.name, "condition": self.surrogate.condition,
               "policy": self.policy_name, "seed": self.seed, "step": n_points,
               "phase": phase, "probe": int(probe)}
        last_x = self.x_observed[n_points - 1]
        for i in range(self.problem.dim):
            row[f"x{i}"] = float(last_x[i])
        row["score"] = campaign_score(self.problem, self.clean_observed[:n_points])
        row["query_regret"] = query_regret(self.problem, self.clean_observed[n_points - 1])
        if has_instrument(self.problem):
            row["instrument"] = self.instrument_history[n_points - 1]
        if has_dose_map(self.problem):
            row["dose_at_query"] = self.dose_at_query[n_points - 1]
        for name in ALL_METRICS:
            row[f"raw:{name}"] = snapshot[name]
            row[f"z:{name}"] = z[name]
        return row

    def run(self):
        args = self.args
        first_fit = self.problem.degree + 2
        warm = []
        for n_points in range(first_fit, args.n_init + 1):
            snapshot, categories, state = self.refit_and_measure(n_points)
            warm.append((n_points, snapshot, state))
        baseline = WarmStartBaseline([snapshot for _, snapshot, _ in warm], args.z_floor, args.z_clip)

        def visible_vector(snapshot):
            z = baseline.z_scores(snapshot)
            return np.array([z[name] for name in self.policy_sees])

        if self.stateful:
            for i in range(len(warm) - 1):
                n_points, snapshot, state = warm[i]
                X = np.array(self.x_observed[:n_points])
                self.surrogate.fit(X, np.array(self.y_observed[:n_points]), refit_number=n_points)
                next_x = self.x_observed[n_points]
                self.policy.observe(state, self.action_for(next_x),
                                    uncertainty_vector=visible_vector(snapshot))

        rows = []
        for n_points, snapshot, _ in warm:
            rows.append(self.row(n_points, "warm", False, snapshot, baseline))

        snapshot, categories, state = self.refit_and_measure(len(self.x_observed))
        for step in range(len(self.x_observed), args.steps):
            candidates = self.candidate_actions()
            visible = visible_vector(snapshot)
            probe = np.random.default_rng([self.seed, 3, step]).random() < args.probe
            if self.policy_name == "random" or probe:
                pick = np.random.default_rng([self.seed, 4, step]).integers(len(candidates))
                action = candidates[int(pick)]
            elif self.policy_name in CLASSIC_POLICIES or self.policy_name in DOSE_POLICIES:
                action = candidates[self.classic_choice(candidates, state, step)]
            elif self.stateful:
                action, _, _ = self.policy.select_next(state, candidates, uncertainty_vector=visible)
            else:
                action, _, _ = self.policy.select_next(state, candidates)
            self.query(action[:self.problem.dim])
            if self.stateful:
                self.policy.observe(state, action, uncertainty_vector=visible)
            snapshot, categories, state = self.refit_and_measure(len(self.x_observed))
            rows.append(self.row(len(self.x_observed), "steer", probe, snapshot, baseline))
        return rows, categories


# ── Command line ──────────────────────────────────────────────────────────────

def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--problem", default="forrester", choices=PROBLEMS)
    p.add_argument("--condition", default="calibrated", choices=CONDITIONS)
    p.add_argument("--policies", nargs="+", default=POLICIES, choices=POLICIES + DOSE_POLICIES)
    p.add_argument("--split", default="default", choices=["default", "random", "all"])
    p.add_argument("--split-index", type=int, default=0, help="which random split (--split random)")
    p.add_argument("--steering", nargs="+", choices=ALL_METRICS,
                   help="explicit steering metrics; overrides --split")
    p.add_argument("--seeds", type=int, default=10, help="number of seeds")
    p.add_argument("--seed-start", type=int, default=0, help="first seed (for SLURM array tasks)")
    p.add_argument("--steps", type=int, default=100, help="total oracle queries per campaign")
    p.add_argument("--n-init", type=int, default=24, help="random warm-start queries (>= degree + 3)")
    p.add_argument("--probe", type=float, default=0.1, help="random probe fraction")
    p.add_argument("--members", type=int, default=20, help="ensemble size (misspecified condition)")
    p.add_argument("--ensemble-size", type=int, default=5,
                   help="bootstrap refits per step for the IWF credal set")
    p.add_argument("--mrf-reps", type=int, default=2,
                   help="refits per subset size for the MRF learning curve")
    p.add_argument("--kappa", type=float, default=1.0)
    p.add_argument("--dose-penalty", type=float, default=1.0,
                   help="dose-aware-lcb: penalty per half dose already absorbed at a candidate")
    p.add_argument("--pi-margin", type=float, default=0.01,
                   help="improvement margin for probability-of-improvement (standardised units)")
    p.add_argument("--rank", type=int, default=4, help="SVD rank of the learned dynamics")
    p.add_argument("--min-fit-obs", type=int, default=5)
    p.add_argument("--gamma-health", type=float, default=0.5)
    p.add_argument("--health-scale", type=float, default=2.0,
                   help="z-score scale k of the health term -mean(tanh(z / k))")
    p.add_argument("--gamma-info", type=float, default=0.5)
    p.add_argument("--z-floor", type=float, default=1e-3,
                   help="smallest warm-start std, as a fraction of the metric's warm-start mean")
    p.add_argument("--z-clip", type=float, default=None, help="limit every z-score to +/- this value")
    p.add_argument("--out", type=Path, default=Path("_results/steering"))
    return p.parse_args(argv)


def write_outputs(args, rows, steering, audit_only, categories, problem):
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "steps.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    config = {}
    for key, value in vars(args).items():
        config[key] = str(value) if isinstance(value, Path) else value
    config["steering"] = steering
    config["audit_only"] = audit_only
    config["categories"] = categories
    config["dim"] = problem.dim
    config["n_objectives"] = problem.n_objectives
    config["score"] = "simple regret" if problem.n_objectives == 1 else "log10 hypervolume difference"
    (args.out / "config.json").write_text(json.dumps(config, indent=2))


def main(argv=None):
    args = parse_args(argv)
    if args.n_init < make_problem(args.problem).degree + 3:
        raise SystemExit(f"--n-init must be at least degree + 3 for {args.problem}")
    steering = choose_steering(args.split, args.split_index, args.steering)
    audit_only = [name for name in ALL_METRICS if name not in steering]

    rows = []
    categories = {}
    start = time.time()
    for policy_name in args.policies:
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            problem = make_problem(args.problem)
            campaign = Campaign(problem, args.condition, policy_name, seed, args, steering)
            campaign_rows, categories = campaign.run()
            rows.extend(campaign_rows)
        print(f"  {policy_name:22s} done ({time.time() - start:.0f}s)", flush=True)

    write_outputs(args, rows, steering, audit_only, categories, problem)

    print(f"\n{args.problem} / {args.condition}")
    print(f"  steering:   {', '.join(steering)}")
    print(f"  audit-only: {', '.join(audit_only) or '(none)'}")
    print(f"\nFinal score (mean over {args.seeds} seeds, lower is better):")
    for policy_name in args.policies:
        finals = [r["score"] for r in rows if r["policy"] == policy_name and r["step"] == args.steps]
        print(f"  {policy_name:22s} {np.mean(finals):.4f} ± {np.std(finals) / np.sqrt(len(finals)):.4f}")
    print(f"\nWrote {args.out / 'steps.csv'}  — next: analyze_steering.py --in {args.out}")


if __name__ == "__main__":
    main()
