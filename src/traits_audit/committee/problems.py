"""Benchmark problems for committee training, and a d-dimensional surrogate.

A ``Problem`` bundles everything the env needs to know about one benchmark:
input dimension, the noisy oracle, its noise-free counterpart (for regret /
hypervolume), a fixed output scale, the grid on which the surrogate's
(mu, sigma) enter the state, and the query-density histogram resolution.

Inputs are always normalised to [0, 1]^dim — the agent's action space.
Oracles return an ``(n, n_objectives)`` array in the problem's original
units. ``n_objectives`` says how many of those columns the audit actually
scores; the env divides each by the matching ``y_scales`` entry before
fitting, so every problem presents (mu, sigma) of similar magnitude to SAC
(observations are not normalised; only rewards are).

Problems
--------
forrester
    The original 1-D benchmark, unchanged: noisy Forrester from
    ``_cal_demo.py`` and its 1-D ``BootstrapSurrogate``. Existing trained
    models depend on this exact behaviour.
branin-currin
    BoTorch's two-objective BraninCurrin on [0, 1]^2 with the observation
    noise used in ``_mobo_demo.py``. Rewards audit Branin only (as that demo
    does); Currin is observed but not scored. 479-dim state.
branin-currin-mo
    The same oracle with **both** objectives audited: one surrogate each, the
    15 audit rewards scored per objective and averaged, and log-hypervolume
    difference in place of simple regret. 932-dim state, so its models are
    not interchangeable with ``branin-currin``'s — the two are separate arms,
    kept side by side deliberately.
color
    LED colour matching with the self-driving-lab-demo light simulator, as
    in ``_sdl_demo.py``: R, G, B in [0, 89] -> Frechet distance to the
    simulator's fixed target spectrum, plus that demo's observation noise.
"""
from __future__ import annotations

import itertools
from functools import cached_property
from typing import Callable, Optional

import numpy as np

from traits_audit._cal_demo import BootstrapSurrogate
from traits_audit._cal_demo import oracle as forrester_oracle
from traits_audit._cal_demo import oracle_clean as forrester_clean


class PolyBootstrapSurrogate:
    """``BootstrapSurrogate`` generalised to d-dimensional inputs.

    Same model — ridge regression (1e-3) on polynomial features, refit on
    ``n_estimators`` bootstrap resamples, sigma = ensemble std * std_scale —
    with features the monomials of total degree <= ``degree`` in ``dim``
    variables (21 terms for dim=2, degree=5).
    """

    def __init__(
        self,
        dim: int,
        degree: int,
        n_estimators: int = 30,
        std_scale: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ):
        self.dim = dim
        self.degree = degree
        self.n_estimators = n_estimators
        self.std_scale = std_scale
        self._rng = rng or np.random.default_rng()
        self._exponents = np.array(
            [e for e in itertools.product(range(degree + 1), repeat=dim) if sum(e) <= degree]
        )
        self._coefs: Optional[np.ndarray] = None  # (n_estimators, n_features)

    def _phi(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1, self.dim)
        return np.prod(x[:, None, :] ** self._exponents[None, :, :], axis=2)

    def fit(self, x: np.ndarray, y: np.ndarray) -> PolyBootstrapSurrogate:
        phi = self._phi(x)
        y = np.asarray(y, dtype=float)
        ridge = 1e-3 * np.eye(phi.shape[1])
        n = len(y)
        coefs = []
        for _ in range(self.n_estimators):
            idx = self._rng.integers(0, n, size=n)
            phi_b, y_b = phi[idx], y[idx]
            coefs.append(np.linalg.solve(phi_b.T @ phi_b + ridge, phi_b.T @ y_b))
        self._coefs = np.stack(coefs)
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        preds = self._phi(x) @ self._coefs.T  # (n_points, n_estimators)
        return preds.mean(1), preds.std(1) * self.std_scale


class _Forrester1DSurrogate:
    """Adapter so the env can hand (n, 1) inputs to the legacy 1-D
    ``BootstrapSurrogate`` while analysis code keeps passing 1-D grids."""

    def __init__(self, inner: BootstrapSurrogate):
        self.inner = inner

    @staticmethod
    def _flat(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return x[:, 0] if x.ndim == 2 else x

    def fit(self, x: np.ndarray, y: np.ndarray) -> _Forrester1DSurrogate:
        self.inner.fit(self._flat(x), y)
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.inner.predict(self._flat(x))


class _MultiSurrogate:
    """One independent surrogate per objective, behind the env's interface.

    ``PolyBootstrapSurrogate.fit`` is already shape-generic (the ridge solve
    handles an ``(n_features, n_obj)`` right-hand side), but ``predict``
    reduces over axis 1 to collapse the bootstrap ensemble, which is the
    wrong axis once ``y`` has columns. Rather than perform axis surgery on a
    class the Forrester and single-objective arms both depend on, hold a list
    and loop — the same adapter approach as :class:`_Forrester1DSurrogate`.

    ``fit`` takes ``(n, n_obj)``; ``predict`` returns ``(mu, sigma)`` each of
    shape ``(n_points, n_obj)``.
    """

    def __init__(self, surrogates: list):
        self.surrogates = surrogates

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_MultiSurrogate":
        y = np.asarray(y, dtype=float)
        for j, s in enumerate(self.surrogates):
            s.fit(x, y[:, j])
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        preds = [s.predict(x) for s in self.surrogates]
        mu = np.stack([p[0] for p in preds], axis=1)
        sigma = np.stack([p[1] for p in preds], axis=1)
        return mu, sigma


def _grid(points_per_dim: int, dim: int) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, points_per_dim)
    mesh = np.meshgrid(*([axis] * dim), indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=1)


class Problem:
    """Interface the env relies on. Subclasses set the class attributes and
    implement :meth:`observe` / :meth:`clean`."""

    name: str
    dim: int
    degree: int
    y_scale: float
    objective_names: tuple[str, ...]
    input_names: tuple[str, ...]
    grid_points_per_dim: int
    density_bins: int  # per input dimension

    # How many objectives the *audit* scores. Deliberately NOT derived from
    # ``len(objective_names)``: BraninCurrinProblem already names two
    # objectives while auditing only Branin, and inferring this would flip
    # its state_dim from 479 to 932 and invalidate every saved SAC model
    # (SB3 raises the shape mismatch at load time, i.e. during analysis,
    # long after the training that produced them).
    n_objectives: int = 1

    @cached_property
    def grid(self) -> np.ndarray:
        """(n_grid, dim) points where the surrogate's (mu, sigma) enter the state."""
        return _grid(self.grid_points_per_dim, self.dim)

    @cached_property
    def y_scales(self) -> np.ndarray:
        """Per-objective divisor applied before the surrogate sees ``y``.

        One entry per audited objective. Defaults to ``y_scale`` repeated, so
        single-objective problems are unchanged. Multi-objective problems must
        override with genuinely per-objective values: observations are not
        normalised before SAC sees them (only rewards are), so a shared scale
        would leave one objective's whole mu/sigma block an order of magnitude
        smaller than the other's and effectively invisible to the policy.
        """
        return np.full(self.n_objectives, float(self.y_scale), dtype=np.float64)

    @property
    def state_dim(self) -> int:
        # mu + sigma on the grid (per objective), summary features, and the
        # query-density histogram. The summary block is
        #   step_norm + per-objective (mean, var, best)
        # = 1 + 3*n_obj, which is the familiar 4 when n_obj == 1.
        return (
            2 * self.n_objectives * len(self.grid)
            + 1 + 3 * self.n_objectives
            + self.density_bins ** self.dim
        )

    def observe(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Noisy oracle: (n, dim) in [0, 1] -> (n, n_objectives), original units."""
        raise NotImplementedError

    def clean(self, x: np.ndarray) -> np.ndarray:
        """Noise-free objectives: (n, dim) in [0, 1] -> (n, n_objectives)."""
        raise NotImplementedError

    @cached_property
    def true_min(self) -> float:
        """Global minimum of the audited objective (``clean(x)[:, 0]``) over
        [0, 1]^dim, for simple-regret scoring. Subclasses override with an
        exact/literature value where one exists; this default does a coarse
        grid search, exact only up to grid resolution."""
        xs = _grid(min(200, int(50_000 ** (1.0 / self.dim))), self.dim)
        return float(self.clean(xs)[:, 0].min())

    def make_surrogate(
        self, degree: int, n_estimators: int, std_scale: float, rng: np.random.Generator,
    ):
        if self.n_objectives == 1:
            return PolyBootstrapSurrogate(
                dim=self.dim, degree=degree,
                n_estimators=n_estimators, std_scale=std_scale, rng=rng,
            )
        # Independent seeds per objective so the two bootstrap ensembles are
        # not resampling in lockstep, which would correlate their sigmas.
        return _MultiSurrogate([
            PolyBootstrapSurrogate(
                dim=self.dim, degree=degree,
                n_estimators=n_estimators, std_scale=std_scale,
                rng=np.random.default_rng(rng.integers(0, 2**63 - 1)),
            )
            for _ in range(self.n_objectives)
        ])

    # -- multi-objective scoring ------------------------------------------
    # Only meaningful when n_objectives > 1; used by analysis/regret.py, never
    # inside the env or the reward path.

    ref_point: Optional[np.ndarray] = None   # hypervolume reference, minimisation
    hv_max: Optional[float] = None           # dense-grid ceiling, for log HV difference

    def hypervolume(self, Y: np.ndarray) -> float:
        """Hypervolume dominated by the non-dominated set of ``Y`` (minimisation).

        ponytail: six lines duplicated from ``_mobo_demo._hypervolume`` rather
        than shared. ``_pybamm_demo`` holds a third copy with the *opposite*
        (maximisation) sign convention, and a common helper taking a
        ``maximise=`` flag is how a third convention bug gets written. The
        demos are standalone reproducibility artifacts; this method is the
        committee's own. ``test_hypervolume_matches_mobo_demo`` pins them
        together.
        """
        import torch
        from botorch.utils.multi_objective.hypervolume import Hypervolume
        from botorch.utils.multi_objective.pareto import is_non_dominated

        if self.ref_point is None:
            raise ValueError(f"{self.name!r} has no ref_point; hypervolume undefined")
        Y_t = torch.tensor(np.asarray(Y, dtype=float), dtype=torch.float64)
        ref = torch.tensor(self.ref_point, dtype=torch.float64)
        # is_non_dominated expects maximisation, so negate on the way in and
        # back out again for Hypervolume.
        mask = is_non_dominated(-Y_t)
        return float(Hypervolume(ref_point=-ref).compute(-Y_t[mask]))


class ForresterProblem(Problem):
    name = "forrester"
    dim = 1
    degree = 5
    y_scale = 1.0
    objective_names = ("forrester",)
    input_names = ("x",)
    grid_points_per_dim = 200
    density_bins = 10

    def __init__(
        self,
        oracle_fn: Optional[Callable[[np.ndarray, np.random.Generator], np.ndarray]] = None,
    ):
        self.oracle_fn = oracle_fn if oracle_fn is not None else forrester_oracle

    def observe(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return np.asarray(self.oracle_fn(np.asarray(x)[:, 0], rng))[:, None]

    def clean(self, x: np.ndarray) -> np.ndarray:
        return forrester_clean(np.asarray(x)[:, 0])[:, None]

    def make_surrogate(
        self, degree: int, n_estimators: int, std_scale: float, rng: np.random.Generator,
    ):
        return _Forrester1DSurrogate(BootstrapSurrogate(
            degree=degree, n_estimators=n_estimators, std_scale=std_scale, rng=rng,
        ))

    @cached_property
    def true_min(self) -> float:
        # Exact 100k-point grid search — matches analysis/regret.py's
        # FORRESTER_TRUE_MIN bit-for-bit (verified in tests). Kept as an
        # explicit override (rather than the base class's dim-scaled grid)
        # so existing Forrester regret figures don't shift.
        xs = np.linspace(0.0, 1.0, 100_000)
        return float(self.clean(xs[:, None])[:, 0].min())


class BraninCurrinProblem(Problem):
    name = "branin-currin"
    dim = 2
    degree = 5
    y_scale = 50.0  # Branin spans ~0.4-308 on [0, 1]^2
    objective_names = ("branin", "currin")
    input_names = ("x1", "x2")
    grid_points_per_dim = 15
    density_bins = 5

    @cached_property
    def _fn(self):
        from botorch.test_functions.multi_objective import BraninCurrin
        return BraninCurrin(negate=False)

    @cached_property
    def noise_std(self) -> np.ndarray:
        from traits_audit._mobo_demo import _NOISE_SE
        return _NOISE_SE.numpy()

    def clean(self, x: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            return self._fn(torch.as_tensor(np.asarray(x, dtype=float))).numpy()

    def observe(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        y = self.clean(x)
        return y + rng.normal(0.0, self.noise_std, size=y.shape)

    # Literature value for Branin's global minimum (it has three, all equal):
    # verified here via multistart L-BFGS-B from each of the three known
    # basins (starting points from the standard reference table), all
    # converging to 0.3978873577297666 to 1e-10. Exact to float64 precision,
    # unlike a grid search.
    true_min: float = 0.3978873577297666


class BraninCurrinMOProblem(BraninCurrinProblem):
    """Branin-Currin with **both** objectives audited.

    Identical oracle, noise, grid and degree to :class:`BraninCurrinProblem` —
    the only difference is that the env fits a surrogate per objective and each
    of the 15 audit rewards is scored on both uncertainty streams, then
    averaged. The single-objective class is left untouched so its trained
    models and figures stay valid as the comparison arm.

    Averaging rather than summing: it keeps the reward on the same scale as the
    single-objective arm (so learning curves are comparable and the downstream
    z-score normaliser behaves the same), and it makes ``MahalanobisOOD`` --
    whose check reads only the queried-x geometry and is objective-independent
    -- identical across the two arms instead of double-counted.
    """

    name = "branin-currin-mo"
    n_objectives = 2

    # Branin std ~50.4, Currin std ~2.6 over [0,1]^2: a shared scale would put
    # Currin's entire mu/sigma block at ~1/20 the magnitude of Branin's, and
    # observations are not normalised before SAC sees them.
    #
    # ponytail: y_scales is the balance knob between the two reward streams.
    # If one objective's raw-delta std stays an order of magnitude larger,
    # retune these before reaching for a per-objective RunningZScore -- the
    # env's output is already z-scored by ZScoreRewardWrapper, and a second
    # normalisation layer inside would mean three sets of running statistics
    # converging on different timescales.
    @cached_property
    def y_scales(self) -> np.ndarray:
        return np.array([50.0, 2.6], dtype=np.float64)

    # Hypervolume reference, in minimisation space. Taken from the project's
    # own MOBO demo (_mobo_demo._REF_POINT) so the committee and the demo
    # score the same objective ranges the same way. Verified to dominate the
    # clean objectives everywhere on [0,1]^2 (see test).
    ref_point: Optional[np.ndarray] = np.array([18.0, 6.0], dtype=np.float64)

    @cached_property
    def hv_max(self) -> float:
        """Practical hypervolume ceiling, from a dense grid of clean values.

        Used as the reference in ``log10(hv_max - HV_t)``. A grid rather than
        the true Pareto front: the front is only reachable in the limit, and a
        ceiling slightly below the theoretical one keeps the log finite and
        the scale interpretable.
        """
        return self.hypervolume(self.clean(_grid(120, self.dim)))


class ColorMatchingProblem(Problem):
    name = "color"
    dim = 3
    degree = 3
    y_scale = 1e4  # Frechet distance spans ~0-130k over the RGB cube
    objective_names = ("frechet",)
    input_names = ("R", "G", "B")
    grid_points_per_dim = 8
    density_bins = 4
    noise_std = 200.0  # _sdl_demo.run(noise_std=200.0)

    @cached_property
    def _sdl(self):
        from traits_audit._sdl_demo import _ensure_sdl_importable
        _ensure_sdl_importable()
        from self_driving_lab_demo import SelfDrivingLabDemoLight
        return SelfDrivingLabDemoLight(autoload=True, simulation=True)

    @cached_property
    def rgb_max(self) -> float:
        return float(self._sdl.bounds["R"][1])

    # Frechet distance is a metric (>= 0, = 0 iff the two spectra are
    # identical), and the simulator's fixed target R/G/B (verified to lie
    # inside [0, rgb_max]^3, so it's reachable by the [0,1]^3 action space)
    # reproduces its own reference spectrum exactly. So 0.0 is the exact
    # global minimum, not merely the smallest value seen in a sample.
    true_min: float = 0.0

    @cached_property
    def target(self) -> Optional[np.ndarray]:
        """The optimum in normalised [0, 1]^3 RGB, or None if unavailable.

        The simulator stores its goal as a *spectrum* and only derives the
        (R, G, B) that produced it on request (``target_inputs`` reads None
        until then), so this asks it rather than hardcoding a triple that
        would silently go stale if the demo's ``target_seed`` ever changed.
        """
        inputs = self._sdl.get_target_inputs()
        if not inputs:
            return None
        return np.array([float(inputs[k]) for k in ("R", "G", "B")]) / self.rgb_max

    def clean(self, x: np.ndarray) -> np.ndarray:
        rgb = np.asarray(x, dtype=float).reshape(-1, 3) * self.rgb_max
        return np.array([
            [self._sdl.evaluate({"R": r, "G": g, "B": b})["frechet"]] for r, g, b in rgb
        ])

    def observe(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        y = self.clean(x)
        return y + rng.normal(0.0, self.noise_std, size=y.shape)


PROBLEMS: dict[str, type[Problem]] = {
    p.name: p for p in (
        ForresterProblem,
        BraninCurrinProblem,
        BraninCurrinMOProblem,
        ColorMatchingProblem,
    )
}


def get_problem(name: str) -> Problem:
    if name not in PROBLEMS:
        raise KeyError(f"Unknown problem {name!r}; choose from {list(PROBLEMS)}")
    return PROBLEMS[name]()
