"""Tests for traits_audit.stability.windowed (ported from battery-forecast exp3/exp4)."""
import numpy as np
import pytest

from traits_audit.stability.dmdc import (
    detrend_trajectory,
    dmdc_r2,
    fit_dmdc,
    stability_convergence,
)
from traits_audit.stability.windowed import (
    score_dmdc_r2,
    stability_convergence_windowed,
    stability_convergence_windowed_pydmd,
)

N_PARAMS = 7   # state is 2 * N_PARAMS = 14-D, matching the production shape these tests came from
N_ACTION = 6


class TestScoreDmdcR2:
    def test_matches_manual_detrend_then_score(self):
        """Contract/drift-guard: score_dmdc_r2 must exactly match manually
        detrending aug via traits_audit.dmdc._detrend_trajectory (the same
        private helper fit_dmdc uses internally) and scoring dmdc_r2 against
        that. If this ever fails after a traits_audit upgrade,
        _detrend_trajectory's behavior (or existence) has changed and
        score_dmdc_r2 needs to be revisited."""
        rng = np.random.default_rng(0)
        aug = rng.standard_normal((12, 6))
        actions = rng.standard_normal((11, 2))
        A_r, B_r, U_r = fit_dmdc(aug, actions, n_components=4)
        expected = dmdc_r2(detrend_trajectory(aug), actions, A_r, B_r, U_r)
        actual = score_dmdc_r2(aug, actions, A_r, B_r, U_r)
        assert actual == pytest.approx(expected, rel=1e-9)

    def test_r2_not_wildly_negative_for_offset_plus_small_dynamics(self):
        """Regression test for the raw-vs-detrended coordinate mismatch.

        A trajectory with a large, roughly-constant baseline offset (as real
        ECM-parameter trajectories have) plus small dynamics: dmdc_r2 scored
        against the RAW (undetrended) aug is catastrophically negative
        because ss_tot mean-centers the offset away while ss_res does not;
        scored against the correctly-detrended aug it is not. Uses the same
        rank-starved regime (T_fit=9 transitions) as this project's real
        exp3 trajectories.
        """
        rng = np.random.default_rng(5)
        n, m, n_steps = 14, 6, 9
        offset = rng.uniform(20, 60, n)
        aug = offset + rng.standard_normal((n_steps + 1, n)) * 0.05
        actions = rng.standard_normal((n_steps, m)) * 0.2

        A_r, B_r, U_r = fit_dmdc(aug, actions, n_components=8)
        r2_raw = dmdc_r2(aug, actions, A_r, B_r, U_r)
        r2_fixed = score_dmdc_r2(aug, actions, A_r, B_r, U_r)

        assert r2_raw < -100          # confirms the fixture reproduces the bug
        assert r2_fixed > 0.3         # not wildly negative once correctly detrended


class TestStabilityConvergenceWindowed:
    def test_window_none_matches_dm_stability_convergence(self):
        """window=None must reproduce traits_audit.dmdc.stability_convergence
        exactly (growing prefix, unbounded) -- the original behaviour."""
        rng = np.random.default_rng(0)
        aug = rng.standard_normal((20, 6))
        actions = rng.standard_normal((19, 2))
        expected = stability_convergence(aug, actions, min_obs=5, n_components=4)
        actual = stability_convergence_windowed(aug, actions, min_obs=5, n_components=4,
                                                   window=None)
        np.testing.assert_allclose(actual, expected)

    def test_windowed_point_matches_a_direct_fit_on_the_same_window(self):
        """Regression test for the windowing itself: the curve's last point
        under window=W must equal |lambda_max| of a fresh fit_dmdc call on
        only the last W+1 states / W actions -- not the full history."""
        window = 6
        rng = np.random.default_rng(1)
        aug = rng.standard_normal((20, 6))
        actions = rng.standard_normal((19, 2))

        conv = stability_convergence_windowed(aug, actions, min_obs=5, n_components=4,
                                                 window=window)

        t = 19  # last prefix index reached (min_obs..T inclusive, T = 19)
        A_t, _, _ = fit_dmdc(aug[t - window:t + 1], actions[t - window:t], n_components=4)
        expected_last = float(np.abs(np.linalg.eigvals(A_t)).max())
        assert conv[-1] == pytest.approx(expected_last)

    def test_windowed_and_unwindowed_diverge_once_history_exceeds_window(self):
        """Confirms windowing actually changes the result once the trajectory
        is longer than the window -- not a silent no-op."""
        rng = np.random.default_rng(2)
        aug = rng.standard_normal((20, 6))
        actions = rng.standard_normal((19, 2))
        unwindowed = stability_convergence_windowed(aug, actions, min_obs=5, n_components=4,
                                                       window=None)
        windowed = stability_convergence_windowed(aug, actions, min_obs=5, n_components=4,
                                                     window=6)
        assert not np.allclose(unwindowed[-1], windowed[-1])


@pytest.mark.skipif(
    __import__('importlib').util.find_spec('pydmd') is None, reason='pydmd not installed'
)
class TestStabilityConvergenceWindowedPydmd:
    def test_known_eigenvalue_recovered_on_synthetic_linear_system(self):
        """Drift-guard: calls the REAL pydmd.DMDc on a tiny synthetic linear
        system z_{t+1} = 0.7*z_t + 1.0*a_t (known eigenvalue 0.7) and
        asserts the recovered |lambda_max| is close to 0.7. If a future
        PyDMD version changes its .eigs accessor or fit(...) signature,
        this fails loudly instead of silently returning NaN/garbage in
        production.

        Note: the function detrends each window before fitting (see its
        docstring -- matches traits_audit.dmdc.fit_dmdc's own default
        behavior, and fixes the same coordinate-frame R^2 issue found
        elsewhere this session), which is why min_obs=30 (past the
        detrender's warmup=20) is used here rather than 10 -- the
        causal-EMA baseline needs to have exited warmup for a z(0)=0,
        mean-reverting-to-0 system like this one to be recovered exactly;
        during warmup the detrender subtracts an expanding global mean
        instead, which would distort the fit for a short prefix.

        Tolerance is loosened to 0.15 (not 0.05): the EMA baseline keeps
        adapting to this system's local (finite-window, not exactly zero)
        mean even past warmup, since it's driven by a random, non-zero
        action realization at every step -- detrending genuinely,
        legitimately perturbs the recovered eigenvalue a bit even for a
        "clean" system, and this test's real purpose is catching a gross
        API break (e.g. .eigs disappearing or returning garbage), not
        validating detrending's numerical precision to two decimal places."""
        rng = np.random.default_rng(0)
        T = 60
        actions = rng.uniform(-1, 1, size=(T, 1))
        z = np.zeros(T + 1)
        for t in range(T):
            z[t + 1] = 0.7 * z[t] + 1.0 * actions[t, 0]
        aug = z[:, None]
        conv, conv_r2, _, _, _ = stability_convergence_windowed_pydmd(
            aug, actions, min_obs=30, window=None, svd_rank=1
        )
        assert conv[-1] == pytest.approx(0.7, abs=0.15)
        # A clean, noise-free linear system should still be fit reasonably
        # well post-detrend (not the ~1.0 a raw, undetrended fit would give
        # -- detrending trades a little precision here for consistency
        # with the rest of this codebase's coordinate-frame handling).
        assert conv_r2[-1] > 0.8

    def test_window_bounds_the_fit_to_the_last_window_steps(self):
        """The last point under window=W must equal a fresh pydmd.DMDc fit
        on only the last W+1 states / W actions -- proves the windowing
        logic actually bounds the fit, not just that it runs."""
        rng = np.random.default_rng(1)
        T, window, min_obs = 25, 6, 5
        actions = rng.uniform(-1, 1, size=(T, 2))
        aug = rng.standard_normal((T + 1, 3))
        conv, conv_r2, _, _, _ = stability_convergence_windowed_pydmd(
            aug, actions, min_obs=min_obs, window=window, svd_rank=2
        )
        from pydmd import DMDc
        start = T - window
        state_win = detrend_trajectory(aug[start:T + 1], None)
        X = state_win.T
        U = actions[start:T].T
        direct = DMDc(svd_rank=2)
        direct.fit(X, U)
        expected = float(np.abs(direct.eigs).max())
        assert conv[-1] == pytest.approx(expected, rel=1e-9)

    def test_r2_matches_manual_one_step_ahead_computation(self):
        """Drift-guard for the R^2 half of the return value: confirms the
        one-step-ahead R^2 computed inside the function (using
        dmdc.operator.as_numpy_array / dmdc.B directly in feature space)
        matches an independent, manual computation of the same quantity.
        If a future PyDMD version changes what .operator.as_numpy_array or
        .B represent (e.g. reduced- vs full-coordinate convention), this
        fails loudly instead of silently scoring R^2 in the wrong frame."""
        from pydmd import DMDc
        rng = np.random.default_rng(4)
        T, min_obs = 20, 15
        actions = rng.uniform(-1, 1, size=(T, 2))
        aug = rng.standard_normal((T + 1, 3))
        _, conv_r2, _, _, _ = stability_convergence_windowed_pydmd(
            aug, actions, min_obs=min_obs, window=None, svd_rank=2
        )
        state_win = detrend_trajectory(aug[:T + 1], None)
        X = state_win.T
        U = actions[:T].T
        direct = DMDc(svd_rank=2)
        direct.fit(X, U)
        A_full = direct.basis @ direct.operator.as_numpy_array @ direct.basis.T
        Z_true = state_win[1:]
        Z_pred = (A_full @ state_win[:-1].T).T + (direct.B @ actions[:T].T).T
        ss_res = np.sum((Z_true - Z_pred) ** 2)
        ss_tot = np.sum((Z_true - Z_true.mean(axis=0)) ** 2)
        expected_r2 = 1.0 - ss_res / ss_tot
        assert conv_r2[-1] == pytest.approx(expected_r2, rel=1e-9)

    def test_lambda_max_and_r2_stay_equal_length_under_real_truncation(self):
        """Regression guard for a real bug found in this exact function:
        under genuine truncation (svd_rank < n_features), PyDMD's
        `dmdc.operator.as_numpy_array` is a REDUCED (r, r) matrix, not a
        full-feature-space one, even though `dmdc.B` already IS full
        feature-space -- using operator.as_numpy_array directly (as an
        earlier version of this function did) raised a shape-mismatch
        ValueError on every single window, which the surrounding
        try/except silently swallowed. Because the exception fired AFTER
        lambda_max_out.append() but BEFORE r2_out.append(), the except
        block's cleanup appended a SECOND (orphan NaN) entry to
        lambda_max_out for every iteration while r2_out only got one --
        silently corrupting the two arrays to different lengths (230 vs
        115 was the observed real-world case) with no exception ever
        surfacing to the caller. This test uses n_features=14 (this
        project's real augmented-state width) with svd_rank=4 (this
        project's real default), guaranteeing genuine truncation, and
        asserts the two returned arrays are the same length -- the
        simplest possible signal that this exact corruption can't recur."""
        rng = np.random.default_rng(6)
        T = 60
        actions = rng.standard_normal((T, N_ACTION))
        aug = rng.standard_normal((T + 1, 2 * N_PARAMS))  # 14-D, matching production
        conv, conv_r2, log_eig_spectra, effective_rank, raw_eigs = stability_convergence_windowed_pydmd(
            aug, actions, min_obs=5, window=None, svd_rank=4
        )
        assert len(conv) == len(conv_r2) == len(log_eig_spectra) == len(effective_rank) == len(raw_eigs)
        assert np.isfinite(conv_r2).any()  # not silently all-NaN either

    def test_log_eig_spectra_full_spectrum_matches_lambda_max(self):
        """Round 8 (Margazoglou & Magri 2023 literature check): the new
        3rd return value must contain the FULL sorted log|eig| set per
        window, not just the max -- and its max must equal
        log(lambda_max_curve[i]) exactly at every step, since lambda_max
        is itself just np.abs(dmdc.eigs).max() -- the same underlying
        eigenvalues, just reported two different ways."""
        rng = np.random.default_rng(7)
        T = 40
        actions = rng.standard_normal((T, N_ACTION))
        aug = rng.standard_normal((T + 1, 2 * N_PARAMS))
        conv, conv_r2, log_eig_spectra, _, _ = stability_convergence_windowed_pydmd(
            aug, actions, min_obs=5, window=None, svd_rank=4
        )
        for i, spec in enumerate(log_eig_spectra):
            if np.isfinite(conv[i]) and len(spec):
                assert spec.max() == pytest.approx(np.log(conv[i]), rel=1e-9)
                # sorted descending
                assert np.all(spec[:-1] >= spec[1:])
            elif not np.isfinite(conv[i]):
                assert len(spec) == 0

    def test_window_none_uses_full_growing_prefix(self):
        """window=None must NOT bound the fit -- the first and last points
        should use different-sized slices (growing prefix), unlike a fixed
        window which would use the same slice length throughout."""
        rng = np.random.default_rng(2)
        T = 20
        actions = rng.uniform(-1, 1, size=(T, 1))
        aug = rng.standard_normal((T + 1, 2))
        conv_unbounded, _, _, _, _ = stability_convergence_windowed_pydmd(
            aug, actions, min_obs=5, window=None, svd_rank=1
        )
        conv_windowed, _, _, _, _ = stability_convergence_windowed_pydmd(
            aug, actions, min_obs=5, window=3, svd_rank=1
        )
        # Not asserting exact values (both are legitimate different fits) --
        # just that they diverge once history exceeds the window, proving
        # window=None isn't silently behaving like a fixed small window.
        assert not np.allclose(conv_unbounded, conv_windowed, equal_nan=True)

    # ── Round 9: rank_mode configurability ──────────────────────────────

    def test_default_svd_rank_is_now_4_not_8(self):
        """Signature-introspection drift guard for the empirically-
        confirmed default change (a fixed svd_rank=8 against a 14-D
        augmented state forces in several near-noise-floor SVD modes,
        overfitting and degrading one-step-ahead R^2; svd_rank=4 roughly
        doubles-to-triples post-warmup median R^2, uniformly across every
        dataset/policy combination tested)."""
        import inspect
        sig = inspect.signature(stability_convergence_windowed_pydmd)
        assert sig.parameters["svd_rank"].default == 4

    def test_invalid_rank_mode_raises(self):
        rng = np.random.default_rng(0)
        aug = rng.standard_normal((30, 4))
        actions = rng.standard_normal((29, 2))
        with pytest.raises(ValueError):
            stability_convergence_windowed_pydmd(
                aug, actions, min_obs=5, rank_mode="bogus"
            )

    def test_energy_mode_resolves_expected_rank_on_synthetic_spectrum(self):
        """Hand-computable ground truth: a state matrix engineered so its
        singular-value spectrum needs exactly 2 components to reach 90%
        cumulative energy. Confirms energy mode's resolved rank matches
        this by construction, not just "some number"."""
        rng = np.random.default_rng(3)
        T = 40
        # Construct a 2-effective-rank system: two dominant, strongly
        # driven state dimensions, plus two near-constant (near-zero-
        # variance, post-detrend) dimensions that contribute negligible
        # singular-value energy.
        actions = rng.uniform(-1, 1, size=(T, 2))
        aug = np.zeros((T + 1, 4))
        for t in range(T):
            aug[t + 1, 0] = 0.6 * aug[t, 0] + 1.0 * actions[t, 0]
            aug[t + 1, 1] = 0.5 * aug[t, 1] + 1.0 * actions[t, 1]
            aug[t + 1, 2:] = rng.standard_normal(2) * 1e-6
        conv, conv_r2, _, effective_rank, _ = stability_convergence_windowed_pydmd(
            aug, actions, min_obs=20, window=None, svd_rank=4,
            rank_mode="energy", energy_threshold=0.90,
        )
        finite_ranks = effective_rank[np.isfinite(effective_rank)]
        assert len(finite_ranks) > 0
        assert np.median(finite_ranks) <= 2

    def test_energy_mode_capped_at_svd_rank_max(self):
        """A spectrum with roughly equal energy across all modes would
        resolve to a HIGH energy-rank -- confirms the max-rank cap is
        actually enforced, not just documented."""
        rng = np.random.default_rng(4)
        T = 40
        actions = rng.standard_normal((T, N_ACTION))
        aug = rng.standard_normal((T + 1, 2 * N_PARAMS))  # noisy, flat-ish spectrum
        conv, conv_r2, _, effective_rank, _ = stability_convergence_windowed_pydmd(
            aug, actions, min_obs=5, window=None, svd_rank=2,
            rank_mode="energy", energy_threshold=0.90,
        )
        finite_ranks = effective_rank[np.isfinite(effective_rank)]
        assert len(finite_ranks) > 0
        assert np.all(finite_ranks <= 2)

    def test_effective_rank_never_exceeds_fixed_svd_rank_under_fixed_mode(self):
        rng = np.random.default_rng(5)
        T = 60
        actions = rng.standard_normal((T, N_ACTION))
        aug = rng.standard_normal((T + 1, 2 * N_PARAMS))
        conv, conv_r2, _, effective_rank, _ = stability_convergence_windowed_pydmd(
            aug, actions, min_obs=5, window=None, svd_rank=4, rank_mode="fixed",
        )
        finite_ranks = effective_rank[np.isfinite(effective_rank)]
        assert len(finite_ranks) > 0
        assert np.all(finite_ranks <= 4)

    def test_raw_eigs_magnitude_matches_lambda_max(self):
        """The 5th return value (raw complex eigenvalues, for a pole-plot
        diagnostic) must be the SAME underlying eigenvalues lambda_max is
        derived from -- max(|raw_eigs[i]|) must equal lambda_max_curve[i]
        exactly at every step."""
        rng = np.random.default_rng(8)
        T = 40
        actions = rng.standard_normal((T, N_ACTION))
        aug = rng.standard_normal((T + 1, 2 * N_PARAMS))
        conv, conv_r2, _, _, raw_eigs = stability_convergence_windowed_pydmd(
            aug, actions, min_obs=5, window=None, svd_rank=4
        )
        assert len(raw_eigs) == len(conv)
        for i, eigs in enumerate(raw_eigs):
            if np.isfinite(conv[i]) and len(eigs):
                assert np.abs(eigs).max() == pytest.approx(conv[i], rel=1e-9)
            elif not np.isfinite(conv[i]):
                assert len(eigs) == 0

    def test_raw_eigs_can_have_nonzero_imaginary_part(self):
        """dmdc.eigs' dtype is only complex when a window's eigenvalues
        genuinely include a complex-conjugate pair (numpy's own
        np.linalg.eig returns real dtype otherwise) -- this test doesn't
        assert dtype (that varies legitimately window-to-window), just
        that AT LEAST ONE window across a real 14-D fit actually produces
        a nonzero imaginary part, confirming raw_eigs preserves the
        oscillatory (complex) poles a pole-plot diagnostic needs, not
        just real ones."""
        rng = np.random.default_rng(0)
        T = 60
        actions = rng.standard_normal((T, N_ACTION))
        aug = rng.standard_normal((T + 1, 2 * N_PARAMS))
        conv, conv_r2, _, _, raw_eigs = stability_convergence_windowed_pydmd(
            aug, actions, min_obs=5, window=None, svd_rank=4
        )
        any_imaginary = any(
            len(e) and np.any(np.abs(np.imag(e)) > 1e-12) for e in raw_eigs
        )
        assert any_imaginary

    def test_energy_mode_rank_is_actually_truncated_not_numpy_int64_bypassed(self):
        """Regression guard for a real bug found in this exact function:
        pydmd.utils._compute_rank returns a numpy.int64, not a plain
        Python int. Passing that numpy.int64 straight to
        DMDc(svd_rank=<numpy.int64>) silently bypasses truncation
        entirely -- PyDMD's OWN internal _compute_rank call (invoked
        again inside DMDc.fit on whatever svd_rank it's given) checks
        `isinstance(svd_rank, int)`, which numpy.int64 fails, silently
        falling through to the "no truncation" branch (`rank = min(rows,
        cols)`) with no exception raised anywhere. Confirmed empirically:
        a resolved energy-rank of 2 silently became an UNTRUNCATED basis
        of shape (n, n) instead of (n, 2) before this was fixed via an
        explicit int(...) cast. This test uses a clean 2-effective-rank
        synthetic system (mirrors
        test_energy_mode_resolves_expected_rank_on_synthetic_spectrum) and
        asserts the resolved rank is STRICTLY LESS than the full state
        width -- the simplest possible signal that truncation is real,
        not silently skipped."""
        rng = np.random.default_rng(6)
        T = 40
        actions = rng.uniform(-1, 1, size=(T, 2))
        aug = np.zeros((T + 1, 4))
        for t in range(T):
            aug[t + 1, 0] = 0.6 * aug[t, 0] + 1.0 * actions[t, 0]
            aug[t + 1, 1] = 0.5 * aug[t, 1] + 1.0 * actions[t, 1]
            aug[t + 1, 2:] = rng.standard_normal(2) * 1e-6
        conv, conv_r2, _, effective_rank, _ = stability_convergence_windowed_pydmd(
            aug, actions, min_obs=20, window=None, svd_rank=4,
            rank_mode="energy", energy_threshold=0.90,
        )
        finite_ranks = effective_rank[np.isfinite(effective_rank)]
        assert len(finite_ranks) > 0
        n_state = aug.shape[1]
        assert np.all(finite_ranks < n_state)
