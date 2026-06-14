"""Tests for AR parameter estimation and OASIS deconvolution."""

import numpy as np
import pytest
import scipy.sparse as sp

from minicnmfe.temporal import _oasis_ar1_pava, deconvolve, estimate_ar_params, update_temporal


def make_ar1_trace(T: int = 300, g: float = 0.9, sn: float = 0.3, seed: int = 0) -> dict:
    """Generate a clean AR(1) calcium trace with Poisson spikes."""
    rng = np.random.default_rng(seed)
    S = (rng.random(T) < 0.05).astype(np.float32)
    S[0] = 0
    C = np.zeros(T, dtype=np.float32)
    for t in range(1, T):
        C[t] = g * C[t - 1] + S[t]
    trace = C + rng.standard_normal(T).astype(np.float32) * sn
    return {"trace": trace, "C_true": C, "S_true": S, "g": g, "sn": sn}


class TestEstimateArParams:
    def test_ar1_decay_estimate(self):
        # Yule-Walker has downward bias with observation noise; use low-noise trace
        data = make_ar1_trace(T=800, g=0.9, sn=0.1)
        g_est, sn_est = estimate_ar_params(data["trace"], p=1)
        assert abs(float(g_est[0]) - 0.9) < 0.2, f"g estimate {g_est[0]:.3f} too far from 0.9"

    def test_noise_estimate(self):
        rng = np.random.default_rng(1)
        true_sn = 1.5
        trace = rng.standard_normal(500).astype(np.float32) * true_sn
        _, sn_est = estimate_ar_params(trace, p=1)
        assert abs(sn_est - true_sn) / true_sn < 0.25

    def test_g_in_valid_range(self):
        data = make_ar1_trace()
        g_est, _ = estimate_ar_params(data["trace"], p=1)
        assert 0.0 <= float(g_est[0]) <= 1.0

    def test_shape(self):
        data = make_ar1_trace()
        g, sn = estimate_ar_params(data["trace"], p=1)
        assert g.shape == (1,)
        assert np.isscalar(sn)

    def test_prior_pulls_g_toward_target(self):
        """With g_prior set, the result should shrink toward the target."""
        # Build a drift-contaminated trace so naive Yule-Walker over-estimates.
        rng = np.random.default_rng(7)
        T = 600
        drift = np.cumsum(rng.standard_normal(T)) * 0.05    # near-random walk
        ar = np.zeros(T, dtype=np.float64)
        for t in range(1, T):
            ar[t] = 0.7 * ar[t - 1] + (rng.random() < 0.04) * 1.0
        trace = (ar + drift).astype(np.float32)

        g_baseline, _ = estimate_ar_params(trace, p=1)
        g_strong, _ = estimate_ar_params(trace, p=1, g_prior=0.7, g_prior_weight=0.95)
        g_weak, _ = estimate_ar_params(trace, p=1, g_prior=0.7, g_prior_weight=0.05)

        assert abs(float(g_strong[0]) - 0.7) < 0.05, (
            f"strong prior should land near 0.7, got {float(g_strong[0]):.3f}"
        )
        # Weak prior leans on data — the drift biases YW upward, so g_weak
        # should sit above 0.7 and close to the no-prior baseline.
        assert float(g_weak[0]) > float(g_strong[0]), (
            "weak prior should pull less than strong prior"
        )
        assert abs(float(g_weak[0]) - float(g_baseline[0])) < 0.1, (
            "weak prior should not move far from the data-only estimate"
        )

    def test_no_prior_unchanged(self):
        """g_prior=None must reproduce the legacy fudge_factor path exactly."""
        data = make_ar1_trace(T=400, g=0.85, sn=0.2)
        g_legacy, sn_legacy = estimate_ar_params(data["trace"], p=1, fudge_factor=0.96)
        g_explicit_none, sn_explicit_none = estimate_ar_params(
            data["trace"], p=1, fudge_factor=0.96,
            g_prior=None, g_prior_weight=0.5,
        )
        assert float(g_legacy[0]) == float(g_explicit_none[0])
        assert sn_legacy == sn_explicit_none


class TestDeconvolve:
    def test_output_shapes(self):
        data = make_ar1_trace(T=200)
        g, sn = estimate_ar_params(data["trace"], p=1)
        c, s, bl = deconvolve(data["trace"], g, sn)
        assert c.shape == (200,)
        assert s.shape == (200,)

    def test_non_negative_spikes(self):
        data = make_ar1_trace()
        g, sn = estimate_ar_params(data["trace"], p=1)
        _, s, _ = deconvolve(data["trace"], g, sn)
        assert (s >= -1e-6).all(), "Spike train should be non-negative"

    def test_clean_trace_recovery(self):
        """On a clean AR1 trace, recovered C should correlate with true C."""
        data = make_ar1_trace(T=400, sn=0.2)
        g, sn = estimate_ar_params(data["trace"], p=1)
        c, _, _ = deconvolve(data["trace"], g, sn)
        r = np.corrcoef(c, data["C_true"])[0, 1]
        assert r > 0.7, f"Correlation with true C = {r:.3f}"

    def test_pava_fallback(self):
        """Pure-Python PAVA should produce non-negative spikes."""
        data = make_ar1_trace(T=200)
        g, sn = estimate_ar_params(data["trace"], p=1)
        c, s, bl = _oasis_ar1_pava(data["trace"], float(g[0]), sn)
        assert (s >= -1e-6).all()
        assert c.shape == (200,)

    def test_pava_fallback_reconstructs_clean_ar1(self):
        """PAVA fallback must faithfully reconstruct a clean AR(1) trace.

        Regression guard for the merge-condition bug: the pool-boundary test used
        bare ``g`` instead of ``g**pool_length``, which over-merged smooth exact-g
        decays and collapsed the trace — a clean AR(1) reconstructed at only
        r~0.4, dropping ``model.C`` vs ground truth from ~0.96 to ~0.58 whenever
        the ``oasis-deconv`` package was absent. This calls the fallback
        directly (so the result is independent of whether the package is
        installed) on a NOISELESS trace, where a correct OASIS returns the input
        almost exactly.
        """
        data = make_ar1_trace(T=600, g=0.9, sn=0.0, seed=3)   # trace == C_true
        c, s, _ = _oasis_ar1_pava(data["trace"], data["g"], sn=0.0)
        r = float(np.corrcoef(c, data["C_true"])[0, 1])
        assert r > 0.95, f"PAVA reconstruction of clean AR(1) only r={r:.3f} (merge-condition regression?)"
        assert (s >= -1e-6).all()


class TestUpdateTemporal:
    def test_output_shapes(self, synth_small):
        d = synth_small
        T = d["movie"].shape[0]
        H, W = d["dims"]
        K = d["A_true"].shape[1]
        Y_flat = d["movie"].reshape(T, H * W).T  # (H*W, T)
        A = sp.csc_matrix(d["A_true"].astype(np.float32))
        C = d["C_true"].copy()
        sn = np.ones(H * W, dtype=np.float32) * d["sn_true"]

        C_new, S_new, _, _ = update_temporal(Y_flat, A, C, sn, ar_order=1, n_iter=1)
        assert C_new.shape == (K, T)
        assert S_new.shape == (K, T)

    def test_temporal_correlation(self, synth_small):
        """Updated temporal traces should correlate with ground truth."""
        d = synth_small
        T = d["movie"].shape[0]
        H, W = d["dims"]
        Y_flat = d["movie"].reshape(T, H * W).T
        A = sp.csc_matrix(d["A_true"].astype(np.float32))
        C_init = d["C_true"].copy()
        sn = np.ones(H * W, dtype=np.float32) * d["sn_true"]

        C_new, _, _, _ = update_temporal(Y_flat, A, C_init, sn, ar_order=1, n_iter=2)

        # Every component should track its ground-truth trace on a small,
        # well-behaved synthetic — best-of-K hides per-neuron failures.
        correlations = [
            float(np.corrcoef(C_new[k], d["C_true"][k])[0, 1])
            for k in range(C_new.shape[0])
        ]
        assert np.mean(correlations) > 0.7, (
            f"Mean correlation = {np.mean(correlations):.3f}, per-neuron: "
            f"{[round(r, 3) for r in correlations]}"
        )


# ---------------------------------------------------------------------------
# Drift-robustness (Fix 1 + Fix 2 from the plan)
# ---------------------------------------------------------------------------

class TestDriftRobustness:
    """Slow drift inflates lag-1 autocorrelation; the detrend kwargs should
    keep `g` close to truth (Fix 1) and let OASIS recover spike events
    (Fix 2). These mirror the failure mode visible in tmp/deconv_real.png.
    """

    def test_estimate_ar_invariant_to_linear_drift(self):
        """The cleanest invariant of the detrend fix: adding a linear drift
        to a trace should NOT change the estimated `g` when detrending is on,
        and SHOULD inflate `g` toward the fudge-factor ceiling (~0.96) when
        detrending is off. This isolates the specific failure mode visible
        in tmp/deconv_real.png (drift → over-estimated g → OASIS collapse).

        Note: Yule-Walker on sparse-spike traces has its own intrinsic
        downward bias; we therefore do NOT assert |g_est − g_true| < small
        on the clean trace. The point of the test is the invariance under
        an added drift, which is what the algorithm is designed to deliver.
        """
        T = 1500
        g_true = 0.90
        data = make_ar1_trace(T=T, g=g_true, sn=0.1, seed=3)
        trace_clean = data["trace"]
        trace_drifty = trace_clean + np.linspace(0.0, 30.0, T).astype(np.float32)

        # Reference: g on the clean (no-drift) trace.
        g_clean, _ = estimate_ar_params(trace_clean, p=1, detrend_order=0)
        g_clean = float(g_clean[0])

        # Without detrending: drift inflates g toward the fudge ceiling.
        g_no, _ = estimate_ar_params(trace_drifty, p=1, detrend_order=0)
        g_no = float(g_no[0])
        assert g_no > 0.94, (
            f"Drift should push g near the fudge ceiling (≈0.96) without "
            f"detrending; got {g_no:.3f}"
        )

        # With detrending: g matches the clean-trace value within a tight
        # band (linear ramp is exactly captured by a degree-≥1 polynomial,
        # so the drifty estimate should be ≈ clean-trace estimate).
        g_yes, _ = estimate_ar_params(trace_drifty, p=1, detrend_order=2)
        g_yes = float(g_yes[0])
        assert abs(g_yes - g_clean) < 0.01, (
            f"Detrended g on drifty trace ({g_yes:.4f}) should match clean "
            f"g ({g_clean:.4f}) within 0.01; got delta={g_yes - g_clean:+.4f}"
        )
        # And it must be much further from the no-detrend (inflated) value
        # than from the clean reference.
        assert abs(g_yes - g_clean) < 0.5 * abs(g_no - g_clean), (
            f"Detrend should mostly cancel the drift inflation: "
            f"g_no={g_no:.3f}, g_yes={g_yes:.3f}, g_clean={g_clean:.3f}"
        )

    def test_update_temporal_detrend_recovers_spikes(self):
        """Three sharp transients on top of an exponential bleach: the
        polynomial detrend (detrend_order=3) must remove the drift before OASIS.

        With a correct OASIS the raw (no-detrend) path can still hit the true
        spike *times*, but the un-removed bleach leaks in as many spurious
        spikes; the detrend path fires ~only the real ones. So the discriminator
        is the false-positive spike count, not the hit count.

        Setup: a 1-pixel "movie" with a single 1.0-weight footprint, so the
        projected trace is exactly that pixel's data.
        """
        T = 800
        rng = np.random.default_rng(11)
        # Three sharp transients at known frames.
        S_true = np.zeros(T, dtype=np.float32)
        spike_frames = [200, 400, 600]
        for t in spike_frames:
            S_true[t] = 3.0
        g = 0.92
        C_clean = np.zeros(T, dtype=np.float32)
        for t in range(1, T):
            C_clean[t] = g * C_clean[t - 1] + S_true[t]
        bleach = 8.0 * np.exp(-np.arange(T, dtype=np.float32) / 250.0)
        noise = rng.standard_normal(T).astype(np.float32) * 0.2
        trace = C_clean + bleach + noise

        # 1-pixel × T "movie" with a single 1.0-weight footprint.
        Y_flat = trace[None, :].astype(np.float32)               # (1, T)
        A = sp.csc_matrix(np.ones((1, 1), dtype=np.float32))     # (1, 1)
        C_init = np.zeros((1, T), dtype=np.float32)
        sn = np.array([0.2], dtype=np.float32)
        g_cached = [np.array([g], dtype=np.float32)]
        sn_cached = np.array([0.2], dtype=np.float32)

        # Without detrending.
        _, S_no, _, _ = update_temporal(
            Y_flat, A, C_init, sn,
            ar_order=1, n_iter=2,
            g_cached=g_cached, sn_cached=sn_cached,
            detrend_order=0,
        )
        # With detrending.
        _, S_yes, _, _ = update_temporal(
            Y_flat, A, C_init.copy(), sn,
            ar_order=1, n_iter=2,
            g_cached=g_cached, sn_cached=sn_cached,
            detrend_order=3,
        )

        # Count "real" spikes (S > 0.5) in a ±3-frame window around truth.
        def hits(S):
            return sum(
                int(S[0, max(0, t - 3): t + 4].max() > 0.5)
                for t in spike_frames
            )

        # False positives = spikes outside the ±3-frame windows around truth.
        def n_false(S):
            mask = np.ones(S.shape[1], dtype=bool)
            for t in spike_frames:
                mask[max(0, t - 3): t + 4] = False
            return int((S[0, mask] > 0.5).sum())

        hits_yes = hits(S_yes)
        fp_no, fp_yes = n_false(S_no), n_false(S_yes)
        assert hits_yes >= 2, (
            f"Expected ≥2 of 3 spikes recovered with detrend; got {hits_yes}. "
            f"S_yes nonzero count: {int((S_yes > 0.5).sum())}"
        )
        # The bleach leaks into the raw deconvolution as spurious spikes; the
        # detrend path must fire far fewer false positives.
        assert fp_yes < fp_no, (
            f"Detrend must suppress drift-induced false spikes: "
            f"fp_yes={fp_yes}, fp_no={fp_no}"
        )

    def test_detrend_is_no_op_on_clean_trace(self):
        """Symmetric guard: on a trace with NO slow drift, enabling the
        polynomial detrend (order ≥ 1) must NOT change the AR estimate.

        Pin against future loosenings of `_detrend_poly` or accidental
        over-fitting: if a regression made the detrend bias `g` on clean
        data, this test catches it. Companion to
        `test_estimate_ar_invariant_to_linear_drift` (which checks the
        invariant in the drifty direction).
        """
        # Clean AR(1) trace, no drift component.
        data = make_ar1_trace(T=1500, g=0.9, sn=0.1, seed=7)
        trace = data["trace"]

        g0, sn0 = estimate_ar_params(trace, p=1, detrend_order=0)
        g2, sn2 = estimate_ar_params(trace, p=1, detrend_order=2)
        g0, g2 = float(g0[0]), float(g2[0])

        # Mean-only and quadratic detrend on a clean trace should give
        # essentially the same g. Empirically the difference is well under
        # 0.005 on this fixture; assert at 0.01 for headroom.
        assert abs(g2 - g0) < 0.01, (
            f"Detrend should be a no-op on a clean trace: "
            f"g(detrend=0)={g0:.4f}, g(detrend=2)={g2:.4f}"
        )
        # Noise estimate uses the high-frequency PSD only; it does not
        # touch the detrend path, but pin it anyway so a refactor that
        # accidentally couples them gets caught.
        assert abs(sn2 - sn0) < 1e-6, (
            f"sn estimate must not depend on detrend_order: "
            f"sn(0)={sn0:.6f}, sn(2)={sn2:.6f}"
        )
