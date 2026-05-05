"""Tests for AR parameter estimation and OASIS deconvolution."""

import numpy as np
import pytest
import scipy.sparse as sp

from cnmfe.temporal import _oasis_ar1_pava, deconvolve, estimate_ar_params, update_temporal


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

        C_new, S_new = update_temporal(Y_flat, A, C, sn, ar_order=1, n_iter=1)
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

        C_new, _ = update_temporal(Y_flat, A, C_init, sn, ar_order=1, n_iter=2)

        # At least one component should have r > 0.5
        correlations = [
            np.corrcoef(C_new[k], d["C_true"][k])[0, 1]
            for k in range(C_new.shape[0])
        ]
        assert max(correlations) > 0.5, f"Best correlation = {max(correlations):.3f}"
