"""Tests for noise estimation and CORR/PNR summary images."""

import numpy as np
import pytest

from cnmfe.preprocess import (
    correlation_pnr,
    estimate_noise,
    local_correlations_fft,
    make_center_surround_psf,
)


class TestCenterSurroundPSF:
    def test_sums_to_zero(self):
        psf = make_center_surround_psf(sigma=3.0)
        assert abs(psf.sum()) < 1e-4, "PSF should sum to ~0"

    def test_shape_is_odd(self):
        psf = make_center_surround_psf(sigma=3.0)
        assert psf.shape[0] % 2 == 1
        assert psf.shape[1] % 2 == 1

    def test_centre_is_positive(self):
        psf = make_center_surround_psf(sigma=3.0)
        cy, cx = psf.shape[0] // 2, psf.shape[1] // 2
        assert psf[cy, cx] > 0, "Centre of DoG kernel should be positive"

    def test_custom_size(self):
        psf = make_center_surround_psf(sigma=2.0, size=11)
        assert psf.shape == (11, 11)


class TestEstimateNoise:
    def test_white_noise(self):
        rng = np.random.default_rng(0)
        T, H, W = 500, 16, 16
        true_std = 2.0
        movie = (rng.standard_normal((T, H, W)) * true_std).astype(np.float32)
        sn = estimate_noise(movie, noise_range=(0.25, 0.5))
        assert sn.shape == (H, W)
        # Estimate should be within 20% of true std
        np.testing.assert_allclose(sn.mean(), true_std, rtol=0.2)

    def test_shape(self):
        movie = np.random.rand(200, 8, 12).astype(np.float32)
        sn = estimate_noise(movie)
        assert sn.shape == (8, 12)

    def test_signal_noise_separation(self):
        """Noise estimate on a signal + noise mix should reflect noise, not signal."""
        rng = np.random.default_rng(1)
        T, H, W = 400, 8, 8
        noise_std = 1.0
        # Low-frequency signal much larger than noise
        t = np.linspace(0, 1, T, dtype=np.float32)
        signal = 10.0 * np.sin(2 * np.pi * 0.05 * t)[:, np.newaxis, np.newaxis]
        noise = rng.standard_normal((T, H, W)).astype(np.float32) * noise_std
        movie = signal + noise
        sn = estimate_noise(movie)
        # Should be close to noise_std (not dominated by the signal)
        np.testing.assert_allclose(sn.mean(), noise_std, rtol=0.3)


class TestLocalCorrelationsFFT:
    def test_correlated_signal(self):
        """Movie where all pixels share the same temporal trace → high correlation."""
        rng = np.random.default_rng(0)
        T, H, W = 200, 16, 16
        # All pixels have same temporal trace (different amplitude, same signal shape)
        signal = np.sin(2 * np.pi * np.arange(T, dtype=np.float32) / 20)
        spatial = rng.uniform(0.5, 1.5, (H, W)).astype(np.float32)
        movie = signal[:, np.newaxis, np.newaxis] * spatial[np.newaxis]
        movie -= movie.mean(axis=0, keepdims=True)
        cn = local_correlations_fft(movie)
        assert cn.shape == (16, 16)
        # All pixels share the signal → correlation near 1
        assert cn.mean() > 0.5

    def test_independent_noise(self):
        """Independent pixel noise → correlation ≈ 0."""
        rng = np.random.default_rng(2)
        movie = rng.standard_normal((500, 16, 16)).astype(np.float32)
        movie -= movie.mean(axis=0, keepdims=True)
        cn = local_correlations_fft(movie)
        assert abs(cn.mean()) < 0.15

    def test_output_shape(self):
        movie = np.random.rand(100, 20, 30).astype(np.float32)
        movie -= movie.mean(axis=0)
        cn = local_correlations_fft(movie)
        assert cn.shape == (20, 30)

    def test_bounded_with_thresholded_input(self):
        """CORR must stay in [-1, 1] even when the input has a non-zero mean.

        Regression: ``correlation_pnr`` thresholds at ``3*sn`` before calling
        this function, which leaves a positive mean (mostly zeros plus
        occasional spikes). Without self-recentering, the formula evaluates
        to ``1/(1-f)`` for two pixels sharing a fraction ``f`` of spike
        times, exceeding 1 for any non-trivial spike rate. Self-recenter
        inside the function fixes this so callers don't have to worry.
        """
        rng = np.random.default_rng(0)
        T, H, W = 200, 10, 10
        # Sparse positive spikes shared across all pixels — f = 0.2,
        # large enough to make the bug obvious (1/(1-0.2) = 1.25).
        movie = np.zeros((T, H, W), dtype=np.float32)
        spike_t = rng.choice(T, size=40, replace=False)
        movie[spike_t] = rng.uniform(2.0, 5.0, size=(40, H, W))
        cn = local_correlations_fft(movie)
        assert (cn <= 1.0 + 1e-5).all(), f"CORR exceeded 1: max = {cn.max():.4f}"
        assert (cn >= -1.0 - 1e-5).all(), f"CORR below -1: min = {cn.min():.4f}"


class TestCorrelationPNR:
    def test_output_shapes(self, synth_small):
        movie = synth_small["movie"]
        T, H, W = movie.shape
        cn, pnr = correlation_pnr(movie, sigma=3.0)
        assert cn.shape == (H, W)
        assert pnr.shape == (H, W)

    def test_pnr_positive(self, synth_small):
        movie = synth_small["movie"]
        cn, pnr = correlation_pnr(movie, sigma=3.0)
        assert (pnr >= 0).all()

    def test_neuron_pixels_high_pnr(self, synth):
        """True neuron centres should have higher PNR than random background pixels."""
        movie = synth["movie"]
        centers = synth["centers"]
        _, pnr = correlation_pnr(movie, sigma=3.0)

        neuron_pnr = pnr[centers[:, 0], centers[:, 1]]
        # Sample some background pixels far from neurons
        H, W = movie.shape[1:]
        bg_pixels = []
        for _ in range(20):
            r, c = np.random.randint(5, H - 5), np.random.randint(5, W - 5)
            far = all(abs(r - cr) + abs(c - cc) > 10 for cr, cc in centers)
            if far:
                bg_pixels.append(pnr[r, c])

        if bg_pixels:
            assert neuron_pnr.mean() > np.mean(bg_pixels), (
                "Neuron centres should have higher PNR than background"
            )

    def test_stride_one_unchanged(self, synth_small):
        """stride=1 must produce bit-equivalent results to the no-stride API."""
        movie = synth_small["movie"]
        cn_a, pnr_a = correlation_pnr(movie, sigma=3.0)
        cn_b, pnr_b = correlation_pnr(movie, sigma=3.0, stride=1)
        np.testing.assert_array_equal(cn_a, cn_b)
        np.testing.assert_array_equal(pnr_a, pnr_b)

    def test_stride_speeds_up(self, synth):
        """stride > 1 should be noticeably faster than stride=1.

        Loose threshold (1.3x) to avoid CI flakes; real-world speedup
        is closer to ``stride``.
        """
        import time
        movie = synth["movie"]
        # Warm caches.
        correlation_pnr(movie, sigma=3.0, stride=1)

        t0 = time.perf_counter()
        for _ in range(2):
            correlation_pnr(movie, sigma=3.0, stride=1)
        t_full = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(2):
            correlation_pnr(movie, sigma=3.0, stride=3)
        t_stride = time.perf_counter() - t0

        assert t_stride < t_full / 1.3, (
            f"stride=3 took {t_stride*1e3:.0f}ms vs stride=1 {t_full*1e3:.0f}ms — "
            f"expected at least 1.3x speedup"
        )
