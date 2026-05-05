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
