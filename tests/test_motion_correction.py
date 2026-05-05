"""Tests for rigid motion correction."""

import numpy as np
import pytest

from cnmfe.motion_correction import apply_shift, estimate_shifts, motion_correct


def make_frame(H: int = 64, W: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Use a structured image (not pure noise) for reliable registration
    frame = np.zeros((H, W), dtype=np.float32)
    for _ in range(5):
        r = rng.integers(10, H - 10)
        c = rng.integers(10, W - 10)
        frame[r - 3 : r + 4, c - 3 : c + 4] = rng.random()
    return frame


class TestEstimateShifts:
    def test_zero_shift(self):
        frame = make_frame()
        shift = estimate_shifts(frame, frame)
        np.testing.assert_allclose(shift, [0.0, 0.0], atol=0.1)

    def test_integer_shift(self):
        frame = make_frame(64, 64)
        template = np.roll(np.roll(frame, 5, axis=0), 3, axis=1)
        shift = estimate_shifts(frame, template, upsample_factor=1, max_shift=(20, 20))
        np.testing.assert_allclose(shift, [5.0, 3.0], atol=0.6)

    def test_subpixel_shift(self):
        frame = make_frame(64, 64)
        # Apply a known subpixel shift to create the template
        known = np.array([2.7, -1.3], dtype=np.float32)
        template = apply_shift(frame, known)
        shift = estimate_shifts(frame, template, upsample_factor=20, max_shift=(20, 20))
        np.testing.assert_allclose(shift, known, atol=0.15)

    def test_max_shift_clipping(self):
        frame = make_frame(64, 64)
        # Shift far beyond max_shift
        shifted = np.roll(frame, 30, axis=0)
        shift = estimate_shifts(frame, shifted, max_shift=(10, 10))
        assert abs(shift[0]) <= 10.0
        assert abs(shift[1]) <= 10.0


class TestApplyShift:
    def test_zero_shift_identity(self):
        frame = make_frame()
        result = apply_shift(frame, np.array([0.0, 0.0]))
        np.testing.assert_allclose(result, frame, atol=1e-4)

    def test_roundtrip(self):
        frame = make_frame()
        shift = np.array([3.5, -2.1], dtype=np.float32)
        shifted = apply_shift(frame, shift)
        recovered = apply_shift(shifted, -shift)
        # PSNR > 30 dB in the interior (borders have wrap-around artefacts)
        interior = slice(10, -10)
        mse = np.mean((frame[interior, interior] - recovered[interior, interior]) ** 2)
        signal_power = np.mean(frame[interior, interior] ** 2)
        psnr = 10 * np.log10(signal_power / (mse + 1e-12))
        assert psnr > 25.0, f"PSNR={psnr:.1f} dB too low"

    def test_output_shape(self):
        frame = make_frame(48, 64)
        result = apply_shift(frame, np.array([1.0, 2.0]))
        assert result.shape == frame.shape


class TestMotionCorrect:
    def test_static_movie(self):
        """A movie with no motion should have near-zero shifts."""
        frame = make_frame(32, 32)
        movie = np.stack([frame] * 20)
        _, shifts = motion_correct(movie, n_iter=1, upsample_factor=5)
        assert shifts.shape == (20, 2)
        np.testing.assert_allclose(shifts, 0.0, atol=0.5)

    def test_shifted_movie(self):
        """Known per-frame shifts should be approximately recovered.

        motion_correct returns the *corrections* applied (≈ -true_motion),
        so the correlation with true_shifts is expected to be strongly negative.
        We check |r| > 0.7 (magnitude, not sign).
        """
        rng = np.random.default_rng(1)
        H, W, T = 48, 48, 60   # larger movie for stable template estimation
        frame = make_frame(H, W)

        true_shifts = rng.uniform(-4, 4, (T, 2)).astype(np.float32)
        movie = np.stack([apply_shift(frame, true_shifts[t]) for t in range(T)])

        _, shifts = motion_correct(movie, n_iter=2, upsample_factor=10, max_shift=(10, 10))
        # |correlation| should be high (sign depends on convention)
        for axis in range(2):
            r = np.corrcoef(shifts[:, axis], true_shifts[:, axis])[0, 1]
            assert abs(r) > 0.7, f"Shift axis {axis} |correlation| {abs(r):.2f} too low"

    def test_output_shape(self):
        movie = np.random.rand(10, 32, 32).astype(np.float32)
        corrected, shifts = motion_correct(movie, n_iter=1)
        assert np.asarray(corrected).shape == (10, 32, 32)
        assert shifts.shape == (10, 2)
