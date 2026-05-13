"""Tests for rigid motion correction."""

import numpy as np
import pytest

from cnmfe.motion_correction import (
    apply_shift,
    estimate_shifts,
    motion_correction_rigid,
)


def make_frame(H: int = 64, W: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
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
        known = np.array([2.7, -1.3], dtype=np.float32)
        template = apply_shift(frame, known)
        shift = estimate_shifts(frame, template, upsample_factor=20, max_shift=(20, 20))
        np.testing.assert_allclose(shift, known, atol=0.3)

    def test_max_shift_clipping(self):
        frame = make_frame(64, 64)
        shifted = np.roll(frame, 30, axis=0)
        shift = estimate_shifts(frame, shifted, max_shift=(10, 10))
        assert abs(shift[0]) <= 10.0
        assert abs(shift[1]) <= 10.0


class TestApplyShift:
    def test_zero_shift_identity(self):
        frame = make_frame()
        result = apply_shift(frame, np.array([0.0, 0.0]))
        np.testing.assert_allclose(result, frame, atol=1e-4)

    def test_output_shape(self):
        frame = make_frame(48, 64)
        result = apply_shift(frame, np.array([1.0, 2.0]))
        assert result.shape == frame.shape


class TestMotionCorrect:
    def test_static_movie(self):
        """A movie with no motion should have near-zero shifts."""
        frame = make_frame(32, 32)
        movie = np.stack([frame] * 20).astype(np.float32)
        _, shifts = motion_correction_rigid(movie, niter_rig=1, upsample_factor=5)
        assert shifts.shape == (20, 2)
        np.testing.assert_allclose(shifts, 0.0, atol=0.5)

    def test_shifted_movie(self):
        """Known per-frame shifts should be approximately recovered.

        motion_correction_rigid returns the corrections applied (≈ -true_motion),
        so correlation with true_shifts is expected to be strongly negative.
        We check |r| > 0.7 (magnitude, not sign).
        """
        rng = np.random.default_rng(1)
        H, W, T = 48, 48, 60
        frame = make_frame(H, W)

        true_shifts = rng.uniform(-4, 4, (T, 2)).astype(np.float32)
        movie = np.stack(
            [apply_shift(frame, true_shifts[t]) for t in range(T)]
        ).astype(np.float32)

        _, shifts = motion_correction_rigid(movie, niter_rig=2, upsample_factor=10,
                                            max_shift=(10, 10))
        for axis in range(2):
            r = np.corrcoef(shifts[:, axis], true_shifts[:, axis])[0, 1]
            assert abs(r) > 0.7, f"Shift axis {axis} |r|={abs(r):.2f} too low"

    def test_output_shape(self):
        movie = np.random.rand(10, 32, 32).astype(np.float32)
        corrected, shifts = motion_correction_rigid(movie, niter_rig=1)
        assert corrected.shape == (10, 32, 32)
        assert shifts.shape == (10, 2)
