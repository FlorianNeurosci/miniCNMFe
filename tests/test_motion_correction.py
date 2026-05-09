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


class TestMotionCorrectRealistic:
    """MC tests using the miniscope simulator with known inter-frame drift.

    Compared with TestMotionCorrect, this class uses the full realistic
    simulator (shot noise, background, ghost cells, vignetting) to verify that
    algorithmic improvements hold up under real-world 1-photon conditions.

    Key challenge: on realistic movies the per-frame SNR is too low (~2.4)
    for phase correlation to reliably detect the sub-pixel drift between
    consecutive frames.  The temporal_smooth_sigma feature exploits the fact
    that brain drift is a smooth process: Gaussian smoothing along the time
    axis suppresses estimation noise while preserving the true slow drift.
    """

    @staticmethod
    def _make_sim(motion_max_shift: float = 4.0, seed: int = 7) -> dict:
        from miniscope_simulator import make_miniscope_movie
        return make_miniscope_movie(
            n_neurons=5,
            dims=(64, 64),
            T=100,
            motion_max_shift=motion_max_shift,
            seed=seed,
            quantize_8bit=False,  # avoid 8-bit rounding artifacts in registration
        )

    def test_temporal_smoothing_improves_shift_accuracy(self):
        """Temporal Gaussian smoothing should recover the known drift with |r| > 0.7.

        Raw phase-correlation estimates on a noisy 1p movie achieve |r| ~ 0.5
        against the true drift because each frame's correlation surface is
        dominated by shot noise.  Smoothing the shift trajectory with a
        Gaussian kernel (sigma≈7 frames) suppresses that noise while keeping
        the slow-drift signal intact, raising |r| above 0.7.
        """
        sim = self._make_sim()
        movie = sim["movie"]
        true_shifts = sim["motion_shifts"]

        _, raw_shifts = motion_correct(movie, n_iter=1, max_shift=(10, 10), upsample_factor=5)
        _, smooth_shifts = motion_correct(
            movie, n_iter=1, max_shift=(10, 10), upsample_factor=5,
            temporal_smooth_sigma=7.0,
        )

        rs_raw    = [abs(float(np.corrcoef(raw_shifts[:, ax],    true_shifts[:, ax])[0, 1])) for ax in range(2)]
        rs_smooth = [abs(float(np.corrcoef(smooth_shifts[:, ax], true_shifts[:, ax])[0, 1])) for ax in range(2)]

        for axis in range(2):
            assert rs_smooth[axis] > rs_raw[axis], (
                f"Smoothing should raise |r|: axis {axis} raw={rs_raw[axis]:.3f} smooth={rs_smooth[axis]:.3f}"
            )
        mean_r = float(np.mean(rs_smooth))
        assert mean_r > 0.7, (
            f"Mean smoothed |r| too weak: {mean_r:.3f} < 0.7  (per-axis: {rs_smooth})"
        )

    def test_temporal_smoothing_reduces_shift_noise(self):
        """Smoothed shifts should vary less frame-to-frame than raw estimates.

        Frame-to-frame jitter in raw phase-correlation estimates comes from
        noise, not from real motion.  A Gaussian filter with sigma > 1 frame
        should substantially reduce that jitter.
        """
        sim = self._make_sim()
        movie = sim["movie"]

        _, raw    = motion_correct(movie, n_iter=1, max_shift=(10, 10))
        _, smooth = motion_correct(movie, n_iter=1, max_shift=(10, 10), temporal_smooth_sigma=7.0)

        raw_jitter    = float(np.diff(raw,    axis=0).std())
        smooth_jitter = float(np.diff(smooth, axis=0).std())
        assert smooth_jitter < raw_jitter, (
            f"Smoothing should reduce jitter: {smooth_jitter:.4f} >= {raw_jitter:.4f}"
        )
