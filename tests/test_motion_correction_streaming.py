"""Tests for the streaming / zarr-backed motion correction path.

These tests verify that:
1. The streaming path (zarr in, zarr out) and the in-memory path produce the
   same shifts on the same movie.
2. Parallel (n_jobs > 1) and serial (n_jobs == 1) produce the same shifts.
3. The output zarr is created with the correct shape, chunks, and dtype, and
   the corrected frames match the in-memory result.
4. Multi-pass (niter_rig > 1) ping-pong works end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from cnmfe.motion_correction import apply_shift, motion_correction_rigid


def _make_movie(T=40, H=48, W=48, seed=0):
    """A small synthetic movie with known per-frame shifts.

    Builds a clean frame with a few bright blobs, then applies a smooth
    drifting shift per frame. Good enough for shift recovery + identity checks.
    """
    rng = np.random.default_rng(seed)
    frame = np.zeros((H, W), dtype=np.float32)
    for _ in range(6):
        r = rng.integers(8, H - 8)
        c = rng.integers(8, W - 8)
        frame[r - 3:r + 4, c - 3:c + 4] = rng.random()
    shifts = rng.uniform(-3, 3, (T, 2)).astype(np.float32)
    movie = np.stack([apply_shift(frame, shifts[t]) for t in range(T)]).astype(np.float32)
    return movie, shifts


def _save_zarr(arr: np.ndarray, path: Path, chunk_t: int = 10) -> zarr.Array:
    """Persist a (T, H, W) numpy array as a zarr store for streaming tests."""
    T, H, W = arr.shape
    z = zarr.open_array(
        str(path), mode="w", shape=(T, H, W),
        chunks=(chunk_t, H, W), dtype="float32",
    )
    z[:] = arr
    return z


class TestStreamingPath:
    def test_zarr_in_zarr_out_matches_in_memory(self, tmp_path):
        """Streaming MC (zarr→zarr) must produce the same shifts as in-memory MC."""
        movie, _true = _make_movie(T=30, H=48, W=48, seed=1)

        # In-memory reference
        _, shifts_mem = motion_correction_rigid(
            movie, max_shift=(8, 8), gSig_filt=None,
            upsample_factor=10, niter_rig=1, verbose=False,
        )

        # Streaming via zarr
        src_path = tmp_path / "src.zarr"
        out_path = tmp_path / "mc.zarr"
        src = _save_zarr(movie, src_path, chunk_t=10)

        out, shifts_str = motion_correction_rigid(
            src, output_path=out_path, max_shift=(8, 8), gSig_filt=None,
            upsample_factor=10, niter_rig=1, batch_size=7,
            template_max_frames=1000, verbose=False,
        )

        # Shifts must match exactly (same algorithm, same template)
        np.testing.assert_allclose(shifts_str, shifts_mem, atol=1e-4)

        # Output zarr exists, has correct shape and dtype
        assert isinstance(out, zarr.Array)
        assert out.shape == (30, 48, 48)
        assert np.dtype(out.dtype) == np.dtype("float32")

    def test_streaming_corrected_frames_match_in_memory(self, tmp_path):
        """Corrected frames written to zarr must equal the in-memory result."""
        movie, _ = _make_movie(T=20, H=40, W=40, seed=2)

        corrected_mem, _ = motion_correction_rigid(
            movie, max_shift=(6, 6), gSig_filt=None,
            upsample_factor=5, niter_rig=1, verbose=False,
        )

        src = _save_zarr(movie, tmp_path / "src.zarr", chunk_t=8)
        out_path = tmp_path / "mc.zarr"
        out, _ = motion_correction_rigid(
            src, output_path=out_path, max_shift=(6, 6), gSig_filt=None,
            upsample_factor=5, niter_rig=1, batch_size=8, verbose=False,
        )

        corrected_streamed = np.asarray(out[:])
        np.testing.assert_allclose(corrected_streamed, corrected_mem, atol=1e-4)

    def test_zarr_input_without_output_path_raises(self, tmp_path):
        """A zarr input with no output_path must raise: forces explicit RAM control."""
        movie, _ = _make_movie(T=10, H=32, W=32, seed=3)
        src = _save_zarr(movie, tmp_path / "src.zarr", chunk_t=5)

        with pytest.raises(ValueError, match="output_path"):
            motion_correction_rigid(src, verbose=False)

    def test_numpy_input_with_output_path_uses_streaming(self, tmp_path):
        """Numpy input + output_path should also route to streaming and write zarr."""
        movie, _ = _make_movie(T=20, H=40, W=40, seed=4)
        out_path = tmp_path / "mc.zarr"
        out, shifts = motion_correction_rigid(
            movie, output_path=out_path, max_shift=(6, 6), gSig_filt=None,
            upsample_factor=5, niter_rig=1, batch_size=8, verbose=False,
        )
        assert isinstance(out, zarr.Array)
        assert out_path.exists()
        assert shifts.shape == (20, 2)

    def test_multi_pass_ping_pong(self, tmp_path):
        """niter_rig=2 should run two passes via scratch zarrs and land at output_path."""
        movie, _ = _make_movie(T=25, H=40, W=40, seed=5)

        # Reference: in-memory niter_rig=2
        _, shifts_mem = motion_correction_rigid(
            movie, max_shift=(6, 6), gSig_filt=None,
            upsample_factor=5, niter_rig=2, verbose=False,
        )

        src = _save_zarr(movie, tmp_path / "src.zarr", chunk_t=8)
        out_path = tmp_path / "mc.zarr"
        out, shifts_str = motion_correction_rigid(
            src, output_path=out_path, max_shift=(6, 6), gSig_filt=None,
            upsample_factor=5, niter_rig=2, batch_size=8, verbose=False,
        )

        # Final output landed at the requested path
        assert isinstance(out, zarr.Array)
        assert out_path.exists()

        # Scratch directories are cleaned up
        assert not (out_path.parent / f".{out_path.name}.scratch_a.zarr").exists()
        assert not (out_path.parent / f".{out_path.name}.scratch_b.zarr").exists()

        # Shifts match in-memory result
        np.testing.assert_allclose(shifts_str, shifts_mem, atol=1e-4)

    def test_output_chunk_t_override(self, tmp_path):
        """User-specified output_chunk_t controls the output zarr chunking."""
        movie, _ = _make_movie(T=20, H=32, W=32, seed=6)
        src = _save_zarr(movie, tmp_path / "src.zarr", chunk_t=10)
        out, _ = motion_correction_rigid(
            src, output_path=tmp_path / "mc.zarr",
            max_shift=(5, 5), gSig_filt=None, upsample_factor=5, niter_rig=1,
            batch_size=8, output_chunk_t=4, verbose=False,
        )
        assert out.chunks[0] == 4


class TestParallelism:
    def test_n_jobs_equals_one_vs_serial(self):
        """n_jobs=1 path is the explicit serial branch; sanity-check it works."""
        movie, _ = _make_movie(T=12, H=32, W=32, seed=7)
        _, shifts = motion_correction_rigid(
            movie, max_shift=(5, 5), gSig_filt=None,
            upsample_factor=5, niter_rig=1, n_jobs=1, verbose=False,
        )
        assert shifts.shape == (12, 2)

    @pytest.mark.parametrize("input_mode", ["numpy", "zarr"])
    def test_parallel_matches_serial(self, tmp_path, input_mode):
        """n_jobs=2 must produce the same shifts as n_jobs=1, on both the
        in-memory (``numpy``) and streaming (``zarr``) input paths."""
        movie, _ = _make_movie(T=20, H=40, W=40, seed=8)

        if input_mode == "numpy":
            _, shifts_s = motion_correction_rigid(
                movie, max_shift=(6, 6), gSig_filt=None,
                upsample_factor=5, niter_rig=1, n_jobs=1, verbose=False,
            )
            _, shifts_p = motion_correction_rigid(
                movie, max_shift=(6, 6), gSig_filt=None,
                upsample_factor=5, niter_rig=1, n_jobs=2, verbose=False,
            )
        else:
            src = _save_zarr(movie, tmp_path / "src.zarr", chunk_t=8)
            _, shifts_s = motion_correction_rigid(
                src, output_path=tmp_path / "mc_s.zarr",
                max_shift=(6, 6), gSig_filt=None, upsample_factor=5,
                niter_rig=1, n_jobs=1, batch_size=8, verbose=False,
            )
            _, shifts_p = motion_correction_rigid(
                src, output_path=tmp_path / "mc_p.zarr",
                max_shift=(6, 6), gSig_filt=None, upsample_factor=5,
                niter_rig=1, n_jobs=2, batch_size=8, verbose=False,
            )
        np.testing.assert_allclose(shifts_p, shifts_s, atol=1e-4)


class TestWithHighPass:
    """The 1-photon path (gSig_filt set) is the common production case — exercise it."""

    def test_streaming_matches_in_memory_with_gsig(self, tmp_path):
        movie, _ = _make_movie(T=25, H=48, W=48, seed=10)
        _, shifts_mem = motion_correction_rigid(
            movie, max_shift=(6, 6), gSig_filt=3,
            upsample_factor=10, niter_rig=1, verbose=False,
        )
        src = _save_zarr(movie, tmp_path / "src.zarr", chunk_t=10)
        _, shifts_str = motion_correction_rigid(
            src, output_path=tmp_path / "mc.zarr",
            max_shift=(6, 6), gSig_filt=3, upsample_factor=10,
            niter_rig=1, batch_size=8, verbose=False,
        )
        np.testing.assert_allclose(shifts_str, shifts_mem, atol=1e-3)
