"""Tests for IO module (zarr save/load)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr

from cnmfe.io import (
    open_zarr,
    open_zarr_pixel_major,
    save_zarr,
    transpose_zarr_to_pixel_major,
)


class TestSaveAndOpenZarr:
    def test_round_trip(self, tmp_path):
        movie = np.random.rand(50, 32, 32).astype(np.float32)
        zarr_path = tmp_path / "test.zarr"
        save_zarr(movie, zarr_path)
        arr = open_zarr(zarr_path)
        np.testing.assert_allclose(np.asarray(arr), movie, rtol=1e-5)

    def test_shape_and_dtype(self, tmp_path):
        movie = np.random.rand(100, 64, 48).astype(np.float64)
        zarr_path = tmp_path / "test.zarr"
        arr = save_zarr(movie, zarr_path, dtype="float32")
        assert arr.shape == (100, 64, 48)
        assert arr.dtype == np.float32

    def test_chunking(self, tmp_path):
        movie = np.random.rand(200, 32, 32).astype(np.float32)
        zarr_path = tmp_path / "test.zarr"
        arr = save_zarr(movie, zarr_path, chunk_t=50)
        assert arr.chunks[0] == 50
        assert arr.chunks[1] == 32
        assert arr.chunks[2] == 32

    def test_open_nonexistent_raises(self, tmp_path):
        with pytest.raises(Exception):
            open_zarr(tmp_path / "does_not_exist.zarr")

    def test_open_bad_shape_raises(self, tmp_path):
        """A 2-D zarr array should raise when opened with open_zarr."""
        zarr_path = tmp_path / "bad.zarr"
        z = zarr.open(str(zarr_path), mode="w", shape=(10, 10), dtype="float32")
        with pytest.raises(ValueError, match="3-D"):
            open_zarr(zarr_path)

    def test_single_frame_access(self, tmp_path):
        """Accessing one frame should not load the full array into memory."""
        movie = np.arange(50 * 32 * 32, dtype=np.float32).reshape(50, 32, 32)
        zarr_path = tmp_path / "test.zarr"
        save_zarr(movie, zarr_path, chunk_t=10)
        arr = open_zarr(zarr_path)
        # Access single frame — zarr loads only the relevant chunk
        frame = arr[5]
        np.testing.assert_allclose(frame, movie[5], rtol=1e-5)


class TestTransposeToPixelMajor:
    """Disk transpose: time-major (T,H,W) → pixel-major (H*W,T).

    Phase F1 of Item 5 (true T-streaming extraction). The transposed
    layout is the canonical input for the streaming extraction path; it
    must match the pixel ordering produced by ``make_2d`` so the rest of
    the pipeline stays agnostic.
    """

    def _save_source(self, tmp_path, T=40, H=12, W=10, seed=0):
        rng = np.random.default_rng(seed)
        src_movie = rng.standard_normal((T, H, W)).astype(np.float32) * 2.0
        src_path = tmp_path / "src.zarr"
        save_zarr(src_movie, src_path, chunk_t=10)
        return src_movie, src_path

    def test_round_trip_matches_make_2d(self, tmp_path):
        """Pixel-major dest must equal ``make_2d(src)`` element-wise."""
        from cnmfe._utils import make_2d

        src_movie, src_path = self._save_source(tmp_path)
        dest_path = tmp_path / "dest.zarr"

        dest_arr = transpose_zarr_to_pixel_major(
            src_path, dest_path,
            pixel_chunk=32, time_chunk=20, src_batch_frames=10,
            verbose=False,
        )

        expected = make_2d(src_movie)               # (H*W, T)
        np.testing.assert_array_equal(np.asarray(dest_arr), expected)

    def test_open_pixel_major_returns_2d(self, tmp_path):
        src_movie, src_path = self._save_source(tmp_path)
        dest_path = tmp_path / "dest.zarr"
        transpose_zarr_to_pixel_major(
            src_path, dest_path, pixel_chunk=32, time_chunk=20, verbose=False
        )

        z = open_zarr_pixel_major(dest_path)
        T, H, W = src_movie.shape
        assert z.shape == (H * W, T)
        assert z.ndim == 2

    def test_skip_if_exists(self, tmp_path):
        """skip_if_exists=True must not overwrite; returns existing handle."""
        src_movie, src_path = self._save_source(tmp_path)
        dest_path = tmp_path / "dest.zarr"
        first = transpose_zarr_to_pixel_major(
            src_path, dest_path, pixel_chunk=32, time_chunk=20, verbose=False
        )
        first_bytes = np.asarray(first).copy()

        # Calling again with skip_if_exists=True (default) is idempotent.
        second = transpose_zarr_to_pixel_major(
            src_path, dest_path, pixel_chunk=32, time_chunk=20, verbose=False
        )
        np.testing.assert_array_equal(np.asarray(second), first_bytes)

    def test_pixel_slice_matches_source_columns(self, tmp_path):
        """Reading pixel rows from dest equals reading the same pixels from src.

        This is the access pattern the streaming extraction will use:
        ``Y_flat[start:end, :]`` returns ``(B, T)`` time series for a
        contiguous pixel batch.
        """
        src_movie, src_path = self._save_source(tmp_path, T=60, H=15, W=10)
        dest_path = tmp_path / "dest.zarr"
        transpose_zarr_to_pixel_major(
            src_path, dest_path, pixel_chunk=20, time_chunk=30, verbose=False
        )
        dest_arr = open_zarr_pixel_major(dest_path)

        T, H, W = src_movie.shape
        # Pick a non-aligned pixel range that straddles a chunk boundary.
        start, end = 25, 75
        chunk = np.asarray(dest_arr[start:end, :])           # (50, T)

        # Convert pixel flat indices back to (h, w) and pick from source.
        expected = np.empty((end - start, T), dtype=np.float32)
        for i, p in enumerate(range(start, end)):
            h, w = divmod(p, W)
            expected[i] = src_movie[:, h, w]
        np.testing.assert_array_equal(chunk, expected)

    def test_open_pixel_major_rejects_3d(self, tmp_path):
        """open_zarr_pixel_major must refuse a time-major (T, H, W) store."""
        _, src_path = self._save_source(tmp_path)
        with pytest.raises(ValueError, match="2-D pixel-major"):
            open_zarr_pixel_major(src_path)
