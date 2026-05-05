"""Tests for IO module (zarr save/load)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr

from cnmfe.io import open_zarr, save_zarr


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
