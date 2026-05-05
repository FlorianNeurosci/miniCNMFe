"""Shared internal helpers — no external deps beyond numpy."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Iterator

import numpy as np

if TYPE_CHECKING:
    from typing import Any


def make_2d(movie: np.ndarray) -> np.ndarray:
    """(T, H, W) → (H*W, T) in Fortran order (pixels × time).

    Fortran order means spatial dimensions vary fastest, matching MATLAB/CaImAn convention
    and making column-slices (single-pixel time series) contiguous in memory.
    """
    T, H, W = movie.shape
    return movie.reshape(T, H * W, order="C").T  # equivalent to F-order pixel layout


def make_3d(Y_flat: np.ndarray, dims: tuple[int, int]) -> np.ndarray:
    """(H*W, T) → (T, H, W). Inverse of make_2d."""
    H, W = dims
    return Y_flat.T.reshape(-1, H, W)


def ensure_float32(arr: np.ndarray) -> np.ndarray:
    """Return arr as float32 without copying if already correct dtype."""
    if arr.dtype == np.float32:
        return arr
    return arr.astype(np.float32)


def iter_frames(movie: "Any", batch_size: int = 200) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (start_idx, batch) where batch is float32 shape (B, H, W).

    Works with zarr arrays, numpy arrays, and any object with __len__ and __getitem__.
    """
    T = len(movie)
    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        batch = np.asarray(movie[start:end], dtype=np.float32)
        yield start, batch


# ---------------------------------------------------------------------------
# GPU / CPU array-module helpers
# ---------------------------------------------------------------------------

def get_xp(device: str = "cpu"):
    """Return cupy if device='cuda' and CuPy is installed, else numpy.

    Accepts 'cpu', 'cuda', or 'gpu' (alias for 'cuda').
    Falls back to numpy with a warning when CuPy is requested but unavailable.
    """
    if device in ("cuda", "gpu"):
        try:
            import cupy as cp
            cp.cuda.runtime.getDeviceCount()   # raises if no GPU
            return cp
        except Exception as exc:
            warnings.warn(
                f"GPU requested but CuPy/CUDA not available ({exc}); "
                "falling back to CPU.",
                stacklevel=2,
            )
    return np


def to_numpy(arr: "Any") -> np.ndarray:
    """Return *arr* as a numpy array, moving off GPU if necessary."""
    try:
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
    except ImportError:
        pass
    return np.asarray(arr)
