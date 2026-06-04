"""Shared internal helpers — no external deps beyond numpy."""

from __future__ import annotations

import time
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

import numpy as np

if TYPE_CHECKING:
    from typing import Any


# ---------------------------------------------------------------------------
# Per-stage wall-clock timing
# ---------------------------------------------------------------------------

class StageTimer:
    """Accumulate labelled wall-clock durations and render a summary.

    Used by ``CNMFe.fit_extract`` to surface where time goes — especially the
    repeated passes over an on-disk ``Y_flat`` store in streaming mode, where
    the dominant cost is network/disk IO rather than compute. Repeated labels
    (e.g. ``compute_W`` across BCD iterations) accumulate into one row with a
    call count.

    Example::

        timer = StageTimer()
        with timer.stage("compute_W"):
            ...
        print(timer.summary())
    """

    def __init__(self) -> None:
        self._secs: dict[str, float] = {}
        self._calls: dict[str, int] = {}
        self._order: list[str] = []

    def add(self, label: str, secs: float) -> None:
        """Record ``secs`` against ``label`` (accumulates if repeated)."""
        if label not in self._secs:
            self._secs[label] = 0.0
            self._calls[label] = 0
            self._order.append(label)
        self._secs[label] += secs
        self._calls[label] += 1

    @contextmanager
    def stage(self, label: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(label, time.perf_counter() - t0)

    def summary(self, title: str = "Stage timings") -> str:
        """Return a formatted table: per-label total seconds, call count, share."""
        if not self._order:
            return f"{title}: (no stages timed)"
        total = sum(self._secs.values())
        width = max(len(lbl) for lbl in self._order)
        lines = [f"{title} (total {total:.1f}s):"]
        for lbl in self._order:
            secs = self._secs[lbl]
            calls = self._calls[lbl]
            count = f" x{calls}" if calls > 1 else ""
            pct = (100.0 * secs / total) if total > 0 else 0.0
            lines.append(f"  {lbl:<{width}}  {secs:7.1f}s  {pct:4.0f}%{count}")
        return "\n".join(lines)


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

    Works with zarr arrays, numpy arrays, and any object supporting .shape or __len__.
    """
    T = movie.shape[0] if hasattr(movie, "shape") else len(movie)
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


# ---------------------------------------------------------------------------
# Footprint centre
# ---------------------------------------------------------------------------

def footprint_center(
    ai: np.ndarray,
    dims: "tuple[int, int] | None" = None,
    smooth_sigma: float = 1.0,
) -> "tuple[int, int]":
    """Locate a footprint's centre as the argmax of a lightly Gaussian-smoothed
    intensity image. Robust to non-convex / multi-peak shapes, unlike the
    intensity-weighted COM which lands between peaks (in the hole of a donut,
    in the dim space between two cells of a multi-cell footprint).

    Args:
        ai: ``(H, W)`` or flat ``(H*W,)`` non-negative footprint.
        dims: ``(H, W)``; required when ``ai`` is 1D.
        smooth_sigma: σ of the pre-argmax Gaussian smooth (px). 1.0 prevents
            a single noisy hot pixel from winning while leaving the actual
            cell peak intact. Set to 0 to skip smoothing.

    Returns:
        ``(cy, cx)`` int pixel coordinates. Returns the image centre
        ``(H // 2, W // 2)`` if the footprint is all-zero.
    """
    if ai.ndim == 1:
        if dims is None:
            raise ValueError("dims=(H, W) required when ai is 1-D")
        H, W = dims
        ai2d = ai.reshape(H, W)
    else:
        ai2d = ai
        H, W = ai2d.shape

    if not (ai2d > 0).any():
        return H // 2, W // 2

    if smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        smoothed = gaussian_filter(ai2d.astype(np.float32, copy=False),
                                   sigma=float(smooth_sigma))
    else:
        smoothed = ai2d
    cy, cx = np.unravel_index(int(np.argmax(smoothed)), (H, W))
    return int(cy), int(cx)
