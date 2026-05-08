"""Lazy movie loading and format conversion.

Architecture: zarr is the canonical format for the pipeline.
Every input format has a converter that streams frames to a zarr store
(chunked along time) without loading the full movie into memory.

Adding a new format = adding a new `<format>_to_zarr()` function.

Uses zarr v3 API (zarr >= 3.0).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import zarr
import zarr.codecs as zcodecs

from cnmfe._utils import ensure_float32, iter_frames


def _open_array(
    path: Path,
    mode: str,
    shape=None,
    chunks=None,
    dtype=None,
    compression: bool = True,
) -> zarr.Array:
    """Create or open a zarr array using v3 API.

    When compression=True (default) uses blosc+lz4+bitshuffle — lossless,
    fast to decompress, and typically 2–10× smaller than uncompressed float32
    or uint8 imaging data.
    """
    if mode == "w":
        codecs = None
        if compression:
            codecs = [
                zcodecs.BytesCodec(),
                zcodecs.BloscCodec(
                    cname="lz4",
                    clevel=5,
                    shuffle=zcodecs.BloscShuffle.bitshuffle,
                ),
            ]
        return zarr.open_array(
            str(path),
            mode="w",
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            codecs=codecs,
        )
    else:
        return zarr.open_array(str(path), mode=mode)


# ---------------------------------------------------------------------------
# Converters: input format → zarr
# ---------------------------------------------------------------------------

def avi_to_zarr(
    src: str | Path,
    dest: str | Path,
    chunk_t: int = 100,
    grayscale: bool = True,
    dtype: str = "uint8",
    compression: bool = True,
) -> zarr.Array:
    """Convert AVI/MP4 to a time-chunked zarr store.

    Reads frames one batch at a time via imageio-ffmpeg — the full movie is
    never in memory at once. Grayscale conversion averages RGB channels.

    The default dtype is uint8 (matching the natural bit depth of miniscope
    AVIs). The pipeline loads with np.asarray(movie, dtype=np.float32) so the
    float32 conversion happens in RAM, not on disk — keeping the zarr 4× smaller.
    Compression is lossless (blosc lz4 + bitshuffle).

    Args:
        src: Path to AVI/MP4 file.
        dest: Output zarr store path (directory). Created if absent.
        chunk_t: Number of frames per time chunk.
        grayscale: Average RGB channels to produce (T, H, W) output.
        dtype: On-disk dtype (default "uint8" for 8-bit miniscope data).
        compression: Use blosc lz4+bitshuffle compression (default True).

    Returns:
        Open zarr.Array with shape (T, H, W).
    """
    import imageio.v3 as iio

    src = Path(src)
    dest = Path(dest)

    props = iio.improps(src, plugin="pyav")
    T = int(props.n_images)
    # props.shape is (T, H, W) or (T, H, W, C) when the full video is described;
    # skip the leading time axis so we always get the spatial (H, W).
    _s = props.shape
    if len(_s) >= 3 and _s[0] == T:
        H, W = int(_s[1]), int(_s[2])
    else:
        H, W = int(_s[0]), int(_s[1])

    store = _open_array(dest, "w", shape=(T, H, W), chunks=(chunk_t, H, W),
                        dtype=dtype, compression=compression)

    for start, batch in _read_video_batches(src, batch_size=chunk_t, grayscale=grayscale):
        end = min(start + len(batch), T)
        store[start:end] = batch.astype(dtype)

    return store


def _read_video_batches(
    path: Path, batch_size: int, grayscale: bool
) -> Iterator[tuple[int, np.ndarray]]:
    import imageio.v3 as iio

    frames: list[np.ndarray] = []
    start = 0
    idx = 0
    for frame in iio.imiter(path, plugin="pyav"):
        frame = np.asarray(frame)  # keep natural dtype (uint8 for 8-bit AVIs)
        if grayscale and frame.ndim == 3:
            frame = frame.mean(axis=-1)  # caller's .astype(dtype) handles final cast
        frames.append(frame)
        idx += 1
        if len(frames) == batch_size:
            yield start, np.stack(frames, axis=0)
            start = idx
            frames = []
    if frames:
        yield start, np.stack(frames, axis=0)


# Future additions follow the same pattern:
# def tiff_to_zarr(src, dest, ...) -> zarr.Array: ...
# def hdf5_to_zarr(src, dest, key="data", ...) -> zarr.Array: ...


# ---------------------------------------------------------------------------
# Zarr access
# ---------------------------------------------------------------------------

def open_zarr(path: str | Path, mode: str = "r") -> zarr.Array:
    """Open an existing zarr store. Returns array with shape (T, H, W)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Zarr store not found: {path}")
    arr = zarr.open_array(str(path), mode=mode)
    if not isinstance(arr, zarr.Array):
        raise ValueError(f"Expected zarr.Array at {path}, got {type(arr)}")
    if arr.ndim != 3:
        raise ValueError(f"Expected 3-D zarr array (T, H, W), got shape {arr.shape}")
    return arr


def save_zarr(
    arr: np.ndarray,
    path: str | Path,
    chunk_t: int = 100,
    dtype: str = "float32",
    compression: bool = True,
) -> zarr.Array:
    """Persist an in-memory (T, H, W) array to a zarr store.

    Useful for saving small intermediates (e.g. motion-corrected chunks).
    Returns the open zarr.Array.
    """
    arr = ensure_float32(arr) if dtype == "float32" else arr.astype(dtype)
    T, H, W = arr.shape
    store = _open_array(Path(path), "w", shape=(T, H, W), chunks=(chunk_t, H, W),
                        dtype=dtype, compression=compression)
    for start, batch in iter_frames(arr, batch_size=chunk_t):
        end = start + len(batch)
        store[start:end] = batch
    return store
