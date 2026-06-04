"""Lazy movie loading and format conversion.

Architecture: zarr is the canonical format for the pipeline.
Every input format has a converter that streams frames to a zarr store
(chunked along time) without loading the full movie into memory.

Adding a new format = adding a new `<format>_to_zarr()` function.

Uses zarr v3 API (zarr >= 3.0).
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import zarr
import zarr.codecs as zcodecs

from minicnmfe._utils import ensure_float32, iter_frames


def _open_array(
    path: Path,
    mode: str,
    shape=None,
    chunks=None,
    dtype=None,
    compression: bool = True,
    clevel: int = 5,
    shuffle: str = "bitshuffle",
) -> zarr.Array:
    """Create or open a zarr array using v3 API.

    When compression=True (default) uses blosc+lz4 — lossless, fast to
    decompress, and typically 2–10× smaller than uncompressed float32 or
    uint8 imaging data.

    ``clevel`` and ``shuffle`` tune the speed/ratio trade-off:
    - ``clevel=5, shuffle="bitshuffle"`` (default): best ratio for float32
      intermediates; ~0.5 GB/s single-thread compress.
    - ``clevel=3, shuffle="shuffle"``: ~3–5× faster compress for uint8 raw
      data, ~10–15 % larger files. Used by ``concat_avis_to_zarr`` so the
      single writer thread doesn't starve decoder workers.
    """
    if mode == "w":
        codecs = None
        if compression:
            shuffle_enum = {
                "bitshuffle": zcodecs.BloscShuffle.bitshuffle,
                "shuffle":    zcodecs.BloscShuffle.shuffle,
                "noshuffle":  zcodecs.BloscShuffle.noshuffle,
            }.get(shuffle)
            if shuffle_enum is None:
                raise ValueError(
                    f"shuffle must be 'bitshuffle' / 'shuffle' / 'noshuffle', "
                    f"got {shuffle!r}"
                )
            codecs = [
                zcodecs.BytesCodec(),
                zcodecs.BloscCodec(
                    cname="lz4",
                    clevel=int(clevel),
                    shuffle=shuffle_enum,
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


# ---------------------------------------------------------------------------
# Pixel-major layout (for true T-streaming extraction)
# ---------------------------------------------------------------------------

def open_zarr_pixel_major(path: str | Path, mode: str = "r") -> zarr.Array:
    """Open a pixel-major zarr (shape ``(H*W, T)``).

    Counterpart to ``open_zarr`` for the layout produced by
    ``transpose_zarr_to_pixel_major``. Pixel-major zarrs let extraction
    read a pixel-row batch with ``O(B·T)`` IO instead of
    ``O(H·W·T)`` for the time-major layout.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Zarr store not found: {path}")
    arr = zarr.open_array(str(path), mode=mode)
    if not isinstance(arr, zarr.Array):
        raise ValueError(f"Expected zarr.Array at {path}, got {type(arr)}")
    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2-D pixel-major zarr (H*W, T), got shape {arr.shape}. "
            f"Use open_zarr() for time-major (T, H, W) stores."
        )
    return arr


def transpose_zarr_to_pixel_major(
    src: str | Path,
    dest: str | Path,
    *,
    pixel_chunk: int = 512,
    time_chunk: int | None = None,
    dtype: str = "float32",
    compression: bool = True,
    src_batch_frames: int = 2000,
    skip_if_exists: bool = True,
    verbose: bool = True,
) -> zarr.Array:
    """Transpose a time-major ``(T, H, W)`` zarr to pixel-major ``(H*W, T)``.

    The pixel ordering matches ``minicnmfe._utils.make_2d``: pixel ``(h, w)``
    lives at flat index ``h * W + w`` (C-order over the spatial axes).
    Reading ``dest[start:end, :]`` therefore returns the same pixel-row
    slice that the in-memory pipeline expects.

    The transpose is a one-time disk pass; afterwards extraction can run
    against ``dest`` without materialising the full movie in RAM.

    **Chunk shape is tuned for the streaming read pattern**, which reads
    contiguous pixel-row batches over the *full* time range. So:
    - ``time_chunk=None`` (default) stores each pixel row's whole time series
      in one chunk — a row read decompresses exactly what's needed, with no
      time-axis read amplification. Caveat for very long recordings: the chunk
      size is ``pixel_chunk * T * 4`` bytes (e.g. ``512 * 60000 * 4`` ≈ 117 MB),
      so cap ``time_chunk`` if that is too large for your write RAM / object
      store.
    - ``pixel_chunk=512`` keeps reads close to the 256-pixel batches used by
      ``compute_W`` / ``update_spatial`` (smaller = less pixel over-read per
      batch; larger = fewer chunk fetches for ``project_onto``'s 4096 batches).

    Args:
        src: Path to the source ``(T, H, W)`` zarr (e.g. ``mc.zarr``).
        dest: Destination zarr path. Created if absent.
        pixel_chunk: Number of pixels per dest chunk along axis 0.
        time_chunk: Frames per dest chunk along axis 1. ``None`` = full ``T``.
        dtype: Output dtype (default ``float32`` for downstream extraction).
        compression: Use blosc lz4+bitshuffle (default ``True``). Keep ``True``
            on a network mount (fewer bytes over the wire); try ``False`` on a
            local SSD (skips per-read decompression, IO-bound only).
        src_batch_frames: Frames read from the source per IO batch.
            Peak RAM ≈ ``src_batch_frames * H * W * 4`` bytes.
        skip_if_exists: If the dest path already exists, return its handle
            without re-writing.
        verbose: Print a progress bar + elapsed time.

    Returns:
        Open zarr.Array with shape ``(H*W, T)``.
    """
    t_start = time.perf_counter()
    src_arr = open_zarr(src)
    T, H, W = src_arr.shape
    n_pixels = H * W

    dest_path = Path(dest)
    if dest_path.exists():
        if skip_if_exists:
            if verbose:
                print(f"Skipping transpose; {dest_path} already exists.")
            return zarr.open_array(str(dest_path), mode="r")
        # Caller asked for a fresh write — remove the old store.
        shutil.rmtree(dest_path)

    time_chunk_eff = T if time_chunk is None else min(time_chunk, T)
    chunks_eff = (min(pixel_chunk, n_pixels), time_chunk_eff)
    if verbose:
        print(
            f"Transposing {src} -> {dest_path}\n"
            f"  src.shape={src_arr.shape}  src.chunks={src_arr.chunks}\n"
            f"  dest.shape=({n_pixels}, {T})  dest.chunks={chunks_eff}  dtype={dtype}"
        )

    dest_arr = _open_array(
        dest_path, "w",
        shape=(n_pixels, T), chunks=chunks_eff,
        dtype=dtype, compression=compression,
    )

    try:
        from tqdm import tqdm as _tqdm
        iterator = _tqdm(range(0, T, src_batch_frames), disable=not verbose,
                         desc="transpose")
    except ImportError:
        iterator = range(0, T, src_batch_frames)

    for t0 in iterator:
        t1 = min(t0 + src_batch_frames, T)
        # Read (B, H, W) -> reshape to (B, H*W) row-major (pixel (h, w) at
        # index h*W + w, matching make_2d) -> transpose to (H*W, B).
        chunk_3d = np.asarray(src_arr[t0:t1], dtype=dtype)
        chunk_2d = chunk_3d.reshape(t1 - t0, n_pixels).T
        # .T returns an F-order view; .copy() makes it C-contiguous for
        # efficient slab-write into the dest zarr (which is C-major chunks).
        dest_arr[:, t0:t1] = np.ascontiguousarray(chunk_2d)

    if verbose:
        print(
            f"Done in {time.perf_counter() - t_start:.1f}s. "
            f"Pixel-major zarr written to: {dest_path}"
        )
    return dest_arr


def stage_zarr_to_local(
    src: str | Path,
    local_dir: str | Path,
    *,
    skip_if_exists: bool = True,
    verbose: bool = True,
) -> zarr.Array:
    """Copy a zarr store to local disk and return the open (read) handle.

    For a store read repeatedly from a network mount — most importantly the
    pixel-major ``Y_flat`` during streaming extraction, which is scanned ~5–6
    times across the BCD loop — staging it to local SSD/tmpfs once turns N
    network read-passes into one network copy + N local passes. Typically the
    single biggest win on a network mount.

    The copy is a plain directory copy (zarr v3 stores are directories), so the
    chunking/compression of ``src`` is preserved verbatim.

    For the auto-derive path (``fit_extract(zarr, output_dir=...)``), setting
    ``CNMFeParams.yflat_dir`` to a local path achieves the same implicitly —
    the transpose then reads ``mc.zarr`` from the network once and writes
    ``Y_flat`` straight to local disk. Use this helper when you already have a
    network-resident ``Y_flat`` (or want to stage ``mc.zarr`` itself).

    Args:
        src: Path to the source zarr store (2-D pixel-major or 3-D time-major).
        local_dir: Destination directory; the store is copied to
            ``local_dir / src.name``.
        skip_if_exists: If the destination already exists, reuse it.
        verbose: Print progress + elapsed time.

    Returns:
        Open zarr.Array handle at the local copy (read mode).
    """
    src_path = Path(src)
    if not src_path.exists():
        raise FileNotFoundError(f"Zarr store not found: {src_path}")
    dest_path = Path(local_dir) / src_path.name

    if dest_path.exists() and skip_if_exists:
        if verbose:
            print(f"Skipping stage; {dest_path} already exists.")
        return zarr.open_array(str(dest_path), mode="r")

    t_start = time.perf_counter()
    if dest_path.exists():
        shutil.rmtree(dest_path)
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Staging {src_path} -> {dest_path} ...")
    shutil.copytree(src_path, dest_path)
    if verbose:
        print(f"Done in {time.perf_counter() - t_start:.1f}s.")
    return zarr.open_array(str(dest_path), mode="r")
