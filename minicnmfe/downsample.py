"""Spatial + temporal downsampling of a (T, H, W) zarr movie.

Stage 1 of the staged "downsample-once" workflow: bin a movie in space
(``ssub``) and time (``tsub``) up front, then run motion correction, extraction
and evaluation entirely on the smaller movie. Use
``CNMFeParams.downscaled(ssub, tsub)`` so parameters expressed in native units
are rescaled to the downsampled grid.

The bin is a streaming block-mean (peak RAM ~ ``src_batch_frames·H·W·4`` bytes,
independent of T), structurally mirroring
``minicnmfe.io.transpose_zarr_to_pixel_major``. Pixel ordering is preserved, so the
output is a normal time-major movie that every downstream stage accepts
unchanged.

NOTE: temporal binning here happens BEFORE motion correction — frames within a
``tsub`` group are averaged prior to registration. Fine for slow drift / small
``tsub``; with large intra-bin motion it blurs neurons. This is the deliberate
trade-off of the downsample-once design.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import zarr
except ImportError:  # pragma: no cover - zarr is a hard dep in practice
    zarr = None

from minicnmfe.io import _open_array, open_zarr


def downsample_movie(
    src: "str | Path",
    dest: "str | Path",
    *,
    ssub: int = 1,
    tsub: int = 1,
    src_batch_frames: int = 2000,
    chunk_t: int = 500,
    dtype: str = "float32",
    compression: bool = True,
    skip_if_exists: bool = True,
    write_meta: bool = True,
    verbose: bool = True,
) -> "zarr.Array":
    """Block-mean downsample a ``(T, H, W)`` zarr by ``ssub`` (space)/``tsub`` (time).

    Output shape is ``(T // tsub, H // ssub, W // ssub)``. When a dimension is
    not an exact multiple of its factor, the trailing remainder is dropped
    (and reported); this keeps the bin a clean mean over full blocks.

    Args:
        src: Path to the source ``(T, H, W)`` zarr.
        dest: Destination zarr path. Created if absent.
        ssub: Spatial bin factor (``H`` and ``W`` divided by this).
        tsub: Temporal bin factor (number of frames averaged per output frame).
        src_batch_frames: Source frames read per IO batch (RAM knob).
        chunk_t: Output zarr time-chunk.
        dtype: Output dtype (default ``float32`` for downstream extraction).
        compression: Use blosc lz4+bitshuffle (default ``True``).
        skip_if_exists: If ``dest`` exists, return its handle without rewriting.
        write_meta: Write ``ds_meta.json`` (ssub/tsub/orig dims) next to ``dest``.
        verbose: Print progress.

    Returns:
        Open ``zarr.Array`` with shape ``(T // tsub, H // ssub, W // ssub)``.
    """
    if zarr is None:
        raise RuntimeError("zarr is required for downsample_movie")
    if ssub < 1 or tsub < 1:
        raise ValueError(f"ssub and tsub must be >= 1 (got {ssub}, {tsub})")

    src_arr = open_zarr(src)
    T, H, W = (int(src_arr.shape[0]), int(src_arr.shape[1]), int(src_arr.shape[2]))

    # Trim each axis down to an exact multiple of its bin factor.
    T_use, H_use, W_use = (T // tsub) * tsub, (H // ssub) * ssub, (W // ssub) * ssub
    T_out, H_out, W_out = T_use // tsub, H_use // ssub, W_use // ssub
    if verbose and (T_use != T or H_use != H or W_use != W):
        print(
            f"downsample: trimming ({T},{H},{W}) -> ({T_use},{H_use},{W_use}) "
            f"so dims divide evenly by (tsub={tsub}, ssub={ssub})."
        )

    dest_path = Path(dest)
    meta = {
        "ssub": int(ssub), "tsub": int(tsub),
        "orig_dims": [H, W], "orig_T": T,
        "ds_dims": [H_out, W_out], "ds_T": T_out,
        "src": str(src), "dest": str(dest_path),
    }

    if dest_path.exists():
        if skip_if_exists:
            if verbose:
                print(f"Skipping downsample; {dest_path} already exists.")
            if write_meta:
                _write_meta(dest_path, meta, verbose)
            return zarr.open_array(str(dest_path), mode="r")
        import shutil
        shutil.rmtree(dest_path)

    chunks_eff = (min(chunk_t, T_out), H_out, W_out)
    if verbose:
        print(
            f"Downsampling {src} -> {dest_path}\n"
            f"  src.shape=({T},{H},{W})  ssub={ssub} tsub={tsub}\n"
            f"  dest.shape=({T_out},{H_out},{W_out})  dest.chunks={chunks_eff}  dtype={dtype}"
        )

    dest_arr = _open_array(
        dest_path, "w",
        shape=(T_out, H_out, W_out), chunks=chunks_eff,
        dtype=dtype, compression=compression,
    )

    # Read ~src_batch_frames input frames per pass => out_batch output frames.
    out_batch = max(1, src_batch_frames // tsub)
    try:
        from tqdm import tqdm as _tqdm
        iterator = _tqdm(range(0, T_out, out_batch), disable=not verbose,
                         desc="downsample")
    except ImportError:
        iterator = range(0, T_out, out_batch)

    for o0 in iterator:
        o1 = min(o0 + out_batch, T_out)
        n_out = o1 - o0
        # Corresponding input frame span (exact multiple of tsub).
        chunk = np.asarray(
            src_arr[o0 * tsub:o1 * tsub, :H_use, :W_use], dtype=np.float32
        )
        # Temporal bin: (n_out, tsub, H_use, W_use) -> mean over tsub.
        if tsub > 1:
            chunk = chunk.reshape(n_out, tsub, H_use, W_use).mean(axis=1)
        # Spatial bin: (n_out, H_out, ssub, W_out, ssub) -> mean over both ssub.
        if ssub > 1:
            chunk = chunk.reshape(n_out, H_out, ssub, W_out, ssub).mean(axis=(2, 4))
        dest_arr[o0:o1] = chunk.astype(dtype)

    if write_meta:
        _write_meta(dest_path, meta, verbose)
    if verbose:
        print(f"Done. Downsampled movie written to: {dest_path}")
    return dest_arr


def _write_meta(dest_path: Path, meta: dict, verbose: bool) -> None:
    meta_path = dest_path.parent / "ds_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    if verbose:
        print(f"Wrote downsample metadata: {meta_path}")


# ---------------------------------------------------------------------------
# Upsampling: map downsampled outputs back onto the native grid / frame rate.
#
# This is plain interpolation of already-extracted, downsampled results — it
# re-expresses footprints/traces on the native grid for overlay/plotting, it
# does NOT recover detail the binning discarded (the native movie is gone in
# the downsample-once workflow). See CNMFe.upsample_to_native.
# ---------------------------------------------------------------------------

def upsample_footprints(A, ds_dims, native_dims, order: int = 1):
    """Interpolate sparse footprints ``(H_ds·W_ds, K)`` to ``(H·W, K)``.

    Each column is reshaped to ``(H_ds, W_ds)`` (C-order, ``h*W+w`` — matching
    ``minicnmfe._utils.make_3d``), resized to the *exact* native ``(H, W)`` with
    ``cv2.resize`` (so a non-integer or trimmed factor is handled), and flattened
    back. ``order=1`` → bilinear (smooth), ``order=0`` → nearest (block-exact
    inverse of the block-mean downsample).

    Returns a ``scipy.sparse.csc_matrix`` of shape ``(H·W, K)`` float32.
    """
    import cv2
    import scipy.sparse as sp

    H_ds, W_ds = int(ds_dims[0]), int(ds_dims[1])
    H, W = int(native_dims[0]), int(native_dims[1])
    A = A.tocsc()
    K = A.shape[1]
    if A.shape[0] != H_ds * W_ds:
        raise ValueError(
            f"A has {A.shape[0]} rows but ds_dims {ds_dims} implies "
            f"{H_ds * W_ds}."
        )
    interp = cv2.INTER_LINEAR if order == 1 else cv2.INTER_NEAREST
    if K == 0:
        return sp.csc_matrix((H * W, 0), dtype=np.float32)

    cols = []
    for k in range(K):
        col = np.asarray(A[:, k].todense(), dtype=np.float32).reshape(H_ds, W_ds)
        up = cv2.resize(col, (W, H), interpolation=interp)   # (H, W)
        cols.append(sp.csc_matrix(up.reshape(-1, 1).astype(np.float32)))
    return sp.hstack(cols, format="csc")


def upsample_traces(C, native_T: int, kind: str = "linear"):
    """Interpolate traces ``(K, T_ds)`` to ``(K, native_T)`` along time.

    ``kind="linear"`` uses ``np.interp`` (endpoints preserved); ``kind="nearest"``
    repeats the closest source sample. Maps ``linspace(0,1,T_ds)`` onto
    ``linspace(0,1,native_T)`` so the exact ``native_T`` is hit despite trimming.
    """
    C = np.asarray(C, dtype=np.float32)
    K, T_ds = C.shape
    native_T = int(native_T)
    if T_ds == native_T:
        return C.copy()
    if T_ds == 0 or K == 0:
        return np.empty((K, native_T), dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, T_ds)
    x_new = np.linspace(0.0, 1.0, native_T)
    if kind == "nearest":
        idx = np.clip(np.round(x_new * (T_ds - 1)).astype(int), 0, T_ds - 1)
        return C[:, idx].astype(np.float32)
    out = np.empty((K, native_T), dtype=np.float32)
    for k in range(K):
        out[k] = np.interp(x_new, x_old, C[k])
    return out
