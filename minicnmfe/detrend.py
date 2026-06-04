"""Per-pixel temporal detrend preprocessing for a (T, H, W) zarr movie.

Subtracts a rolling lower-percentile baseline from each pixel along the time
axis. Robust to bright transients (the percentile window is dominated by
inter-event baseline, not spikes) and to step jumps in luminance (the window
slides past the step in ``window_frames`` and the new baseline takes over).

This is a *preprocessing* step that runs between motion correction and
extraction: the LED warm-up / focus-step pattern that produces
all-components-fire-in-lockstep OASIS spike trains in the first ~N frames of a
session lives in the per-pixel baseline drift, which neither
``ar_detrend_order`` nor ``temporal_detrend_order`` can heal because those run
*after* greedy init has already seeded the ghost components.

The implementation is a single overlapping-batches pass over the source zarr:
each batch reads the central ``batch_t`` frames plus a half-window of overlap
on each side, computes the baseline at anchor frames spaced ``anchor_stride``
apart and linearly interpolates between them, then writes only the central
frames to the destination. Total disk IO is
``(1 + window_frames / batch_t) * T * H * W`` instead of the
``3 × T × H × W`` a transpose / detrend / transpose-back pipeline would cost.
Anchoring trades a tiny smoothing of the baseline (which is slow by
construction — it's a 30 s lower percentile) for a ~20× CPU win over a
per-frame ``scipy.ndimage.percentile_filter`` sweep.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import zarr
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from minicnmfe.io import _open_array, open_zarr


def _anchored_percentile_baseline(
    block_slab: np.ndarray,
    window_frames: int,
    percentile: float,
    anchor_stride: int,
) -> np.ndarray:
    """Per-pixel rolling lower-percentile baseline via anchor + linear interp.

    For each anchor frame ``t`` (spaced ``anchor_stride`` apart, plus the last
    frame), computes ``np.percentile(block[t - half : t + half + 1],
    percentile, axis=0)``. The per-anchor baselines are then linearly
    interpolated along time onto the full ``(T, H, W)`` grid in one vectorised
    pass.

    The exact per-frame rolling percentile that ``scipy.ndimage.percentile_filter``
    would produce changes very slowly with t (the lower percentile of a long
    window is a slow baseline by design); sparse anchors + linear interpolation
    reproduce it to within a small smoothing error while skipping the O(T)
    per-output-pixel sliding-window machinery.

    Module-level so joblib's spawn-pickling works (matches the convention in
    ``minicnmfe/motion_correction.py``).
    """
    T_block = block_slab.shape[0]
    half = window_frames // 2

    anchors = list(range(0, T_block, anchor_stride))
    if anchors[-1] != T_block - 1:
        anchors.append(T_block - 1)
    anchors_arr = np.asarray(anchors, dtype=np.int64)
    n_anchors = anchors_arr.size

    base = np.empty((n_anchors,) + block_slab.shape[1:], dtype=np.float32)
    for i, t in enumerate(anchors):
        lo = max(0, t - half)
        hi = min(T_block, t + half + 1)
        base[i] = np.percentile(
            block_slab[lo:hi], float(percentile), axis=0,
        ).astype(np.float32, copy=False)

    if n_anchors == 1:
        return np.broadcast_to(base[0], block_slab.shape).astype(
            np.float32, copy=True,
        )

    t_axis = np.arange(T_block, dtype=np.int64)
    idx_right = np.clip(
        np.searchsorted(anchors_arr, t_axis, side="right"),
        1, n_anchors - 1,
    )
    idx_left = idx_right - 1
    t_left = anchors_arr[idx_left]
    t_right = anchors_arr[idx_right]
    denom = (t_right - t_left).astype(np.float32)
    denom[denom == 0] = 1.0
    alpha = ((t_axis - t_left).astype(np.float32) / denom)[:, None, None]
    return (1.0 - alpha) * base[idx_left] + alpha * base[idx_right]


def _detrend_slab(
    block_slab: np.ndarray,
    window_frames: int,
    percentile: float,
    anchor_stride: int,
) -> np.ndarray:
    """Subtract the rolling lower-percentile baseline along axis 0."""
    baseline = _anchored_percentile_baseline(
        block_slab, window_frames, percentile, anchor_stride,
    )
    return (block_slab - baseline).astype(np.float32, copy=False)


def detrend_movie(
    src: "str | Path",
    dest: "str | Path",
    *,
    window_s: float = 30.0,
    percentile: float = 10.0,
    frame_rate_hz: float,
    batch_t: int = 2000,
    anchor_stride: "int | None" = None,
    chunk_t: "int | None" = None,
    n_jobs: int = 1,
    skip_if_exists: bool = True,
    verbose: bool = True,
) -> zarr.Array:
    """Per-pixel rolling-percentile temporal detrend on a (T, H, W) zarr.

    For each pixel, subtracts a rolling lower-percentile baseline of width
    ``window_frames = round(window_s * frame_rate_hz)`` along the time axis.

    Args:
        src: Path to the source ``(T, H, W)`` zarr (typically ``mc.zarr``).
        dest: Path to the destination ``(T, H, W)`` zarr. Created if absent.
        window_s: Rolling-window length in seconds.
        percentile: Lower percentile used for the baseline (default 10).
        frame_rate_hz: Acquisition frame rate. Combined with ``window_s`` to
            derive the window length in frames.
        batch_t: Number of *central* frames written per IO batch. The actual
            read window is ``batch_t + window_frames`` (half a window of
            overlap on each side so the rolling filter has context).
        anchor_stride: Spacing (in frames) at which the lower percentile is
            evaluated. The baseline is linearly interpolated between anchors.
            ``None`` (default) = ``max(1, window_frames // 10)`` — typically
            a ~20× CPU win with negligible accuracy loss because the
            ``window_frames``-wide lower percentile is a slow signal by
            design. Set ``anchor_stride=1`` to evaluate at every frame
            (still faster than ``scipy.ndimage.percentile_filter``); raise it
            to trade a touch more smoothing for more speed.
        chunk_t: Time chunk for the destination zarr. ``None`` = match source.
        n_jobs: Parallel workers for the per-batch percentile filter. The
            spatial frame is split into ``n_jobs`` horizontal slabs;
            ``np.percentile`` releases the GIL, so threads scale well.
            ``-1`` = all CPUs, ``1`` = serial.
        skip_if_exists: If ``dest`` already exists, return it without
            recomputing (matches the idempotency pattern used by
            ``transpose_zarr_to_pixel_major`` and the live-session notebook).
        verbose: Print progress lines.

    Returns:
        Read-mode handle on the destination zarr.
    """
    src_arr = open_zarr(src)
    if src_arr.ndim != 3:
        raise ValueError(f"src must be 3D (T, H, W); got shape {src_arr.shape}")
    T, H, W = src_arr.shape

    window_frames = max(2, int(round(window_s * frame_rate_hz)))
    if window_frames >= T:
        raise ValueError(
            f"window_frames ({window_frames}) >= T ({T}); pick a smaller "
            f"window_s or run on a longer movie."
        )
    half_w = window_frames // 2
    anchor_stride_eff = (
        int(anchor_stride) if anchor_stride is not None
        else max(1, window_frames // 10)
    )
    if anchor_stride_eff < 1:
        raise ValueError(f"anchor_stride must be >= 1, got {anchor_stride_eff}")

    dest_path = Path(dest)
    if dest_path.exists():
        if skip_if_exists:
            if verbose:
                print(f"Skipping detrend; {dest_path} already exists.")
            return zarr.open_array(str(dest_path), mode="r")
        shutil.rmtree(dest_path)

    src_chunk_t = src_arr.chunks[0]
    chunk_t_eff = int(chunk_t) if chunk_t is not None else src_chunk_t
    out = _open_array(
        dest_path, "w",
        shape=(T, H, W),
        chunks=(chunk_t_eff, H, W),
        dtype="float32",
        compression=True,  # heavyweight defaults match mc.zarr
    )

    if verbose:
        print(
            f"detrend: window={window_frames} frames ({window_s:g} s @ "
            f"{frame_rate_hz:g} Hz), percentile={percentile}, batch_t={batch_t}, "
            f"anchor_stride={anchor_stride_eff}, n_jobs={n_jobs}"
        )
        print(f"         src {src_arr.shape} -> dest {out.shape} (float32)")

    t_start = time.perf_counter()
    n_batches = (T + batch_t - 1) // batch_t

    # Pre-compute slab boundaries for the parallel split. Re-used across batches.
    if n_jobs == 1 or n_jobs == 0:
        slabs = [(0, H)]
    else:
        n_workers = n_jobs if n_jobs > 0 else None  # joblib resolves -1
        # Decide the number of slabs from the (resolved) worker count.
        import os
        eff = n_workers if n_workers is not None else (os.cpu_count() or 1)
        eff = max(1, min(eff, H))
        edges = np.linspace(0, H, eff + 1).astype(int)
        slabs = [(int(edges[i]), int(edges[i + 1])) for i in range(eff)]

    written = 0
    for b in range(n_batches):
        t0 = b * batch_t
        t1 = min(t0 + batch_t, T)
        r0 = max(0, t0 - half_w)
        r1 = min(T, t1 + half_w)
        block = np.asarray(src_arr[r0:r1], dtype=np.float32)

        if len(slabs) == 1:
            detrended = _detrend_slab(
                block, window_frames, percentile, anchor_stride_eff,
            )
        else:
            with threadpool_limits(limits=1, user_api="blas"):
                slab_results = Parallel(n_jobs=n_jobs, prefer="threads")(
                    delayed(_detrend_slab)(
                        block[:, h0:h1, :], window_frames, percentile,
                        anchor_stride_eff,
                    )
                    for (h0, h1) in slabs
                )
            detrended = np.empty_like(block)
            for (h0, h1), result in zip(slabs, slab_results):
                detrended[:, h0:h1, :] = result

        # Write only the central [t0:t1] slice — the half-window overlap was
        # context for the rolling filter at the batch boundaries.
        out[t0:t1] = detrended[t0 - r0 : t1 - r0]
        written += t1 - t0
        if verbose:
            elapsed = time.perf_counter() - t_start
            rate = written / elapsed if elapsed > 0 else 0.0
            print(
                f"  batch {b + 1}/{n_batches}: wrote frames {t0}..{t1}  "
                f"({written}/{T}, {rate:.0f} frames/s)"
            )

    if verbose:
        print(f"detrend done in {time.perf_counter() - t_start:.1f}s")

    return zarr.open_array(str(dest_path), mode="r")
