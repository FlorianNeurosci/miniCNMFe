"""Outlier-frame rejection preprocessor.

Detects frames where the spatial mean is a strong outlier (camera saturation,
brief illumination flashes, missed motion-correction at a sudden jump) and
replaces them by linear interpolation between the nearest non-outlier frames.

Use when the global-mean trace shows isolated spikes that survive
``detrend_movie`` — the rolling lower-percentile window is too wide to flag a
few-frame flash. The replacement is conservative: frames are only touched
when ``|mean(t) − median(mean)| > k_mad · 1.4826 · MAD``, so a recording with
no flashes typically yields zero replacements and the dest is a clean copy of
the source.

Caveat: if the recording has genuine population-wide synchronous bursts (a
real event that lifts the global mean), a too-strict ``k_mad`` will replace
those bursts too. Inspect the returned ``outlier_idx`` (or the printed
fraction) and raise ``k_mad`` if it looks like real activity is being
flagged.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import zarr

from minicnmfe.io import _open_array, open_zarr


def _detect_outlier_frames(means: np.ndarray, k_mad: float) -> np.ndarray:
    """Return indices of frames whose mean is outside the MAD-based gate."""
    med = float(np.median(means))
    mad = float(np.median(np.abs(means - med)))
    if mad == 0.0:
        return np.empty(0, dtype=np.int64)
    thr = k_mad * 1.4826 * mad
    return np.where(np.abs(means - med) > thr)[0]


def _build_interp_plan(
    outlier_idx: np.ndarray, T: int,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Per outlier: nearest non-outlier indices on each side and the
    linear-interp weight on the right side.

    A value of ``-1`` in ``lefts`` / ``rights`` means there is no
    non-outlier on that side (only possible if the outlier run touches
    the start / end of the movie).
    """
    is_out = np.zeros(T, dtype=bool)
    is_out[outlier_idx] = True

    left_nonout = np.full(T, -1, dtype=np.int64)
    last = -1
    for t in range(T):
        if not is_out[t]:
            last = t
        left_nonout[t] = last

    right_nonout = np.full(T, -1, dtype=np.int64)
    nxt = -1
    for t in range(T - 1, -1, -1):
        if not is_out[t]:
            nxt = t
        right_nonout[t] = nxt

    lefts = left_nonout[outlier_idx]
    rights = right_nonout[outlier_idx]
    alphas = np.zeros_like(outlier_idx, dtype=np.float32)
    for i, t in enumerate(outlier_idx):
        l, r = int(lefts[i]), int(rights[i])
        if l == -1 and r == -1:
            alphas[i] = 0.0
        elif l == -1:
            alphas[i] = 1.0
        elif r == -1:
            alphas[i] = 0.0
        else:
            alphas[i] = (int(t) - l) / float(r - l)
    return lefts, rights, alphas


def reject_outlier_frames(
    src: "str | Path",
    dest: "str | Path",
    *,
    k_mad: float = 5.0,
    batch_t: int = 1000,
    chunk_t: "int | None" = None,
    skip_if_exists: bool = True,
    verbose: bool = True,
) -> "tuple[zarr.Array, np.ndarray]":
    """Replace per-frame-mean outliers with neighbour interpolation.

    Two streaming passes over ``src``:

    1. Compute the per-frame mean ``means[t] = src[t].mean()``; flag outliers
       at ``|means[t] − median(means)| > k_mad · 1.4826 · MAD``.
    2. Copy ``src`` to ``dest`` batch-by-batch, replacing each outlier
       frame with the linear blend of its nearest non-outlier neighbours.

    Args:
        src: Source ``(T, H, W)`` zarr (typically ``mc.zarr``).
        dest: Destination ``(T, H, W)`` zarr. Created if absent.
        k_mad: MAD threshold. Default 5.0 ≈ 3.3σ on a Gaussian-shaped
            ``means`` distribution; expect ~0.1 % false positives on a
            clean recording (≈ a few dozen frames in 40k). Bump to 7–10
            on recordings with real synchronous bursts; drop to 3–4 to
            catch subtle flashes.
        batch_t: Frames per IO batch.
        chunk_t: Output time chunk. ``None`` = match source.
        skip_if_exists: If ``dest`` exists, return it without recomputing.
            (Outlier indices are persisted as a sidecar ``<dest>.outliers.npy``
            so they can be reloaded.)
        verbose: Print progress.

    Returns:
        ``(dest, outlier_idx)`` — open zarr handle on the destination, plus
        the int64 array of replaced frame indices (empty when no outliers).
    """
    src_arr = open_zarr(src)
    if src_arr.ndim != 3:
        raise ValueError(f"src must be 3D (T, H, W); got shape {src_arr.shape}")
    T, H, W = src_arr.shape

    dest_path = Path(dest)
    sidecar = dest_path.parent / f"{dest_path.name}.outliers.npy"
    if dest_path.exists():
        if skip_if_exists:
            if verbose:
                print(f"Skipping reject; {dest_path} already exists.")
            existing = zarr.open_array(str(dest_path), mode="r")
            if sidecar.exists():
                return existing, np.load(sidecar)
            return existing, np.empty(0, dtype=np.int64)
        shutil.rmtree(dest_path)

    src_chunk_t = src_arr.chunks[0]
    chunk_t_eff = int(chunk_t) if chunk_t is not None else src_chunk_t

    t_start = time.perf_counter()
    if verbose:
        print(f"reject: pass 1/2 — computing per-frame mean (T={T})...")
    means = np.empty(T, dtype=np.float32)
    for s in range(0, T, batch_t):
        e = min(s + batch_t, T)
        means[s:e] = np.asarray(src_arr[s:e]).reshape(e - s, -1).mean(axis=1)

    outlier_idx = _detect_outlier_frames(means, k_mad)
    if verbose:
        if len(outlier_idx) == 0:
            print(f"  no outliers at k_mad={k_mad}; dest will be a copy.")
        else:
            pct = 100.0 * len(outlier_idx) / T
            print(
                f"  {len(outlier_idx)} outlier frames "
                f"({pct:.2f}% of T) at k_mad={k_mad}"
            )
            print(f"  e.g. {outlier_idx[:8].tolist()}")

    out = _open_array(
        dest_path, "w",
        shape=(T, H, W),
        chunks=(chunk_t_eff, H, W),
        dtype="float32",
        compression=True,
    )

    if len(outlier_idx) == 0:
        if verbose:
            print("reject: pass 2/2 — direct copy (no outliers).")
        for s in range(0, T, batch_t):
            e = min(s + batch_t, T)
            out[s:e] = np.asarray(src_arr[s:e], dtype=np.float32)
    else:
        lefts, rights, alphas = _build_interp_plan(outlier_idx, T)
        boundary_idx = np.unique(np.concatenate([
            lefts[lefts >= 0], rights[rights >= 0],
        ]).astype(np.int64))
        boundary_cache = {
            int(bi): np.asarray(src_arr[int(bi)], dtype=np.float32)
            for bi in boundary_idx
        }
        outlier_to_plan_idx = {int(t): i for i, t in enumerate(outlier_idx)}
        outlier_set = set(outlier_to_plan_idx.keys())

        if verbose:
            print("reject: pass 2/2 — copying with replacements...")
        for s in range(0, T, batch_t):
            e = min(s + batch_t, T)
            batch = np.asarray(src_arr[s:e], dtype=np.float32)
            for local_t in range(e - s):
                global_t = s + local_t
                if global_t not in outlier_set:
                    continue
                i = outlier_to_plan_idx[global_t]
                l, r, a = int(lefts[i]), int(rights[i]), float(alphas[i])
                if l == -1 and r == -1:
                    continue
                if l == -1:
                    batch[local_t] = boundary_cache[r]
                elif r == -1:
                    batch[local_t] = boundary_cache[l]
                else:
                    batch[local_t] = (
                        (1.0 - a) * boundary_cache[l]
                        + a * boundary_cache[r]
                    )
            out[s:e] = batch

    np.save(sidecar, outlier_idx)
    if verbose:
        print(
            f"reject done in {time.perf_counter() - t_start:.1f}s "
            f"({len(outlier_idx)} replaced)  sidecar -> {sidecar.name}"
        )
    return zarr.open_array(str(dest_path), mode="r"), outlier_idx
