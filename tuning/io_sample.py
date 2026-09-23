"""Sampling / decoding helpers for the tuner.

Cheap ways to get a small slice of a recording into RAM — a strided AVI sample
(for the motion-correction heuristics) and a strided ``mc.zarr`` sample (for the
init heuristics) — plus ``pick_cutout`` (choose a representative spatial+temporal
window for the sweep) and ``quick_fused_mc`` (run a fast fused AVI->mc.zarr so
the extraction sweep has something to run on).

Pure IO + numpy; no matplotlib, no pipeline internals beyond ``CNMFe``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def list_avis(folder: "str | Path", pattern: str = "*.avi") -> list[Path]:
    """AVIs in ``folder``, in concatenation order.

    Delegates to ``concat_avis_to_zarr.discover_avis`` so the tuner sees exactly
    the files (and the order) the rest of the pipeline will use: numeric names
    first, then the single-file / timestamped layouts written by the FFV1
    acquisition.
    """
    from minicnmfe.concat_avis_to_zarr import discover_avis

    return discover_avis(folder, pattern)


# Files at least this long are sampled by seeking instead of decoding every frame.
# Chunked miniscope recordings (0.avi, 1.avi, ...) are ~1000 frames per file and keep
# the original full-decode path; the FFV1 acquisition writes ONE file per recording
# (51k frames on the 2024 PV cohort), where "decode 8 files" meant decoding the whole
# movie on one thread -- 316 s of a 462 s MC-tuning.
_SEEK_MIN_FRAMES = 5000


def _n_frames(path) -> int:
    """Frame count from the container header (0 if the container does not say)."""
    import av

    with av.open(str(path)) as c:
        return int(c.streams.video[0].frames or 0)


def _decode_frames_at(path, frame_idx) -> list:
    """Decode the frames at ``frame_idx`` (0-based, ascending) by seeking.

    Seeks to the keyframe at or before each target and decodes forward to it, so
    the returned frames are exactly those a full sequential decode would give at
    those indices. Consecutive targets inside one GOP reuse the open decoder
    instead of seeking again.
    """
    import av

    out = []
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        stream.thread_type = "FRAME"
        start = int(stream.start_time or 0)
        # pts per frame in the stream's time base (1 for the FFV1 files: tb = 1/fps)
        step = float(1 / (stream.average_rate * stream.time_base))
        frames = None
        cur = None
        for i in frame_idx:
            target = start + int(round(i * step))
            if cur is None or cur.pts > target or target - cur.pts > 16 * step:
                c.seek(target, stream=stream, backward=True, any_frame=False)
                frames = c.decode(stream)
                cur = next(frames)
            while cur.pts < target:
                cur = next(frames)
            if cur.pts != target:
                raise ValueError(f"{path}: seek for frame {i} (pts {target}) landed on pts {cur.pts}")
            out.append(cur.to_ndarray(format="gray8"))
    return out


def decode_strided_sample(avi_paths, n_avis: int, stride: int) -> np.ndarray:
    """Decode a strided sample of frames from a strided subset of AVIs into RAM.

    Lifted verbatim from ``live_runs/estimate_params.ipynb`` (the
    ``decode_strided_sample`` helper). Returns a ``(T_sample, H, W)`` float32
    stack — enough to build std / median projections and a shift histogram
    without touching a zarr.

    Every ``stride``-th frame of each picked file is kept. Files of at least
    ``_SEEK_MIN_FRAMES`` frames are read by seeking to those frames rather than
    decoding all of them; the frames returned are the same either way.
    """
    import av

    k = min(n_avis, len(avi_paths))
    picks = np.linspace(0, len(avi_paths) - 1, k).astype(int)
    pool = []
    for i in picks:
        path = avi_paths[int(i)]
        n = _n_frames(path)
        if n >= _SEEK_MIN_FRAMES:
            pool.extend(_decode_frames_at(path, range(0, n, stride)))
            continue
        container = av.open(str(path))
        try:
            stream = container.streams.video[0]
            stream.thread_type = "FRAME"
            for j, frame in enumerate(container.decode(stream)):
                if j % stride == 0:
                    pool.append(frame.to_ndarray(format="gray8"))
        finally:
            container.close()
    return np.stack(pool, axis=0).astype(np.float32)


def decode_contiguous_clip(avi_paths, n_frames: int, start_frac: float = 0.4) -> np.ndarray:
    """Decode up to ``n_frames`` **consecutive** frames into RAM.

    Unlike ``decode_strided_sample`` (which subsamples for projections /
    histograms), the MC parameter search needs a temporally contiguous clip:
    motion is continuous in time, so registration quality can only be judged on
    successive frames. Starts ``start_frac`` of the way through the recording
    (avoids LED-warmup at the very start) and spans consecutive AVIs until
    ``n_frames`` are collected. Returns a ``(T<=n_frames, H, W)`` float32 stack.

    Chunked recordings start at the first frame of file ``int(start_frac * n_files)``.
    When the files are long (``_SEEK_MIN_FRAMES`` or more) that rounds to file 0 for a
    single-file recording -- i.e. the LED warm-up -- so the start is instead
    ``start_frac`` of the total frame count, reached by seeking.
    """
    import av

    counts = [_n_frames(p) for p in avi_paths]
    if counts and min(counts) >= _SEEK_MIN_FRAMES:
        g = int(start_frac * sum(counts))
        start_avi = 0
        while g >= counts[start_avi]:
            g -= counts[start_avi]
            start_avi += 1
        first_local = g
    else:
        start_avi = min(len(avi_paths) - 1, int(start_frac * len(avi_paths)))
        first_local = 0
    pool: list = []
    for p in avi_paths[start_avi:]:
        if first_local:
            want = min(n_frames - len(pool), counts[start_avi] - first_local)
            pool.extend(_decode_frames_at(p, range(first_local, first_local + want)))
            first_local = 0
            if len(pool) >= n_frames:
                break
            continue
        container = av.open(str(p))
        try:
            stream = container.streams.video[0]
            stream.thread_type = "FRAME"
            for frame in container.decode(stream):
                pool.append(frame.to_ndarray(format="gray8"))
                if len(pool) >= n_frames:
                    break
        finally:
            container.close()
        if len(pool) >= n_frames:
            break
    return np.stack(pool, axis=0).astype(np.float32)


def load_mc_sample(mc_zarr, n_frames: int) -> "tuple[np.ndarray, np.ndarray]":
    """Load a chunk-aligned, time-spread sample of an ``mc.zarr`` into RAM.

    Returns ``(sample, idx)`` where ``sample`` is ``(n, H, W)`` float32 and
    ``idx`` is the global frame index of each sampled frame (so a temporal
    window can be mapped back to full-T coordinates).

    The frames are read as ``K`` **contiguous, chunk-aligned blocks** spread
    across the recording rather than as ``n`` linspace-strided single frames.
    The old strided read defeated zarr chunking: with a time-chunk of e.g. 100
    frames and a stride > 100, every requested frame lands in a distinct chunk,
    so reading 400 frames decompressed ~400 full chunks (tens of GB) off the
    store. Reading whole chunks at ``K = ceil(n / blk)`` evenly-spaced,
    chunk-snapped starts pulls only the data actually needed (~one chunk per
    block) while still covering the whole recording in time.
    """
    T = int(mc_zarr.shape[0])
    n = int(min(T, max(1, n_frames)))
    # Time-chunk size (frames per chunk); fall back to a sane block if unchunked.
    chunks = getattr(mc_zarr, "chunks", None)
    blk = int(chunks[0]) if chunks else min(T, 256)
    blk = max(1, min(blk, T))
    k = max(1, -(-n // blk))  # ceil(n / blk) blocks
    if k * blk >= T:
        # Few/large blocks cover the whole movie — just take a strided view.
        idx = np.linspace(0, T - 1, n).astype(int)
        sample = np.asarray(mc_zarr[:], dtype=np.float32)[idx]
        return sample, idx
    # Evenly-spaced, chunk-aligned block starts across [0, T - blk].
    starts = np.unique(((np.linspace(0, T - blk, k)).astype(int) // blk) * blk)
    parts, idxs = [], []
    for s in starts:
        s = int(s)
        parts.append(np.asarray(mc_zarr[s:s + blk], dtype=np.float32))
        idxs.append(np.arange(s, s + blk))
    sample = np.concatenate(parts, axis=0)
    idx = np.concatenate(idxs)
    if len(sample) > n:  # trim evenly to exactly n frames
        keep = np.linspace(0, len(sample) - 1, n).astype(int)
        sample, idx = sample[keep], idx[keep]
    return sample, idx.astype(int)


def pick_cutout(
    cn: np.ndarray,
    *,
    T: int,
    cutout_hw: "tuple[int, int]",
    window_t: int,
    sample: "np.ndarray | None" = None,
    sample_idx: "np.ndarray | None" = None,
) -> "tuple[tuple[int, int, int, int], tuple[int, int]]":
    """Choose a representative ``(spatial_crop, temporal_crop)`` for the sweep.

    Spatial: slide a ``cutout_hw`` window over the **correlation image** ``cn``
    (activity-dense, not the mean projection which chases bright vasculature /
    vignette — see CLAUDE.md) and pick the window with the largest summed CORR
    via an O(H·W) integral image. Temporal: centre a ``window_t``-frame window
    on the highest-activity sampled frame (per-frame variance of the strided
    ``sample``); falls back to the first ``window_t`` frames if no sample given.

    Returns native-coordinate ``(y0, y1, x0, x1)`` and ``(t0, t1)`` (t1
    exclusive), both clamped to the FOV / movie length.
    """
    H, W = cn.shape
    ch, cw = int(min(cutout_hw[0], H)), int(min(cutout_hw[1], W))

    # Integral image of cn for O(1) window sums.
    finite = np.nan_to_num(cn, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    ii = finite.cumsum(axis=0).cumsum(axis=1)
    ii = np.pad(ii, ((1, 0), (1, 0)), mode="constant")
    # window sum at top-left (y, x): ii[y+ch,x+cw]-ii[y,x+cw]-ii[y+ch,x]+ii[y,x]
    win = (
        ii[ch:, cw:] - ii[:-ch, cw:] - ii[ch:, :-cw] + ii[:-ch, :-cw]
    )
    if win.size == 0:
        y0, x0 = 0, 0
    else:
        best = np.unravel_index(int(np.argmax(win)), win.shape)
        y0, x0 = int(best[0]), int(best[1])
    y1, x1 = y0 + ch, x0 + cw

    # Temporal window.
    if window_t >= T:
        t0, t1 = 0, T
    elif sample is not None and sample_idx is not None and len(sample) > 1:
        med = np.median(sample, axis=0)
        activity = ((sample - med) ** 2).mean(axis=(1, 2))
        centre = int(sample_idx[int(np.argmax(activity))])
        half = window_t // 2
        t0 = max(0, min(centre - half, T - window_t))
        t1 = t0 + window_t
    else:
        t0, t1 = 0, min(window_t, T)

    return (int(y0), int(y1), int(x0), int(x1)), (int(t0), int(t1))


def quick_fused_mc(
    avi_folder: "str | Path",
    out_dir: "str | Path",
    params,
    *,
    ssub: int = 1,
    tsub: int = 1,
    n_template_avis: int = 8,
    max_avis: "int | None" = None,
    pattern: str = "*.avi",
):
    """Fast fused AVI -> ``mc.zarr`` so the extraction sweep has an input.

    Thin wrapper over ``CNMFe(params.downscaled(ssub, tsub)).fit_mc_from_avis``.
    When ``max_avis`` is set, an evenly-spaced subset of the AVIs is symlinked
    into a temporary ``_mc_subset`` dir (named ``0.avi, 1.avi, ...``) and fused
    from there — a fast approximation to the full-session shifts. ``mc.zarr``
    lands in ``out_dir``.

    The subset links are placed on **local temp storage**, not under ``out_dir``:
    ``out_dir`` is often a network share (CIFS/NFS) that rejects symlinks with
    ``OSError(EOPNOTSUPP)`` (errno 95). The links target the absolute AVI paths,
    so they resolve back to the share regardless of where the dir lives.
    """
    import shutil
    import tempfile

    from minicnmfe.pipeline import CNMFe

    avi_folder = Path(avi_folder)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    folder = avi_folder
    subset_dir = None
    if max_avis is not None:
        avis = list_avis(avi_folder, pattern)
        if max_avis < len(avis):
            picks = np.linspace(0, len(avis) - 1, max_avis).astype(int)
            subset_dir = Path(tempfile.mkdtemp(prefix="minicnmfe_mc_subset_"))
            for new_i, src_i in enumerate(picks):
                src = avis[int(src_i)].resolve()
                link = subset_dir / f"{new_i}.avi"
                try:
                    link.symlink_to(src)
                except OSError:
                    # even local temp can't symlink -> copy as a last resort
                    shutil.copy2(src, link)
            folder = subset_dir

    model = CNMFe(params.downscaled(ssub, tsub))
    try:
        mc_zarr = model.fit_mc_from_avis(
            folder, out_dir, pattern=pattern, ssub=ssub, tsub=tsub,
        )
    finally:
        if subset_dir is not None:
            shutil.rmtree(subset_dir, ignore_errors=True)
    return mc_zarr, model.shifts
