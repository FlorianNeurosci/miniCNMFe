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
    """Numerically-sorted list of ``0.avi, 1.avi, ...`` in ``folder``.

    Reuses ``concat_avis_to_zarr._numeric_key`` so the ordering matches the
    rest of the pipeline (and drops non-numerically-named files).
    """
    from minicnmfe.concat_avis_to_zarr import _numeric_key

    folder = Path(folder)
    avis = sorted(folder.glob(pattern), key=_numeric_key)
    avis = [p for p in avis if _numeric_key(p) >= 0]
    if not avis:
        raise FileNotFoundError(f"No numerically-named AVIs ({pattern}) in {folder}")
    return avis


def decode_strided_sample(avi_paths, n_avis: int, stride: int) -> np.ndarray:
    """Decode a strided sample of frames from a strided subset of AVIs into RAM.

    Lifted verbatim from ``live_runs/estimate_params.ipynb`` (the
    ``decode_strided_sample`` helper). Returns a ``(T_sample, H, W)`` float32
    stack — enough to build std / median projections and a shift histogram
    without touching a zarr.
    """
    import av

    k = min(n_avis, len(avi_paths))
    picks = np.linspace(0, len(avi_paths) - 1, k).astype(int)
    pool = []
    for i in picks:
        container = av.open(str(avi_paths[int(i)]))
        try:
            stream = container.streams.video[0]
            stream.thread_type = "FRAME"
            for j, frame in enumerate(container.decode(stream)):
                if j % stride == 0:
                    pool.append(frame.to_ndarray(format="gray8"))
        finally:
            container.close()
    return np.stack(pool, axis=0).astype(np.float32)


def load_mc_sample(mc_zarr, n_frames: int) -> "tuple[np.ndarray, np.ndarray]":
    """Load a linspace-strided sample of an ``mc.zarr`` into RAM.

    Returns ``(sample, idx)`` where ``sample`` is ``(n, H, W)`` float32 and
    ``idx`` is the global frame index of each sampled frame (so a temporal
    window can be mapped back to full-T coordinates).
    """
    T = int(mc_zarr.shape[0])
    idx = np.linspace(0, T - 1, min(T, n_frames)).astype(int)
    sample = np.stack([np.asarray(mc_zarr[int(i)]) for i in idx]).astype(np.float32)
    return sample, idx


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
