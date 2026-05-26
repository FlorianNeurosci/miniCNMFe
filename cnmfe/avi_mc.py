"""Fused AVI -> motion-corrected zarr in a single pass.

Eliminates the intermediate ``session.zarr`` from the live-session workflow:
decode the AVI folder and apply rigid motion correction in the same pipeline,
writing only the corrected ``mc.zarr`` to disk. On a network mount this saves
~5 min and ~6 GB compared with running ``concat_avis_to_zarr`` followed by
``motion_correction_rigid`` separately.

Reuses two existing pieces:
- The producer-consumer AVI decoder pipeline from ``concat_avis_to_zarr``
  (``_decode_avi_worker`` pushes ``(start, batch)`` tuples onto a queue).
- The per-frame MC work from ``cnmfe/motion_correction.py``
  (``_process_batch`` takes a batch + pre-filtered template, returns
  ``(corrected, shifts)`` parallelised over frames).

Only ``mc_n_iter == 1`` is supported. For multi-iteration MC (where each
iteration rebuilds the template from the previous iteration's corrected
output) use the separate ``concat_avis_to_zarr`` + ``CNMFe.fit_mc`` flow.
"""

from __future__ import annotations

import os
import queue
import shutil
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import zarr

from cnmfe.io import _open_array, open_zarr
from cnmfe.motion_correction import (
    _process_batch,
    _sample_frame_indices,
    caiman_bin_median,
    high_pass_filter_space,
)
from concat_avis_to_zarr import (
    _DECODER_DONE,
    _count_and_shape,
    _decode_avi_worker,
    _numeric_key,
)


def concat_avis_to_mc_zarr(
    folder: "str | Path",
    output_path: "str | Path",
    params,
    *,
    pattern: str = "*.avi",
    n_jobs: "int | None" = None,
    skip_if_exists: bool = False,
    n_template_avis: int = 10,
    ssub: int = 1,
    tsub: int = 1,
    verbose: bool = True,
) -> "tuple[zarr.Array, np.ndarray]":
    """Decode an AVI folder and apply motion correction in one pass.

    Pipeline:
      1. Discover + pre-scan every AVI (frame counts, spatial dims).
      2. Build a CaImAn-style bin-median template by decoding a strided
         subset of AVIs (~``n_template_avis`` files) into a RAM buffer
         and median-filtering the high-pass-filtered samples.
      3. Re-decode every AVI in parallel; each batch goes through
         ``_process_batch`` (per-frame MC) and is written to the output
         zarr at the offset given by the pre-scan counts. Shifts are
         accumulated into a ``(T, 2)`` float32 array.

    Args:
        folder: Directory containing numbered AVI files (``0.avi``, ...).
        output_path: Path to the output ``mc.zarr``.
        params: ``CNMFeParams``. Read fields: ``max_shift``,
            ``upsample_factor``, ``mc_n_iter`` (must be 1),
            ``mc_gSig_filt``, ``mc_batch_size``,
            ``mc_template_max_frames``, ``mc_output_chunk_t``,
            ``mc_output_dtype``, ``n_jobs``.
        pattern: Glob pattern for AVI selection.
        n_jobs: Override for ``params.n_jobs`` in the decoder thread
            count. ``None`` (default) uses the same auto-pick as
            ``concat_avis_to_zarr`` (``min(cpu_count, len(avis))``).
        skip_if_exists: If ``output_path`` already exists, open it and
            return (rather than raising or overwriting).
        n_template_avis: Number of strided AVIs to decode into RAM for
            template building. The template is then strided down to
            ``params.mc_template_max_frames`` frames.
        ssub: Spatial bin factor (block-mean over ``ssub×ssub`` pixels),
            applied to the raw AVI frames *before* MC. ``1`` = none.
        tsub: Temporal bin factor (mean of ``tsub`` consecutive frames),
            per file, applied before MC. ``1`` = none. Output frame count
            is ``sum(n_i // tsub)``; the trailing ``< tsub`` frames of each
            file are dropped.
            **When downsampling, pass MC params in downsampled units** —
            i.e. ``params = native_params.downscaled(ssub, tsub)`` so
            ``max_shift`` / ``mc_gSig_filt`` match the binned frames. The
            fused output ``mc.zarr`` is the *only* zarr written (no
            intermediate full-resolution store).
        verbose: Print progress + per-phase timing lines.

    Returns:
        (mc_zarr, shifts) — ``zarr.Array`` shape ``(T_out, H//ssub, W//ssub)``
        float32 + ``np.ndarray`` ``(T_out, 2)`` float32, where
        ``T_out == sum(n_i // tsub)``.

    Raises:
        ValueError: bad input, or ``params.mc_n_iter > 1``.
        FileNotFoundError: ``folder`` does not exist.
        FileExistsError: ``output_path`` exists and ``skip_if_exists``
            is False.
    """
    if params.mc_n_iter < 1:
        raise ValueError(
            f"mc_n_iter must be >= 1, got {params.mc_n_iter}"
        )
    if ssub < 1 or tsub < 1:
        raise ValueError(f"ssub and tsub must be >= 1 (got {ssub}, {tsub})")

    t0_total = time.time()

    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a directory: {folder}")

    out_path = Path(output_path)

    # Idempotency: reuse the existing zarr (and shifts.npy if present).
    if out_path.exists():
        if skip_if_exists:
            if verbose:
                print(f"Output already exists, reusing: {out_path}")
            mc_zarr = open_zarr(out_path)
            shifts_path = out_path.parent / "shifts.npy"
            shifts = np.load(shifts_path) if shifts_path.exists() else None
            return mc_zarr, shifts
        raise FileExistsError(
            f"Output already exists: {out_path}. "
            f"Delete it or pass skip_if_exists=True to reuse it."
        )

    # --- Discover AVIs ------------------------------------------------------
    candidates = sorted(folder.glob(pattern), key=_numeric_key)
    avis = [p for p in candidates if _numeric_key(p) >= 0]
    if not avis:
        raise ValueError(
            f"No numerically-named AVI files found in {folder} "
            f"matching '{pattern}'."
        )
    if verbose:
        print(f"Found {len(avis)} AVI files: "
              f"{avis[0].name} ... {avis[-1].name}")

    # --- Pre-scan: walk every AVI to count frames + check spatial dims ------
    t0_probe = time.time()
    if verbose:
        print("Scanning frame counts ...", flush=True)
    counts: list[int] = []
    ref_H = ref_W = None
    for avi in avis:
        n, H, W = _count_and_shape(avi)
        if ref_H is None:
            ref_H, ref_W = H, W
        elif (H, W) != (ref_H, ref_W):
            raise ValueError(
                f"Spatial mismatch: {avi.name} is {H}x{W} but first "
                f"file is {ref_H}x{ref_W}"
            )
        counts.append(n)
        if verbose:
            print(f"  {avi.name}: {n} frames  ({H}x{W})", flush=True)
    T_total = sum(counts)
    # Output (post-bin) dims + per-file output counts. Temporal binning is
    # per file, so T_out = sum(n_i // tsub); the trailing < tsub frames of
    # each file are dropped.
    downsampling = (ssub > 1 or tsub > 1)
    out_H, out_W = ref_H // ssub, ref_W // ssub
    out_counts = [c // tsub for c in counts] if tsub > 1 else list(counts)
    T_out = int(sum(out_counts))
    if verbose:
        print(f"\nTotal: {T_total} frames  x  {ref_H}x{ref_W} px")
        if downsampling:
            print(f"Downsample ssub={ssub} tsub={tsub} -> "
                  f"{T_out} frames  x  {out_H}x{out_W} px (binned before MC)")
    probe_secs = time.time() - t0_probe

    n_jobs_eff = _resolve_n_jobs(n_jobs, params, len(avis))

    # --- Phase 1: build template from a strided AVI subset ------------------
    # Built from binned frames so it matches the (downsampled) data MC sees.
    t0_template = time.time()
    template = _build_template_from_strided_avis(
        avis,
        counts,
        out_H,
        out_W,
        n_template_avis=n_template_avis,
        template_max_frames=params.mc_template_max_frames,
        gSig_filt=params.mc_gSig_filt,
        bin_window=10,
        ssub=ssub,
        tsub=tsub,
        n_jobs=n_jobs_eff,
        verbose=verbose,
    )
    filtered_template = (
        high_pass_filter_space(template, params.mc_gSig_filt)
        if params.mc_gSig_filt is not None else template
    )
    template_secs = time.time() - t0_template

    # --- Allocate fused-pass-1 output zarr + shifts buffer ------------------
    # When mc_n_iter == 1 the fused pass writes directly to `out_path`.
    # When mc_n_iter > 1 it writes to a scratch (".<name>.fused.zarr") that
    # we hand off to `motion_correction_rigid` for the remaining
    # iterations, then delete in a finally block.
    if params.mc_n_iter == 1:
        pass1_path = out_path
    else:
        pass1_path = out_path.parent / f".{out_path.name}.fused.zarr"
        if pass1_path.exists():
            shutil.rmtree(pass1_path)

    chunk_t = params.mc_output_chunk_t or min(params.mc_batch_size, T_out)
    mc_zarr = _open_array(
        pass1_path, "w",
        shape=(T_out, out_H, out_W),
        chunks=(chunk_t, out_H, out_W),
        dtype=params.mc_output_dtype,
        compression=True,
        # Heavy compression: the mc.zarr is read many times during
        # extraction, so ratio matters more than write speed here.
        clevel=5,
        shuffle="bitshuffle",
    )
    shifts = np.zeros((T_out, 2), dtype=np.float32)
    if verbose:
        print(f"\nWriting -> {pass1_path}")
        print(f"  shape={mc_zarr.shape}  chunks={mc_zarr.chunks}  "
              f"dtype={params.mc_output_dtype}", flush=True)

    # --- Phase 2: fused decode + MC + write (iteration 1) -------------------
    t0_mc = time.time()
    _run_avi_mc_parallel(
        avis,
        out_counts,
        mc_zarr,
        shifts,
        filtered_template=filtered_template,
        gSig_filt=params.mc_gSig_filt,
        upsample_factor=params.upsample_factor,
        max_shift=params.max_shift,
        chunk_t=params.mc_batch_size,
        ssub=ssub,
        tsub=tsub,
        n_jobs=n_jobs_eff,
        n_jobs_inner=n_jobs_eff,
        queue_maxsize=2 * n_jobs_eff,
        verbose=verbose,
    )
    mc_secs = time.time() - t0_mc

    if params.mc_n_iter == 1:
        if verbose:
            total_secs = time.time() - t0_total
            print(f"\nDone. mc.zarr written to: {out_path}")
            print(f"  Timing: pre-scan {probe_secs:.1f}s  "
                  f"template {template_secs:.1f}s  "
                  f"decode+MC+write {mc_secs:.1f}s  "
                  f"total {total_secs:.1f}s")
        return mc_zarr, shifts

    # --- Phase 3: hand off remaining iterations to motion_correction_rigid --
    # The fused pass 1 corrected output is at `pass1_path`. The existing
    # streaming MC builds the iteration-2 template from this corrected
    # source (motion_correction.py:580), then ping-pongs through its own
    # `.scratch_a.zarr` / `.scratch_b.zarr` for any further iterations,
    # and renames the final scratch to `output_path`.
    if verbose:
        print(f"\nFused pass 1 done in {mc_secs:.1f}s; "
              f"running {params.mc_n_iter - 1} additional MC "
              f"iteration{'s' if params.mc_n_iter > 2 else ''} via "
              f"motion_correction_rigid ...", flush=True)
    t0_handoff = time.time()
    try:
        # Local import to avoid a module-level cycle: motion_correction
        # itself doesn't depend on avi_mc, and a top-level import here
        # would slow `import cnmfe` for callers that never touch the
        # fused path.
        from cnmfe.motion_correction import motion_correction_rigid

        final_zarr, shifts_rest = motion_correction_rigid(
            mc_zarr,
            output_path=out_path,
            max_shift=params.max_shift,
            gSig_filt=params.mc_gSig_filt,
            upsample_factor=params.upsample_factor,
            niter_rig=params.mc_n_iter - 1,
            batch_size=params.mc_batch_size,
            n_jobs=n_jobs_eff,
            template_max_frames=params.mc_template_max_frames,
            output_chunk_t=params.mc_output_chunk_t,
            output_dtype=params.mc_output_dtype,
            verbose=verbose,
        )
        # Shifts compose additively across iterations (matches the
        # existing `shifts_total += shifts_iter` convention in
        # motion_correction.py:575).
        total_shifts = shifts + shifts_rest
    finally:
        shutil.rmtree(pass1_path, ignore_errors=True)
    handoff_secs = time.time() - t0_handoff

    if verbose:
        total_secs = time.time() - t0_total
        print(f"\nDone. mc.zarr written to: {out_path}")
        print(f"  Timing: pre-scan {probe_secs:.1f}s  "
              f"template {template_secs:.1f}s  "
              f"fused-pass-1 {mc_secs:.1f}s  "
              f"handoff (iter 2..{params.mc_n_iter}) {handoff_secs:.1f}s  "
              f"total {total_secs:.1f}s")

    return final_zarr, total_shifts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_n_jobs(n_jobs_arg, params, n_avis: int) -> int:
    """Pick a decoder thread count. Matches concat_avis_to_zarr's auto-pick.

    Order of precedence: ``n_jobs_arg`` > ``params.n_jobs`` > auto.
    """
    if n_jobs_arg is not None:
        return max(1, int(n_jobs_arg))
    raw = getattr(params, "n_jobs", -1)
    if raw == -1 or raw is None:
        return min(os.cpu_count() or 1, n_avis)
    return max(1, int(raw))


def _build_template_from_strided_avis(
    avis,
    counts,
    H: int,
    W: int,
    *,
    n_template_avis: int,
    template_max_frames: int,
    gSig_filt,
    bin_window: int,
    ssub: int,
    tsub: int,
    n_jobs: int,
    verbose: bool,
) -> np.ndarray:
    """Decode a strided subset of AVIs into a RAM buffer, then build a
    CaImAn-style bin-median template from a strided sample of those frames.

    Picks ``n_template_avis`` evenly-spaced AVIs across the file list.
    Frame counts come from the caller's pre-scan so no extra pyav opens
    happen here. ``H`` / ``W`` are the OUTPUT (post-bin) dims; the sample
    frames are binned by ``ssub`` / ``tsub`` so the template matches the
    downsampled data MC operates on.
    """
    if not avis:
        raise ValueError("no AVIs provided to template builder")

    k = min(n_template_avis, len(avis))
    if k == len(avis):
        chosen_idx = list(range(k))
    else:
        chosen_idx = np.linspace(0, len(avis) - 1, k).astype(int).tolist()
    chosen_avis = [avis[i] for i in chosen_idx]
    chosen_counts = [counts[i] for i in chosen_idx]
    if verbose:
        name_preview = ", ".join(p.name for p in chosen_avis[:3])
        if k > 3:
            name_preview += ", ..."
        print(f"\nBuilding template from {k} strided AVIs ({name_preview})")

    pool = _decode_avis_to_buffer(
        chosen_avis, chosen_counts, H, W,
        ssub=ssub, tsub=tsub,
        n_jobs=min(n_jobs, k), verbose=verbose,
    )  # (T_subset_out, H, W) — H, W already post-bin

    # Stride-sample the pool down to template_max_frames, high-pass filter,
    # and median-bin to get the template.
    idx = _sample_frame_indices(pool.shape[0], template_max_frames)
    sampled = np.empty((len(idx), H, W), dtype=np.float32)
    for i, t in enumerate(idx):
        frame = pool[int(t)].astype(np.float32)
        if gSig_filt is not None:
            frame = high_pass_filter_space(frame, gSig_filt)
        sampled[i] = frame
    return caiman_bin_median(sampled, window=bin_window)


def _decode_avis_to_buffer(
    avis,
    counts,
    H: int,
    W: int,
    *,
    ssub: int = 1,
    tsub: int = 1,
    n_jobs: int,
    verbose: bool,
) -> np.ndarray:
    """Decode the given AVIs (with known per-file frame counts) into a single
    contiguous RAM buffer.

    ``H`` / ``W`` are the OUTPUT (post-bin) dims; frames are binned by
    ``ssub`` / ``tsub`` inline. The buffer is ``(sum(n_i // tsub), H, W)``,
    float32 when binning (to keep the fractional means) else uint8. Uses the
    same parallel decoder threads as the main MC pass, draining into RAM.
    """
    downsampling = (ssub > 1 or tsub > 1)
    out_counts = [c // tsub for c in counts] if tsub > 1 else list(counts)
    T = int(sum(out_counts))
    offsets = np.concatenate([[0], np.cumsum(out_counts[:-1])]).astype(int).tolist()
    dec_dtype = "float32" if downsampling else "uint8"
    buf = np.empty((T, H, W), dtype=np.float32 if downsampling else np.uint8)

    out_q: "queue.Queue" = queue.Queue(maxsize=2 * max(1, n_jobs))
    errors: list = []
    stop_event = threading.Event()
    sem = threading.Semaphore(max(1, n_jobs))

    def _decode_with_admission(*args):
        with sem:
            _decode_avi_worker(*args)

    decoders: list[threading.Thread] = []
    for i, (path, offset) in enumerate(zip(avis, offsets)):
        t = threading.Thread(
            target=_decode_with_admission,
            args=(path, i, offset, 200,    # chunk_t in template phase
                  True, "luma", dec_dtype,
                  ssub, tsub,
                  out_q, errors, stop_event),
            name=f"tmpl-decoder-{i}-{path.name}",
            daemon=True,
        )
        decoders.append(t)
        t.start()

    if verbose:
        try:
            from tqdm import tqdm
            progress = tqdm(total=T, unit="frame", desc="template decode")
        except ImportError:
            progress = None
    else:
        progress = None

    try:
        sentinels_seen = 0
        while sentinels_seen < len(avis):
            start, batch = out_q.get()
            if batch is _DECODER_DONE:
                sentinels_seen += 1
                continue
            end = start + len(batch)
            buf[start:end] = batch
            if progress is not None:
                progress.update(len(batch))
        if errors:
            raise errors[0]
    finally:
        if progress is not None:
            progress.close()
        for t in decoders:
            t.join(timeout=5)

    return buf


def _run_avi_mc_parallel(
    avis,
    out_counts,
    mc_zarr,
    shifts_buf,
    *,
    filtered_template,
    gSig_filt,
    upsample_factor,
    max_shift,
    chunk_t: int,
    ssub: int,
    tsub: int,
    n_jobs: int,
    n_jobs_inner: int,
    queue_maxsize: int,
    verbose: bool,
) -> None:
    """The fused decode+(bin)+MC+write loop.

    Mirrors ``concat_avis_to_zarr._run_parallel`` but swaps the
    "write batch to zarr" writer for a "MC the batch, write the corrected
    output + shifts" writer. Decoders push batches (binned to the output
    resolution when ``ssub`` / ``tsub`` > 1); the writer runs
    ``_process_batch`` (per-frame MC, parallelised across the batch via
    ``n_jobs_inner``) against the (binned) template and writes the float32
    result. ``out_counts`` are per-file OUTPUT frame counts, so batch
    ``start`` indices land at the right downsampled positions.
    """
    out_q: "queue.Queue" = queue.Queue(maxsize=queue_maxsize)
    errors: list = []
    stop_event = threading.Event()

    offsets = np.concatenate([[0], np.cumsum(out_counts[:-1])]).astype(int).tolist()
    sem = threading.Semaphore(max(1, n_jobs))
    # Keep the fractional binned means: decode straight to float32 when
    # downsampling (uint8 would round each averaged pixel before MC).
    dec_dtype = "float32" if (ssub > 1 or tsub > 1) else "uint8"

    def _decode_with_admission(*args):
        with sem:
            _decode_avi_worker(*args)

    decoders: list[threading.Thread] = []
    for i, (path, offset) in enumerate(zip(avis, offsets)):
        t = threading.Thread(
            target=_decode_with_admission,
            args=(path, i, offset, chunk_t,
                  True, "luma", dec_dtype,
                  ssub, tsub,
                  out_q, errors, stop_event),
            name=f"mc-decoder-{i}-{path.name}",
            daemon=True,
        )
        decoders.append(t)
        t.start()

    T_total = int(sum(out_counts))
    if verbose:
        try:
            from tqdm import tqdm
            progress = tqdm(total=T_total, unit="frame", desc="decode+MC+write")
        except ImportError:
            progress = None
    else:
        progress = None

    try:
        sentinels_seen = 0
        while sentinels_seen < len(avis):
            start, batch = out_q.get()
            if batch is _DECODER_DONE:
                sentinels_seen += 1
                continue
            corrected, batch_shifts = _process_batch(
                batch, filtered_template,
                gSig_filt, upsample_factor, max_shift,
                n_jobs_inner,
            )
            end = start + len(corrected)
            mc_zarr[start:end] = corrected.astype(mc_zarr.dtype)
            shifts_buf[start:end] = batch_shifts
            if progress is not None:
                progress.update(len(corrected))
        if errors:
            raise errors[0]
    finally:
        if progress is not None:
            progress.close()
        for t in decoders:
            t.join(timeout=5)
