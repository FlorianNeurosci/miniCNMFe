"""Concatenate sequentially numbered AVI files into a single zarr store.

Usage (from project root or anywhere):
    python concat_avis_to_zarr.py /path/to/folder/
    python concat_avis_to_zarr.py /path/to/folder/ --output /path/to/movie.zarr
    python concat_avis_to_zarr.py /path/to/folder/ --pattern "*.avi"

The AVI files are sorted by the integer embedded in their filename:
    0.avi, 1.avi, ..., 65.avi   (numeric order, not lexicographic)

Files whose names are not purely numeric (e.g. "preview.avi") are skipped
unless you pass --pattern to change the glob.

Output zarr is time-chunked (100 frames/chunk), uint8, shape (T_total, H, W),
with lossless blosc lz4+bitshuffle compression. Use --dtype float32 for
float-valued intermediates.
It can be opened lazily with  cnmfe.io.open_zarr(output_path).

Optional inline downsampling (``--ssub`` / ``--tsub``) bins the frames as they
are decoded, so the *only* zarr written is the (downsampled) output — no
intermediate full-resolution store. Spatial binning is a block-mean over
``ssub×ssub`` pixel blocks; temporal binning averages ``tsub`` consecutive
frames. Temporal grouping is **per file** (a bin never spans two AVIs; the
trailing ``< tsub`` frames of each file are dropped). Binned means are
fractional — pass ``--dtype float32`` to keep that precision (uint8 would
round each averaged pixel).

Programmatic use:
    from concat_avis_to_zarr import concat_avis_to_zarr
    z = concat_avis_to_zarr(folder, output_path="session.zarr",
                            pattern="*.avi", skip_if_exists=True)
    # Single-pass downsample-on-write (e.g. 2x spatial, 2x temporal):
    z = concat_avis_to_zarr(folder, "ds.zarr", ssub=2, tsub=2, dtype="float32")
"""

from __future__ import annotations

import argparse
import os
import queue
import re
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import zarr as zarr_pkg  # noqa: F401


# A "done" sentinel pushed onto the queue when a decoder finishes its file.
# Using a unique singleton object so we never confuse it with a real batch.
_DECODER_DONE = object()


def _iter_frames(path: Path, grayscale: bool = True):
    """Yield frames from a single AVI file in their natural dtype.

    Used only by the serial (n_jobs=1) path. The parallel path goes through
    `_decode_avi_worker` which talks to pyav directly.
    """
    import imageio.v3 as iio
    for frame in iio.imiter(str(path), plugin="pyav"):
        frame = np.asarray(frame)  # keep natural dtype (uint8 for 8-bit AVIs)
        if grayscale and frame.ndim == 3:
            frame = frame.mean(axis=-1)  # store assignment handles final cast
        yield frame


def _count_and_shape(path: Path) -> tuple[int, int, int]:
    """Return (n_frames, H, W) for a single AVI without loading pixel data."""
    import imageio.v3 as iio
    props = iio.improps(str(path), plugin="pyav")
    n = int(props.n_images)
    _s = props.shape
    if len(_s) >= 3 and _s[0] == n:
        H, W = int(_s[1]), int(_s[2])
    else:
        H, W = int(_s[0]), int(_s[1])
    return n, H, W


def _numeric_key(path: Path) -> int:
    """Sort key: the integer in the filename stem, or -1 if not purely numeric."""
    m = re.fullmatch(r"\d+", path.stem)
    return int(m.group()) if m else -1


def _spatial_bin(frame: np.ndarray, ssub: int) -> np.ndarray:
    """Block-mean a single (H, W) frame by ``ssub`` -> (H//ssub, W//ssub) float32.

    Trailing rows/cols that don't fill a full ``ssub×ssub`` block are dropped.
    Grayscale (2-D) only — inline downsampling requires ``grayscale=True``.
    """
    H, W = frame.shape[0], frame.shape[1]
    Hu, Wu = (H // ssub) * ssub, (W // ssub) * ssub
    f = frame[:Hu, :Wu].astype(np.float32)
    return f.reshape(Hu // ssub, ssub, Wu // ssub, ssub).mean(axis=(1, 3))


def _binned_file_frames(raw_iter, ssub: int, tsub: int):
    """Yield (spatially + temporally) binned output frames for ONE file.

    Temporal grouping is per call (per file): ``tsub`` consecutive frames are
    averaged into one output frame, and the trailing ``< tsub`` frames are
    dropped. With ``ssub == tsub == 1`` this is a passthrough.
    """
    tacc = None
    tfill = 0
    for fr in raw_iter:
        if ssub > 1:
            fr = _spatial_bin(fr, ssub)
        if tsub > 1:
            if tacc is None:
                tacc = np.zeros(fr.shape, dtype=np.float32)
                tfill = 0
            tacc += fr
            tfill += 1
            if tfill < tsub:
                continue
            yield tacc / tsub
            tacc = None
            tfill = 0
        else:
            yield fr


# ---------------------------------------------------------------------------
# Parallel decode pipeline (n_jobs >= 2)
# ---------------------------------------------------------------------------

def _decode_avi_worker(
    path: Path,
    file_idx: int,
    start_offset: int,
    chunk_t: int,
    grayscale: bool,
    grayscale_method: str,
    dtype: str,
    ssub: int,
    tsub: int,
    out_q: "queue.Queue",
    errors: list,
    stop_event: threading.Event,
    crop_bbox: "tuple[int, int, int, int] | None" = None,
    mask_local: "np.ndarray | None" = None,
    frame_lo: int = 0,
    frame_hi: "int | None" = None,
) -> None:
    """Decode one AVI in a worker thread and push (start, batch) to out_q.

    Cutout (optional): only local frames in ``[frame_lo, frame_hi)`` are
    processed (temporal crop); each is sliced to ``crop_bbox`` ``(y0,y1,x0,x1)``
    and multiplied by ``mask_local`` (bool, bbox-shaped) before binning
    (spatial crop + ROI mask). All native-pixel coords; applied before
    ``ssub``/``tsub`` binning. Grayscale only.

    Each batch is a `(<=chunk_t, H, W)` (or `(.., H, W, 3)` for colour)
    ndarray already cast to `dtype`. Frames within a file stay in order;
    the absolute `start` index carried by each batch lets the writer place
    them correctly when batches from different files arrive interleaved.

    When ``ssub`` / ``tsub`` > 1 frames are binned inline (block-mean) before
    buffering, so ``start_offset`` and every emitted index are in OUTPUT
    (downsampled) frame coordinates. Temporal grouping is per file — the
    trailing ``< tsub`` frames are dropped.

    ``start_offset`` is the file's exact absolute position in the output
    zarr, computed by the caller from the (downsampled) pre-scan frame counts.

    On any exception, the exception is appended to `errors`, `stop_event`
    is set, and a `_DECODER_DONE` sentinel is sent so the writer can
    bail out.

    The Y-plane fast path (`grayscale_method="luma"`) decodes straight to
    a `(H, W) uint8` frame via pyav's `format="gray8"`, skipping any
    RGB intermediate. For grayscale-encoded sources (R==G==B in the
    original) this is pixel-identical to the historical `frame.mean(axis=-1)`
    behaviour.
    """
    try:
        import av  # pyav

        # batch buffer + how many frames have been written into it
        buf = None       # lazy-allocated once we know the (binned) frame shape
        fill = 0
        cur_start = start_offset
        # Temporal-bin accumulator (per file): sum of the current group.
        tacc = None
        tfill = 0
        local_idx = 0    # 0-based index of the decoded frame within this file

        container = av.open(str(path))
        try:
            stream = container.streams.video[0]
            # FRAME-level multithread decode helps MJPEG (each frame is
            # intra-coded). Harmless on other codecs.
            stream.thread_type = "FRAME"

            for frame in container.decode(stream):
                if stop_event.is_set():
                    return

                # Temporal cutout: skip frames before the window; stop once past it.
                i_local = local_idx
                local_idx += 1
                if i_local < frame_lo:
                    continue
                if frame_hi is not None and i_local >= frame_hi:
                    break

                if grayscale and grayscale_method == "luma":
                    arr = frame.to_ndarray(format="gray8")  # (H, W) uint8
                elif grayscale and grayscale_method == "mean":
                    rgb = frame.to_ndarray(format="rgb24")  # (H, W, 3) uint8
                    arr = rgb.mean(axis=-1)                  # float64 (H, W)
                else:
                    arr = frame.to_ndarray(format="rgb24")   # keep RGB

                # Spatial cutout: crop to bbox, then zero outside the ROI mask.
                if crop_bbox is not None:
                    y0, y1, x0, x1 = crop_bbox
                    arr = arr[y0:y1, x0:x1]
                if mask_local is not None:
                    arr = arr * mask_local

                # Inline downsample (block-mean). `out_arr` is the frame to
                # buffer, or we `continue` while a temporal group still fills.
                if ssub > 1:
                    arr = _spatial_bin(arr, ssub)
                if tsub > 1:
                    if tacc is None:
                        tacc = np.zeros(arr.shape, dtype=np.float32)
                        tfill = 0
                    tacc += arr
                    tfill += 1
                    if tfill < tsub:
                        continue
                    out_arr = tacc / tsub
                    tacc = None
                    tfill = 0
                else:
                    out_arr = arr

                if buf is None:
                    buf = np.empty((chunk_t,) + out_arr.shape, dtype=dtype)
                buf[fill] = out_arr
                fill += 1

                if fill == chunk_t:
                    out_q.put((cur_start, buf[:fill].copy()))
                    cur_start += fill
                    fill = 0
        finally:
            container.close()

        # Flush the tail (fewer than chunk_t frames).
        if buf is not None and fill > 0:
            out_q.put((cur_start, buf[:fill].copy()))
    except BaseException as exc:  # noqa: BLE001 — propagate to main thread
        errors.append(exc)
        stop_event.set()
    finally:
        out_q.put((file_idx, _DECODER_DONE))


def _writer_loop(
    store,
    out_q: "queue.Queue",
    n_files: int,
    errors: list,
    progress=None,
) -> int:
    """Consume (start, batch) tuples and write them to `store`.

    Returns total frames written. Exits once `n_files` decoder sentinels
    have been seen, or once any decoder reports an error.
    """
    sentinels_seen = 0
    total_written = 0
    while sentinels_seen < n_files:
        start, batch = out_q.get()
        if batch is _DECODER_DONE:
            sentinels_seen += 1
            continue

        end = start + len(batch)
        store[start:end] = batch
        total_written += len(batch)
        if progress is not None:
            progress.update(len(batch))

    if errors:
        raise errors[0]
    return total_written


def concat_avis_to_zarr(
    folder: "str | Path",
    output_path: "str | Path | None" = None,
    pattern: str = "*.avi",
    chunk_t: int = 500,
    dtype: str = "uint8",
    grayscale: bool = True,
    skip_if_exists: bool = False,
    n_jobs: "int | None" = None,
    grayscale_method: str = "luma",
    queue_maxsize: "int | None" = None,
    verbose: bool = True,
    clevel: int = 3,
    shuffle: str = "shuffle",
    ssub: int = 1,
    tsub: int = 1,
):
    """Concatenate numbered AVIs in `folder` into a single time-chunked zarr.

    AVI files are sorted by the integer in their filename stem (so 2.avi
    comes before 10.avi). Files whose stem isn't purely numeric are skipped.

    For multi-file sessions the parallel path (N decoder threads + 1 writer)
    gives a roughly N× speed-up on intra-coded codecs (MJPEG / motion-JPEG —
    the common miniscope case), capped by IO and memory bandwidth.

    Pipeline:
      1. Pre-scan every AVI via ``_count_and_shape`` (a pyav walk that
         counts frames) so the output zarr is sized exactly and per-file
         offsets are known up front.
      2. Allocate the output zarr.
      3. Spawn `n_jobs` decoder threads (one per AVI, admitted via a
         semaphore) that decode frames in parallel and push them through
         a bounded queue.
      4. The writer drains the queue and writes batches to the zarr at
         the position dictated by each file's pre-scan offset.

    Args:
        folder: Directory containing the AVI files.
        output_path: Output zarr path. Default: `<folder>/movie.zarr`.
        pattern: Glob pattern. Default `"*.avi"`. Use e.g. `"realistic_*.avi"`
            to pick a subset.
        chunk_t: Frames per time chunk in the output zarr. Default 500 —
            chosen so a 100k-frame session uses ~200 chunk writes instead
            of ~1000; matters on network-mounted output stores where each
            chunk write is a round-trip.
        dtype: On-disk dtype. Default `"uint8"` (matches 8-bit miniscope AVIs).
            Use `"float32"` for already-corrected/float intermediates.
        grayscale: Convert RGB → gray at read time. Default True.
        skip_if_exists: If True and output_path already exists, open and
            return it instead of raising / overwriting. Useful for notebook
            idempotency.
        n_jobs: Number of parallel decoder threads. ``1`` keeps the original
            serial path (single decoder + inline writes). ``>=2`` enables the
            producer-consumer pipeline. ``None`` (default) picks
            ``min(cpu_count, len(avis))`` — more in-flight reads materially
            help when the source AVIs sit on a network mount.
        grayscale_method: ``"luma"`` (default, fast) decodes directly to a
            ``(H, W) uint8`` Y-plane via pyav's ``format="gray8"``.
            ``"mean"`` decodes RGB and averages channels (the historical
            behaviour). For grayscale-encoded sources (R==G==B; all
            miniscope data) the two produce identical pixels.
        queue_maxsize: Bounded-queue depth for the parallel pipeline.
            ``None`` (default) = ``2 * n_jobs``. Larger uses more RAM, smaller
            risks starving the writer.
        verbose: Print progress + per-phase timing lines.
        clevel: blosc compression level for the output zarr. Default 3 —
            ~3× faster compress than the project-wide default of 5, with
            ~10 % size penalty on uint8 imaging data. The writer thread is
            single-threaded so freeing it up keeps decoders un-stalled.
        shuffle: blosc shuffle filter — ``"shuffle"`` (byte, default;
            fast), ``"bitshuffle"`` (slower, ~10 % smaller), or
            ``"noshuffle"``.
        ssub: Spatial bin factor (block-mean over ``ssub×ssub`` pixels).
            ``1`` (default) = no spatial downsampling. Requires ``grayscale``.
        tsub: Temporal bin factor (mean of ``tsub`` consecutive frames).
            ``1`` (default) = no temporal downsampling. Binning is **per file**
            — a group never spans two AVIs, and each file's trailing
            ``< tsub`` frames are dropped, so the output frame count is
            ``sum(n_i // tsub)``. Pass ``dtype="float32"`` to keep the
            fractional precision of the binned means.

    Returns:
        Open zarr.Array with shape (T_out, H//ssub, W//ssub), where
        ``T_out == sum(n_i // tsub)`` (``== T_total`` when ``tsub == 1``).

    Raises:
        FileNotFoundError: `folder` is not a directory.
        ValueError: no matching numerically-named AVIs found, or spatial
            dimensions disagree across files, or unknown ``grayscale_method``.
        FileExistsError: output exists and `skip_if_exists` is False.
    """
    import time as _time
    _t0_total = _time.time()

    if grayscale_method not in ("luma", "mean"):
        raise ValueError(
            f"grayscale_method must be 'luma' or 'mean', got {grayscale_method!r}"
        )
    if ssub < 1 or tsub < 1:
        raise ValueError(f"ssub and tsub must be >= 1 (got {ssub}, {tsub})")
    if (ssub > 1 or tsub > 1) and not grayscale:
        raise ValueError("inline downsampling (ssub/tsub > 1) requires grayscale=True")
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a directory: {folder}")

    out_path = Path(output_path) if output_path is not None else folder / "movie.zarr"

    # Early exit if caller wants idempotent behaviour.
    if out_path.exists():
        if skip_if_exists:
            if verbose:
                print(f"Output already exists, reusing: {out_path}")
            from cnmfe.io import open_zarr
            return open_zarr(out_path)
        raise FileExistsError(
            f"Output already exists: {out_path}. "
            f"Delete it or pass skip_if_exists=True to reuse it."
        )

    # --- Collect AVI files, sort numerically --------------------------------
    candidates = sorted(folder.glob(pattern), key=_numeric_key)
    avis = [p for p in candidates if _numeric_key(p) >= 0]
    if not avis:
        raise ValueError(
            f"No numerically-named AVI files found in {folder} "
            f"matching '{pattern}'. Expected files like 0.avi, 1.avi, ..."
        )

    if verbose:
        print(f"Found {len(avis)} AVI files: "
              f"{avis[0].name} ... {avis[-1].name}")

    # --- Pre-scan: walk every AVI to count frames + check spatial dims ------
    # Necessary because the output zarr must be created with an exact T;
    # decoder threads then write to known per-file offsets and there is no
    # resize-at-end pass. Over a network mount this can be a few seconds
    # per file (pyav has to scan the stream to count frames in MJPG); the
    # cost is reported in the timing line at the end.
    _t0_probe = _time.time()
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
                f"Spatial mismatch: {avi.name} is {H}x{W} "
                f"but first file is {ref_H}x{ref_W}"
            )
        counts.append(n)
        if verbose:
            print(f"  {avi.name}: {n} frames  ({H}x{W})", flush=True)
    T_total = sum(counts)
    # Per-file OUTPUT counts after (optional) temporal binning, and the
    # downsampled spatial dims. Binning is per file, so the output frame
    # count is sum(n_i // tsub) — the trailing < tsub frames of each file
    # are dropped.
    out_counts = [c // tsub for c in counts] if tsub > 1 else list(counts)
    T_out = int(sum(out_counts))
    out_H, out_W = ref_H // ssub, ref_W // ssub
    if verbose:
        print(f"\nTotal: {T_total} frames  x  {ref_H}x{ref_W} px")
        if ssub > 1 or tsub > 1:
            print(f"Downsample ssub={ssub} tsub={tsub} -> "
                  f"{T_out} frames  x  {out_H}x{out_W} px")
    _probe_secs = _time.time() - _t0_probe

    # --- Create zarr --------------------------------------------------------
    _t0_create = _time.time()
    from cnmfe.io import _open_array
    store = _open_array(out_path, "w",
                        shape=(T_out, out_H, out_W),
                        chunks=(min(chunk_t, T_out), out_H, out_W),
                        dtype=dtype,
                        compression=True,
                        clevel=clevel,
                        shuffle=shuffle)
    _create_secs = _time.time() - _t0_create
    if verbose:
        print(f"\nWriting -> {out_path}")
        print(f"  shape={store.shape}  chunks={store.chunks}  dtype={dtype}  "
              f"compression=lz4 clevel={clevel} shuffle={shuffle}", flush=True)

    # --- Pick parallel vs serial path --------------------------------------
    if n_jobs is None:
        # Raise the historical 4-cap: on network-mounted sources more
        # in-flight reads materially help. Locally, decoders are cheap
        # relative to the (single) compressing writer.
        n_jobs = min(os.cpu_count() or 1, len(avis))
    n_jobs = max(1, int(n_jobs))

    _t0_decode = _time.time()
    if n_jobs == 1:
        write_start = _run_serial(
            avis, counts, store, chunk_t, dtype, grayscale, grayscale_method,
            ssub, tsub, verbose,
        )
    else:
        if queue_maxsize is None:
            queue_maxsize = 2 * n_jobs
        write_start = _run_parallel(
            avis, out_counts, store, chunk_t, dtype, grayscale, grayscale_method,
            ssub, tsub, n_jobs, queue_maxsize, verbose,
        )
    _decode_secs = _time.time() - _t0_decode

    if verbose:
        total_secs = _time.time() - _t0_total
        bytes_total = T_out * out_H * out_W * np.dtype(dtype).itemsize
        gbps = bytes_total / max(total_secs, 1e-3) / 1e9
        print(f"\nDone. Zarr written to: {out_path}")
        print(f"  Total frames written: {write_start}")
        print(f"  Timing: pre-scan {_probe_secs:.1f}s  "
              f"create {_create_secs:.2f}s  "
              f"decode+write {_decode_secs:.1f}s  "
              f"total {total_secs:.1f}s  "
              f"({gbps:.2f} GB/s raw equivalent)")

    return store


# ---------------------------------------------------------------------------
# Serial vs parallel runners
# ---------------------------------------------------------------------------

def _run_serial(avis, counts, store, chunk_t, dtype, grayscale,
                grayscale_method, ssub, tsub, verbose) -> int:
    """Single-threaded path: same loop as before, plus the luma fast-path.

    Kept as a verbatim alternative to the parallel pipeline for debugging
    and test reproducibility. Inline downsampling (``ssub`` / ``tsub``) is
    applied per file via ``_binned_file_frames``; the chunk buffer may still
    span file boundaries (write granularity only), but temporal groups never
    do. ``counts`` are raw per-file frame counts, used only for the log line.
    """
    write_start = 0
    buf: list[np.ndarray] = []

    def _flush(buf: list, start: int) -> int:
        if not buf:
            return start
        batch = np.stack(buf, axis=0).astype(dtype)
        end = start + len(batch)
        store[start:end] = batch
        return end

    # Luma fast-path: use the same direct-to-gray8 decoder as the parallel
    # path so serial output is byte-equal to parallel output. "mean" falls
    # back to the imageio iterator (historical behaviour).
    use_luma = grayscale and grayscale_method == "luma"
    if use_luma:
        import av

    for avi_idx, avi in enumerate(avis):
        if verbose:
            print(f"  [{avi_idx + 1}/{len(avis)}] {avi.name} ...",
                  end=" ", flush=True)
        if use_luma:
            container = av.open(str(avi))
            try:
                stream = container.streams.video[0]
                stream.thread_type = "FRAME"
                raw_iter = (frame.to_ndarray(format="gray8")
                            for frame in container.decode(stream))
                for out_fr in _binned_file_frames(raw_iter, ssub, tsub):
                    buf.append(out_fr)
                    if len(buf) == chunk_t:
                        write_start = _flush(buf, write_start)
                        buf = []
            finally:
                container.close()
        else:
            raw_iter = _iter_frames(avi, grayscale=grayscale)
            for out_fr in _binned_file_frames(raw_iter, ssub, tsub):
                buf.append(out_fr)
                if len(buf) == chunk_t:
                    write_start = _flush(buf, write_start)
                    buf = []
        if verbose:
            print(f"{counts[avi_idx]} frames", flush=True)

    write_start = _flush(buf, write_start)
    return write_start


def _run_parallel(avis, out_counts, store, chunk_t, dtype, grayscale,
                  grayscale_method, ssub, tsub, n_jobs, queue_maxsize,
                  verbose) -> int:
    """Producer-consumer parallel pipeline.

    Spawns up to `n_jobs` decoder threads (one per AVI, capped at `n_jobs`
    in flight) and one writer thread. Frame batches travel through a
    bounded `queue.Queue` so RAM is `O(queue_maxsize * chunk_t * H * W)`.

    ``out_counts`` are the per-file OUTPUT frame counts (``count // tsub``
    when temporal binning, else raw counts); offsets and the progress total
    are derived from them so writes land at the right downsampled positions.
    """
    out_q: "queue.Queue" = queue.Queue(maxsize=queue_maxsize)
    errors: list = []
    stop_event = threading.Event()

    # Each AVI's absolute OUTPUT-frame offset from the (downsampled) counts.
    offsets = np.concatenate([[0], np.cumsum(out_counts[:-1])]).astype(int).tolist()

    # Spawn decoders. We cap concurrent threads at n_jobs by spawning in
    # batches; a simple Semaphore controls admission. For typical sessions
    # (len(avis) ~ n_jobs) all decoders run concurrently from the start.
    sem = threading.Semaphore(n_jobs)

    def _decode_with_admission(*args):
        with sem:
            _decode_avi_worker(*args)

    decoder_threads: list[threading.Thread] = []
    for i, (path, offset) in enumerate(zip(avis, offsets)):
        t = threading.Thread(
            target=_decode_with_admission,
            args=(path, i, offset, chunk_t, grayscale, grayscale_method,
                  dtype, ssub, tsub, out_q, errors, stop_event),
            name=f"decoder-{i}-{path.name}",
            daemon=True,
        )
        decoder_threads.append(t)
        t.start()

    # Run the writer on the main thread (no extra thread needed; cleaner
    # exception propagation). Use tqdm for a frames-written progress bar.
    T_total = int(sum(out_counts))
    if verbose:
        try:
            from tqdm import tqdm
            progress = tqdm(total=T_total, unit="frame", desc="decode+write")
        except ImportError:
            progress = None
    else:
        progress = None

    try:
        total_written = _writer_loop(store, out_q, n_files=len(avis),
                                     errors=errors, progress=progress)
    finally:
        if progress is not None:
            progress.close()
        # All decoders should have finished by now (writer waits for all
        # sentinels). Join them so any exceptions surface cleanly.
        for t in decoder_threads:
            t.join(timeout=5)

    return total_written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path,
                        help="Directory containing numbered AVI files")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output zarr path (default: <folder>/movie.zarr)")
    parser.add_argument("--pattern", default="*.avi",
                        help="Glob pattern for AVI files (default: *.avi)")
    parser.add_argument("--chunk-t", type=int, default=500,
                        help="Frames per time chunk in zarr (default: 500)")
    parser.add_argument("--dtype", default="uint8",
                        help="On-disk dtype (default: uint8; use float32 for float intermediates)")
    parser.add_argument("--color", action="store_true",
                        help="Keep colour channels (default: convert to grayscale)")
    parser.add_argument("--skip-if-exists", action="store_true",
                        help="If output zarr already exists, reuse it (default: error)")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="Parallel decoder threads (default: auto = min(cpu, len(avis)); "
                             "1 = original serial path)")
    parser.add_argument("--grayscale-method", choices=("luma", "mean"),
                        default="luma",
                        help="Grayscale conversion: 'luma' (Y-plane, fast, default) or "
                             "'mean' (RGB average, historical)")
    parser.add_argument("--queue-maxsize", type=int, default=None,
                        help="Bounded-queue depth in the parallel pipeline "
                             "(default: 2 * n_jobs)")
    parser.add_argument("--clevel", type=int, default=3,
                        help="blosc compression level for the output zarr "
                             "(default: 3, fast)")
    parser.add_argument("--shuffle", choices=("shuffle", "bitshuffle", "noshuffle"),
                        default="shuffle",
                        help="blosc shuffle filter (default: 'shuffle' = byte "
                             "shuffle, fast)")
    parser.add_argument("--ssub", type=int, default=1,
                        help="Spatial bin factor, block-mean (default: 1 = none)")
    parser.add_argument("--tsub", type=int, default=1,
                        help="Temporal bin factor, per-file mean (default: 1 = none)")
    args = parser.parse_args()

    try:
        store = concat_avis_to_zarr(
            folder=args.folder,
            output_path=args.output,
            pattern=args.pattern,
            chunk_t=args.chunk_t,
            dtype=args.dtype,
            grayscale=not args.color,
            skip_if_exists=args.skip_if_exists,
            n_jobs=args.n_jobs,
            grayscale_method=args.grayscale_method,
            queue_maxsize=args.queue_maxsize,
            clevel=args.clevel,
            shuffle=args.shuffle,
            ssub=args.ssub,
            tsub=args.tsub,
            verbose=True,
        )
    except (FileNotFoundError, ValueError, FileExistsError) as exc:
        parser.error(str(exc))
        return  # unreachable

    out_path = args.output if args.output else args.folder.resolve() / "movie.zarr"
    print(f"\nLoad lazily with:")
    print(f"  from cnmfe.io import open_zarr")
    print(f"  z = open_zarr('{out_path}')")
    _ = store  # silence linters


if __name__ == "__main__":
    main()
