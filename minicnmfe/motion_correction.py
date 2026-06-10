from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    import zarr as zarr_pkg


def _is_zarr_array(obj) -> bool:
    """Duck-type check for a zarr.Array (avoids hard zarr import)."""
    try:
        import zarr
        return isinstance(obj, zarr.Array)
    except ImportError:
        return False


# =============================================================================
# CAIMAN-COMPATIBLE HIGH PASS FILTER
# =============================================================================

def high_pass_filter_space(img_orig, gSig_filt):

    if np.isscalar(gSig_filt):
        gSig_filt = [gSig_filt, gSig_filt]

    # ksize must be an int for cv2 (gSig_filt may be a float, e.g. after
    # CNMFeParams.downscaled scales it by ssub). For integer gSig_filt this
    # is identical to the historical expression.
    ksize = tuple([int((3 * i) // 2 * 2 + 1) for i in gSig_filt])

    ker = cv2.getGaussianKernel(ksize[0], gSig_filt[0])
    ker2D = ker.dot(ker.T)

    nz = np.nonzero(ker2D >= ker2D[:, 0].max())
    zz = np.nonzero(ker2D < ker2D[:, 0].max())

    ker2D[nz] -= ker2D[nz].mean()
    ker2D[zz] = 0

    return cv2.filter2D(
        img_orig.astype(np.float32),
        -1,
        ker2D,
        borderType=cv2.BORDER_REFLECT
    )


# =============================================================================
# CAIMAN BIN MEDIAN
# =============================================================================

def caiman_bin_median(mat, window=10):

    T, d1, d2 = mat.shape

    if T < window:
        window = T

    num_windows = int(T // window)

    if num_windows == 0:
        return np.median(mat, axis=0)

    num_frames = num_windows * window

    return np.nanmedian(
        np.nanmean(
            np.reshape(
                mat[:num_frames],
                (window, num_windows, d1, d2)
            ),
            axis=0
        ),
        axis=0
    )


# =============================================================================
# OPENCV FFT
# =============================================================================

def cv2_fft2(img):

    dft = cv2.dft(
        img.astype(np.float32),
        flags=cv2.DFT_COMPLEX_OUTPUT + cv2.DFT_SCALE
    )

    return dft[:, :, 0] + 1j * dft[:, :, 1]


def cv2_ifft2(freq):

    freq_cv = np.dstack([
        np.real(freq),
        np.imag(freq)
    ]).astype(np.float32)

    out = cv2.dft(
        freq_cv,
        flags=cv2.DFT_INVERSE + cv2.DFT_SCALE
    )

    return out[:, :, 0] + 1j * out[:, :, 1]


# =============================================================================
# UPSAMPLED DFT
# =============================================================================
def _upsampled_dft(data,
                   upsampled_region_size,
                   upsample_factor=1,
                   axis_offsets=None):

    """
    Upsampled DFT used for subpixel registration.

    Directly adapted from skimage/CaImAn implementation.
    """

    if axis_offsets is None:
        axis_offsets = [0, 0]

    im2pi = 1j * 2 * np.pi

    nr, nc = data.shape

    # -----------------------------------------------------------------
    # kernel for columns
    # shape:
    #   (nc, upsampled_region_size)
    # -----------------------------------------------------------------

    kernc = np.exp(
        (-im2pi / (nc * upsample_factor))
        * (
            (np.fft.ifftshift(np.arange(nc)) - np.floor(nc / 2))[:, None]
        )
        * (
            np.arange(upsampled_region_size)[None, :]
            - axis_offsets[1]
        )
    )

    # -----------------------------------------------------------------
    # kernel for rows
    # shape:
    #   (upsampled_region_size, nr)
    # -----------------------------------------------------------------

    kernr = np.exp(
        (-im2pi / (nr * upsample_factor))
        * (
            (np.arange(upsampled_region_size)[:, None])
            - axis_offsets[0]
        )
        * (
            np.fft.ifftshift(np.arange(nr))[None, :]
            - np.floor(nr / 2)
        )
    )

    return kernr @ data @ kernc

# =============================================================================
# CAIMAN-COMPATIBLE SHIFT ESTIMATION
# =============================================================================

def register_translation_caiman(
        src_image,
        target_image,
        upsample_factor=10,
        max_shifts=(20, 20),
):

    src_freq = cv2_fft2(src_image)
    tgt_freq = cv2_fft2(target_image)

    image_product = src_freq * tgt_freq.conj()

    eps = np.finfo(np.float32).eps

    image_product /= np.maximum(
        np.abs(image_product),
        100 * eps
    )

    cross_corr = cv2_ifft2(image_product)
    cross_corr = np.abs(cross_corr)

    # -----------------------------------------------------------------
    # constrain shifts EXACTLY like CaImAn
    # -----------------------------------------------------------------

    constrained = cross_corr.copy()

    constrained[
        max_shifts[0]:-max_shifts[0],
        :
    ] = 0

    constrained[
        :,
        max_shifts[1]:-max_shifts[1]
    ] = 0

    maxima = np.unravel_index(
        np.argmax(constrained),
        constrained.shape
    )

    midpoints = np.array([
        np.fix(axis_size / 2)
        for axis_size in src_image.shape
    ])

    shifts = np.array(maxima, dtype=np.float64)

    shifts[shifts > midpoints] -= np.array(
        src_image.shape
    )[shifts > midpoints]

    # -----------------------------------------------------------------
    # subpixel refinement
    # -----------------------------------------------------------------

    if upsample_factor > 1:

        upsampled_region_size = int(
            np.ceil(upsample_factor * 1.5)
        )

        dftshift = np.fix(
            upsampled_region_size / 2.0
        )

        sample_region_offset = (
            dftshift - shifts * upsample_factor
        )

        cross_corr_up = _upsampled_dft(
            image_product.conj(),
            upsampled_region_size,
            upsample_factor,
            sample_region_offset
        ).conj()

        cross_corr_up = np.abs(cross_corr_up)

        maxima = np.unravel_index(
            np.argmax(cross_corr_up),
            cross_corr_up.shape
        )

        maxima = np.array(
            maxima,
            dtype=np.float64
        )

        maxima -= dftshift

        shifts += maxima / upsample_factor

    return (
        float(shifts[0]),
        float(shifts[1])
    )


# =============================================================================
# APPLY SHIFT EXACTLY LIKE CAIMAN
# =============================================================================
def apply_shift_caiman(img, shift):

    row_shift, col_shift = shift

    h, w = img.shape

    M = np.array([
        [1, 0, col_shift],
        [0, 1, row_shift]
    ], dtype=np.float32)

    shifted = cv2.warpAffine(
        img.astype(np.float32),
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    shifted = np.nan_to_num(shifted)

    shifted = np.clip(
        shifted,
        0,
        None
    )

    return shifted


# def apply_shift_caiman(img, shift):
# # we thought this function introduces some artefacts, but that is likely wrong. so consider switching back but the
# # other function is giving the same correspondence to caimans mc.
#
#     row_shift, col_shift = shift
#
#     h, w = img.shape
#
#     M = np.array([
#         [1, 0, col_shift],
#         [0, 1, row_shift]
#     ], dtype=np.float32)
#
#     min_ = np.nanmin(img)
#     max_ = np.nanmax(img)
#
#     shifted = cv2.warpAffine(
#         img.astype(np.float32),
#         M,
#         (w, h),
#         flags=cv2.INTER_CUBIC,
#         borderMode=cv2.BORDER_REFLECT
#     )
#
#     shifted = np.clip(
#         shifted,
#         min_,
#         max_
#     )
#
#     return shifted


# =============================================================================
# Worker (module-level for joblib spawn pickling)
# =============================================================================

def _filter_estimate_apply(
    frame, filtered_template, gSig_filt, upsample_factor, max_shift,
):
    """Per-frame work: high-pass filter -> estimate shift vs template -> warp.

    Module-level so joblib's spawn-based pickling works on Windows.
    `filtered_template` must already be high-pass-filtered (we re-filter only
    the frame here, not the template, to avoid duplicate work).

    Returns:
        corrected: (H, W) float32
        shift: (2,) float32 — (dy, dx)
    """
    frame = np.asarray(frame, dtype=np.float32)
    if gSig_filt is not None:
        filtered_frame = high_pass_filter_space(frame, gSig_filt)
    else:
        filtered_frame = frame
    dy, dx = register_translation_caiman(
        filtered_template, filtered_frame,
        upsample_factor=upsample_factor, max_shifts=max_shift,
    )
    corrected = apply_shift_caiman(frame, (dy, dx))
    return corrected, np.array([dy, dx], dtype=np.float32)


def _process_batch(
    batch, filtered_template, gSig_filt, upsample_factor, max_shift, n_jobs,
):
    """Process a (B, H, W) batch in parallel. Returns (corrected_batch, shifts_batch).

    Frames within a batch are independent given a fixed template, so they
    parallelize trivially via joblib. For n_jobs=1 the serial path avoids
    Parallel's process-pool overhead.
    """
    B = batch.shape[0]
    if n_jobs == 1 or B == 1:
        out = np.empty_like(batch, dtype=np.float32)
        shifts = np.empty((B, 2), dtype=np.float32)
        for i in range(B):
            out[i], shifts[i] = _filter_estimate_apply(
                batch[i], filtered_template, gSig_filt, upsample_factor, max_shift,
            )
        return out, shifts

    from joblib import Parallel, delayed
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_filter_estimate_apply)(
            batch[i], filtered_template, gSig_filt, upsample_factor, max_shift,
        )
        for i in range(B)
    )
    out = np.stack([r[0] for r in results], axis=0)
    shifts = np.stack([r[1] for r in results], axis=0)
    return out, shifts


# =============================================================================
# Streaming helpers (zarr-backed)
# =============================================================================

def _sample_frame_indices(T: int, max_frames: int) -> np.ndarray:
    """Return up to max_frames evenly-spaced int indices into [0, T)."""
    if T <= max_frames:
        return np.arange(T, dtype=int)
    return np.linspace(0, T - 1, max_frames).astype(int)


def _build_template_streaming(
    source,            # zarr.Array or np.ndarray supporting .shape and slicing
    gSig_filt,
    bin_window,
    batch_size,
    template_max_frames,
    verbose=True,
):
    """Build a CaImAn-style bin-median template from a strided sample of frames.

    Reads source in batches (chunk-friendly), high-pass filters the sampled
    frames, then runs caiman_bin_median on the subsample.

    Memory budget: ~template_max_frames * H * W * 4 + batch_size * H * W * 4 bytes.
    """
    T, H, W = source.shape[0], source.shape[1], source.shape[2]
    idx = _sample_frame_indices(T, template_max_frames)
    sampled = np.empty((len(idx), H, W), dtype=np.float32)
    fill = 0

    # Group sampled indices by batch so we read each batch at most once.
    iter_ = range(0, T, batch_size)
    if verbose:
        iter_ = tqdm(iter_, desc="MC template")
    for start in iter_:
        end = min(start + batch_size, T)
        in_batch = [int(t) for t in idx if start <= t < end]
        if not in_batch:
            continue
        batch = np.asarray(source[start:end], dtype=np.float32)
        for t in in_batch:
            frame = batch[t - start]
            if gSig_filt is not None:
                frame = high_pass_filter_space(frame, gSig_filt)
            sampled[fill] = frame
            fill += 1

    return caiman_bin_median(sampled[:fill], window=bin_window)


def _create_output_zarr(path, shape, chunks, dtype, compression):
    """Create an empty output zarr store via the project's _open_array."""
    from minicnmfe.io import _open_array
    return _open_array(
        Path(path), "w", shape=shape, chunks=chunks,
        dtype=dtype, compression=compression,
    )


def _run_pass_zarr(
    src, dst, filtered_template, gSig_filt, upsample_factor,
    max_shift, batch_size, n_jobs, verbose=True, desc="MC pass",
):
    """One rigid MC pass: src -> dst. Returns shifts (T, 2).

    src and dst can be zarr.Array or numpy. We never hold more than one batch
    of source + one batch of corrected output in RAM at a time.
    """
    T = src.shape[0]
    shifts = np.zeros((T, 2), dtype=np.float32)
    dst_dtype = np.dtype(dst.dtype)
    iter_ = range(0, T, batch_size)
    if verbose:
        iter_ = tqdm(iter_, desc=desc)
    for start in iter_:
        end = min(start + batch_size, T)
        batch = np.asarray(src[start:end], dtype=np.float32)
        corrected_batch, shifts_batch = _process_batch(
            batch, filtered_template, gSig_filt, upsample_factor, max_shift, n_jobs,
        )
        if corrected_batch.dtype != dst_dtype:
            corrected_batch = corrected_batch.astype(dst_dtype)
        dst[start:end] = corrected_batch
        shifts[start:end] = shifts_batch
    return shifts


# =============================================================================
# Streaming main path
# =============================================================================

def _motion_correction_streaming(
    src,
    output_path,
    max_shift,
    gSig_filt,
    upsample_factor,
    niter_rig,
    bin_window,
    template,
    batch_size,
    n_jobs,
    template_max_frames,
    output_chunk_t,
    output_dtype,
    compression,
    verbose,
):
    output_path = Path(output_path)
    T, H, W = src.shape[0], src.shape[1], src.shape[2]

    if output_chunk_t is None:
        # Match source chunking when possible; otherwise pick a reasonable default.
        chunk_t = int(src.chunks[0]) if hasattr(src, "chunks") else min(batch_size, T)
    else:
        chunk_t = int(output_chunk_t)
    chunks_out = (chunk_t, H, W)

    if template is None:
        if verbose:
            print("Building template from strided sample...")
        template = _build_template_streaming(
            src, gSig_filt, bin_window, batch_size, template_max_frames, verbose=verbose,
        )

    shifts_total = np.zeros((T, 2), dtype=np.float32)

    if niter_rig <= 1:
        # Single pass: src -> output_path directly.
        if output_path.exists():
            shutil.rmtree(output_path)
        dst = _create_output_zarr(
            output_path, (T, H, W), chunks_out, output_dtype, compression,
        )
        filtered_template = (
            high_pass_filter_space(template, gSig_filt)
            if gSig_filt is not None else template
        )
        if verbose:
            print(f"Rigid iteration 1/{niter_rig}")
        shifts_total = _run_pass_zarr(
            src, dst, filtered_template, gSig_filt, upsample_factor,
            max_shift, batch_size, n_jobs, verbose=verbose,
        )
        return dst, shifts_total

    # niter_rig >= 2: ping-pong between two scratch zarrs, rename final to output_path.
    scratch_a = output_path.parent / (f".{output_path.name}.scratch_a.zarr")
    scratch_b = output_path.parent / (f".{output_path.name}.scratch_b.zarr")
    for p in (scratch_a, scratch_b, output_path):
        if p.exists():
            shutil.rmtree(p)

    current_src = src
    final_scratch = None
    for iteration in range(niter_rig):
        dst_path = scratch_a if (iteration % 2 == 0) else scratch_b
        if dst_path.exists():
            shutil.rmtree(dst_path)
        dst = _create_output_zarr(
            dst_path, (T, H, W), chunks_out, output_dtype, compression,
        )
        filtered_template = (
            high_pass_filter_space(template, gSig_filt)
            if gSig_filt is not None else template
        )
        if verbose:
            print(f"Rigid iteration {iteration + 1}/{niter_rig}")
        shifts_iter = _run_pass_zarr(
            current_src, dst, filtered_template, gSig_filt, upsample_factor,
            max_shift, batch_size, n_jobs, verbose=verbose,
            desc=f"MC pass {iteration + 1}/{niter_rig}",
        )
        shifts_total += shifts_iter

        if iteration < niter_rig - 1:
            if verbose:
                print("  Updating template from corrected frames...")
            template = _build_template_streaming(
                dst, gSig_filt, bin_window, batch_size, template_max_frames,
                verbose=verbose,
            )
        current_src = dst
        final_scratch = dst_path

    # Move the last scratch to output_path and clean up the other.
    other = scratch_a if final_scratch == scratch_b else scratch_b
    if other.exists():
        try:
            shutil.rmtree(other)
        except OSError:
            pass
    # shutil.move is portable across filesystems; same-fs rename is atomic.
    # If the move fails (cross-fs error, permission, AV lock), make sure both
    # scratch zarrs are removed so we don't accrue 2-3x the movie size on disk
    # across retries.
    try:
        shutil.move(str(final_scratch), str(output_path))
    finally:
        for p in (scratch_a, scratch_b):
            if p != output_path and p.exists():
                try:
                    shutil.rmtree(p)
                except OSError:
                    pass

    import zarr
    return zarr.open_array(str(output_path), mode="r+"), shifts_total


# =============================================================================
# In-memory main path (preserved for small numpy inputs)
# =============================================================================

def _motion_correction_in_memory(
    movie,
    max_shift,
    gSig_filt,
    upsample_factor,
    niter_rig,
    bin_window,
    template,
    batch_size,
    n_jobs,
    template_max_frames,
    verbose,
    in_place=False,
):
    """In-memory rigid MC with bounded peak RAM.

    Peak RAM ~= 1x the movie when ``in_place=True`` (frames are warped back into
    the input buffer), ~2x otherwise (one owned output buffer). Either way the
    template is built from a strided sample via ``_build_template_streaming`` —
    not a full-movie high-pass copy — so the previous ~3-4x peak (an extra
    ``filtered`` / ``filtered_corrected`` / ``corrected_iter`` full-movie float32
    buffer each) is gone. float32 throughout.

    ``in_place=True`` mutates the caller's array and is only safe when the caller
    treats the movie as consumed (it does NOT, e.g., recompute a "raw" image from
    it afterwards). Default False keeps the input intact.
    """
    movie = np.asarray(movie, dtype=np.float32)
    T, H, W = movie.shape

    if template is None:
        # Sample-based template (same builder the streaming path uses, so the two
        # paths agree). Allocates ~template_max_frames*H*W*4, not T*H*W*4.
        template = _build_template_streaming(
            movie, gSig_filt, bin_window, batch_size, template_max_frames,
            verbose=verbose,
        )

    corrected = movie if in_place else np.empty_like(movie)
    shifts_total = np.zeros((T, 2), dtype=np.float32)

    for iteration in range(niter_rig):
        if verbose:
            print(f"\nRigid iteration {iteration + 1}/{niter_rig}")

        filtered_template = (
            high_pass_filter_space(template, gSig_filt)
            if gSig_filt is not None else template
        )

        # Read the original movie on the first pass, the running corrected movie
        # thereafter. _process_batch returns a fresh array per batch, so writing
        # it back into `corrected` is safe even when corrected aliases the source
        # (frames are independent given a fixed template).
        source = movie if iteration == 0 else corrected
        shifts_iter = np.zeros((T, 2), dtype=np.float32)
        iter_ = range(0, T, batch_size)
        if verbose:
            iter_ = tqdm(iter_)
        for start in iter_:
            end = min(start + batch_size, T)
            corrected_batch, shifts_batch = _process_batch(
                source[start:end], filtered_template, gSig_filt,
                upsample_factor, max_shift, n_jobs,
            )
            corrected[start:end] = corrected_batch
            shifts_iter[start:end] = shifts_batch

        shifts_total += shifts_iter

        # Rebuild the template from the corrected frames for the NEXT pass only
        # (the last pass's template would be unused). Sample-based again.
        if iteration + 1 < niter_rig:
            template = _build_template_streaming(
                corrected, gSig_filt, bin_window, batch_size, template_max_frames,
                verbose=verbose,
            )

    return corrected, shifts_total


# =============================================================================
# MAIN PUBLIC FUNCTION
# =============================================================================

def motion_correction_rigid(
    movie,
    output_path: "str | Path | None" = None,
    max_shift=(20, 20),
    gSig_filt=7,
    upsample_factor=10,
    niter_rig=1,
    bin_window=10,
    template=None,
    batch_size: int = 200,
    n_jobs: int = 1,
    template_max_frames: int = 2000,
    output_chunk_t: "int | None" = None,
    output_dtype: str = "float32",
    compression: bool = True,
    verbose: bool = True,
    in_place: bool = False,
):
    """CaImAn-compatible rigid motion correction.

    Two execution paths, chosen automatically:

    - **Streaming (zarr-backed)** — used when the input is a zarr.Array or when
      ``output_path`` is given. Reads/writes batches of frames; peak RAM is
      ``(batch_size + template_max_frames) * H * W * 4`` bytes, independent of T.
      Returns the output zarr handle and the (T, 2) shifts.
    - **In-memory** — used when the input is a numpy array and no ``output_path``
      is given (the small-movie / test path). Same algorithm, returns a numpy
      corrected movie. Parallelized over frames via joblib when ``n_jobs > 1``.

    Args:
        movie: Input movie, shape (T, H, W). zarr.Array or np.ndarray.
        output_path: If given, write corrected movie as a zarr store here and
            return the zarr handle. **Required when ``movie`` is a zarr.Array.**
        max_shift: Maximum allowed (dy, dx) shift, in pixels.
        gSig_filt: Sigma for the high-pass filter applied before cross-
            correlation. Required for 1-photon data; set to None for 2p.
        upsample_factor: Subpixel refinement factor (10 = 0.1 px precision).
        niter_rig: Number of rigid passes. Each pass re-estimates the template
            from the previously corrected frames.
        bin_window: Frame-binning window for the median template (CaImAn default
            10).
        template: Optional precomputed (H, W) template. When None, built from a
            strided sample of the movie.
        batch_size: Frames per batch in the streaming/parallel loop. Tune for
            your RAM/IO trade-off. Default 200.
        n_jobs: CPU workers for per-frame work within a batch. 1 = serial,
            -1 = all cores.
        template_max_frames: Cap on frames sampled for template estimation
            (uniformly strided over the movie). Bounds RAM.
        output_chunk_t: Time-axis chunk size for the output zarr. Default:
            match source chunks if available, else ``batch_size``.
        output_dtype: dtype of the output zarr (default "float32").
        compression: Use blosc lz4+bitshuffle compression on the output zarr.
        verbose: tqdm progress bars + iteration log lines.

    Returns:
        corrected: zarr.Array if ``output_path`` was given (or input was zarr),
            else np.ndarray of shape (T, H, W) float32.
        shifts: np.ndarray (T, 2) float32, (dy, dx) per frame. Sum of per-
            iteration shifts when niter_rig > 1.
    """
    is_zarr_in = _is_zarr_array(movie)
    use_streaming = is_zarr_in or output_path is not None

    if use_streaming and output_path is None:
        # Reachable only if is_zarr_in and not output_path
        raise ValueError(
            "motion_correction_rigid: zarr input requires output_path "
            "(streaming MC writes the corrected movie to a zarr store). "
            "Pass output_path='.../mc.zarr' or pre-load the movie as a "
            "numpy array if it fits in RAM."
        )

    if use_streaming:
        return _motion_correction_streaming(
            movie, output_path, max_shift, gSig_filt, upsample_factor,
            niter_rig, bin_window, template, batch_size, n_jobs,
            template_max_frames, output_chunk_t, output_dtype, compression,
            verbose,
        )

    return _motion_correction_in_memory(
        movie, max_shift, gSig_filt, upsample_factor, niter_rig, bin_window,
        template, batch_size, n_jobs, template_max_frames, verbose,
        in_place=in_place,
    )


# ---------------------------------------------------------------------------
# Convenience aliases used by the rest of the package
# ---------------------------------------------------------------------------

def apply_shift(img: np.ndarray, shift) -> np.ndarray:
    """Alias for apply_shift_caiman — apply (dy, dx) shift via cv2.warpAffine."""
    return apply_shift_caiman(img, shift)


def estimate_shifts(
    frame: np.ndarray,
    template: np.ndarray,
    upsample_factor: int = 10,
    max_shift=(20, 20),
    gSig_filt: float | None = None,
) -> np.ndarray:
    """Estimate subpixel (dy, dx) shift between frame and template.

    Thin wrapper around register_translation_caiman that optionally applies
    the CaImAn high-pass filter before cross-correlation.

    Returns:
        shift: float32 array shape (2,) — (dy, dx)
    """
    f = frame.astype(np.float32)
    t = template.astype(np.float32)
    if gSig_filt is not None and gSig_filt > 0:
        f = high_pass_filter_space(f, float(gSig_filt))
        t = high_pass_filter_space(t, float(gSig_filt))
    dy, dx = register_translation_caiman(t, f, upsample_factor=upsample_factor,
                                          max_shifts=max_shift)
    return np.array([dy, dx], dtype=np.float32)