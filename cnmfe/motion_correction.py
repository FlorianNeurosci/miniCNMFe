"""Rigid motion correction via FFT phase cross-correlation.

Algorithm
---------
1. Optionally high-pass filter both frame and template (1p miniscope data
   has slow background structure that corrupts low-frequency phase, dragging
   the cross-correlation peak off-true; CaImAn's `gSig_filt` does the same).
2. Compute the phase-normalized cross-power spectrum and its inverse FFT.
3. Mask the cross-correlation surface outside the allowed shift region —
   the peak search is constrained to be in-bounds, so a corrupted peak that
   would have landed at 60 px gets the best in-bounds peak instead, rather
   than the post-hoc-clipped wrong answer.
4. Refine to subpixel via parabolic interpolation in a 3×3 neighborhood.
5. Apply the shift as a Fourier-domain phase ramp (no interpolation artifacts).

The algorithmic shape mirrors CaImAn's `register_translation` /
`apply_shifts_dft` but is small enough that we own it directly.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import convolve
from tqdm import tqdm

from cnmfe._utils import ensure_float32, get_xp, iter_frames, to_numpy
from cnmfe.io import save_zarr

if TYPE_CHECKING:
    import zarr


def _high_pass_filter_space(img: np.ndarray, gSig_filt: float) -> np.ndarray:
    """CaImAn-style centered Gaussian band-pass kernel.

    Builds a 2D Gaussian of size (3*gSig_filt//2)*2+1, then for the central
    blob (pixels ≥ the kernel's edge value) subtracts the blob's mean and
    zeros the surrounding pixels. The result is a band-pass kernel: it
    suppresses scales much larger than gSig_filt (because the kernel
    integrates to ~0) while preserving features on scales ~gSig_filt.
    Used only on the inputs to shift estimation; the actual frame that gets
    shifted is the unfiltered original.
    """
    ksize = int((3 * gSig_filt) // 2) * 2 + 1
    if ksize < 3:
        ksize = 3
    ax = (np.arange(ksize, dtype=np.float64) - ksize // 2)
    g1 = np.exp(-0.5 * (ax / float(gSig_filt)) ** 2)
    g1 /= g1.sum()
    ker = np.outer(g1, g1)
    # Match CaImAn: keep only the central blob (pixels ≥ edge value of the
    # last column), zero-DC over that blob, zero outside it.
    threshold = ker[:, 0].max()
    nz = ker >= threshold
    ker[nz] -= ker[nz].mean()
    ker[~nz] = 0.0
    return convolve(img.astype(np.float32), ker.astype(np.float32),
                    mode="constant", cval=0.0)


def estimate_shifts(
    frame: np.ndarray,
    template: np.ndarray,
    upsample_factor: int = 10,
    max_shift: tuple[int, int] = (20, 20),
    gSig_filt: float | None = None,
) -> np.ndarray:
    """Compute subpixel (dy, dx) shift between frame and template.

    Phase correlation with the cross-correlation surface masked to the
    allowed shift region BEFORE peak finding (so a corrupted peak that
    would have escaped the bounds gets the best in-bounds peak instead),
    then refined to subpixel via 3×3 parabolic interpolation.

    Args:
        frame: (H, W) frame to register.
        template: (H, W) reference image.
        upsample_factor: kept for API compatibility; subpixel uses parabolic
            interpolation regardless.
        max_shift: (dy_max, dx_max) — peak search is constrained to this
            box around zero shift.
        gSig_filt: if not None, apply a centered Gaussian high-pass with this
            sigma to BOTH frame and template before phase correlation. The
            standard preprocessing for 1p miniscope data.

    Returns:
        shift: float32 array shape (2,) — (dy, dx) such that
        ``apply_shift(frame, shift) ≈ template``.
    """
    f = ensure_float32(frame)
    t = ensure_float32(template)
    if gSig_filt is not None and gSig_filt > 0:
        f = _high_pass_filter_space(f, float(gSig_filt))
        t = _high_pass_filter_space(t, float(gSig_filt))

    F_t = np.fft.fft2(t)
    F_f = np.fft.fft2(f)
    R = F_t * F_f.conj()
    eps = 100.0 * np.finfo(R.real.dtype).eps
    R /= np.maximum(np.abs(R), eps)                 # phase normalization

    cc = np.fft.fftshift(np.real(np.fft.ifft2(R)))
    H, W = cc.shape
    cy, cx = H // 2, W // 2
    msy = min(int(max_shift[0]), cy)
    msx = min(int(max_shift[1]), cx)

    # Constrain the peak search to the allowed shift region (CaImAn-style).
    cc_m = np.full_like(cc, -np.inf)
    cc_m[cy - msy : cy + msy + 1, cx - msx : cx + msx + 1] = (
        cc[cy - msy : cy + msy + 1, cx - msx : cx + msx + 1]
    )

    py, px = np.unravel_index(int(np.argmax(cc_m)), cc.shape)
    dy_int = py - cy
    dx_int = px - cx

    # Subpixel refinement via parabolic fit in a 3×3 neighborhood.
    def _parabolic_offset(a: float, b: float, c: float) -> float:
        denom = a - 2.0 * b + c
        if abs(denom) < 1e-12:
            return 0.0
        off = 0.5 * (a - c) / denom
        # Clamp to [-0.5, 0.5] — only meaningful for a true local max.
        if off > 0.5:
            return 0.5
        if off < -0.5:
            return -0.5
        return off

    if 0 < py < H - 1:
        dy_sub = _parabolic_offset(
            float(cc[py - 1, px]), float(cc[py, px]), float(cc[py + 1, px])
        )
    else:
        dy_sub = 0.0
    if 0 < px < W - 1:
        dx_sub = _parabolic_offset(
            float(cc[py, px - 1]), float(cc[py, px]), float(cc[py, px + 1])
        )
    else:
        dx_sub = 0.0

    return np.array([dy_int + dy_sub, dx_int + dx_sub], dtype=np.float32)


def apply_shift(frame: np.ndarray, shift: np.ndarray, xp=np) -> np.ndarray:
    """Apply a (dy, dx) subpixel shift via Fourier-domain phase multiplication.

    This is mathematically equivalent to a shift in the spatial domain but
    avoids interpolation artifacts. Shifted-in border regions are filled by the
    implicit periodicity of the DFT; in practice they are masked during analysis.

    Algorithm:
        F = FFT2(frame)
        phase_ramp = exp(1j * 2π * (-dy * Nr/H - dx * Nc/W))
        corrected = real(IFFT2(F * phase_ramp))
    where Nr, Nc are the frequency indices arranged by ifftshift.

    Args:
        frame: (H, W) frame — numpy or cupy array.
        shift: (2,) array of (dy, dx) shifts.
        xp: Array module (numpy or cupy). Defaults to numpy.

    Returns:
        corrected: (H, W) float32 shifted frame (same device as xp).
    """
    frame = xp.asarray(frame, dtype=xp.float32)
    H, W = frame.shape

    F = xp.fft.fft2(frame)

    Nr = xp.fft.ifftshift(xp.arange(-(H // 2), H - H // 2)).reshape(-1, 1).astype(xp.float32)
    Nc = xp.fft.ifftshift(xp.arange(-(W // 2), W - W // 2)).reshape(1, -1).astype(xp.float32)

    dy, dx = float(shift[0]), float(shift[1])
    phase_ramp = xp.exp(1j * 2 * xp.pi * (-dy * Nr / H - dx * Nc / W))

    return xp.real(xp.fft.ifft2(F * phase_ramp)).astype(xp.float32)


def _shift_and_correct_frame(
    frame: np.ndarray,
    template: np.ndarray,
    upsample_factor: int,
    max_shift: tuple[int, int],
    gSig_filt: float | None = None,
    roi: "tuple[slice, slice] | None" = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate and apply shift for one frame (module-level for pickling).

    Always uses CPU (numpy). Used by the joblib parallel path.

    Args:
        template: already cropped to roi when roi is set (caller's responsibility).
        roi: if set, crop frame before shift estimation; shift is applied to full frame.
    """
    frame_for_est = frame[roi] if roi is not None else frame
    shift = estimate_shifts(frame_for_est, template, upsample_factor, max_shift, gSig_filt)
    return to_numpy(apply_shift(frame, shift)), shift


def select_roi(
    movie: "zarr.Array | np.ndarray",
    frac_h: float = 0.5,
    frac_w: float = 0.5,
    n_frames: int = 800,
    border_margin: int = 40,
    hp_sigma: float = 25.0,
    blob_sigma: float = 3.0,
    use_blobness: bool = False,
) -> "tuple[slice, slice]":
    """Find the most neuron-dense rectangular crop of the movie.

    Strategy:
    1. Temporal-std image on a subsampled movie (neurons flicker; background is slow).
    2. Spatial high-pass (subtract Gaussian blur) to remove broad gradients.
    3. Optional LoG blobness filter to further favour blob-shaped structures.
    4. Mask borders to avoid edge artefacts winning.
    5. Find the crop of size (frac_h*H, frac_w*W) with highest total score via
       an integral-image (O(H*W) vectorised, no Python loop).

    Args:
        movie: (T, H, W) zarr or numpy array.
        frac_h: Crop height as a fraction of H.
        frac_w: Crop width as a fraction of W.
        n_frames: Number of evenly-spaced frames for the std image.
        border_margin: Pixels excluded from each edge of the score map.
        hp_sigma: Gaussian sigma for the background-subtraction high-pass.
        blob_sigma: LoG sigma used when use_blobness=True.
        use_blobness: Apply LoG on top of the high-passed std image to prefer
                      blob-shaped structures over vessels or bright edges.

    Returns:
        (y_slice, x_slice) — the best crop as a pair of slice objects.
    """
    from scipy.ndimage import gaussian_filter, gaussian_laplace

    T = movie.shape[0]
    H, W = movie.shape[1], movie.shape[2]

    stride    = max(1, T // n_frames)
    t_idx     = np.arange(0, T, stride)
    frames    = np.asarray(movie[t_idx], dtype=np.float32)   # (N, H, W)

    # Temporal std: neurons flicker; background is slow
    score_map = frames.std(axis=0)

    # Spatial high-pass: remove slow background gradients
    bg        = gaussian_filter(score_map, sigma=hp_sigma)
    score_map = np.clip(score_map - bg, 0, None)

    # Optional LoG blobness to prefer compact blob-like structures
    if use_blobness:
        score_map = np.clip(-gaussian_laplace(score_map, sigma=blob_sigma), 0, None)

    # Mask borders
    m = border_margin
    if m > 0:
        score_map[:m, :]  = 0
        score_map[-m:, :] = 0
        score_map[:, :m]  = 0
        score_map[:, -m:] = 0

    # Best crop via integral image (vectorised sliding-window sum)
    crop_h = min(max(8, int(H * frac_h)), H)
    crop_w = min(max(8, int(W * frac_w)), W)
    ii  = np.pad(score_map, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    y2  = np.arange(crop_h, H + 1)[:, None]
    x2  = np.arange(crop_w, W + 1)[None, :]
    y1, x1 = y2 - crop_h, x2 - crop_w
    sums = ii[y2, x2] - ii[y1, x2] - ii[y2, x1] + ii[y1, x1]
    y0, x0 = np.unravel_index(sums.argmax(), sums.shape)
    return slice(int(y0), int(y0) + crop_h), slice(int(x0), int(x0) + crop_w)


def motion_correct(
    movie: "zarr.Array | np.ndarray",
    upsample_factor: int = 10,
    max_shift: tuple[int, int] = (20, 20),
    n_iter: int = 2,
    output_path: str | Path | None = None,
    template_frames: int = 200,
    update_interval: int = 100,
    n_jobs: int = 1,
    device: str = "cpu",
    gSig_filt: float | None = None,
    roi: "tuple[slice, slice] | None" = None,
) -> tuple["zarr.Array", np.ndarray]:
    """Rigidly motion-correct a (T, H, W) movie.

    Template strategy:
    - One static template — the mean of the first `template_frames` frames
      of the raw movie — is built before the first pass and reused across
      every pass. Pass 2+ then runs against the same template on already-
      corrected frames; residual misalignments surface as small refinement
      shifts.
    - The template is NOT updated between passes. Iterative refinement
      (median-of-corrected-frames between passes, CaImAn-style) was tried
      and produced positive-feedback divergence on movies whose pass-1
      alignment is imperfect: median across partially-aligned frames stays
      smear-y, pass 2 makes mistakes correlated with pass 1's mistakes, and
      `max_shift` saturation grows with each iteration. The static template
      is empirically more stable.

    Within each batch, shift estimation and application are independent and run
    in parallel when n_jobs != 1.

    Args:
        movie: Input movie, shape (T, H, W). zarr or numpy array.
        upsample_factor: Subpixel precision = 1/upsample_factor pixels.
        max_shift: Maximum allowed shift (dy, dx) in pixels.
        n_iter: Number of correction passes (2 recommended).
        output_path: Write corrected frames here as zarr. If None, keep in memory.
        template_frames: Number of frames averaged for initial template.
        update_interval: Recompute running template every this many frames.
        n_jobs: Number of parallel workers for per-frame registration
                (-1 = all CPUs, 1 = serial). Ignored when device='cuda'.
        device: 'cpu' (default) or 'cuda'. GPU accelerates apply_shift (FFT);
                shift estimation always runs on CPU.
        gSig_filt: if not None, apply a centered Gaussian high-pass with this
                sigma to frame and template before shift estimation. Strongly
                recommended for 1p miniscope data (slow background otherwise
                corrupts low-frequency phase). The shift is applied to the
                unfiltered original frame.
        roi: optional (y_slice, x_slice). When given, shift estimation uses only
                the cropped sub-region (e.g. a neuron-dense area found by
                select_roi()), while the shift is applied to the full frame.
                Use select_roi() to find a good crop automatically.

    Returns:
        corrected: zarr.Array (or np.ndarray) of shape (T, H, W).
        shifts: np.ndarray of shape (T, 2), per-frame (dy, dx) shifts.
    """
    xp = get_xp(device)

    T = len(movie)
    first_batch = np.asarray(movie[0], dtype=np.float32)
    H, W = first_batch.shape

    n_init = min(template_frames, T)
    init_frames = np.asarray(movie[:n_init], dtype=np.float32)
    template = init_frames.mean(axis=0)
    # Pre-crop template once; all workers receive the cropped version.
    template_for_est = template[roi] if roi is not None else template

    cumulative_shifts = np.zeros((T, 2), dtype=np.float32)
    corrected_buf = np.empty((T, H, W), dtype=np.float32)

    if n_jobs != 1 and xp is np:
        from joblib import Parallel, delayed

    for iteration in range(n_iter):
        desc = f"Motion correction pass {iteration + 1}/{n_iter}"
        pass_shifts = np.zeros((T, 2), dtype=np.float32)

        for start, batch in tqdm(iter_frames(movie, batch_size=update_interval), desc=desc,
                                 total=(T + update_interval - 1) // update_interval):
            end = start + len(batch)

            if xp is not np:
                # GPU path: estimate shifts on CPU, apply on GPU
                results = []
                for frame in batch:
                    frame_for_est = frame[roi] if roi is not None else frame
                    shift = estimate_shifts(frame_for_est, template_for_est, upsample_factor, max_shift, gSig_filt)
                    corrected_frame = to_numpy(apply_shift(frame, shift, xp=xp))
                    results.append((corrected_frame, shift))
            elif n_jobs == 1:
                results = [
                    _shift_and_correct_frame(frame, template_for_est, upsample_factor, max_shift, gSig_filt, roi)
                    for frame in batch
                ]
            else:
                results = Parallel(n_jobs=n_jobs)(
                    delayed(_shift_and_correct_frame)(
                        frame, template_for_est, upsample_factor, max_shift, gSig_filt, roi
                    )
                    for frame in batch
                )

            for i, (corrected_frame, shift) in enumerate(results):
                corrected_buf[start + i] = corrected_frame
                pass_shifts[start + i] = shift

        cumulative_shifts += pass_shifts

        if iteration < n_iter - 1:
            movie = corrected_buf

    # Surface silent failures: phase correlation may report shifts beyond
    # max_shift, which estimate_shifts then clips. If a non-trivial fraction
    # of frames hit the clip on either axis (across any pass), the template
    # likely doesn't represent the data well and the correction is unreliable.
    sat_tol = 1e-3
    sat_per_pass = max(1, n_iter)
    sat_dy = np.abs(cumulative_shifts[:, 0]) >= (max_shift[0] * sat_per_pass - sat_tol)
    sat_dx = np.abs(cumulative_shifts[:, 1]) >= (max_shift[1] * sat_per_pass - sat_tol)
    n_saturated = int((sat_dy | sat_dx).sum())
    if T > 0 and n_saturated / T > 0.01:
        warnings.warn(
            f"Motion correction: {n_saturated}/{T} frames "
            f"({100.0 * n_saturated / T:.1f}%) saturated the max_shift={max_shift} "
            f"clip across all {n_iter} pass(es). Phase correlation is pointing "
            f"to shifts beyond the clip ceiling, so the correction is likely "
            f"unreliable on those frames. Consider increasing template_frames, "
            f"raising max_shift if true motion is large, or inspecting the "
            f"movie for non-rigid drift / extreme intensity changes.",
            stacklevel=2,
        )

    if output_path is not None:
        corrected_out = save_zarr(corrected_buf, output_path)
    else:
        corrected_out = corrected_buf  # type: ignore[assignment]

    return corrected_out, cumulative_shifts  # type: ignore[return-value]
