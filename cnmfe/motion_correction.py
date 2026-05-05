"""Rigid motion correction via FFT phase cross-correlation.

Algorithm
---------
1. Compute the cross-power spectrum of a frame and the template in Fourier space.
2. Find the peak of the inverse FFT → coarse integer shift.
3. Refine to subpixel accuracy using an upsampled DFT (matrix-multiply DFT).
4. Apply the shift as a Fourier-domain phase ramp (no interpolation artifacts).

This matches the algorithm in CaImAn's `register_translation` /
`apply_shifts_dft` but uses numpy.fft throughout (no cv2 dependency).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from skimage.registration import phase_cross_correlation
from tqdm import tqdm

from cnmfe._utils import ensure_float32, get_xp, iter_frames, to_numpy
from cnmfe.io import save_zarr

if TYPE_CHECKING:
    import zarr


def estimate_shifts(
    frame: np.ndarray,
    template: np.ndarray,
    upsample_factor: int = 10,
    max_shift: tuple[int, int] = (20, 20),
) -> np.ndarray:
    """Compute subpixel (dy, dx) shift between frame and template.

    Uses phase cross-correlation with upsampled DFT refinement.
    The detected shift is clipped to max_shift to prevent runaway corrections.

    Returns:
        shift: float array shape (2,) — (dy, dx) in pixels.
    """
    frame = ensure_float32(frame)
    template = ensure_float32(template)

    shift, _, _ = phase_cross_correlation(
        template,
        frame,
        upsample_factor=upsample_factor,
        normalization=None,
    )
    # Clip to max_shift
    shift = np.clip(shift, -np.array(max_shift), np.array(max_shift))
    return shift.astype(np.float32)


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
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate and apply shift for one frame (module-level for pickling).

    Always uses CPU (numpy). Used by the joblib parallel path.
    """
    shift = estimate_shifts(frame, template, upsample_factor, max_shift)
    return to_numpy(apply_shift(frame, shift)), shift


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
) -> tuple["zarr.Array", np.ndarray]:
    """Rigidly motion-correct a (T, H, W) movie.

    Template strategy:
    - Pass 1: initialize template from the mean of the first `template_frames` frames,
      then update as a running mean every `update_interval` frames.
    - Pass 2 (if n_iter >= 2): restart with the final template from pass 1.

    Within each batch, shift estimation and application are independent and run
    in parallel when n_jobs != 1. Template updates remain sequential.

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
                shift estimation always runs on CPU (skimage requirement).

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
                # GPU path: estimate shifts on CPU (skimage), apply on GPU
                results = []
                for frame in batch:
                    shift = estimate_shifts(frame, template, upsample_factor, max_shift)
                    corrected_frame = to_numpy(apply_shift(frame, shift, xp=xp))
                    results.append((corrected_frame, shift))
            elif n_jobs == 1:
                results = [
                    _shift_and_correct_frame(frame, template, upsample_factor, max_shift)
                    for frame in batch
                ]
            else:
                results = Parallel(n_jobs=n_jobs)(
                    delayed(_shift_and_correct_frame)(
                        frame, template, upsample_factor, max_shift
                    )
                    for frame in batch
                )

            corrected_frames = []
            for i, (corrected_frame, shift) in enumerate(results):
                corrected_buf[start + i] = corrected_frame
                pass_shifts[start + i] = shift
                corrected_frames.append(corrected_frame)

            template = np.stack(corrected_frames, axis=0).mean(axis=0)

        cumulative_shifts += pass_shifts

        if iteration < n_iter - 1:
            movie = corrected_buf

    if output_path is not None:
        corrected_out = save_zarr(corrected_buf, output_path)
    else:
        corrected_out = corrected_buf  # type: ignore[assignment]

    return corrected_out, cumulative_shifts  # type: ignore[return-value]
