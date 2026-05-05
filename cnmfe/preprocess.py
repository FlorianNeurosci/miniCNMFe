"""Preprocessing: noise estimation and CORR/PNR summary images.

These images are the inputs to seed detection in CNMFe initialization.
All operations work chunk-by-chunk to avoid loading the full movie.

References (algorithmic only, no imports):
    CaImAn summary_images.py:correlation_pnr (line 286)
    CaImAn pre_processing.py:get_noise_fft (line 128)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.ndimage as ndi

from cnmfe._utils import ensure_float32, get_xp, iter_frames, to_numpy

if TYPE_CHECKING:
    import zarr


def make_center_surround_psf(sigma: float, size: int | None = None) -> np.ndarray:
    """Build a centered Gaussian PSF for 1-photon background suppression.

    The kernel is a Gaussian with its mean subtracted on the nonzero support.
    This makes the kernel sum to zero, suppressing spatially uniform background
    while preserving signal at the neuron scale (radius ≈ sigma pixels).

    Args:
        sigma: Gaussian standard deviation in pixels.
        size: Kernel side length. Defaults to 2 * ceil(3*sigma) + 1.

    Returns:
        psf: 2D float32 kernel, sum ≈ 0.
    """
    import math

    if size is None:
        size = 2 * math.ceil(3 * sigma) + 1

    half = size // 2
    y, x = np.ogrid[-half : half + 1, -half : half + 1]
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2)).astype(np.float32)

    # Create a disk mask that excludes the very centre (centre-surround)
    outer_disk = (x**2 + y**2) <= (3 * sigma) ** 2
    g[~outer_disk] = 0.0

    # Subtract mean so the kernel integrates to zero on its support
    nonzero = g > 0
    g[nonzero] -= g[nonzero].mean()
    g[~nonzero] = 0.0
    return g


def estimate_noise(
    movie: "zarr.Array | np.ndarray",
    noise_range: tuple[float, float] = (0.25, 0.5),
    method: str = "logmexp",
) -> np.ndarray:
    """Estimate per-pixel noise std from the high-frequency power spectrum.

    For each pixel the power spectral density is estimated via rfft along time.
    Frequencies in `noise_range` * Nyquist are assumed to be pure noise,
    giving a noise floor estimate that is robust to low-frequency calcium signals.

    Args:
        movie: (T, H, W) array.
        noise_range: Fraction of Nyquist to use for noise estimation.
        method: 'mean', 'median', or 'logmexp' (geometric mean via log).

    Returns:
        sn: (H, W) float32 noise std estimate.
    """
    T = len(movie)
    # Load full movie in chunks and accumulate PSD
    # rfft gives T//2+1 frequency bins; select the noise range
    n_fft = T
    freqs = np.fft.rfftfreq(n_fft)
    noise_mask = (freqs >= noise_range[0]) & (freqs <= noise_range[1])

    _, first = next(iter_frames(movie, batch_size=T))
    H, W = first.shape[1], first.shape[2]
    psd_sum = np.zeros((H, W), dtype=np.float64)
    count = np.zeros((H, W), dtype=np.float64)

    # Process full time axis in one go (we need all T frames for the FFT)
    # For very large movies this could be adapted to Welch's method
    all_frames = np.asarray(movie, dtype=np.float32)  # (T, H, W)
    Xf = np.fft.rfft(all_frames, axis=0)             # (T//2+1, H, W)
    psd = (np.abs(Xf[noise_mask]) ** 2) / T * 2       # one-sided PSD

    if method == "mean":
        noise_var = psd.mean(axis=0)
    elif method == "median":
        noise_var = np.median(psd, axis=0)
    elif method == "logmexp":
        log_psd = np.log(psd + 1e-10)
        noise_var = np.exp(log_psd.mean(axis=0))
    else:
        raise ValueError(f"Unknown noise method: {method!r}")

    return np.sqrt(noise_var).astype(np.float32)


def local_correlations_fft(movie: np.ndarray, xp=np) -> np.ndarray:
    """Compute local (8-neighbor) correlation image.

    Each pixel's value is the mean Pearson correlation with its 8 neighbors
    over time. Implemented via FFT-based shift for efficiency.

    Args:
        movie: (T, H, W) float32, with time-mean already subtracted.
        xp: Array module — numpy (default) or cupy for GPU computation.

    Returns:
        cn: (H, W) float32 correlation image (always numpy).
    """
    Y = xp.asarray(movie, dtype=xp.float32)
    T, H, W = Y.shape

    std = Y.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    Y = Y / std  # (T, H, W), unit-std traces

    Yf = xp.fft.fft2(Y, axes=(1, 2))

    cn = xp.zeros((H, W), dtype=xp.float64)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            Nr = xp.fft.ifftshift(xp.arange(-(H // 2), H - H // 2)).reshape(1, -1, 1)
            Nc = xp.fft.ifftshift(xp.arange(-(W // 2), W - W // 2)).reshape(1, 1, -1)
            phase = xp.exp(1j * 2 * xp.pi * (dy * Nr / H + dx * Nc / W))
            Y_shifted = xp.real(xp.fft.ifft2(Yf * phase, axes=(1, 2)))
            cn += (Y * Y_shifted).mean(axis=0)

    cn /= 8.0
    return to_numpy(cn.astype(xp.float32))


def correlation_pnr(
    movie: "zarr.Array | np.ndarray",
    sigma: float | None = None,
    center_psf: bool = True,
    noise_range: tuple[float, float] = (0.25, 0.5),
    n_jobs: int = 1,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute local correlation (CORR) and peak-to-noise ratio (PNR) images.

    These two images together reveal neuron locations:
    - High CORR: pixel is temporally correlated with its neighbors (structured signal).
    - High PNR: pixel's peak fluorescence is large relative to noise.
    Seeds for initialization are detected from the CORR × PNR product.

    Args:
        movie: (T, H, W) array — the motion-corrected movie.
        sigma: Gaussian sigma for center-surround spatial filter. If None,
               no filtering is applied (not recommended for 1p data).
        center_psf: Use center-surround kernel (True) vs plain Gaussian (False).
        noise_range: Frequency range for noise estimation.
        n_jobs: Number of parallel workers for per-frame filtering (-1 = all CPUs).
                Ignored when device='cuda' (GPU handles parallelism internally).
        device: 'cpu' or 'cuda' — where to run filtering and correlation.

    Returns:
        cn: (H, W) local correlation image.
        pnr: (H, W) peak-to-noise ratio image.
    """
    movie = np.asarray(movie, dtype=np.float32)
    xp = get_xp(device)

    # --- Spatial filtering ---
    if sigma is not None:
        if xp is not np:
            # GPU path: use cupyx.scipy.ndimage for convolution
            import cupyx.scipy.ndimage as cp_ndi
            psf = make_center_surround_psf(sigma) if center_psf else None
            movie_xp = xp.asarray(movie)
            if center_psf:
                psf_xp = xp.asarray(psf)
                filtered_xp = xp.stack(
                    [cp_ndi.convolve(frame, psf_xp, mode="reflect") for frame in movie_xp]
                )
            else:
                import cupyx.scipy.ndimage as cp_ndi  # noqa: F811
                filtered_xp = xp.stack(
                    [cp_ndi.gaussian_filter(frame, sigma) for frame in movie_xp]
                )
            filtered = to_numpy(filtered_xp)
        elif center_psf:
            psf = make_center_surround_psf(sigma)
            if n_jobs == 1:
                filtered = np.stack(
                    [ndi.convolve(frame, psf, mode="reflect") for frame in movie], axis=0
                )
            else:
                from joblib import Parallel, delayed
                filtered = np.stack(
                    Parallel(n_jobs=n_jobs)(
                        delayed(ndi.convolve)(frame, psf, mode="reflect")
                        for frame in movie
                    ),
                    axis=0,
                )
        else:
            if n_jobs == 1:
                filtered = np.stack(
                    [ndi.gaussian_filter(frame, sigma) for frame in movie], axis=0
                )
            else:
                from joblib import Parallel, delayed
                filtered = np.stack(
                    Parallel(n_jobs=n_jobs)(
                        delayed(ndi.gaussian_filter)(frame, sigma)
                        for frame in movie
                    ),
                    axis=0,
                )
    else:
        filtered = movie.copy()

    # --- Noise and PNR (always CPU — rfft is fast, result is H×W) ---
    filtered -= filtered.mean(axis=0, keepdims=True)
    sn = estimate_noise(filtered, noise_range=noise_range)

    peak = filtered.max(axis=0)
    pnr = np.where(sn > 0, peak / sn, 0.0).astype(np.float32)
    pnr[pnr < 0] = 0.0

    # --- Local correlation (GPU-aware via xp) ---
    thresh_data = filtered.copy()
    thresh_data[thresh_data < 3 * sn[np.newaxis]] = 0.0
    cn = local_correlations_fft(thresh_data, xp=xp)

    return cn, pnr
