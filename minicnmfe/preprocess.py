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

from minicnmfe._utils import ensure_float32, get_xp, to_numpy

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


def _noise_band(movie, h0, h1, noise_mask, T, method):
    """Noise variance for one row-band -> ``(h1-h0, W)``. Module-level for joblib.

    The per-pixel rfft along time is independent across pixels; ``np.fft.rfft``
    (pocketfft) releases the GIL, so dispatching bands with ``prefer="threads"``
    gives real parallelism even under a global BLAS thread cap.
    """
    block = np.asarray(movie[:, h0:h1, :], dtype=np.float32)  # (T, dh, W)
    Xf = np.fft.rfft(block, axis=0)                  # (T//2+1, dh, W)
    psd = (np.abs(Xf[noise_mask]) ** 2) / T * 2       # one-sided PSD
    if method == "mean":
        return psd.mean(axis=0)
    elif method == "median":
        return np.median(psd, axis=0)
    return np.exp(np.log(psd + 1e-10).mean(axis=0))   # logmexp


def estimate_noise(
    movie: "zarr.Array | np.ndarray",
    noise_range: tuple[float, float] = (0.25, 0.5),
    method: str = "logmexp",
    n_jobs: int = 1,
) -> np.ndarray:
    """Estimate per-pixel noise std from the high-frequency power spectrum.

    For each pixel the power spectral density is estimated via rfft along time.
    Frequencies in `noise_range` * Nyquist are assumed to be pure noise,
    giving a noise floor estimate that is robust to low-frequency calcium signals.

    Args:
        movie: (T, H, W) array.
        noise_range: Fraction of Nyquist to use for noise estimation.
        method: 'mean', 'median', or 'logmexp' (geometric mean via log).
        n_jobs: Parallel workers for the row-band rfft loop (-1 = all CPUs,
            1 = serial). The rfft is pocketfft (single-threaded, never BLAS-
            threaded), so on a many-core box the serial loop runs on ONE core;
            ``n_jobs>1`` dispatches bands across joblib threads (GIL released in
            rfft) for a near-linear speedup. Bit-identical to the serial result.

    Returns:
        sn: (H, W) float32 noise std estimate.
    """
    T, H, W = int(movie.shape[0]), int(movie.shape[1]), int(movie.shape[2])
    # rfft gives T//2+1 frequency bins; select the noise range.
    freqs = np.fft.rfftfreq(T)
    noise_mask = (freqs >= noise_range[0]) & (freqs <= noise_range[1])
    n_freq = int(noise_mask.size)

    if method not in ("mean", "median", "logmexp"):
        raise ValueError(f"Unknown noise method: {method!r}")

    from joblib import effective_n_jobs
    nw = 1 if n_jobs == 1 else max(1, effective_n_jobs(n_jobs))

    # Tile the spatial axis into row-bands and reduce each band to its
    # (H_band, W) noise variance. This bounds peak RAM to the resident complex
    # slabs (n_freq x dh x W x 16 B) instead of the full (T//2+1, H, W)
    # transform (~2x the float32 movie), and produces the identical per-pixel
    # result. Serial: one ~512 MB slab. Parallel: shrink dh so the nw bands held
    # concurrently stay within ~4 GB total.
    if nw == 1:
        band_budget = 512 * 1024 * 1024
    else:
        band_budget = max(64 * 1024 * 1024, (4 * 1024 * 1024 * 1024) // nw)
    dh = max(1, band_budget // (max(n_freq, 1) * max(W, 1) * 16))

    bands = [(h0, min(h0 + dh, H)) for h0 in range(0, H, dh)]
    noise_var = np.empty((H, W), dtype=np.float64)

    if nw == 1 or len(bands) == 1:
        for h0, h1 in bands:
            noise_var[h0:h1] = _noise_band(movie, h0, h1, noise_mask, T, method)
    else:
        from joblib import Parallel, delayed
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1, user_api="blas"):
            results = Parallel(n_jobs=nw, prefer="threads")(
                delayed(_noise_band)(movie, h0, h1, noise_mask, T, method)
                for h0, h1 in bands
            )
        for (h0, h1), res in zip(bands, results):
            noise_var[h0:h1] = res

    return np.sqrt(noise_var).astype(np.float32)


def local_correlations_fft(movie: np.ndarray, xp=np, *, threshold=None) -> np.ndarray:
    """Compute local (8-neighbor) correlation image.

    Each pixel's value is the mean Pearson correlation with its 8 neighbors
    over time. Spatial-domain integer shifts via interior-slice multiplies —
    no FFT, no complex64/128 allocations. Memory ≈ ~2× the input movie: one
    owned entry copy (so the in-place recenter/normalize below never write
    through to the caller) plus the one transient inside ``Y.std``.

    Edge pixels are divided by their actual neighbor count (5 at corners,
    8 in the bulk), not always 8.

    The function self-recenters (subtracts the per-pixel time-mean) before
    dividing by std, matching CaImAn's reference implementation. This makes
    the result a proper Pearson correlation regardless of the input's mean,
    so the output is bounded in [-1, 1]. Callers (e.g. ``correlation_pnr``)
    that pass thresholded inputs no longer need to pre-center.

    Args:
        movie: (T, H, W) float32. No pre-conditions on the time-mean.
        xp: Array module — numpy (default) or cupy for GPU computation.
        threshold: Optional (H, W) / (1, H, W) array broadcast over time. When
            given, pixels below it are zeroed on the owned entry copy *before*
            recentering — equivalent to passing ``np.where(movie < threshold,
            0, movie)`` but without the caller materializing that extra full
            (T, H, W) array. Default None leaves the input untouched.

    Returns:
        cn: (H, W) float32 correlation image (always numpy).
    """
    # One OWNED copy: every op below is in place, so this must not alias the
    # caller's array (xp.asarray would return a contiguous-float32 input
    # unchanged and the in-place writes would corrupt it).
    Y = xp.array(movie, dtype=xp.float32)
    T, H, W = Y.shape

    if threshold is not None:
        # Fused thresholding on the owned copy — replaces a caller-side
        # ``np.where(movie < threshold, 0, movie)`` full (T,H,W) allocation.
        Y[Y < threshold] = 0.0

    # Self-recenter so the formula reduces to Pearson r regardless of caller
    # preprocessing. Without this, thresholded inputs (e.g. the 3*sn step in
    # correlation_pnr) yield biased products that can exceed 1. In place.
    Y -= Y.mean(axis=0, keepdims=True)

    std = Y.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    Y /= std  # (T, H, W), zero-mean unit-std traces (in place)

    cn = xp.zeros((H, W), dtype=xp.float32)
    counts = xp.zeros((H, W), dtype=xp.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            y0, y1 = max(0, dy), H + min(0, dy)
            x0, x1 = max(0, dx), W + min(0, dx)
            ny0, ny1 = max(0, -dy), H + min(0, -dy)
            nx0, nx1 = max(0, -dx), W + min(0, -dx)
            corr = (Y[:, y0:y1, x0:x1] * Y[:, ny0:ny1, nx0:nx1]).mean(axis=0)
            cn[y0:y1, x0:x1] += corr
            counts[y0:y1, x0:x1] += 1.0

    cn /= xp.maximum(counts, 1.0)
    return to_numpy(cn)


def correlation_pnr(
    movie: "zarr.Array | np.ndarray",
    sigma: float | None = None,
    center_psf: bool = True,
    noise_range: tuple[float, float] = (0.25, 0.5),
    n_jobs: int = 1,
    device: str = "cpu",
    stride: int = 1,
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
        stride: Subsample time before computing CORR/PNR. The reductions
            (per-pixel mean / max / neighbour correlation) only need a
            representative slice of frames; ``stride=2-5`` typically halves
            to fifths the wall time with negligible impact on the seed map.
            Default ``1`` keeps the current behaviour (no subsampling).

    Returns:
        cn: (H, W) local correlation image.
        pnr: (H, W) peak-to-noise ratio image.
    """
    movie = np.asarray(movie, dtype=np.float32)
    if stride > 1:
        # Strided view is non-contiguous — materialise contiguously so the
        # downstream per-frame loops and FFTs aren't bottlenecked on cache misses.
        movie = np.ascontiguousarray(movie[::stride])
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
                from threadpoolctl import threadpool_limits
                # Threads: ndi.convolve is pure C and releases the GIL; loky
                # would pickle ~1.4 MB per frame x T_init frames per call.
                # Cap inner BLAS to 1 -- see spatial.py for the rationale.
                with threadpool_limits(limits=1, user_api="blas"):
                    filtered = np.stack(
                        Parallel(n_jobs=n_jobs, prefer="threads")(
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
                from threadpoolctl import threadpool_limits
                # Cap inner BLAS to 1 -- see spatial.py for the rationale.
                with threadpool_limits(limits=1, user_api="blas"):
                    filtered = np.stack(
                        Parallel(n_jobs=n_jobs, prefer="threads")(
                            delayed(ndi.gaussian_filter)(frame, sigma)
                            for frame in movie
                        ),
                        axis=0,
                    )
    else:
        filtered = movie.copy()

    # --- Noise and PNR (always CPU — rfft is fast, result is H×W) ---
    filtered -= filtered.mean(axis=0, keepdims=True)
    sn = estimate_noise(filtered, noise_range=noise_range, n_jobs=n_jobs)

    peak = filtered.max(axis=0)
    pnr = np.where(sn > 0, peak / sn, 0.0).astype(np.float32)
    pnr[pnr < 0] = 0.0

    # --- Local correlation (GPU-aware via xp) ---
    # Threshold ``filtered`` in place. Its pristine values are no longer
    # needed after the PNR step above, so we skip the second ``(T, H, W)``
    # copy this used to do.
    filtered[filtered < 3 * sn[np.newaxis]] = 0.0
    cn = local_correlations_fft(filtered, xp=xp)

    return cn, pnr
