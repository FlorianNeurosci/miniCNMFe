"""Greedy CORR-PNR initialization for CNMFe.

Finds initial estimates of spatial footprints A and temporal traces C by
iteratively selecting the pixel with the highest correlation × PNR score,
extracting the component there, subtracting it, and repeating.

Reference (algorithmic only):
    CaImAn initialization.py:init_neurons_corr_pnr (line 1380)
    CaImAn initialization.py:extract_ac (line 1841)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import scipy.ndimage as ndi
import scipy.sparse as sp
from skimage.feature import peak_local_max

from minicnmfe._utils import ensure_float32, get_xp, to_numpy
from minicnmfe.preprocess import (
    estimate_noise,
    local_correlations_fft,
    make_center_surround_psf,
)
from minicnmfe.temporal import deconvolve, estimate_ar_params

if TYPE_CHECKING:
    import zarr


# ---------------------------------------------------------------------------
# Spatial constraint helpers
# ---------------------------------------------------------------------------

def circular_constraint(
    ai: np.ndarray,
    max_dist_factor: float = 2.5,
    max_radius: float | None = None,
) -> np.ndarray:
    """Zero pixels far from the component's centre of mass.

    Enforces a roughly circular shape by removing pixels that are more than
    `max_dist_factor * radius` from the centroid, where radius is estimated
    from the component area.

    `max_radius` (optional, in pixels) adds an **absolute** cap on the clip
    distance: the cutoff becomes `min(max_dist_factor * radius, max_radius)`.
    The area-derived radius grows with a sprawled footprint, so on its own the
    constraint barely clips an already-bloated footprint; capping the cutoff at
    a physical neuron radius (e.g. `factor * sigma`) breaks that feedback loop.
    `None` (default) keeps the pure area-derived behaviour (bit-for-bit).
    """
    ai = ai.copy()
    total = ai.sum()
    if total == 0:
        return ai

    rows, cols = np.indices(ai.shape)
    cy = (rows * ai).sum() / total
    cx = (cols * ai).sum() / total
    area = (ai > 0).sum()
    radius = np.sqrt(area / np.pi)

    cutoff = radius * max_dist_factor
    if max_radius is not None:
        cutoff = min(cutoff, float(max_radius))

    dist = np.sqrt((rows - cy) ** 2 + (cols - cx) ** 2)
    ai[dist > cutoff] = 0.0
    return ai


def connectivity_constraint(ai: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component in a spatial footprint."""
    binary = ai > 0
    labeled, n_labels = ndi.label(binary)
    if n_labels <= 1:
        return ai
    # Find the largest label
    sizes = ndi.sum(binary, labeled, range(1, n_labels + 1))
    largest = int(np.argmax(sizes)) + 1
    ai = ai.copy()
    ai[labeled != largest] = 0.0
    return ai


# ---------------------------------------------------------------------------
# Seed detection
# ---------------------------------------------------------------------------

def detect_seeds(
    cn: np.ndarray,
    pnr: np.ndarray,
    min_corr: float,
    min_pnr: float,
    min_distance: int = 5,
) -> np.ndarray:
    """Find candidate seed pixels as local maxima of cn × pnr.

    Args:
        cn: (H, W) local correlation image.
        pnr: (H, W) peak-to-noise ratio image.
        min_corr: Minimum local correlation for a seed.
        min_pnr: Minimum PNR for a seed.
        min_distance: Minimum pixels between two seeds.

    Returns:
        centers: (N, 2) array of (row, col) coordinates, sorted by cn*pnr score descending.
    """
    score = cn * pnr
    score[(cn < min_corr) | (pnr < min_pnr)] = 0.0

    peaks = peak_local_max(score, min_distance=min_distance, threshold_abs=0.0)
    if len(peaks) == 0:
        return np.empty((0, 2), dtype=np.int32)

    # Sort by score descending
    peak_scores = score[peaks[:, 0], peaks[:, 1]]
    order = np.argsort(peak_scores)[::-1]
    return peaks[order].astype(np.int32)


# ---------------------------------------------------------------------------
# Single-component extraction
# ---------------------------------------------------------------------------

def extract_spatial_temporal(
    data_filtered: np.ndarray,
    data_raw: np.ndarray,
    seed_rc: tuple[int, int],
    patch_radius: int,
    min_corr_neuron: float = 0.8,
    max_corr_bg: float = 0.4,
    circular_max_dist_factor: float = 2.5,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Extract spatial footprint and temporal trace for one neuron.

    Given a patch centred on the seed pixel, this function:
    1. Identifies "neuron pixels" (high correlation with seed trace).
    2. Estimates a rough local background from low-correlation pixels.
    3. Solves a 3-component OLS to get the spatial footprint.
    4. Applies shape constraints (circular + connected).

    Returns:
        ai: (patch_h, patch_w) float32 spatial footprint (patch coordinates).
        ci: (T,) float32 temporal trace (baseline-subtracted).
        success: False if the component is degenerate (empty trace or too small).
    """
    H_full, W_full = data_filtered.shape[1:]
    row, col = seed_rc
    r = patch_radius

    r0, r1 = max(0, row - r), min(H_full, row + r + 1)
    c0, c1 = max(0, col - r), min(W_full, col + r + 1)

    patch_f = data_filtered[:, r0:r1, c0:c1]   # (T, ph, pw)
    patch_r = data_raw[:, r0:r1, c0:c1]
    T, ph, pw = patch_f.shape

    patch_f_flat = patch_f.reshape(T, -1)        # (T, n_patch)
    patch_r_flat = patch_r.reshape(T, -1)

    # Seed pixel index within the patch
    seed_in_patch_r = min(row - r0, ph - 1)
    seed_in_patch_c = min(col - c0, pw - 1)
    seed_flat = seed_in_patch_r * pw + seed_in_patch_c

    # Normalise each pixel trace
    centered = patch_f_flat - patch_f_flat.mean(axis=0)
    std = np.sqrt((centered ** 2).sum(axis=0))
    std[std == 0] = 1.0
    normed = centered / std                       # (T, n_patch)

    y0 = normed[:, seed_flat]                     # seed trace, unit norm
    tmp_corr = (y0 @ normed)                      # correlation with seed

    ind_neuron = tmp_corr > min_corr_neuron
    ind_bg = tmp_corr < max_corr_bg

    ci = normed[:, ind_neuron].mean(axis=1) if ind_neuron.any() else y0
    if ci @ ci == 0:
        return np.zeros((ph, pw), np.float32), np.zeros(T, np.float32), False

    y_bg = (
        np.median(patch_r_flat[:, ind_bg], axis=1).reshape(-1, 1)
        if ind_bg.any()
        else np.ones((T, 1), np.float32)
    )

    # 3-component OLS: [ci, y_bg, 1] → pixel intensities → ai
    X = np.column_stack([ci, y_bg[:, 0], np.ones(T, np.float32)])   # (T, 3)
    try:
        coef = np.linalg.lstsq(X, patch_r_flat, rcond=None)[0]      # (3, n_patch)
    except np.linalg.LinAlgError:
        return np.zeros((ph, pw), np.float32), np.zeros(T, np.float32), False

    ai_flat = coef[0].clip(0)                     # spatial footprint (non-negative)
    ai = ai_flat.reshape(ph, pw).astype(np.float32)
    ai = circular_constraint(ai, max_dist_factor=circular_max_dist_factor)
    ai = connectivity_constraint(ai)

    if ai.sum() == 0 or (ai > 0).sum() < 1:
        return ai, np.zeros(T, np.float32), False

    # Baseline-subtract temporal trace
    sn = float(np.sqrt(np.mean(ci ** 2)) * 0.1)  # rough noise estimate
    y_diff = np.concatenate([[-1.0], np.diff(ci)])
    baseline = float(np.median(ci[(y_diff >= 0) & (y_diff < max(sn, 1e-6))]))
    ci = (ci - baseline).astype(np.float32)

    return ai, ci, True


# ---------------------------------------------------------------------------
# Local CN/PNR update against a cached noise map
# ---------------------------------------------------------------------------

def _local_cn_pnr_box(
    data_filtered_box: np.ndarray,
    noise_pixel_box: np.ndarray,
    thresh_init: float = 3.0,
    corrpnr_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Refresh CN and PNR on a small residual patch using a cached noise map.

    Mirrors CaImAn's per-seed local update (init.py:1791-1808). Does NOT
    re-estimate noise (it's already cached globally) and does NOT re-filter the
    patch through the center-surround PSF (we only re-filter `ai` before
    subtracting it from data_filtered, so the residual is already correct).

    Args:
        data_filtered_box: (T, nr2, nc2) already-filtered residual patch.
        noise_pixel_box: (nr2, nc2) slice of the cached global noise map.
        thresh_init: PSD threshold multiplier used in `correlation_pnr`. 3.0 = CaImAn default.

    Returns:
        cn_box: (nr2, nc2) local correlation image.
        pnr_box: (nr2, nc2) peak-to-noise ratio.
    """
    # PNR: peak-to-noise must see the FULL time axis — striding before the max
    # discards transient peaks and (divided by the cached full-T noise) collapses.
    max_box = data_filtered_box.max(axis=0)
    pnr_box = np.where(
        noise_pixel_box > 0, max_box / noise_pixel_box, 0.0
    ).astype(np.float32)
    pnr_box[pnr_box < 0] = 0.0

    # CN: local_correlations_fft is the per-seed hot spot; subsample time for it
    # only (correlation tolerates striding far better than a peak max).
    cn_src = data_filtered_box[::corrpnr_stride] if corrpnr_stride > 1 else data_filtered_box
    thresholded = np.where(
        cn_src < thresh_init * noise_pixel_box, 0.0, cn_src,
    )
    cn_box = local_correlations_fft(thresholded)
    cn_box[np.isnan(cn_box) | (cn_box < 0)] = 0.0
    return cn_box, pnr_box


# ---------------------------------------------------------------------------
# Greedy initialization
# ---------------------------------------------------------------------------

def greedy_corr_pnr(
    movie: "zarr.Array | np.ndarray",
    sigma: float,
    min_corr: float = 0.8,
    min_pnr: float = 10.0,
    max_neurons: int | None = None,
    min_pixel: int = 3,
    border_px: int = 0,
    ar_order: int = 1,
    n_jobs: int = 1,
    device: str = "cpu",
    min_corr_neuron: float = 0.8,
    max_corr_bg: float = 0.4,
    seed_suppress_factor: float = 2.0,
    circular_max_dist_factor: float = 2.5,
    corrpnr_stride: int = 1,
    g_prior: float | None = None,
    g_prior_weight: float = 0.5,
) -> tuple[sp.csc_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """Find initial neurons using a greedy CORR-PNR strategy.

    Algorithm (CaImAn `init_neurons_corr_pnr` parity, init.py:1381-1825):
    1. Filter movie spatially with a center-surround PSF (cv2.filter2D).
    2. Mean-center each pixel and estimate per-pixel noise ONCE globally.
    3. Compute initial CN and PNR; build `ind_search` (persistent bitmask of
       pixels already tried / inside an accepted neuron's support).
    4. For each seed (highest CN*PNR first, skipping `ind_search` pixels):
       a. Extract spatial footprint (ai) and temporal trace (ci) by OLS.
       b. Deconvolve ci with OASIS to get a clean calcium trace.
       c. Subtract ai*ci from data_raw on the gSiz patch.
       d. Re-filter ai through the PSF and subtract ai_filtered*ci from
          data_filtered on the 2*gSiz halo (so the filtered residual stays
          clean and no halo ghosts seed subsequent iterations).
       e. Mark `ai > ai.max()/2` pixels in `ind_search`.
       f. Refresh CN/PNR locally against the cached noise map — no FFT,
          no PSF re-filter of the whole patch.
    5. Rebuild the sorted seed list when the current pass is exhausted; stop
       when no progress or max_neurons reached.

    Args:
        movie: (T, H, W) motion-corrected movie.
        sigma: Gaussian sigma for center-surround spatial filter (pixels).
        min_corr: Minimum local correlation for a seed pixel.
        min_pnr: Minimum PNR for a seed pixel.
        max_neurons: Stop after finding this many neurons (None = no limit).
        min_pixel: Minimum number of nonzero pixels in a valid footprint.
        border_px: Ignore seeds within this many pixels of the image border.
        ar_order: AR model order for deconvolution (1 or 2).
        n_jobs: Number of parallel workers for the initial PSF filtering pass
                (-1 = all CPUs, 1 = serial). Ignored when device='cuda'.
        device: 'cpu' or 'cuda'. GPU accelerates the initial PSF convolution
                over all frames. The greedy extraction loop is always sequential.
        seed_suppress_factor: DEPRECATED. The old circular-disk suppression has
                been replaced by CaImAn-style `ai > ai.max()/2` support-mask
                suppression via `ind_search`. The parameter is kept for API
                stability and is no longer consulted.

    Returns:
        A: (H*W, K) sparse csc_matrix of spatial footprints.
        C: (K, T) denoised calcium traces.
        C_raw: (K, T) raw (un-deconvolved) traces.
        centers: (K, 2) neuron centre coordinates (row, col).
    """
    movie = np.asarray(movie, dtype=np.float32)
    T, H, W = movie.shape
    xp = get_xp(device)

    # ----- Initial filtering pass -----
    # CaImAn (init.py:1473) uses cv2.filter2D for the per-frame center-surround
    # filter. cv2 is 5-10x faster than scipy.ndimage.convolve here because the
    # kernel is small (~6*sigma+1 on each side) and cv2's filter2D is heavily
    # SIMD-optimised. The PSF is rotationally symmetric so convolution ==
    # correlation, which is what cv2.filter2D applies by default.
    psf = make_center_surround_psf(sigma)
    if xp is not np:
        import cupyx.scipy.ndimage as cp_ndi
        psf_xp = xp.asarray(psf)
        movie_xp = xp.asarray(movie)
        data_filtered = to_numpy(xp.stack(
            [cp_ndi.convolve(frame, psf_xp, mode="reflect") for frame in movie_xp]
        ))
    elif n_jobs == 1:
        data_filtered = np.empty_like(movie)
        for t in range(T):
            data_filtered[t] = cv2.filter2D(
                movie[t], -1, psf, borderType=cv2.BORDER_REFLECT,
            )
    else:
        from joblib import Parallel, delayed
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1, user_api="blas"):
            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(cv2.filter2D)(
                    frame, -1, psf, borderType=cv2.BORDER_REFLECT,
                )
                for frame in movie
            )
        data_filtered = np.stack(results, axis=0).astype(np.float32)
    data_raw = movie.copy()

    # ----- One-shot global noise + CN/PNR (CaImAn init.py:1480-1489) -----
    # Mean-center each pixel's time series; this is the global counterpart
    # of `data_filtered -= data_filtered.mean(axis=0)` in CaImAn.
    data_filtered -= data_filtered.mean(axis=0, keepdims=True)
    noise_pixel = estimate_noise(data_filtered, noise_range=(0.25, 0.5))

    # Initial CN/PNR. PNR is a peak/max statistic over the cached full-T
    # `noise_pixel`, so its max MUST see the full time axis — striding it
    # discards transient peaks and collapses PNR (strided-max / full-noise is a
    # systematic under-estimate), starving seeding to ~0 on long movies.
    # corrpnr_stride > 1 therefore subsamples time for the CN only (the cost
    # driver: local_correlations_fft), never for PNR.
    thresh_init = 3.0
    pnr = np.where(
        noise_pixel > 0, data_filtered.max(axis=0) / noise_pixel, 0.0
    ).astype(np.float32)
    if corrpnr_stride > 1:
        df_stride = np.ascontiguousarray(data_filtered[::corrpnr_stride])
        sn_stride = estimate_noise(df_stride, noise_range=(0.25, 0.5))
        tmp = np.where(df_stride < thresh_init * sn_stride, 0.0, df_stride)
        cn = local_correlations_fft(tmp)
        del tmp, df_stride
    else:
        tmp = np.where(data_filtered < thresh_init * noise_pixel, 0.0, data_filtered)
        cn = local_correlations_fft(tmp)
        del tmp
    pnr[pnr < 0] = 0.0
    cn[np.isnan(cn) | (cn < 0)] = 0.0

    patch_radius = max(int(3 * sigma), 5)
    gSiz = patch_radius

    # ----- CaImAn-style search state -----
    v_search = (cn * pnr).astype(np.float32)
    v_search[(cn < min_corr) | (pnr < min_pnr)] = 0.0
    # ind_search: persistent (H, W) bitmask of pixels that won't be tried as
    # seeds. Initialised from sub-threshold v_search and the border mask;
    # accepted neurons add their `ai > ai.max()/2` support each iteration
    # (CaImAn init.py:1763), and every tried seed is marked too (CaImAn :1660).
    ind_search = (v_search <= 0)
    if border_px > 0:
        ind_search[:border_px] = True
        ind_search[-border_px:] = True
        ind_search[:, :border_px] = True
        ind_search[:, -border_px:] = True

    min_v_search = float(min_corr) * float(min_pnr)

    A_cols: list[sp.csc_matrix] = []
    C_list: list[np.ndarray] = []
    C_raw_list: list[np.ndarray] = []
    centers_list: list[tuple[int, int]] = []

    while True:
        if max_neurons is not None and len(A_cols) >= max_neurons:
            break

        # Rebuild the sorted candidate list once per outer pass. With
        # peak_local_max we want a fresh scan after multiple accepts have
        # reshaped v_search; we don't break-and-rebuild per accept.
        score = v_search.copy()
        score[ind_search | (cn < min_corr) | (pnr < min_pnr)] = 0.0
        peaks = peak_local_max(
            score,
            min_distance=max(1, int(sigma)),
            threshold_abs=min_v_search,
        )
        if len(peaks) == 0:
            break
        order = np.argsort(score[peaks[:, 0], peaks[:, 1]])[::-1]
        seeds_sorted = peaks[order]

        progress = False
        for seed in seeds_sorted:
            row, col = int(seed[0]), int(seed[1])
            if ind_search[row, col] or v_search[row, col] < min_v_search:
                continue
            ind_search[row, col] = True  # mark this pixel as tried (CaImAn :1660)

            # Diff-noise guard (CaImAn init.py:1670-73): reject pure-noise pixels
            y0_diff = np.diff(data_filtered[:, row, col])
            if y0_diff.size and y0_diff.max() < 3.0 * y0_diff.std():
                v_search[row, col] = 0.0
                continue

            r0 = max(0, row - gSiz)
            r1 = min(H, row + gSiz + 1)
            c0 = max(0, col - gSiz)
            c1 = min(W, col + gSiz + 1)

            ai, ci, success = extract_spatial_temporal(
                data_filtered, data_raw, (row, col), gSiz,
                min_corr_neuron=min_corr_neuron,
                max_corr_bg=max_corr_bg,
                circular_max_dist_factor=circular_max_dist_factor,
            )

            if not success or (ai > 0).sum() < min_pixel:
                continue

            ph = r1 - r0
            pw = c1 - c0
            ai_patch = ai[:ph, :pw]

            # Deconvolve temporal trace
            try:
                g, sn_ar = estimate_ar_params(
                    ci, p=ar_order,
                    g_prior=g_prior, g_prior_weight=g_prior_weight,
                )
                c_clean, s, bl = deconvolve(ci, g, sn_ar)
            except Exception:
                c_clean = ci.copy()

            # Store component. Slice assignment instead of the old double for-loop.
            ai_full_2d = np.zeros((H, W), dtype=np.float32)
            ai_full_2d[r0:r1, c0:c1] = ai_patch
            A_cols.append(sp.csc_matrix(ai_full_2d.reshape(-1, 1)))
            C_list.append(c_clean)
            C_raw_list.append(ci)
            centers_list.append((row, col))

            # ----- Subtraction -----
            # data_raw: subtract on the gSiz extraction box.
            data_raw[:, r0:r1, c0:c1] -= (
                ai_patch[np.newaxis] * c_clean[:, None, None]
            )

            # data_filtered: subtract ai_filtered (NOT ai) on the 2*gSiz halo
            # so the spatially-filtered residual stays clean. The old code
            # subtracted unfiltered ai here, which left a halo at the soma's
            # PSF sidelobes -- ghost seeds in subsequent iterations.
            r2_0 = max(0, row - 2 * gSiz)
            r2_1 = min(H, row + 2 * gSiz + 1)
            c2_0 = max(0, col - 2 * gSiz)
            c2_1 = min(W, col + 2 * gSiz + 1)
            ai_box_full = np.zeros((r2_1 - r2_0, c2_1 - c2_0), dtype=np.float32)
            ai_box_full[r0 - r2_0:r1 - r2_0, c0 - c2_0:c1 - c2_0] = ai_patch
            ai_filtered = cv2.filter2D(
                ai_box_full, -1, psf, borderType=cv2.BORDER_REFLECT,
            ).astype(np.float32)
            data_filtered[:, r2_0:r2_1, c2_0:c2_1] -= (
                ai_filtered[np.newaxis] * c_clean[:, None, None]
            )

            # ----- Suppression: support mask only (pure CaImAn parity) -----
            # `seed_suppress_factor` is no longer consulted; it remains on the
            # signature for API stability.
            if ai_patch.max() > 0:
                support = ai_patch > (ai_patch.max() / 2.0)
                ind_search[r0:r1, c0:c1] |= support

            # ----- Local CN/PNR update against the cached noise -----
            # Subsample time by corrpnr_stride, MATCHING the global CORR/PNR pass
            # (lines ~353): the initial cn/pnr that seed the search were computed
            # on strided frames, so the per-seed refresh should use the same
            # frames for consistency — and it's the per-seed hot spot
            # (local_correlations_fft ≈ 40% of greedy), so striding cuts it
            # ~corrpnr_stride×. No-op when corrpnr_stride == 1.
            box = data_filtered[:, r2_0:r2_1, c2_0:c2_1]
            cn_box, pnr_box = _local_cn_pnr_box(
                box, noise_pixel[r2_0:r2_1, c2_0:c2_1],
                thresh_init=thresh_init, corrpnr_stride=corrpnr_stride,
            )
            cn[r2_0:r2_1, c2_0:c2_1] = cn_box
            pnr[r2_0:r2_1, c2_0:c2_1] = pnr_box
            v_search[r2_0:r2_1, c2_0:c2_1] = cn_box * pnr_box
            v_search[ind_search] = 0.0

            progress = True
            if max_neurons is not None and len(A_cols) >= max_neurons:
                break

        if not progress:
            break

    if not A_cols:
        # Return empty results
        return (
            sp.csc_matrix((H * W, 0), dtype=np.float32),
            np.empty((0, T), dtype=np.float32),
            np.empty((0, T), dtype=np.float32),
            np.empty((0, 2), dtype=np.int32),
        )

    A = sp.hstack(A_cols, format="csc")
    C = np.vstack(C_list)
    C_raw = np.vstack(C_raw_list)
    centers = np.array(centers_list, dtype=np.int32)
    return A, C, C_raw, centers


# ---------------------------------------------------------------------------
# [NON-STANDARD speed] Patch-based parallel initialization
#
# The greedy loop above is inherently sequential (each extraction mutates the
# residual the next seed reads), so it can't be threaded directly. Instead we
# tile the FOV into OVERLAPPING spatial patches, run the proven `greedy_corr_pnr`
# on each patch in parallel *processes* (the loop is GIL-bound — the one place
# we deviate from the `prefer="threads"` convention), remap footprints to global
# coordinates, and de-duplicate neurons detected in patch overlaps via
# `merge_components`. Opt-in through CNMFeParams.init_patches.
# ---------------------------------------------------------------------------

def _tile_grid(
    H: int, W: int, patch_size: int, patch_overlap: int
) -> list[tuple[int, int, int, int]]:
    """Overlapping patch grid covering (H, W). Returns (r0, r1, c0, c1) tiles.

    Adjacent patches step by ``patch_size - patch_overlap``; the last patch on
    each axis is clamped to the edge so the whole FOV is covered.
    """
    step = max(1, patch_size - patch_overlap)

    def axis_starts(length: int) -> list[int]:
        if length <= patch_size:
            return [0]
        starts = list(range(0, length - patch_size + 1, step))
        if starts[-1] != length - patch_size:
            starts.append(length - patch_size)
        return starts

    tiles: list[tuple[int, int, int, int]] = []
    for r0 in axis_starts(H):
        r1 = min(r0 + patch_size, H)
        for c0 in axis_starts(W):
            c1 = min(c0 + patch_size, W)
            tiles.append((r0, r1, c0, c1))
    return tiles


def _greedy_patch_worker(
    patch_movie: np.ndarray,
    row_off: int,
    col_off: int,
    H: int,
    W: int,
    sigma: float,
    min_corr: float,
    min_pnr: float,
    min_pixel: int,
    ar_order: int,
    min_corr_neuron: float,
    max_corr_bg: float,
    seed_suppress_factor: float,
    circular_max_dist_factor: float,
    corrpnr_stride: int,
    g_prior: "float | None",
    g_prior_weight: float,
) -> "tuple[sp.csc_matrix, np.ndarray, np.ndarray, np.ndarray]":
    """Run `greedy_corr_pnr` on ONE patch; remap footprints/centres to global.

    Module-level for loky pickling. Inner parallelism is forced off
    (``n_jobs=1, device="cpu"``) — patch-level parallelism is the outer layer.
    ``border_px=0`` because patch borders are interior FOV (edge rejection is
    applied globally by the driver). ``max_neurons=None`` — the cap is global.
    """
    from threadpoolctl import threadpool_limits

    ph, pw = patch_movie.shape[1], patch_movie.shape[2]
    # The parent's threadpool_limits context does NOT cross the process
    # boundary under loky, so cap inner BLAS here too.
    with threadpool_limits(limits=1, user_api="blas"):
        A_p, C_p, C_raw_p, centers_p = greedy_corr_pnr(
            patch_movie,
            sigma=sigma,
            min_corr=min_corr,
            min_pnr=min_pnr,
            max_neurons=None,
            min_pixel=min_pixel,
            border_px=0,
            ar_order=ar_order,
            n_jobs=1,
            device="cpu",
            min_corr_neuron=min_corr_neuron,
            max_corr_bg=max_corr_bg,
            seed_suppress_factor=seed_suppress_factor,
            circular_max_dist_factor=circular_max_dist_factor,
            corrpnr_stride=corrpnr_stride,
            g_prior=g_prior,
            g_prior_weight=g_prior_weight,
        )

    k = A_p.shape[1]
    if k == 0:
        T_init = patch_movie.shape[0]
        return (
            sp.csc_matrix((H * W, 0), dtype=np.float32),
            np.empty((0, T_init), dtype=np.float32),
            np.empty((0, T_init), dtype=np.float32),
            np.empty((0, 2), dtype=np.int32),
        )

    # Remap patch-local pixel indices (lr*pw + lc) to global flat indices.
    lr, lc = np.divmod(np.arange(ph * pw), pw)
    gidx = (row_off + lr) * W + (col_off + lc)
    coo = A_p.tocoo()
    A_global = sp.csc_matrix(
        (coo.data, (gidx[coo.row], coo.col)), shape=(H * W, k), dtype=np.float32
    )
    centers_global = (centers_p + np.array([row_off, col_off])).astype(np.int32)
    return A_global, C_p, C_raw_p, centers_global


def _empty_init(H: int, W: int, T: int):
    return (
        sp.csc_matrix((H * W, 0), dtype=np.float32),
        np.empty((0, T), dtype=np.float32),
        np.empty((0, T), dtype=np.float32),
        np.empty((0, 2), dtype=np.int32),
    )


def greedy_corr_pnr_patched(
    movie: np.ndarray,
    sigma: float,
    min_corr: float = 0.8,
    min_pnr: float = 10.0,
    max_neurons: "int | None" = None,
    min_pixel: int = 3,
    border_px: int = 0,
    ar_order: int = 1,
    min_corr_neuron: float = 0.8,
    max_corr_bg: float = 0.4,
    seed_suppress_factor: float = 2.0,
    circular_max_dist_factor: float = 2.5,
    corrpnr_stride: int = 1,
    g_prior: "float | None" = None,
    g_prior_weight: float = 0.5,
    patch_size: int = 64,
    patch_overlap: int = 16,
    n_jobs: int = 1,
    merge_thr_corr: float = 0.85,
    merge_thr_overlap: float = 0.5,
    merge_centre_dist_factor: float = 2.0,
) -> "tuple[sp.csc_matrix, np.ndarray, np.ndarray, np.ndarray]":
    """Patch-parallel greedy init. Same return contract as `greedy_corr_pnr`.

    Tiles the in-RAM ``(T, H, W)`` movie into overlapping patches, runs
    `greedy_corr_pnr` per patch in parallel processes, concatenates the
    global-remapped components, and merges border duplicates. Peak extra RAM is
    ``≈ n_jobs × T × patch_size² × 4`` bytes (per-worker patch copies).
    """
    from joblib import Parallel, delayed

    from minicnmfe.merging import merge_components

    movie = np.asarray(movie, dtype=np.float32)
    T, H, W = movie.shape
    tiles = _tile_grid(H, W, patch_size, patch_overlap)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_greedy_patch_worker)(
            np.ascontiguousarray(movie[:, r0:r1, c0:c1]),
            r0, c0, H, W,
            sigma, min_corr, min_pnr, min_pixel, ar_order,
            min_corr_neuron, max_corr_bg, seed_suppress_factor,
            circular_max_dist_factor, corrpnr_stride, g_prior, g_prior_weight,
        )
        for (r0, r1, c0, c1) in tiles
    )

    results = [r for r in results if r[0].shape[1] > 0]
    if not results:
        return _empty_init(H, W, T)

    A_all = sp.hstack([r[0] for r in results], format="csc")
    C_all = np.vstack([r[1] for r in results]).astype(np.float32)
    C_raw_all = np.vstack([r[2] for r in results]).astype(np.float32)
    centers_all = np.vstack([r[3] for r in results]).astype(np.int32)

    # Reject detections within border_px of the global FOV edge (each patch ran
    # with border_px=0 since patch borders are interior, not image edges).
    if border_px > 0:
        keep = (
            (centers_all[:, 0] >= border_px)
            & (centers_all[:, 0] < H - border_px)
            & (centers_all[:, 1] >= border_px)
            & (centers_all[:, 1] < W - border_px)
        )
        if not keep.any():
            return _empty_init(H, W, T)
        A_all = A_all[:, keep].tocsc()
        C_all = C_all[keep]
        C_raw_all = C_raw_all[keep]
        centers_all = centers_all[keep]

    # De-duplicate the same neuron detected in two overlapping patches: its two
    # copies have near-identical traces (high |Pearson|) and centres within a
    # few px — exactly the centre-distance fallback case in merge_components.
    A_dedup, C_dedup, _, members = merge_components(
        A_all, C_all,
        thr_corr=merge_thr_corr,
        thr_overlap=merge_thr_overlap,
        ar_order=ar_order,
        sigma=sigma,
        dims=(H, W),
        centre_dist_factor=merge_centre_dist_factor,
    )
    # Keep C_raw / centres aligned with the merged order (mirror pipeline.fit).
    C_raw_dedup = np.vstack(
        [C_raw_all[m].mean(axis=0).clip(min=0) for m in members]
    ).astype(np.float32)
    centers_dedup = np.vstack([centers_all[m[0]] for m in members]).astype(np.int32)

    # Global max_neurons cap (top-K by footprint energy ‖a‖²), post-dedup.
    if max_neurons is not None and A_dedup.shape[1] > max_neurons:
        energy = np.asarray(A_dedup.power(2).sum(axis=0)).ravel()
        top = np.sort(np.argsort(energy)[::-1][:max_neurons])
        A_dedup = A_dedup[:, top].tocsc()
        C_dedup = C_dedup[top]
        C_raw_dedup = C_raw_dedup[top]
        centers_dedup = centers_dedup[top]

    return A_dedup, C_dedup, C_raw_dedup, centers_dedup
