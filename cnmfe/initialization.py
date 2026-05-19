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

import numpy as np
import scipy.ndimage as ndi
import scipy.sparse as sp
from skimage.feature import peak_local_max

from cnmfe._utils import ensure_float32, get_xp, to_numpy
from cnmfe.preprocess import correlation_pnr, make_center_surround_psf
from cnmfe.temporal import deconvolve, estimate_ar_params

if TYPE_CHECKING:
    import zarr


# ---------------------------------------------------------------------------
# Spatial constraint helpers
# ---------------------------------------------------------------------------

def circular_constraint(ai: np.ndarray, max_dist_factor: float = 2.5) -> np.ndarray:
    """Zero pixels far from the component's centre of mass.

    Enforces a roughly circular shape by removing pixels that are more than
    `max_dist_factor * radius` from the centroid, where radius is estimated
    from the component area.
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

    dist = np.sqrt((rows - cy) ** 2 + (cols - cx) ** 2)
    ai[dist > radius * max_dist_factor] = 0.0
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
) -> tuple[sp.csc_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """Find initial neurons using a greedy CORR-PNR strategy.

    Algorithm:
    1. Filter movie spatially with a center-surround PSF.
    2. Compute CORR and PNR summary images.
    3. For each seed (highest CORR×PNR first):
       a. Extract spatial footprint (ai) and temporal trace (ci).
       b. Deconvolve ci with OASIS to get a clean calcium trace.
       c. Subtract ai*ci from both raw and filtered data (in-place).
       d. Update CORR and PNR locally around the subtracted component.
    4. Stop when no seeds remain above threshold or max_neurons reached.

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

    Returns:
        A: (H*W, K) sparse csc_matrix of spatial footprints.
        C: (K, T) denoised calcium traces.
        C_raw: (K, T) raw (un-deconvolved) traces.
        centers: (K, 2) neuron centre coordinates (row, col).
    """
    movie = np.asarray(movie, dtype=np.float32)
    T, H, W = movie.shape
    xp = get_xp(device)

    # Spatial filtering across all frames
    psf = make_center_surround_psf(sigma)
    if xp is not np:
        import cupyx.scipy.ndimage as cp_ndi
        psf_xp = xp.asarray(psf)
        movie_xp = xp.asarray(movie)
        data_filtered = to_numpy(xp.stack(
            [cp_ndi.convolve(frame, psf_xp, mode="reflect") for frame in movie_xp]
        ))
    elif n_jobs == 1:
        data_filtered = np.stack(
            [ndi.convolve(frame, psf, mode="reflect") for frame in movie], axis=0
        )
    else:
        from joblib import Parallel, delayed
        # Threads: ndi.convolve is pure C and releases the GIL. Each frame is
        # ~1.4 MB; with T_init=5000 frames loky would pickle ~7 GB per call.
        data_filtered = np.stack(
            Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(ndi.convolve)(frame, psf, mode="reflect") for frame in movie
            ),
            axis=0,
        )
    data_raw = movie.copy()

    # Summary images. Subsample time when corrpnr_stride > 1 — CORR/PNR are
    # per-pixel reductions, so a strided slice gives a near-identical seed
    # map at a fraction of the cost. The greedy loop's local CORR/PNR
    # updates (after each detection) still run on the full-T patch so
    # extraction stays sharp.
    cn, pnr = correlation_pnr(
        data_filtered, sigma=None, center_psf=False, stride=corrpnr_stride,
    )

    # Border mask
    if border_px > 0:
        cn[:border_px] = 0
        cn[-border_px:] = 0
        cn[:, :border_px] = 0
        cn[:, -border_px:] = 0
        pnr[:border_px] = 0
        pnr[-border_px:] = 0
        pnr[:, :border_px] = 0
        pnr[:, -border_px:] = 0

    patch_radius = max(int(3 * sigma), 5)

    A_cols: list[sp.csc_matrix] = []
    C_list: list[np.ndarray] = []
    C_raw_list: list[np.ndarray] = []
    centers_list: list[tuple[int, int]] = []

    while True:
        if max_neurons is not None and len(A_cols) >= max_neurons:
            break

        seeds = detect_seeds(cn, pnr, min_corr, min_pnr, min_distance=max(1, int(sigma)))
        if len(seeds) == 0:
            break

        found = False
        for seed in seeds:
            row, col = int(seed[0]), int(seed[1])

            ai, ci, success = extract_spatial_temporal(
                data_filtered, data_raw, (row, col), patch_radius,
                min_corr_neuron=min_corr_neuron,
                max_corr_bg=max_corr_bg,
                circular_max_dist_factor=circular_max_dist_factor,
            )

            if not success or (ai > 0).sum() < min_pixel:
                # Mark this seed as used by zeroing score
                cn[row, col] = 0
                pnr[row, col] = 0
                continue

            # Deconvolve temporal trace
            try:
                g, sn = estimate_ar_params(ci, p=ar_order)
                c_clean, s, bl = deconvolve(ci, g, sn)
            except Exception:
                c_clean = ci.copy()

            # Store component
            ai_full = np.zeros(H * W, dtype=np.float32)
            r0 = max(0, row - patch_radius)
            r1 = min(H, row + patch_radius + 1)
            c0 = max(0, col - patch_radius)
            c1 = min(W, col + patch_radius + 1)
            ph = r1 - r0
            pw = c1 - c0
            ai_patch = ai[:ph, :pw]
            for rr in range(ph):
                for cc in range(pw):
                    ai_full[(r0 + rr) * W + (c0 + cc)] = ai_patch[rr, cc]

            ai_sparse = sp.csc_matrix(ai_full.reshape(-1, 1))
            A_cols.append(ai_sparse)
            C_list.append(c_clean)
            C_raw_list.append(ci)
            centers_list.append((row, col))

            # Subtract component from data. Use the raw trace `ci`, NOT the
            # OASIS-deconvolved `c_clean`. OASIS smooths brief transients
            # forward in time (the c[t] >= g*c[t-1] constraint), so subtracting
            # c_clean leaves structured residuals at the spike locations and
            # feeds halo-driven re-seeding. Subtracting ci is the faithful
            # least-squares residual of the per-pixel OLS extraction.
            sub = ai_patch[np.newaxis] * ci[:, np.newaxis, np.newaxis]       # (T, ph, pw)
            data_raw[:, r0:r1, c0:c1] -= sub
            data_filtered[:, r0:r1, c0:c1] -= sub

            # Update CORR/PNR locally around the subtracted region
            update_r0 = max(0, r0 - patch_radius)
            update_r1 = min(H, r1 + patch_radius)
            update_c0 = max(0, c0 - patch_radius)
            update_c1 = min(W, c1 + patch_radius)
            local_cn, local_pnr = correlation_pnr(
                data_filtered[:, update_r0:update_r1, update_c0:update_c1],
                sigma=None,
                center_psf=False,
            )
            cn[update_r0:update_r1, update_c0:update_c1] = local_cn
            pnr[update_r0:update_r1, update_c0:update_c1] = local_pnr

            # After the update, suppress cn/pnr near every found centre so the
            # same neuron cannot be re-detected from a neighbouring pixel.
            # Must cover the neuron's actual support (FWHM ≈ 2*sigma) so the
            # residual halo cannot seed a duplicate just outside the disk.
            suppress_r = max(int(seed_suppress_factor * sigma), int(2 * sigma + 1))
            rr_grid, cc_grid = np.ogrid[:H, :W]
            for (fr, fc) in centers_list:
                mask = (rr_grid - fr) ** 2 + (cc_grid - fc) ** 2 <= suppress_r ** 2
                cn[mask] = 0.0
                pnr[mask] = 0.0

            # Enforce border mask again after update
            if border_px > 0:
                cn[:border_px] = 0
                cn[-border_px:] = 0
                cn[:, :border_px] = 0
                cn[:, -border_px:] = 0
                pnr[:border_px] = 0
                pnr[-border_px:] = 0
                pnr[:, :border_px] = 0
                pnr[:, -border_px:] = 0

            found = True
            break  # Found one neuron; recompute seeds

        if not found:
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
