"""Spatial footprint update via per-pixel non-negative LASSO regression.

For each pixel p, the data is modelled as:
    Y[p, :] = sum_{k in active(p)} A[p, k] * C[k, :] + noise

where active(p) is the set of components whose footprints overlap pixel p
(within a dilation of the current footprint). This small local set makes the
regression fast and well-conditioned.

Reference (algorithmic only): CaImAn spatial.py:update_spatial_components (line 29).
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi
import scipy.sparse as sp
from sklearn.linear_model import LassoLars


# ---------------------------------------------------------------------------
# Module-level worker (must be importable for multiprocessing pickling)
# ---------------------------------------------------------------------------

def _spatial_pixel_batch(
    pixel_start: int,
    Y_batch: np.ndarray,
    C: np.ndarray,
    support_batch: list,
    sn_batch: np.ndarray,
    T: int,
) -> list[tuple[int, int, float]]:
    """Process one contiguous batch of pixels.

    Args:
        pixel_start: Global index of the first pixel in this batch.
        Y_batch: (batch_size, T) — temporal traces for this batch only.
        C: (K, T) — all component temporal traces.
        support_batch: List of int arrays; support_batch[i] = active components
                       for global pixel pixel_start + i.
        sn_batch: (batch_size,) noise std for this batch.
        T: Number of time points.

    Returns:
        List of (global_pixel_idx, component_idx, coefficient_value) tuples
        with coefficient_value > 0.
    """
    results: list[tuple[int, int, float]] = []
    for i, active in enumerate(support_batch):
        if len(active) == 0:
            continue
        C_active = C[active]              # (n_active, T)
        y_p = Y_batch[i]                  # (T,)
        sn_p = float(sn_batch[i])

        gram = C_active @ C_active.T
        max_energy = float(np.max(np.diag(gram))) if gram.size > 0 else 1.0
        lam = 0.5 * sn_p * np.sqrt(max(max_energy, 1e-10)) / T
        if lam == 0:
            continue

        try:
            reg = LassoLars(alpha=lam, positive=True, fit_intercept=False, max_iter=200)
            reg.fit(C_active.T, y_p)
            coef = reg.coef_
        except Exception:
            coef = np.zeros(len(active), dtype=np.float32)

        p_global = pixel_start + i
        for idx, k in enumerate(active):
            val = float(coef[idx])
            if val > 0:
                results.append((p_global, int(k), val))
    return results


# ---------------------------------------------------------------------------
# Support computation
# ---------------------------------------------------------------------------

def compute_support(
    A: sp.csc_matrix,
    dims: tuple[int, int],
    dilation_radius: int = 3,
) -> list[np.ndarray]:
    """For each pixel, find which components could contribute to it.

    Each component's binary footprint is dilated by `dilation_radius` pixels.
    Pixel p is in the support of component k if p falls inside the dilated
    footprint of k.

    Returns:
        support: List of length H*W. support[p] is an int array of component
                 indices that are active near pixel p.
    """
    H, W = dims
    n_pixels = H * W
    K = A.shape[1]

    struct = ndi.generate_binary_structure(2, 1)
    struct = ndi.iterate_structure(struct, dilation_radius)

    support: list[list[int]] = [[] for _ in range(n_pixels)]

    for k in range(K):
        col = np.asarray(A.getcol(k).todense()).ravel()
        footprint_2d = col.reshape(H, W)
        binary = footprint_2d > 0
        dilated = ndi.binary_dilation(binary, structure=struct)
        active_pixels = np.where(dilated.ravel())[0]
        for p in active_pixels:
            support[p].append(k)

    return [np.array(s, dtype=np.int32) for s in support]


def threshold_footprint(
    ai: np.ndarray,
    dims: tuple[int, int],
    max_thr: float = 0.1,
) -> np.ndarray:
    """Post-process a single spatial footprint to enforce compactness.

    Steps:
    1. Reshape to 2D and apply a 3×3 median filter (removes isolated pixels).
    2. Zero pixels below max_thr * max(ai).
    3. Keep only the largest connected component.

    Args:
        ai: (H*W,) flat footprint.
        dims: (H, W) spatial dimensions.
        max_thr: Fraction of maximum value below which pixels are zeroed.

    Returns:
        Cleaned footprint, same shape as ai.
    """
    H, W = dims
    ai2d = ai.reshape(H, W).copy()

    ai2d = ndi.median_filter(ai2d, size=3)
    ai2d = ai2d.clip(0)

    if ai2d.max() == 0:
        return np.zeros(H * W, dtype=np.float32)

    ai2d[ai2d < max_thr * ai2d.max()] = 0.0

    labeled, n = ndi.label(ai2d > 0)
    if n > 1:
        sizes = ndi.sum(ai2d > 0, labeled, range(1, n + 1))
        largest = int(np.argmax(sizes)) + 1
        ai2d[labeled != largest] = 0.0

    return ai2d.ravel().astype(np.float32)


# ---------------------------------------------------------------------------
# Main update
# ---------------------------------------------------------------------------

def update_spatial(
    Y_flat: np.ndarray,
    C: np.ndarray,
    A: sp.csc_matrix,
    sn: np.ndarray,
    dims: tuple[int, int],
    dilation_radius: int = 3,
    n_jobs: int = 1,
    max_thr: float = 0.1,
) -> sp.csc_matrix:
    """Refine spatial footprints by per-pixel non-negative LASSO regression.

    For each pixel p:
        active = compute_support[p]   # components near this pixel
        lambda_p = 0.5 * sn[p] * sqrt(max(diag(C_active @ C_active.T))) / T
        a[active] = LassoLars(alpha=lambda_p, positive=True).fit(
                        C[active].T,   # (T, n_active) — regressors
                        Y_flat[p],     # (T,) — target
                    ).coef_

    After all pixels are processed, each component's footprint is cleaned
    with threshold_footprint().

    Args:
        Y_flat: (H*W, T) background-subtracted movie.
        C: (K, T) temporal traces.
        A: (H*W, K) current spatial footprints (initial support guess).
        sn: (H*W,) per-pixel noise std.
        dims: (H, W) spatial dimensions.
        dilation_radius: Footprint dilation for support computation.
        n_jobs: Number of parallel workers (-1 = all CPUs, 1 = serial).
        max_thr: Pixels below max_thr * max(ai) are zeroed after LASSO.

    Returns:
        A_new: Updated sparse (H*W, K) footprints.
    """
    H, W = dims
    n_pixels, T = Y_flat.shape
    K = A.shape[1]

    support = compute_support(A, dims, dilation_radius)

    # Build list of (start, end) batch ranges
    batch_size = 256
    batches = [
        (s, min(s + batch_size, n_pixels))
        for s in range(0, n_pixels, batch_size)
    ]

    if n_jobs == 1:
        # Serial path — avoids joblib import overhead
        all_results: list[tuple[int, int, float]] = []
        for start, end in batches:
            all_results.extend(
                _spatial_pixel_batch(
                    start,
                    Y_flat[start:end],
                    C,
                    support[start:end],
                    sn[start:end],
                    T,
                )
            )
    else:
        from joblib import Parallel, delayed

        batch_lists = Parallel(n_jobs=n_jobs)(
            delayed(_spatial_pixel_batch)(
                start,
                Y_flat[start:end],
                C,
                support[start:end],
                sn[start:end],
                T,
            )
            for start, end in batches
        )
        all_results = [item for batch in batch_lists for item in batch]

    # Accumulate per-component pixel values
    new_data: dict[int, dict[int, float]] = {k: {} for k in range(K)}
    for p_global, k, val in all_results:
        new_data[k][p_global] = val

    # Build new sparse matrix and clean each component's footprint
    rows_all: list[np.ndarray] = []
    cols_all: list[np.ndarray] = []
    data_all: list[np.ndarray] = []

    for k in range(K):
        if not new_data[k]:
            continue
        pixel_ids = np.array(list(new_data[k].keys()), dtype=np.int32)
        values = np.array(list(new_data[k].values()), dtype=np.float32)

        ai_flat = np.zeros(n_pixels, dtype=np.float32)
        ai_flat[pixel_ids] = values
        ai_flat = threshold_footprint(ai_flat, dims, max_thr=max_thr)

        nz = np.where(ai_flat > 0)[0]
        if len(nz) == 0:
            continue

        rows_all.append(nz)
        cols_all.append(np.full(len(nz), k, dtype=np.int32))
        data_all.append(ai_flat[nz])

    if rows_all:
        rows = np.concatenate(rows_all)
        cols = np.concatenate(cols_all)
        data = np.concatenate(data_all)
        A_new = sp.csc_matrix(
            (data, (rows, cols)),
            shape=(n_pixels, K),
            dtype=np.float32,
        )
    else:
        A_new = sp.csc_matrix((n_pixels, K), dtype=np.float32)

    return A_new
