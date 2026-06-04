"""Component merging for CNMFe.

After each round of spatial/temporal updates, pairs of components that are
both spatially overlapping (high Jaccard) and temporally correlated (high
Pearson r) are merged into a single component.

Reference (algorithmic only): CaImAn merging.py:merge_components (line 19).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from minicnmfe.spatial import threshold_footprint


def merge_components(
    A: sp.csc_matrix,
    C: np.ndarray,
    thr_corr: float = 0.85,
    thr_overlap: float = 0.5,
    ar_order: int = 1,
    sigma: float | None = None,
    dims: tuple[int, int] | None = None,
    centre_dist_factor: float = 2.0,
    max_thr: float = 0.1,
) -> tuple[sp.csc_matrix, np.ndarray, int, list[np.ndarray]]:
    """Merge spatially overlapping and temporally correlated components.

    Two components i, j are merged if their traces are correlated AND they
    either share spatial support OR have centres of mass within a few pixels:

        |Pearson(C[i], C[j])| > thr_corr  AND
        ( Jaccard(i, j) > thr_overlap  OR  centre_dist(i, j) < centre_dist_factor * sigma )

    The centre-distance fallback catches duplicate detections of the same
    neuron whose footprints have been thresholded at different peak pixels —
    their Jaccard can be near zero even though they refer to the same neuron.

    Merged spatial footprint: sum of individual footprints, then re-thresholded
    via threshold_footprint() to keep the fused blob compact.
    Merged temporal trace: mean of individual traces (clipped non-negative).
    Re-deconvolution is intentionally deferred to the caller's next
    update_temporal pass, which uses the persistent per-component AR cache —
    re-estimating g here would re-introduce the fudge_factor drift.

    Args:
        A: (H*W, K) sparse spatial footprints.
        C: (K, T) temporal traces.
        thr_corr: Minimum Pearson correlation between traces to trigger merge.
        thr_overlap: Minimum Jaccard overlap between footprints.
        ar_order: AR model order (kept for API compat — no longer used internally).
        sigma: Neuron radius (px). Required to enable the centre-distance
               fallback; if None or dims is None, only Jaccard is used.
        dims: (H, W) image dimensions; required for centre-of-mass and for
              re-thresholding merged footprints.
        centre_dist_factor: Centre-distance threshold = factor * sigma.
        max_thr: Threshold passed to threshold_footprint for merged blobs.

    Returns:
        A_merged: (H*W, K_new) sparse footprints.
        C_merged: (K_new, T) calcium traces.
        n_merged: Number of merge events (groups merged > 1 component).
        members_per_group: list of length K_new; members_per_group[j] is an
            int array of original component indices that fused into output j.
            Singletons have len 1; merges have len > 1. Lets the caller update
            any per-component cache (e.g. AR coefficients).
    """
    del ar_order  # unused — re-deconvolution deferred to update_temporal
    K, T = C.shape
    if K <= 1:
        members_per_group = [np.array([k], dtype=np.int32) for k in range(K)]
        return A, C, 0, members_per_group

    # --- Spatial overlap (Jaccard) ---
    O = (A.T @ A).toarray()              # (K, K) — dot product of footprints
    nA = np.maximum(np.diag(O), 1e-10)  # (K,) — squared norm of each footprint
    # Jaccard[i,j] = O[i,j] / (nA[i] + nA[j] - O[i,j])
    denom = nA[:, np.newaxis] + nA[np.newaxis, :] - O
    denom = np.maximum(denom, 1e-10)
    jaccard = O / denom                  # (K, K)
    np.fill_diagonal(jaccard, 0.0)

    # --- Centre-of-mass distance (fallback for duplicates with disjoint supports) ---
    centre_close = np.zeros_like(jaccard, dtype=bool)
    if sigma is not None and dims is not None:
        H, W = dims
        rows = np.arange(H * W) // W
        cols = np.arange(H * W) % W
        col_sums = np.asarray(A.sum(axis=0)).ravel()
        col_sums = np.maximum(col_sums, 1e-10)
        weighted_rows = np.asarray(A.multiply(rows[:, np.newaxis]).sum(axis=0)).ravel()
        weighted_cols = np.asarray(A.multiply(cols[:, np.newaxis]).sum(axis=0)).ravel()
        cy = weighted_rows / col_sums
        cx = weighted_cols / col_sums
        d2 = (cy[:, np.newaxis] - cy[np.newaxis, :]) ** 2 + \
             (cx[:, np.newaxis] - cx[np.newaxis, :]) ** 2
        centre_close = d2 < (centre_dist_factor * sigma) ** 2
        np.fill_diagonal(centre_close, False)

    # --- Temporal correlation ---
    C_std = C - C.mean(axis=1, keepdims=True)
    C_norm = np.sqrt((C_std ** 2).sum(axis=1, keepdims=True))
    C_norm = np.maximum(C_norm, 1e-10)
    C_normed = C_std / C_norm
    R = C_normed @ C_normed.T            # (K, K) Pearson matrix
    np.fill_diagonal(R, 0.0)

    # --- Merge graph: temporally correlated AND (spatially overlapping OR centres close) ---
    spatial_link = (jaccard > thr_overlap) | centre_close
    merge_mask = spatial_link & (np.abs(R) > thr_corr)
    # Symmetrize and treat as undirected graph
    merge_graph = sp.csr_matrix(merge_mask.astype(np.float32))
    n_comp, labels = connected_components(merge_graph, directed=False)

    n_merged = 0
    A_new_cols: list[sp.csc_matrix] = []
    C_new_rows: list[np.ndarray] = []
    members_per_group: list[np.ndarray] = []

    for comp_id in range(n_comp):
        members = np.where(labels == comp_id)[0]
        members_per_group.append(members.astype(np.int32))
        if len(members) == 1:
            A_new_cols.append(A.getcol(members[0]).tocsc())
            C_new_rows.append(C[members[0]])
            continue

        n_merged += 1
        # Merge: sum spatial, re-threshold to keep the blob compact, mean
        # temporal (clipped non-negative). The caller's next update_temporal
        # will deconvolve the merged trace with the cached AR coefficient.
        A_merged_flat = np.asarray(A[:, members].sum(axis=1)).ravel().astype(np.float32)
        if dims is not None:
            A_merged_flat = threshold_footprint(A_merged_flat, dims, max_thr=max_thr)
        A_merged_sparse = sp.csc_matrix(A_merged_flat.reshape(-1, 1))

        C_merged_row = C[members].mean(axis=0).clip(0).astype(np.float32)

        A_new_cols.append(A_merged_sparse)
        C_new_rows.append(C_merged_row)

    A_out = sp.hstack(A_new_cols, format="csc") if A_new_cols else sp.csc_matrix(A.shape)
    C_out = np.vstack(C_new_rows) if C_new_rows else np.empty((0, T), dtype=np.float32)
    return A_out, C_out, n_merged, members_per_group
