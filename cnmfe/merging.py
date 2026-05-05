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

from cnmfe.temporal import deconvolve, estimate_ar_params


def merge_components(
    A: sp.csc_matrix,
    C: np.ndarray,
    thr_corr: float = 0.85,
    thr_overlap: float = 0.5,
    ar_order: int = 1,
) -> tuple[sp.csc_matrix, np.ndarray, int]:
    """Merge spatially overlapping and temporally correlated components.

    Two components i, j are merged if:
        Jaccard(i, j) > thr_overlap   AND   |Pearson(C[i], C[j])| > thr_corr

    Merged spatial footprint: sum of individual footprints, re-normalised.
    Merged temporal trace: mean of individual traces, then re-deconvolved.

    Args:
        A: (H*W, K) sparse spatial footprints.
        C: (K, T) temporal traces.
        thr_corr: Minimum Pearson correlation between traces to trigger merge.
        thr_overlap: Minimum Jaccard overlap between footprints.
        ar_order: AR model order for re-deconvolution after merge.

    Returns:
        A_merged: (H*W, K_new) sparse footprints.
        C_merged: (K_new, T) calcium traces.
        n_merged: Number of merge events (groups merged > 1 component).
    """
    K, T = C.shape
    if K <= 1:
        return A, C, 0

    # --- Spatial overlap (Jaccard) ---
    O = (A.T @ A).toarray()              # (K, K) — dot product of footprints
    nA = np.maximum(np.diag(O), 1e-10)  # (K,) — squared norm of each footprint
    # Jaccard[i,j] = O[i,j] / (nA[i] + nA[j] - O[i,j])
    denom = nA[:, np.newaxis] + nA[np.newaxis, :] - O
    denom = np.maximum(denom, 1e-10)
    jaccard = O / denom                  # (K, K)
    np.fill_diagonal(jaccard, 0.0)

    # --- Temporal correlation ---
    C_std = C - C.mean(axis=1, keepdims=True)
    C_norm = np.sqrt((C_std ** 2).sum(axis=1, keepdims=True))
    C_norm = np.maximum(C_norm, 1e-10)
    C_normed = C_std / C_norm
    R = C_normed @ C_normed.T            # (K, K) Pearson matrix
    np.fill_diagonal(R, 0.0)

    # --- Merge graph ---
    merge_mask = (jaccard > thr_overlap) & (np.abs(R) > thr_corr)
    # Symmetrize and treat as undirected graph
    merge_graph = sp.csr_matrix(merge_mask.astype(np.float32))
    n_comp, labels = connected_components(merge_graph, directed=False)

    n_merged = 0
    A_new_cols: list[sp.csc_matrix] = []
    C_new_rows: list[np.ndarray] = []

    for comp_id in range(n_comp):
        members = np.where(labels == comp_id)[0]
        if len(members) == 1:
            A_new_cols.append(A.getcol(members[0]).tocsc())
            C_new_rows.append(C[members[0]])
            continue

        n_merged += 1
        # Merge: sum spatial, mean temporal, then re-deconvolve
        A_merged_col = A[:, members].sum(axis=1)
        A_merged_sparse = sp.csc_matrix(A_merged_col)

        C_merged_row = C[members].mean(axis=0)
        try:
            g, sn = estimate_ar_params(C_merged_row, p=ar_order)
            c_clean, _, _ = deconvolve(C_merged_row, g, sn)
        except Exception:
            c_clean = C_merged_row.clip(0)

        A_new_cols.append(A_merged_sparse)
        C_new_rows.append(c_clean.astype(np.float32))

    A_out = sp.hstack(A_new_cols, format="csc") if A_new_cols else sp.csc_matrix(A.shape)
    C_out = np.vstack(C_new_rows) if C_new_rows else np.empty((0, T), dtype=np.float32)
    return A_out, C_out, n_merged
