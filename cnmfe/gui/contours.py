"""Footprint contour + centroid helpers for the curation GUI.

All functions operate on the standard CNMFe layout: ``A`` is a sparse CSC
``(H*W, K)`` matrix with row-major pixel order (``h * W + w``, see
``cnmfe._utils.make_2d``).

``precompute_centroids`` walks the CSC directly without densification — fine
for K = many hundreds. ``component_contour`` densifies a single column.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def precompute_centroids(
    A_csc: sp.csc_matrix, H: int, W: int
) -> np.ndarray:
    """Return ``(K, 2)`` weighted centroids ``[y, x]`` per footprint.

    Each centroid is the value-weighted mean of the footprint's nonzero
    pixels. Empty columns get the FOV centre as a stable fallback.
    """
    if not sp.issparse(A_csc):
        raise TypeError("A must be a scipy.sparse matrix")
    A = A_csc.tocsc()
    K = A.shape[1]
    centroids = np.zeros((K, 2), dtype=np.float32)
    indptr = A.indptr
    indices = A.indices
    data = A.data
    for k in range(K):
        lo, hi = indptr[k], indptr[k + 1]
        if hi <= lo:
            centroids[k] = [(H - 1) / 2.0, (W - 1) / 2.0]
            continue
        rows = indices[lo:hi]
        vals = np.asarray(data[lo:hi], dtype=np.float64)
        # row -> (y, x)
        ys = rows // W
        xs = rows % W
        wsum = vals.sum()
        if wsum <= 0:
            centroids[k] = [(H - 1) / 2.0, (W - 1) / 2.0]
            continue
        cy = float((ys * vals).sum() / wsum)
        cx = float((xs * vals).sum() / wsum)
        centroids[k] = [cy, cx]
    return centroids


def footprint_image(A_csc: sp.csc_matrix, k: int, H: int, W: int) -> np.ndarray:
    """Return ``(H, W)`` float32 dense footprint for component ``k``."""
    col = A_csc[:, int(k)]
    arr = np.asarray(col.todense()).ravel().astype(np.float32)
    return arr.reshape(H, W)


def component_contour(
    a_2d: np.ndarray, level_frac: float = 0.3
) -> list[np.ndarray]:
    """Find iso-contours of a footprint at ``level_frac * max(a_2d)``.

    Returns a list of ``(N, 2)`` arrays in ``(row, col)`` order, exactly as
    ``skimage.measure.find_contours`` does. Caller plots with ``ax.plot(c[:,1],
    c[:,0])`` (matplotlib wants ``x, y``). Empty list if the footprint is all
    zero.
    """
    from skimage.measure import find_contours

    peak = float(a_2d.max())
    if peak <= 0:
        return []
    level = peak * float(level_frac)
    return [np.asarray(c, dtype=np.float32) for c in find_contours(a_2d, level)]
