"""Ring-model background estimation for 1-photon CNMFe.

The 1-photon background is spatially correlated: each pixel's background
fluctuation is well-predicted by a weighted sum of the pixels in a ring
around it. The ring model solves:

    min_{W, b0}  ||X - W * X||_F
    subject to   W[i, j] = 0  for j not in ring(i)

where X = Y - A @ C - b0[:, None]  (data minus neural signal minus baseline).

This produces a sparse weight matrix W and per-pixel baseline b0.

Reference (algorithmic only): CaImAn initialization.py:compute_W (line 1900).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from skimage.morphology import disk as morpho_disk

from cnmfe._utils import get_xp, to_numpy


# ---------------------------------------------------------------------------
# Module-level worker (must be importable for multiprocessing pickling)
# ---------------------------------------------------------------------------

def _ring_pixel_batch(
    pixel_start: int,
    X_batch: np.ndarray,
    ring_indices: list,
    X_full: np.ndarray,
    lambda_reg: float,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Process one batch of pixels for ring-model weight estimation.

    Args:
        pixel_start: Global index of the first pixel in this batch.
        X_batch: (batch_size, T) — centred residual for these pixels (targets).
        ring_indices: List of int arrays, ring_indices[i] = flat indices of
                      ring pixels around global pixel pixel_start + i.
        X_full: (H*W, T) — full centred residual (needed to read ring pixels).
        lambda_reg: Ridge regularization strength.

    Returns:
        List of (pixel_idx, ring_flat_indices, weights) for pixels that have
        at least one ring neighbour.
    """
    results = []
    for i, ring in enumerate(ring_indices):
        if len(ring) == 0:
            continue
        B = X_full[ring]                      # (n_ring, T)
        BTB = B @ B.T                          # (n_ring, n_ring)
        reg = lambda_reg * np.trace(BTB)
        BTB.flat[:: len(ring) + 1] += reg
        try:
            w = np.linalg.solve(BTB, B @ X_batch[i])
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(BTB, B @ X_batch[i], rcond=None)[0]
        results.append((pixel_start + i, ring, w.astype(np.float32)))
    return results


# ---------------------------------------------------------------------------
# Ring index precomputation
# ---------------------------------------------------------------------------

def build_ring_indices(dims: tuple[int, int], radius: float) -> list[np.ndarray]:
    """Precompute flat pixel indices of the ring around each pixel.

    The ring contains pixels at Euclidean distance in (radius, radius+1].
    Pixels near the border will have fewer ring neighbors (clipped at boundary).

    Args:
        dims: (H, W) image dimensions.
        radius: Inner radius of the ring in pixels.

    Returns:
        ring_idx: List of length H*W; ring_idx[p] is an int array of flat
                  (C-order) indices of the ring pixels around pixel p.
    """
    H, W = dims
    inner = morpho_disk(int(radius))
    outer = morpho_disk(int(radius) + 1)
    pad = outer.shape[0] // 2 - inner.shape[0] // 2
    ring_mask = outer.copy()
    ring_mask[pad : pad + inner.shape[0], pad : pad + inner.shape[1]] -= inner
    ring_mask = ring_mask > 0

    ring_offsets = np.argwhere(ring_mask) - np.array(ring_mask.shape) // 2

    ring_idx: list[np.ndarray] = []
    for p in range(H * W):
        row, col = divmod(p, W)
        rows = row + ring_offsets[:, 0]
        cols = col + ring_offsets[:, 1]
        valid = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
        flat = rows[valid] * W + cols[valid]
        ring_idx.append(flat.astype(np.int32))

    return ring_idx


# ---------------------------------------------------------------------------
# Weight estimation
# ---------------------------------------------------------------------------

def _compute_W_gpu(
    X: np.ndarray,
    ring_idx: list,
    lambda_reg: float,
    xp,
) -> tuple[list, list, list]:
    """Vectorized GPU ring regression using batched linalg.solve.

    Groups pixels by ring size so all pixels with the same neighbourhood
    count can be solved in a single batched call, removing Python-per-pixel
    overhead for the dominant interior-pixel group.

    Returns three lists (rows, cols, data) ready for sparse assembly.
    """
    from collections import defaultdict

    n_pixels = X.shape[0]
    X_xp = xp.asarray(X)

    # Group pixel indices by ring neighbourhood size
    size_to_pixels: dict[int, list[int]] = defaultdict(list)
    for p, ring in enumerate(ring_idx):
        size_to_pixels[len(ring)].append(p)

    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []
    data_list: list[np.ndarray] = []

    for ring_size, pixel_list in size_to_pixels.items():
        if ring_size == 0:
            continue
        pixel_arr = np.array(pixel_list, dtype=np.int32)
        n_batch = len(pixel_arr)

        # ring_mat[i, k] = flat index of k-th ring neighbour of pixel i
        ring_mat = np.array([ring_idx[p] for p in pixel_list], dtype=np.int32)

        # B[i] = X[ring_mat[i]], shape (n_batch, ring_size, T)
        B = X_xp[ring_mat]  # advanced indexing — works on CuPy

        # BTB[i] = B[i] @ B[i].T, shape (n_batch, ring_size, ring_size)
        BTB = xp.einsum("brt,bst->brs", B, B)

        # Regularise: add lambda_reg * trace(BTB[i]) * I to each block
        traces = xp.trace(BTB, axis1=1, axis2=2)        # (n_batch,)
        eye = xp.eye(ring_size, dtype=BTB.dtype)
        BTB = BTB + (lambda_reg * traces)[:, None, None] * eye[None]

        # RHS: Bx[i] = B[i] @ X[pixel_i], shape (n_batch, ring_size)
        X_targets = X_xp[pixel_arr]                      # (n_batch, T)
        Bx = xp.einsum("brt,bt->br", B, X_targets)

        # Batched solve: BTB[i] @ w[i] = Bx[i]
        try:
            w = xp.linalg.solve(BTB, Bx)                # (n_batch, ring_size)
        except Exception:
            w = xp.stack([xp.linalg.solve(BTB[i], Bx[i]) for i in range(n_batch)])

        w_cpu = to_numpy(w).astype(np.float32)
        for i, p in enumerate(pixel_list):
            rows_list.append(np.full(ring_size, p, dtype=np.int32))
            cols_list.append(ring_idx[p])
            data_list.append(w_cpu[i])

    return rows_list, cols_list, data_list


def compute_W(
    Y_flat: np.ndarray,
    A: sp.csc_matrix,
    C: np.ndarray,
    dims: tuple[int, int],
    radius: float,
    lambda_reg: float = 1e-5,
    n_jobs: int = 1,
    device: str = "cpu",
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Fit ring-model weights W and per-pixel baseline b0.

    For each pixel i, solve:
        w_i = argmin_{w} ||X[i] - w @ X[ring_i]||^2 + lambda * ||w||^2
    where X = Y - A @ C - b0[:, None] is the neural-signal-subtracted residual.

    The baseline b0 is estimated as the temporal mean of X before fitting W.

    Args:
        Y_flat: (H*W, T) raw movie in pixel-major order.
        A: (H*W, K) sparse spatial footprints.
        C: (K, T) temporal traces.
        dims: (H, W) spatial dimensions.
        radius: Ring inner radius in pixels.
        lambda_reg: Ridge regularization as fraction of trace(B @ B.T).
        n_jobs: Number of parallel workers (-1 = all CPUs, 1 = serial).
                Ignored when device='cuda'.
        device: 'cpu' or 'cuda'. GPU uses batched linalg.solve (fast for
                large movies with many interior pixels).

    Returns:
        W: Sparse (H*W, H*W) weight matrix (row i = weights for pixel i).
        b0: (H*W,) baseline per pixel.
    """
    H, W = dims
    n_pixels, T = Y_flat.shape
    xp = get_xp(device)

    X = Y_flat - A.dot(C)
    b0 = X.mean(axis=1)
    X -= b0[:, np.newaxis]

    ring_idx = build_ring_indices(dims, radius)

    if xp is not np:
        # GPU path: vectorized batched solve grouped by ring size
        rows_list, cols_list, data_list = _compute_W_gpu(X, ring_idx, lambda_reg, xp)
    else:
        # CPU path: serial or joblib-parallel batches
        batch_size = 500
        batches = [
            (s, min(s + batch_size, n_pixels))
            for s in range(0, n_pixels, batch_size)
        ]

        if n_jobs == 1:
            raw_results: list[tuple[int, np.ndarray, np.ndarray]] = []
            for start, end in batches:
                raw_results.extend(
                    _ring_pixel_batch(start, X[start:end], ring_idx[start:end], X, lambda_reg)
                )
        else:
            from joblib import Parallel, delayed
            batch_lists = Parallel(n_jobs=n_jobs)(
                delayed(_ring_pixel_batch)(start, X[start:end], ring_idx[start:end], X, lambda_reg)
                for start, end in batches
            )
            raw_results = [item for batch in batch_lists for item in batch]

        rows_list = [np.full(len(ring), p, dtype=np.int32) for p, ring, _ in raw_results]
        cols_list = [ring for _, ring, _ in raw_results]
        data_list = [w for _, _, w in raw_results]

    # Assemble sparse W (always on CPU via scipy.sparse)
    if rows_list:
        W_mat = sp.csr_matrix(
            (np.concatenate(data_list), (np.concatenate(rows_list), np.concatenate(cols_list))),
            shape=(n_pixels, n_pixels),
            dtype=np.float32,
        )
    else:
        W_mat = sp.csr_matrix((n_pixels, n_pixels), dtype=np.float32)

    return W_mat, b0.astype(np.float32)


def subtract_background(
    Y_flat: np.ndarray,
    W: sp.csr_matrix,
    b0: np.ndarray,
) -> np.ndarray:
    """Subtract ring-model background from the raw movie.

    The model predicts background at pixel i as:
        bg_i = b0[i] + sum_j W[i, j] * (Y[j] - b0[j])
    So the background-free signal is:
        Y_res = Y - b0[:,None] - W @ (Y - b0[:,None])
              = (I - W) @ (Y - b0[:,None])

    Args:
        Y_flat: (H*W, T) raw movie.
        W: (H*W, H*W) sparse ring weight matrix.
        b0: (H*W,) per-pixel baseline.

    Returns:
        Y_res: (H*W, T) background-subtracted movie.
    """
    X = Y_flat - b0[:, np.newaxis]
    return X - W.dot(X)
