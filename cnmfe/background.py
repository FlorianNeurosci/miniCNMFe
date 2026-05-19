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


def _ring_pixel_batch_slab(
    pixel_starts: np.ndarray,
    target_local_idx: np.ndarray,
    ring_local_indices: list,
    ring_global_indices: list,
    X_slab: np.ndarray,
    lambda_reg: float,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Like ``_ring_pixel_batch`` but reads from a local slab of pixels.

    Used by the streaming (zarr-backed) ``compute_W`` path: rather than
    pass the full ``(H*W, T_sub)`` residual, the caller pre-extracts just
    the rows needed for this pixel batch (the batch pixels themselves
    plus the union of their ring neighbours) into ``X_slab``.

    Args:
        pixel_starts: (B,) global pixel indices for the batch's targets.
        target_local_idx: (B,) row index of each batch pixel in ``X_slab``.
        ring_local_indices: list of length B; each entry is an int array
            of row indices into ``X_slab`` for that pixel's ring neighbours.
        ring_global_indices: parallel list — original ring flat indices in
            the full ``H*W`` array. Carried through so the caller can
            assemble the global sparse W.
        X_slab: ``(slab_size, T_sub)`` dense residual for the slab pixels.
        lambda_reg: Ridge regularization strength.
    """
    results = []
    for i, ring_local in enumerate(ring_local_indices):
        if len(ring_local) == 0:
            continue
        target_row = X_slab[target_local_idx[i]]            # (T_sub,)
        B = X_slab[ring_local]                              # (n_ring, T_sub)
        BTB = B @ B.T
        reg = lambda_reg * np.trace(BTB)
        BTB.flat[:: len(ring_local) + 1] += reg
        try:
            w = np.linalg.solve(BTB, B @ target_row)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(BTB, B @ target_row, rcond=None)[0]
        results.append((int(pixel_starts[i]), ring_global_indices[i], w.astype(np.float32)))
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


def _compute_w_streaming(
    Y_flat,
    A: sp.csc_matrix,
    C_sub: np.ndarray,
    b0: np.ndarray,
    ring_idx: list,
    actual_tsub: int,
    lambda_reg: float,
    n_jobs: int,
) -> list:
    """Per-pixel-batch ring weight solve against a zarr-backed Y_flat.

    The in-memory path of ``compute_W`` materialises the entire
    ``(H*W, T_sub)`` residual X. With zarr-backed input that defeats the
    point — the whole movie would land in RAM. Here we iterate over pixel
    batches; for each batch we read only the rows we actually need (the
    batch's targets plus the union of their ring neighbours), build the
    local X slab on the fly, and feed it to ``_ring_pixel_batch_slab``.

    Peak RAM per batch ≈ ``slab_size * T_sub * 4`` bytes, with
    ``slab_size`` bounded by the batch size times the per-pixel ring size
    plus the batch itself. For batch=256 and typical ring size ≈24 this
    is ~7k pixels × T_sub × 4 bytes — negligible relative to the full
    movie.
    """
    n_pixels = Y_flat.shape[0]
    batch_size = 256

    if actual_tsub > 1:
        time_idx = np.arange(0, Y_flat.shape[1], actual_tsub)
    else:
        time_idx = None

    def _process_batch(start: int, end: int):
        ring_indices_batch = ring_idx[start:end]
        # Collect the rows we need to read from Y_flat: the batch's targets
        # plus the union of their ring neighbours. Empty-ring entries are
        # skipped (those pixels contribute nothing to W).
        nonempty = [r for r in ring_indices_batch if len(r) > 0]
        if not nonempty:
            return []
        batch_pixels = np.arange(start, end, dtype=np.int64)
        union = np.unique(np.concatenate([batch_pixels, *nonempty]))

        # Read the slab from Y_flat — orthogonal selection for zarr-style
        # arrays, falls back to fancy indexing for anything else.
        if time_idx is not None and hasattr(Y_flat, "get_orthogonal_selection"):
            Y_slab = np.asarray(
                Y_flat.get_orthogonal_selection((union, time_idx)),
                dtype=np.float32,
            )
        elif time_idx is not None:
            # Generic fallback (numpy array): fancy row index + strided cols.
            Y_slab = np.asarray(Y_flat[union][:, ::actual_tsub], dtype=np.float32)
        else:
            Y_slab = np.asarray(Y_flat[union], dtype=np.float32)

        A_slab = A[union] if not sp.issparse(A) else A.tocsr()[union]
        AC_slab = np.asarray(A_slab @ C_sub, dtype=np.float32)
        X_slab = Y_slab - AC_slab - b0[union, None]

        # Build slab-local indices: each target pixel's row, and each
        # ring neighbour's row, expressed as positions in `union`.
        target_local_idx = np.searchsorted(union, batch_pixels)
        ring_local_indices = []
        ring_global_indices = []
        pixel_starts = []
        for offset, ring in enumerate(ring_indices_batch):
            if len(ring) == 0:
                continue
            ring_local_indices.append(
                np.searchsorted(union, ring).astype(np.int32)
            )
            ring_global_indices.append(ring)
            pixel_starts.append(start + offset)

        # Re-key target_local_idx to match only the nonempty entries.
        kept_offsets = np.array(
            [i for i, r in enumerate(ring_indices_batch) if len(r) > 0],
            dtype=np.int64,
        )
        target_local_idx_kept = target_local_idx[kept_offsets]

        return _ring_pixel_batch_slab(
            np.asarray(pixel_starts, dtype=np.int64),
            target_local_idx_kept,
            ring_local_indices,
            ring_global_indices,
            X_slab,
            lambda_reg,
        )

    batches = [(s, min(s + batch_size, n_pixels))
               for s in range(0, n_pixels, batch_size)]

    if n_jobs == 1:
        raw_results: list = []
        for start, end in batches:
            raw_results.extend(_process_batch(start, end))
        return raw_results

    # Thread-based parallelism. Process-based (loky) would have to ship
    # A, C_sub, b0, and the zarr handle to every worker per call — that's
    # ~12 MB × n_batches of IPC, more expensive than the compute itself.
    # The heavy ops in _process_batch (zarr reads, np.linalg.solve, einsum,
    # sparse mat-vec) all release the GIL, so threads get real speedup
    # without the serialisation tax.
    from joblib import Parallel, delayed
    from threadpoolctl import threadpool_limits

    # See spatial.py for the rationale: cap inner BLAS to 1 so the n_jobs
    # worker threads don't each spawn n_cores OpenMP threads.
    with threadpool_limits(limits=1, user_api="blas"):
        per_batch = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_process_batch)(start, end) for start, end in batches
        )
    raw_results = [item for batch_out in per_batch for item in batch_out]
    return raw_results


def compute_W(
    Y_flat: np.ndarray,
    A: sp.csc_matrix,
    C: np.ndarray,
    dims: tuple[int, int],
    radius: float,
    lambda_reg: float = 1e-5,
    n_jobs: int = 1,
    device: str = "cpu",
    tsub: int = 1,
    W_cached: "sp.csr_matrix | None" = None,
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
        tsub: Subsample factor along time for the expensive BTB solve.
        W_cached: If given, reuse this ring weight matrix instead of solving.
                  Only b0 is refit from the current (A, C). Use this when
                  W has already been computed on similar (A, C) earlier
                  in the same fit() call and only the baseline needs to
                  track residual changes.

    Returns:
        W: Sparse (H*W, H*W) weight matrix (row i = weights for pixel i).
        b0: (H*W,) baseline per pixel.
    """
    H, W = dims
    n_pixels, T = Y_flat.shape
    xp = get_xp(device)
    Y_is_numpy = isinstance(Y_flat, np.ndarray)

    # Streaming b0: (Y.sum(axis=1) - A @ C.sum(axis=1)) / T is identically
    # (Y - A @ C).mean(axis=1) by linearity, without ever materialising the
    # full (H*W, T) residual. For non-numpy Y_flat (e.g. zarr), sum along
    # axis=1 by streaming pixel batches so we never load the whole store.
    C_sum = np.asarray(C.sum(axis=1), dtype=np.float32)
    if Y_is_numpy:
        Y_sum = np.asarray(Y_flat.sum(axis=1), dtype=np.float32)
    else:
        Y_sum = np.zeros(n_pixels, dtype=np.float32)
        sum_batch = 4096
        for s in range(0, n_pixels, sum_batch):
            e = min(s + sum_batch, n_pixels)
            Y_sum[s:e] = np.asarray(Y_flat[s:e], dtype=np.float32).sum(axis=1)
    AC_sum = np.asarray(A @ C_sum, dtype=np.float32).ravel()
    b0 = ((Y_sum - AC_sum) / float(T)).astype(np.float32)

    if W_cached is not None:
        # Reuse the ring weight matrix; only b0 needs refreshing as A, C drift.
        return W_cached, b0

    # Optionally subsample time for the expensive BTB = B @ B.T solve.
    # b0 is kept from the full T; only the ring regression uses fewer frames.
    # Cap tsub so at least 200 frames are used (prevents noisy W on short movies).
    actual_tsub = max(1, min(tsub, T // 200)) if tsub > 1 else 1
    ring_idx = build_ring_indices(dims, radius)

    if Y_is_numpy:
        # In-memory path: build X_fit once at subsampled time resolution,
        # then dispatch per-pixel batches over the dense slab.
        if actual_tsub > 1:
            Y_sub = np.asarray(Y_flat[:, ::actual_tsub], dtype=np.float32)
            C_sub = C[:, ::actual_tsub]
        else:
            Y_sub = np.asarray(Y_flat, dtype=np.float32)
            C_sub = C
        AC_sub = np.asarray(A.dot(C_sub), dtype=np.float32)
        X_fit = Y_sub - AC_sub - b0[:, np.newaxis]
        del Y_sub, AC_sub
    else:
        # Zarr / on-disk: defer X_fit construction to the per-batch loop
        # below so the full ``(H*W, T/tsub)`` slab is never materialised.
        C_sub = C[:, ::actual_tsub] if actual_tsub > 1 else C
        X_fit = None

    if xp is not np and Y_is_numpy:
        # GPU path: vectorized batched solve grouped by ring size (in-memory only).
        rows_list, cols_list, data_list = _compute_W_gpu(X_fit, ring_idx, lambda_reg, xp)
    elif not Y_is_numpy:
        # Streaming CPU path for zarr-backed Y_flat.
        # Per pixel batch: pull only ring-neighbour rows into a small slab,
        # build X_fit just for those rows, run the per-pixel solve.
        raw_results = _compute_w_streaming(
            Y_flat, A, C_sub, b0, ring_idx, actual_tsub, lambda_reg, n_jobs,
        )
        rows_list = [np.full(len(ring), p, dtype=np.int32) for p, ring, _ in raw_results]
        cols_list = [ring for _, ring, _ in raw_results]
        data_list = [w for _, _, w in raw_results]
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
                    _ring_pixel_batch(start, X_fit[start:end], ring_idx[start:end], X_fit, lambda_reg)
                )
        else:
            from joblib import Parallel, delayed
            from threadpoolctl import threadpool_limits

            # Threads: each call passes a per-batch slice of X_fit plus the
            # full X_fit (for ring lookups). loky would pickle the full X_fit
            # per call -- threads share it. Inner ops (linalg.solve, einsum)
            # release the GIL.
            # threadpool_limits caps BLAS to 1 thread per worker -- see
            # spatial.py for the rationale.
            with threadpool_limits(limits=1, user_api="blas"):
                batch_lists = Parallel(n_jobs=n_jobs, prefer="threads")(
                    delayed(_ring_pixel_batch)(start, X_fit[start:end], ring_idx[start:end], X_fit, lambda_reg)
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

    Note:
        This materialises the full (H*W, T) result and allocates two more
        full-size temporaries. For large T use ``BackgroundSubtractor`` and
        request pixel-row slices on demand instead.
    """
    X = Y_flat - b0[:, np.newaxis]
    return X - W.dot(X)


class BackgroundSubtractor:
    """Lazy ring-background subtractor — on-demand pixel-row slices.

    Computes pixel-row slices of ``Y_bg = (I - W) @ (Y - b0[:, None])``
    without materialising the full ``(H*W, T)`` result. Used by extraction
    to bound peak RAM.

    Mathematical identity used per slice:

        Y_bg[start:end] = Y[start:end]
                        - b0[start:end, None]
                        - W[start:end] @ Y
                        + (W[start:end] @ b0)[:, None]

    The two W matmuls are sparse-dense and sparse@vector — they never need
    a materialised ``X = Y - b0`` intermediate.

    Supports ``bg[start:end]`` (equivalent to ``bg.slice(start, end)``) and
    ``bg.project_onto(A)`` (streaming ``Y_bg.T @ A``) so existing call sites
    that read ``Y_bg`` only via slicing and matmul can swap a numpy array
    for this object without other changes.

    Args:
        Y_flat: (H*W, T) source movie. numpy array or zarr.Array.
        W: (H*W, H*W) sparse ring weight matrix.
        b0: (H*W,) per-pixel baseline.
    """

    def __init__(
        self,
        Y_flat: np.ndarray,
        W: sp.csr_matrix,
        b0: np.ndarray,
    ) -> None:
        self.Y_flat = Y_flat
        self.W = W if sp.isspmatrix_csr(W) else W.tocsr()
        self.b0 = np.asarray(b0, dtype=np.float32)
        self.shape = (int(Y_flat.shape[0]), int(Y_flat.shape[1]))
        self.dtype = np.dtype(np.float32)

    def slice(self, start: int, end: int) -> np.ndarray:
        """Return ``Y_bg[start:end, :]`` as a fresh ``(end-start, T)`` array.

        When ``Y_flat`` is a zarr (or any non-numpy array), the
        ``W_chunk @ Y_flat`` matmul cannot be done directly — it would read
        the whole on-disk store. Instead we extract just the
        ring-neighbour rows (the union of nonzero columns of
        ``W_chunk``), remap the sparse column indices to that small
        buffer, and do the matmul on the dense slab.
        """
        W_chunk = self.W[start:end]
        Y_chunk = np.asarray(self.Y_flat[start:end], dtype=np.float32)

        # Extract only the ring-neighbour rows needed for this batch and do the
        # sparse matmul on that small C-contiguous slab. This avoids the
        # pathological access pattern of `W_chunk @ self.Y_flat` when Y_flat
        # is the F-contiguous transposed view produced by make_2d(): scipy's
        # CSR matmul would access Y_flat[j, :] with 1.44 MB stride per time
        # step on a 15+ GB array, causing ~99 % DRAM cache miss rate (~94 min
        # for a 600×600 × 11k movie). Fancy indexing `self.Y_flat[needed]`
        # always returns a C-contiguous copy, so the subsequent sparse matmul
        # stays cache-friendly regardless of Y_flat's layout (numpy or zarr).
        indices = W_chunk.indices
        if indices.size == 0:
            W_Y = np.zeros_like(Y_chunk)
        else:
            needed = np.unique(indices)
            Y_needed = np.asarray(self.Y_flat[needed], dtype=np.float32)
            remap = np.searchsorted(needed, indices)
            W_chunk_remapped = sp.csr_matrix(
                (W_chunk.data, remap, W_chunk.indptr),
                shape=(W_chunk.shape[0], needed.size),
            )
            W_Y = np.asarray(W_chunk_remapped @ Y_needed, dtype=np.float32)

        W_b0 = np.asarray(W_chunk @ self.b0, dtype=np.float32)
        out = Y_chunk - self.b0[start:end, None] - W_Y
        out += W_b0[:, None]
        return out

    def __getitem__(self, key) -> np.ndarray:
        if isinstance(key, slice):
            start = 0 if key.start is None else int(key.start)
            stop = self.shape[0] if key.stop is None else int(key.stop)
            if key.step not in (None, 1):
                raise ValueError("BackgroundSubtractor only supports contiguous slices")
            return self.slice(start, stop)
        raise TypeError(
            f"BackgroundSubtractor supports only slice indexing, got {type(key).__name__}"
        )

    def project_onto(
        self,
        A: "sp.spmatrix | np.ndarray",
        batch_size: int = 4096,
    ) -> np.ndarray:
        """Streaming ``Y_bg.T @ A``  →  ``(T, K)`` without materialising Y_bg.

        Iterates over pixel batches, accumulating partial contributions to
        the projection. Equivalent to ``self[:].T @ A`` but bounded RAM.
        """
        T = self.shape[1]
        K = int(A.shape[1])
        YA = np.zeros((T, K), dtype=np.float32)
        if K == 0:
            return YA
        A_csr = A.tocsr() if sp.issparse(A) else A
        n_pix = self.shape[0]
        for start in range(0, n_pix, batch_size):
            end = min(start + batch_size, n_pix)
            Y_chunk = self.slice(start, end)        # (B, T)
            A_chunk = A_csr[start:end]               # (B, K), sparse or dense
            YA += np.asarray(Y_chunk.T @ A_chunk, dtype=np.float32)
        return YA
