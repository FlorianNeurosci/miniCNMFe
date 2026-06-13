"""Spatial footprint update via per-pixel non-negative LASSO regression.

For each pixel p, the data is modelled as:
    Y[p, :] = sum_{k in active(p)} A[p, k] * C[k, :] + noise

where active(p) is the set of components whose footprints overlap pixel p
(within a dilation of the current footprint). This small local set makes the
regression fast and well-conditioned.

Reference (algorithmic only): CaImAn spatial.py:update_spatial_components (line 29).
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.ndimage as ndi
import scipy.sparse as sp
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model._cd_fast import enet_coordinate_descent_gram

# Cyclic CD doesn't need randomness, but the Cython entry point still wants
# a RandomState object. Build one once at import time and pass it on every
# call -- never mutated when random=0.
_CD_RNG = np.random.RandomState(0)

# Worker-thread cap for the per-pixel CD parallel path. The loop in
# `_spatial_pixel_batch` is dominated by GIL-held Python/numpy glue
# (fancy-indexing C[active], the tiny C_active@C_active.T, np.diag/np.max,
# array allocation) wrapped around a *short* GIL-released Cython solve. A
# GIL-bound loop can't use more than ~one Python thread at a time, so handing
# it the full core budget (e.g. 256) just thrashes the GIL and collapses to
# ~1 effective core. Capping at a sweet spot saturates the GIL-released
# fraction without the thrash. Tunable via CNMFeParams.spatial_thread_cap.
_SPATIAL_THREAD_CAP = 16


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
    max_iter: int = 1000,
    tol: float = 1e-4,
    ridge: float = 1e-2,
) -> tuple[list[tuple[int, int, float]], tuple[int, int, int, int, int]]:
    """Process one contiguous batch of pixels.

    Args:
        pixel_start: Global index of the first pixel in this batch.
        Y_batch: (batch_size, T) — temporal traces for this batch only.
        C: (K, T) — all component temporal traces.
        support_batch: List of int arrays; support_batch[i] = active components
                       for global pixel pixel_start + i.
        sn_batch: (batch_size,) noise std for this batch.
        T: Number of time points.
        max_iter: Per-pixel CD iteration cap (passed straight through to
            ``enet_coordinate_descent_gram``).
        tol: Per-pixel CD convergence tolerance (passed straight through).
        ridge: Elastic-net L2 fraction. The solver's ``beta`` is set to
            ``ridge * max(diag(Gram))`` per pixel, which bounds the condition
            number of ``Gram + beta*I`` to ~``1/ridge``. This is what keeps the
            CD converging in tens of iterations when active components have
            correlated/near-duplicate traces (a near-singular Gram) — without it
            (``ridge=0``, pure LASSO) such pixels crawl to thousands of
            iterations and may hit ``max_iter``. The coefficient shrinkage it
            introduces is ~``ridge`` (≈1% at the default), negligible and
            confined to the degenerate components it stabilises.

    Returns:
        Tuple of (results, stats):
          - results: list of (global_pixel_idx, component_idx, coef) for
            coefficients > 0.
          - stats: (n_unconverged, n_ran, iter_sum, iter_max, active_sum)
            diagnostic counters aggregated by ``update_spatial``:
              * n_unconverged: pixels whose CD hit ``max_iter`` without
                converging (also drives the existing warning line).
              * n_ran: pixels that actually ran CD (non-empty support, lam>0).
              * iter_sum / iter_max: sum / max of CD iteration counts over
                ``n_ran`` pixels (mean iter = iter_sum / n_ran).
              * active_sum: sum of active-set sizes over ``n_ran`` pixels
                (mean active = active_sum / n_ran).
            These let ``update_spatial`` print one line distinguishing
            "slow because CD never converges" from "slow because active
            sets are huge" from "slow because there are many pixels".
    """
    # Call the underlying Cython CD solver directly instead of going
    # through sklearn's Lasso estimator. The estimator's .fit() path
    # spends ~30% of its time in input validation (validate_data,
    # check_X_y, _assert_all_finite, ...) -- pure-Python overhead that
    # also holds the GIL and defeats thread-based parallelism. The
    # Cython solver (enet_coordinate_descent_gram) is what Lasso ends up
    # calling anyway; we just skip the wrapper. It releases the GIL
    # during the CD inner loop so joblib threads now actually parallelise.
    #
    # Math match with `Lasso(alpha=lam, positive=True, fit_intercept=False)`:
    #   sklearn parameterises Lasso as
    #     (1/(2*n_samples)) * ||y - Xw||^2 + alpha * |w|_1
    #   The Cython entry takes the un-normalised L1 penalty as its first
    #   arg, so we pass `lam * T` to match `Lasso(alpha=lam)`.
    results: list[tuple[int, int, float]] = []
    n_unconverged = 0
    n_ran = 0
    iter_sum = 0
    iter_max = 0
    active_sum = 0
    # Suppress sklearn's per-pixel ConvergenceWarning once for the whole batch
    # (entering a catch_warnings context per pixel — ~256× here, 90k× per
    # update — is pure-Python GIL-bound overhead that threads can't hide). The
    # aggregate unconverged count is reported once by update_spatial.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for i, active in enumerate(support_batch):
            if len(active) == 0:
                continue
            C_active = C[active]              # (n_active, T)
            y_p = Y_batch[i]                  # (T,)
            sn_p = float(sn_batch[i])

            # X = C_active.T  (samples x features), so Gram = X.T @ X.
            # Cython solver is float64-only; the casts are cheap (small Gram /
            # Xy / coef arrays, n_active typically 1-5).
            gram = np.ascontiguousarray(C_active @ C_active.T, dtype=np.float64)
            max_energy = float(np.max(np.diag(gram))) if gram.size > 0 else 1.0
            lam = 0.5 * sn_p * np.sqrt(max(max_energy, 1e-10)) / T
            if lam == 0:
                continue

            Xy = np.ascontiguousarray(C_active @ y_p, dtype=np.float64)
            y_p_64 = np.ascontiguousarray(y_p, dtype=np.float64)
            w = np.zeros(len(active), dtype=np.float64)

            try:
                w, _, _, n_iter = enet_coordinate_descent_gram(
                    w,                       # initial coef, mutated in place
                    float(lam) * T,          # alpha * n_samples
                    float(ridge) * max_energy,  # beta = ridge * max diag(Q):
                    #                          # bounds cond(Q+beta*I) ~ 1/ridge so
                    #                          # CD converges even for correlated comps
                    gram,                    # Q = X.T @ X
                    Xy,                      # q = X.T @ y
                    y_p_64,                  # y (for dual-gap check)
                    int(max_iter),
                    float(tol),
                    _CD_RNG,                 # rng (unused with cyclic CD)
                    0,                       # random=0 -> cyclic, deterministic
                    1,                       # positive
                )
                n_iter_i = int(n_iter)
                if n_iter_i >= int(max_iter):
                    n_unconverged += 1
                n_ran += 1
                iter_sum += n_iter_i
                if n_iter_i > iter_max:
                    iter_max = n_iter_i
                active_sum += len(active)
                coef = w
            except Exception:
                coef = np.zeros(len(active), dtype=np.float64)

            p_global = pixel_start + i
            for idx, k in enumerate(active):
                val = float(coef[idx])
                if val > 0:
                    results.append((p_global, int(k), val))
    return results, (n_unconverged, n_ran, iter_sum, iter_max, active_sum)


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
    closing_radius: int = 1,
    circular_max_dist_factor: float = 1.5,
) -> np.ndarray:
    """Post-process a single spatial footprint to enforce compactness.

    Steps:
    1. Reshape to 2D and apply a 3×3 median filter (removes isolated pixels).
    2. Zero pixels below max_thr * max(ai).
    3. Binary-close the support (fills 1-pixel gaps / hollow interiors so the
       LASSO's jagged ring-with-holes survives as a single connected blob
       instead of being split). Skipped when ``closing_radius == 0``.
    4. Keep only the largest connected component.
    5. NON-STANDARD bandaid: clip pixels further than
       ``circular_max_dist_factor * sqrt(area/pi)`` from the footprint
       centroid (``circular_constraint`` from initialization.py). Targets
       thin "tendril" extensions toward neighbouring components that the
       LASSO can produce. Same shape prior already applied at greedy init.
       Skipped when ``circular_max_dist_factor <= 0``.

    Args:
        ai: (H*W,) flat footprint.
        dims: (H, W) spatial dimensions.
        max_thr: Fraction of maximum value below which pixels are zeroed.
        closing_radius: Radius (in pixels) of the morphological binary closing
            applied between thresholding and largest-CC extraction. ``0``
            disables (legacy behaviour); ``1`` (default, 3×3 SE) matches
            CaImAn.
        circular_max_dist_factor: NON-STANDARD. Radius factor for the
            post-update circular constraint (step 5). ``0`` disables;
            ``1.5`` (default) is tighter than the greedy-init factor
            because post-BCD-refinement the footprint shape is known
            better — a generous factor lets thin tendrils survive.

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

    if closing_radius > 0:
        # Run the largest-CC selection against the *closed* binary mask, so
        # bridge pixels filled by the closing unify components for the CC
        # step. The grayscale ai2d is then masked by the chosen CC — the
        # bridge pixels remain 0 in the output (they don't get fake
        # intensity), but both real clusters survive together.
        se = ndi.generate_binary_structure(2, 2)   # 3×3 (8-connectivity)
        if closing_radius > 1:
            se = ndi.iterate_structure(se, closing_radius)
        closed = ndi.binary_closing(ai2d > 0, structure=se)
        labeled, n = ndi.label(closed)
        if n > 1:
            sizes = ndi.sum(closed, labeled, range(1, n + 1))
            largest = int(np.argmax(sizes)) + 1
            keep = labeled == largest
        else:
            keep = closed
        ai2d = ai2d * keep.astype(ai2d.dtype)
    else:
        labeled, n = ndi.label(ai2d > 0)
        if n > 1:
            sizes = ndi.sum(ai2d > 0, labeled, range(1, n + 1))
            largest = int(np.argmax(sizes)) + 1
            ai2d[labeled != largest] = 0.0

    if circular_max_dist_factor > 0 and (ai2d > 0).any():
        # Imported locally to avoid an import cycle with `initialization`.
        from minicnmfe.initialization import circular_constraint
        ai2d = circular_constraint(
            ai2d, max_dist_factor=float(circular_max_dist_factor),
        )

    return ai2d.ravel().astype(np.float32)


def _threshold_one_component(
    k: int,
    pixel_ids: np.ndarray,
    values: np.ndarray,
    n_pixels: int,
    dims: tuple[int, int],
    max_thr: float,
    closing_radius: int,
    circular_max_dist_factor: float,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Build + clean one component's footprint. Module-level for joblib.

    `threshold_footprint` is built on `scipy.ndimage` (median filter, closing,
    label) which releases the GIL, so dispatching this per-component with
    `prefer="threads"` gives real concurrency. Returns ``(k, nz, values_nz)``.
    """
    ai_flat = np.zeros(n_pixels, dtype=np.float32)
    ai_flat[pixel_ids] = values
    ai_flat = threshold_footprint(
        ai_flat, dims, max_thr=max_thr, closing_radius=closing_radius,
        circular_max_dist_factor=circular_max_dist_factor,
    )
    nz = np.where(ai_flat > 0)[0]
    return k, nz, ai_flat[nz]


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
    closing_radius: int = 1,
    max_iter: int = 1000,
    tol: float = 1e-4,
    circular_max_dist_factor: float = 1.5,
    spatial_ridge: float = 1e-2,
    spatial_thread_cap: int = _SPATIAL_THREAD_CAP,
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
        closing_radius: Radius of the binary closing inside
            ``threshold_footprint`` (0 = legacy / no closing; 1 = CaImAn 3×3 SE).
        max_iter: Per-pixel coordinate-descent iteration cap.
        tol: Per-pixel CD convergence tolerance. If any pixels hit
            ``max_iter`` without converging, a single summary line is
            printed (the per-pixel sklearn ConvergenceWarning is suppressed).
        circular_max_dist_factor: Forwarded to ``threshold_footprint``;
            radius factor for the post-update circular constraint.
            ``0`` disables; ``1.5`` (default) is tighter than the
            init-time factor because BCD-refined footprints are known
            better and a generous factor lets thin tendrils survive.
        spatial_ridge: Elastic-net L2 fraction for the per-pixel solve (see
            ``_spatial_pixel_batch``). The default ``1e-2`` keeps the CD
            converging in tens of iterations on real data, where correlated /
            duplicate components otherwise make the Gram near-singular and the
            pure-LASSO CD run to ``max_iter``. ``0`` restores pure LASSO.
        spatial_thread_cap: Max worker threads for the parallel (``n_jobs>1``)
            per-pixel CD. The loop is GIL-bound, so more than ~16 threads thrash
            the GIL instead of helping (observed ~1 effective core at
            ``n_jobs=256``). The effective worker count is
            ``min(n_jobs, spatial_thread_cap)``. Does not affect the serial path.

    Returns:
        A_new: Updated sparse (H*W, K) footprints.
    """
    H, W = dims
    n_pixels, T = Y_flat.shape
    K = A.shape[1]

    support = compute_support(A, dims, dilation_radius)

    if n_jobs == 1:
        # Serial path — avoids joblib import overhead. Fixed 256-pixel batches
        # (kept byte-identical: the n_jobs=1 path is pinned bit-for-bit by
        # tests/test_stage_split.py).
        batch_size = 256
        batches = [
            (s, min(s + batch_size, n_pixels))
            for s in range(0, n_pixels, batch_size)
        ]
        per_batch = [
            _spatial_pixel_batch(
                start,
                Y_flat[start:end],
                C,
                support[start:end],
                sn[start:end],
                T,
                max_iter,
                tol,
                spatial_ridge,
            )
            for start, end in batches
        ]
    else:
        from joblib import Parallel, delayed, effective_n_jobs
        from threadpoolctl import threadpool_limits

        # Threads, not processes: the object passed as Y_flat is a lazy
        # BackgroundSubtractor referencing the full sparse W + full movie, so
        # loky would have to marshal those per task. The Cython CD solver
        # releases the GIL during its inner loop, so threads get real
        # concurrency on the solve without any pickling tax.
        #
        # Cap the worker count (see _SPATIAL_THREAD_CAP): the per-pixel loop is
        # GIL-bound, so >~16 threads thrash the GIL rather than help (observed
        # ~1 effective core at n_jobs=256). With the cap, size batches to ~4x
        # workers so dispatch overhead is small while a batch's dense slab
        # (batch_size*T*4 bytes) stays bounded (~64 MB cap).
        n_workers = max(1, min(effective_n_jobs(n_jobs), int(spatial_thread_cap)))
        batch_size = (n_pixels + 4 * n_workers - 1) // (4 * n_workers)
        batch_size = max(256, batch_size)
        max_batch_by_mem = max(256, (64 * 1024 * 1024) // (4 * max(int(T), 1)))
        batch_size = min(batch_size, max_batch_by_mem)
        batches = [
            (s, min(s + batch_size, n_pixels))
            for s in range(0, n_pixels, batch_size)
        ]

        # threadpool_limits caps inner BLAS threads to 1 for the duration
        # of the parallel section. Without this cap, on Linux each worker
        # thread's BLAS calls (the small Gram matmul, the CD inner solve)
        # try to spawn up to n_cores OpenMP threads, producing
        # n_workers * n_cores OS threads competing for n_cores.
        with threadpool_limits(limits=1, user_api="blas"):
            per_batch = Parallel(n_jobs=n_workers, prefer="threads")(
                delayed(_spatial_pixel_batch)(
                    start,
                    Y_flat[start:end],
                    C,
                    support[start:end],
                    sn[start:end],
                    T,
                    max_iter,
                    tol,
                    spatial_ridge,
                )
                for start, end in batches
            )

    all_results = [item for batch_res, _ in per_batch for item in batch_res]

    # Aggregate per-batch diagnostic counters (see _spatial_pixel_batch stats).
    total_unconverged = sum(s[0] for _, s in per_batch)
    total_ran = sum(s[1] for _, s in per_batch)
    total_iter = sum(s[2] for _, s in per_batch)
    max_iter_seen = max((s[3] for _, s in per_batch), default=0)
    total_active = sum(s[4] for _, s in per_batch)

    if total_ran > 0:
        mean_iter = total_iter / total_ran
        mean_active = total_active / total_ran
        # One always-on summary that distinguishes the three slowness causes:
        # convergence (mean_iter near max_iter), active-set size (mean_active
        # large -> heavy per-pixel Gram matmul), or sheer pixel count.
        print(
            f"  update_spatial stats: {total_ran}/{n_pixels} pixels ran CD; "
            f"mean_iter={mean_iter:.1f} max_iter_seen={max_iter_seen} "
            f"(cap={max_iter}); {total_unconverged} hit cap; "
            f"mean_active={mean_active:.1f}"
        )
    if total_unconverged > 0:
        print(
            f"  update_spatial: {total_unconverged}/{n_pixels} pixels hit "
            f"max_iter={max_iter} before converging (tol={tol}). Increase "
            f"CNMFeParams.spatial_max_iter or loosen spatial_tol to tighten."
        )

    # Accumulate per-component pixel values
    new_data: dict[int, dict[int, float]] = {k: {} for k in range(K)}
    for p_global, k, val in all_results:
        new_data[k][p_global] = val

    # Clean each component's footprint. Each component is independent and
    # threshold_footprint releases the GIL (scipy.ndimage), so parallelize
    # across K — this is otherwise a serial single-core stretch that grows
    # with component count and FOV.
    ks = [k for k in range(K) if new_data[k]]
    pixel_ids = {
        k: np.fromiter(new_data[k].keys(), dtype=np.int32, count=len(new_data[k]))
        for k in ks
    }
    values = {
        k: np.fromiter(new_data[k].values(), dtype=np.float32, count=len(new_data[k]))
        for k in ks
    }

    if n_jobs == 1:
        cleaned = [
            _threshold_one_component(
                k, pixel_ids[k], values[k], n_pixels, dims,
                max_thr, closing_radius, circular_max_dist_factor,
            )
            for k in ks
        ]
    else:
        from joblib import Parallel, delayed
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1, user_api="blas"):
            cleaned = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_threshold_one_component)(
                    k, pixel_ids[k], values[k], n_pixels, dims,
                    max_thr, closing_radius, circular_max_dist_factor,
                )
                for k in ks
            )

    # Build new sparse matrix (COO assembly is order-independent).
    rows_all: list[np.ndarray] = []
    cols_all: list[np.ndarray] = []
    data_all: list[np.ndarray] = []
    for k, nz, vals_nz in cleaned:
        if len(nz) == 0:
            continue
        rows_all.append(nz)
        cols_all.append(np.full(len(nz), k, dtype=np.int32))
        data_all.append(vals_nz)

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
