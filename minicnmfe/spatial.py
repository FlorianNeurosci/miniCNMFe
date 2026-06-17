"""Spatial footprint update via per-pixel non-negative LASSO regression.

For each pixel p, the data is modelled as:
    Y[p, :] = sum_{k in active(p)} A[p, k] * C[k, :] + noise

where active(p) is the set of components whose footprints overlap pixel p
(within a dilation of the current footprint). This small local set makes the
regression fast and well-conditioned.

Reference (algorithmic only): CaImAn spatial.py:update_spatial_components (line 29).
"""

from __future__ import annotations

import math
import time
import warnings

import numpy as np
import scipy.ndimage as ndi
import scipy.sparse as sp
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model._cd_fast import enet_coordinate_descent_gram

# Optional numba acceleration for the parallel (n_jobs>1) per-pixel CD. The
# Python per-pixel loop is GIL-bound, so joblib threads can't parallelise it on
# a many-core box; a numba njit(parallel=True) prange kernel runs the CD in
# compiled nogil code and uses every core regardless of the GIL or loky nesting.
# Graceful fallback to the threaded path if numba is absent (pip-only dep).
try:  # pragma: no cover - exercised by the env that has numba
    import numba
    from numba import njit, prange

    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    _HAS_NUMBA = False

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
# numba nogil/prange per-pixel CD kernel (GIL-free parallelism)
# ---------------------------------------------------------------------------
# Mirrors the math of `_spatial_pixel_batch` exactly: per pixel build the small
# Gram (C_active @ C_active.T) and q (C_active @ y), then run the non-negative
# elastic-net coordinate descent that sklearn's `enet_coordinate_descent_gram`
# implements (L1 alpha = lam*T, L2 beta = ridge*max_energy, positive, cyclic,
# H = Gram@w maintained incrementally). Because it is compiled nogil code with a
# prange over pixels, it scales across all cores irrespective of the GIL — which
# the threaded path cannot, the per-pixel work being mostly GIL-held Python glue.
if _HAS_NUMBA:

    @njit(parallel=True, cache=True, fastmath=False)
    def _spatial_cd_kernel(
        C, Y_block, sn_block, supp_idx, supp_off, T, max_iter, tol, ridge,
        lambda_scale, out_coef, out_iter, out_ran, out_unconv, out_active,
    ):  # pragma: no cover - compiled
        n = Y_block.shape[0]
        for i in prange(n):
            a0 = supp_off[i]
            a1 = supp_off[i + 1]
            na = a1 - a0
            out_iter[i] = 0
            out_ran[i] = 0
            out_unconv[i] = 0
            out_active[i] = 0
            if na == 0:
                continue

            gram = np.zeros((na, na))
            q = np.zeros(na)
            for a in range(na):
                ia = supp_idx[a0 + a]
                qa = 0.0
                for t in range(T):
                    qa += C[ia, t] * Y_block[i, t]
                q[a] = qa
                for b in range(a, na):
                    ib = supp_idx[a0 + b]
                    g = 0.0
                    for t in range(T):
                        g += C[ia, t] * C[ib, t]
                    gram[a, b] = g
                    gram[b, a] = g

            max_energy = 0.0
            for a in range(na):
                if gram[a, a] > max_energy:
                    max_energy = gram[a, a]
            me = max_energy if max_energy > 1e-10 else 1e-10
            lam = lambda_scale * 0.5 * sn_block[i] * math.sqrt(me) / T
            if lam == 0.0:
                continue
            alpha = lam * T
            beta = ridge * max_energy

            w = np.zeros(na)
            H = np.zeros(na)  # H = gram @ w, starts at 0 (w = 0)
            n_iter = 0
            for it in range(max_iter):
                w_max = 0.0
                d_w_max = 0.0
                for ii in range(na):
                    Qii = gram[ii, ii]
                    if Qii == 0.0:
                        continue
                    w_ii = w[ii]
                    if w_ii != 0.0:
                        for j in range(na):
                            H[j] -= w_ii * gram[j, ii]
                    tmp = q[ii] - H[ii]
                    if tmp < 0.0:
                        neww = 0.0
                    else:
                        neww = tmp - alpha
                        if neww < 0.0:
                            neww = 0.0
                        neww = neww / (Qii + beta)
                    w[ii] = neww
                    if neww != 0.0:
                        for j in range(na):
                            H[j] += neww * gram[j, ii]
                    d = abs(neww - w_ii)
                    if d > d_w_max:
                        d_w_max = d
                    aw = abs(neww)
                    if aw > w_max:
                        w_max = aw
                n_iter = it + 1
                if w_max == 0.0 or (d_w_max / w_max) < tol:
                    break

            out_iter[i] = n_iter
            out_ran[i] = 1
            out_active[i] = na
            if n_iter >= max_iter:
                out_unconv[i] = 1
            for a in range(na):
                out_coef[a0 + a] = w[a]


def _materialize_slab(Y_flat, sn, rows, tsub=1):
    """Materialize one (len(rows), T) background-subtracted slab + its sn.

    This is the expensive bit of update_spatial on real data: reading the slab
    triggers BackgroundSubtractor.slice_rows -> a GIL-releasing sparse W@Y matmul
    (the ring-background subtraction). Run under joblib threads it parallelises.

    ``rows`` is the array of *global* pixel indices for this block — only pixels
    with non-empty support (the ones that will run CD) are passed, so the
    background subtraction is never wasted on the ~70% of empty-support pixels.
    Falls back to fancy indexing for a plain-ndarray Y_flat (used in tests).
    """
    if hasattr(Y_flat, "slice_rows"):
        sub = Y_flat.slice_rows(rows, tsub)           # subsample at the source (W@Y over T/tsub)
    else:
        sub = Y_flat[rows][:, ::tsub] if tsub > 1 else Y_flat[rows]
    Y_block = np.ascontiguousarray(sub, dtype=np.float64)
    sn_block = np.ascontiguousarray(sn[rows], dtype=np.float64)
    return Y_block, sn_block


def _spatial_numba_update(Y_flat, C, support, sn, T, n_pixels, n_jobs,
                          max_iter, tol, ridge, n_workers, lambda_scale=1.0,
                          tsub=1):
    """Per-pixel CD via the numba kernel, with THREADED slab materialization.

    The dominant cost on real (mean_active~2) data is not the CD but the
    background-subtraction slab (Y_flat[start:end] = sparse W@Y). That matmul
    releases the GIL, so we materialize slabs in **waves of n_workers via joblib
    threads** (bounded memory: ~n_workers * block * T * 8 bytes), then run the
    numba kernel on each slab between waves (the kernel itself pranges over the
    block's pixels using the same thread budget — cheap when mean_active is low,
    parallel when the FOV is dense). Prints a slab-vs-cd timing breakdown.
    Returns ``(all_results, totals)`` matching ``_aggregate_spatial_batches``.
    """
    from joblib import Parallel, delayed
    from threadpoolctl import threadpool_limits

    n_workers = max(1, min(int(n_workers), int(numba.config.NUMBA_NUM_THREADS)))
    numba.set_num_threads(n_workers)

    # spatial_tsub: solve the per-pixel LASSO on a time-subsample. The kernel's
    # gram/q/max_energy all scale ~1/tsub but the L1 penalty (alpha) only
    # ~1/sqrt(tsub), so correct lambda_scale by 1/sqrt(tsub) to keep the same
    # footprints (validated by live_runs/spatial_tsub_probe.py: coef cos ~0.9999).
    # Default tsub=1 -> Csub is C, lam_eff is lambda_scale -> bit-for-bit.
    Csub = C[:, ::tsub] if tsub > 1 else C
    C64 = np.ascontiguousarray(Csub, dtype=np.float64)   # (K, T_eff), read-only
    T_eff = C64.shape[1]
    lam_eff = lambda_scale / (tsub ** 0.5) if tsub > 1 else lambda_scale

    # Only pixels with non-empty support run CD and contribute to A_new; the
    # other ~70% would have their (expensive) background-subtracted slab
    # materialised for nothing. Restrict the whole slab/CD loop to those
    # support pixels -- bit-exact (the skipped pixels produce no footprint
    # value either way), ~5x less slab work on real data.
    supp_lens = np.fromiter((len(s) for s in support), dtype=np.int64, count=n_pixels)
    active_pix = np.flatnonzero(supp_lens > 0)
    n_active = int(active_pix.size)

    all_results: list[tuple[int, int, float]] = []
    total_unconverged = total_ran = total_iter = max_iter_seen = total_active = 0
    slab_wall = cd_wall = 0.0

    if n_active == 0:
        print(f"  update_spatial timing: slab={slab_wall:.1f}s cd={cd_wall:.1f}s "
              f"(x{n_workers} threads, 0 blocks)")
        totals = (total_unconverged, total_ran, total_iter, max_iter_seen, total_active)
        return all_results, totals

    # Small blocks so a wave of n_workers slabs stays bounded (~n_workers*32 MB).
    # Blocks index into `active_pix`; each block's *global* pixel rows are
    # `active_pix[bs:be]`.
    block_px = max(256, int(32 * 1024 * 1024) // (8 * max(int(T), 1)))
    blocks = [(bs, min(bs + block_px, n_active))
              for bs in range(0, n_active, block_px)]

    for ws in range(0, len(blocks), n_workers):
        wave = blocks[ws:ws + n_workers]
        wave_rows = [active_pix[bs:be] for bs, be in wave]

        # Phase 1: materialize this wave's slabs in parallel (GIL-released W@Y).
        t0 = time.perf_counter()
        with threadpool_limits(limits=1, user_api="blas"):
            slabs = Parallel(n_jobs=n_workers, prefer="threads")(
                delayed(_materialize_slab)(Y_flat, sn, rows, tsub) for rows in wave_rows
            )
        slab_wall += time.perf_counter() - t0

        # Phase 2: run the (cheap, or prange-parallel) CD kernel on each slab.
        t1 = time.perf_counter()
        for rows, (Y_block, sn_block) in zip(wave_rows, slabs):
            nb = rows.size
            supp = [support[p] for p in rows]
            lens = np.fromiter((len(s) for s in supp), dtype=np.int64, count=nb)
            supp_off = np.zeros(nb + 1, dtype=np.int64)
            np.cumsum(lens, out=supp_off[1:])
            tot = int(supp_off[-1])
            supp_idx = (np.concatenate([np.asarray(s, dtype=np.int64) for s in supp])
                        if tot > 0 else np.empty(0, dtype=np.int64))

            out_coef = np.zeros(tot, dtype=np.float64)
            out_iter = np.zeros(nb, dtype=np.int64)
            out_ran = np.zeros(nb, dtype=np.int64)
            out_unconv = np.zeros(nb, dtype=np.int64)
            out_active = np.zeros(nb, dtype=np.int64)

            # Y_block already comes back at T_eff (subsampled in _materialize_slab).
            _spatial_cd_kernel(
                C64, Y_block, sn_block, supp_idx, supp_off, int(T_eff),
                int(max_iter), float(tol), float(ridge), float(lam_eff),
                out_coef, out_iter, out_ran, out_unconv, out_active,
            )

            total_unconverged += int(out_unconv.sum())
            total_ran += int(out_ran.sum())
            total_iter += int(out_iter.sum())
            if out_iter.size and int(out_iter.max()) > max_iter_seen:
                max_iter_seen = int(out_iter.max())
            total_active += int(out_active.sum())

            for i in range(nb):
                a0 = int(supp_off[i])
                a1 = int(supp_off[i + 1])
                p_global = int(rows[i])
                for a in range(a0, a1):
                    val = float(out_coef[a])
                    if val > 0.0:
                        all_results.append((p_global, int(supp_idx[a]), val))
        cd_wall += time.perf_counter() - t1

    print(f"  update_spatial timing: slab={slab_wall:.1f}s cd={cd_wall:.1f}s "
          f"(x{n_workers} threads, {len(blocks)} blocks)")
    totals = (total_unconverged, total_ran, total_iter, max_iter_seen, total_active)
    return all_results, totals


def _aggregate_spatial_batches(per_batch):
    """Flatten per-batch ``(results, stats)`` into ``(all_results, totals)``."""
    all_results = [item for batch_res, _ in per_batch for item in batch_res]
    total_unconverged = sum(s[0] for _, s in per_batch)
    total_ran = sum(s[1] for _, s in per_batch)
    total_iter = sum(s[2] for _, s in per_batch)
    max_iter_seen = max((s[3] for _, s in per_batch), default=0)
    total_active = sum(s[4] for _, s in per_batch)
    return all_results, (total_unconverged, total_ran, total_iter,
                         max_iter_seen, total_active)


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
    lambda_scale: float = 1.0,
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
        lambda_scale: Multiplier on the per-pixel L1 penalty
            ``lam = lambda_scale * 0.5 * sn_p * sqrt(max_energy) / T``. ``1.0``
            (default) is standard CNMF-E. ``>1`` raises the threshold a pixel's
            ``C_k·y`` must clear to be nonzero, so footprints come out tighter at
            the regression source (rather than only via post-hoc ``max_thr``
            zeroing) — useful for dense FOVs where footprints sprawl into
            neighbours.

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
            lam = lambda_scale * 0.5 * sn_p * np.sqrt(max(max_energy, 1e-10)) / T
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
    sigma: float | None = None,
    max_radius_factor: float = 0.0,
    thr_method: str = "max",
    nrg_thr: float = 0.9999,
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
        sigma: Estimated neuron radius (px). Only used together with
            ``max_radius_factor`` to set the absolute circular-constraint cap.
        max_radius_factor: NON-STANDARD. When ``> 0`` (and ``sigma`` is given),
            cap the circular-constraint clip distance at
            ``max_radius_factor * sigma`` px — an absolute physical bound on
            footprint radius that, unlike the area-derived radius, still bites
            once a footprint has sprawled. ``0`` (default) disables (bit-for-bit).
        thr_method: How step 2 zeroes faint pixels. ``"max"`` (default) =
            peak-relative (drop below ``max_thr * peak``, the original
            behaviour). ``"nrg"`` = energy thresholding (CaImAn
            ``thr_method='nrg'``): keep the brightest pixels whose summed
            ``a²`` reaches ``nrg_thr`` of the total. Energy thresholding drops
            dim skirts more cleanly (squaring discounts them), so it tightens
            low-contrast / sprawled footprints that ``"max"`` keeps.
        nrg_thr: Fraction of total footprint energy to retain when
            ``thr_method='nrg'``. ``0.9999`` (default, CaImAn's loose value)
            drops only the faintest pixels; tightening lives at ~0.90–0.97.

    Returns:
        Cleaned footprint, same shape as ai.
    """
    H, W = dims
    ai2d = ai.reshape(H, W).copy()

    ai2d = ndi.median_filter(ai2d, size=3)
    ai2d = ai2d.clip(0)

    if ai2d.max() == 0:
        return np.zeros(H * W, dtype=np.float32)

    if thr_method == "nrg":
        # Energy thresholding (CaImAn thr_method='nrg'): keep the smallest set of
        # brightest pixels whose summed L2 energy (Σ a²) reaches nrg_thr of the
        # total, zero the rest. Squaring discounts the dim skirt, so it is
        # dropped cleanly — unlike the peak-relative ``max_thr`` step, which
        # keeps a low-contrast (sprawled) footprint's broad skirt.
        flat = ai2d.ravel().copy()
        order = np.argsort(flat)[::-1]
        csum = np.cumsum(flat[order] ** 2)
        total = float(csum[-1])
        if total > 0:
            cutoff = int(np.searchsorted(csum, nrg_thr * total))
            cutoff = min(cutoff, flat.size - 1)
            keep = np.zeros(flat.size, dtype=bool)
            keep[order[: cutoff + 1]] = True
            flat[~keep] = 0.0
            ai2d = flat.reshape(H, W)
    else:
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
        # Absolute radius cap (in px) from the physical neuron-size prior. The
        # area-derived radius grows with a sprawled footprint, so without this
        # cap the constraint barely clips already-bloated footprints in dense
        # FOVs; `factor * sigma` bounds the clip distance to a fixed physical
        # radius. None = off (pure area-derived, bit-for-bit).
        max_radius = (
            float(max_radius_factor) * float(sigma)
            if (max_radius_factor > 0 and sigma)
            else None
        )
        ai2d = circular_constraint(
            ai2d, max_dist_factor=float(circular_max_dist_factor),
            max_radius=max_radius,
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
    sigma: float | None = None,
    max_radius_factor: float = 0.0,
    thr_method: str = "max",
    nrg_thr: float = 0.9999,
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
        sigma=sigma, max_radius_factor=max_radius_factor,
        thr_method=thr_method, nrg_thr=nrg_thr,
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
    lambda_scale: float = 1.0,
    sigma: float | None = None,
    max_radius_factor: float = 0.0,
    thr_method: str = "max",
    nrg_thr: float = 0.9999,
    spatial_tsub: int = 1,
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
        lambda_scale: Multiplier on the per-pixel LASSO penalty (forwarded to
            the CD). ``1.0`` (default) = standard; ``>1`` yields tighter
            footprints at the regression source. See ``_spatial_pixel_batch``.
        sigma: Estimated neuron radius (px); forwarded to ``threshold_footprint``
            for the absolute circular-constraint cap (only used with
            ``max_radius_factor``).
        max_radius_factor: When ``> 0`` (and ``sigma`` given), cap each
            footprint's circular-constraint radius at ``max_radius_factor *
            sigma`` px. ``0`` (default) disables. Helps dense/long FOVs where
            footprints sprawl past their area-derived radius.
        thr_method: ``"max"`` (default, peak-relative) or ``"nrg"`` (energy
            thresholding); forwarded to ``threshold_footprint``.
        nrg_thr: Energy fraction retained when ``thr_method='nrg'`` (default
            ``0.9999``). See ``threshold_footprint``.

    Returns:
        A_new: Updated sparse (H*W, K) footprints.
    """
    H, W = dims
    n_pixels, T = Y_flat.shape
    K = A.shape[1]

    support = compute_support(A, dims, dilation_radius)

    spatial_path = "serial"   # diagnostic: which CD path ran (see stats print)
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
                lambda_scale,
            )
            for start, end in batches
        ]
        all_results, totals = _aggregate_spatial_batches(per_batch)
    elif _HAS_NUMBA:
        from joblib import effective_n_jobs

        # numba prange kernel: the per-pixel CD runs in compiled nogil code, so
        # it scales across all cores irrespective of the GIL (the threaded path
        # below cannot — the per-pixel loop is GIL-bound). numba is not GIL-bound
        # so the _SPATIAL_THREAD_CAP (a GIL-thrash mitigation) does not apply;
        # use the full inner budget. set_num_threads bounds it so candidates
        # nested in loky processes don't oversubscribe (each pranges over its
        # own inner_jobs share). Clamp to NUMBA_NUM_THREADS (the launchable max,
        # = cpu_count) — set_num_threads rejects anything larger.
        n_workers = max(1, min(effective_n_jobs(n_jobs),
                               int(numba.config.NUMBA_NUM_THREADS)))
        spatial_path = f"numba x{n_workers} (slab-parallel)"
        all_results, totals = _spatial_numba_update(
            Y_flat, C, support, sn, T, n_pixels, n_jobs,
            max_iter, tol, spatial_ridge, n_workers, lambda_scale, spatial_tsub,
        )
    else:
        from joblib import Parallel, delayed, effective_n_jobs
        from threadpoolctl import threadpool_limits

        # Threaded fallback (numba unavailable). Threads, not processes: the
        # object passed as Y_flat is a lazy BackgroundSubtractor referencing the
        # full sparse W + full movie, so loky would have to marshal those per
        # task. The Cython CD solver releases the GIL during its inner loop, but
        # the surrounding per-pixel Python glue does not, so this path is
        # GIL-bound — cap the worker count (see _SPATIAL_THREAD_CAP): >~16
        # threads thrash the GIL rather than help. Size batches to ~4x workers so
        # dispatch overhead is small while a batch's dense slab stays bounded.
        n_workers = max(1, min(effective_n_jobs(n_jobs), int(spatial_thread_cap)))
        spatial_path = f"threaded x{n_workers} GIL-bound (pip install numba to scale)"
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
                    lambda_scale,
                )
                for start, end in batches
            )
        all_results, totals = _aggregate_spatial_batches(per_batch)

    total_unconverged, total_ran, total_iter, max_iter_seen, total_active = totals

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
            f"mean_active={mean_active:.1f} [{spatial_path}]"
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
                sigma, max_radius_factor, thr_method, nrg_thr,
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
                    sigma, max_radius_factor, thr_method, nrg_thr,
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
