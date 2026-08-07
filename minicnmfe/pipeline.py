"""High-level CNMFe pipeline.

Orchestrates motion correction → preprocessing → initialization →
ring background → iterative spatial/temporal refinement.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

from minicnmfe._utils import StageTimer, make_2d
from minicnmfe.background import BackgroundSubtractor, compute_W
from minicnmfe.evaluate import auto_evaluate_components
from minicnmfe.initialization import (
    _resolve_patch_workers,
    greedy_corr_pnr,
    greedy_corr_pnr_patched,
)
from minicnmfe.io import transpose_zarr_to_pixel_major
from minicnmfe.merging import merge_components
from minicnmfe.motion_correction import motion_correction_rigid
from minicnmfe.preprocess import correlation_pnr, estimate_noise
from minicnmfe.spatial import update_spatial
from minicnmfe.temporal import estimate_ar_params, update_temporal


def _zarr_store_path(arr) -> "str | None":
    """Best-effort: return the on-disk path backing ``arr`` (a zarr.Array).

    Returns None if the array is in-memory or has no resolvable path —
    callers should fall back to materialisation in that case.
    """
    store = getattr(arr, "store", None)
    if store is None:
        return None
    # zarr v3 LocalStore exposes .root; FsspecStore exposes .path.
    for attr in ("root", "path"):
        val = getattr(store, attr, None)
        if val is not None:
            return str(val)
    return None


def _sn_from_footprint(a_k, sn_flat: np.ndarray) -> float:
    """Closed-form noise std of the projected trace ``(a · Y) / ‖a‖²``.

    Assuming pixel noise is independent with per-pixel std ``sn_flat``
    (the assumption already baked into ``compute_W`` and
    ``update_spatial``), the projection is a fixed linear combination
    of pixel traces, so its variance is

        Var(proj) = (Σᵢ aᵢ² · sn_flat[i]²) / ‖a‖⁴
        sn_k      = ‖a · sn_flat‖₂ / ‖a‖²

    Used instead of ``estimate_ar_params(C_raw[k], ...)`` for the
    per-component noise std at init — that path runs PSD on the
    center-surround-filtered ``C_raw`` whose high-frequency content
    has been smoothed out, returning a near-zero ``sn`` that drives
    OASIS into collapse.
    """
    if sp.issparse(a_k):
        a_k = a_k.toarray().ravel()
    else:
        a_k = np.asarray(a_k).ravel()
    aa = float(a_k @ a_k)
    if aa <= 0:
        return 1.0
    weighted = a_k.astype(np.float32) * sn_flat.astype(np.float32)
    return float(np.sqrt(np.sum(weighted ** 2)) / aa)


def _normalize_to_trace_amplitude(A, C, S, YrA, C_raw, sn_per_k):
    """Relabel the ``A·C`` factorization into CaImAn's scale convention.

    CNMF-E factorizes ``Y ≈ A·C``, which is invariant under
    ``A[:,k] *= s_k`` / ``C[k] /= s_k``. minicnmfe's init parks the
    per-component gain in the footprints (large ``A``, unit-L2-norm traces);
    CaImAn parks it in the traces (unit-L2-norm footprints, amplitude in
    ``C``). This moves the gain into the traces:

        s_k = ‖A[:,k]‖₂
        A[:,k] /= s_k ;  C[k] *= s_k ;  S[k] *= s_k ;
        YrA[k] *= s_k ;  C_raw[k] *= s_k ;  sn_per_k[k] *= s_k

    ``A·C`` (and every scale-invariant quantity — correlations, SNR, spike
    timing) is unchanged. Returns ``(A, C, S, YrA, C_raw, sn_per_k, s)``
    where ``s`` is the per-component original L2 norm — keep it so the
    auto-eval SNR (which depends on ‖a_k‖²) can be reconstructed from the
    now-unit-norm footprints (see ``CNMFe.evaluate``).
    """
    K = A.shape[1]
    if K == 0:
        return A, C, S, YrA, C_raw, sn_per_k, np.zeros(0, dtype=np.float32)

    s = np.sqrt(np.asarray(A.power(2).sum(axis=0)).ravel()).astype(np.float32)
    s_safe = np.where(s > 0, s, 1.0).astype(np.float32)

    A = (A.tocsc() @ sp.diags(1.0 / s_safe)).tocsc()
    if C is not None:
        C = C * s_safe[:, None]
    if S is not None:
        S = S * s_safe[:, None]
    if YrA is not None:
        YrA = YrA * s_safe[:, None]
    if C_raw is not None and C_raw.shape[0] == K:
        C_raw = C_raw * s_safe[:, None]
    if sn_per_k is not None:
        sn_per_k = np.asarray(sn_per_k) * s_safe
    return A, C, S, YrA, C_raw, sn_per_k, s


def _Y_times_vec(Y_flat, v, axis: int) -> np.ndarray:
    """Compute ``Y_flat @ v`` (axis=1, output shape (H*W,)) or ``Y_flat.T @ v``
    (axis=0, output shape (T,)) without materialising a zarr ``Y_flat`` in full.
    """
    if isinstance(Y_flat, np.ndarray):
        return (Y_flat @ v if axis == 1 else Y_flat.T @ v).astype(np.float32)
    n_pix = int(Y_flat.shape[0])
    T = int(Y_flat.shape[1])
    batch = 4096
    if axis == 1:
        out = np.zeros(n_pix, dtype=np.float32)
        for s in range(0, n_pix, batch):
            e = min(s + batch, n_pix)
            out[s:e] = np.asarray(Y_flat[s:e], dtype=np.float32) @ v
        return out
    out = np.zeros(T, dtype=np.float32)
    for s in range(0, n_pix, batch):
        e = min(s + batch, n_pix)
        out += np.asarray(Y_flat[s:e], dtype=np.float32).T @ v[s:e]
    return out


def _yflat_proj_batch(Y_flat, A_csr, s: int, e: int) -> np.ndarray:
    """One pixel-batch contribution to ``Y_flat.T @ A`` → ``(T, K)``.

    Module-level for joblib pickling. Dispatched with ``prefer="threads"``;
    each batch reads a zarr slab and does a GIL-releasing BLAS matmul.
    """
    Y_chunk = np.asarray(Y_flat[s:e], dtype=np.float32)
    return np.asarray(Y_chunk.T @ A_csr[s:e], dtype=np.float32)


def _fit_global_bg_rank1(
    Y_flat,
    A: "sp.csc_matrix",
    C: np.ndarray,
    W_mat: "sp.csr_matrix",
    b0: np.ndarray,
    bf: "np.ndarray | None",
    f: "np.ndarray | None",
    n_iter: int = 2,
) -> "tuple[np.ndarray, np.ndarray]":
    """Alternating LS for the rank-1 temporal background ``b_f · f(t)``.

    Fits ``R ≈ b_f · f(t)`` where
        R = (I − W)(Y − b0) − A C
    is the ring-subtracted, neural-signal-subtracted residual. ``bf`` and ``f``
    are unconstrained (drift can have either sign per pixel relative to b0);
    standard CNMF non-negativity would force a monotonic-only background and
    is not appropriate when modelling a *deviation* from b0.

    On first call pass ``bf=None, f=None`` for init via the per-frame mean of
    ``Y_bg``; subsequent calls warm-start from the previous values.

    Algebra (avoids materialising R):

        u  = (I − W^T) bf
        f  = (Y^T u  −  u·b0  −  (bf·A) C) / (bf·bf)

        v  = Y f − b0·(f.sum())
        bf = ((I − W) v  −  A (C f)) / (f·f)

    All terms use sparse matvec + dense matvec; cost per iteration is
    O(H·W·T) (the ``Y @ f`` and ``Y.T @ u`` calls) plus O(H·W·K + K·T) for
    the A/C terms. Streams cleanly on zarr Y_flat via ``_Y_times_vec``.
    """
    n_pix = int(Y_flat.shape[0])
    T = int(Y_flat.shape[1])
    A_csr = A.tocsr() if sp.issparse(A) else sp.csr_matrix(A)

    if bf is None or f is None:
        # Init: f_0 = per-frame mean of Y_bg (drift signal).
        # mean_p (I-W)(Y-b0) = (1/HW) · (1 − W^T 1)^T · (Y − b0)
        ones = np.ones(n_pix, dtype=np.float32) / float(n_pix)
        u0 = ones - (W_mat.T @ ones)
        f = _Y_times_vec(Y_flat, u0, axis=0) - float(u0 @ b0)
        f = (f - f.mean()).astype(np.float32)
        if not np.any(f):
            return (np.zeros(n_pix, dtype=np.float32),
                    np.zeros(T, dtype=np.float32))
        # bf_0: LS projection of Y_bg onto f.
        # bf = ((I-W)(Y - b0) f) / (f·f) — neural term skipped on init.
        v = _Y_times_vec(Y_flat, f, axis=1) - b0 * float(f.sum())
        bf = (v - (W_mat @ v)).astype(np.float32) / max(float(f @ f), 1e-10)

    for _ in range(n_iter):
        # Update f given bf
        u = (bf - (W_mat.T @ bf)).astype(np.float32)
        yTu = _Y_times_vec(Y_flat, u, axis=0)
        ub0 = float(u @ b0)
        if A_csr.shape[1] > 0:
            bfA = np.asarray(bf @ A_csr, dtype=np.float32)
            bfA_C = bfA @ C
        else:
            bfA_C = np.zeros(T, dtype=np.float32)
        denom_f = max(float(bf @ bf), 1e-10)
        f = ((yTu - ub0 - bfA_C) / denom_f).astype(np.float32)

        # Update bf given f
        v = _Y_times_vec(Y_flat, f, axis=1) - b0 * float(f.sum())
        Iv_minus_Wv = v - (W_mat @ v)
        if A_csr.shape[1] > 0:
            ACf = np.asarray(A_csr @ (C @ f), dtype=np.float32)
        else:
            ACf = np.zeros(n_pix, dtype=np.float32)
        denom_b = max(float(f @ f), 1e-10)
        bf = ((Iv_minus_Wv - ACf) / denom_b).astype(np.float32)

    return bf, f

if TYPE_CHECKING:
    import zarr


@dataclass
class CNMFeParams:
    """All CNMFe algorithm parameters.

    Deviations from standard CNMF-E (Zhou et al. 2018) are tagged
    ``[NON-STANDARD]`` next to the field. With every ``[NON-STANDARD]``
    *algorithm* flag at its default, the math matches the published CNMF-E.
    Fields tagged ``[NON-STANDARD speed]`` are speed/memory trade-offs
    (subsampling, parallelism, etc.) that converge to the standard
    algorithm at their slow-but-faithful settings (``tsub=1``,
    ``stride=1``, ``skip_first_deconv=False``, ``sample_frames=T``).
    Implementation mechanics like ``n_jobs``, ``device``, and ``mc_*``
    are not tagged.
    """

    # --- Motion correction ---
    max_shift: tuple[int, int] = (20, 20)
    upsample_factor: int = 10
    mc_n_iter: int = 1                 # number of rigid MC passes (max passes when mc_converge_tol is set)
    mc_converge_tol: "float | None" = None  # if set (e.g. 0.01), stop MC passes early once template sharpness gain falls below it (GT-free convergence; mc_n_iter is then just the cap)
    mc_sharpen_template: bool = True   # build the MC template by aligning the in-RAM frame sample (recovers full drift amplitude even at mc_n_iter=1, ~one-pass cost on long movies); False = legacy smeared-median template
    mc_gSig_filt: float | None = None  # 1p high-pass sigma; set ≈ sigma to enable
    mc_batch_size: int = 200           # frames per streaming/parallel batch in MC
    mc_template_max_frames: int = 2000 # cap on frames sampled to build the MC template
    mc_output_chunk_t: "int | None" = None   # output zarr chunk on time axis (None = match source)
    mc_output_dtype: str = "float32"   # output zarr dtype for the corrected movie

    # --- Spatial filtering / PSF ---
    sigma: float = 3.0        # Gaussian sigma in pixels (neuron size)
    center_psf: bool = True   # Use center-surround kernel for 1p background rejection

    # --- Initialization (GreedyCorr) ---
    min_corr: float = 0.8
    min_pnr: float = 10.0
    min_pixel: int = 3        # Minimum nonzero pixels in a valid footprint
    border_px: int = 5        # Ignore seeds within this many border pixels
    max_neurons: int | None = None  # Stop early (None = no limit)
    init_min_corr_neuron: float = 0.8         # "Neuron pixel" threshold inside extract_spatial_temporal
    init_max_corr_bg: float = 0.4             # "Background pixel" threshold inside extract_spatial_temporal
    seed_suppress_factor: float = 2.0         # DEPRECATED no-op (kept for API stability). Greedy init now suppresses found neurons with an `ind_search` support mask (ai > ai.max()/2), not a disk; see initialization.greedy_corr_pnr.
    circular_max_dist_factor: float = 2.5     # circular_constraint cutoff = factor * estimated_radius
    # [NON-STANDARD speed] Temporal stride for greedy init (None = auto: max(1, T//5000)).
    init_stride: "int | None" = None
    # [NON-STANDARD speed] Extra stride for the CN (correlation) part of the
    # initial seed sweep inside greedy init. PNR is NEVER strided (it is a
    # peak/max over the full-T noise — striding it under-estimates PNR and
    # starves seeding to ~0 on long movies). None -> 1 (no stride): we do not
    # trade seeds for speed by default. Set > 1 to opt into a faster CN sweep.
    init_corrpnr_stride: "int | None" = None
    # [speed] Patch-based PARALLEL initialization (ON by default). The greedy
    # seed loop is inherently serial, so for a large FOV we tile it into
    # overlapping spatial patches run in parallel processes and merge border
    # duplicates (validated: seeds land on the same cells as the serial path,
    # ~3x faster at n_jobs=8, more with more cores). Auto-skips when it can't
    # help or could hurt: small FOV (min(H,W) < init_patch_min_fov), GPU
    # (device != cpu), a streaming/zarr movie (patches materialise, so in-RAM
    # only -> no OOM), and nested parallelism (the tuning sweep sets this False
    # for its candidates, since loky can't nest and candidates already run in
    # parallel). Set False to force the bit-for-bit serial greedy.
    seed_method: str = "mixture"       # 'mixture' (self-calibrating local-maxima seeds) | 'threshold' (legacy min_corr/min_pnr gate)
    init_patches: bool = True
    init_patch_size: "int | None" = None     # patch side px; None -> max(int(12*sigma), 48)
    init_patch_overlap: "int | None" = None  # overlap px; None -> int(4*sigma) (> patch_radius ~3*sigma)
    init_patch_min_fov: int = 128            # only tile when min(H, W) >= this; else global init
    init_patch_n_jobs: "int | None" = None   # patch workers; None -> n_jobs
    init_patch_max_workers: int = 32         # upper bound on patch-init loky processes
                                             # (caps process-count RAM on many-core
                                             # boxes; raise to use more, or set huge
                                             # to disable). Results are unchanged.

    # --- Background (ring model) ---
    ring_size_factor: float = 1.5  # ring radius = ring_size_factor * (2*sigma+1)
    ring_lambda: float = 1e-5      # Ridge regularization for ring regression
    # [NON-STANDARD] Enforce Σ_j W[i, j] = 1 per row of the ring weight matrix
    # (Lagrangian-constrained ridge LS). Cancels spatially-uniform brightness
    # events (LED flicker, uniform photobleaching) exactly in the residual —
    # the unconstrained ring shrinks row sums below 1, so a fraction of every
    # global event leaks into the extracted traces. Standard CNMF-E uses
    # unconstrained ridge LS (i.e. False). Negligible cost (one extra RHS in
    # the same per-pixel solve).
    ring_constrain_sum: bool = False
    # [NON-STANDARD] Add a rank-1 temporal background ``b_f · f(t)`` on top of
    # the ring (CNMF-style ``nb=1``). Captures *spatially-non-uniform* slow
    # drift (vignetting-coupled photobleaching, scope warmup) that the ring
    # cannot — the constrained ring only cancels spatially-uniform modes.
    # ``0`` = standard CNMF-E (ring only). ``1`` = one rank-1 term, fit via
    # alternating non-negative LS in each BCD iteration. Higher ranks are
    # deferred — ``1`` covers the typical bleach pattern.
    global_bg_rank: int = 0

    # --- Spatial update ---
    # Neighbourhood (in pixels) around each component's current binary
    # footprint that's eligible to enter the per-pixel LASSO active set.
    # Was 3 historically; reduced to 2 to suppress crosstalk between
    # adjacent neurons (which otherwise produced thin "tendril" connections
    # in the extracted footprints — see tmp/output_simple_movie_vs_caiman.png).
    # Smaller = less crosstalk, slower footprint growth across BCD iterations.
    # Larger = faster correction of under-shot init footprints, more crosstalk.
    dilation_radius: int = 2
    spatial_max_thr: float = 0.1    # Zero footprint pixels below this fraction of the peak
    # [NON-STANDARD; matches CaImAn] Radius (in pixels) of the morphological
    # binary closing applied inside `threshold_footprint`, between the
    # max-threshold and the largest-connected-component extraction. Fills
    # 1-pixel gaps so a jagged LASSO support survives as one rounded blob
    # instead of being split into multiple disconnected pieces (only the
    # largest of which would be kept) — the cause of small fragmented
    # footprints when this step is skipped. ``0`` disables (recovers the
    # exact prior behaviour); ``1`` (default, 3×3 SE) is what CaImAn uses.
    spatial_close_radius: int = 1
    # Per-pixel LASSO coordinate-descent budget (sklearn's
    # `enet_coordinate_descent_gram`). With `spatial_ridge` (below) the CD
    # converges in tens of iterations even on real data, so this cap is rarely
    # reached; it is just a backstop. Increase it only if the pipeline logs
    # "N pixels hit max_iter ..." at the end of update_spatial (and prefer
    # raising `spatial_ridge` slightly first); loosening `spatial_tol` also
    # speeds convergence at the cost of LASSO tightness.
    spatial_max_iter: int = 1000
    spatial_tol: float = 1e-4
    # [PERF/STABILITY] Elastic-net L2 fraction for the per-pixel solve. The
    # solver's beta is set to `spatial_ridge * max(diag(Gram))`, bounding the
    # condition number of the per-pixel Gram (`C_active @ C_active.T`) to
    # ~`1/spatial_ridge`. This is what keeps the CD converging when active
    # components have correlated/near-duplicate traces (a near-singular Gram) —
    # the cause of pixels running to thousands of iterations / not converging on
    # real recordings. Shrinkage on the coefficients is ~`spatial_ridge` (≈1% at
    # the default), negligible and confined to the degenerate components it
    # stabilises. Set ``0.0`` to restore pure LASSO (pre-change behaviour).
    spatial_ridge: float = 1e-2
    # [PERF] Max worker threads for the parallel per-pixel CD in update_spatial.
    # That loop is GIL-bound, so handing it the full core budget (e.g. n_jobs=256
    # on a big server) thrashes the GIL and collapses to ~1 effective core; the
    # effective worker count is min(n_jobs, spatial_thread_cap). 16 is a sweet
    # spot. Does not affect the serial (n_jobs=1) path or extracted results.
    spatial_thread_cap: int = 16
    # [NON-STANDARD; bandaid for LASSO spread] Apply `circular_constraint`
    # (initialization.py:33-53) as the final step of `threshold_footprint`.
    # Clips pixels further than `factor * sqrt(area/pi)` from the footprint
    # centroid, suppressing thin tendril-shaped extensions toward
    # neighbouring neurons (visible in tmp/after_tendril_correction.png on
    # the 10-neuron simulator). The same prior is already used at greedy
    # init (`circular_max_dist_factor`, default 2.5), but **post-BCD-
    # refinement we know the footprint shape much better, so we use a
    # tighter default here** (1.5 ≈ 50% above the area-derived natural
    # radius). At factor=2.5 the cutoff is too generous to clip the short
    # residual tendrils that appear with `dilation_radius=2`.
    # Set ``0.0`` to disable (recovers pre-change behaviour, e.g. for
    # dendritic imaging where non-circular footprints are expected).
    spatial_circular_max_dist_factor: float = 1.5
    # [NON-STANDARD; dense-FOV footprint tightening] Multiplier on the per-pixel
    # LASSO penalty in update_spatial: lam = spatial_lambda_scale * 0.5 * sn_p *
    # sqrt(max_energy) / T. ``1.0`` (default) = standard CNMF-E. ``>1`` raises
    # the threshold a pixel's C_k·y must clear to be nonzero, so footprints come
    # out tighter **at the regression source** rather than only via the post-hoc
    # ``spatial_max_thr`` zeroing — the principled knob for dense FOVs where
    # footprints sprawl into neighbours (~1.5 a good starting point). 1.0 leaves
    # results bit-for-bit unchanged.
    spatial_lambda_scale: float = 1.0
    # [NON-STANDARD; dense-FOV footprint tightening] Cap the circular-constraint
    # clip radius (in threshold_footprint) at ``spatial_max_radius_factor *
    # sigma`` px. The default circular constraint derives its radius from the
    # footprint's own area, so once a footprint sprawls its radius grows too and
    # the constraint stops biting; this adds an absolute physical-radius cap that
    # still clips bloated footprints. ``0.0`` (default) = off (area-derived only,
    # bit-for-bit). ~2.0 recommended for dense/long recordings.
    spatial_max_radius_factor: float = 0.0
    # [CaImAn thr_method] How threshold_footprint zeroes faint pixels.
    # ``"nrg"`` (DEFAULT) = energy thresholding: keep the brightest pixels whose
    # summed a² reaches ``spatial_nrg_thr`` of the total. ``"max"`` = peak-relative
    # (drop below spatial_max_thr * peak, the legacy behaviour). Energy
    # thresholding drops dim skirts more cleanly (squaring discounts them), so it
    # tightens low-contrast / sprawled footprints that "max" keeps — the
    # shape-aware footprint-size control. Validated on a real dense FOV: at matched
    # K it gives smaller, better-separated footprints AND higher trace purity than
    # "max" (corr(C,C+YrA) 0.850->0.871), optimum ~0.95 (live_runs/nrg_compare.py).
    # NOTE the threshold_footprint() *function* default stays "max" (only this
    # CNMFeParams field defaults to nrg) — direct callers/tests are unaffected.
    spatial_thr_method: str = "nrg"
    # Energy fraction retained when spatial_thr_method="nrg". ``0.95`` (default) is
    # the validated sweet spot; higher (->0.9999, CaImAn's loose value) keeps more
    # of the skirt, lower (~0.90) tightens further but past ~0.90 over-tightens
    # (footprints clip real signal and fall below min_pixel). Ignored for "max".
    spatial_nrg_thr: float = 0.95
    # Time-subsample factor for the per-pixel footprint LASSO in update_spatial
    # (analogous to bg_tsub for compute_W). The slab (ring W@Y background
    # subtraction) is the bandwidth-bound bottleneck (~45% of extraction on a big
    # FOV) and is unhelpable by threads; subsampling time cuts it ∝ 1/tsub. The
    # LASSO penalty is auto-corrected by 1/sqrt(tsub) so footprints match full-T
    # (validated: coef cos ~0.9999 at tsub=2). NUMBA path only. Default 1 = off
    # (bit-for-bit). Try 2.
    spatial_tsub: int = 1

    # --- Temporal update / deconvolution ---
    ar_order: int = 1
    global_ar: bool = True  # True = one g estimated from pooled C_raw; False = per-neuron
    n_iter_temporal: int = 2
    # [NON-STANDARD knob, standard *value*] Shrinkage on the Yule-Walker AR
    # estimate. 0.96 is the historical CNMF-E default (Friedrich 2017) and
    # avoids over-estimating the decay on clean traces. The downside: on
    # data with slow-background contamination — *any* miniscope recording
    # with photobleach or non-stationary neuropil — Yule-Walker over-shoots
    # toward 1, and 0.96 clamps it there instead of correcting it. Drop to
    # 0.90 for fast indicators (GCaMP6f, GCaMP8) on bleach-heavy recordings.
    # Exposed per recording rather than tuned globally because the correct
    # value depends on the indicator's true τ and the recording's drift
    # characteristics, neither of which the algorithm can observe directly.
    fudge_factor: float = 0.96
    # Bayesian-prior path for `g`: when both `decay_time_ms` and
    # `frame_rate_hz` are set, derive
    #     g_target = exp(-1 / (frame_rate_hz * decay_time_ms / 1000))
    # and shrink the Yule-Walker estimate toward it:
    #     g = (1 - g_prior_weight) * g_yw + g_prior_weight * g_target
    # `fudge_factor` is skipped on the prior path (the prior already
    # encodes the physical bound). When either field is `None`, fall back
    # to the legacy `fudge_factor` shrinkage.
    #
    # Suggested decay_time_ms (single-AP τ, somatic):
    #   GCaMP6f ~140   jGCaMP7f ~160
    #   jGCaMP8f ~70   jGCaMP8m ~180   jGCaMP8s ~350
    #   GCaMP6s/7s ~1000
    # Values vary 1.5–2× with cell type, AP count, expression level.
    decay_time_ms: float | None = None
    frame_rate_hz: float | None = None
    # Shrinkage weight on the prior: 0 = pure Yule-Walker (legacy),
    # 1 = pin at g_target. Default 0.5 balances data fit against the
    # physical prior. Bump toward 1 when the recording has heavy slow
    # drift (Yule-Walker is upward-biased); leave at 0.5 otherwise.
    g_prior_weight: float = 0.5
    # [NON-STANDARD] Polynomial order subtracted from each trace before the
    # Yule-Walker autocorrelation that estimates `g`. A slow bleach trend
    # has lag-1 autocorrelation ≈ 1 and, when not detrended, pushes `g`
    # toward 1.0 — OASIS then explains the whole trace as one decaying tail
    # and the deconvolved C collapses to ~0 after the first transient.
    # Default 0 matches standard CNMF-E (mean-only centring). Set ≥1 when
    # traces carry a slow bleach component (order 2 absorbs typical
    # exponential photobleaching).
    ar_detrend_order: int = 0
    # [NON-STANDARD] Polynomial order subtracted from each component's trace
    # `YrA[:,k]/nA[k] + C[k]` immediately before OASIS, inside the BCD loop.
    # Standard CNMF-E feeds OASIS the raw projection and lets OASIS fit a
    # single-scalar median baseline; that cannot track a long slow drift,
    # so deconvolved spikes get suppressed. Set ≥1 to opt in. Default 0:
    # on activity-rich recordings the least-squares polynomial gets pulled
    # upward by the spike envelope, depressing the inter-spike baseline
    # below truth so OASIS reconstructs inflated transients, and the BCD
    # spatial update then propagates that distortion into A — visible as
    # overshoot in `C + YrA` vs ground truth.
    temporal_detrend_order: int = 0

    # --- Merging ---
    merge_thr_corr: float = 0.85
    merge_thr_overlap: float = 0.5  # Min Jaccard spatial overlap to consider merging
    # [NON-STANDARD] Centre-distance fallback for merging: pairs whose footprints
    # have ended up disjoint after threshold_footprint can still be merged if
    # their centres are within `factor * sigma`. Standard CNMF-E merges on spatial
    # overlap only. Disable by setting factor=0.
    merge_centre_dist_factor: float = 2.0

    # --- Main loop ---
    n_iter_main: int = 2  # Full spatial + temporal + merge cycles

    # --- Parallelism ---
    n_jobs: int = 1      # Workers for pixel-parallel steps (-1 = all CPUs)
    device: str = "cpu"  # 'cpu' or 'cuda' (requires CuPy + CUDA GPU)

    # --- Statistics sampling ---
    # [NON-STANDARD speed] Max frames used for noise + CORR/PNR (evenly sampled
    # when T > this). Standard CNMF-E uses all frames.
    sample_frames: int = 1000

    # --- Speed / accuracy trade-offs ---
    # [NON-STANDARD speed] Use NNLS (p=0) for the first temporal pass; OASIS on all
    # others. Saves the dominant cost on iteration 0 with no measurable accuracy hit.
    skip_first_deconv: bool = True
    # [NON-STANDARD speed] Temporal subsampling factor for the ring W solve. The b0
    # baseline still uses the full T; only the per-pixel BTB regression sees a strided
    # slice. Standard CNMF-E uses tsub=1.
    bg_tsub: int = 5

    # --- Streaming store layout (true T-streaming extraction only) ---
    # Control the pixel-major ``Y_flat`` store the auto-derive path builds when
    # you call ``fit_extract(zarr, output_dir=...)`` (or run_extract.py without
    # --in-memory). These only affect IO speed of the on-disk store — never the
    # extracted results. See minicnmfe.io.transpose_zarr_to_pixel_major.
    # yflat_dir: where to write Y_flat_pixel.zarr. None = under output_dir.
    #   Point at a LOCAL SSD/tmpfs to stage off a network mount (the transpose
    #   reads mc.zarr from the network once; all BCD passes then read locally).
    yflat_dir: "str | None" = None
    # yflat_pixel_chunk / yflat_time_chunk: dest chunk shape. time_chunk None =
    #   full T (one chunk per pixel row — best for the full-time read pattern).
    yflat_pixel_chunk: int = 512
    yflat_time_chunk: "int | None" = None
    # yflat_compression: blosc lz4+bitshuffle. Keep True on a network mount
    #   (fewer bytes over the wire); try False on a local SSD (no per-read
    #   decompression, IO-bound only — costs ~H*W*T*4 bytes of disk).
    yflat_compression: bool = True

    # --- Auto evaluation (post-BCD quality tagging — report-only, OPT-IN gate) ---
    # `evaluate()` ALWAYS computes per-component quality metrics into
    # ``model.eval_info`` (``snr_amp`` = mean(a^2)/mean(sn_pixel^2), a
    # scale-invariant SNR, and ``pixel_count``) for inspection. The acceptance
    # *gate* — flagging components in ``model.accepted_mask`` — is **off by
    # default** (`auto_eval_snr_amp_thr = 0.0` → every component accepted on the
    # SNR check; the `min_pixel` floor is the init floor and real cells clear it).
    # Rationale: with seed thresholds tuned to the noise floor you don't get
    # ghosts, so a default gate mostly produces false negatives (rejects real
    # dim cells). Ghost control belongs upstream in `min_corr`/`min_pnr`.
    # To opt back IN (e.g. loose-seeding workflows with ghosts): raise
    # `auto_eval_snr_amp_thr` (~3 separates real σ=3 cells from noise seeds)
    # and/or `min_pixel`, then filter with ``model.A[:, model.accepted_mask]``.
    # Nothing is ever removed from the model; the gate is purely a mask.
    auto_eval_snr_amp_thr: float = 0.0

    # --- Cutout (crop the movie before extraction; NATIVE coordinates) ---
    # Applied ONCE at ingestion, before motion correction (see minicnmfe/cutout.py).
    # All None (default) = no cutout = bit-for-bit unchanged behaviour.
    temporal_crop: "tuple[int, int] | None" = None       # (t0, t1), t1 exclusive
    spatial_crop: "tuple[int, int, int, int] | None" = None  # (y0, y1, x0, x1)
    spatial_mask_path: "str | None" = None               # path to a bool .npy (H, W)

    # --- Serialisation ---------------------------------------------------------

    def to_json(self, path: "str | Path") -> None:
        """Write all parameters to a JSON file."""
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: "str | Path") -> "CNMFeParams":
        """Construct ``CNMFeParams`` from a JSON file written by ``to_json``.

        Tuple fields (e.g. ``max_shift``) come back as lists from JSON; this
        method restores their declared types. Unknown keys are dropped so old
        save dirs remain loadable after fields are added or removed.
        """
        raw = json.loads(Path(path).read_text())
        valid_names = {f.name for f in fields(cls)}
        data = {k: v for k, v in raw.items() if k in valid_names}
        for key in ("max_shift", "temporal_crop", "spatial_crop"):
            if isinstance(data.get(key), list):
                data[key] = tuple(data[key])
        return cls(**data)

    def downscaled(self, ssub: int, tsub: int) -> "CNMFeParams":
        """Return a copy with spatial/temporal params rescaled for a movie
        downsampled by ``ssub`` (space) and ``tsub`` (time).

        Lets a user express parameters in NATIVE (full-resolution) units once
        and run the whole pipeline on a movie produced by
        ``minicnmfe.downsample.downsample_movie``. Pixel-valued fields shrink with
        ``ssub`` (areas with ``ssub²``); the frame rate that feeds the AR-``g``
        prior shrinks with ``tsub``. Fields derived from these (e.g. the ring
        radius = ``ring_size_factor·(2·sigma+1)``) follow automatically.

        ``decay_time_ms`` is a physical time and is left unchanged — only
        ``frame_rate_hz`` changes, which correctly raises the per-frame decay.

        Cutout fields (``temporal_crop`` / ``spatial_crop`` /
        ``spatial_mask_path``) are **cleared**: a cutout is applied at native
        resolution upstream of binning, so the downsampled movie this params
        copy describes is already cropped.
        """
        if ssub < 1 or tsub < 1:
            raise ValueError(f"ssub and tsub must be >= 1 (got {ssub}, {tsub})")
        return replace(
            self,
            sigma=self.sigma / ssub,
            min_pixel=max(1, self.min_pixel // (ssub * ssub)),
            border_px=self.border_px // ssub,
            max_shift=(self.max_shift[0] // ssub, self.max_shift[1] // ssub),
            mc_gSig_filt=(None if self.mc_gSig_filt is None
                          else self.mc_gSig_filt / ssub),
            frame_rate_hz=(None if self.frame_rate_hz is None
                           else self.frame_rate_hz / tsub),
            temporal_crop=None,
            spatial_crop=None,
            spatial_mask_path=None,
        )


class CNMFe:
    """Clean CNMFe for 1-photon calcium imaging.

    Example::

        from minicnmfe import CNMFe, CNMFeParams
        from minicnmfe.io import avi_to_zarr

        movie = avi_to_zarr("recording.avi", "/tmp/movie.zarr")
        model = CNMFe(CNMFeParams(sigma=3.0, min_corr=0.8, min_pnr=10))
        model.fit(movie)

        A = model.A   # (H*W, K) sparse spatial footprints
        C = model.C   # (K, T)   calcium traces
        S = model.S   # (K, T)   spike trains
    """

    def __init__(self, params: CNMFeParams | None = None) -> None:
        self.params = params or CNMFeParams()

        # Results — populated by fit()
        self.A: sp.csc_matrix | None = None
        self.C: np.ndarray | None = None
        self.S: np.ndarray | None = None
        self.C_raw: np.ndarray | None = None
        self.YrA: np.ndarray | None = None    # residual at each footprint; C + YrA = noisy projection
        self.W: sp.csr_matrix | None = None
        self.b0: np.ndarray | None = None
        self.b_f: np.ndarray | None = None     # (H*W,) — rank-1 spatial bg
        self.f: np.ndarray | None = None       # (T,)   — rank-1 temporal bg
        self.sn: np.ndarray | None = None
        self.shifts: np.ndarray | None = None
        self.dims: tuple[int, int] | None = None
        self.g: list[np.ndarray] | None = None    # per-component AR coefs
        self.sn_per_k: np.ndarray | None = None   # per-component noise std
        self.A_norm: np.ndarray | None = None     # (K,) original ‖a_k‖₂ before unit-norm (CaImAn scale)
        self.mc_roi: "tuple[slice, slice] | None" = None  # ROI used for shift estimation
        self.accepted_mask: np.ndarray | None = None    # (K,) bool — passed auto-eval
        self.eval_info: dict | None = None              # full dict from auto_evaluate_components
        self.cutout: dict | None = None                 # crop meta (minicnmfe/cutout.py) if any

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def C_projected(self) -> np.ndarray:
        """The shape-faithful "noisy projected trace" ``C + YrA``.

        - ``self.C`` is OASIS-deconvolved (clean AR(1) shape); use it for
          spike-event detection or smooth amplitude features where the
          shape constraint ``c[t] >= g·c[t-1]`` is acceptable.
        - ``self.C + self.YrA`` is the residual at each footprint added
          back to ``C``; it has the same shape as the underlying data
          and typically correlates >0.9 with ground truth, vs ~0.6–0.85
          for ``C`` alone. Use it for cross-correlation, regression
          against external signals, or any shape-sensitive analysis.

        Raises:
            RuntimeError: if ``fit()`` has not been called yet.
        """
        if self.C is None or self.YrA is None:
            raise RuntimeError("Model has not been fit; C and YrA are unset.")
        return self.C + self.YrA

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, output_dir: "str | Path") -> None:
        """Persist results + params to ``output_dir`` as standalone files.

        Layout (mirrors ``full_pipeline.py`` so existing analysis scripts
        keep working):

        - ``A.npz``        sparse CSC footprints ``(H*W, K)``
        - ``C.npy``        OASIS-deconvolved traces ``(K, T)``
        - ``S.npy``        spike trains ``(K, T)``
        - ``YrA.npy``      residual; ``C + YrA`` is the noisy projection
        - ``C_raw.npy``    raw init traces (when available)
        - ``sn.npy``       per-pixel noise std ``(H, W)``
        - ``shifts.npy``   per-frame motion shifts ``(T, 2)`` (when available)
        - ``b0.npy``       per-pixel ring-bg baseline (when available)
        - ``W.npz``        sparse CSR ring weights (when available)
        - ``g.npy``        ``(K, ar_order)`` stacked AR coefs (when available)
        - ``sn_per_k.npy`` ``(K,)`` per-component noise std (when available)
        - ``A_norm.npy``   ``(K,)`` original ``||a_k||_2`` before the CaImAn-scale
          unit-norm relabeling (when available)
        - ``accepted_mask.npy`` ``(K,)`` bool from auto-eval (when available)
        - ``eval_info.npz`` per-component auto-eval stats (when available)
        - ``params.json``  the ``CNMFeParams`` dataclass
        - ``manifest.json`` non-parameter metadata: ``dims``, ``K``, ``T``.

        Raises:
            RuntimeError: if ``fit()`` has not been called yet.
        """
        if self.A is None or self.C is None:
            raise RuntimeError("Cannot save a model that has not been fit.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        sp.save_npz(output_dir / "A.npz", self.A.tocsc())
        np.save(output_dir / "C.npy", self.C)
        np.save(output_dir / "S.npy", self.S)
        np.save(output_dir / "YrA.npy", self.YrA)
        if self.C_raw is not None:
            np.save(output_dir / "C_raw.npy", self.C_raw)
        if self.sn is not None:
            np.save(output_dir / "sn.npy", self.sn)
        if self.shifts is not None:
            np.save(output_dir / "shifts.npy", self.shifts)
        if self.b0 is not None:
            np.save(output_dir / "b0.npy", self.b0)
        if self.b_f is not None:
            np.save(output_dir / "b_f.npy", self.b_f)
        if self.f is not None:
            np.save(output_dir / "f.npy", self.f)
        if self.W is not None:
            sp.save_npz(output_dir / "W.npz", self.W.tocsr())
        if self.g is not None and len(self.g) > 0:
            np.save(output_dir / "g.npy", np.stack(self.g))
        if self.sn_per_k is not None:
            np.save(output_dir / "sn_per_k.npy", self.sn_per_k)
        if self.A_norm is not None:
            np.save(output_dir / "A_norm.npy", self.A_norm)
        if self.accepted_mask is not None:
            np.save(output_dir / "accepted_mask.npy", self.accepted_mask)
        if self.eval_info is not None:
            # eval_info has arrays (pixel_count, snr_amp, pixel_pass, snr_pass)
            # plus scalars (min_pixel, snr_amp_thr); npz stores both fine.
            np.savez(output_dir / "eval_info.npz", **self.eval_info)

        self.params.to_json(output_dir / "params.json")

        manifest = {
            "dims": list(self.dims) if self.dims is not None else None,
            "K": int(self.A.shape[1]),
            "T": int(self.C.shape[1]),
            "cutout": self.cutout,
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    @classmethod
    def load(cls, output_dir: "str | Path") -> "CNMFe":
        """Reconstruct a ``CNMFe`` from a directory written by ``save``.

        All result attributes (``A``, ``C``, ``S``, ``YrA``, ``sn``, etc.)
        are restored. Optional files (``shifts``, ``b0``, ``W``, ``g``,
        ``sn_per_k``, ``C_raw``, ``accepted_mask``, ``eval_info``) are
        loaded if present; otherwise the corresponding attribute stays
        ``None``.
        """
        output_dir = Path(output_dir)
        if not output_dir.is_dir():
            raise FileNotFoundError(f"Save directory not found: {output_dir}")

        params = CNMFeParams.from_json(output_dir / "params.json")
        model = cls(params)

        manifest = json.loads((output_dir / "manifest.json").read_text())
        if manifest.get("dims") is not None:
            model.dims = tuple(manifest["dims"])
        model.cutout = manifest.get("cutout")

        model.A = sp.load_npz(output_dir / "A.npz").tocsc()
        model.C = np.load(output_dir / "C.npy")
        model.S = np.load(output_dir / "S.npy")
        model.YrA = np.load(output_dir / "YrA.npy")

        if (output_dir / "C_raw.npy").exists():
            model.C_raw = np.load(output_dir / "C_raw.npy")
        if (output_dir / "sn.npy").exists():
            model.sn = np.load(output_dir / "sn.npy")
        if (output_dir / "shifts.npy").exists():
            model.shifts = np.load(output_dir / "shifts.npy")
        if (output_dir / "b0.npy").exists():
            model.b0 = np.load(output_dir / "b0.npy")
        if (output_dir / "b_f.npy").exists():
            model.b_f = np.load(output_dir / "b_f.npy")
        if (output_dir / "f.npy").exists():
            model.f = np.load(output_dir / "f.npy")
        if (output_dir / "W.npz").exists():
            model.W = sp.load_npz(output_dir / "W.npz").tocsr()
        if (output_dir / "g.npy").exists():
            g_arr = np.load(output_dir / "g.npy")
            model.g = [g_arr[k] for k in range(g_arr.shape[0])]
        if (output_dir / "sn_per_k.npy").exists():
            model.sn_per_k = np.load(output_dir / "sn_per_k.npy")
        if (output_dir / "A_norm.npy").exists():
            model.A_norm = np.load(output_dir / "A_norm.npy")
        if (output_dir / "accepted_mask.npy").exists():
            model.accepted_mask = np.load(output_dir / "accepted_mask.npy")
        if (output_dir / "eval_info.npz").exists():
            with np.load(output_dir / "eval_info.npz") as f:
                model.eval_info = {
                    k: f[k].item() if f[k].ndim == 0 else f[k] for k in f.files
                }

        return model

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _resolve_mc_template(self, movie, template, template_window, dims):
        """Resolve an optional precomputed MC template (in cropped coords).

        Returns an ``(H, W)`` float32 template or ``None`` (auto, built from a
        strided sample inside ``motion_correction_rigid``). At most one of
        ``template`` / ``template_window`` may be given.

        - ``template``: a precomputed ``(H, W)`` array (raw, *not* high-pass
          filtered — MC filters it internally with ``mc_gSig_filt``).
        - ``template_window=(t0, t1)``: build the template as the mean of frames
          ``[t0:t1)``. A *short, low-motion* window gives a sharp, single-position
          template that avoids the smeared-full-movie under-tracking — only that
          slice is read, so it is cheap even for a zarr input.
        """
        if template is not None and template_window is not None:
            raise ValueError(
                "fit_mc: pass at most one of template / template_window"
            )
        if template is not None:
            t = np.asarray(template, dtype=np.float32)
            if t.shape != dims:
                raise ValueError(
                    f"fit_mc: template shape {t.shape} != movie dims {dims} "
                    "(template must match the post-cutout frame size)"
                )
            return t
        if template_window is not None:
            t0, t1 = int(template_window[0]), int(template_window[1])
            T = int(movie.shape[0])
            if not (0 <= t0 < t1 <= T):
                raise ValueError(
                    f"fit_mc: template_window {(t0, t1)} out of range for T={T}"
                )
            return np.asarray(movie[t0:t1], dtype=np.float32).mean(axis=0)
        return None

    def fit_mc(
        self,
        movie: "zarr.Array | np.ndarray",
        output_dir: str | Path | None = None,
        in_place: bool = False,
        template: "np.ndarray | None" = None,
        template_window: "tuple[int, int] | None" = None,
    ) -> "zarr.Array | np.ndarray":
        """Run only the motion correction step.

        Stores self.shifts and self.dims.

        - If ``output_dir`` is given **or** the input is a zarr.Array, runs the
          streaming MC path: reads/writes batches of frames, peak RAM is
          ``(mc_batch_size + mc_template_max_frames) * H * W * 4`` bytes
          regardless of T. The corrected movie is written to
          ``<output_dir>/mc.zarr`` and the zarr handle is returned.
        - Otherwise (numpy input, no output_dir): returns a float32 numpy array
          in memory. Per-frame work is parallelized over ``params.n_jobs``.

        Call ``model.fit(mc, do_motion_correction=False)`` afterward to run
        extraction without re-running correction.

        Args:
            movie: Input movie, shape (T, H, W). zarr or numpy array.
            output_dir: If given, write ``mc.zarr`` here and return zarr handle.
            template: Optional precomputed ``(H, W)`` template to register
                against, instead of the auto strided-sample template. Useful
                when the auto template smears under large drift (it averages
                over every drift position) — supply a sharp, single-position
                template so even a couple of passes recover full amplitude.
                Must match the post-cutout frame size; given raw (MC high-pass
                filters it internally).
            template_window: Optional ``(t0, t1)`` — build the template as the
                mean of frames ``[t0:t1)`` instead of supplying one. Pick a
                *short, low-motion* window; only that slice is read. Mutually
                exclusive with ``template``.

        Returns:
            Corrected movie — zarr handle (if output_dir given / zarr input)
            or np.ndarray (if numpy input + no output_dir).
        """
        p = self.params

        # Crop (if a cutout is set) before MC. Materialises the crop in RAM.
        movie = self._ingest_cutout(movie)

        # Read shape without materializing the full movie when it's a zarr.
        T, H, W = int(movie.shape[0]), int(movie.shape[1]), int(movie.shape[2])
        self.dims = (H, W)

        # Optional sharp / precomputed template (resolved in cropped coords).
        mc_template = self._resolve_mc_template(
            movie, template, template_window, (H, W)
        )

        output_path: "Path | None" = None
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "mc.zarr"

        corrected, self.shifts = motion_correction_rigid(
            movie,
            output_path=output_path,
            max_shift=p.max_shift,
            gSig_filt=p.mc_gSig_filt,
            upsample_factor=p.upsample_factor,
            niter_rig=p.mc_n_iter,
            converge_tol=p.mc_converge_tol,
            sharpen_template=p.mc_sharpen_template,
            template=mc_template,
            batch_size=p.mc_batch_size,
            n_jobs=p.n_jobs,
            template_max_frames=p.mc_template_max_frames,
            output_chunk_t=p.mc_output_chunk_t,
            output_dtype=p.mc_output_dtype,
            in_place=in_place,
        )
        return corrected

    def fit_mc_from_avis(
        self,
        folder: "str | Path",
        output_dir: "str | Path",
        *,
        pattern: str = "*.avi",
        skip_if_exists: bool = False,
        ssub: int = 1,
        tsub: int = 1,
    ) -> "zarr.Array":
        """Fused AVI -> motion-corrected zarr in one pass.

        Skips the intermediate `session.zarr` that the two-step
        `concat_avis_to_zarr` + `fit_mc` flow produces. On a network mount
        this typically saves ~5 min and ~6 GB of disk for a 100k-frame
        session.

        Writes ``<output_dir>/mc.zarr`` (float32, motion-corrected) and
        ``<output_dir>/shifts.npy``. Stores ``self.shifts`` and ``self.dims``
        for symmetry with ``fit_mc``.

        Handles ``params.mc_n_iter`` ≥ 1. For ``mc_n_iter == 1`` the fused
        path writes ``mc.zarr`` directly from the AVIs. For
        ``mc_n_iter > 1`` it writes the fused first pass to a scratch zarr
        and hands the remaining iterations off to
        ``motion_correction_rigid`` (which rebuilds the template from the
        corrected output of each pass). The scratch is cleaned up
        automatically.

        Args:
            folder: Directory containing numbered AVI files (0.avi, ...).
            output_dir: Output directory. ``mc.zarr`` + ``shifts.npy`` go
                here.
            pattern: Glob pattern for AVI selection.
            skip_if_exists: If ``mc.zarr`` is already in ``output_dir``,
                reuse it (and ``shifts.npy`` if present).
            ssub: Spatial bin factor applied to the raw AVI frames before MC
                (``1`` = none). The fused ``mc.zarr`` is the only zarr written.
            tsub: Temporal bin factor (per file) applied before MC (``1`` = none).
                **When downsampling, construct this model with downscaled
                params**, e.g. ``CNMFe(params.downscaled(ssub, tsub))``, so
                ``max_shift`` / ``mc_gSig_filt`` match the binned frames.

        Returns:
            Open zarr.Array of the corrected movie, shape
            ``(T_out, H//ssub, W//ssub)`` float32.
        """
        # Local import: avi_mc depends on the top-level concat_avis_to_zarr
        # script, which we don't want as a hard dep of `pipeline`.
        from minicnmfe.avi_mc import concat_avis_to_mc_zarr

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        mc_path = output_dir / "mc.zarr"

        mc_zarr, shifts = concat_avis_to_mc_zarr(
            folder,
            mc_path,
            self.params,
            pattern=pattern,
            skip_if_exists=skip_if_exists,
            ssub=ssub,
            tsub=tsub,
            verbose=True,
        )

        self.shifts = shifts
        self.dims = (int(mc_zarr.shape[1]), int(mc_zarr.shape[2]))

        if shifts is not None:
            np.save(output_dir / "shifts.npy", shifts)

        # If a cutout was applied during decode, concat_avis_to_mc_zarr wrote a
        # cutout.json sidecar; load it so place_in_full_fov works downstream.
        cutout_json = output_dir / "cutout.json"
        if cutout_json.exists():
            self.cutout = json.loads(cutout_json.read_text())

        return mc_zarr

    def _ingest_cutout(self, movie, *, Y_flat_zarr=None):
        """Apply the params cutout (crop) to a raw ``(T, H, W)`` movie, once.

        Returns the cropped float32 array (numpy) and records ``self.cutout``.
        No-op (returns ``movie`` unchanged) when no cutout fields are set.
        Raises if a cutout is combined with a pre-built ``Y_flat_zarr`` (the
        crop must be baked into that store upstream).
        """
        from minicnmfe.cutout import apply_cutout, public_spec, resolve_cutout

        spec = resolve_cutout(
            self.params,
            (int(movie.shape[1]), int(movie.shape[2])),
            int(movie.shape[0]),
        )
        if spec is None:
            return movie
        if Y_flat_zarr is not None:
            raise ValueError(
                "A cutout (temporal_crop/spatial_crop/spatial_mask_path) cannot "
                "be combined with a pre-built Y_flat_zarr; bake the crop in "
                "upstream or clear the crop params."
            )
        cropped = apply_cutout(movie, spec)
        self.cutout = public_spec(spec)
        self.cutout["spatial_mask_path"] = self.params.spatial_mask_path
        y0, y1, x0, x1 = spec["bbox"]
        t0, t1 = spec["t_range"]
        print(
            f"Cutout: T[{t0}:{t1}] H[{y0}:{y1}] W[{x0}:{x1}]"
            f"{' +mask' if spec['masked'] else ''} -> {cropped.shape}"
        )
        return cropped

    def fit(
        self,
        movie: "zarr.Array | np.ndarray",
        do_motion_correction: bool = True,
        output_dir: str | Path | None = None,
        Y_flat_zarr: "zarr.Array | None" = None,
        evaluate: bool = True,
    ) -> "CNMFe":
        """Run the full CNMFe pipeline on a (T, H, W) movie.

        Thin wrapper that composes the standalone stages:
        ``fit_mc`` (optional, in-memory) -> ``fit_extract`` -> ``evaluate``.
        Behaviour is unchanged from the historical monolithic ``fit``; the
        stages are also callable individually for a disk-handoff workflow.

        Args:
            movie: Input movie. zarr.Array or numpy array, shape (T, H, W).
            do_motion_correction: Run rigid motion correction before extraction.
            output_dir: If given, derive the pixel-major ``Y_flat`` store here
                (streaming path).
            Y_flat_zarr: Optional pixel-major ``(H*W, T)`` zarr (see
                ``fit_extract``). Incompatible with ``do_motion_correction``.
            evaluate: Run the non-destructive auto-evaluation pass at the end
                (sets ``accepted_mask`` / ``eval_info``).

        Returns:
            self (for chaining).
        """
        p = self.params

        # Crop the movie (if a cutout is set) before MC / extraction.
        movie = self._ingest_cutout(movie, Y_flat_zarr=Y_flat_zarr)

        if do_motion_correction:
            if Y_flat_zarr is not None:
                raise ValueError(
                    "Y_flat_zarr is the already-corrected movie; pass "
                    "do_motion_correction=False (and run fit_mc separately if needed)."
                )
            # In-memory MC: extraction below needs the full movie in RAM
            # anyway, so streaming-to-zarr would only add disk IO. Use
            # fit_mc(zarr, output_dir=...) directly when the input is too big.
            movie_arr = np.asarray(movie, dtype=np.float32)
            movie_arr, self.shifts = motion_correction_rigid(
                movie_arr,
                max_shift=p.max_shift,
                gSig_filt=p.mc_gSig_filt,
                upsample_factor=p.upsample_factor,
                niter_rig=p.mc_n_iter,
                batch_size=p.mc_batch_size,
                n_jobs=p.n_jobs,
                template_max_frames=p.mc_template_max_frames,
            )
            return self.fit_extract(
                movie_arr, output_dir=output_dir, evaluate=evaluate,
            )

        return self.fit_extract(
            movie, Y_flat_zarr=Y_flat_zarr, output_dir=output_dir,
            evaluate=evaluate,
        )

    def evaluate(self) -> "CNMFe":
        """Compute per-component quality metrics (non-destructive, report-only).

        Always populates ``self.eval_info`` (per-component ``snr_amp`` =
        mean(a^2)/mean(sn^2) and ``pixel_count``) for inspection. The acceptance
        *gate* (``self.accepted_mask``) is **off by default**
        (``auto_eval_snr_amp_thr=0.0`` → all accepted); raise that threshold
        (and/or ``min_pixel``) to opt in to ghost filtering. See
        ``minicnmfe/evaluate.py``.

        Reads only ``self.A``, ``self.sn`` and ``self.A_norm`` (the cached
        original footprint norms, used to recover the un-normalized SNR after
        the CaImAn-scale relabeling), so it can be called on a freshly
        ``load()``-ed model to retune thresholds without re-extracting. **No
        components are dropped**; filter post-hoc via ``model.accepted_mask``.

        Sets ``self.accepted_mask`` and ``self.eval_info``. Returns self.
        """
        if self.A is None:
            raise RuntimeError("Cannot evaluate before extraction (self.A is None).")
        if self.sn is None:
            raise RuntimeError(
                "evaluate() needs the per-pixel noise map self.sn; run "
                "fit_extract first or load() a directory that contains sn.npy."
            )
        p = self.params
        if self.A.shape[1] == 0:
            self.accepted_mask = np.zeros(0, dtype=bool)
            self.eval_info = None
            return self

        keep, eval_info = auto_evaluate_components(
            self.A,
            sn_flat=self.sn.ravel(),
            min_pixel=p.min_pixel,
            snr_amp_thr=p.auto_eval_snr_amp_thr,
            a_norm=self.A_norm,
        )
        self.accepted_mask = keep
        self.eval_info = eval_info
        n_drop = int((~keep).sum())
        n_px_fail = int((~eval_info["pixel_pass"]).sum())
        n_snr_fail = int((~eval_info["snr_pass"]).sum())
        print(
            f"Auto-evaluation: {int(keep.sum())}/{self.A.shape[1]} accepted "
            f"(flagged {n_drop}: {n_px_fail} fail pixel_count<{p.min_pixel}, "
            f"{n_snr_fail} fail snr_amp<{p.auto_eval_snr_amp_thr}). "
            f"All components retained; filter via model.accepted_mask."
        )
        return self

    def upsample_to_native(
        self,
        *,
        orig_dims: "tuple[int, int] | None" = None,
        orig_T: "int | None" = None,
        ssub: "int | None" = None,
        tsub: "int | None" = None,
        ds_meta: "dict | str | Path | None" = None,
        spatial_order: int = 1,
    ) -> "CNMFe":
        """Return a NEW model with footprints/traces interpolated to native res.

        Re-expresses a downsampled extraction on the native pixel grid and frame
        rate — for overlaying footprints on a native-resolution reference image
        and plotting traces against native-rate signals.

        **This is interpolation, not recovery.** The downsample-once workflow
        keeps only the downsampled movie, so this upsamples the *already-extracted*
        downsampled results; it does NOT recover detail the binning discarded.
        Footprints become smooth (bilinear) blobs and traces are linearly
        interpolated.

        Non-destructive — ``self`` is left unchanged. The returned model is for
        **inspection / overlay / native-rate plotting**, NOT for re-running the
        BCD: the ring/background model (``W``, ``b0``, ``b_f``, ``f``) and
        ``shifts`` are dropped, and the deconvolved spike train ``S`` is left at
        the **downsampled** rate (upsampling a spike train is ambiguous), so
        ``S.shape[1]`` will not match the upsampled ``C``.

        Args:
            orig_dims: Native ``(H, W)``. Required (or supplied via ``ds_meta``).
            orig_T: Native frame count. Required (or via ``ds_meta``). Must be
                ``>=`` the downsampled ``T``.
            ssub, tsub: Recorded for reference only; the interpolation derives
                everything from ``self.dims``/``orig_dims`` and ``T``/``orig_T``.
            ds_meta: Optional ``ds_meta.json`` dict or path written by
                ``downsample_movie``; if given, ``orig_dims``/``orig_T`` are read
                from it.
            spatial_order: 1 = bilinear (default), 0 = nearest (block-exact).

        Returns:
            A new ``CNMFe`` with ``A``/``C``/``YrA``/``C_raw`` at native
            resolution, ``dims = orig_dims``.
        """
        from minicnmfe.downsample import upsample_footprints, upsample_traces

        if self.A is None or self.C is None:
            raise RuntimeError("upsample_to_native requires a fitted model.")
        if self.dims is None:
            raise RuntimeError("self.dims is None; cannot upsample.")

        if ds_meta is not None:
            if isinstance(ds_meta, (str, Path)):
                ds_meta = json.loads(Path(ds_meta).read_text())
            orig_dims = tuple(ds_meta["orig_dims"]) if orig_dims is None else orig_dims
            orig_T = int(ds_meta["orig_T"]) if orig_T is None else orig_T
            ssub = ds_meta.get("ssub", ssub)
            tsub = ds_meta.get("tsub", tsub)
        if orig_dims is None or orig_T is None:
            raise ValueError(
                "provide orig_dims and orig_T (or a ds_meta with those keys)."
            )
        orig_dims = (int(orig_dims[0]), int(orig_dims[1]))
        orig_T = int(orig_T)
        T_ds = int(self.C.shape[1])
        if orig_T < T_ds:
            raise ValueError(
                f"orig_T ({orig_T}) < downsampled T ({T_ds}); is it really native?"
            )

        new = CNMFe(self.params)
        new.dims = orig_dims
        new.A = upsample_footprints(self.A, self.dims, orig_dims, order=spatial_order)
        new.C = upsample_traces(self.C, orig_T)
        new.YrA = upsample_traces(self.YrA, orig_T) if self.YrA is not None else None
        new.C_raw = upsample_traces(self.C_raw, orig_T) if self.C_raw is not None else None
        # S stays at the downsampled rate (documented).
        new.S = None if self.S is None else self.S.copy()
        # Per-component metadata is resolution-independent.
        new.g = self.g
        new.sn_per_k = self.sn_per_k
        # A_norm carried best-effort: bilinear footprint interpolation perturbs
        # the column norms slightly, so it is approximate on this
        # inspection-only view (don't re-run the BCD / evaluate() here).
        new.A_norm = self.A_norm
        new.accepted_mask = self.accepted_mask
        new.eval_info = self.eval_info
        if self.sn is not None:
            import cv2
            new.sn = cv2.resize(
                np.asarray(self.sn, dtype=np.float32),
                (orig_dims[1], orig_dims[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        # Extraction internals / ds-rate quantities aren't meaningful at the
        # native grid; the upsampled model is for inspection, not re-fitting.
        new.W = None
        new.b0 = None
        new.b_f = None
        new.f = None
        new.shifts = None
        return new

    def place_in_full_fov(self, *, place_time: bool = True) -> "CNMFe":
        """Return a NEW model with footprints/traces mapped back to the full FOV.

        Inverse of the cutout: footprints are padded back to the original
        ``(H, W)`` at the crop ``(y0, x0)`` offset, and (when ``place_time``)
        the traces ``C``/``YrA``/``C_raw``/``S`` are embedded in the full
        ``orig_T`` timeline at ``[t0:t1]`` (zeros outside the window). For
        overlaying on the uncropped movie / anatomy and aligning with
        full-length signals.

        Non-destructive (``self`` unchanged). The returned model is for
        inspection/overlay: the background model (``W``/``b0``/``b_f``/``f``)
        and ``shifts`` are dropped. Requires that this model was run on a
        cutout (``self.cutout`` is set).
        """
        from minicnmfe.cutout import place_footprints_in_fov, place_traces_in_timeline

        if self.cutout is None:
            raise RuntimeError(
                "place_in_full_fov requires a cutout; this model was not run on one."
            )
        if self.A is None or self.C is None:
            raise RuntimeError("place_in_full_fov requires a fitted model.")
        c = self.cutout
        orig_dims = tuple(c["orig_dims"])
        orig_T, bbox, t_range = c["orig_T"], c["bbox"], c["t_range"]

        new = CNMFe(self.params)
        new.dims = orig_dims
        new.A = place_footprints_in_fov(self.A, bbox, orig_dims)
        if place_time:
            new.C = place_traces_in_timeline(self.C, t_range, orig_T)
            new.YrA = place_traces_in_timeline(self.YrA, t_range, orig_T)
            new.C_raw = place_traces_in_timeline(self.C_raw, t_range, orig_T)
            new.S = place_traces_in_timeline(self.S, t_range, orig_T)
        else:
            new.C = self.C.copy()
            new.YrA = None if self.YrA is None else self.YrA.copy()
            new.C_raw = None if self.C_raw is None else self.C_raw.copy()
            new.S = None if self.S is None else self.S.copy()
        new.g = self.g
        new.sn_per_k = self.sn_per_k
        # Zero-padding into the full FOV preserves each column's L2 norm exactly.
        new.A_norm = self.A_norm
        new.accepted_mask = self.accepted_mask
        new.eval_info = self.eval_info
        if self.sn is not None:
            y0, y1, x0, x1 = bbox
            full_sn = np.zeros(orig_dims, dtype=np.float32)
            full_sn[y0:y1, x0:x1] = self.sn
            new.sn = full_sn
        new.W = None
        new.b0 = None
        new.b_f = None
        new.f = None
        new.shifts = None
        return new

    def fit_extract(
        self,
        movie: "zarr.Array | np.ndarray",
        *,
        Y_flat_zarr: "zarr.Array | None" = None,
        output_dir: str | Path | None = None,
        evaluate: bool = True,
    ) -> "CNMFe":
        """Run extraction on an ALREADY motion-corrected (T, H, W) movie.

        This is the historical ``fit`` body minus motion correction: noise
        estimation, greedy init, ring background, the spatial/temporal/merge
        BCD loop, the final temporal pass, and YrA. Optionally finishes with
        the non-destructive auto-evaluation (``evaluate=True``).

        Args:
            movie: Already-corrected movie. zarr.Array or numpy array, shape
                (T, H, W). Used for noise estimation and greedy init (the
                latter on a strided sample so the 3D movie need not fully
                materialise).
            Y_flat_zarr: Optional pixel-major ``(H*W, T)`` zarr produced by
                ``transpose_zarr_to_pixel_major``. When provided, extraction
                (compute_W, BackgroundSubtractor, update_spatial,
                update_temporal) runs directly against this on-disk array
                so the full ``(H*W, T)`` movie is never held in RAM.
                ``movie`` must be a zarr.Array in this mode (we need
                strided 3D access for init); it is **not** materialised.
                Shape must match the spatial dims of ``movie``.
            output_dir: If given (with a zarr ``movie`` and no Y_flat_zarr),
                the pixel-major ``Y_flat`` store is derived here.
            evaluate: Run the non-destructive auto-evaluation pass at the end.

        Returns:
            self (for chaining).
        """
        p = self.params
        timer = StageTimer()

        # --- Auto-derive Y_flat_zarr from a zarr movie + output_dir -----------
        # Passing a zarr.Array `movie` without Y_flat_zarr used to silently
        # asarray() the whole thing into RAM (86 GB for 60k×600×600). When
        # the user also gives an `output_dir` we know where to put a derived
        # pixel-major store: transpose once (idempotent — skip_if_exists),
        # then route through the existing streaming branch below.
        auto_derived_y_flat = False
        if Y_flat_zarr is None:
            try:
                import zarr as _zarr_pkg
                is_zarr_movie = isinstance(movie, _zarr_pkg.Array)
            except ImportError:
                is_zarr_movie = False
            if is_zarr_movie and output_dir is not None:
                src_path = _zarr_store_path(movie)
                if src_path is None:
                    raise ValueError(
                        "Cannot auto-derive Y_flat_zarr: the input zarr has no "
                        "resolvable on-disk path. Pass Y_flat_zarr= explicitly "
                        "(e.g. via minicnmfe.io.transpose_zarr_to_pixel_major)."
                    )
                # Default the store next to the results; `yflat_dir` can divert
                # it to a local SSD/tmpfs so the BCD passes read off the network.
                yflat_parent = Path(p.yflat_dir) if p.yflat_dir else Path(output_dir)
                pixel_path = yflat_parent / "Y_flat_pixel.zarr"
                pixel_path.parent.mkdir(parents=True, exist_ok=True)
                with timer.stage("transpose -> Y_flat"):
                    Y_flat_zarr = transpose_zarr_to_pixel_major(
                        src_path, pixel_path,
                        pixel_chunk=p.yflat_pixel_chunk,
                        time_chunk=p.yflat_time_chunk,
                        compression=p.yflat_compression,
                        verbose=True,
                    )
                auto_derived_y_flat = True

        # --- Streaming mode (Y_flat_zarr supplied or auto-derived) ------------
        streaming = Y_flat_zarr is not None
        # L3: in-memory path on a zarr movie -> materialise AFTER greedy init.
        defer_materialize = False
        if streaming:
            try:
                import zarr as _zarr_pkg
                if not isinstance(movie, _zarr_pkg.Array):
                    raise TypeError(
                        "Y_flat_zarr requires `movie` to be a zarr.Array (3D) "
                        "so the strided init sample can be read lazily."
                    )
            except ImportError as exc:
                raise RuntimeError("zarr is required for Y_flat_zarr") from exc
            T, H, W = int(movie.shape[0]), int(movie.shape[1]), int(movie.shape[2])
            if Y_flat_zarr.shape != (H * W, T):
                raise ValueError(
                    f"Y_flat_zarr shape {Y_flat_zarr.shape} must equal "
                    f"(H*W, T) = ({H * W}, {T}) derived from movie {movie.shape}."
                )
            movie_arr = movie    # zarr handle; never asarray'd in full
        else:
            # In-memory path. When `movie` is a zarr (not numpy) we DEFER the
            # full 23 GB materialisation until after greedy init (L3): the full
            # movie is not touched during the greedy peak — only the strided
            # init sample (`movie_arr[::init_stride]`) and the noise-stats slice
            # are, and both read lazily from the zarr exactly like the streaming
            # branch above. Keeping the movie on disk until then removes the
            # resident-movie floor from under the greedy peak. It is materialised
            # below, just before `make_2d`, once greedy's buffers are freed.
            # A numpy `movie` is already resident -> nothing to defer.
            defer_materialize = is_zarr_movie
            if defer_materialize:
                movie_arr = movie    # zarr handle; full asarray happens post-init
                T, H, W = (
                    int(movie.shape[0]), int(movie.shape[1]), int(movie.shape[2]),
                )
            else:
                movie_arr = np.asarray(movie, dtype=np.float32)
                T, H, W = movie_arr.shape

        dims = (H, W)
        self.dims = dims

        # --- Log effective config so users can see what they got ---
        if streaming:
            stream_str = f"yes (Y_flat_zarr={'auto-derived' if auto_derived_y_flat else 'user-supplied'})"
        elif defer_materialize:
            stream_str = "no (movie materialised in RAM after greedy init)"
        else:
            stream_str = "no (movie materialised in RAM)"
        print(
            f"Extraction config: n_jobs={p.n_jobs} device={p.device} "
            f"T={T} H={H} W={W} streaming={stream_str}"
        )
        if p.n_jobs == 1:
            print(
                "  Note: n_jobs=1 (serial). "
                "Set CNMFeParams(n_jobs=-1) to use all CPU cores."
            )

        # --- Steps 2-3: sample frames for statistics if T > sample_frames ---
        # T can come from a zarr handle (streaming) or a numpy array (in-memory).
        T = int(movie_arr.shape[0])
        stride = max(1, T // p.sample_frames)
        if stride > 1:
            t_idx = np.arange(0, T, stride)
            stats_movie = movie_arr[t_idx]  # advanced indexing already copies
        else:
            stats_movie = movie_arr

        # --- Step 2: Noise estimation ---
        print("Estimating noise...")
        _t = time.perf_counter()
        self.sn = estimate_noise(stats_movie, n_jobs=p.n_jobs)   # (H, W)
        timer.add("noise estimation", time.perf_counter() - _t)

        # # --- Step 3: Summary images ---
        # print("Computing CORR and PNR images...")
        # cn, pnr = correlation_pnr(
        #     stats_movie, sigma=p.sigma, center_psf=p.center_psf,
        #     n_jobs=p.n_jobs, device=p.device,
        # )

        # L3 chunked materialisation of a deferred zarr movie into a fresh
        # float32 array. np.empty + chunked copy, NOT np.asarray(zarr, float32)
        # (which reads the native array then .astype-copies it -> ~3x resident).
        # Caps materialisation at ~1x. Used by the stride==1 path (before greedy)
        # and the stride>1 path (after the init bootstrap), so the full movie is
        # never co-resident with the strided init sample.
        def _materialise_full_movie():
            nonlocal movie_arr
            _t_mat = time.perf_counter()
            _zsrc = movie_arr
            movie_arr = np.empty(_zsrc.shape, dtype=np.float32)
            for _s in range(0, _zsrc.shape[0], 1000):
                _e = min(_s + 1000, _zsrc.shape[0])
                movie_arr[_s:_e] = _zsrc[_s:_e]
            del _zsrc
            timer.add("materialise movie (post-init)", time.perf_counter() - _t_mat)

        # --- Step 4: Initialization ---
        # Greedy init allocates two full (T, H, W) float32 copies inside
        # (`data_filtered` after PSF + `data_raw`). On a 10k × 600 × 600
        # movie that is ~28 GB of transient overhead. Running init on a
        # strided sample cuts this by `stride`. The footprints A are
        # spatial — independent of T — so spatial recovery is unaffected;
        # the temporal traces are then re-projected at full T below.
        init_stride = p.init_stride
        if init_stride is None:
            # PNR-safe default: do NOT sub-sample the movie. init_stride shrinks
            # the whole movie before init, which sub-samples the PNR peak (a
            # max-over-time statistic) and silently drops cells. We keep full-T
            # detection and take the long-movie speed-up from the CORR-only
            # stride (init_corrpnr_stride) below instead, which never touches PNR.
            init_stride = 1

        if init_stride > 1:
            # In deferred mode movie_arr is a zarr handle; `[::init_stride]`
            # reads the strided sample as a numpy array (the only allocation
            # greedy needs), leaving the full movie on disk until after init.
            # On a network mount this strided read touches every time-chunk.
            _t_isr = time.perf_counter()
            init_movie = movie_arr[::init_stride]
            timer.add("init strided read", time.perf_counter() - _t_isr)
        else:
            if defer_materialize:
                # stride 1: greedy reads every frame, so deferral saves no peak
                # (the movie must be resident during init regardless) and would
                # also leave init_movie as a zarr handle, disabling patched init
                # (which needs a numpy init_movie). Materialise now so the path
                # matches the numpy case exactly.
                _materialise_full_movie()
                defer_materialize = False
            init_movie = movie_arr
        T_init = int(init_movie.shape[0])

        # Resolve the secondary stride for the initial CORR/PNR sweep (CN only;
        # PNR always uses full T — see greedy_corr_pnr). Default is 1 (no stride):
        # PNR is a peak/max statistic, and even with the CN-only stride the
        # correlation values shift slightly, so we never sacrifice seeds for
        # speed by default. Opt into corrpnr_stride > 1 explicitly for a faster
        # CN sweep on very long movies.
        corrpnr_stride = p.init_corrpnr_stride
        if corrpnr_stride is None:
            # Auto speed-up for long movies: stride ONLY the CORR sweep (PNR keeps
            # full T → no cells lost), leaving ~2000 frames for the correlation —
            # plenty for a stable CN, and the per-seed CN refresh is the init hot
            # spot. Short movies (T <= 2000) get stride 1 = bit-for-bit unchanged.
            corrpnr_stride = max(1, T_init // 2000)

        if init_stride > 1 or corrpnr_stride > 1:
            print(
                f"Running greedy CORR-PNR initialization "
                f"(init_stride={init_stride}, T_init={T_init}; "
                f"corrpnr_stride={corrpnr_stride}, T_corrpnr={T_init // corrpnr_stride})..."
            )
        else:
            print("Running greedy CORR-PNR initialization...")

        # Bayesian-prior target for g, derived from indicator τ + frame rate.
        # Threaded into every estimate_ar_params call (greedy init,
        # pipeline init, update_temporal fallback) so g doesn't pin at the
        # fudge_factor ceiling on data with un-subtracted slow background.
        g_target: float | None = None
        if p.decay_time_ms is not None and p.frame_rate_hz is not None:
            # NB: this is exactly temporal.g_from_decay_time(decay_time_ms,
            # frame_rate_hz); kept inline only to avoid an import here — keep the
            # two in sync (flagged for a cleanup PR to call the helper).
            g_target = float(np.exp(
                -1.0 / (p.frame_rate_hz * p.decay_time_ms / 1000.0)
            ))
            print(
                f"Using Bayesian g prior: decay_time_ms={p.decay_time_ms} "
                f"frame_rate_hz={p.frame_rate_hz} -> g_target={g_target:.4f} "
                f"(weight={p.g_prior_weight})"
            )

        _t = time.perf_counter()
        use_patches = (
            p.init_patches
            and min(H, W) >= p.init_patch_min_fov
            and p.device == "cpu"
            # in-RAM only: greedy_corr_pnr_patched does np.asarray(movie), so a
            # streaming/zarr init_movie would be fully materialised (OOM risk).
            and isinstance(init_movie, np.ndarray)
        )
        if use_patches:
            patch_size = p.init_patch_size or max(int(12 * p.sigma), 48)
            patch_overlap = (
                p.init_patch_overlap
                if p.init_patch_overlap is not None
                else int(4 * p.sigma)
            )
            requested_jobs = p.init_patch_n_jobs or p.n_jobs
            patch_n_jobs = _resolve_patch_workers(
                requested_jobs, p.init_patch_max_workers
            )
            capped = patch_n_jobs < (
                (os.cpu_count() or 1) if requested_jobs < 0 else requested_jobs
            )
            print(
                f"  Patch-parallel init: patch_size={patch_size}, "
                f"overlap={patch_overlap}, n_jobs={patch_n_jobs}"
                + (
                    f" (capped to init_patch_max_workers={p.init_patch_max_workers})"
                    if capped
                    else ""
                )
            )
            A, C_init, C_raw_init, centers = greedy_corr_pnr_patched(
                init_movie,
                sigma=p.sigma,
                min_corr=p.min_corr,
                min_pnr=p.min_pnr,
                max_neurons=p.max_neurons,
                min_pixel=p.min_pixel,
                border_px=p.border_px,
                ar_order=p.ar_order,
                min_corr_neuron=p.init_min_corr_neuron,
                max_corr_bg=p.init_max_corr_bg,
                seed_suppress_factor=p.seed_suppress_factor,
                circular_max_dist_factor=p.circular_max_dist_factor,
                corrpnr_stride=corrpnr_stride,
                g_prior=g_target,
                g_prior_weight=p.g_prior_weight,
                patch_size=patch_size,
                patch_overlap=patch_overlap,
                n_jobs=patch_n_jobs,
                merge_thr_corr=p.merge_thr_corr,
                merge_thr_overlap=p.merge_thr_overlap,
                merge_centre_dist_factor=p.merge_centre_dist_factor,
                seed_mode=p.seed_method,
            )
        else:
            A, C_init, C_raw_init, centers = greedy_corr_pnr(
                init_movie,
                sigma=p.sigma,
                min_corr=p.min_corr,
                min_pnr=p.min_pnr,
                max_neurons=p.max_neurons,
                min_pixel=p.min_pixel,
                border_px=p.border_px,
                ar_order=p.ar_order,
                n_jobs=p.n_jobs,
                device=p.device,
                min_corr_neuron=p.init_min_corr_neuron,
                max_corr_bg=p.init_max_corr_bg,
                seed_suppress_factor=p.seed_suppress_factor,
                circular_max_dist_factor=p.circular_max_dist_factor,
                corrpnr_stride=corrpnr_stride,
                g_prior=g_target,
                g_prior_weight=p.g_prior_weight,
                seed_mode=p.seed_method,
            )
        timer.add("greedy init", time.perf_counter() - _t)
        # NOTE: the strided init movie is freed below, after it is reused to
        # bootstrap a background-aware init projection (see the init_stride>1
        # branch). Freeing it here would force the contaminated raw projection.
        print(f"  Found {A.shape[1]} initial components.")

        if A.shape[1] == 0:
            print("No neurons found. Try lowering min_corr or min_pnr.")
            self.A = A
            self.C = np.empty((0, T), dtype=np.float32)
            self.S = np.empty((0, T), dtype=np.float32)
            self.C_raw = np.empty((0, T), dtype=np.float32)
            self.A_norm = np.zeros(0, dtype=np.float32)
            return self

        # Flatten movie to (H*W, T) for all subsequent steps. In streaming
        # mode the user-supplied pixel-major zarr IS our Y_flat — no
        # materialisation.
        if streaming:
            Y_flat = Y_flat_zarr
        elif defer_materialize and init_stride > 1:
            # L3 + strided init: DEFER the full materialisation past the init
            # bootstrap below (its compute_W needs only the strided init sample).
            # The full movie is brought in just before project_onto, AFTER
            # init_movie is freed, so the full movie and the strided sample are
            # never co-resident (the ~movie + 2*sample peak). Y_flat set there.
            Y_flat = None
        else:
            # Non-deferred (numpy in), or stride==1 (already materialised above).
            # Greedy is done and its (T_init,H,W) buffers are freed, so the
            # resident-movie floor no longer stacks under the peak.
            if defer_materialize:
                _materialise_full_movie()
            Y_flat = make_2d(movie_arr)     # (H*W, T) view
        sn_flat = self.sn.ravel()           # (H*W,)

        # `C_init` / `C_raw_init` from greedy_corr_pnr are the per-pixel
        # OLS-extracted traces at the seed pixels — narrow, mostly-clean
        # traces (the seed pixel is, by construction, the local peak of
        # CORR×PNR). They contain orders of magnitude less background
        # contamination than the full-movie projection `(A.T @ Y) / ‖A‖²`,
        # which polls every pixel in the footprint including the noisy
        # halo. Phase D (commit 8a91b4e) replaced these with the projection
        # to unify the strided init code path; on the realistic-miniscope
        # fixture this dropped `r(C+YrA, truth)` from 0.87 to 0.18. Keep
        # the strided projection ONLY when stride>1 (where greedy returned
        # T_init traces, not full T); for stride==1 use greedy's traces
        # directly.
        # Ring radius (used both for the init-projection bootstrap below and
        # for the first full ring background fit further down).
        ring_radius = p.ring_size_factor * (2 * p.sigma + 1)

        AA_init = (A.T @ A).toarray()
        nA_init = np.maximum(np.diag(AA_init), 1e-10).astype(np.float32)
        if init_stride > 1:
            # Strided init returned T_init-length traces; we need full T.
            #
            # A naive full-T projection ``C = (Y.T @ A) / ‖A‖²`` re-introduces
            # the broad 1p background into ``A·C`` from frame 0. That then
            # BLINDS the first ``compute_W`` (fit on the residual
            # ``X = Y − A·C − b0``): the shared background it can no longer see
            # in ``X`` leaks into every neuron trace for the rest of the BCD,
            # a stable bad fixed point. Empirically (PICAST cutout) this made
            # the deconvolved traces share an ~81 %-variance PC1 that is ≈ the
            # global background, median pairwise |r| ~0.45 — and more BCD
            # iterations did NOT escape it (see live_runs/bg_leak_diag.py).
            #
            # Fix: bootstrap a ring background from the *clean* strided greedy
            # traces (``C_init`` comes from the center-surround-filtered movie,
            # so ``A·C_init`` carries little background), then project the
            # full-T init traces through it. This mirrors what the stride==1
            # path gets for free (its first ``compute_W`` sees clean traces),
            # so the first full ``compute_W`` below is no longer blinded.
            # Subsample time like the main ring fit (was tsub=1). This bootstrap
            # only needs a ROUGH W0/b0 to de-leak the strided-init projection, and
            # compute_W clamps actual_tsub to keep >=200 frames, so the per-pixel
            # ring regression stays hugely overdetermined -> W0/b0 ~unchanged but
            # ~bg_tsub x faster (it was the single largest fit_extract stage, run
            # at full time resolution on the strided sample).
            _t_boot = time.perf_counter()
            W0, b0_0 = compute_W(
                make_2d(init_movie), A, C_init, dims, ring_radius,
                lambda_reg=p.ring_lambda, n_jobs=p.n_jobs, device=p.device,
                tsub=p.bg_tsub, constrain_sum=p.ring_constrain_sum,
            )
            timer.add("init bootstrap compute_W", time.perf_counter() - _t_boot)
            # init_movie is no longer needed: project_onto uses the full Y_flat +
            # W0/b0/A, not the strided sample. Free it BEFORE materialising the
            # full movie (fix #2: never hold the full movie and the strided init
            # sample together — that was the ~movie + 2*sample bootstrap co-peak),
            # then bring the full movie in for the projection and the BCD.
            del init_movie
            if Y_flat is None:        # deferred (L3 + stride>1) -> materialise now
                _materialise_full_movie()
                Y_flat = make_2d(movie_arr)
            # project_onto handles numpy and zarr Y_flat (streaming) and
            # returns the background-subtracted projection Y_bg.T @ A → (T, K).
            _t_proj = time.perf_counter()
            YA_init = BackgroundSubtractor(Y_flat, W0, b0_0).project_onto(
                A, n_jobs=p.n_jobs
            )
            timer.add("init trace projection", time.perf_counter() - _t_proj)
            del W0, b0_0
            C_raw = (YA_init / nA_init[None, :]).T.astype(np.float32)     # (K, T)
            C = C_raw.copy()
        else:
            # stride==1: greedy already produced full-T per-pixel-OLS traces.
            C_raw = C_raw_init.astype(np.float32)
            C = C_init.astype(np.float32)
            # (init_movie == movie_arr here; it stays as the working movie.)

        # Estimate the AR coefficient `g` ONCE from the pooled raw traces.
        #
        # Per-component Yule-Walker on T~300 traces has high variance (~0.1
        # spread even on clean ground truth) because the lag-1 / lag-2
        # autocorrelations are noisy. Pooling all components into one long
        # concatenated trace gives an effective sample length of K*T and a
        # much more accurate estimate. We assume all neurons share the same
        # calcium indicator dynamics — the common case for one recording.
        #
        # The estimate is persisted and passed into every update_temporal
        # call instead of being re-estimated each iteration, which would
        # re-apply the fudge_factor 0.96 multiplier to already-deconvolved
        # traces and drift g toward 0.
        K_init = A.shape[1]
        g_per_k: list[np.ndarray] = []
        sn_per_k = np.zeros(K_init, dtype=np.float32)

        _t_g = time.perf_counter()
        if p.global_ar:
            try:
                g_global, _ = estimate_ar_params(
                    C_raw.ravel().astype(np.float32),
                    p=p.ar_order,
                    detrend_order=p.ar_detrend_order,
                    fudge_factor=p.fudge_factor,
                    g_prior=g_target,
                    g_prior_weight=p.g_prior_weight,
                )
            except Exception:
                fallback_g = (
                    float(g_target) if g_target is not None
                    else 0.9 ** (1.0 / max(p.ar_order, 1))
                )
                g_global = np.array([fallback_g] * p.ar_order, dtype=np.float32)
            for k in range(K_init):
                g_per_k.append(g_global.copy())
                sn_per_k[k] = _sn_from_footprint(A[:, k], sn_flat)
        else:
            for k in range(K_init):
                try:
                    g_k, _ = estimate_ar_params(
                        C_raw[k], p=p.ar_order,
                        detrend_order=p.ar_detrend_order,
                        fudge_factor=p.fudge_factor,
                        g_prior=g_target,
                        g_prior_weight=p.g_prior_weight,
                    )
                except Exception:
                    fallback_g = (
                        float(g_target) if g_target is not None
                        else 0.9 ** (1.0 / max(p.ar_order, 1))
                    )
                    g_k = np.array([fallback_g] * p.ar_order, dtype=np.float32)
                g_per_k.append(g_k)
                sn_per_k[k] = _sn_from_footprint(A[:, k], sn_flat)
        timer.add("AR g estimation", time.perf_counter() - _t_g)

        # --- Step 5: Initial ring background ---
        # ring_radius computed above (reused by the init-projection bootstrap).
        print(f"Fitting ring-model background (radius={ring_radius:.1f}px, tsub={p.bg_tsub})...")
        _t = time.perf_counter()
        W_mat, b0 = compute_W(
            Y_flat, A, C, dims, ring_radius,
            lambda_reg=p.ring_lambda, n_jobs=p.n_jobs, device=p.device,
            tsub=p.bg_tsub,
            constrain_sum=p.ring_constrain_sum,
        )
        timer.add("compute_W", time.perf_counter() - _t)

        # --- Step 5b: Rank-1 global background bf · f(t) (opt-in, NON-STANDARD) ---
        bf: np.ndarray | None = None
        f_bg: np.ndarray | None = None
        if p.global_bg_rank == 1:
            print("Fitting rank-1 global background b_f · f(t) (initial)...")
            _t_gbg = time.perf_counter()
            bf, f_bg = _fit_global_bg_rank1(
                Y_flat, A, C, W_mat, b0, bf=None, f=None, n_iter=2,
            )
            timer.add("global bg rank-1", time.perf_counter() - _t_gbg)

        def _cache_after_merge(members_per_group: list[np.ndarray]) -> None:
            """Update g_per_k / sn_per_k after merge_components reorders K."""
            nonlocal g_per_k, sn_per_k
            new_g: list[np.ndarray] = []
            new_sn = np.zeros(len(members_per_group), dtype=np.float32)
            for j, members in enumerate(members_per_group):
                # Inherit from the first (typically strongest-seeded) member.
                # No re-estimation -> no fudge_factor drift.
                m0 = int(members[0])
                new_g.append(g_per_k[m0])
                new_sn[j] = sn_per_k[m0]
            g_per_k = new_g
            sn_per_k = new_sn

        # --- Step 6: Main refinement loop ---
        for iteration in range(p.n_iter_main):
            print(f"Refinement iteration {iteration + 1}/{p.n_iter_main}...")

            # Early merge: catch duplicates from greedy init while their footprints
            # still overlap, before threshold_footprint() in update_spatial separates them.
            if iteration == 0 and A.shape[1] >= 2:
                print("  Pre-merging duplicate seeds...")
                _t_m = time.perf_counter()
                A, C, n_pre_merged, members_per_group = merge_components(
                    A, C_raw,
                    thr_corr=p.merge_thr_corr,
                    thr_overlap=p.merge_thr_overlap,
                    ar_order=p.ar_order,
                    sigma=p.sigma,
                    dims=dims,
                    centre_dist_factor=p.merge_centre_dist_factor,
                )
                timer.add("merge", time.perf_counter() - _t_m)
                _cache_after_merge(members_per_group)
                # Keep C_raw aligned with A's new column order: merge groups
                # become the mean of their members' rows (matches what
                # merge_components does for C).
                C_raw = np.vstack([
                    C_raw[m].mean(axis=0).clip(0) for m in members_per_group
                ]).astype(np.float32)
                if n_pre_merged:
                    print(f"  {A.shape[1]} components ({n_pre_merged} pre-merged).")

            Y_bg = BackgroundSubtractor(Y_flat, W_mat, b0, bf=bf, f=f_bg)  # lazy (H*W, T)

            print("  Updating spatial footprints...")
            _t = time.perf_counter()
            A = update_spatial(
                Y_bg, C, A, sn_flat, dims,
                p.dilation_radius, p.n_jobs, p.spatial_max_thr,
                closing_radius=p.spatial_close_radius,
                max_iter=p.spatial_max_iter,
                tol=p.spatial_tol,
                circular_max_dist_factor=p.spatial_circular_max_dist_factor,
                spatial_ridge=p.spatial_ridge,
                spatial_thread_cap=p.spatial_thread_cap,
                lambda_scale=p.spatial_lambda_scale,
                sigma=p.sigma,
                max_radius_factor=p.spatial_max_radius_factor,
                thr_method=p.spatial_thr_method,
                nrg_thr=p.spatial_nrg_thr,
                spatial_tsub=p.spatial_tsub,
            )
            timer.add("update_spatial", time.perf_counter() - _t)

            # Remove dead components (all-zero footprints)
            nA = np.asarray(A.power(2).sum(axis=0)).ravel()
            alive = nA > 0
            if not alive.all():
                A = A[:, alive]
                C = C[alive]
                C_raw = C_raw[alive]
                alive_idx = np.where(alive)[0]
                g_per_k = [g_per_k[i] for i in alive_idx]
                sn_per_k = sn_per_k[alive_idx]

            if A.shape[1] == 0:
                print("  All components died. Stopping.")
                break

            print("  Updating temporal traces...")
            _deconvolve = (iteration > 0) or (not p.skip_first_deconv)
            _t = time.perf_counter()
            C, S, g_per_k, sn_per_k = update_temporal(
                Y_bg, A, C, sn_flat, p.ar_order, p.n_iter_temporal,
                n_jobs=p.n_jobs, device=p.device,
                g_cached=g_per_k, sn_cached=sn_per_k,
                deconvolve=_deconvolve,
                detrend_order=p.temporal_detrend_order,
                g_prior=g_target, g_prior_weight=p.g_prior_weight,
            )
            timer.add("update_temporal", time.perf_counter() - _t)

            print("  Merging correlated components...")
            _t_m = time.perf_counter()
            A, C, n_merged, members_per_group = merge_components(
                A, C,
                thr_corr=p.merge_thr_corr,
                thr_overlap=p.merge_thr_overlap,
                ar_order=p.ar_order,
                sigma=p.sigma,
                dims=dims,
                centre_dist_factor=p.merge_centre_dist_factor,
            )
            timer.add("merge", time.perf_counter() - _t_m)
            _cache_after_merge(members_per_group)
            C_raw = np.vstack([
                C_raw[m].mean(axis=0).clip(0) for m in members_per_group
            ]).astype(np.float32)
            if n_merged:
                _t = time.perf_counter()
                C, S, g_per_k, sn_per_k = update_temporal(
                    Y_bg, A, C, sn_flat, p.ar_order, 1,
                    n_jobs=p.n_jobs, device=p.device,
                    g_cached=g_per_k, sn_cached=sn_per_k,
                    deconvolve=True,
                    detrend_order=p.temporal_detrend_order,
                    g_prior=g_target, g_prior_weight=p.g_prior_weight,
                )
                timer.add("update_temporal", time.perf_counter() - _t)
            print(f"  {A.shape[1]} components ({n_merged} merged).")

            # Refresh the per-pixel baseline b0 from the refined (A, C).
            # Reuse the ring weight matrix W from the initial solve — the
            # ring's spatial structure is a property of the data, not of A/C,
            # so it remains valid across BCD iterations. Saves the expensive
            # per-pixel BTB solve every iteration (speedup.md Change 2).
            _t = time.perf_counter()
            W_mat, b0 = compute_W(
                Y_flat, A, C, dims, ring_radius,
                lambda_reg=p.ring_lambda, n_jobs=p.n_jobs, device=p.device,
                tsub=p.bg_tsub,
                W_cached=W_mat,
                constrain_sum=p.ring_constrain_sum,
            )
            timer.add("compute_W (b0 refresh)", time.perf_counter() - _t)

            # Refresh the rank-1 global background warm-started from the
            # previous iteration's (bf, f). Two alternating-LS sweeps converge
            # fast since A, C are nearly converged here.
            if p.global_bg_rank == 1:
                bf, f_bg = _fit_global_bg_rank1(
                    Y_flat, A, C, W_mat, b0, bf=bf, f=f_bg, n_iter=2,
                )

        # BCD ended with no surviving components — skip the final temporal
        # pass + YrA recomputation, return empty arrays.
        if A.shape[1] == 0:
            print("No components survived refinement.")
            self.A = A
            self.C = np.empty((0, T), dtype=np.float32)
            self.S = np.empty((0, T), dtype=np.float32)
            self.C_raw = C_raw
            self.YrA = np.empty((0, T), dtype=np.float32)
            self.W = W_mat
            self.b0 = b0
            self.b_f = bf
            self.f = f_bg
            self.g = g_per_k
            self.sn_per_k = sn_per_k
            self.A_norm = np.zeros(0, dtype=np.float32)
            return self

        # Final deconvolution pass to get spike trains
        print("Final temporal update...")
        Y_bg = BackgroundSubtractor(Y_flat, W_mat, b0, bf=bf, f=f_bg)
        _t = time.perf_counter()
        C, S, g_per_k, sn_per_k = update_temporal(
            Y_bg, A, C, sn_flat, p.ar_order, p.n_iter_temporal,
            n_jobs=p.n_jobs, device=p.device,
            g_cached=g_per_k, sn_cached=sn_per_k,
            detrend_order=p.temporal_detrend_order,
            g_prior=g_target, g_prior_weight=p.g_prior_weight,
        )
        timer.add("final update_temporal", time.perf_counter() - _t)

        # Compute the residual projected onto each footprint:
        #   YrA[k, t] = (a_k . (Y_bg - A @ C)[:, t]) / ||a_k||^2
        # The "noisy projected trace" with the same shape as the underlying
        # data is C + YrA. OASIS-deconvolved C alone correlates only ~0.6
        # with ground truth on synthetic data because the shape constraint
        # c[t] >= g * c[t-1] introduces small spike-timing distortions; the
        # noisy projection preserves shape and typically correlates > 0.9.
        AA_final = (A.T @ A).toarray()
        nA_final = np.maximum(np.diag(AA_final), 1e-10)
        _t = time.perf_counter()
        YA_final = Y_bg.project_onto(A, n_jobs=p.n_jobs)                  # (T, K)
        timer.add("final YrA projection", time.perf_counter() - _t)
        crosstalk = AA_final @ C - np.diag(AA_final)[:, None] * C        # (K, T)
        YrA = (YA_final.T - crosstalk) / nA_final[:, None] - C           # (K, T)

        assert C_raw.shape[0] == A.shape[1], (
            f"C_raw/A K mismatch: C_raw has {C_raw.shape[0]} rows, A has {A.shape[1]} cols"
        )
        self.A = A
        self.C = C
        self.S = S
        self.C_raw = C_raw
        self.YrA = YrA
        self.W = W_mat
        self.b0 = b0
        self.b_f = bf
        self.f = f_bg
        self.g = g_per_k
        self.sn_per_k = sn_per_k

        # --- Relabel to CaImAn's scale convention (unit-L2-norm footprints,
        # amplitude in the traces) as the final canonicalization step. A·C is
        # unchanged; the original ‖a_k‖₂ is kept on self.A_norm so the auto-eval
        # SNR (∝ ‖a_k‖²) survives the normalization. Run unconditionally —
        # before the evaluate gate — so fit() and the staged
        # fit_extract(evaluate=False)+evaluate() paths are bit-for-bit identical.
        (self.A, self.C, self.S, self.YrA, self.C_raw,
         self.sn_per_k, self.A_norm) = _normalize_to_trace_amplitude(
            self.A, self.C, self.S, self.YrA, self.C_raw, self.sn_per_k,
        )

        # --- Step 7: Auto-evaluation (informational, non-destructive) ---
        # Reads only self.A + self.sn (+ self.A_norm to recover the original
        # footprint scale), so order relative to the final temporal pass is
        # irrelevant; runs here so a standalone fit_extract is complete.
        if evaluate:
            self.evaluate()

        print(f"Done. Extracted {A.shape[1]} neurons.")
        print(timer.summary())
        return self
