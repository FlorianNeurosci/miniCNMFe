"""High-level CNMFe pipeline.

Orchestrates motion correction → preprocessing → initialization →
ring background → iterative spatial/temporal refinement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

from cnmfe._utils import make_2d
from cnmfe.background import BackgroundSubtractor, compute_W
from cnmfe.initialization import greedy_corr_pnr
from cnmfe.merging import merge_components
from cnmfe.motion_correction import motion_correction_rigid
from cnmfe.preprocess import correlation_pnr, estimate_noise
from cnmfe.spatial import update_spatial
from cnmfe.temporal import estimate_ar_params, update_temporal

if TYPE_CHECKING:
    import zarr


@dataclass
class CNMFeParams:
    """All CNMFe algorithm parameters."""

    # --- Motion correction ---
    max_shift: tuple[int, int] = (20, 20)
    upsample_factor: int = 10
    mc_n_iter: int = 1                 # number of rigid MC passes (CaImAn default is 1)
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
    seed_suppress_factor: float = 2.0         # Suppression disk radius after extraction = factor * sigma
    circular_max_dist_factor: float = 2.5     # circular_constraint cutoff = factor * estimated_radius

    # --- Background (ring model) ---
    ring_size_factor: float = 1.5  # ring radius = ring_size_factor * (2*sigma+1)
    ring_lambda: float = 1e-5      # Ridge regularization for ring regression

    # --- Spatial update ---
    dilation_radius: int = 3
    spatial_max_thr: float = 0.1    # Zero footprint pixels below this fraction of the peak

    # --- Temporal update / deconvolution ---
    ar_order: int = 1
    global_ar: bool = True  # True = one g estimated from pooled C_raw; False = per-neuron
    n_iter_temporal: int = 2

    # --- Merging ---
    merge_thr_corr: float = 0.85
    merge_thr_overlap: float = 0.5  # Min Jaccard spatial overlap to consider merging
    merge_centre_dist_factor: float = 2.0  # Centre-distance fallback = factor * sigma (in pixels)

    # --- Main loop ---
    n_iter_main: int = 2  # Full spatial + temporal + merge cycles

    # --- Parallelism ---
    n_jobs: int = 1      # Workers for pixel-parallel steps (-1 = all CPUs)
    device: str = "cpu"  # 'cpu' or 'cuda' (requires CuPy + CUDA GPU)

    # --- Statistics sampling ---
    sample_frames: int = 1000  # Max frames used for noise + CORR/PNR (evenly sampled when T > this)

    # --- Speed / accuracy trade-offs ---
    skip_first_deconv: bool = True  # Use NNLS (p=0) for first temporal pass; OASIS on all others
    bg_tsub: int = 5                # Temporal subsampling factor for ring-background W solve


class CNMFe:
    """Clean CNMFe for 1-photon calcium imaging.

    Example::

        from cnmfe import CNMFe, CNMFeParams
        from cnmfe.io import avi_to_zarr

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
        self.sn: np.ndarray | None = None
        self.shifts: np.ndarray | None = None
        self.dims: tuple[int, int] | None = None
        self.g: list[np.ndarray] | None = None    # per-component AR coefs
        self.sn_per_k: np.ndarray | None = None   # per-component noise std
        self.mc_roi: "tuple[slice, slice] | None" = None  # ROI used for shift estimation

    def fit_mc(
        self,
        movie: "zarr.Array | np.ndarray",
        output_dir: str | Path | None = None,
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

        Returns:
            Corrected movie — zarr handle (if output_dir given / zarr input)
            or np.ndarray (if numpy input + no output_dir).
        """
        p = self.params

        # Read shape without materializing the full movie when it's a zarr.
        T, H, W = int(movie.shape[0]), int(movie.shape[1]), int(movie.shape[2])
        self.dims = (H, W)

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
            batch_size=p.mc_batch_size,
            n_jobs=p.n_jobs,
            template_max_frames=p.mc_template_max_frames,
            output_chunk_t=p.mc_output_chunk_t,
            output_dtype=p.mc_output_dtype,
        )
        return corrected

    def fit(
        self,
        movie: "zarr.Array | np.ndarray",
        do_motion_correction: bool = True,
        output_dir: str | Path | None = None,
    ) -> "CNMFe":
        """Run the full CNMFe pipeline on a (T, H, W) movie.

        Args:
            movie: Input movie. zarr.Array or numpy array, shape (T, H, W).
            do_motion_correction: Run rigid motion correction before extraction.
            output_dir: If given, save motion-corrected movie and intermediate
                        results here as zarr stores.

        Returns:
            self (for chaining).
        """
        p = self.params
        movie_arr = np.asarray(movie, dtype=np.float32)
        T, H, W = movie_arr.shape
        dims = (H, W)
        self.dims = dims

        # --- Step 1: Motion correction ---
        if do_motion_correction:
            # In-memory path here: extraction steps below need the full movie
            # in RAM anyway, so streaming-to-zarr would only add disk IO.
            # Use fit_mc(zarr, output_dir=...) directly when the input is too
            # big to materialize.
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

        # --- Steps 2-3: sample frames for statistics if T > sample_frames ---
        T = len(movie_arr)
        stride = max(1, T // p.sample_frames)
        if stride > 1:
            t_idx = np.arange(0, T, stride)
            stats_movie = movie_arr[t_idx]  # advanced indexing already copies
        else:
            stats_movie = movie_arr

        # --- Step 2: Noise estimation ---
        print("Estimating noise...")
        self.sn = estimate_noise(stats_movie)   # (H, W)

        # # --- Step 3: Summary images ---
        # print("Computing CORR and PNR images...")
        # cn, pnr = correlation_pnr(
        #     stats_movie, sigma=p.sigma, center_psf=p.center_psf,
        #     n_jobs=p.n_jobs, device=p.device,
        # )

        # --- Step 4: Initialization ---
        print("Running greedy CORR-PNR initialization...")
        A, C, C_raw, centers = greedy_corr_pnr(
            movie_arr,
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
        )
        print(f"  Found {A.shape[1]} initial components.")

        if A.shape[1] == 0:
            print("No neurons found. Try lowering min_corr or min_pnr.")
            self.A = A
            self.C = C
            self.S = np.empty((0, T), dtype=np.float32)
            self.C_raw = C_raw
            return self

        # Flatten movie to (H*W, T) for all subsequent steps
        Y_flat = make_2d(movie_arr)     # (H*W, T)
        sn_flat = self.sn.ravel()       # (H*W,)

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

        if p.global_ar:
            try:
                g_global, _ = estimate_ar_params(C_raw.ravel().astype(np.float32), p=p.ar_order)
            except Exception:
                g_global = np.array([0.9 ** (1.0 / max(p.ar_order, 1))] * p.ar_order,
                                    dtype=np.float32)
            for k in range(K_init):
                g_per_k.append(g_global.copy())
                try:
                    _, sn_k = estimate_ar_params(C_raw[k], p=p.ar_order)
                except Exception:
                    sn_k = float(np.std(C_raw[k])) if np.std(C_raw[k]) > 0 else 1.0
                sn_per_k[k] = sn_k
        else:
            for k in range(K_init):
                try:
                    g_k, sn_k = estimate_ar_params(C_raw[k], p=p.ar_order)
                except Exception:
                    g_k = np.array([0.9 ** (1.0 / max(p.ar_order, 1))] * p.ar_order,
                                   dtype=np.float32)
                    sn_k = float(np.std(C_raw[k])) if np.std(C_raw[k]) > 0 else 1.0
                g_per_k.append(g_k)
                sn_per_k[k] = sn_k

        # --- Step 5: Initial ring background ---
        ring_radius = p.ring_size_factor * (2 * p.sigma + 1)
        print(f"Fitting ring-model background (radius={ring_radius:.1f}px, tsub={p.bg_tsub})...")
        W_mat, b0 = compute_W(
            Y_flat, A, C, dims, ring_radius,
            lambda_reg=p.ring_lambda, n_jobs=p.n_jobs, device=p.device,
            tsub=p.bg_tsub,
        )

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
                A, C, n_pre_merged, members_per_group = merge_components(
                    A, C_raw,
                    thr_corr=p.merge_thr_corr,
                    thr_overlap=p.merge_thr_overlap,
                    ar_order=p.ar_order,
                    sigma=p.sigma,
                    dims=dims,
                    centre_dist_factor=p.merge_centre_dist_factor,
                )
                _cache_after_merge(members_per_group)
                if n_pre_merged:
                    print(f"  {A.shape[1]} components ({n_pre_merged} pre-merged).")

            Y_bg = BackgroundSubtractor(Y_flat, W_mat, b0)  # lazy (H*W, T)

            print("  Updating spatial footprints...")
            A = update_spatial(Y_bg, C, A, sn_flat, dims, p.dilation_radius, p.n_jobs, p.spatial_max_thr)

            # Remove dead components (all-zero footprints)
            nA = np.asarray(A.power(2).sum(axis=0)).ravel()
            alive = nA > 0
            if not alive.all():
                A = A[:, alive]
                C = C[alive]
                alive_idx = np.where(alive)[0]
                g_per_k = [g_per_k[i] for i in alive_idx]
                sn_per_k = sn_per_k[alive_idx]

            if A.shape[1] == 0:
                print("  All components died. Stopping.")
                break

            print("  Updating temporal traces...")
            _deconvolve = (iteration > 0) or (not p.skip_first_deconv)
            C, S, g_per_k, sn_per_k = update_temporal(
                Y_bg, A, C, sn_flat, p.ar_order, p.n_iter_temporal,
                n_jobs=p.n_jobs, device=p.device,
                g_cached=g_per_k, sn_cached=sn_per_k,
                deconvolve=_deconvolve,
            )

            print("  Merging correlated components...")
            A, C, n_merged, members_per_group = merge_components(
                A, C,
                thr_corr=p.merge_thr_corr,
                thr_overlap=p.merge_thr_overlap,
                ar_order=p.ar_order,
                sigma=p.sigma,
                dims=dims,
                centre_dist_factor=p.merge_centre_dist_factor,
            )
            _cache_after_merge(members_per_group)
            if n_merged:
                C, S, g_per_k, sn_per_k = update_temporal(
                    Y_bg, A, C, sn_flat, p.ar_order, 1,
                    n_jobs=p.n_jobs, device=p.device,
                    g_cached=g_per_k, sn_cached=sn_per_k,
                    deconvolve=True,
                )
            print(f"  {A.shape[1]} components ({n_merged} merged).")

            # Refresh the per-pixel baseline b0 from the refined (A, C).
            # Reuse the ring weight matrix W from the initial solve — the
            # ring's spatial structure is a property of the data, not of A/C,
            # so it remains valid across BCD iterations. Saves the expensive
            # per-pixel BTB solve every iteration (speedup.md Change 2).
            W_mat, b0 = compute_W(
                Y_flat, A, C, dims, ring_radius,
                lambda_reg=p.ring_lambda, n_jobs=p.n_jobs, device=p.device,
                tsub=p.bg_tsub,
                W_cached=W_mat,
            )

        # Final deconvolution pass to get spike trains
        print("Final temporal update...")
        Y_bg = BackgroundSubtractor(Y_flat, W_mat, b0)
        C, S, g_per_k, sn_per_k = update_temporal(
            Y_bg, A, C, sn_flat, p.ar_order, p.n_iter_temporal,
            n_jobs=p.n_jobs, device=p.device,
            g_cached=g_per_k, sn_cached=sn_per_k,
        )

        # Compute the residual projected onto each footprint:
        #   YrA[k, t] = (a_k . (Y_bg - A @ C)[:, t]) / ||a_k||^2
        # The "noisy projected trace" with the same shape as the underlying
        # data is C + YrA. OASIS-deconvolved C alone correlates only ~0.6
        # with ground truth on synthetic data because the shape constraint
        # c[t] >= g * c[t-1] introduces small spike-timing distortions; the
        # noisy projection preserves shape and typically correlates > 0.9.
        AA_final = (A.T @ A).toarray()
        nA_final = np.maximum(np.diag(AA_final), 1e-10)
        YA_final = Y_bg.project_onto(A)                                   # (T, K)
        crosstalk = AA_final @ C - np.diag(AA_final)[:, None] * C        # (K, T)
        YrA = (YA_final.T - crosstalk) / nA_final[:, None] - C           # (K, T)

        self.A = A
        self.C = C
        self.S = S
        self.C_raw = C_raw
        self.YrA = YrA
        self.W = W_mat
        self.b0 = b0
        self.g = g_per_k
        self.sn_per_k = sn_per_k
        print(f"Done. Extracted {A.shape[1]} neurons.")
        return self
