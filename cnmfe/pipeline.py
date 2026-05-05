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
from cnmfe.background import compute_W, subtract_background
from cnmfe.initialization import greedy_corr_pnr
from cnmfe.merging import merge_components
from cnmfe.motion_correction import motion_correct
from cnmfe.preprocess import correlation_pnr, estimate_noise
from cnmfe.spatial import update_spatial
from cnmfe.temporal import update_temporal

if TYPE_CHECKING:
    import zarr


@dataclass
class CNMFeParams:
    """All CNMFe algorithm parameters."""

    # --- Motion correction ---
    max_shift: tuple[int, int] = (20, 20)
    upsample_factor: int = 10
    mc_n_iter: int = 2

    # --- Spatial filtering / PSF ---
    sigma: float = 3.0        # Gaussian sigma in pixels (neuron size)
    center_psf: bool = True   # Use center-surround kernel for 1p background rejection

    # --- Initialization (GreedyCorr) ---
    min_corr: float = 0.8
    min_pnr: float = 10.0
    min_pixel: int = 3        # Minimum nonzero pixels in a valid footprint
    border_px: int = 5        # Ignore seeds within this many border pixels
    max_neurons: int | None = None  # Stop early (None = no limit)

    # --- Background (ring model) ---
    ring_size_factor: float = 1.5  # ring radius = ring_size_factor * (2*sigma+1)
    ring_lambda: float = 1e-5      # Ridge regularization for ring regression

    # --- Spatial update ---
    dilation_radius: int = 3

    # --- Temporal update / deconvolution ---
    ar_order: int = 1
    n_iter_temporal: int = 2

    # --- Merging ---
    merge_thr_corr: float = 0.85

    # --- Main loop ---
    n_iter_main: int = 2  # Full spatial + temporal + merge cycles

    # --- Parallelism ---
    n_jobs: int = 1      # Workers for pixel-parallel steps (-1 = all CPUs)
    device: str = "cpu"  # 'cpu' or 'cuda' (requires CuPy + CUDA GPU)


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
        self.W: sp.csr_matrix | None = None
        self.b0: np.ndarray | None = None
        self.sn: np.ndarray | None = None
        self.shifts: np.ndarray | None = None
        self.dims: tuple[int, int] | None = None

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
            mc_path = Path(output_dir) / "mc.zarr" if output_dir else None
            movie_arr, self.shifts = motion_correct(
                movie_arr,
                upsample_factor=p.upsample_factor,
                max_shift=p.max_shift,
                n_iter=p.mc_n_iter,
                output_path=mc_path,
                n_jobs=p.n_jobs,
                device=p.device,
            )
            movie_arr = np.asarray(movie_arr, dtype=np.float32)

        # --- Step 2: Noise estimation ---
        print("Estimating noise...")
        self.sn = estimate_noise(movie_arr)   # (H, W)

        # --- Step 3: Summary images ---
        print("Computing CORR and PNR images...")
        cn, pnr = correlation_pnr(
            movie_arr, sigma=p.sigma, center_psf=p.center_psf,
            n_jobs=p.n_jobs, device=p.device,
        )

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

        # --- Step 5: Initial ring background ---
        ring_radius = p.ring_size_factor * (2 * p.sigma + 1)
        print(f"Fitting ring-model background (radius={ring_radius:.1f}px)...")
        W_mat, b0 = compute_W(
            Y_flat, A, C, dims, ring_radius,
            lambda_reg=p.ring_lambda, n_jobs=p.n_jobs, device=p.device,
        )

        # --- Step 6: Main refinement loop ---
        for iteration in range(p.n_iter_main):
            print(f"Refinement iteration {iteration + 1}/{p.n_iter_main}...")

            Y_bg = subtract_background(Y_flat, W_mat, b0)  # (H*W, T)

            print("  Updating spatial footprints...")
            A = update_spatial(Y_bg, C, A, sn_flat, dims, p.dilation_radius, p.n_jobs)

            # Remove dead components (all-zero footprints)
            nA = np.asarray(A.power(2).sum(axis=0)).ravel()
            alive = nA > 0
            if not alive.all():
                A = A[:, alive]
                C = C[alive]

            if A.shape[1] == 0:
                print("  All components died. Stopping.")
                break

            print("  Updating temporal traces...")
            C, S = update_temporal(
                Y_bg, A, C, sn_flat, p.ar_order, p.n_iter_temporal,
                n_jobs=p.n_jobs, device=p.device,
            )

            print("  Merging correlated components...")
            A, C, n_merged = merge_components(A, C, thr_corr=p.merge_thr_corr)
            if n_merged:
                C, S = update_temporal(
                    Y_bg, A, C, sn_flat, p.ar_order, 1,
                    n_jobs=p.n_jobs, device=p.device,
                )
            print(f"  {A.shape[1]} components ({n_merged} merged).")

            # Update background with refined components
            W_mat, b0 = compute_W(
                Y_flat, A, C, dims, ring_radius,
                lambda_reg=p.ring_lambda, n_jobs=p.n_jobs, device=p.device,
            )

        # Final deconvolution pass to get spike trains
        print("Final temporal update...")
        Y_bg = subtract_background(Y_flat, W_mat, b0)
        C, S = update_temporal(
            Y_bg, A, C, sn_flat, p.ar_order, p.n_iter_temporal,
            n_jobs=p.n_jobs, device=p.device,
        )

        self.A = A
        self.C = C
        self.S = S
        self.C_raw = C_raw
        self.W = W_mat
        self.b0 = b0
        print(f"Done. Extracted {A.shape[1]} neurons.")
        return self
