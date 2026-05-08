"""Run the full CNMFe pipeline on a zarr movie and save results to disk.

Usage:
    python full_pipeline.py /path/to/movie.zarr [options]

The zarr must be a 3-D float32 store with shape (T, H, W), as produced by
concat_avis_to_zarr.py or convert_to_zarr.py.

Results are written to  <output_dir>/  (default: a 'results' folder next to the zarr):
    A.npz        -- sparse spatial footprints (scipy CSC, shape H*W × K)
    C.npy        -- OASIS-deconvolved traces (K × T)
    S.npy        -- spike trains (K × T)
    YrA.npy      -- residuals; C + YrA is the noisy projected trace (K × T)
    shifts.npy   -- per-frame motion correction shifts (T × 2), dy/dx in pixels
    sn.npy       -- per-pixel noise std (H × W)
    params.json  -- all pipeline parameters used
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("zarr", type=Path,
                        help="Path to zarr store (T, H, W)")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output directory (default: <zarr_parent>/results/)")

    # Core params
    parser.add_argument("--sigma",     type=float, default=3.0,
                        help="Neuron radius in pixels (default: 3.0)")
    parser.add_argument("--min-corr",  type=float, default=0.8,
                        help="Minimum local correlation for seed detection (default: 0.8)")
    parser.add_argument("--min-pnr",   type=float, default=10.0,
                        help="Minimum peak-to-noise ratio for seed detection (default: 10.0)")
    parser.add_argument("--n-iter",    type=int,   default=1,
                        help="Number of main refinement cycles (default: 1)")

    # Parallelism
    parser.add_argument("--n-jobs",    type=int,   default=-1,
                        help="CPU workers (-1 = all cores, default: -1)")

    # Motion correction
    parser.add_argument("--no-mc",     action="store_true",
                        help="Skip motion correction")
    parser.add_argument("--mc-iter",   type=int,   default=2,
                        help="Motion correction passes (default: 2)")
    parser.add_argument("--max-shift", type=int,   default=20,
                        help="Max allowed shift in pixels (default: 20)")

    # Merge / spatial / temporal
    parser.add_argument("--merge-corr",    type=float, default=0.85,
                        help="Temporal correlation threshold for merging (default: 0.85)")
    parser.add_argument("--spatial-thr",   type=float, default=0.1,
                        help="Footprint peak-fraction threshold (default: 0.1)")
    parser.add_argument("--global-ar",     action="store_true",
                        help="Use a single pooled AR coefficient (default: per-neuron)")
    args = parser.parse_args()

    zarr_path: Path = args.zarr.resolve()
    if not zarr_path.exists():
        parser.error(f"Zarr not found: {zarr_path}")

    out_dir: Path = args.output if args.output else zarr_path.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load zarr lazily ---------------------------------------------------
    from cnmfe.io import open_zarr
    z = open_zarr(zarr_path)
    T, H, W = z.shape
    print(f"Movie  : {z.shape}  chunks={z.chunks}  dtype={z.dtype}")
    print(f"Output : {out_dir}")

    # --- Build params -------------------------------------------------------
    from cnmfe.pipeline import CNMFe, CNMFeParams
    params = CNMFeParams(
        sigma=args.sigma,
        min_corr=args.min_corr,
        min_pnr=args.min_pnr,
        n_iter_main=args.n_iter,
        n_iter_temporal=2,
        mc_n_iter=args.mc_iter,
        max_shift=(args.max_shift, args.max_shift),
        spatial_max_thr=args.spatial_thr,
        merge_thr_corr=args.merge_corr,
        merge_thr_overlap=0.5,
        global_ar=args.global_ar,
        n_jobs=args.n_jobs,
    )

    print(f"\nParameters:")
    print(f"  sigma={params.sigma}  min_corr={params.min_corr}  min_pnr={params.min_pnr}")
    print(f"  n_iter_main={params.n_iter_main}  n_jobs={params.n_jobs}")
    print(f"  motion_correction={'disabled' if args.no_mc else f'enabled ({params.mc_n_iter} passes)'}")
    print()

    # --- Run pipeline -------------------------------------------------------
    t0 = time.time()
    model = CNMFe(params).fit(z, do_motion_correction=not args.no_mc)
    elapsed = time.time() - t0

    K = model.A.shape[1]
    print(f"\nExtracted {K} neurons in {elapsed:.1f}s")

    # --- Save results -------------------------------------------------------
    import scipy.sparse as sp

    sp.save_npz(out_dir / "A.npz", model.A.tocsc())
    np.save(out_dir / "C.npy",   model.C)
    np.save(out_dir / "S.npy",   model.S)
    np.save(out_dir / "YrA.npy", model.YrA)
    np.save(out_dir / "sn.npy",  model.sn)

    if model.shifts is not None:
        np.save(out_dir / "shifts.npy", model.shifts)
    else:
        np.save(out_dir / "shifts.npy", np.zeros((T, 2), dtype=np.float32))

    params_dict = {
        "zarr": str(zarr_path),
        "movie_shape": list(z.shape),
        "K_extracted": K,
        "wall_time_s": round(elapsed, 2),
        "sigma": params.sigma,
        "min_corr": params.min_corr,
        "min_pnr": params.min_pnr,
        "n_iter_main": params.n_iter_main,
        "n_iter_temporal": params.n_iter_temporal,
        "mc_enabled": not args.no_mc,
        "mc_n_iter": params.mc_n_iter,
        "max_shift": args.max_shift,
        "spatial_max_thr": params.spatial_max_thr,
        "merge_thr_corr": params.merge_thr_corr,
        "merge_thr_overlap": params.merge_thr_overlap,
        "global_ar": params.global_ar,
        "n_jobs": params.n_jobs,
    }
    (out_dir / "params.json").write_text(json.dumps(params_dict, indent=2))

    print(f"\nResults saved to {out_dir}/")
    print(f"  A.npz      -- footprints  ({H * W} × {K}, sparse CSC)")
    print(f"  C.npy      -- deconvolved traces  ({K} × {T})")
    print(f"  S.npy      -- spike trains  ({K} × {T})")
    print(f"  YrA.npy    -- residuals (C + YrA = projected trace)  ({K} × {T})")
    print(f"  shifts.npy -- motion shifts  ({T} × 2)")
    print(f"  sn.npy     -- noise map  ({H} × {W})")
    print(f"  params.json")

    print(f"\nLoad results:")
    print(f"  import numpy as np, scipy.sparse as sp")
    print(f"  A   = sp.load_npz('{out_dir}/A.npz')   # ({H*W}, {K})")
    print(f"  C   = np.load('{out_dir}/C.npy')        # ({K}, {T})")
    print(f"  YrA = np.load('{out_dir}/YrA.npy')      # ({K}, {T})")
    print(f"  # noisy projected trace (shape-faithful):")
    print(f"  C_proj = C + YrA")


if __name__ == "__main__":
    main()
