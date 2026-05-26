"""Stage 2 of the staged pipeline: motion correction only.

Reads a (T, H, W) zarr movie and writes a motion-corrected ``mc.zarr`` plus
``shifts.npy`` to the output directory. Streaming: peak RAM is independent of T.

    python run_mc.py /path/to/movie.zarr -o /path/to/stage_dir [options]

Outputs (in <output_dir>/):
    mc.zarr       -- motion-corrected movie (T, H, W) float32
    shifts.npy    -- per-frame (dy, dx) shifts (T, 2)
    params.json   -- the CNMFeParams used (picked up by run_extract.py)

Next stage:
    python run_extract.py <output_dir>/mc.zarr -o <results_dir> --params <output_dir>/params.json

Pass ``--params p.json`` to reuse a saved CNMFeParams (e.g. one rescaled by
run_preprocess.py); CLI flags below override individual fields on top of it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("zarr", type=Path, help="Path to input zarr (T, H, W)")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Stage output dir (default: <zarr_parent>/mc/)")
    parser.add_argument("--params", type=Path, default=None,
                        help="CNMFeParams JSON to load as the base config")
    # Overrides (only applied when explicitly given).
    parser.add_argument("--mc-iter", type=int, default=None,
                        help="Number of rigid MC passes")
    parser.add_argument("--max-shift", type=int, default=None,
                        help="Max allowed shift in pixels (applied to both axes)")
    parser.add_argument("--gsig-filt", type=float, default=None,
                        help="1p high-pass sigma for MC (set ~sigma to enable)")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="CPU workers (-1 = all cores)")
    args = parser.parse_args()

    from cnmfe.io import open_zarr
    from cnmfe.pipeline import CNMFe, CNMFeParams

    zarr_path: Path = args.zarr.resolve()
    if not zarr_path.exists():
        parser.error(f"Zarr not found: {zarr_path}")
    out_dir: Path = args.output if args.output else zarr_path.parent / "mc"
    out_dir.mkdir(parents=True, exist_ok=True)

    params = CNMFeParams.from_json(args.params) if args.params else CNMFeParams()
    if args.mc_iter is not None:
        params.mc_n_iter = args.mc_iter
    if args.max_shift is not None:
        params.max_shift = (args.max_shift, args.max_shift)
    if args.gsig_filt is not None:
        params.mc_gSig_filt = args.gsig_filt
    if args.n_jobs is not None:
        params.n_jobs = args.n_jobs

    z = open_zarr(zarr_path)
    print(f"Movie  : {z.shape}  chunks={z.chunks}  dtype={z.dtype}")
    print(f"Output : {out_dir}")
    print(f"MC     : passes={params.mc_n_iter} max_shift={params.max_shift} "
          f"gSig_filt={params.mc_gSig_filt} n_jobs={params.n_jobs}")

    model = CNMFe(params)
    mc = model.fit_mc(z, output_dir=out_dir)
    np.save(out_dir / "shifts.npy", model.shifts)
    params.to_json(out_dir / "params.json")

    print(f"\nWrote {out_dir}/mc.zarr  shape={mc.shape}")
    print(f"Wrote {out_dir}/shifts.npy  ({model.shifts.shape[0]} × 2)")
    print(f"Next : python run_extract.py {out_dir}/mc.zarr "
          f"-o <results_dir> --params {out_dir}/params.json")


if __name__ == "__main__":
    main()
