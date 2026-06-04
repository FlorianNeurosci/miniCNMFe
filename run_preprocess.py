"""Stage 1 of the staged pipeline: spatial + temporal downsampling.

Bins a (T, H, W) zarr movie by ``--ssub`` (space) and ``--tsub`` (time), writing
a smaller time-major zarr plus ``ds_meta.json``. Motion correction, extraction
and evaluation then run on the downsampled movie; pass the ``ds_meta.json`` to
``run_extract.py --ds-meta`` so native-unit parameters are auto-rescaled.

    python run_preprocess.py /path/to/movie.zarr -o /path/to/ds.zarr --ssub 2 --tsub 2

Next stage:
    python run_mc.py <ds.zarr> -o <mc_dir>
    python run_extract.py <mc_dir>/mc.zarr -o <results> --ds-meta <ds_dir>/ds_meta.json
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("zarr", type=Path, help="Path to input zarr (T, H, W)")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output zarr path (default: <zarr_parent>/ds.zarr)")
    parser.add_argument("--ssub", type=int, default=1,
                        help="Spatial bin factor (default: 1 = no spatial bin)")
    parser.add_argument("--tsub", type=int, default=1,
                        help="Temporal bin factor (default: 1 = no temporal bin)")
    parser.add_argument("--chunk-t", type=int, default=500,
                        help="Output zarr time chunk (default: 500)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rewrite the output even if it already exists")
    args = parser.parse_args()

    from minicnmfe.downsample import downsample_movie

    zarr_path: Path = args.zarr.resolve()
    if not zarr_path.exists():
        parser.error(f"Zarr not found: {zarr_path}")
    if args.ssub < 1 or args.tsub < 1:
        parser.error("--ssub and --tsub must be >= 1")
    dest: Path = args.output if args.output else zarr_path.parent / "ds.zarr"

    out = downsample_movie(
        zarr_path, dest,
        ssub=args.ssub, tsub=args.tsub,
        chunk_t=args.chunk_t,
        skip_if_exists=not args.overwrite,
    )
    print(f"\nWrote {dest}  shape={out.shape}")
    print(f"Metadata: {dest.parent / 'ds_meta.json'}")
    print(f"Next : python run_mc.py {dest} -o <mc_dir>")


if __name__ == "__main__":
    main()
