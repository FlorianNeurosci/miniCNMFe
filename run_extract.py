"""Stage 3 of the staged pipeline: extraction on an already-corrected movie.

Runs greedy init -> ring background -> spatial/temporal/merge BCD -> final
temporal pass + YrA (and, unless --no-eval, the non-destructive auto-eval) on a
motion-corrected (T, H, W) zarr. Saves all result arrays via ``model.save``.

    python run_extract.py /path/to/mc.zarr -o /path/to/results [options]

For a downsampled run, point --ds-meta at the ``ds_meta.json`` written by
run_preprocess.py: parameters expressed in NATIVE units are auto-rescaled to the
downsampled grid via ``CNMFeParams.downscaled`` before extraction.

By default a zarr input is processed in streaming mode (a pixel-major Y_flat
store is derived under the results dir; peak RAM independent of T). Use
--in-memory to materialise the movie instead (small movies / tests).

Next stage:
    python run_evaluate.py <results_dir>            # retune accept thresholds
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("zarr", type=Path, help="Path to corrected zarr (T, H, W)")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Results dir (default: <zarr_parent>/results/)")
    parser.add_argument("--params", type=Path, default=None,
                        help="CNMFeParams JSON to load as the base config")
    parser.add_argument("--ds-meta", type=Path, default=None,
                        help="ds_meta.json from run_preprocess.py; rescales "
                             "native-unit params to the downsampled grid")
    parser.add_argument("--in-memory", action="store_true",
                        help="Materialise the movie in RAM instead of streaming")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip the auto-evaluation pass (run it later with "
                             "run_evaluate.py)")
    # Overrides (only applied when explicitly given).
    parser.add_argument("--sigma", type=float, default=None,
                        help="Neuron radius in pixels (NATIVE units if --ds-meta)")
    parser.add_argument("--min-corr", type=float, default=None)
    parser.add_argument("--min-pnr", type=float, default=None)
    parser.add_argument("--n-iter", type=int, default=None,
                        help="Main refinement cycles")
    parser.add_argument("--n-jobs", type=int, default=None)
    # Streaming store layout (only affects on-disk IO speed, not results).
    parser.add_argument("--yflat-dir", type=Path, default=None,
                        help="Where to write the pixel-major Y_flat store "
                             "(default: under the results dir). Point at a LOCAL "
                             "SSD/tmpfs to stage off a network mount.")
    parser.add_argument("--yflat-pixel-chunk", type=int, default=None,
                        help="Pixels per Y_flat chunk (default 512)")
    parser.add_argument("--yflat-time-chunk", type=int, default=None,
                        help="Frames per Y_flat chunk (default: full T)")
    parser.add_argument("--yflat-no-compress", action="store_true",
                        help="Write Y_flat uncompressed (try on local SSD)")
    args = parser.parse_args()

    from minicnmfe.io import open_zarr
    from minicnmfe.pipeline import CNMFe, CNMFeParams

    zarr_path: Path = args.zarr.resolve()
    if not zarr_path.exists():
        parser.error(f"Zarr not found: {zarr_path}")
    out_dir: Path = args.output if args.output else zarr_path.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    params = CNMFeParams.from_json(args.params) if args.params else CNMFeParams()
    # Native-unit overrides applied BEFORE downscaling so the rescale is correct.
    if args.sigma is not None:
        params.sigma = args.sigma
    if args.min_corr is not None:
        params.min_corr = args.min_corr
    if args.min_pnr is not None:
        params.min_pnr = args.min_pnr
    if args.n_iter is not None:
        params.n_iter_main = args.n_iter
    if args.n_jobs is not None:
        params.n_jobs = args.n_jobs
    # Streaming store layout (resolution-independent; downscaled() preserves these).
    if args.yflat_dir is not None:
        params.yflat_dir = str(args.yflat_dir)
    if args.yflat_pixel_chunk is not None:
        params.yflat_pixel_chunk = args.yflat_pixel_chunk
    if args.yflat_time_chunk is not None:
        params.yflat_time_chunk = args.yflat_time_chunk
    if args.yflat_no_compress:
        params.yflat_compression = False

    ssub = tsub = 1
    if args.ds_meta is not None:
        meta = json.loads(Path(args.ds_meta).read_text())
        ssub, tsub = int(meta["ssub"]), int(meta["tsub"])
        params = params.downscaled(ssub, tsub)
        print(f"Downsampled run: ssub={ssub} tsub={tsub} -> "
              f"sigma={params.sigma:.3f} min_pixel={params.min_pixel} "
              f"max_shift={params.max_shift}")

    z = open_zarr(zarr_path)
    print(f"Movie  : {z.shape}  chunks={z.chunks}  dtype={z.dtype}")
    print(f"Output : {out_dir}")

    model = CNMFe(params)
    t0 = time.time()
    if args.in_memory:
        movie = np.asarray(z, dtype=np.float32)
        model.fit_extract(movie, evaluate=not args.no_eval)
    else:
        # zarr + output_dir => streaming: Y_flat pixel-major store derived here.
        model.fit_extract(z, output_dir=out_dir, evaluate=not args.no_eval)
    elapsed = time.time() - t0

    K = model.A.shape[1]
    print(f"\nExtracted {K} neurons in {elapsed:.1f}s")

    model.save(out_dir)
    run_info = {
        "zarr": str(zarr_path),
        "movie_shape": list(z.shape),
        "K_extracted": K,
        "wall_time_s": round(elapsed, 2),
        "ssub": ssub,
        "tsub": tsub,
        "evaluated": not args.no_eval,
    }
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))
    print(f"Results saved to {out_dir}/")
    if args.no_eval:
        print(f"Next : python run_evaluate.py {out_dir}")


if __name__ == "__main__":
    main()
