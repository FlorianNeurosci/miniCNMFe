"""Stage 4 of the staged pipeline: (re-)run the non-destructive auto-evaluation.

Loads a results directory written by run_extract.py / full_pipeline.py, runs the
per-component quality tagging (pixel-count floor + scale-invariant mean-amplitude
SNR), and re-saves ``accepted_mask.npy`` / ``eval_info.npz``. No components are
dropped — downstream code filters via ``model.A[:, model.accepted_mask]``.

    python run_evaluate.py /path/to/results [--min-pixel N] [--snr-amp-thr X]

Because evaluation depends only on A + the per-pixel noise map (sn.npy), this is
cheap and can be rerun with different thresholds without re-extracting.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results", type=Path,
                        help="Results dir written by run_extract.py")
    parser.add_argument("--min-pixel", type=int, default=None,
                        help="Override min footprint pixel count")
    parser.add_argument("--snr-amp-thr", type=float, default=None,
                        help="Override mean-amplitude SNR acceptance threshold")
    args = parser.parse_args()

    from minicnmfe.pipeline import CNMFe

    results_dir: Path = args.results.resolve()
    if not results_dir.is_dir():
        parser.error(f"Results dir not found: {results_dir}")

    model = CNMFe.load(results_dir)
    if model.sn is None:
        parser.error("sn.npy missing from results dir; cannot evaluate.")

    if args.min_pixel is not None:
        model.params.min_pixel = args.min_pixel
    if args.snr_amp_thr is not None:
        model.params.auto_eval_snr_amp_thr = args.snr_amp_thr

    model.evaluate()
    model.save(results_dir)
    print(f"Re-evaluated {results_dir}/ "
          f"(min_pixel={model.params.min_pixel}, "
          f"snr_amp_thr={model.params.auto_eval_snr_amp_thr}); "
          f"updated accepted_mask.npy + eval_info.npz.")


if __name__ == "__main__":
    main()
