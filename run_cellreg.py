"""Cross-session cell registration (CellReg-style).

Register the same neurons across multiple already-extracted sessions. Each
session is a results dir written by ``model.save`` (e.g. by run_extract.py).
Footprints are rigid-body aligned to a reference, nearby cell pairs are scored
by centroid distance + spatial correlation, matched one-to-one per session
pair, then clustered into a ``cell_to_index_map`` (global cells x sessions).

    python run_cellreg.py session1/results session2/results [...] -o reg/

Distance thresholds are in pixels unless --microns-per-pixel is given, in which
case --max-distance-um / --dist-thr-um are interpreted in microns.

Output (in the results dir):
    cell_to_index_map.npy   (n_global, n_sessions) int, -1 = absent
    transforms.npy          (n_sessions, 3) rigid (dy, dx, theta)
    aligned_centroids.npz   per-session aligned centroids
    cellreg_info.json       metadata + params
    run_info.json           run metadata
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("sessions", type=Path, nargs="+",
                        help="Results dirs (>= 2), each written by model.save")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output dir (default: <first_session>/cellreg/)")
    parser.add_argument("--microns-per-pixel", type=float, default=None,
                        help="Physical scale; makes distance thresholds microns")
    parser.add_argument("--max-distance-um", type=float, default=12.0,
                        help="Neighbour search radius (um if --microns-per-pixel)")
    parser.add_argument("--max-distance-px", type=float, default=None,
                        help="Neighbour search radius in px (overrides --max-distance-um)")
    parser.add_argument("--align", choices=["translation", "rotation", "none"],
                        default="translation", help="Rigid alignment mode")
    parser.add_argument("--reference", type=int, default=0,
                        help="Index of the reference session for alignment")
    parser.add_argument("--accepted-only", action="store_true",
                        help="Use only auto-eval accepted components when available")
    # Phase 1 thresholds
    parser.add_argument("--dist-thr-um", type=float, default=5.0,
                        help="Match distance threshold (um if --microns-per-pixel)")
    parser.add_argument("--dist-thr-px", type=float, default=None,
                        help="Match distance threshold in px (overrides --dist-thr-um)")
    parser.add_argument("--corr-thr", type=float, default=0.65,
                        help="Match spatial-correlation threshold")
    parser.add_argument("--corr-weight", type=float, default=0.5,
                        help="Score blend: 0 = distance only, 1 = correlation only")
    # CellReg P_same model (DEFAULT) — data-driven threshold, auto model select
    parser.add_argument("--registration-approach",
                        choices=["probabilistic_model", "threshold"],
                        default="probabilistic_model",
                        help="DEFAULT probabilistic_model = CellReg lognormal/beta "
                             "P_same with a data-driven threshold (auto-falls back "
                             "to threshold matching on sparse/degenerate fields); "
                             "threshold = hand-set corr_thr/dist_thr")
    parser.add_argument("--psame-feature", choices=["auto", "spatial", "centroid"],
                        default="auto",
                        help="P_same feature for the model approach (auto = "
                             "choose_best_model by FP+FN+MSE)")
    parser.add_argument("--clustering", choices=["iterative", "greedy"],
                        default="iterative",
                        help="Cluster refinement for the model approach")
    parser.add_argument("--p-same-thr", type=float, default=0.5,
                        help="P_same acceptance cutoff")
    # Legacy GMM probabilistic (only with --registration-approach threshold)
    parser.add_argument("--probabilistic", action="store_true",
                        help="Legacy GMM P_same (requires --registration-approach "
                             "threshold)")
    parser.add_argument("--model", choices=["centroid", "spatial", "joint"],
                        default="spatial", help="legacy GMM feature(s)")
    parser.add_argument("--min-sessions", type=int, default=2,
                        help="Report cells present in >= this many sessions")
    args = parser.parse_args()

    if len(args.sessions) < 2:
        parser.error("need at least 2 sessions")

    from minicnmfe.cellreg import register_sessions

    out_dir = args.output or (args.sessions[0] / "cellreg")

    t0 = time.time()
    result = register_sessions(
        [str(p) for p in args.sessions],
        microns_per_pixel=args.microns_per_pixel,
        max_distance_um=args.max_distance_um,
        max_distance_px=args.max_distance_px,
        align=args.align,
        reference=args.reference,
        accepted_only=args.accepted_only,
        dist_thr_um=args.dist_thr_um,
        dist_thr_px=args.dist_thr_px,
        corr_thr=args.corr_thr,
        corr_weight=args.corr_weight,
        registration_approach=args.registration_approach,
        psame_feature=args.psame_feature,
        clustering=args.clustering,
        probabilistic=args.probabilistic,
        model=args.model,
        p_same_thr=args.p_same_thr,
    )
    elapsed = time.time() - t0

    result.save(out_dir)

    n_reg = result.n_registered(min_sessions=args.min_sessions)
    print(f"Registered {result.n_sessions} sessions in {elapsed:.1f}s")
    print(f"  approach: {result.params.get('registration_approach')}"
          f" (feature={result.params.get('psame_feature')},"
          f" data-driven thr={result.params.get('data_driven_threshold')})")
    print(f"  global cells: {result.n_global}")
    print(f"  present in >= {args.min_sessions} sessions: {n_reg}")
    if result.uncertain_fraction is not None:
        print(f"  uncertain fraction: {result.uncertain_fraction:.3f}"
              f"  (model FP={result.model_false_positive:.3f},"
              f" FN={result.model_false_negative:.3f})")
    print(f"  saved to: {out_dir}")

    run_info = {
        "sessions": [str(p) for p in args.sessions],
        "output_dir": str(out_dir),
        "elapsed_s": elapsed,
        "n_global": result.n_global,
        "n_registered_min_sessions": {str(args.min_sessions): n_reg},
        "transforms": result.transforms.tolist(),
        "params": result.params,
    }
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))


if __name__ == "__main__":
    main()
