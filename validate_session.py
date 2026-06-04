"""Full-recording validation CLI — extract at one or more threshold sets.

Runs the fused full-recording MC once, transposes to Y_flat once, then runs
``fit_extract`` for each ``min_corr:min_pnr`` threshold set (reusing the
threshold-independent Y_flat), with diagnostic figures and a comparison table.
Uses the long-real-recording defaults from
``live_runs/tuning_picast/LEARNINGS.md`` (global_bg_rank=1, low min_pixel + SNR
ghost cut, physical-decay prior, pinned init_stride).

    python validate_session.py /path/to/miniscope_video -o out/ \\
        --indicator gcamp8m --thresholds "0.8:10,0.7:6"

Frame rate and dims are auto-read from the session's metaData.json when present.
"""

from __future__ import annotations

import argparse
from pathlib import Path


_TAU = {"gcamp6f": 140, "jgcamp7f": 160, "gcamp7f": 160, "jgcamp8f": 70,
        "gcamp8f": 70, "jgcamp8m": 180, "gcamp8m": 180, "jgcamp8s": 350,
        "gcamp8s": 350, "gcamp6s": 1000, "gcamp7s": 1000}


def _parse_thresholds(s: "str | None"):
    if not s:
        return None
    out = []
    for i, part in enumerate(s.split(",")):
        mc, mp = part.split(":")
        out.append((f"set{i}", float(mc), float(mp)))
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, help="AVI session folder (0.avi..N.avi)")
    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Output root (default: <input_parent>/validation/)")
    p.add_argument("--frame-rate", type=float, default=None,
                   help="Hz (default: read from metaData.json, else 20)")
    p.add_argument("--decay-time-ms", type=float, default=None)
    p.add_argument("--indicator", type=str, default=None,
                   help="gcamp6f/7f/8f/8m/8s/6s/7s -> decay τ")
    p.add_argument("--ssub", type=int, default=1)
    p.add_argument("--tsub", type=int, default=1)
    p.add_argument("--sigma", type=float, default=6.0, help="NATIVE neuron radius px")
    p.add_argument("--min-corr", type=float, default=0.8)
    p.add_argument("--min-pnr", type=float, default=10.0)
    p.add_argument("--thresholds", type=str, default=None,
                   help='comma list of min_corr:min_pnr, e.g. "0.8:10,0.7:6". '
                        "Default: recommended + a lower-recall set.")
    p.add_argument("--reuse-mc", type=Path, default=None,
                   help="Existing mc.zarr to reuse instead of fusing AVIs")
    p.add_argument("--n-jobs", type=int, default=-1)
    args = p.parse_args()

    if not args.input.exists():
        p.error(f"input not found: {args.input}")

    import matplotlib
    matplotlib.use("Agg")
    from tuning.validate import good_defaults, read_session_meta, validate_session

    meta = read_session_meta(args.input)
    fps = args.frame_rate or meta.get("fps") or 20.0

    decay = args.decay_time_ms
    if decay is None and args.indicator:
        key = args.indicator.lower().replace("-", "").replace("_", "")
        if key not in _TAU:
            p.error(f"unknown indicator {args.indicator}; options: {sorted(_TAU)}")
        decay = float(_TAU[key])
    if decay is None:
        decay = 180.0  # jGCaMP8m default — FLAG and override if the line differs

    native = good_defaults(frame_rate_hz=fps, decay_time_ms=decay, sigma=args.sigma,
                           min_corr=args.min_corr, min_pnr=args.min_pnr, n_jobs=args.n_jobs)
    out = args.output or (args.input.parent / "validation")

    print(f"Validating {args.input}")
    print(f"  fps={fps} (measured {meta.get('fps_measured')})  dims={meta.get('dims')}  "
          f"decay_time_ms={decay} (jGCaMP8m default unless --indicator given)")

    res = validate_session(
        args.input, out, native_params=native, ssub=args.ssub, tsub=args.tsub,
        threshold_sets=_parse_thresholds(args.thresholds), reuse_mc=args.reuse_mc)

    print(f"\nDone ({res['wall_s']:.0f}s). Comparison: {Path(res['out_dir']) / 'comparison.md'}")
    for r in res["rows"]:
        print(f"  [{r['label']}] min_corr={r['min_corr']} min_pnr={r['min_pnr']}: "
              f"K={r['K']} accepted={r['K_accepted']} "
              f"cprojcorr={r['cprojcorr_median']:.3f}")


if __name__ == "__main__":
    main()
