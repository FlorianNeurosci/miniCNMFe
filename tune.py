"""CNMFe parameter-tuning CLI — the single fire-and-forget front door.

Point it at one recording (an AVI folder or an ``mc.zarr``) and it tests the
recording, suggests motion-correction + extraction parameters, runs a graded
extraction sweep, (by default) validates the recommendation on the FULL
recording, and writes a self-contained review folder so you can judge quality
by eye — no AI needed:

    python tune.py /path/to/avis --indicator gcamp8m --n-jobs -1 &   # launch, walk away
    # then open  runs/tune_<name>_<ts>/report.html  in a browser

Batch (one background process, BLAS-capped, never sub-agents):

    python tune.py --sessions sessions.txt -o runs/batch --indicator gcamp8m --jobs 2 --cores 6 &

Outputs in the run folder: ``report.html`` (open this), ``report.md``,
``recommended_params.json`` + ``downsample.json``, ``fig_*.png``, and (when
validating) ``full/`` with ``comparison.md`` + per-run diagnostic figures.

Two depth modes (``--mode``):
  heuristic   fast image-based suggestions only (no full extraction)
  sweep       run fit_extract across a grid of key knobs, score each candidate
  both        heuristics seed the grid, sweep refines + adds temporal knobs

Two sweep regions (``--region``):
  cutout      run the grid on a representative spatial+temporal window (fast, default)
  full        run the grid on the whole recording (faithful, slow)

The ``recommended_params.json`` is written in NATIVE units and feeds the staged
pipeline directly:

    python run_mc.py <movie> -o mc/ --params <run>/recommended_params.json
    python run_extract.py mc/mc.zarr -o results/ --params <run>/recommended_params.json \\
        --ds-meta <run>/downsample.json
"""

from __future__ import annotations

import argparse
import shlex
import sys
import time
from pathlib import Path


def _floats(s: "str | None"):
    return None if not s else [float(x) for x in s.split(",") if x.strip()]


def _ints(s: "str | None"):
    return None if not s else [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, nargs="?", default=None,
                        help="AVI folder (0.avi..N.avi) or an mc.zarr path "
                             "(omit when using --sessions)")
    parser.add_argument("--sessions", type=str, nargs="+", default=None,
                        help="Batch mode: session paths and/or a .txt list — "
                             "delegates to batch_tune (one background process)")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Parent for the timestamped run folder "
                             "(default: ./runs/, which is gitignored)")
    parser.add_argument("--params", type=Path, default=None,
                        help="Base CNMFeParams JSON (heuristics/sweep override on top)")
    parser.add_argument("--mode", choices=["heuristic", "sweep", "both"], default="both")
    parser.add_argument("--region", choices=["cutout", "full"], default="cutout")

    parser.add_argument("--frame-rate", type=float, default=None,
                        help="Acquisition frame rate (Hz, native); "
                             "default: read from metaData.json (AVI input), else 20")
    parser.add_argument("--decay-time-ms", type=float, default=None,
                        help="Indicator single-AP decay τ (ms)")
    parser.add_argument("--indicator", type=str, default=None,
                        help="Shortcut for --decay-time-ms (gcamp6f/7f/8f/8m/8s/6s/7s)")

    parser.add_argument("--ssub", type=int, default=None, help="Spatial bin (AVI input; auto if unset)")
    parser.add_argument("--tsub", type=int, default=None, help="Temporal bin (AVI input; auto if unset)")
    parser.add_argument("--reuse-mc-zarr", type=Path, default=None,
                        help="Use this mc.zarr for extraction tuning (skip quick MC)")
    parser.add_argument("--max-avis", type=int, default=None,
                        help="Cap AVIs fused in the quick MC (evenly-spaced subset)")
    parser.add_argument("--pattern", default="*.avi")

    parser.add_argument("--n-template-avis", type=int, default=8)
    parser.add_argument("--stride-within-avi", type=int, default=50)
    parser.add_argument("--n-init-frames", type=int, default=400)
    parser.add_argument("--n-shift-frames", type=int, default=200)
    parser.add_argument("--cutout-hw", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--window-t", type=int, default=3000)

    parser.add_argument("--grid-sigma", type=str, default=None, help="comma list, e.g. 3,4,5")
    parser.add_argument("--grid-min-corr", type=str, default=None)
    parser.add_argument("--grid-min-pnr", type=str, default=None)
    parser.add_argument("--grid-merge-thr", type=str, default=None)
    parser.add_argument("--grid-bg-rank", type=str, default=None, help="comma list, e.g. 0,1")
    parser.add_argument("--grid-init-stride", type=str, default=None)
    parser.add_argument("--max-candidates", type=int, default=24)
    parser.add_argument("--existing-results", type=Path, default=None,
                        help="Pre-fit results dir for the temporal heuristics "
                             "(when not running a sweep)")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="total core budget for the sweep; split across "
                             "candidate-level processes x inner per-fit threads "
                             "(default 1 = fully serial). Set near your physical "
                             "core count.")

    # -- consolidated front-door flags --
    parser.add_argument("--validate", action=argparse.BooleanOptionalAction, default=None,
                        help="Validate the recommendation on the FULL recording "
                             "(default: on for an AVI folder, off for an mc.zarr)")
    parser.add_argument("--no-lowthr", action="store_true",
                        help="Validation: single threshold set (skip the lower-recall compare)")
    parser.add_argument("--html", action=argparse.BooleanOptionalAction, default=True,
                        help="Write the self-contained report.html (default: on)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the resolved plan + output layout and exit")
    # batch-mode passthroughs (used only with --sessions)
    parser.add_argument("--jobs", type=int, default=2, help="(batch) sessions run concurrently")
    parser.add_argument("--cores", type=int, default=6, help="(batch) cores per session")

    args = parser.parse_args()

    if args.sessions and args.input is not None:
        parser.error("give either a single input or --sessions, not both")
    if not args.sessions and args.input is None:
        parser.error("an input path (or --sessions) is required")
    if args.input is not None and not args.input.exists():
        parser.error(f"input not found: {args.input}")

    # -- Batch mode: delegate to the BLAS-capped background orchestrator. --
    if args.sessions:
        import batch_tune
        out = args.output or (Path.cwd() / "runs" / "batch")
        batch_tune.run_batch(
            args.sessions, out, indicator=args.indicator or "gcamp8m",
            jobs=args.jobs, cores=args.cores, lowthr=not args.no_lowthr,
            dry_run=args.dry_run)
        return

    _TAU = {"gcamp6f": 140, "jgcamp7f": 160, "gcamp7f": 160, "jgcamp8f": 70,
            "gcamp8f": 70, "jgcamp8m": 180, "gcamp8m": 180, "jgcamp8s": 350,
            "gcamp8s": 350, "gcamp6s": 1000, "gcamp7s": 1000}
    decay = args.decay_time_ms
    if decay is None and args.indicator:
        key = args.indicator.lower().replace("-", "").replace("_", "")
        if key not in _TAU:
            parser.error(f"unknown indicator {args.indicator}; options: {sorted(_TAU)}")
        decay = float(_TAU[key])
    if decay is None:
        decay = 180.0  # GCaMP8m default

    kind = "avi" if (args.input.is_dir() and not str(args.input).endswith(".zarr")) else "zarr"
    do_validate = args.validate if args.validate is not None else (kind == "avi")
    out_parent = args.output or (Path.cwd() / "runs")
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(out_parent) / f"tune_{args.input.name}_{ts}"

    # -- dry-run: print the resolved plan + output layout, exit (cheap, testable) --
    if args.dry_run:
        print(f"[dry-run] tune: {args.input}  (kind={kind}, mode={args.mode}, "
              f"region={args.region}, decay_time_ms={decay})")
        print(f"[dry-run] validate={do_validate} (lowthr={not args.no_lowthr})  "
              f"html={args.html}  n_jobs={args.n_jobs}")
        print(f"[dry-run] output folder: {run_dir}")
        print(f"[dry-run]   report.html  report.md  recommended_params.json  "
              f"downsample.json  fig_*.png" + ("  full/" if do_validate else ""))
        return

    # Lazy imports so --help / --dry-run stay instant and matplotlib is headless.
    import matplotlib
    matplotlib.use("Agg")
    from minicnmfe.pipeline import CNMFeParams
    from tuning.sweep import SweepSpec
    from tuning.tuner import TunerConfig
    from tuning.validate import read_session_meta, tune_then_validate

    # Resolve frame rate: explicit flag wins; else auto-read metaData.json from an
    # AVI session folder (or its parent), matching batch_tune / validate_session;
    # fall back to 20 Hz. A wrong fps silently mis-tunes the decay-time g prior.
    if args.frame_rate is not None:
        fps, fps_src = args.frame_rate, "flag"
    else:
        fps = None
        if kind == "avi":
            fps = (read_session_meta(args.input).get("fps")
                   or read_session_meta(args.input.parent).get("fps"))
        fps, fps_src = (fps, "metaData.json") if fps else (20.0, "default")
        print(f"frame rate: {fps} Hz ({fps_src})")

    base_params = CNMFeParams.from_json(args.params) if args.params else None
    spec = SweepSpec(
        sigma=_floats(args.grid_sigma), min_corr=_floats(args.grid_min_corr),
        min_pnr=_floats(args.grid_min_pnr), merge_thr_corr=_floats(args.grid_merge_thr),
        global_bg_rank=_ints(args.grid_bg_rank), init_stride=_ints(args.grid_init_stride))

    cfg = TunerConfig(
        input_path=args.input, output_dir=run_dir, mode=args.mode, region=args.region,
        frame_rate_hz=fps, decay_time_ms=decay, base_params=base_params,
        ssub=args.ssub, tsub=args.tsub, reuse_mc_zarr=args.reuse_mc_zarr,
        max_avis=args.max_avis, pattern=args.pattern,
        n_template_avis=args.n_template_avis, stride_within_avi=args.stride_within_avi,
        n_init_frames=args.n_init_frames, n_shift_frames=args.n_shift_frames,
        cutout_hw=tuple(args.cutout_hw), window_t=args.window_t,
        sweep=spec, max_candidates=args.max_candidates,
        existing_results=args.existing_results, n_jobs=args.n_jobs,
        name=args.input.name, timestamp=ts,
        cli="python " + " ".join(shlex.quote(a) for a in sys.argv),
    )

    print(f"Tuning {args.input}  (mode={args.mode}, region={args.region}, "
          f"validate={do_validate}) -> {run_dir}")
    # A provided mc.zarr is itself the full recording → reuse it for validation.
    reuse_mc = args.input if (kind == "zarr" and do_validate) else None
    out = tune_then_validate(cfg, validate=do_validate, lowthr=not args.no_lowthr,
                             write_html=args.html, reuse_mc=reuse_mc)
    result = out["result"]

    print(f"\nWrote review folder: {run_dir}")
    if args.html:
        print(f"  - {run_dir / 'report.html'}   <- open this in a browser")
    print(f"  - {run_dir / 'report.md'}")
    print(f"  - {run_dir / 'recommended_params.json'}")
    print(f"  - {run_dir / 'downsample.json'}  (ssub={result['ssub']}, tsub={result['tsub']})")
    if out["validation"]:
        print(f"  - {run_dir / 'full' / 'comparison.md'}  (full-recording validation)")
        for r in out["validation"]["rows"]:
            print(f"      [{r['label']}] min_corr={r['min_corr']} min_pnr={r['min_pnr']}: "
                  f"K={r['K']} accepted={r['K_accepted']} "
                  f"cprojcorr={r['cprojcorr_median']:.3f}")
    print("\nApply the recommended params directly:")
    print(f"  python run_mc.py <movie_or_zarr> -o mc/ --params {run_dir / 'recommended_params.json'}")
    print(f"  python run_extract.py mc/mc.zarr -o results/ "
          f"--params {run_dir / 'recommended_params.json'} --ds-meta {run_dir / 'downsample.json'}")


if __name__ == "__main__":
    main()
