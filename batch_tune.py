"""Batch tune + validate many sessions in ONE process (no sub-agents).

For each session: run the tuner (`tune.py`), read its recommended
sigma/ssub/thresholds, then validate on the full recording
(`validate_session.py`) at the recommended threshold + a lower-recall set.
Sessions run with bounded concurrency; BLAS is capped so `JOBS·CORES` threads
don't oversubscribe. Writes `<out>/batch_summary.md`.

This is the **lean** batch path: the model launches it in the background (so the
hours of compute cost no model tokens) and only reads the text summary +
optionally views a curated set of figures afterward. Contrast with spawning one
interpreting sub-agent per session, which is what made the batch token-heavy.

    python batch_tune.py sessions.txt -o live_runs/tuning_batch \\
        --indicator gcamp8m --jobs 2 --cores 6

Use --dry-run to print the plan (resolved sessions + per-session commands)
without running anything.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_BLAS = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                          "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                          "VECLIB_MAXIMUM_THREADS")}


def _tune_cmd(sess, out, fps, indicator, nj, lowthr):
    """One consolidated `tune.py` call per session: tune + full-recording
    validation in a single BLAS-capped subprocess (tune.py owns both stages)."""
    # No --grid-bg-rank: global_bg_rank is pinned to the long-recording base (=1)
    # because the short cutout sweep can't see its benefit (see tuning/tuner.py).
    # No --grid-min-corr/--grid-min-pnr: min_corr/min_pnr are auto-detected per
    # recording by image-threshold morphology (suggest_corr_pnr); the sweep tests
    # a small auto-anchored range around them, so a hardcoded grid is unneeded.
    cmd = [sys.executable, "-u", str(ROOT / "tune.py"), str(sess),
           "-o", str(out), "--frame-rate", str(fps),
           "--indicator", indicator, "--mode", "both", "--region", "cutout",
           "--max-avis", "6",
           "--n-jobs", str(nj), "--validate", "--html"]
    if not lowthr:
        cmd.append("--no-lowthr")
    return cmd


def _run_one(sess, out_root, indicator, nj, lowthr):
    """Tune + validate one session via a single tune.py subprocess. Never raises."""
    name = Path(sess).parent.name if Path(sess).name == "miniscope_video" else Path(sess).name
    out = Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "batch.log").open("w")
    env = {**os.environ, **_BLAS}
    t0 = time.time()
    try:
        from tuning.validate import read_session_meta
        fps = read_session_meta(sess).get("fps") or 20.0
        log.write(f"== tune+validate {name} (fps={fps}) ==\n"); log.flush()
        subprocess.run(_tune_cmd(sess, out, fps, indicator, nj, lowthr),
                       check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
        run_dirs = sorted(out.glob("tune_*"))
        run_dir = run_dirs[-1] if run_dirs else out
        rec = (json.loads((run_dir / "recommended_params.json").read_text())
               if (run_dir / "recommended_params.json").exists() else {})
        ds = (json.loads((run_dir / "downsample.json").read_text())
              if (run_dir / "downsample.json").exists() else {"ssub": 1})
        comp = run_dir / "full" / "comparison.md"
        return {"name": name, "ok": True, "out": str(run_dir),
                "sigma": rec.get("sigma"), "ssub": int(ds.get("ssub", 1)),
                "min_corr": rec.get("min_corr"), "min_pnr": rec.get("min_pnr"),
                "comparison": comp.read_text() if comp.exists() else "",
                "wall_s": round(time.time() - t0)}
    except Exception as exc:  # noqa: BLE001
        return {"name": name, "ok": False, "out": str(out),
                "error": repr(exc), "log": str(out / "batch.log"),
                "wall_s": round(time.time() - t0)}
    finally:
        log.close()


def run_batch(inputs, output, *, indicator="gcamp8m", jobs=2, cores=6,
              lowthr=True, dry_run=False) -> "dict | None":
    """Tune + validate a list of sessions in one background process.

    The programmatic core shared by ``batch_tune.py`` and ``tune.py --sessions``.
    Each session runs in its own BLAS-capped ``tune.py --validate`` subprocess
    (so ``jobs·cores`` threads don't oversubscribe), tuned independently. Writes
    ``<output>/<name>/`` per session + ``<output>/batch_summary.md``. Returns the
    summary dict (or None on ``dry_run``).
    """
    from tuning.validate import resolve_session_paths

    output = Path(output)
    sessions = resolve_session_paths(inputs)
    output.mkdir(parents=True, exist_ok=True)

    print(f"{len(sessions)} session(s); jobs={jobs} cores/session={cores} "
          f"indicator={indicator} lowthr={lowthr}")
    for s in sessions:
        print(f"  {s}")
    if dry_run:
        ex = sessions[0]
        print("\nper-session plan (example):")
        print("  ", " ".join(_tune_cmd(ex, output / "<name>", 20, indicator, cores, lowthr)))
        return None

    t0 = time.time()
    rows = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futs = {pool.submit(_run_one, s, output, indicator, cores, lowthr): s
                for s in sessions}
        for fut in as_completed(futs):
            r = fut.result()
            rows.append(r)
            tag = "OK  " if r["ok"] else "FAIL"
            print(f"[{tag}] {r['name']} ({r['wall_s']}s)"
                  + ("" if r["ok"] else f"  -> {r['log']}"), flush=True)

    rows.sort(key=lambda r: r["name"])
    lines = ["# Batch tuning summary", "",
             f"{len(rows)} sessions, total {round(time.time()-t0)}s "
             f"(jobs={jobs}, cores/session={cores})", "",
             "Per-session recommended params (NATIVE units) + full-recording "
             "validation. `decay_time_ms` is set from the indicator "
             f"(`{indicator}`), NOT the drift-inflated data estimate.", ""]
    for r in rows:
        lines.append(f"## {r['name']}")
        if not r["ok"]:
            lines += [f"**FAILED** — {r['error']} (log: `{r['log']}`)", ""]
            continue
        sigma_s = f"{r['sigma']:.2f}" if isinstance(r["sigma"], (int, float)) else str(r["sigma"])
        lines += [f"recommended: sigma(native)={sigma_s}  ssub={r['ssub']}  "
                  f"min_corr={r['min_corr']}  min_pnr={r['min_pnr']}  ({r['wall_s']}s)",
                  "", r["comparison"], f"outputs: `{r['out']}/`", ""]
    (output / "batch_summary.md").write_text("\n".join(lines))
    print(f"\nWrote {output / 'batch_summary.md'}")
    return {"rows": rows, "summary": str(output / "batch_summary.md")}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="session paths and/or a .txt list")
    p.add_argument("--output", "-o", type=Path, required=True)
    p.add_argument("--indicator", default="gcamp8m")
    p.add_argument("--jobs", type=int, default=2, help="sessions run concurrently")
    p.add_argument("--cores", type=int, default=6, help="cores (n-jobs) per session")
    p.add_argument("--no-lowthr", action="store_true",
                   help="single threshold per session (skip the lower-recall compare)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    run_batch(args.inputs, args.output, indicator=args.indicator, jobs=args.jobs,
              cores=args.cores, lowthr=not args.no_lowthr, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
