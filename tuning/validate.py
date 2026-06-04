"""Full-recording validation: extract at one or more threshold sets + diagnose.

Generalises the bespoke ``live_runs/tuning_picast/run_full*.py`` scripts. Given
a session AVI folder (or an existing ``mc.zarr``), runs the fused full-recording
motion correction **once**, transposes to a pixel-major ``Y_flat`` store
**once**, then runs ``fit_extract`` for each requested ``(min_corr, min_pnr)``
threshold set — reusing the (threshold-independent) ``Y_flat`` so each extra
candidate skips the expensive MC + transpose. Writes per-run results + a shared
set of diagnostic figures + a comparison table.

The defaults baked into :func:`good_defaults` are the long-real-recording
overrides learned in ``live_runs/tuning_picast/LEARNINGS.md`` (global rank-1
background, low ``min_pixel`` floor with SNR doing the ghost rejection, the
physical-decay prior, pinned ``init_stride``).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from minicnmfe.io import open_zarr, open_zarr_pixel_major, transpose_zarr_to_pixel_major
from minicnmfe.pipeline import CNMFe, CNMFeParams
from minicnmfe.preprocess import correlation_pnr
from tuning import report as R
from tuning.metrics import model_quality


def resolve_session_paths(items) -> "list[Path]":
    """Expand a session-list argument into validated, deduped session paths.

    ``items`` is a string or list of strings; each entry is either:

    - a ``.txt`` file → read it, one session path per line (blank lines and
      ``#`` comments skipped);
    - a session directory or ``mc.zarr`` path → used as-is.

    Order is preserved, duplicates dropped, missing paths skipped (a note is
    printed). Raises ``FileNotFoundError`` if nothing valid remains.
    """
    if isinstance(items, (str, Path)):
        items = [items]
    raw: "list[str]" = []
    for it in items:
        p = Path(str(it).strip())
        if p.suffix == ".txt" and p.exists():
            for line in p.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    raw.append(line)
        else:
            raw.append(str(it).strip())

    out: "list[Path]" = []
    seen = set()
    missing = []
    for s in raw:
        if not s:
            continue
        p = Path(s)
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.exists():
            out.append(p)
        else:
            missing.append(s)
    if missing:
        print(f"resolve_session_paths: skipping {len(missing)} missing path(s): "
              f"{missing[:3]}{' ...' if len(missing) > 3 else ''}")
    if not out:
        raise FileNotFoundError("no existing session paths after resolution")
    return out


def read_session_meta(folder: "str | Path") -> dict:
    """Parse acquisition settings from a Miniscope session folder.

    Reads ``metaData.json`` (``frameRate`` like ``"20FPS"`` → int; ``ROI``
    width/height → dims) and, when present, estimates the true frame rate from
    ``timeStamps.csv`` (``Time Stamp (ms)`` column). All fields are best-effort;
    missing files yield ``None`` values rather than raising.
    """
    folder = Path(folder)
    meta: dict = {"fps": None, "dims": None, "fps_measured": None,
                  "device": None, "n_frames_ts": None}

    mj = folder / "metaData.json"
    if mj.exists():
        d = json.loads(mj.read_text())
        fr = str(d.get("frameRate", ""))
        m = re.search(r"(\d+(?:\.\d+)?)\s*FPS", fr, re.IGNORECASE)
        if m:
            meta["fps"] = float(m.group(1))
        roi = d.get("ROI", {})
        if "height" in roi and "width" in roi:
            meta["dims"] = (int(roi["height"]), int(roi["width"]))
        meta["device"] = d.get("deviceType")

    ts = folder / "timeStamps.csv"
    if ts.exists():
        try:
            import csv
            rows = list(csv.reader(ts.open()))
            t = np.array([float(r[1]) for r in rows[1:] if len(r) > 1])
            if len(t) > 1:
                dur = (t[-1] - t[0]) / 1000.0
                meta["fps_measured"] = float((len(t) - 1) / dur) if dur > 0 else None
                meta["n_frames_ts"] = int(len(t))
        except Exception:
            pass
    return meta


def good_defaults(*, frame_rate_hz: float, decay_time_ms: float = 180.0,
                  sigma: float = 6.0, max_shift=(5, 3), mc_gSig_filt: float = 6.0,
                  min_corr: float = 0.8, min_pnr: float = 10.0,
                  n_jobs: int = -1) -> CNMFeParams:
    """Native-unit ``CNMFeParams`` with the long-real-recording overrides.

    See ``live_runs/tuning_picast/LEARNINGS.md`` for the rationale behind each.
    """
    return CNMFeParams(
        sigma=sigma, max_shift=tuple(max_shift), mc_gSig_filt=mc_gSig_filt,
        min_corr=min_corr, min_pnr=min_pnr,
        min_pixel=60,                 # floor only; SNR is the ghost discriminator
        global_bg_rank=1,             # absorb slow drift on long recordings
        auto_eval_snr_amp_thr=20.0,   # real ghost cut given typical SNR spreads
        decay_time_ms=decay_time_ms,  # physical τ, NOT the drift-inflated estimate
        frame_rate_hz=frame_rate_hz, g_prior_weight=0.6,
        init_stride=2,                # auto under-seeds long movies
        n_iter_main=2, n_jobs=n_jobs,
    )


def tune_then_validate(cfg, *, validate: bool = True, lowthr: bool = True,
                       write_html: bool = True, reuse_mc=None,
                       verbose: bool = True) -> dict:
    """Tune, then optionally validate on the full recording + write the HTML.

    The single-session core shared by ``tune.py`` and ``batch_tune.py``: run the
    tuner (:func:`tuning.tuner.run_tuning`), then — using the long-recording
    :func:`good_defaults` plus the tuner's recommended ``sigma`` and thresholds —
    run full-recording :func:`validate_session` at the recommended (+ a
    lower-recall) threshold set, and emit the self-contained HTML report
    alongside the markdown one.

    Args:
        cfg: a fully-built :class:`tuning.tuner.TunerConfig`.
        validate: run full-recording validation after tuning.
        lowthr: add the lower-recall threshold set to the comparison.
        write_html: emit ``report.html`` next to ``report.md``.
        reuse_mc: an existing full-recording ``mc.zarr`` for validation to reuse
            (e.g. when the tuner input was itself an ``mc.zarr``).

    Returns ``{"result", "validation", "run_dir", "html"}``.
    """
    from tuning.tuner import run_tuning

    result = run_tuning(cfg)
    run_dir = Path(cfg.output_dir)
    validation = None
    if validate:
        rec = result["recommended"]
        ssub, tsub = int(result["ssub"]), int(result["tsub"])
        # Match validate_session.py: good_defaults (the long-recording overrides)
        # carries everything except sigma + thresholds + ssub/tsub + decay.
        native = good_defaults(
            frame_rate_hz=cfg.frame_rate_hz, decay_time_ms=cfg.decay_time_ms,
            sigma=float(rec["sigma"]), min_corr=float(rec["min_corr"]),
            min_pnr=float(rec["min_pnr"]), n_jobs=cfg.n_jobs)
        thresholds = [("recommended", float(rec["min_corr"]), float(rec["min_pnr"]))]
        if lowthr:
            thresholds.append(("lowthr",
                               max(0.6, round(float(rec["min_corr"]) - 0.1, 2)),
                               max(3.0, float(rec["min_pnr"]) - 4)))
        validation = validate_session(
            cfg.input_path, run_dir / "full", native_params=native,
            ssub=ssub, tsub=tsub, threshold_sets=thresholds,
            reuse_mc=reuse_mc, verbose=verbose)

    if write_html:
        from tuning.report_html import write_html_report
        write_html_report(run_dir, result, validation)

    return {"result": result, "validation": validation, "run_dir": str(run_dir),
            "html": str(run_dir / "report.html") if write_html else None}


def _diagnostics(model, cn, sample, run_dir, *, with_shifts=True):
    """Write the standard diagnostic figure set for one fitted model."""
    figs = run_dir / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    if model.A is not None and model.A.shape[1] > 0:
        R.fig_sweep_footprints(model, cn, out_path=figs / "footprints_on_corr.png")
        R.fig_sweep_traces(model, out_path=figs / "traces.png", n=12)
        R.fig_npix_accepted(model, out_path=figs / "npix_dist.png")
    if model.eval_info is not None and "snr_amp" in model.eval_info:
        ev = {"snr": np.asarray(model.eval_info["snr_amp"], float),
              "current_thr": model.params.auto_eval_snr_amp_thr,
              "snr_suggested": model.params.auto_eval_snr_amp_thr}
        R.fig_snr_eval(ev, model, out_path=figs / "snr_eval.png")
    if with_shifts and model.shifts is not None:
        R.fig_mc_shifts(model.shifts, out_path=figs / "mc_shifts.png")


def validate_session(
    folder: "str | Path", out_dir: "str | Path", *,
    native_params: CNMFeParams, ssub: int = 1, tsub: int = 1,
    threshold_sets: "list[tuple[str, float, float]] | None" = None,
    reuse_mc: "str | Path | None" = None, n_template_avis: int = 10,
    verbose: bool = True,
) -> dict:
    """Run full-recording extraction at each threshold set, sharing MC + Y_flat.

    Args:
        folder: AVI session folder (``0.avi…N.avi``).
        out_dir: output root; gets ``mc/``, ``Y_flat_pixel.zarr``, ``run_<label>/``.
        native_params: NATIVE-unit params (typically from :func:`good_defaults`);
            ``downscaled(ssub, tsub)`` is applied internally.
        threshold_sets: list of ``(label, min_corr, min_pnr)``. Default = the
            params' own thresholds (``recommended``) + a lower-recall set
            (``lowthr``: ``min_corr−0.1``, ``min_pnr−4``, floored).
        reuse_mc: an existing ``mc.zarr`` to use instead of fusing the AVIs.

    Returns a dict with ``rows`` (one per run) and paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = native_params.downscaled(ssub, tsub)

    if threshold_sets is None:
        threshold_sets = [
            ("recommended", native_params.min_corr, native_params.min_pnr),
            ("lowthr", max(0.6, native_params.min_corr - 0.1),
             max(3.0, native_params.min_pnr - 4)),
        ]

    t0 = time.time()
    # --- MC once ---
    if reuse_mc is not None:
        mc = open_zarr(reuse_mc)
        mc_path = Path(reuse_mc)
        shifts = None
        sh_npy = mc_path.parent / "shifts.npy"
        if sh_npy.exists():
            shifts = np.load(sh_npy)
    else:
        mc_model = CNMFe(grid)
        mc = mc_model.fit_mc_from_avis(folder, out_dir / "mc", ssub=ssub, tsub=tsub,
                                       skip_if_exists=True)
        mc_path = out_dir / "mc" / "mc.zarr"
        shifts = mc_model.shifts
    if verbose:
        print(f">>> mc.zarr {tuple(mc.shape)} ready ({time.time()-t0:.0f}s)", flush=True)

    # --- Y_flat once (threshold-independent) ---
    yf_path = out_dir / "Y_flat_pixel.zarr"
    yf = transpose_zarr_to_pixel_major(mc_path, yf_path, skip_if_exists=True,
                                       verbose=verbose)
    if verbose:
        print(f">>> Y_flat {tuple(yf.shape)} ready ({time.time()-t0:.0f}s)", flush=True)

    # --- correlation image once (shared by footprint overlays) ---
    T = int(mc.shape[0])
    idx = np.linspace(0, T - 1, min(T, 1500)).astype(int)
    sample = np.stack([np.asarray(mc[int(i)]) for i in idx]).astype(np.float32)
    cn, _pnr = correlation_pnr(sample, sigma=grid.sigma, n_jobs=grid.n_jobs)
    np.save(out_dir / "cn.npy", cn)

    # --- one extraction per threshold set, reusing Y_flat ---
    rows = []
    for label, mcorr, mpnr in threshold_sets:
        tr = time.time()
        params = replace(grid, min_corr=float(mcorr), min_pnr=float(mpnr))
        model = CNMFe(params)
        if verbose:
            print(f">>> extract [{label}] min_corr={mcorr} min_pnr={mpnr} ...", flush=True)
        model.fit_extract(mc, Y_flat_zarr=yf, evaluate=True)
        if shifts is not None:
            model.shifts = shifts
        run_dir = out_dir / f"run_{label}"
        model.save(run_dir)
        _diagnostics(model, cn, sample, run_dir)
        q = model_quality(model)
        row = {"label": label, "min_corr": float(mcorr), "min_pnr": float(mpnr),
               "wall_s": round(time.time() - tr, 1), **q}
        rows.append(row)
        (run_dir / "summary.txt").write_text(
            f"{label}: min_corr={mcorr} min_pnr={mpnr}\n"
            f"K={q['K']} accepted={q['K_accepted']} ({q['accepted_frac']:.2f})\n"
            f"cprojcorr median={q['cprojcorr_median']:.3f}  "
            f"npix median/IQR={q['npix_median']:.0f}/{q['npix_iqr']:.0f}  "
            f"SNR median={q['snr_median']:.1f}\n")
        if verbose:
            print(f"    -> K={q['K']} accepted={q['K_accepted']} "
                  f"cprojcorr={q['cprojcorr_median']:.3f} ({row['wall_s']:.0f}s)", flush=True)

    # --- comparison table ---
    cols = ["label", "min_corr", "min_pnr", "K", "K_accepted", "accepted_frac",
            "cprojcorr_median", "npix_median", "npix_iqr", "snr_median", "wall_s"]
    lines = ["# Full-recording validation — threshold comparison", "",
             f"mc.zarr: `{tuple(mc.shape)}`  (ssub={ssub}, tsub={tsub})  "
             f"total {time.time()-t0:.0f}s", "",
             "| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    (out_dir / "comparison.md").write_text("\n".join(lines) + "\n")

    return {"rows": rows, "out_dir": str(out_dir), "mc_path": str(mc_path),
            "yflat_path": str(yf_path), "wall_s": round(time.time() - t0, 1)}
