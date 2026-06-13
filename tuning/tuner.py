"""Tuner orchestration: ``TunerConfig`` + ``run_tuning``.

Wires the sampling helpers, the per-knob heuristics, the optional extraction
sweep, and the report writer into one call. Unit bookkeeping is the one subtle
part: the extraction sweep runs on the ``mc.zarr`` grid (possibly downsampled),
but ``recommended_params.json`` is written in **native** units (sigma·ssub,
min_pixel·ssub², thresholds unchanged) so it feeds ``run_extract.py --ds-meta``
correctly. ``ssub``/``tsub`` are carried separately in ``downsample.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import numpy as np

from minicnmfe.io import open_zarr
from minicnmfe.pipeline import CNMFe, CNMFeParams
from tuning import heuristics as H
from tuning import io_sample as S
from tuning.metrics import mc_quality
from tuning.sweep import SweepSpec, resolve_offset_grid, resolve_sigma_grid, run_sweep
from tuning.validate import good_defaults


@dataclass
class TunerConfig:
    input_path: Path
    output_dir: Path
    mode: str = "both"          # 'heuristic' | 'sweep' | 'both'
    region: str = "cutout"      # 'cutout' | 'full'
    frame_rate_hz: float = 20.0
    decay_time_ms: float = 180.0
    base_params: "CNMFeParams | None" = None
    # downsample (AVI input only). None ssub/tsub => auto from the heuristic.
    ssub: "int | None" = None
    tsub: "int | None" = None
    reuse_mc_zarr: "Path | None" = None
    max_avis: "int | None" = None
    pattern: str = "*.avi"
    # subset sizes
    n_template_avis: int = 8
    stride_within_avi: int = 50
    n_init_frames: int = 400
    n_shift_frames: int = 200
    cutout_hw: "tuple[int, int]" = (256, 256)
    window_t: int = 3000
    # sweep
    sweep: SweepSpec = field(default_factory=SweepSpec)
    max_candidates: int = 24
    existing_results: "Path | None" = None
    n_jobs: int = 1
    name: str = "session"
    cli: str = ""
    timestamp: str = ""


def _detect_kind(path: Path) -> str:
    path = Path(path)
    if path.is_dir() and not str(path).endswith(".zarr"):
        return "avi"
    return "zarr"


def run_tuning(cfg: TunerConfig) -> dict:
    """Run the tuning workflow and write the report folder. Returns the result dict."""
    from tuning import report as R

    # Default base = the long-recording overrides (global_bg_rank=1, n_iter_main=2,
    # init_stride=2, min_pixel floor + SNR ghost cut, physical-decay prior). The
    # tuner overwrites only the fields it has data-driven values for, so the
    # recommendation *contains* these wins and validation can run it verbatim.
    # An explicit --params base is respected unchanged.
    base = cfg.base_params or good_defaults(
        frame_rate_hz=cfg.frame_rate_hz, decay_time_ms=cfg.decay_time_ms,
        n_jobs=cfg.n_jobs)
    # Patch-parallel greedy init is ON by default, but the sweep runs candidates
    # in parallel loky processes — patched-greedy inside a candidate would nest
    # loky (joblib serializes the inner patches → slower, not faster). The sweep
    # already parallelizes at the candidate level, so force serial greedy per
    # candidate here. (The final MiniCnmfeExtraction is a single fit → keeps the
    # default patched path.)
    base = replace(base, n_jobs=cfg.n_jobs,
                   frame_rate_hz=cfg.frame_rate_hz, decay_time_ms=cfg.decay_time_ms,
                   init_patches=False)
    kind = _detect_kind(cfg.input_path)
    run_dir = Path(cfg.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    stages: dict = {}
    recommended: dict = {"frame_rate_hz": cfg.frame_rate_hz,
                         "decay_time_ms": cfg.decay_time_ms}
    sources: dict = {"frame_rate_hz": "user", "decay_time_ms": "user"}
    rationale: dict = {}

    sigma_native = 4.0  # fallback; set from the raw sample (AVI) or mc sample (zarr) below
    ssub = cfg.ssub if cfg.ssub is not None else 1
    tsub = cfg.tsub if cfg.tsub is not None else 1

    # -- Stage 1: motion correction (AVI input only) --
    if kind == "avi":
        avis = S.list_avis(cfg.input_path, cfg.pattern)
        sample = S.decode_strided_sample(avis, cfg.n_template_avis, cfg.stride_within_avi)
        median_img = np.median(sample, axis=0)

        mc_gsig, sigma_native, ev = H.suggest_mc_gsig_and_sigma(sample)
        stages["mc_gsig"] = ev
        recommended["mc_gSig_filt"] = mc_gsig
        sources["mc_gSig_filt"] = "heuristic"
        rationale["mc_gSig_filt"] = "≈ median neuron radius (blob_log on temporal-std)"

        max_shift, border_px, ev = H.suggest_max_shift(sample, median_img, mc_gsig,
                                                       n_shift_frames=cfg.n_shift_frames)
        stages["max_shift"] = ev
        recommended["max_shift"] = max_shift
        recommended["border_px"] = border_px
        sources["max_shift"] = sources["border_px"] = "heuristic"
        rationale["max_shift"] = "99th-pct per-frame shift + 2 px margin"
        rationale["border_px"] = "max(max_shift) — trims warpAffine fill"

        s_auto, t_auto, ev = H.suggest_downsample(sigma_native, cfg.frame_rate_hz,
                                                  cfg.decay_time_ms)
        stages["downsample"] = ev
        if cfg.ssub is None:
            ssub = s_auto
        if cfg.tsub is None:
            tsub = t_auto

    # -- Resolve the mc.zarr the extraction stages run on --
    mc_shifts = None
    if cfg.reuse_mc_zarr is not None:
        mc = open_zarr(cfg.reuse_mc_zarr)
        mc_path = Path(cfg.reuse_mc_zarr)
        sh_npy = mc_path.parent / "shifts.npy"
        if sh_npy.exists():
            mc_shifts = np.load(sh_npy)
    elif kind == "avi":
        mc, mc_shifts = S.quick_fused_mc(
            cfg.input_path, run_dir / "mc", base, ssub=ssub, tsub=tsub,
            n_template_avis=cfg.n_template_avis, max_avis=cfg.max_avis, pattern=cfg.pattern)
        mc_path = run_dir / "mc" / "mc.zarr"
    else:
        # Extraction-only path: a provided zarr is already motion-corrected
        # (e.g. the DB's mc.zarr). Stage 1 is skipped entirely, so MC params
        # (mc_gSig_filt / max_shift / mc_n_iter) are NEVER (re-)estimated here —
        # they are chosen upstream by tuning.mc_tune. This tuner only tunes the
        # extraction params on the corrected movie.
        mc = open_zarr(cfg.input_path)
        mc_path = Path(cfg.input_path)
        ssub = tsub = 1  # a provided zarr is taken as-is
        sh_npy = mc_path.parent / "shifts.npy"
        if sh_npy.exists():
            mc_shifts = np.load(sh_npy)

    dims = (int(mc.shape[1]), int(mc.shape[2]))
    T = int(mc.shape[0])
    mc_sample, sample_idx = S.load_mc_sample(mc, cfg.n_init_frames)

    # For a directly-provided zarr there was no raw-AVI sample to measure the
    # neuron radius from (the AVI path sets sigma_native at stage 1). Estimate it
    # from the mc sample instead of leaving the hardcoded default — it seeds the
    # CORR/PNR PSF width for the sigma refit below. ssub==1 here (provided zarr is
    # taken as-is), so the mc grid is native and the estimate is in native units.
    if kind == "zarr":
        _gsig, sigma_native, ev = H.suggest_mc_gsig_and_sigma(mc_sample)
        stages["sigma_native"] = ev

    # -- Stage 3: initialisation (on the mc.zarr grid) --
    sigma_seed = max(2.0, sigma_native / ssub) if ssub > 1 else sigma_native
    sigma_ds, cn, pnr, ev = H.suggest_sigma_extraction(mc_sample, sigma_seed, n_jobs=cfg.n_jobs)
    stages["sigma"] = ev
    min_corr, min_pnr, ev = H.suggest_corr_pnr(cn, pnr, sigma_ds)
    stages["corr_pnr"] = ev
    min_pixel_ds, ev = H.suggest_min_pixel(mc_sample, sigma_ds, min_corr, min_pnr,
                                           dims, n_jobs=cfg.n_jobs)
    stages["min_pixel"] = ev

    # Grid-unit base params for the sweep / extraction.
    base_grid = replace(base, sigma=float(sigma_ds), min_corr=float(min_corr),
                        min_pnr=float(min_pnr), min_pixel=int(min_pixel_ds),
                        frame_rate_hz=cfg.frame_rate_hz / tsub)

    # Defaults (native) from the heuristics; overwritten by sweep-best below.
    best_swept = {"sigma": sigma_ds, "min_corr": min_corr, "min_pnr": min_pnr,
                  "merge_thr_corr": base.merge_thr_corr,
                  "global_bg_rank": base.global_bg_rank, "init_stride": base.init_stride}
    for f in best_swept:
        sources.setdefault(f, "heuristic")

    best_model = None
    sweep_result = None

    # -- Sweep --
    if cfg.mode in ("sweep", "both"):
        region_crop = None
        if cfg.region == "cutout":
            region_crop = S.pick_cutout(cn, T=T, cutout_hw=cfg.cutout_hw,
                                        window_t=cfg.window_t, sample=mc_sample,
                                        sample_idx=sample_idx)
        # Resolve the (possibly heuristic-relative) grids against the measured
        # heuristics before the spec is expanded, so each data-driven value is
        # always a candidate (see tuning.sweep.resolve_offset_grid). Omitted
        # min_corr/min_pnr collapse to the single detected value (no sweep).
        sweep = cfg.sweep or SweepSpec()
        sigma_grid = resolve_sigma_grid(sweep.sigma, sigma_ds)
        corr_grid = resolve_offset_grid(sweep.min_corr, min_corr, floor=0.3, clip_max=0.98)
        pnr_grid = resolve_offset_grid(sweep.min_pnr, min_pnr, floor=2.0)
        sweep = replace(sweep, sigma=sigma_grid, min_corr=corr_grid, min_pnr=pnr_grid)
        stages["sigma_grid"] = {"values": sigma_grid, "heuristic": float(sigma_ds)}
        stages["thr_grid"] = {"min_corr": corr_grid, "min_pnr": pnr_grid,
                              "detected": (float(min_corr), float(min_pnr))}
        rows, best_params, best_model = run_sweep(
            mc_path, base_grid, sweep, region_crop=region_crop,
            n_jobs=cfg.n_jobs, workdir=run_dir / "sweep",
            max_candidates=cfg.max_candidates, cn=cn)
        sweep_result = {"rows": rows, "region": cfg.region, "region_crop": region_crop}
        for f in best_swept:
            # global_bg_rank / init_stride are long-recording wins the short sweep
            # cutout can't evaluate (see LEARNINGS.md): keep them pinned to the base
            # so a misleading cutout sweep can't downgrade them in the recommendation.
            if f in ("global_bg_rank", "init_stride"):
                sources[f] = "base"
                continue
            best_swept[f] = getattr(best_params, f)
            sources[f] = "sweep"

    # -- Stage 4: temporal / merge / eval (needs a fitted model) --
    stage4_model = best_model
    if stage4_model is None and cfg.existing_results is not None:
        stage4_model = CNMFe.load(cfg.existing_results)
        if stage4_model.eval_info is None and stage4_model.sn is not None:
            stage4_model.evaluate()

    if stage4_model is not None and stage4_model.C_raw is not None and stage4_model.A.shape[1] > 0:
        fr_grid = base_grid.frame_rate_hz
        decay_ms, ev = H.suggest_decay_time(stage4_model, fr_grid)
        stages["decay"] = ev
        # Diagnostic only: the data Yule-Walker τ is drift-inflated on long
        # recordings (LEARNINGS.md), so the recommendation keeps the physical
        # indicator τ (cfg.decay_time_ms, already in `recommended`).
        rationale["decay_time_ms"] = (
            f"physical indicator τ (data Yule-Walker τ ≈ {decay_ms:.0f} ms is "
            "drift-inflated; shown for diagnosis, not used)")

        gpw, ev = H.suggest_g_prior_weight(ev["g_yw"], fr_grid, cfg.decay_time_ms)
        stages["g_prior"] = ev
        recommended["g_prior_weight"] = gpw
        sources["g_prior_weight"] = "data"
        rationale["g_prior_weight"] = "spread of YW g vs physical target"

        merge_thr, ev = H.suggest_merge_thr(stage4_model)
        stages["merge"] = ev
        best_swept["merge_thr_corr"] = merge_thr
        sources["merge_thr_corr"] = "data"

        snr_thr, ev = H.suggest_snr_thr(stage4_model)
        stages["snr"] = ev
        recommended["auto_eval_snr_amp_thr"] = snr_thr
        sources["auto_eval_snr_amp_thr"] = "data"
        rationale["auto_eval_snr_amp_thr"] = "largest gap in the low-SNR (ghost) region"

    # -- Fold swept/stage values into native-unit recommended params --
    recommended["sigma"] = float(best_swept["sigma"]) * ssub
    recommended["min_pixel"] = max(1, int(min_pixel_ds) * ssub * ssub)
    recommended["min_corr"] = float(best_swept["min_corr"])
    recommended["min_pnr"] = float(best_swept["min_pnr"])
    recommended["merge_thr_corr"] = float(best_swept["merge_thr_corr"])
    recommended["global_bg_rank"] = int(best_swept["global_bg_rank"])
    if best_swept["init_stride"] is not None:
        recommended["init_stride"] = int(best_swept["init_stride"])
    sources["min_pixel"] = "heuristic"
    rationale["sigma"] = "blob_log on CORR·PNR (×ssub → native)"
    rationale["min_corr"] = rationale["min_pnr"] = (
        "best separation of CORR/PNR at detected neuron blobs vs background")
    rationale["min_pixel"] = "25th-pct footprint area (×ssub² → native)"

    # One merged CNMFeParams: the long-recording base with the tuner's data-driven
    # native-unit fields layered on. This is the single source of truth — it is
    # serialized to recommended_params.json AND validated verbatim on the full
    # recording (tuning.validate.tune_then_validate), so what you see validated is
    # what you apply downstream.
    _valid = {f.name for f in fields(CNMFeParams)}
    rec_params = replace(base, **{k: v for k, v in recommended.items() if k in _valid})

    result = {
        "config": {"name": cfg.name, "input": str(cfg.input_path),
                   "input_kind": kind, "mode": cfg.mode, "region": cfg.region,
                   "ssub": ssub, "tsub": tsub, "frame_rate_hz": cfg.frame_rate_hz,
                   "decay_time_ms": cfg.decay_time_ms, "n_jobs": cfg.n_jobs,
                   "timestamp": cfg.timestamp, "cli": cfg.cli},
        "recommended": recommended, "recommended_params": rec_params,
        "sources": sources, "rationale": rationale,
        "ssub": ssub, "tsub": tsub, "stages": stages,
        "sweep": sweep_result, "mc_quality": mc_quality(mc_shifts),
    }

    R.write_report(run_dir, result, best_model=stage4_model, cn=cn, pnr=pnr)
    return result
