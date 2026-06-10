"""Pre-motion-correction tuning: pick ssub/tsub + crispness-validated MC params.

Runs *before* downsampling / MC on a recording's raw AVIs. It (1) estimates the
downsample factors and the MC high-pass/shift **seeds** from a cheap strided AVI
sample (reusing ``tuning.heuristics``), (2) loads one contiguous high-activity
clip and block-mean downsamples it to the grid MC will actually run on, and
(3) hands that clip to ``tuning.mc_search.search_mc_params`` to choose
``mc_gSig_filt`` / ``max_shift`` / ``mc_n_iter`` by corrected-clip crispness.

This is the counterpart to ``tuning.tuner.run_tuning`` — which, *post*-MC, tunes
only the extraction params on the resulting ``mc.zarr``. Splitting the two means
the extraction sweep no longer runs against an internally-produced, unvalidated
quick MC: MC is chosen on its own merits first, then extraction is tuned on the
real corrected movie.

Units: heuristic seeds come out in NATIVE pixels; the search clip and the
returned ``mc_params`` are in GRID units (downsampled by ``ssub``/``tsub``),
because the DB runs MC on the downsampled ``ds.zarr``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tuning import heuristics as H
from tuning import io_sample as S
from tuning.mc_search import McSearchSpec, search_mc_params
from tuning.validate import good_defaults


@dataclass
class McTuneConfig:
    input_path: Path                       # raw-AVI folder
    output_dir: Path
    frame_rate_hz: float = 20.0
    decay_time_ms: float = 180.0
    ssub: "int | None" = None              # None => auto from the heuristic
    tsub: "int | None" = None
    pattern: str = "*.avi"
    # seed-estimation sample sizes (shared with the tuner's stage 1)
    n_template_avis: int = 8
    stride_within_avi: int = 50
    n_shift_frames: int = 200
    # contiguous clip the short MCs run on
    clip_frames: int = 1500
    clip_start_frac: float = 0.4
    spec: McSearchSpec = field(default_factory=McSearchSpec)
    n_jobs: int = 1
    name: str = "session"
    write_report: bool = True


def _block_mean(clip: np.ndarray, ssub: int, tsub: int) -> np.ndarray:
    """Block-mean downsample a (T, H, W) numpy clip by ssub (space) / tsub (time).

    Mirrors ``minicnmfe.downsample.downsample_movie`` (trailing remainder on each
    axis dropped so every bin is a full-block mean) but stays in RAM — the clip
    is small and we avoid a zarr round-trip.
    """
    if ssub == 1 and tsub == 1:
        return clip.astype(np.float32)
    T, Hh, Ww = clip.shape
    T2, H2, W2 = (T // tsub) * tsub, (Hh // ssub) * ssub, (Ww // ssub) * ssub
    c = clip[:T2, :H2, :W2]
    c = c.reshape(T2 // tsub, tsub, H2 // ssub, ssub, W2 // ssub, ssub)
    return c.mean(axis=(1, 3, 5)).astype(np.float32)


def run_mc_tuning(cfg: McTuneConfig) -> dict:
    """Estimate ssub/tsub + MC params for one recording. Returns a result dict.

    Keys: ``ssub``, ``tsub``, ``mc_params`` (grid units: ``mc_gSig_filt``,
    ``max_shift``, ``mc_n_iter``, ``border_px``), ``rows`` (per-candidate, for
    the DB part table + report), ``best``, ``sources``, ``rationale``,
    ``evidence``.
    """
    run_dir = Path(cfg.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    avis = S.list_avis(cfg.input_path, cfg.pattern)
    sample = S.decode_strided_sample(avis, cfg.n_template_avis, cfg.stride_within_avi)
    median_img = np.median(sample, axis=0)

    # -- seeds (native units) --
    seed_gsig, sigma_native, ev_g = H.suggest_mc_gsig_and_sigma(sample)
    seed_ms, border_native, ev_ms = H.suggest_max_shift(
        sample, median_img, seed_gsig, n_shift_frames=cfg.n_shift_frames)
    s_auto, t_auto, ev_ds = H.suggest_downsample(
        sigma_native, cfg.frame_rate_hz, cfg.decay_time_ms)
    ssub = int(cfg.ssub) if cfg.ssub is not None else int(s_auto)
    tsub = int(cfg.tsub) if cfg.tsub is not None else int(t_auto)

    # -- contiguous clip, downsampled to the MC grid --
    clip_native = S.decode_contiguous_clip(avis, cfg.clip_frames, cfg.clip_start_frac)
    clip = _block_mean(clip_native, ssub, tsub)

    # -- seeds -> grid units (the clip & MC run on the downsampled grid) --
    seed_gsig_grid = max(1.0, float(seed_gsig) / ssub)
    seed_ms_grid = (max(1, int(seed_ms[0]) // ssub), max(1, int(seed_ms[1]) // ssub))

    base = good_defaults(frame_rate_hz=cfg.frame_rate_hz,
                         decay_time_ms=cfg.decay_time_ms, n_jobs=cfg.n_jobs)
    best, rows, ev_search = search_mc_params(
        clip, base, seed_gsig_grid, seed_ms_grid, cfg.spec, n_jobs=cfg.n_jobs)
    for i, r in enumerate(rows):
        r["idx"] = i

    border_px = max(int(best["max_shift"][0]), int(best["max_shift"][1]))
    mc_params = {"mc_gSig_filt": float(best["mc_gSig_filt"]),
                 "max_shift": (int(best["max_shift"][0]), int(best["max_shift"][1])),
                 "mc_n_iter": int(best["mc_n_iter"]), "border_px": int(border_px)}

    sources = {"ssub": "user" if cfg.ssub is not None else "heuristic",
               "tsub": "user" if cfg.tsub is not None else "heuristic",
               "mc_gSig_filt": "search", "max_shift": "search",
               "mc_n_iter": "search", "border_px": "derived"}
    rationale = {
        "mc_gSig_filt": "crispness-best of a grid seeded at the median neuron radius",
        "max_shift": "crispness-best of a grid seeded at the 99th-pct probe shift",
        "mc_n_iter": "crispness-best of {1,2} rigid passes",
        "border_px": "max(max_shift) — trims warpAffine fill",
        "ssub": "neuron FWHM >= 4 px on the binned grid",
        "tsub": "binned frame period <= decay_time/2",
    }
    result = {
        "config": {"name": cfg.name, "input": str(cfg.input_path),
                   "ssub": ssub, "tsub": tsub, "frame_rate_hz": cfg.frame_rate_hz,
                   "decay_time_ms": cfg.decay_time_ms, "n_jobs": cfg.n_jobs},
        "ssub": ssub, "tsub": tsub, "mc_params": mc_params, "best": best,
        "rows": rows, "sources": sources, "rationale": rationale,
        "evidence": {"seeds_native": {"mc_gSig_filt": seed_gsig,
                                      "max_shift": list(seed_ms),
                                      "sigma": sigma_native, "border_px": border_native},
                     "search": ev_search, "clip_shape": list(clip.shape)},
    }

    if cfg.write_report:
        try:
            _write_mc_report(run_dir, result, clip, best)
        except Exception as e:  # a failed plot must not abort an otherwise-good run
            print(f"mc_tune: report writing failed ({e}); continuing.")
    return result


def _write_mc_report(run_dir: Path, result: dict, clip: np.ndarray, best: dict) -> None:
    """Write ``mc_tuning.json`` + a one-page diagnostic PNG (lazy matplotlib)."""
    import json

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from minicnmfe.motion_correction import motion_correction_rigid

    json_safe = {k: result[k] for k in ("config", "ssub", "tsub", "mc_params",
                                        "sources", "rationale")}
    json_safe["rows"] = [{kk: (list(vv) if isinstance(vv, tuple) else vv)
                          for kk, vv in r.items()} for r in result["rows"]]
    (run_dir / "mc_tuning.json").write_text(json.dumps(json_safe, indent=2, default=float))

    from tuning.metrics import correlation_image

    # Re-run the winning MC once on the clip for before/after correlation images
    # (Cn) — the cell-focused view the ranking uses, not the background-dominated
    # mean.
    corrected, _ = motion_correction_rigid(
        np.asarray(clip, dtype=np.float32), output_path=None,
        max_shift=tuple(best["max_shift"]), gSig_filt=float(best["mc_gSig_filt"]),
        niter_rig=int(best["mc_n_iter"]), verbose=False)
    raw_cn = correlation_image(clip)
    cor_cn = correlation_image(np.asarray(corrected))
    vmax = float(np.percentile(np.concatenate([raw_cn.ravel(), cor_cn.ravel()]), 99.5))

    rows = result["rows"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    labels = [f"g{r['mc_gSig_filt']:.1f}\nms{min(r['max_shift'])}\nn{r['mc_n_iter']}"
              for r in rows]
    score = [r["score"] if np.isfinite(r["score"]) else 0.0 for r in rows]
    colors = ["tab:green" if r["is_best"]
              else ("tab:red" if (r["saturated"] or not r["beats_raw"]) else "tab:blue")
              for r in rows]
    ax[0].bar(range(len(rows)), score, color=colors)
    ax[0].set_xticks(range(len(rows)))
    ax[0].set_xticklabels(labels, fontsize=7)
    ax[0].axhline(result["evidence"]["search"]["raw_score"], ls="--", c="k", lw=1)
    ax[0].set_title("candidate score = mean Cn (green=best, red=rejected, --=raw)")
    ax[1].imshow(raw_cn, cmap="viridis", vmin=0, vmax=vmax)
    ax[1].set_title("raw clip Cn"); ax[1].axis("off")
    ax[2].imshow(cor_cn, cmap="viridis", vmin=0, vmax=vmax)
    ax[2].set_title("corrected (best) Cn"); ax[2].axis("off")
    fig.tight_layout()
    fig.savefig(run_dir / "mc_tuning.png", dpi=110)
    plt.close(fig)
