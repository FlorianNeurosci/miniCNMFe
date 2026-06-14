"""max vs nrg thresholding: does energy thresholding beat peak-relative?

Runs four extractions on the first 5000 frames of the PICAST mc.zarr, all with
min_pixel lowered (footprints shrink -> the acceptance floor must track):
  - max@0.1   (baseline peak-relative)
  - nrg@0.99
  - nrg@0.95
  - nrg@0.90

Figures are written **incrementally** to live_runs/nrg_compare_out/: each
setting's own overlay (full FOV + zoom) is saved as soon as its fit finishes, and
results.json is updated after every fit. A final side-by-side panel is written at
the end. Decision question: does nrg give a *more shape-appropriate* footprint
distribution and better trace purity (higher corr(C, C+YrA)) than max?
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minicnmfe.pipeline import CNMFe, CNMFeParams  # noqa: E402
from minicnmfe.preprocess import correlation_pnr  # noqa: E402

SESSION = Path(
    "/media/server/archive/projects/2023_intercontext/PICAST/data/"
    "1_preprocessed/20260505_m0010800_wt_1557/miniscope_video/minicnmfe_mc_mcid_0"
)
MC_ZARR = SESSION / "mc.zarr"
N_FRAMES = 5000
MIN_PIXEL = 70

# nrg levels: default sweep, or pass on the CLI e.g. `python nrg_compare.py 0.8 0.7 0.6`.
_args = sys.argv[1:]
LEVELS = [float(a) for a in _args] if _args else [0.99, 0.95, 0.90]
_TAG = ("_" + "-".join(f"{lv:g}" for lv in LEVELS)) if _args else ""
OUT_DIR = Path(__file__).parent / f"nrg_compare_out{_TAG}"

SETTINGS = [("01_max@0.1 (baseline)", dict())] + [
    (f"{i + 2:02d}_nrg@{lv:g}", dict(spatial_thr_method="nrg", spatial_nrg_thr=lv))
    for i, lv in enumerate(LEVELS)
]


def load_recommended_params() -> dict:
    cands = sorted(SESSION.glob("minicnmfe_tuning_taskid_0__*/recommended_params.json"),
                   key=lambda p: p.stat().st_mtime)
    with open(cands[-1]) as f:
        raw = json.load(f)
    valid = {f.name for f in dataclasses.fields(CNMFeParams)}
    kw = {k: v for k, v in raw.items() if k in valid}
    kw["min_pixel"] = MIN_PIXEL
    return kw


def footprint_npix(model):
    A = model.A.tocsc()
    return np.diff(A.indptr)


def corr_C_vs_CplusYrA(model):
    C = np.asarray(model.C)
    proj = C + np.asarray(model.YrA)
    rs = np.full(C.shape[0], np.nan)
    for k in range(C.shape[0]):
        if C[k].std() > 0 and proj[k].std() > 0:
            rs[k] = np.corrcoef(C[k], proj[k])[0, 1]
    order = np.argsort(C.max(axis=1))[::-1]
    return np.nanmean(rs), np.nanmean(rs[order[:30]])


def fit(movie, base_kw, **ov):
    kw = dict(base_kw); kw.update(ov); kw["n_jobs"] = -1
    m = CNMFe(CNMFeParams(**kw))
    m.fit(movie, do_motion_correction=False)
    return m


def draw_contours(ax, model, xs, ys):
    H, W = model.dims
    K = model.A.shape[1]
    A_dense = np.asarray(model.A.todense()).reshape(H, W, K)
    mask = getattr(model, "accepted_mask", None)
    for k in range(K):
        fp = A_dense[..., k]
        if fp.max() <= 0:
            continue
        accepted = (mask is None) or (len(mask) == K and bool(mask[k]))
        ax.contour(xs, ys, fp, levels=[0.3 * fp.max()],
                   colors="lime" if accepted else "red", linewidths=0.6)


def panel(ax, cn, model, xs, ys, cvmin, cvmax, title, zoom=None):
    ax.imshow(cn, cmap="gray", vmin=cvmin, vmax=cvmax)
    draw_contours(ax, model, xs, ys)
    if zoom is not None:
        zy0, zy1, zx0, zx1 = zoom
        ax.set_xlim(zx0, zx1); ax.set_ylim(zy1, zy0)
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"out dir: {OUT_DIR}", flush=True)
    print(f"Loading {N_FRAMES} frames...", flush=True)
    z = zarr.open(str(MC_ZARR), mode="r")
    movie = np.asarray(z[:N_FRAMES], dtype=np.float32)
    base_kw = load_recommended_params()
    sigma = float(base_kw.get("sigma", 3.0))
    print(f"  {movie.shape}; min_pixel={MIN_PIXEL} sigma={sigma}", flush=True)

    print("Computing correlation image...", flush=True)
    cn, pnr = correlation_pnr(movie, sigma=sigma, center_psf=True, n_jobs=-1, stride=2)
    H, W = cn.shape
    sm = uniform_filter(np.nan_to_num(cn) * np.nan_to_num(pnr), size=40)
    cy, cx = np.unravel_index(np.argmax(sm), sm.shape)
    half = 70
    zoom = (max(0, cy - half), min(H, cy + half), max(0, cx - half), min(W, cx + half))
    xs, ys = np.arange(W), np.arange(H)
    cvmax, cvmin = np.nanpercentile(cn, 99.7), np.nanpercentile(cn, 50)

    results, models = [], []
    for label, ov in SETTINGS:
        print(f"\n=== {label} ===", flush=True)
        t0 = time.perf_counter()
        m = fit(movie, base_kw, **ov)
        npix = footprint_npix(m)
        acc = int(m.accepted_mask.sum()) if m.accepted_mask is not None else -1
        r_all, r_top = corr_C_vs_CplusYrA(m)
        q25, q50, q75 = np.percentile(npix, [25, 50, 75]) if npix.size else (0, 0, 0)
        rec = {"label": label, "K": int(m.A.shape[1]), "accepted": acc,
               "npix_median": float(q50), "npix_iqr": [float(q25), float(q75)],
               "corr_mean": float(r_all), "corr_top30": float(r_top),
               "secs": time.perf_counter() - t0}
        results.append(rec); models.append((label, m))
        print(f"  K={rec['K']} acc={acc} npix med={q50:.0f} IQR=[{q25:.0f},{q75:.0f}] "
              f"corr mean={r_all:.3f} top30={r_top:.3f} ({rec['secs']:.0f}s)", flush=True)

        # --- incremental: this setting's own figure (full FOV + zoom) ---
        cap = (f"{label}\nK={rec['K']} acc={acc} npix med={q50:.0f} "
               f"IQR=[{q25:.0f},{q75:.0f}] corr={r_all:.3f}/{r_top:.3f}")
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        panel(ax[0], cn, m, xs, ys, cvmin, cvmax, cap)
        panel(ax[1], cn, m, xs, ys, cvmin, cvmax, f"{label} — zoom", zoom=zoom)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{label.replace('@', '').replace(' ', '_')}.png",
                    dpi=110, bbox_inches="tight")
        plt.close(fig)
        with open(OUT_DIR / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"  wrote panel + results.json", flush=True)

    # --- final side-by-side ---
    n = len(models)
    fig, axes = plt.subplots(2, n, figsize=(7 * n, 14))
    for j, (label, m) in enumerate(models):
        r = results[j]
        panel(axes[0, j], cn, m, xs, ys, cvmin, cvmax,
              f"{label}\nK={r['K']} acc={r['accepted']} npix={r['npix_median']:.0f} "
              f"corr={r['corr_mean']:.3f}")
        panel(axes[1, j], cn, m, xs, ys, cvmin, cvmax, f"{label} — zoom", zoom=zoom)
    fig.suptitle(f"max vs nrg thresholding (PICAST 5k snippet, sigma={sigma}, "
                 f"min_pixel={MIN_PIXEL})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUT_DIR / "side_by_side.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {OUT_DIR}/side_by_side.png", flush=True)


if __name__ == "__main__":
    main()
