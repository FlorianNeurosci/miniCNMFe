"""Figures for the footprint-size fix (radius cap + min_pixel retune).

Re-runs three extractions on the first 5000 frames of the PICAST mc.zarr and
overlays footprint contours on the correlation image:
  A. defaults (min_pixel=211)
  B. radius=2.5, min_pixel=211   -> shows the min_pixel gotcha (all rejected)
  C. radius=2.5, min_pixel=70    -> the fix (tight footprints, accepted)

Accepted footprints drawn green, rejected red. Full FOV + a zoom on the
densest region. Writes live_runs/spatial_size_compare.png.
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
OUT = Path(__file__).parent / "spatial_size_compare.png"


def load_recommended_params() -> dict:
    cands = sorted(SESSION.glob("minicnmfe_tuning_taskid_0__*/recommended_params.json"),
                   key=lambda p: p.stat().st_mtime)
    with open(cands[-1]) as f:
        raw = json.load(f)
    valid = {f.name for f in dataclasses.fields(CNMFeParams)}
    return {k: v for k, v in raw.items() if k in valid}


def footprint_npix(model):
    A = model.A.tocsc()
    return np.diff(A.indptr)


def fit(movie, base_kw, **overrides):
    kw = dict(base_kw)
    kw.update(overrides)
    kw["n_jobs"] = -1
    p = CNMFeParams(**kw)
    m = CNMFe(p)
    m.fit(movie, do_motion_correction=False)
    return m


def draw_contours(ax, model, xs, ys):
    H, W = model.dims
    K = model.A.shape[1]
    A_dense = np.asarray(model.A.todense()).reshape(H, W, K)
    mask = getattr(model, "accepted_mask", None)
    n_acc = 0
    for k in range(K):
        fp = A_dense[..., k]
        if fp.max() <= 0:
            continue
        accepted = (mask is None) or (len(mask) == K and bool(mask[k]))
        n_acc += int(accepted)
        col = "lime" if accepted else "red"
        ax.contour(xs, ys, fp, levels=[0.3 * fp.max()], colors=col, linewidths=0.6)
    return K, n_acc


def main():
    print(f"Loading {N_FRAMES} frames...", flush=True)
    z = zarr.open(str(MC_ZARR), mode="r")
    movie = np.asarray(z[:N_FRAMES], dtype=np.float32)
    print(f"  {movie.shape} ({movie.nbytes/1e9:.1f} GB)", flush=True)

    base_kw = load_recommended_params()
    sigma = float(base_kw.get("sigma", 3.0))

    print("Computing correlation image...", flush=True)
    cn, pnr = correlation_pnr(movie, sigma=sigma, center_psf=True, n_jobs=-1, stride=2)

    # Zoom on the densest region (peak of a smoothed CORR*PNR product).
    prod = np.nan_to_num(cn) * np.nan_to_num(pnr)
    from scipy.ndimage import uniform_filter
    sm = uniform_filter(prod, size=40)
    cy, cx = np.unravel_index(np.argmax(sm), sm.shape)
    H, W = cn.shape
    half = 70
    zy0, zy1 = max(0, cy - half), min(H, cy + half)
    zx0, zx1 = max(0, cx - half), min(W, cx + half)
    print(f"  zoom @ ({cy},{cx}) -> rows {zy0}:{zy1} cols {zx0}:{zx1}", flush=True)

    settings = [
        ("A. defaults", dict()),
        ("B. radius=2.5 (min_pixel=211)", dict(spatial_max_radius_factor=2.5)),
        ("C. radius=2.5 + min_pixel=70 (fix)",
         dict(spatial_max_radius_factor=2.5, min_pixel=70)),
    ]

    models = []
    for label, ov in settings:
        print(f"\n=== {label} ===", flush=True)
        t0 = time.perf_counter()
        m = fit(movie, base_kw, **ov)
        npix = footprint_npix(m)
        acc = int(m.accepted_mask.sum()) if m.accepted_mask is not None else -1
        print(f"  K={m.A.shape[1]} accepted={acc} npix_median={np.median(npix):.0f} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
        models.append((label, m))

    xs, ys = np.arange(W), np.arange(H)
    cvmax = np.nanpercentile(cn, 99.7)
    cvmin = np.nanpercentile(cn, 50)

    fig, axes = plt.subplots(2, 3, figsize=(21, 14))
    for j, (label, m) in enumerate(models):
        # Full FOV
        ax = axes[0, j]
        ax.imshow(cn, cmap="gray", vmin=cvmin, vmax=cvmax)
        K, n_acc = draw_contours(ax, m, xs, ys)
        ax.set_title(f"{label}\nK={K} accepted={n_acc} "
                     f"(green=accepted, red=rejected)", fontsize=11)
        ax.axis("off")
        # Zoom
        axz = axes[1, j]
        axz.imshow(cn, cmap="gray", vmin=cvmin, vmax=cvmax)
        draw_contours(axz, m, xs, ys)
        axz.set_xlim(zx0, zx1); axz.set_ylim(zy1, zy0)
        axz.set_title(f"{label} — zoom", fontsize=11)
        axz.axis("off")

    fig.suptitle("Footprint size: defaults vs sigma radius cap (PICAST 5k snippet, "
                 f"sigma={sigma})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
