"""A/B for todo/oasis_oversmoothing.md — fudge_factor=0.85 + temporal_detrend_order=1.

Mirrors demo_notebooks/02_extract_components.ipynb cells 1-7 on the demo
session's mc.zarr. Saves diagnostic PNGs + numerical stats so the trace
quality can be compared against the existing notebook outputs.

Outputs (in tmp/ab_oasis/):
  trace_plot.png   — top-10 accepted traces (C, C+YrA, S)
  raw_5_traces.png — first 5 C_projected (mirrors cell 18)
  footprints.png   — mean image + footprint contours
  stats.txt        — K, accepted, g min/max/mean, sn min/max/mean, wall time
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cnmfe.io import open_zarr
from cnmfe.pipeline import CNMFe, CNMFeParams

PROJECT_ROOT = Path("/home/fs539/code/simpler_cnmfe")
MC_ZARR = PROJECT_ROOT / "demo_movies" / "demo_session" / "mc_output" / "mc.zarr"
OUT_DIR = PROJECT_ROOT / "tmp" / "ab_oasis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

t_total = time.time()

# --- Load (mirrors notebook cells 2-3) ---
mc_zarr = open_zarr(MC_ZARR)
T_full, H, W = mc_zarr.shape
t0 = time.time()
movie = np.asarray(mc_zarr, dtype=np.float32)
movie = movie[:10000]
T = movie.shape[0]
print(f"loaded {movie.shape} ({movie.nbytes/1e9:.2f} GB) in {time.time()-t0:.1f}s")

# --- Params (same as notebook cell 10 with A/B knobs) ---
SIGMA = 5.0
params = CNMFeParams(
    sigma=SIGMA,
    min_corr=0.9,
    min_pnr=15.0,
    max_neurons=None,
    n_iter_main=2,
    n_iter_temporal=2,
    merge_thr_corr=0.85,
    merge_thr_overlap=0.5,
    merge_centre_dist_factor=2.0,
    spatial_max_thr=0.1,
    ar_order=1,
    global_ar=True,
    skip_first_deconv=True,
    fudge_factor=0.85,                # A/B
    temporal_detrend_order=1,         # A/B
    init_stride=10,
    init_corrpnr_stride=None,
    min_pixel=3,
    auto_eval_snr_amp_thr=3.0,
    n_jobs=-1,
    device="cpu",
    ring_constrain_sum=True,
)
print(f"A/B: fudge_factor={params.fudge_factor} temporal_detrend_order={params.temporal_detrend_order}")

# --- Fit ---
model = CNMFe(params)
t0 = time.time()
model.fit(movie, do_motion_correction=False)
fit_elapsed = time.time() - t0
K = model.A.shape[1]
acc_idx = np.flatnonzero(model.accepted_mask)
print(f"fit {K} components ({len(acc_idx)} accepted) in {fit_elapsed:.1f}s")

# --- Stats ---
g_arr = np.array([float(g[0]) for g in model.g])
sn_arr = np.asarray(model.sn_per_k)
stats_txt = OUT_DIR / "stats.txt"
stats_txt.write_text(
    f"A/B params: fudge_factor={params.fudge_factor} "
    f"temporal_detrend_order={params.temporal_detrend_order}\n"
    f"n_iter_main={params.n_iter_main} n_iter_temporal={params.n_iter_temporal}\n"
    f"T={T} H={H} W={W} K={K} accepted={len(acc_idx)}\n"
    f"wall time fit: {fit_elapsed:.1f}s\n"
    f"g  per-component (AR coef): "
    f"min={g_arr.min():.4f} mean={g_arr.mean():.4f} max={g_arr.max():.4f}\n"
    f"sn per-component         : "
    f"min={sn_arr.min():.4f} mean={sn_arr.mean():.4f} max={sn_arr.max():.4f}\n"
)
print(stats_txt.read_text())

# --- Footprints (mirrors notebook cell 16) ---
mean_img = movie.mean(axis=0)
A_dense = np.asarray(model.A.todense()).reshape(H, W, K)
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
axes[0].imshow(
    mean_img, cmap="gray",
    vmin=np.percentile(mean_img, 1), vmax=np.percentile(mean_img, 99),
)
for k in range(K):
    fp = A_dense[..., k]
    if fp.max() > 0:
        axes[0].contour(fp, levels=[fp.max() * 0.3], colors="lime", linewidths=0.7)
axes[0].set_title(f"Mean image + {K} footprint contours")
axes[0].axis("off")
axes[1].imshow(A_dense.max(axis=2), cmap="hot")
axes[1].set_title("Footprint max-projection")
axes[1].axis("off")
plt.tight_layout()
plt.savefig(OUT_DIR / "footprints.png", dpi=110, bbox_inches="tight")
plt.close()

# --- Top-10 accepted traces (mirrors notebook cell 17) ---
N_TRACES = min(10, len(acc_idx))
peak_acc = model.C[acc_idx].max(axis=1)
top_k = acc_idx[np.argsort(-peak_acc)[:N_TRACES]]
C_proj = model.C_projected

fig, axes = plt.subplots(N_TRACES, 1, figsize=(11, 1.6 * N_TRACES), sharex=True)
if N_TRACES == 1:
    axes = [axes]
for ax, k in zip(axes, top_k):
    ax.plot(C_proj[k], lw=0.7, color="steelblue", alpha=0.7, label="C + YrA")
    ax.plot(model.C[k], lw=1.0, color="forestgreen", label="C (OASIS)")
    ax2 = ax.twinx()
    nz = np.flatnonzero(model.S[k])
    if nz.size:
        ax2.vlines(nz, 0, model.S[k][nz], color="tomato", alpha=0.6, linewidth=1.0)
    ax2.set_yticks([])
    ax.set_ylabel(f"k = {k}", fontsize=8)
    ax.legend(loc="upper right", fontsize=7)
axes[-1].set_xlabel("Frame")
plt.suptitle(
    f"A/B fudge={params.fudge_factor} detrend={params.temporal_detrend_order} — "
    f"C (green), C+YrA (blue), S (red)",
    y=1.02,
)
plt.tight_layout()
plt.savefig(OUT_DIR / "trace_plot.png", dpi=110, bbox_inches="tight")
plt.close()

# --- 5-trace raw plot (mirrors notebook cell 18) ---
plt.figure()
plt.plot(C_proj.T[:2000, :5])
plt.title(f"C_projected first 5 components (frames 0-2000)")
plt.xlabel("Frame")
plt.savefig(OUT_DIR / "raw_5_traces.png", dpi=110, bbox_inches="tight")
plt.close()

print(f"total: {time.time()-t_total:.1f}s")
print(f"outputs in {OUT_DIR}")
