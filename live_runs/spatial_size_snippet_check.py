"""Throwaway: real 5k-frame snippet check of the footprint-size knobs.

Loads the first 5000 frames of an already-MC'd mc.zarr into RAM and runs the
extraction three ways:
  1. recommended params as-is (defaults: lambda_scale=1.0, max_radius_factor=0.0)
  2. + spatial_lambda_scale=1.5
  3. + spatial_lambda_scale=1.5, spatial_max_radius_factor=2.0

Reports per setting: K (alive / accepted), footprint npix median/IQR, and
corr(C, C+YrA) mean + top-30-amplitude (the density<->purity proxy).
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import zarr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minicnmfe.pipeline import CNMFe, CNMFeParams  # noqa: E402

SESSION = Path(
    "/media/server/archive/projects/2023_intercontext/PICAST/data/"
    "1_preprocessed/20260505_m0010800_wt_1557/miniscope_video/minicnmfe_mc_mcid_0"
)
MC_ZARR = SESSION / "mc.zarr"
N_FRAMES = 5000


def load_recommended_params() -> dict:
    cands = sorted(SESSION.glob("minicnmfe_tuning_taskid_0__*/recommended_params.json"),
                   key=lambda p: p.stat().st_mtime)
    with open(cands[-1]) as f:
        raw = json.load(f)
    valid = {f.name for f in dataclasses.fields(CNMFeParams)}
    return {k: v for k, v in raw.items() if k in valid}


def footprint_npix(model) -> np.ndarray:
    A = model.A.tocsc()
    return np.diff(A.indptr)


def corr_C_vs_CplusYrA(model):
    C = np.asarray(model.C)
    YrA = np.asarray(model.YrA)
    proj = C + YrA
    K = C.shape[0]
    rs = np.full(K, np.nan)
    for k in range(K):
        a, b = C[k], proj[k]
        if a.std() > 0 and b.std() > 0:
            rs[k] = np.corrcoef(a, b)[0, 1]
    amp = C.max(axis=1)
    order = np.argsort(amp)[::-1]
    top = order[:30]
    return np.nanmean(rs), np.nanmean(rs[top])


def run(movie, base_kw, label, **overrides):
    kw = dict(base_kw)
    kw.update(overrides)
    kw["n_jobs"] = -1
    p = CNMFeParams(**kw)
    print(f"\n===== {label} =====", flush=True)
    t0 = time.perf_counter()
    model = CNMFe(p)
    model.fit(movie, do_motion_correction=False)
    dt = time.perf_counter() - t0
    npix = footprint_npix(model)
    acc = int(model.accepted_mask.sum()) if model.accepted_mask is not None else -1
    r_all, r_top = corr_C_vs_CplusYrA(model)
    q25, q50, q75 = np.percentile(npix, [25, 50, 75]) if npix.size else (0, 0, 0)
    print(f"[{label}] K_alive={model.A.shape[1]} K_accepted={acc} "
          f"npix median={q50:.0f} IQR=[{q25:.0f},{q75:.0f}] "
          f"corr(C,C+YrA) mean={r_all:.3f} top30={r_top:.3f} "
          f"({dt:.0f}s)", flush=True)
    return {"label": label, "K_alive": int(model.A.shape[1]), "K_accepted": acc,
            "npix_median": float(q50), "npix_iqr": [float(q25), float(q75)],
            "corr_mean": float(r_all), "corr_top30": float(r_top), "secs": dt}


def main():
    print(f"Loading first {N_FRAMES} frames of {MC_ZARR} ...", flush=True)
    z = zarr.open(str(MC_ZARR), mode="r")
    t0 = time.perf_counter()
    movie = np.asarray(z[:N_FRAMES], dtype=np.float32)
    print(f"  loaded {movie.shape} in {time.perf_counter()-t0:.0f}s "
          f"({movie.nbytes/1e9:.1f} GB)", flush=True)

    base_kw = load_recommended_params()
    base_kw["spatial_lambda_scale"] = 1.0
    base_kw["spatial_max_radius_factor"] = 0.0
    print("Base extraction params:", json.dumps(base_kw, default=str), flush=True)

    results = []
    results.append(run(movie, base_kw, "1_defaults"))
    results.append(run(movie, base_kw, "2_lambda1.5", spatial_lambda_scale=1.5))
    results.append(run(movie, base_kw, "3_lambda1.5_radius2.0",
                       spatial_lambda_scale=1.5, spatial_max_radius_factor=2.0))

    print("\n===== SUMMARY =====", flush=True)
    for r in results:
        print(json.dumps(r), flush=True)
    out = Path(__file__).parent / "spatial_size_snippet_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
