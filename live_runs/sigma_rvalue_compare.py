"""Does r_value (bbox/ΔF) discriminate sigma=3 (good) vs sigma=4 (the merged
winner)? Fit both at the default nrg=0.95 on the winner cutout; compare median
r_value, npix, multipeak_frac, K. If sigma=3 >> sigma=4 on r_value, the dominant
fix is the sigma selection (thread A), validated by r_value as a quality metric.
"""
from __future__ import annotations
import dataclasses, json, sys, time
from pathlib import Path
import numpy as np
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from minicnmfe.pipeline import CNMFe, CNMFeParams  # noqa: E402
from minicnmfe.preprocess import correlation_pnr  # noqa: E402
from tuning import metrics as M  # noqa: E402
from live_runs.rvalue_nrg_probe import spatial_r_values, cn_proxy, montage  # noqa: E402

MC = Path("/media/server/archive/projects/2023_intercontext/PICAST/data/1_preprocessed/"
          "20260505_m0010800_wt_1557/miniscope_video/minicnmfe_mc_mcid_0/mc.zarr")
REC = Path("/media/server/archive/projects/2023_intercontext/PICAST/data/2_processed/"
           "20260505_m0010800_wt_1557/miniscope_video/minicnmfe_mc_mcid_0/"
           "minicnmfe_tuning_taskid_0__20260614_194430_137177_0dbb4d/recommended_params.json")
Y0, Y1, X0, X1, T0, T1 = 67, 323, 246, 502, 14876, 17876
OUT = ROOT / "live_runs" / "sigma_rvalue_compare_out"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = zarr.open(str(MC), mode="r")
    cut = np.asarray(z[T0:T1, Y0:Y1, X0:X1], dtype=np.float32)
    dims = cut.shape[1:]
    print(f"cutout {cut.shape}", flush=True)
    raw = json.load(open(REC))
    valid = {f.name for f in dataclasses.fields(CNMFeParams)}
    base = {k: v for k, v in raw.items() if k in valid}

    rows = []
    for sigma in (3.0, 4.0):
        kw = dict(base); kw.update(sigma=sigma, spatial_nrg_thr=0.95, min_pixel=1, n_jobs=-1)
        p = CNMFeParams(**kw)
        print(f"\n=== sigma={sigma} (nrg=0.95) ===", flush=True)
        t0 = time.time()
        m = CNMFe(p); m.fit_extract(cut, evaluate=True)
        print(f"  fit {time.time()-t0:.0f}s K={m.A.shape[1]}", flush=True)
        cn, pnr = correlation_pnr(cut, sigma=sigma, center_psf=True, n_jobs=-1, stride=2)
        C = np.asarray(m.C); YrA = np.asarray(m.YrA)
        A = m.A.tocsc()
        rv = spatial_r_values(A, C, cut, dims)
        cp = cn_proxy(A, cn, dims)
        q = M.model_quality(m)
        npix = np.diff(A.indptr)
        rec = dict(sigma=sigma, K=int(m.A.shape[1]),
                   rvalue_median=float(np.nanmedian(rv)),
                   rvalue_p75=float(np.nanpercentile(rv, 75)),
                   cnproxy_median=float(np.nanmedian(cp)),
                   npix_median=float(np.median(npix)),
                   multipeak_frac=float(q.get("multipeak_frac", float("nan"))),
                   cprojcorr_median=float(q.get("cprojcorr_median", float("nan"))))
        rows.append(rec)
        print(f"  r_med={rec['rvalue_median']:.3f} r_p75={rec['rvalue_p75']:.3f} "
              f"cn={rec['cnproxy_median']:.3f} npix={rec['npix_median']:.0f} "
              f"multipeak={rec['multipeak_frac']:.2f} cproj={rec['cprojcorr_median']:.3f}",
              flush=True)
        # montage: 8 highest-npix cells (the suspect oversized/merged ones)
        order = [int(k) for k in np.argsort(-npix)[:8]]
        montage(A, C, YrA, cn, dims, order, rv, cp,
                f"sigma={sigma} nrg=0.95 (8 largest footprints)",
                OUT / f"montage_sigma{int(sigma)}.png")

    json.dump(rows, open(OUT / "compare.json", "w"), indent=2)
    print("\n=== SUMMARY ===", flush=True)
    for r in rows:
        print(json.dumps(r), flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
