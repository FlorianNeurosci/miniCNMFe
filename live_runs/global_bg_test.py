"""Does raising global_bg_rank (or a temporal detrend) decontaminate the traces?
Fit the winner cutout under several background configs; measure how dominant the
shared signal stays (PC1 variance, median pairwise trace corr) without wrecking
footprints (K, spatialcorr, npix)."""
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
from tuning.metrics import per_cell_spatial_corr  # noqa: E402

MC = Path("/media/server/archive/projects/2023_intercontext/PICAST/data/1_preprocessed/"
          "20260505_m0010800_wt_1557/miniscope_video/minicnmfe_mc_mcid_0/mc.zarr")
REC = Path("/media/server/archive/projects/2023_intercontext/PICAST/data/2_processed/"
           "20260505_m0010800_wt_1557/miniscope_video/minicnmfe_mc_mcid_0/"
           "minicnmfe_tuning_taskid_0__20260614_194430_137177_0dbb4d/recommended_params.json")
Y0, Y1, X0, X1, T0, T1 = 67, 323, 246, 502, 14876, 17876
OUT = ROOT / "live_runs" / "global_bg_test_out"

CONFIGS = [
    ("rank1", dict(global_bg_rank=1)),
    ("rank2", dict(global_bg_rank=2)),
    ("rank3", dict(global_bg_rank=3)),
    ("rank1+detrend3", dict(global_bg_rank=1, temporal_detrend_order=3)),
    ("rank2+detrend3", dict(global_bg_rank=2, temporal_detrend_order=3)),
]


def pc1_and_pairwise(C_acc):
    Xc = C_acc - C_acc.mean(axis=1, keepdims=True)
    sub = Xc if Xc.shape[0] <= 200 else Xc[np.linspace(0, Xc.shape[0]-1, 200).astype(int)]
    s = np.linalg.svd(sub, compute_uv=False)
    var1 = float(s[0] ** 2 / (s ** 2).sum())
    cc = np.corrcoef(sub)
    iu = np.triu_indices(cc.shape[0], 1)
    v = np.abs(cc[iu]); v = v[np.isfinite(v)]
    return var1, float(np.median(v)) if v.size else float("nan")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = zarr.open(str(MC), mode="r")
    cut = np.asarray(z[T0:T1, Y0:Y1, X0:X1], dtype=np.float32)
    H, W = cut.shape[1:]
    raw = json.load(open(REC))
    valid = {f.name for f in dataclasses.fields(CNMFeParams)}
    base = {k: v for k, v in raw.items() if k in valid}
    base.update(sigma=3.0, min_corr=0.822, min_pnr=4.859, spatial_nrg_thr=0.95,
                min_pixel=1, n_jobs=-1)
    print("cn...", flush=True)
    cn, _ = correlation_pnr(cut, sigma=3.0, center_psf=True, n_jobs=-1, stride=2)

    rows = []
    for label, ov in CONFIGS:
        kw = dict(base); kw.update(ov)
        p = CNMFeParams(**kw)
        print(f"\n=== {label} ===", flush=True)
        t0 = time.time()
        m = CNMFe(p); m.fit_extract(cut, evaluate=True)
        C = np.asarray(m.C); K = C.shape[0]
        mask = m.accepted_mask if m.accepted_mask is not None else np.ones(K, bool)
        acc = np.flatnonzero(mask)
        var1, medpair = pc1_and_pairwise(C[acc]) if len(acc) >= 2 else (float("nan"), float("nan"))
        sc = per_cell_spatial_corr(m.A, cn, (H, W))
        npix = np.diff(m.A.tocsc().indptr)
        rec = dict(label=label, K=int(K), accepted=int(len(acc)),
                   pc1_var=var1, pairwise_corr=medpair,
                   spatialcorr_median=float(np.nanmedian(sc)),
                   npix_median=float(np.median(npix)), secs=time.time()-t0)
        rows.append(rec)
        print(f"  K={K} acc={len(acc)} PC1var={var1:.1%} pairwise={medpair:.3f} "
              f"spatialcorr={rec['spatialcorr_median']:.3f} npix={rec['npix_median']:.0f} "
              f"({rec['secs']:.0f}s)", flush=True)
        # top-8 traces overlaid (visual decontamination check)
        cpc = []
        Cproj = C + np.asarray(m.YrA)
        for k in acc:
            a = C[k]-C[k].mean(); b = Cproj[k]-Cproj[k].mean()
            d = np.sqrt((a**2).sum()*(b**2).sum())+1e-12
            cpc.append((a*b).sum()/d)
        top = acc[np.argsort(-np.array(cpc))[:8]]
        fig, ax = plt.subplots(figsize=(12, 5))
        for i, k in enumerate(top):
            tr = C[k]; tr = (tr-tr.min())/(np.ptp(tr)+1e-9)
            ax.plot(tr+i, lw=0.5)
        ax.set_title(f"{label}: top-8 traces (PC1var={var1:.0%} pairwise|r|={medpair:.2f})")
        ax.set_yticks([]); fig.tight_layout()
        fig.savefig(OUT / f"traces_{label}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

    json.dump(rows, open(OUT / "results.json", "w"), indent=2)
    # summary bars
    labels = [r["label"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
    ax[0].bar(labels, [r["pc1_var"] for r in rows], color="tab:red")
    ax[0].set_title("PC1 variance explained (lower=less global dominance)")
    ax[1].bar(labels, [r["pairwise_corr"] for r in rows], color="tab:orange")
    ax[1].set_title("median pairwise |trace corr| (lower=cleaner)")
    ax[2].bar(labels, [r["spatialcorr_median"] for r in rows], color="tab:green")
    ax[2].set_title("spatialcorr median (footprint quality; want stable)")
    for a in ax:
        a.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(OUT / "summary.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("\n=== SUMMARY ===", flush=True)
    for r in rows:
        print(json.dumps(r), flush=True)
    print(f"out: {OUT}", flush=True)


if __name__ == "__main__":
    main()
