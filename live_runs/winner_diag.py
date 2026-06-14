"""Diagnostic on the sigma=3 winner (PICAST cutout):
Q1 why are the top-corr traces mutually correlated, and Q2 are footprints really
oversized or is it a cn display-threshold artifact.
"""
from __future__ import annotations
import dataclasses, json, sys, time
from pathlib import Path
import numpy as np
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from minicnmfe.pipeline import CNMFe, CNMFeParams  # noqa: E402
from minicnmfe.preprocess import correlation_pnr  # noqa: E402
from tuning.metrics import _per_cell_corr  # noqa: E402

MC = Path("/media/server/archive/projects/2023_intercontext/PICAST/data/1_preprocessed/"
          "20260505_m0010800_wt_1557/miniscope_video/minicnmfe_mc_mcid_0/mc.zarr")
REC = Path("/media/server/archive/projects/2023_intercontext/PICAST/data/2_processed/"
           "20260505_m0010800_wt_1557/miniscope_video/minicnmfe_mc_mcid_0/"
           "minicnmfe_tuning_taskid_0__20260614_194430_137177_0dbb4d/recommended_params.json")
Y0, Y1, X0, X1, T0, T1 = 67, 323, 246, 502, 14876, 17876
OUT = ROOT / "live_runs" / "winner_diag_out"


def footprint_centroids(A_csc, H, W):
    cents = []
    for k in range(A_csc.shape[1]):
        s, e = A_csc.indptr[k], A_csc.indptr[k + 1]
        rows = A_csc.indices[s:e]
        if rows.size:
            cents.append(((rows // W).mean(), (rows % W).mean()))
        else:
            cents.append((np.nan, np.nan))
    return np.array(cents)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = zarr.open(str(MC), mode="r")
    cut = np.asarray(z[T0:T1, Y0:Y1, X0:X1], dtype=np.float32)
    H, W = cut.shape[1:]
    print(f"cutout {cut.shape}", flush=True)

    raw = json.load(open(REC))
    valid = {f.name for f in dataclasses.fields(CNMFeParams)}
    kw = {k: v for k, v in raw.items() if k in valid}
    # winner config from the rerun: sigma=3 + its thresholds
    kw.update(sigma=3.0, min_corr=0.822, min_pnr=4.859, spatial_nrg_thr=0.95,
              min_pixel=1, n_jobs=-1)
    p = CNMFeParams(**kw)
    print("fit winner sigma=3 ...", flush=True)
    t0 = time.time()
    m = CNMFe(p); m.fit_extract(cut, evaluate=True)
    print(f"  fit {time.time()-t0:.0f}s K={m.A.shape[1]}", flush=True)
    cn, pnr = correlation_pnr(cut, sigma=p.sigma, center_psf=True, n_jobs=-1, stride=2)

    A = m.A.tocsc()
    C = np.asarray(m.C); YrA = np.asarray(m.YrA)
    K = C.shape[0]
    mask = m.accepted_mask if m.accepted_mask is not None else np.ones(K, bool)
    acc = np.flatnonzero(mask)
    cpc = _per_cell_corr(C, C + YrA)

    # ---------- Q2: footprints on cn at a vmin ladder ----------
    A_dense = np.asarray(A.todense()).reshape(H, W, K)
    xs, ys = np.arange(W), np.arange(H)
    vmins = [("min_corr=0.82", 0.82), ("p50", np.nanpercentile(cn, 50)),
             ("p20", np.nanpercentile(cn, 20)), ("p2", np.nanpercentile(cn, 2))]
    cvmax = np.nanpercentile(cn, 99.7)
    fig, ax = plt.subplots(1, 4, figsize=(26, 7))
    for j, (lbl, vmin) in enumerate(vmins):
        ax[j].imshow(cn, cmap="gray", vmin=vmin, vmax=cvmax)
        for k in range(K):
            fp = A_dense[..., k]
            if fp.max() <= 0:
                continue
            ax[j].contour(xs, ys, fp, levels=[0.3 * fp.max()],
                          colors="lime" if mask[k] else "red", linewidths=0.5)
        ax[j].set_title(f"cn vmin={lbl}", fontsize=11); ax[j].axis("off")
    fig.suptitle(f"Q2: winner sigma=3 footprints (0.3·peak contour) over cn at "
                 f"different DISPLAY thresholds — K={K}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "footprints_vs_cnthresh.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote footprints_vs_cnthresh.png", flush=True)

    # contour-level variant (fixed low vmin) — core vs extent
    fig, ax = plt.subplots(1, 3, figsize=(20, 7))
    for j, lev in enumerate([0.2, 0.3, 0.5]):
        ax[j].imshow(cn, cmap="gray", vmin=np.nanpercentile(cn, 20), vmax=cvmax)
        for k in range(K):
            fp = A_dense[..., k]
            if fp.max() <= 0:
                continue
            ax[j].contour(xs, ys, fp, levels=[lev * fp.max()],
                          colors="lime" if mask[k] else "red", linewidths=0.5)
        ax[j].set_title(f"contour at {lev:.1f}·peak", fontsize=11); ax[j].axis("off")
    fig.suptitle("Q2b: footprint contour level (over cn vmin=p20) — core vs extent",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "footprints_contour_levels.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote footprints_contour_levels.png", flush=True)

    # ---------- Q1: top-trace correlation ----------
    accC = C[acc]
    cpc_acc = cpc[acc]
    topN = min(30, len(acc))
    top_local = np.argsort(-cpc_acc)[:topN]
    top = acc[top_local]
    rng = np.random.RandomState(0)
    rnd = acc[np.sort(rng.choice(len(acc), size=min(topN, len(acc)), replace=False))]

    def med_pair(idx):
        if len(idx) < 2:
            return float("nan")
        cc = np.corrcoef(C[idx])
        iu = np.triu_indices(len(idx), 1)
        v = np.abs(cc[iu]); v = v[np.isfinite(v)]
        return float(np.median(v)) if v.size else float("nan")

    # PC1 of accepted traces
    Xc = accC - accC.mean(axis=1, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pc1 = Vt[0]
    var1 = float(S[0] ** 2 / (S ** 2).sum())
    Xc_resid = Xc - np.outer(Xc @ pc1, pc1)
    # map back to accepted indexing for top cells
    top_in_acc = np.array([np.where(acc == t)[0][0] for t in top])

    def med_pair_resid(local_idx):
        sub = Xc_resid[local_idx]
        cc = np.corrcoef(sub)
        iu = np.triu_indices(len(local_idx), 1)
        v = np.abs(cc[iu]); v = v[np.isfinite(v)]
        return float(np.median(v)) if v.size else float("nan")

    # corr of each top trace with the rank-1 background f(t)
    fbg = np.asarray(m.f).ravel() if getattr(m, "f", None) is not None else None
    f_corr = None
    if fbg is not None and fbg.std() > 0:
        f_corr = np.array([abs(np.corrcoef(C[t], fbg)[0, 1]) for t in top])

    print("\n===== Q1 STATS =====", flush=True)
    print(f"K={K} accepted={len(acc)}", flush=True)
    print(f"median |pairwise corr|  top30={med_pair(top):.3f}  random30={med_pair(rnd):.3f}",
          flush=True)
    print(f"PC1 variance explained (accepted traces) = {var1:.2%}", flush=True)
    print(f"median |pairwise corr| top30 AFTER removing PC1 = {med_pair_resid(top_in_acc):.3f}",
          flush=True)
    if f_corr is not None:
        print(f"top30 |corr with rank-1 bg f(t)|: median={np.median(f_corr):.3f} "
              f"max={f_corr.max():.3f}", flush=True)

    # figure: top traces raw vs PC1-removed
    fig, ax = plt.subplots(1, 2, figsize=(20, 10), sharex=True)
    for i, t in enumerate(top):
        tr = C[t]; tr = (tr - tr.min()) / (np.ptp(tr) + 1e-9)
        ax[0].plot(tr + i, lw=0.5, color="tab:blue")
    ax[0].set_title(f"top-30 by cprojcorr — RAW C (median pairwise |r|={med_pair(top):.2f})")
    for i, lk in enumerate(top_in_acc):
        tr = Xc_resid[lk]; tr = (tr - tr.min()) / (np.ptp(tr) + 1e-9)
        ax[1].plot(tr + i, lw=0.5, color="tab:green")
    ax[1].set_title(f"same, PC1 removed (PC1 expl {var1:.0%}; "
                    f"median pairwise |r|={med_pair_resid(top_in_acc):.2f})")
    for a in ax:
        a.set_yticks([]); a.set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(OUT / "top_trace_corr.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote top_trace_corr.png", flush=True)

    # pairwise corr heatmap (top30)
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    im0 = ax[0].imshow(np.corrcoef(C[top]), vmin=-1, vmax=1, cmap="RdBu_r")
    ax[0].set_title("top-30 pairwise corr (raw)"); fig.colorbar(im0, ax=ax[0], shrink=0.8)
    im1 = ax[1].imshow(np.corrcoef(Xc_resid[top_in_acc]), vmin=-1, vmax=1, cmap="RdBu_r")
    ax[1].set_title("top-30 pairwise corr (PC1 removed)"); fig.colorbar(im1, ax=ax[1], shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "pairwise_corr_heatmap.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote pairwise_corr_heatmap.png", flush=True)

    # centroids of top cells on cn
    cents = footprint_centroids(A, H, W)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cn, cmap="gray", vmin=np.nanpercentile(cn, 20), vmax=cvmax)
    ax.scatter(cents[top, 1], cents[top, 0], s=60, facecolors="none",
               edgecolors="red", linewidths=1.5)
    ax.set_title("top-30-by-cprojcorr centroids on cn (clustered=local/dup, spread=global)")
    ax.axis("off"); fig.tight_layout()
    fig.savefig(OUT / "top_centroids_on_cn.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote top_centroids_on_cn.png", flush=True)
    print(f"\nout: {OUT}", flush=True)


if __name__ == "__main__":
    main()
