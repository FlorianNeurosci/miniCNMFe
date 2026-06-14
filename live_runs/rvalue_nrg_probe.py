"""PROTOTYPE (throwaway): does a CaImAn-style r_value + a cn-patch proxy find the
right footprint size? Probe on the real PICAST winner cutout.

Fit once at a LOOSE nrg (0.999) so footprints keep maximal support, then
re-threshold tighter across a range of nrg_thr (no re-fit, C/YrA held fixed) and
track: median true r_value, median cn-proxy, npix_median, corr(C,C+YrA). Emits
curves + per-cell montages (footprint over local CORR + trace) at several levels.
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
from minicnmfe._utils import make_2d  # noqa: E402
from minicnmfe.spatial import threshold_footprint  # noqa: E402
from minicnmfe.preprocess import correlation_pnr  # noqa: E402

MC = Path("/media/server/archive/projects/2023_intercontext/PICAST/data/1_preprocessed/"
          "20260505_m0010800_wt_1557/miniscope_video/minicnmfe_mc_mcid_0/mc.zarr")
REC = Path("/media/server/archive/projects/2023_intercontext/PICAST/data/2_processed/"
           "20260505_m0010800_wt_1557/miniscope_video/minicnmfe_mc_mcid_0/"
           "minicnmfe_tuning_taskid_0__20260614_194430_137177_0dbb4d/recommended_params.json")
# winner cutout from the tuning report: ((y0,y1,x0,x1),(t0,t1))
Y0, Y1, X0, X1 = 67, 323, 246, 502
T0, T1 = 14876, 17876
LEVELS = [0.999, 0.99, 0.97, 0.95, 0.92, 0.90, 0.87, 0.85, 0.80, 0.75]
OUT = ROOT / "live_runs" / "rvalue_nrg_probe_out"


def _bbox(rows, H, W, pad):
    ys, xs = rows // W, rows % W
    return (max(0, int(ys.min()) - pad), min(H, int(ys.max()) + 1 + pad),
            max(0, int(xs.min()) - pad), min(W, int(xs.max()) + 1 + pad))


def _fp_box(rows, vals, y0, y1, x0, x1, W):
    """Footprint values laid into the bbox (zeros outside support)."""
    ys, xs = rows // W, rows % W
    box = np.zeros((y1 - y0, x1 - x0), dtype=np.float64)
    box[ys - y0, xs - x0] = vals
    return box.ravel()


def spatial_r_values(A_csc, C, cut, dims, peak_frac=0.95, min_peak=10, pad=4):
    """CaImAn-style space corr over a bounding box (footprint + dark surround) of
    the ACTIVITY image (ΔF = mean over peak frames − mean over all frames)."""
    H, W = dims
    K = A_csc.shape[1]
    mean_all = cut.mean(axis=0)  # (H,W) baseline
    r = np.full(K, np.nan)
    for k in range(K):
        s, e = A_csc.indptr[k], A_csc.indptr[k + 1]
        if s == e:
            continue
        rows = A_csc.indices[s:e]
        vals = A_csc.data[s:e].astype(np.float64)
        y0, y1, x0, x1 = _bbox(rows, H, W, pad)
        c = C[k]
        thr = np.quantile(c, peak_frac)
        peak = np.where(c >= thr)[0]
        if peak.size < min_peak:
            peak = np.argsort(c)[-min_peak:]
        df = (cut[peak, y0:y1, x0:x1].mean(axis=0) - mean_all[y0:y1, x0:x1]).ravel()
        fp = _fp_box(rows, vals, y0, y1, x0, x1, W)
        if df.std() > 0 and fp.std() > 0:
            r[k] = float(np.corrcoef(df, fp)[0, 1])
        else:
            r[k] = 0.0
    return r


def cn_proxy(A_csc, cn, dims, pad=4):
    """Footprint vs local cn over a bounding box (footprint + surround)."""
    H, W = dims
    K = A_csc.shape[1]
    r = np.full(K, np.nan)
    for k in range(K):
        s, e = A_csc.indptr[k], A_csc.indptr[k + 1]
        if s == e:
            continue
        rows = A_csc.indices[s:e]
        vals = A_csc.data[s:e].astype(np.float64)
        y0, y1, x0, x1 = _bbox(rows, H, W, pad)
        cvals = np.nan_to_num(cn[y0:y1, x0:x1]).ravel()
        fp = _fp_box(rows, vals, y0, y1, x0, x1, W)
        if cvals.std() > 0 and fp.std() > 0:
            r[k] = float(np.corrcoef(fp, cvals)[0, 1])
        else:
            r[k] = 0.0
    return r


def rethreshold(A_loose, dims, p, nrg):
    """Re-threshold every footprint of A_loose at nrg level -> new csc."""
    n_pix, K = A_loose.shape
    cols, rows, data = [], [], []
    A_csc = A_loose.tocsc()
    for k in range(K):
        s, e = A_csc.indptr[k], A_csc.indptr[k + 1]
        if s == e:
            continue
        flat = np.zeros(n_pix, dtype=np.float32)
        flat[A_csc.indices[s:e]] = A_csc.data[s:e]
        new = threshold_footprint(
            flat, dims, max_thr=p.spatial_max_thr,
            closing_radius=p.spatial_close_radius,
            circular_max_dist_factor=p.spatial_circular_max_dist_factor,
            sigma=p.sigma, max_radius_factor=p.spatial_max_radius_factor,
            thr_method="nrg", nrg_thr=nrg)
        nz = np.where(new > 0)[0]
        rows.append(nz); cols.append(np.full(nz.size, k)); data.append(new[nz])
    if rows:
        return sp.csc_matrix((np.concatenate(data),
                              (np.concatenate(rows), np.concatenate(cols))),
                             shape=(n_pix, K), dtype=np.float32)
    return sp.csc_matrix((n_pix, K), dtype=np.float32)


def montage(A_csc, C, YrA, cn, dims, cell_ids, rvals, cnp, title, path):
    H, W = dims
    n = len(cell_ids)
    fig, ax = plt.subplots(n, 2, figsize=(11, 2.0 * n))
    for i, k in enumerate(cell_ids):
        col = np.asarray(A_csc.getcol(k).todense()).ravel().reshape(H, W)
        ys, xs = np.where(col > 0)
        if ys.size:
            cy, cx = int(ys.mean()), int(xs.mean())
        else:
            cy, cx = H // 2, W // 2
        r = 22
        y0, y1 = max(0, cy - r), min(H, cy + r)
        x0, x1 = max(0, cx - r), min(W, cx + r)
        a0 = ax[i, 0]
        a0.imshow(cn[y0:y1, x0:x1], cmap="gray",
                  vmin=np.nanpercentile(cn, 50), vmax=np.nanpercentile(cn, 99.7),
                  extent=[x0, x1, y1, y0])
        if col.max() > 0:
            a0.contour(np.arange(W), np.arange(H), col,
                       levels=[0.3 * col.max()], colors="lime", linewidths=1.0)
        a0.set_xlim(x0, x1); a0.set_ylim(y1, y0); a0.axis("off")
        a0.set_title(f"k={k} npix={int((col>0).sum())} r={rvals[k]:.2f} cn={cnp[k]:.2f}",
                     fontsize=8)
        a1 = ax[i, 1]
        a1.plot(C[k] + YrA[k], color="0.6", lw=0.5)
        a1.plot(C[k], color="tab:blue", lw=0.6)
        a1.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"out: {OUT}", flush=True)
    z = zarr.open(str(MC), mode="r")
    t0 = time.time()
    cut = np.asarray(z[T0:T1, Y0:Y1, X0:X1], dtype=np.float32)
    dims = cut.shape[1:]
    print(f"cutout {cut.shape} ({cut.nbytes/1e9:.2f} GB) in {time.time()-t0:.0f}s", flush=True)

    raw = json.load(open(REC))
    valid = {f.name for f in dataclasses.fields(CNMFeParams)}
    kw = {k: v for k, v in raw.items() if k in valid}
    kw.update(spatial_nrg_thr=0.999, min_pixel=1, n_jobs=-1)
    p = CNMFeParams(**kw)
    print(f"fit: sigma={p.sigma} nrg=0.999 (loose) ...", flush=True)
    t0 = time.time()
    m = CNMFe(p); m.fit_extract(cut, evaluate=True)
    print(f"  fit done in {time.time()-t0:.0f}s, K={m.A.shape[1]}", flush=True)

    print("cn on cutout...", flush=True)
    cn, pnr = correlation_pnr(cut, sigma=p.sigma, center_psf=True, n_jobs=-1, stride=2)
    C = np.asarray(m.C); YrA = np.asarray(m.YrA)

    # temporal corr(C, C+YrA) — constant across re-threshold (C held fixed)
    proj = C + YrA
    tcorr = np.array([np.corrcoef(C[k], proj[k])[0, 1]
                      if C[k].std() > 0 and proj[k].std() > 0 else np.nan
                      for k in range(C.shape[0])])

    rows = []
    A_loose = m.A.tocsc()
    montage_cells = None
    for lvl in LEVELS:
        A_l = rethreshold(A_loose, dims, p, lvl)
        npix = np.diff(A_l.indptr)
        alive = npix > 0
        rv = spatial_r_values(A_l, C, cut, dims)
        cp = cn_proxy(A_l, cn, dims)
        rec = dict(nrg=lvl, K_alive=int(alive.sum()),
                   npix_median=float(np.median(npix[alive])) if alive.any() else 0.0,
                   rvalue_median=float(np.nanmedian(rv[alive])) if alive.any() else np.nan,
                   cnproxy_median=float(np.nanmedian(cp[alive])) if alive.any() else np.nan,
                   tcorr_median=float(np.nanmedian(tcorr)))
        rows.append(rec)
        print(f"  nrg={lvl}: K={rec['K_alive']} npix={rec['npix_median']:.0f} "
              f"r={rec['rvalue_median']:.3f} cn={rec['cnproxy_median']:.3f}", flush=True)
        # pick montage cells once (largest footprints at the loosest level = the 'too big' ones)
        if montage_cells is None:
            order = np.argsort(-npix)
            montage_cells = [int(k) for k in order[:8]]
        # montage at a few representative levels
        if lvl in (0.99, 0.90, 0.80):
            montage(A_l, C, YrA, cn, dims, montage_cells, rv, cp,
                    f"nrg={lvl}  (same 8 cells)", OUT / f"montage_nrg{lvl}.png")
            print(f"  wrote montage_nrg{lvl}.png", flush=True)

    json.dump(rows, open(OUT / "curve.json", "w"), indent=2)

    # curves
    nrg = [r["nrg"] for r in rows]
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
    ax[0].plot(nrg, [r["rvalue_median"] for r in rows], "o-"); ax[0].set_title("median true r_value"); ax[0].set_xlabel("nrg_thr")
    ax[1].plot(nrg, [r["cnproxy_median"] for r in rows], "o-", color="tab:green"); ax[1].set_title("median cn-proxy"); ax[1].set_xlabel("nrg_thr")
    ax[2].plot(nrg, [r["npix_median"] for r in rows], "o-", color="tab:red"); ax[2].set_title("npix_median"); ax[2].set_xlabel("nrg_thr")
    ax[3].plot(nrg, [r["K_alive"] for r in rows], "o-", color="0.4"); ax[3].set_title("K alive"); ax[3].set_xlabel("nrg_thr")
    for a in ax:
        a.invert_xaxis(); a.grid(alpha=0.3)
    fig.suptitle(f"r_value / cn-proxy / size vs nrg_thr (winner cutout, sigma={p.sigma})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {OUT}/curves.png + montages + curve.json", flush=True)


if __name__ == "__main__":
    main()
