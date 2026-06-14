"""Step 1 of the background-leak diagnosis (plan: check-this-out...steady-pnueli).

Question: the winner extraction's deconvolved traces share an 81%-variance PC1
across spatially-distributed cells. Is this residual 1p background leaking into
the neuron traces, and which knobs move it?

For each setting in a small matrix we fit the SAME PICAST cutout and report, on
the ACCEPTED traces:
  - PC1 variance-explained of C
  - median pairwise |r| of C (raw) and after removing PC1
  - corr(PC1, model.f)              -> is the shared mode the rank-1 bg shape?
  - corr(PC1, global spatial mean)  -> is it the global background fluctuation?
  - K, accepted count

Hypotheses being tested:
  - init_stride=1 (filtered init trace) << init_stride=2 (raw projection init)
  - n_iter_main up -> PC1% down (fixed-point / too-few-iters)
  - global_bg_rank=0 vs 1

Run (live output, env python directly):
  /home/fs539/miniforge3/envs/claude_cnmfe/bin/python live_runs/bg_leak_diag.py
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
from tuning.metrics import _per_cell_corr  # noqa: E402

MC = Path("/media/server/archive/projects/2023_intercontext/PICAST/data/1_preprocessed/"
          "20260505_m0010800_wt_1557/miniscope_video/minicnmfe_mc_mcid_0/mc.zarr")
REC = Path("/media/server/archive/projects/2023_intercontext/PICAST/data/2_processed/"
           "20260505_m0010800_wt_1557/miniscope_video/minicnmfe_mc_mcid_0/"
           "minicnmfe_tuning_taskid_0__20260614_194430_137177_0dbb4d/recommended_params.json")
Y0, Y1, X0, X1, T0, T1 = 67, 323, 246, 502, 14876, 17876
OUT = ROOT / "live_runs" / "bg_leak_diag_out"

# winner thresholds from winner_diag.py (sigma=3 rerun)
WINNER = dict(sigma=3.0, min_corr=0.822, min_pnr=4.859, spatial_nrg_thr=0.95,
              min_pixel=1, n_jobs=-1)

# Settings matrix: (label, overrides-on-top-of-winner)
MATRIX = [
    ("baseline (stride2,iter2,rank1)", {}),
    ("init_stride=1",                  dict(init_stride=1)),
    ("n_iter_main=4",                  dict(n_iter_main=4)),
    ("n_iter_main=6",                  dict(n_iter_main=6)),
    ("global_bg_rank=0",              dict(global_bg_rank=0)),
    ("stride1 + iter4",               dict(init_stride=1, n_iter_main=4)),
]


def base_kwargs():
    raw = json.load(open(REC))
    valid = {f.name for f in dataclasses.fields(CNMFeParams)}
    kw = {k: v for k, v in raw.items() if k in valid}
    kw.update(WINNER)
    return kw


def pc1_stats(C, acc):
    """Return dict of PC1% / pairwise-|r| / pc1 vector on accepted C."""
    accC = C[acc]
    if accC.shape[0] < 2:
        return None, None
    Xc = accC - accC.mean(axis=1, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pc1 = Vt[0]
    var1 = float(S[0] ** 2 / (S ** 2).sum())
    Xc_resid = Xc - np.outer(Xc @ pc1, pc1)

    def med_pair(M):
        cc = np.corrcoef(M)
        iu = np.triu_indices(M.shape[0], 1)
        v = np.abs(cc[iu]); v = v[np.isfinite(v)]
        return float(np.median(v)) if v.size else float("nan")

    return {
        "pc1_var": var1,
        "med_pair_raw": med_pair(accC),
        "med_pair_resid": med_pair(Xc_resid),
    }, pc1


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = zarr.open(str(MC), mode="r")
    cut = np.asarray(z[T0:T1, Y0:Y1, X0:X1], dtype=np.float32)
    H, W = cut.shape[1:]
    print(f"cutout {cut.shape}", flush=True)
    # global spatial-mean trace (the candidate background fluctuation)
    gmean = cut.reshape(cut.shape[0], -1).mean(axis=1).astype(np.float32)

    results = []
    for label, ov in MATRIX:
        kw = base_kwargs()
        kw.update(ov)
        p = CNMFeParams(**kw)
        print(f"\n===== {label} =====", flush=True)
        print(f"  init_stride={p.init_stride} n_iter_main={p.n_iter_main} "
              f"global_bg_rank={p.global_bg_rank}", flush=True)
        t0 = time.time()
        m = CNMFe(p); m.fit_extract(cut, evaluate=True)
        dt = time.time() - t0
        K = m.A.shape[1]
        mask = m.accepted_mask if m.accepted_mask is not None else np.ones(K, bool)
        acc = np.flatnonzero(mask)
        C = np.asarray(m.C)
        stats, pc1 = pc1_stats(C, acc)
        rec = dict(label=label, K=int(K), accepted=int(len(acc)),
                   init_stride=int(p.init_stride), n_iter_main=int(p.n_iter_main),
                   global_bg_rank=int(p.global_bg_rank), fit_s=round(dt, 1))
        if stats is None:
            rec.update(pc1_var=None, med_pair_raw=None, med_pair_resid=None,
                       pc1_corr_f=None, pc1_corr_gmean=None)
            print("  <2 accepted; skipping PC1", flush=True)
        else:
            rec.update({k: round(v, 4) for k, v in stats.items()})
            # corr(PC1, model.f) and corr(PC1, global mean)
            f = np.asarray(m.f).ravel() if getattr(m, "f", None) is not None else None
            cf = (abs(np.corrcoef(pc1, f)[0, 1])
                  if f is not None and f.std() > 0 else None)
            cg = abs(np.corrcoef(pc1, gmean)[0, 1])
            rec["pc1_corr_f"] = None if cf is None else round(float(cf), 4)
            rec["pc1_corr_gmean"] = round(float(cg), 4)
            print(f"  K={K} acc={len(acc)} PC1var={stats['pc1_var']:.1%} "
                  f"medR_raw={stats['med_pair_raw']:.3f} "
                  f"medR_resid={stats['med_pair_resid']:.3f} "
                  f"corr(PC1,f)={rec['pc1_corr_f']} "
                  f"corr(PC1,gmean)={rec['pc1_corr_gmean']} ({dt:.0f}s)", flush=True)
        results.append(rec)
        # incremental save
        json.dump(results, open(OUT / "bg_leak_summary.json", "w"), indent=2)

    # ---- figure: PC1% + median pairwise |r| per setting ----
    labels = [r["label"] for r in results]
    pc1v = [(r["pc1_var"] or 0) * 100 for r in results]
    mraw = [(r["med_pair_raw"] or 0) for r in results]
    cf = [(r["pc1_corr_f"] or 0) for r in results]
    cg = [(r["pc1_corr_gmean"] or 0) for r in results]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))
    ax[0].bar(x, pc1v, color="tab:red", alpha=0.8)
    ax[0].set_ylabel("PC1 variance explained (%)")
    ax[0].set_title("Shared-mode dominance per setting (lower = less leak)")
    for xi, v, kacc in zip(x, pc1v, [r["accepted"] for r in results]):
        ax[0].text(xi, v + 1, f"{v:.0f}%\nacc={kacc}", ha="center", fontsize=8)
    w = 0.27
    ax[1].bar(x - w, mraw, w, label="median |r| raw C", color="tab:red", alpha=0.8)
    ax[1].bar(x, cf, w, label="|corr(PC1, f)|", color="tab:blue", alpha=0.8)
    ax[1].bar(x + w, cg, w, label="|corr(PC1, global mean)|", color="tab:green", alpha=0.8)
    ax[1].set_ylim(0, 1); ax[1].legend(fontsize=9)
    ax[1].set_title("Is PC1 the background? (corr with f / global mean)")
    for a in ax:
        a.set_xticks(x); a.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "bg_leak_summary.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {OUT}/bg_leak_summary.png + .json", flush=True)


if __name__ == "__main__":
    main()
