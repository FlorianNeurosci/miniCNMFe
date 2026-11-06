"""Check minicnmfe's stages against CaImAn's, on a simulated or real movie.

Three reports:

* same inputs: both implementations get identical inputs, stage by stage;
* stage swaps: our loop runs end to end with CaImAn doing one or all stages;
* correlation by distance: mean pairwise r(C+YrA) against centroid distance. This
  needs no ground truth, and a ring that reads neurons shows up as a negative dip.

Run from the repo root, in the ``claude_cnmfe`` env:

    python -m validation.caiman_swap.run_swap --sim-seed 1
    python -m validation.caiman_swap.run_swap --zarr /path/mc.zarr --n-frames 5000 --sigma 3.1
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tests")]

import numpy as np                                                    # noqa: E402
import scipy.sparse as sp                                             # noqa: E402
from scipy.spatial.distance import cdist                              # noqa: E402

from validation.caiman_swap.adapters import CaimanStages              # noqa: E402 (sets env)
import minicnmfe.pipeline as P                                        # noqa: E402
from minicnmfe.pipeline import CNMFe, CNMFeParams                     # noqa: E402

BINS = ((6, 9), (9, 12), (12, 15), (15, 18), (18, 24), (24, 32), (48, 400))
SWAPS = (("compute_W",), ("update_temporal",), ("update_spatial",), ("merge_components",),
         ("greedy_corr_pnr",),
         ("compute_W", "update_temporal", "update_spatial", "merge_components"))


def aligned_ring_factor(sigma):
    """A ``ring_size_factor`` that puts both rings on the same pixels.

    Ours truncates the radius and CaImAn rounds it, so a radius like 19.5 lands on
    different 1-px rings (19 vs 20) and the ``W`` supports do not overlap at all.
    """
    return (round(1.5 * (4 * sigma + 1)) + 0.25) / (4 * sigma + 1)


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (np.sqrt((a @ a) * (b @ b)) + 1e-12))


def _fit(movie, params, patches=None):
    model = CNMFe(params)
    with contextlib.ExitStack() as stack:
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        stack.enter_context(warnings.catch_warnings())
        warnings.simplefilter("ignore")
        if patches:
            stack.enter_context(mock.patch.multiple(P, **patches))
        model.fit_extract(movie, evaluate=False)
    return model


def same_input_report(movie, params, stages):
    """Hand both implementations the inputs our loop produced; compare the outputs.

    To compare the ring solves exactly set ``bg_tsub=1`` (at ``tsub > 1`` ours strides
    and CaImAn block-averages, so their ``W`` differ by construction) and use
    :func:`aligned_ring_factor`.
    """
    cap = {}
    o_cw, o_sp, o_tp = P.compute_W, P.update_spatial, P.update_temporal

    def cw(Y, A, C, dims, radius, *a, **k):
        out = o_cw(Y, A, C, dims, radius, *a, **k)
        cap.setdefault("W", (Y, A.copy(), np.array(C), radius, k.get("tsub", 1), out))
        return out

    def spatial(Y_bg, C, A, sn, dims, *a, **k):
        out = o_sp(Y_bg, C, A, sn, dims, *a, **k)
        cap["S"] = (Y_bg, np.array(C), A.copy(), np.array(sn), out)
        return out

    def temporal(Y_bg, A, C, sn, ar, n, **k):
        out = o_tp(Y_bg, A, C, sn, ar, n, **k)
        cap["T"] = (Y_bg, A.copy(), np.array(C), ar, n, dict(k), out)
        return out

    _fit(movie, params, dict(compute_W=cw, update_spatial=spatial, update_temporal=temporal))

    Y, A, C, radius, tsub, (W_o, b0_o) = cap["W"]
    W_c, b0_c = stages.compute_W(Y, A, C, stages.dims, radius, tsub=tsub)
    dW = W_o - W_c
    support = len(set(zip(*W_o.nonzero())) & set(zip(*W_c.nonzero())))
    union = len(set(zip(*W_o.nonzero())) | set(zip(*W_c.nonzero())))
    report = {"W_rel_l2": float(np.sqrt(dW.multiply(dW).sum() / W_c.multiply(W_c).sum())),
              "W_support_jaccard": support / union,
              "b0_rel_diff": float(np.abs(b0_o - b0_c).max() / np.abs(b0_o).max())}

    Y_bg, C_s, A_s, sn, A_out = cap["S"]
    a, c = A_out.toarray(), stages.update_spatial(Y_bg, C_s, A_s, sn, stages.dims).toarray()
    r = [pearson(a[:, k], c[:, k]) for k in range(a.shape[1]) if a[:, k].any() and c[:, k].any()]
    report.update(spatial_r_median=float(np.median(r)), spatial_r_min=float(np.min(r)),
                  spatial_npix=(int(np.median((a > 0).sum(0))), int(np.median((c > 0).sum(0)))))

    Y_bg, A_t, C_t, ar, n, kw, (C_out, *_) = cap["T"]
    C_c = stages.update_temporal(Y_bg, A_t, C_t, None, ar, n, **kw)[0]
    r = [pearson(C_out[k], C_c[k]) for k in range(C_out.shape[0])]
    report.update(temporal_r_median=float(np.median(r)), temporal_r_min=float(np.min(r)))
    return report


def centroids(A, dims):
    img = np.abs(A).reshape(*dims, -1)
    yy, xx = np.mgrid[:dims[0], :dims[1]]
    s = img.sum((0, 1)) + 1e-12
    return np.c_[(yy[..., None] * img).sum((0, 1)) / s, (xx[..., None] * img).sum((0, 1)) / s]


def distance_curve(traces, cents, bins=BINS):
    """Mean pairwise trace correlation per centroid-distance bin."""
    iu = np.triu_indices(len(cents), 1)
    r, d = np.corrcoef(traces)[iu], cdist(cents, cents)[iu]
    return [float(r[(d >= lo) & (d < hi)].mean()) if ((d >= lo) & (d < hi)).any() else np.nan
            for lo, hi in bins]


def truth_scores(model, a_true, c_true):
    """Mean spatial r, r(C), r(C+YrA) over reciprocal-best matches to ground truth."""
    z = lambda X: (X - X.mean(0)) / (np.linalg.norm(X - X.mean(0), axis=0) + 1e-12)
    A, C = model.A.toarray(), np.asarray(model.C)
    Y = C + np.asarray(model.YrA)
    R = z(A).T @ z(a_true)
    best_t, best_k = R.argmax(1), R.argmax(0)
    M = [(k, best_t[k]) for k in range(A.shape[1]) if best_k[best_t[k]] == k and R[k, best_t[k]] > 0.5]
    return (len(M), float(np.mean([R[k, t] for k, t in M])),
            float(np.mean([pearson(C[k], c_true[t]) for k, t in M])),
            float(np.mean([pearson(Y[k], c_true[t]) for k, t in M])))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--zarr", help="motion-corrected (T, H, W) zarr")
    ap.add_argument("--n-frames", type=int, default=5000)
    ap.add_argument("--sim-seed", type=int, default=1)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args()

    truth = None
    if args.zarr:
        from minicnmfe.io import open_zarr
        movie = np.asarray(open_zarr(args.zarr)[:args.n_frames], dtype=np.float32)
    else:
        from miniscope_simulator import make_miniscope_movie
        sim = make_miniscope_movie(n_neurons=60, dims=(128, 128), T=2000, seed=args.sim_seed)
        movie = np.asarray(sim["movie"], dtype=np.float32)
        truth = np.asarray(sim["A_true"], float), np.asarray(sim["C_true"], float)
    dims = movie.shape[1:]

    # min_corr/min_pnr only reach CaImAn's seeding; our default seed_method ignores them.
    params = CNMFeParams(sigma=args.sigma, n_jobs=args.n_jobs, init_patches=False,
                         min_corr=0.6, min_pnr=5.0,
                         spatial_thr_method="max", frame_rate_hz=20.0, decay_time_ms=180.0,
                         g_prior_weight=0.6, auto_eval_snr_amp_thr=0.0)
    stages = CaimanStages(dims, sigma=args.sigma, frame_rate_hz=20.0, decay_time_ms=180.0,
                          merge_thr=params.merge_thr_corr)

    print("== same inputs, ours vs CaImAn ==")
    exact = replace(params, bg_tsub=1, ring_size_factor=aligned_ring_factor(args.sigma))
    for key, value in same_input_report(movie, exact, stages).items():
        print(f"  {key:20s} {value}")

    print("\n== our loop, CaImAn stages swapped in ==")
    print(f"  {'arm':62s}   K  npix  time" + ("  matched spat_r r(C) r(C+YrA)" if truth else ""))
    for names in ((),) + SWAPS:
        label = " + ".join(names) or "ours"
        t = time.time()
        try:
            m = _fit(movie, params, stages.patches(*names) if names else None)
        except Exception as err:                   # a failed arm must not hide the others
            print(f"  {label:62s}   FAILED: {type(err).__name__}: {err}")
            continue
        A = m.A.toarray()
        line = f"  {label:62s} {A.shape[1]:3d} {int(np.median((A > 0).sum(0))):5d} {time.time() - t:4.0f}s"
        if truth:
            n, sr, rc, ry = truth_scores(m, *truth)
            line += f"  {n:3d}/{truth[1].shape[0]} {sr:.3f}  {rc:.3f} {ry:.3f}"
        print(line, flush=True)

    print("\n== mean pairwise r(C+YrA) by distance, px ==")
    print(f"  {'':24s}" + "".join(f"{f'{a}-{b}':>8s}" for a, b in BINS))
    old = 1.5 * (2 * args.sigma + 1) / (4 * args.sigma + 1)
    for label, p in (("published model", params),
                     ("legacy", replace(params, ring_apply_to_raw=True, ring_fit_once=True,
                                        ring_size_factor=old))):
        m = _fit(movie, p)
        trace = np.asarray(m.C) + np.asarray(m.YrA)
        print(f"  {label:24s}" + "".join(f"{x:+8.3f}" for x in
                                         distance_curve(trace, centroids(m.A.toarray(), dims))))


if __name__ == "__main__":
    main()
