"""Figure generation + report folder writer for the tuner.

The **only** matplotlib-importing module. Each ``fig_*`` function takes an
evidence dict (from ``tuning.heuristics``) or sweep rows / a fitted model and
either saves a PNG (``out_path`` given) or returns the figure (for inline
display in ``live_runs/tune.ipynb``). ``write_report`` consumes the ``result``
dict assembled by ``tuning.tuner`` and writes ``recommended_params.json``,
``downsample.json``, ``report.md`` and every figure into the run folder.
"""

from __future__ import annotations

import json
from dataclasses import fields as _fields
from pathlib import Path

import numpy as np

# Shared prose so the markdown report, the HTML report and the docs stay in
# sync from one source (imported by ``tuning.report_html`` and referenced by
# ``wiki/parameter-tuning.md``).
METRICS_BLURB = (
    "These are **ground-truth-free proxies**, not validation. `cprojcorr_median` "
    "(median `corr(C, C+YrA)`) is the primary purity signal — in a dense FOV it "
    "*falls* as the cell count rises (YrA cross-talk), so there is a density↔purity "
    "sweet spot rather than a 'more cells = better' rule. `multipeak_frac` is the "
    "fraction of footprints carrying ≥2 distinct soma-scale peaks — the signature of "
    "`sigma` too large fusing neighbours; `npix_oversize` (diagnostic) is the median "
    "footprint area over one expected soma. `composite_score` ranks candidates as "
    "`cprojcorr_median + 0.5·accepted_frac − 0.25·(npix_iqr/npix_median) − 0.5·multipeak_frac`; "
    "re-rank from the table with your own weights if needed. See the roadmap "
    "(C1/C2) for true validation."
)

# Single source of truth for the by-eye troubleshooting table (was duplicated in
# the tune-session skill, the wiki and tuning_picast/LEARNINGS.md). The HTML
# report embeds this; the wiki links to it.
SYMPTOM_CAUSE_KNOB = """\
| Symptom (what you see) | Most likely cause | Knob to try first | Inspect in |
|---|---|---|---|
| **Two or more bright peaks** in many footprints | `sigma` too large → seed-suppression disk wipes neighbours → LASSO merges them in space | `sigma` 5→3 (single biggest lever), also reduce `min_pnr` to let more seeds in | footprint_grid |
| **Soft amoeba blobs** filling crop window | Halo dominated; `spatial_max_thr` too low, residual background drift | `spatial_max_thr` 0.1→0.25; `global_bg_rank` 0→1; verify detrend ran | footprint_grid, eccentricity |
| **Sprawling footprints + `sigma`/`ssub`/`min_pixel` ~2× a comparable session** | Hazy / out-of-focus FOV: `blob_log` measured the neuron radius off broad background, not the cells | Auto-fixed by the `highpass_sigma` pre-filter in the radius estimate; check `fig_mc_gsig` (histogram should sit at neuron scale), raise `highpass_sigma` if it persists | mc_gsig, footprint_grid |
| **Donut / hollow ring** footprints | Ring background ate the soma | `ring_constrain_sum=True`; `ring_size_factor` 1.5→1.2 | footprint_grid |
| **Crescent / arc** footprints | Asymmetric ring subtraction; usually paired with multi-peak | Fix `sigma` first; then `spatial_circular_max_dist_factor` 1.5→1.0 | footprint_grid |
| **Streaky / elongated** (ecc > 0.8 common) | Vasculature contamination, or two close neurons unsplit | `spatial_circular_max_dist_factor` 1.5→1.0; for vasculature draw an ROI mask | footprint_grid, eccentricity |
| **Many tiny / fragmented** dots | `min_pixel` too low; or `spatial_close_radius=0` | `min_pixel` 3→25; ensure `spatial_close_radius=1` | footprint_grid, eccentricity |
| **Hard circular outer boundary** (clipped circle) | `circular_constraint` cutoff too tight | raise `spatial_circular_max_dist_factor` (1.2→1.5→2.0) | footprint_grid |
| **Heavy spatial + trace overlap** between pairs that didn't merge | `merge_thr_corr` too strict | `merge_thr_corr` 0.85→0.75 | jaccard_merge |
| **Ghost components** (dots on dark areas; bimodal peak histogram) | `min_corr` / `min_pnr` too lax for this recording | raise `min_pnr` (10→15); auto-eval flags ghosts by SNR | eccentricity, centroid_drift |
| **Bright CORR·PNR blobs with no footprint** (low `blob_recall`) | thresholds too tight, or init under-seeding a long movie | lower `min_corr`/`min_pnr`; pin `init_stride` to 1–2 on long recordings | blob_coverage |
| **Footprints sitting on no CORR·PNR blob** (low `footprint_precision`) | thresholds too lax → ghosts, or `sigma` mismatch | raise `min_pnr`/`min_corr`, or `auto_eval_snr_amp_thr` to cut ghosts | blob_coverage, snr_eval |
| **`C+YrA` far from `C` / traces correlated across all cells** (low `cprojcorr_median`, high `trace_corr_median`) | over-split or unremoved global drift in a dense FOV (YrA cross-talk) | `global_bg_rank=1`; gentler merge; `n_iter_main≥2` + tighter footprints (`spatial_max_thr`↑, `spatial_circular_max_dist_factor`↓) | traces, jaccard_merge |
| **All traces fire in lockstep** | Global drift not removed (bleach, LED warm-up, focus step) | per-pixel detrend; or `global_bg_rank=1` | traces, mean_proj_and_activity |
| **Monotonic ramp at the start** of every trace | LED warm-up | crop first N frames via `temporal_crop`; or detrend | traces |
| **Saturated / square-wave `C`** with normal `C + YrA` | OASIS over-smoothing; `g` pinned at the prior target | `g_prior_weight` 0.7→0.3; or unset `decay_time_ms` | traces, decay |
| **Long tail toward g → 1** | Yule-Walker contaminated by drift | `ar_detrend_order=2`; or detrend the movie | decay |
| **CORR map uniformly bright everywhere** | Unremoved global signal | detrend the movie first | centroid_drift |
| **Footprints clipped at crop border** | Real cells extend past the spatial crop | expand `spatial_crop` to leave ≳ `max_shift` px margin | footprints_on_corr |
"""


def _finish(fig, out_path):
    import matplotlib.pyplot as plt

    if out_path is not None:
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


# --------------------------------------------------------------------------
# Stage 1 — motion correction
# --------------------------------------------------------------------------


def fig_mc_gsig(ev, out_path=None):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(ev["std_img"], cmap="magma")
    for y, x, s in ev.get("blobs_top", []):
        axes[0].add_patch(plt.Circle((x, y), s * np.sqrt(2), color="cyan",
                                     fill=False, lw=0.8))
    axes[0].set_title(f"temporal std + top {len(ev.get('blobs_top', []))} blobs")
    axes[0].axis("off")
    sig = ev.get("sigmas_top", np.empty(0))
    if len(sig):
        axes[1].hist(sig, bins=12, color="C0", edgecolor="k")
        axes[1].axvline(ev["median_sigma"], color="C3", ls="--", lw=1.5,
                        label=f"median = {ev['median_sigma']:.1f}")
        axes[1].legend()
    axes[1].set_xlabel("blob sigma (px)")
    axes[1].set_ylabel("count")
    axes[1].set_title("fitted neuron-radius distribution")
    fig.tight_layout()
    return _finish(fig, out_path)


def fig_max_shift(ev, out_path=None):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, vals, p99, lab in [(axes[0], ev["abs_dy"], ev["p99_dy"], "dy"),
                               (axes[1], ev["abs_dx"], ev["p99_dx"], "dx")]:
        ax.hist(vals, bins=30, color="C0", edgecolor="k")
        ax.axvline(p99, color="C3", ls="--", lw=1.5, label=f"99th pct = {p99:.2f}")
        ax.set_xlabel(f"|{lab}| (px)")
        ax.set_ylabel("frame count")
        ax.set_title(f"absolute {lab} shifts (max={vals.max():.2f})")
        ax.legend()
    fig.tight_layout()
    return _finish(fig, out_path)


def fig_downsample(ev, out_path=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.axis("off")
    lines = [
        f"native neuron FWHM ≈ {ev['fwhm_native']:.2f} px",
        f"native dt = {ev['dt_ms']:.1f} ms   decay = {ev['decay_time_ms']:g} ms",
        "",
        "ssub   FWHM_ds(px)   ok?",
    ]
    for ssub, fwhm_ds, ok in ev["ssub_rows"]:
        lines.append(f"  {ssub}      {fwhm_ds:7.2f}     {'OK' if ok else 'too small'}")
    lines += ["", "tsub   dt_ds(ms)   dt/decay   ok?"]
    for tsub, dt_ds, ratio, ok in ev["tsub_rows"]:
        lines.append(f"  {tsub}    {dt_ds:8.1f}    {ratio:6.2f}    "
                     f"{'OK' if ok else 'under-samples'}")
    ax.text(0.0, 0.98, "\n".join(lines), family="monospace", va="top",
            transform=ax.transAxes, fontsize=11)
    ax.set_title("downsample candidates (ssub: FWHM≥4 px,  tsub: dt≤decay/2)")
    fig.tight_layout()
    return _finish(fig, out_path)


# --------------------------------------------------------------------------
# Stage 3 — initialisation
# --------------------------------------------------------------------------


def fig_corr_pnr_sigma(ev, out_path=None):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(ev["cn"], cmap="viridis"); axes[0].set_title("CORR"); axes[0].axis("off")
    axes[1].imshow(ev["pnr"], cmap="magma"); axes[1].set_title("PNR"); axes[1].axis("off")
    axes[2].imshow(ev["product"], cmap="magma")
    blobs, order = ev.get("blobs", np.empty((0, 3))), ev.get("order", np.empty(0, int))
    for y, x, s in (blobs[order] if len(order) else []):
        axes[2].add_patch(plt.Circle((x, y), s * np.sqrt(2), color="cyan",
                                     fill=False, lw=0.6))
    axes[2].set_title(f"CORR·PNR + top blobs (median σ={ev['sigma_refit']:.2f})")
    axes[2].axis("off")
    fig.tight_layout()
    return _finish(fig, out_path)


def fig_corr_pnr_separation(ev, out_path=None):
    """min_corr/min_pnr by neuron-vs-background separation (mirrors the diagnosis).

    Top row: histograms of the CORR / PNR values **at the detected neuron blob
    centres** (signal) vs the **background** pixels (noise), with the chosen
    threshold marked — it sits where the two populations separate best (max
    Youden's J). Bottom row: the CORR·PNR product with the detected blobs circled
    (the cells the eye picks out), and the product masked by the chosen thresholds
    (neurons retained, background removed).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    cn, pnr = ev["cn"], ev["pnr"]
    mc, mp = ev["min_corr"], ev["min_pnr"]
    sigma = float(ev["sigma"])
    rc = ev.get("blob_rc", np.zeros((0, 2), int))
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    def _hist(a, neuron, bg, thr, xlabel, title):
        if len(bg):
            a.hist(bg, bins=40, density=True, color="0.6", alpha=0.7, label="background")
        if len(neuron):
            a.hist(neuron, bins=40, density=True, color="C2", alpha=0.7, label="neuron centres")
        a.axvline(thr, c="k", ls="--", lw=1.5, label=f"thr={thr:.2f}")
        a.set_xlabel(xlabel); a.set_ylabel("density"); a.set_title(title); a.legend(fontsize=8)

    _hist(ax[0, 0], ev.get("corr_neuron", []), ev.get("corr_bg", []), mc,
          "CORR", f"CORR separation  ->  min_corr={mc:.2f}")
    _hist(ax[0, 1], ev.get("pnr_neuron", []), ev.get("pnr_bg", []), mp,
          "PNR", f"PNR separation  ->  min_pnr={mp:.1f}")

    product = cn * pnr
    pvmax = float(np.percentile(product, 99.5)) if np.isfinite(product).any() else None
    ax[1, 0].imshow(product, cmap="magma", vmax=pvmax)
    for y, x in rc:
        ax[1, 0].add_patch(plt.Circle((x, y), 1.5 * sigma, color="cyan", fill=False, lw=0.6))
    ax[1, 0].set_title(f"CORR·PNR + {len(rc)} detected neurons"); ax[1, 0].axis("off")

    kept = product * ((cn >= mc) & (pnr >= mp))
    ax[1, 1].imshow(kept, cmap="magma", vmax=pvmax)
    ax[1, 1].set_title("CORR·PNR kept by (corr≥min_corr & pnr≥min_pnr)"); ax[1, 1].axis("off")
    fig.suptitle(f"min_corr / min_pnr from neuron-vs-background separation (σ={sigma:.1f})",
                 y=1.0, fontsize=10)
    fig.tight_layout()
    return _finish(fig, out_path)


def fig_min_pixel(ev, out_path=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    pc = ev.get("pixel_counts", np.empty(0))
    if len(pc):
        ax.hist(pc, bins=20, color="C0", edgecolor="k")
        ax.axvline(ev["p25"], color="C3", ls="--", lw=1.5, label=f"25th pct = {ev['p25']}")
        ax.axvline(ev["p50"], color="C2", ls=":", lw=1.5, label=f"median = {ev['p50']}")
        ax.legend()
    ax.set_xlabel("footprint pixel count (>5% peak)")
    ax.set_ylabel("component count")
    ax.set_title(f"footprint area distribution (K={ev.get('K', 0)})")
    fig.tight_layout()
    return _finish(fig, out_path)


# --------------------------------------------------------------------------
# Stage 4 — temporal / merge / eval
# --------------------------------------------------------------------------


def fig_decay(ev, out_path=None):
    import matplotlib.pyplot as plt

    g_yw, tau_ms, tau_med = ev["g_yw"], ev["tau_ms"], ev["tau_med"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(g_yw, bins=30, color="C0", edgecolor="k")
    axes[0].set_xlabel("AR(1) g (Yule-Walker, no prior)"); axes[0].set_ylabel("components")
    axes[0].set_title(f"g distribution (median={np.median(g_yw):.3f})")
    hi = np.percentile(tau_ms, 99) if len(tau_ms) else 1.0
    axes[1].hist(tau_ms[tau_ms < hi], bins=30, color="C1", edgecolor="k")
    axes[1].axvline(tau_med, color="C3", ls="--", lw=1.5, label=f"median = {tau_med:.0f} ms")
    axes[1].set_xlabel("tau (ms)"); axes[1].set_ylabel("components")
    axes[1].set_title("per-component decay (clipped at 99th pct)"); axes[1].legend()
    fig.tight_layout()
    return _finish(fig, out_path)


def fig_g_prior(ev, out_path=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.hist(ev["g_yw"], bins=30, color="C0", edgecolor="k", alpha=0.8, label="per-component YW g")
    ax.axvline(ev["g_target"], color="C3", ls="--", lw=1.5,
               label=f"g_target = {ev['g_target']:.3f}")
    ax.set_xlabel("g"); ax.set_ylabel("components")
    ax.set_title(f"YW g vs target (median |rel dev| = {ev['spread']*100:.1f}%)")
    ax.legend()
    fig.tight_layout()
    return _finish(fig, out_path)


def fig_merge_corr(ev, out_path=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    rs = ev.get("rs", np.empty(0))
    if len(rs):
        ax.hist(rs, bins=80, color="C0", edgecolor="k")
        ax.set_yscale("log")
        ax.axvline(ev["p99"], color="C2", ls=":", lw=1.0, label=f"99th pct = {ev['p99']:.2f}")
        ax.legend()
    ax.set_xlabel("Pearson(C_raw[i], C_raw[j])  (i<j)")
    ax.set_ylabel("pair count (log)")
    ax.set_title(f"pairwise component correlations ({len(rs)} pairs)")
    fig.tight_layout()
    return _finish(fig, out_path)


def fig_snr_eval(ev, model, out_path=None):
    import matplotlib.pyplot as plt

    snr = ev.get("snr", np.empty(0))
    cur = ev["current_thr"]
    fig = plt.figure(figsize=(14, 7))
    ax_h = plt.subplot2grid((3, 6), (0, 0), colspan=6)
    if len(snr):
        hi = np.percentile(snr, 99)
        ax_h.hist(snr[snr < hi], bins=60, color="C0", edgecolor="k")
        ax_h.set_yscale("log")
        ax_h.axvline(cur, color="k", ls=":", lw=1.0, label=f"current = {cur:.1f}")
        ax_h.axvline(ev["snr_suggested"], color="C3", ls="--", lw=1.5,
                     label=f"suggested = {ev['snr_suggested']:.1f}")
        ax_h.legend()
    ax_h.set_xlabel("mean-amplitude SNR")
    ax_h.set_title(f"SNR distribution ({len(snr)} components)")

    if len(snr) and model.dims is not None:
        K = len(snr)
        H, W = model.dims
        A_dense = np.asarray(model.A.todense()).reshape(H, W, K)
        above = np.where(snr >= cur)[0]
        below = np.where(snr < cur)[0]
        above_sel = above[np.argsort(snr[above])[:6]]
        below_sel = below[np.argsort(-snr[below])[:6]]

        def _row(r, indices, prefix):
            axes_row = [plt.subplot2grid((3, 6), (r, c)) for c in range(6)]
            for ax, k in zip(axes_row, indices):
                ax.imshow(A_dense[..., k], cmap="magma")
                ax.set_title(f"{prefix} k={k}\nsnr={snr[k]:.2f}", fontsize=8)
                ax.axis("off")
            for ax in axes_row[len(indices):]:
                ax.axis("off")
        _row(1, above_sel, "ABOVE")
        _row(2, below_sel, "BELOW")
    fig.tight_layout()
    return _finish(fig, out_path)


# --------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------


def fig_sweep_scatter(rows, out_path=None):
    import matplotlib.pyplot as plt

    ok = [r for r in rows if r.get("error") is None and r.get("K", 0) > 0]
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    if ok:
        K = np.array([r["K"] for r in ok])
        cc = np.array([r.get("cprojcorr_median", np.nan) for r in ok])
        acc = np.array([r.get("accepted_frac", 0.0) for r in ok])
        score = np.array([r["score"] for r in ok])
        sc = ax.scatter(K, cc, s=40 + 200 * acc, c=score, cmap="viridis",
                        edgecolor="k", zorder=3)
        plt.colorbar(sc, ax=ax, label="composite score")
        best = ok[int(np.argmax(score))]
        ax.scatter([best["K"]], [best.get("cprojcorr_median", np.nan)],
                   marker="*", s=420, color="gold", edgecolor="k", zorder=4,
                   label=f"best (idx {best['idx']})")
        ax.legend()
    ax.set_xlabel("K (components extracted)")
    ax.set_ylabel("median corr(C, C+YrA)  — purity")
    ax.set_title("sweep: density vs purity (point size = accepted fraction)")
    fig.tight_layout()
    return _finish(fig, out_path)


def fig_sweep_footprints(model, cn, out_path=None, region_crop=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    if cn is not None:
        # σ-smoothed CORR backdrop (gray, vmin=min_corr) — the cutout-selection
        # look; cosmetic only, contours/data below are unchanged.
        cn = np.asarray(cn, dtype=np.float32)
        sigma = float(getattr(model.params, "sigma", 0.0) or 0.0)
        if sigma > 0:
            from scipy.ndimage import gaussian_filter
            cn_disp = gaussian_filter(cn, sigma=sigma)
        else:
            cn_disp = cn
        vmin = float(getattr(model.params, "min_corr", 0.0) or 0.0)
        vmax = np.nanpercentile(cn_disp, 99.5) if np.isfinite(cn_disp).any() else None
        ax.imshow(cn_disp, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title("best-candidate footprints over the σ-smoothed CORRELATION image")
    H, W = model.dims
    # The sweep extracts on a cutout (crop-local coords) but ``cn`` is the full
    # FOV correlation image. Offset the contours by the crop origin so they land
    # on the right pixels; without this they are drawn up-and-left of the cells.
    y0, x0 = 0, 0
    if region_crop is not None:
        (y0, _y1, x0, _x1), _t = region_crop
    xs = np.arange(W) + x0
    ys = np.arange(H) + y0
    K = model.A.shape[1]
    mask = getattr(model, "accepted_mask", None)
    A_dense = np.asarray(model.A.todense()).reshape(H, W, K)
    for k in range(K):
        fp = A_dense[..., k]
        if fp.max() <= 0:
            continue
        col = "lime" if (mask is None or (len(mask) == K and mask[k])) else "red"
        ax.contour(xs, ys, fp, levels=[0.3 * fp.max()], colors=col, linewidths=0.7)
    ax.axis("off")
    fig.tight_layout()
    return _finish(fig, out_path)


def _plot_trace_stack(ax, C, Cproj, order, title):
    """Stack of normalized C (blue) vs C+YrA (grey) traces for the given cells."""
    for row, k in enumerate(order):
        off = row * 1.2
        tr = Cproj[k]
        tr = (tr - tr.min()) / (tr.max() - tr.min() + 1e-8)
        cc = C[k]
        cc = (cc - cc.min()) / (cc.max() - cc.min() + 1e-8)
        ax.plot(tr + off, color="0.6", lw=0.5)
        ax.plot(cc + off, color="C0", lw=0.8)
        ax.text(-0.01 * C.shape[1], off + 0.5, f"k={k}", fontsize=7, ha="right", va="center")
    ax.set_yticks([])
    ax.set_xlabel("frame")
    ax.set_title(title)


def fig_sweep_traces(model, out_path=None, n: int = 10):
    import matplotlib.pyplot as plt

    C = np.asarray(model.C)
    Cproj = C + np.asarray(model.YrA)
    K = C.shape[0]
    n = min(n, K)

    # Restrict to auto-eval-ACCEPTED cells. Noise/ghost components are
    # uncorrelated with everything (and have poor C vs C+YrA agreement), so
    # showing them is misleading; both panels below sample only accepted cells.
    mask = getattr(model, "accepted_mask", None)
    if mask is not None and len(mask) == K and int(np.sum(mask)) >= 2:
        pool = np.flatnonzero(np.asarray(mask, dtype=bool))
    else:
        pool = np.arange(K)
    # Cap the correlation work on very dense models (full-FOV path).
    if len(pool) > 400:
        pool = pool[np.argsort(-C[pool].max(axis=1))[:400]]
    corr = np.corrcoef(C[pool]) if len(pool) > 1 else np.array([[1.0]])
    corr = np.nan_to_num(corr)
    nshow = min(n, len(pool))

    # LEFT: an unbiased RANDOM sample of accepted cells (fixed seed for
    # reproducibility) — the honest "what do typical cells look like" view.
    rng = np.random.RandomState(0)
    rand_local = np.sort(rng.choice(len(pool), size=nshow, replace=False))
    rand_order = [int(pool[c]) for c in rand_local]

    # RIGHT: greedy DIVERSE pick — seed with the highest-amplitude accepted cell,
    # then add the component LEAST correlated with those already chosen. Shows the
    # most distinct signals available (best-case diversity).
    seed = int(np.argmax(C[pool].max(axis=1)))
    chosen = [seed]
    while len(chosen) < nshow:
        maxc = np.abs(corr[chosen]).max(axis=0)
        maxc[chosen] = np.inf  # don't reselect
        chosen.append(int(np.argmin(maxc)))
    div_order = [int(pool[c]) for c in chosen]

    # THIRD: highest per-cell corr(C, C+YrA) — the best-fit / cleanest accepted
    # cells (the demixed trace explains the raw footprint projection well).
    a = C[pool] - C[pool].mean(axis=1, keepdims=True)
    b = Cproj[pool] - Cproj[pool].mean(axis=1, keepdims=True)
    cell_corr = (a * b).sum(axis=1) / (
        np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1)) + 1e-12
    )
    corr_order = [int(pool[c]) for c in np.argsort(-cell_corr)[:nshow]]

    # redundancy annotation: median |pairwise corr| among accepted cells
    if corr.shape[0] > 1:
        iu = np.triu_indices(corr.shape[0], k=1)
        pair = np.abs(corr[iu])
        pair = pair[np.isfinite(pair)]
        med = float(np.median(pair)) if pair.size else float("nan")
    else:
        med = float("nan")

    fig, axes = plt.subplots(1, 3, figsize=(28, 1.0 * nshow + 1.5), sharex=True)
    _plot_trace_stack(axes[0], C, Cproj, rand_order, "random accepted sample")
    _plot_trace_stack(axes[1], C, Cproj, div_order, "most-decorrelated accepted")
    _plot_trace_stack(axes[2], C, Cproj, corr_order,
                      "highest C vs C+YrA corr (best-fit)")
    verdict = "high → synchronized / shared signal" if (med == med and med > 0.5) \
        else "low → distinct signals"
    fig.suptitle(
        "accepted-cell traces: C (blue) vs C+YrA (grey)  —  "
        f"median pairwise trace corr = {med:.2f} ({verdict})"
    )
    fig.tight_layout()
    return _finish(fig, out_path)


# --------------------------------------------------------------------------
# Full-recording diagnostics (shared by validate.py / run_full*.py / notebook)
# --------------------------------------------------------------------------


def fig_mc_shifts(shifts, out_path=None):
    import matplotlib.pyplot as plt

    sh = np.asarray(shifts)
    fig, ax = plt.subplots(2, 1, figsize=(11, 6))
    ax[0].plot(sh[:, 0], lw=0.5, label="dy"); ax[0].plot(sh[:, 1], lw=0.5, label="dx")
    ax[0].set_xlabel("frame"); ax[0].set_ylabel("shift (px)")
    ax[0].set_title(f"MC shifts (|max| dy={np.abs(sh[:,0]).max():.1f} "
                    f"dx={np.abs(sh[:,1]).max():.1f})"); ax[0].legend()
    ax[1].hist(np.linalg.norm(sh, axis=1), bins=60, color="C0", edgecolor="k")
    ax[1].set_xlabel("|shift| (px)"); ax[1].set_ylabel("frames")
    fig.tight_layout()
    return _finish(fig, out_path)


def fig_npix_accepted(model, out_path=None):
    import matplotlib.pyplot as plt
    import scipy.sparse as sp

    A = model.A.tocsc() if sp.issparse(model.A) else sp.csc_matrix(model.A)
    npix = np.diff(A.indptr)
    mask = model.accepted_mask
    if mask is None or len(mask) != len(npix):
        mask = np.ones(len(npix), dtype=bool)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    if len(npix):
        bins = np.linspace(0, npix.max() + 1, 30)
        ax.hist(npix[mask], bins=bins, color="C2", alpha=0.7, label=f"accepted ({int(mask.sum())})")
        ax.hist(npix[~mask], bins=bins, color="C3", alpha=0.7, label=f"rejected ({int((~mask).sum())})")
        ax.axvline(model.params.min_pixel, color="k", ls="--",
                   label=f"min_pixel={model.params.min_pixel}")
        ax.legend()
    ax.set_xlabel("footprint npix (extraction grid)"); ax.set_ylabel("components")
    ax.set_title("footprint area: accepted vs rejected")
    fig.tight_layout()
    return _finish(fig, out_path)


def fig_projections(sample, cn, out_path=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    ax[0].imshow(np.asarray(sample).mean(0), cmap="gray")
    ax[0].set_title("mean projection"); ax[0].axis("off")
    ax[1].imshow(cn, cmap="viridis"); ax[1].set_title("correlation image"); ax[1].axis("off")
    fig.tight_layout()
    return _finish(fig, out_path)


# --------------------------------------------------------------------------
# Packaged component diagnostics
# --------------------------------------------------------------------------
# These were previously notebook-only cells (``live_runs/diagnostics.ipynb`` +
# ``cutout_analysis.ipynb``). Each follows the same ``out_path`` contract as the
# figures above and reuses ``minicnmfe._utils.footprint_center`` /
# ``minicnmfe.preprocess.correlation_pnr``, so they are callable from both the HTML
# report and the notebooks (``from tuning.report import fig_eccentricity``).


def _dense_components(model):
    """Return ``(A_dense (H,W,K), (H,W), K)`` from a fitted model.

    ``(None, None, 0)`` when there is nothing to show (no ``A`` / no ``dims`` /
    zero components).
    """
    import scipy.sparse as sp

    A = getattr(model, "A", None)
    if A is None or getattr(model, "dims", None) is None or A.shape[1] == 0:
        return None, None, 0
    H, W = model.dims
    K = int(A.shape[1])
    A_arr = np.asarray(A.todense()) if sp.issparse(A) else np.asarray(A)
    return A_arr.reshape(H, W, K), (H, W), K


def _eccentricity(fp_bool) -> float:
    """Eccentricity ``sqrt(1 - lam_min/lam_max)`` of a binary footprint's pixel
    covariance. 0 = circular, →1 = elongated. Returns 0 for <2 pixels."""
    ys, xs = np.where(fp_bool)
    if ys.size < 2:
        return 0.0
    cov = np.cov(np.vstack([ys, xs]))
    eig = np.linalg.eigvalsh(cov)
    return float(np.sqrt(max(0.0, 1 - eig.min() / max(eig.max(), 1e-9))))


def mean_proj_and_activity(arr, *, max_pts: int = 2000, chunk_t=None):
    """Single chunk-aligned pass over a ``(T, H, W)`` numpy/zarr movie.

    Streams the mean projection while sampling a per-frame spatial-mean activity
    trace on a global stride; reads each zarr chunk exactly once. Returns
    ``(mean_img (H,W) float32, t_idx (n,) int, activity (n,) float32)``. The
    activity trace exposes drift / photobleaching / lockstep firing at a glance.
    """
    T = arr.shape[0]
    if chunk_t is None:
        ck = getattr(arr, "chunks", None)
        chunk_t = ck[0] if ck else min(T, 500)
    step = max(1, T // max_pts)
    acc = np.zeros(arr.shape[1:], dtype=np.float64)
    idx_list, vals_list = [], []
    for s in range(0, T, chunk_t):
        e = min(s + chunk_t, T)
        batch = np.asarray(arr[s:e], dtype=np.float32)   # one chunk decompress
        acc += batch.sum(0, dtype=np.float64)
        local_offset = (-s) % step                       # keep global stride
        if local_offset < e - s:
            local_idx = np.arange(local_offset, e - s, step)
            idx_list.append(s + local_idx)
            vals_list.append(batch[local_idx].reshape(local_idx.size, -1).mean(axis=1))
    mean_img = (acc / max(T, 1)).astype(np.float32)
    t_idx = np.concatenate(idx_list) if idx_list else np.array([], dtype=int)
    activity = (np.concatenate(vals_list).astype(np.float32)
                if vals_list else np.array([], dtype=np.float32))
    return mean_img, t_idx, activity


def fig_eccentricity(model, out_path=None):
    """Per-component footprint shape/size: area, peak and eccentricity
    histograms. High eccentricity (→1) flags elongated / merged / vasculature-
    contaminated footprints; area ≫ ``π(2σ)²`` flags halo or multi-cell blobs.
    Mirrors ``diagnostics.ipynb`` section C."""
    import matplotlib.pyplot as plt

    A_dense, _dims, K = _dense_components(model)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    if K:
        flat = A_dense.reshape(-1, K)
        npix = (flat > 0).sum(axis=0)
        peaks = flat.max(axis=0)
        eccs = np.array([_eccentricity(A_dense[..., k] > 0) for k in range(K)])
        sigma = float(getattr(model.params, "sigma", 3.0))
        expected = np.pi * (2 * sigma) ** 2
        axes[0].hist(npix, bins=40, color="C0", edgecolor="k")
        axes[0].axvline(expected, color="C3", ls="--", label=f"π(2σ)² = {expected:.0f}")
        axes[0].set_title("footprint area (px)"); axes[0].legend(fontsize=8)
        axes[1].hist(peaks, bins=40, color="C1", edgecolor="k")
        axes[1].set_title("peak value")
        axes[2].hist(eccs, bins=20, color="C2", edgecolor="k")
        axes[2].axvline(0.8, color="C3", ls="--", label="0.8")
        axes[2].set_title(f"eccentricity (median={np.median(eccs):.2f}, "
                          f"{(eccs > 0.8).mean() * 100:.0f}% > 0.8)")
        axes[2].legend(fontsize=8)
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "no components", ha="center", va="center",
                    transform=ax.transAxes); ax.axis("off")
    fig.suptitle("footprint shape / size distribution")
    fig.tight_layout()
    return _finish(fig, out_path)


def fig_jaccard_merge(model, *, jaccard_thr=None, corr_thr=None, max_k: int = 400,
                      out_path=None):
    """K×K spatial-Jaccard + trace-correlation matrices, annotated with the
    'should-have-merged' pair count (both above threshold). Heavy spatial+trace
    overlap on pairs that did NOT merge → ``merge_thr_corr`` too strict. Mirrors
    ``diagnostics.ipynb`` section E. Capped at ``max_k`` to bound the O(K²)
    render (a note is drawn instead when exceeded)."""
    import matplotlib.pyplot as plt

    A_dense, _dims, K = _dense_components(model)
    mt = corr_thr if corr_thr is not None else float(getattr(model.params, "merge_thr_corr", 0.85))
    mo = jaccard_thr if jaccard_thr is not None else float(getattr(model.params, "merge_thr_overlap", 0.5))
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    note = ""
    if K == 0 or getattr(model, "C", None) is None:
        for a in ax:
            a.text(0.5, 0.5, "no components", ha="center", va="center",
                   transform=a.transAxes); a.axis("off")
    elif K > max_k:
        for a in ax:
            a.text(0.5, 0.5, f"K={K} > max_k={max_k}\n(skipped — O(K²))",
                   ha="center", va="center", transform=a.transAxes); a.axis("off")
    else:
        A_bin = (A_dense.reshape(-1, K) > 0).astype(np.float32)
        inter = A_bin.T @ A_bin
        areas = A_bin.sum(0)
        union = areas[:, None] + areas[None, :] - inter
        jac = inter / np.maximum(union, 1.0)
        np.fill_diagonal(jac, 0.0)
        tc = np.corrcoef(np.asarray(model.C))
        tc = np.atleast_2d(tc)
        if tc.shape != (K, K):
            tc = np.zeros((K, K))
        np.fill_diagonal(tc, 0.0)
        im0 = ax[0].imshow(jac, cmap="magma", vmin=0, vmax=max(jac.max(), 0.1))
        ax[0].set_title("spatial Jaccard overlap")
        plt.colorbar(im0, ax=ax[0], fraction=0.046)
        im1 = ax[1].imshow(tc, cmap="RdBu_r", vmin=-1, vmax=1)
        ax[1].set_title("trace Pearson correlation")
        plt.colorbar(im1, ax=ax[1], fraction=0.046)
        both = int(((tc > mt) & (jac > mo)).sum() // 2)
        note = (f"pairs trace_corr>{mt:.2f}: {int((tc > mt).sum() // 2)}   "
                f"jaccard>{mo:.2f}: {int((jac > mo).sum() // 2)}   "
                f"BOTH (should-merge): {both}")
    fig.suptitle("pairwise overlap / correlation" + (f"\n{note}" if note else ""))
    fig.tight_layout()
    return _finish(fig, out_path)


def fig_centroid_drift(model, cn, pnr=None, out_path=None):
    """Overlay argmax centres (``footprint_center`` — what the algorithm uses in
    circular_constraint/merge) vs binary-mask COM centres on the CORR (and
    optionally PNR / CORR·PNR) images, and quantify the COM-vs-argmax drift. A
    large drift means the 'centroids miss the cells' look is a COM overlay
    artifact, not bad seeding. Mirrors ``diagnostics.ipynb`` section H."""
    import matplotlib.pyplot as plt
    from scipy.ndimage import center_of_mass

    from minicnmfe._utils import footprint_center

    A_dense, dims, K = _dense_components(model)
    panels = [("CORR", cn)]
    if pnr is not None:
        panels += [("PNR", pnr), ("CORR·PNR", cn * pnr)]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    axes = np.atleast_1d(axes)
    arg = com = None
    if K:
        peaks = A_dense.reshape(-1, K).max(axis=0)
        has = peaks > 0
        arg = np.array([footprint_center(A_dense[..., k]) if has[k] else [np.nan, np.nan]
                        for k in range(K)], float)
        com = np.array([list(center_of_mass(A_dense[..., k] > 0))
                        if (A_dense[..., k] > 0).any() else [np.nan, np.nan]
                        for k in range(K)], float)
    for ax, (title, img) in zip(axes, panels):
        vmax = (np.nanpercentile(img, 99.5)
                if img is not None and np.isfinite(img).any() else None)
        ax.imshow(img, cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(title); ax.axis("off")
    if arg is not None:
        axes[-1].scatter(com[:, 1], com[:, 0], s=10, c="0.55", alpha=0.7, label="COM")
        axes[-1].scatter(arg[:, 1], arg[:, 0], s=10, c="cyan", alpha=0.9,
                         label="argmax (algorithm)")
        leg = axes[-1].legend(loc="upper right", fontsize=8, framealpha=0.4)
        for t in leg.get_texts():
            t.set_color("white")
        valid = np.isfinite(arg[:, 0]) & np.isfinite(com[:, 0])
        if valid.any():
            drift = np.hypot(arg[valid, 0] - com[valid, 0], arg[valid, 1] - com[valid, 1])
            fig.suptitle("centre estimators — COM-vs-argmax drift: "
                         f"median {np.median(drift):.1f} px, "
                         f"90th {np.percentile(drift, 90):.1f} px")
    fig.tight_layout()
    return _finish(fig, out_path)


def fig_blob_coverage(model, cn, pnr, sigma, *, min_corr, min_pnr,
                      radius_factor: float = 1.5, out_path=None):
    """Blob-coverage check, drawn like the maintainer's manual selection overlay.

    Single panel over the **σ-smoothed CORR image** (``gaussian_filter(cn, sigma)``,
    gray, ``vmin=min_corr`` — the exact backdrop of the cutout-selection notebooks).
    On top, in one view so detected blobs and the extracted footprints can be
    compared by eye:

    - **A footprint contours** at ``0.3·max`` — **cyan** accepted / **gray** rejected
      (``model.accepted_mask``); the same overlay drawn when choosing components.
    - **detected cell blobs** (:func:`tuning.metrics.detect_cell_blobs` — ``blob_log``
      on the CORR·PNR product, thresholded by ``min_corr``/``min_pnr``) as open
      circles: **green** when an accepted footprint peak lands within
      ``radius_factor·sigma`` px, **red** when none does (a bright blob with no
      footprint).
    - **magenta ✕** at accepted-footprint peaks that sit on no blob (possible
      spurious/ghost component).

    Title carries ``blob_recall`` / ``footprint_precision``. Detection uses the raw
    ``cn``/``pnr``; only the *backdrop* is smoothed. Shares its geometry with
    :func:`tuning.metrics.blob_coverage` so figure and metric never diverge."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from scipy.ndimage import gaussian_filter

    from tuning.metrics import _footprint_peaks, detect_cell_blobs

    cn = np.asarray(cn, dtype=np.float32)
    radius = float(radius_factor * sigma)
    blobs = detect_cell_blobs(cn, pnr, sigma, min_corr=float(min_corr),
                              min_pnr=float(min_pnr))
    peaks = _footprint_peaks(model, accepted_only=True)

    def _covered(pts_a, pts_b):
        if len(pts_a) == 0 or len(pts_b) == 0:
            return np.zeros(len(pts_a), dtype=bool)
        d = np.hypot(pts_a[:, None, 0] - pts_b[None, :, 0],
                     pts_a[:, None, 1] - pts_b[None, :, 1])
        return (d <= radius).any(axis=1)

    blob_ok = _covered(blobs, peaks)
    peak_ok = _covered(peaks, blobs)
    recall = float(blob_ok.mean()) if len(blobs) else float("nan")
    prec = float(peak_ok.mean()) if len(peaks) else float("nan")

    fig, ax = plt.subplots(figsize=(8, 8))
    cn_s = gaussian_filter(cn, sigma=float(sigma)) if sigma and sigma > 0 else cn
    vmax = np.nanpercentile(cn_s, 99.5) if np.isfinite(cn_s).any() else None
    ax.imshow(cn_s, cmap="gray", vmin=float(min_corr), vmax=vmax)

    # A footprint contours (the "compare to A" overlay) — cyan accepted / gray rejected.
    A_dense, dims, K = _dense_components(model)
    if K:
        mask = getattr(model, "accepted_mask", None)
        for k in range(K):
            fp = A_dense[..., k]
            if fp.max() <= 0:
                continue
            acc = mask is None or (len(mask) == K and mask[k])
            ax.contour(fp, levels=[0.3 * fp.max()],
                       colors=("cyan" if acc else "0.5"), linewidths=0.7)

    # Detected blobs: green=covered by a footprint peak, red=uncovered.
    for (r, c), ok in zip(blobs, blob_ok):
        ax.add_patch(Circle((c, r), radius, fill=False,
                            ec=("lime" if ok else "red"), lw=1.3, alpha=0.9))
    # Footprint peaks sitting on no blob — possible ghosts.
    if len(peaks):
        off = peaks[~peak_ok]
        if len(off):
            ax.scatter(off[:, 1], off[:, 0], s=40, c="magenta", marker="x", lw=1.5)
    ax.set_title(
        f"blob coverage — recall {recall:.2f} ({int(blob_ok.sum())}/{len(blobs)} blobs), "
        f"precision {prec:.2f} ({int(peak_ok.sum())}/{len(peaks)} footprints)\n"
        "σ-smoothed CORR · cyan/gray=A footprint (acc/rej) · "
        "green/red circle=covered/uncovered blob · magenta✕=footprint on no blob")
    ax.axis("off")
    fig.tight_layout()
    return _finish(fig, out_path)


def fig_footprint_grid(model, *, n: int = 24, ncols: int = 8, out_path=None):
    """Panel grid of the top-``n`` footprints by peak, each cropped around its
    ``footprint_center`` (so the soma is centred, not pushed to the crop edge by
    COM drift), titled ``k<id> n<npix> e<ecc>`` with a red title for auto-eval-
    rejected components. The first thing to look at. Mirrors
    ``diagnostics.ipynb`` section A."""
    import matplotlib.pyplot as plt

    from minicnmfe._utils import footprint_center

    A_dense, dims, K = _dense_components(model)
    if K == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "no components", ha="center", va="center",
                transform=ax.transAxes); ax.axis("off")
        return _finish(fig, out_path)
    H, W = dims
    sigma = float(getattr(model.params, "sigma", 3.0))
    crop_r = max(8, int(3 * sigma))
    peaks = A_dense.reshape(-1, K).max(axis=0)
    mask = getattr(model, "accepted_mask", None)
    order = np.argsort(-peaks)[:min(n, K)]
    nrows = int(np.ceil(len(order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.5 * ncols, 1.7 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, k in zip(axes, order):
        fp = A_dense[..., k]
        cy, cx = footprint_center(fp) if (fp > 0).any() else (H // 2, W // 2)
        y0, y1 = max(0, cy - crop_r), min(H, cy + crop_r + 1)
        x0, x1 = max(0, cx - crop_r), min(W, cx + crop_r + 1)
        ax.imshow(fp[y0:y1, x0:x1], cmap="magma", vmin=0, vmax=max(peaks[k], 1e-9))
        acc = mask is None or (len(mask) == K and mask[k])
        ax.set_title(f"k{k} n{int((fp > 0).sum())} e{_eccentricity(fp > 0):.2f}",
                     fontsize=7, color=("k" if acc else "C3"))
        ax.axis("off")
    for ax in axes[len(order):]:
        ax.axis("off")
    fig.suptitle(f"top {len(order)} footprints (centred on argmax; "
                 "red title = auto-eval rejected)")
    fig.tight_layout()
    return _finish(fig, out_path)


def fig_mean_proj_and_activity(movie, *, max_pts: int = 2000, out_path=None):
    """Streamed mean projection + per-frame spatial-mean activity trace, in one
    chunk-aligned pass over a numpy or zarr movie. Mirrors
    ``cutout_analysis.ipynb``'s ``mean_proj_and_activity`` helper (the reduction
    is exposed standalone as :func:`mean_proj_and_activity`)."""
    import matplotlib.pyplot as plt

    mean_img, t_idx, activity = mean_proj_and_activity(movie, max_pts=max_pts)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].imshow(mean_img, cmap="gray"); ax[0].set_title("mean projection")
    ax[0].axis("off")
    ax[1].plot(t_idx, activity, lw=0.7)
    ax[1].set_title("spatial-mean activity over time")
    ax[1].set_xlabel("frame"); ax[1].set_ylabel("mean intensity")
    fig.tight_layout()
    return _finish(fig, out_path)


# Discoverable registry of the model-only diagnostic figures (each takes a
# fitted ``model`` first; ``fig_centroid_drift`` additionally needs ``cn``).
DIAGNOSTIC_FIGS = {
    "footprint_grid": fig_footprint_grid,
    "eccentricity": fig_eccentricity,
    "jaccard_merge": fig_jaccard_merge,
    "centroid_drift": fig_centroid_drift,
}


# --------------------------------------------------------------------------
# Report writer
# --------------------------------------------------------------------------

_FIG_FOR_STAGE = {
    "mc_gsig": fig_mc_gsig, "max_shift": fig_max_shift, "downsample": fig_downsample,
    "sigma": fig_corr_pnr_sigma, "corr_pnr": fig_corr_pnr_separation, "min_pixel": fig_min_pixel,
    "decay": fig_decay, "g_prior": fig_g_prior, "merge": fig_merge_corr,
}


def _params_json(recommended: dict):
    from minicnmfe.pipeline import CNMFeParams

    valid = {f.name for f in _fields(CNMFeParams)}
    filtered = {k: v for k, v in recommended.items() if k in valid}
    return CNMFeParams(**filtered), filtered


def write_report(run_dir, result, *, best_model=None, cn=None, pnr=None):
    """Write figures + recommended_params.json + downsample.json + report.md.

    ``result`` is the dict assembled by ``tuning.tuner.run_tuning``; ``best_model``
    (live) and ``cn`` (correlation image) feed the stage-4 / sweep figures. When
    ``pnr`` (the matching peak-to-noise image) is also given, the sweep gets a
    seed-coverage figure for the best candidate. Returns the run_dir path.
    """
    import matplotlib
    matplotlib.use("Agg")

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    stages = result.get("stages", {})
    saved = {}

    for key, fn in _FIG_FOR_STAGE.items():
        if key in stages:
            out = run_dir / f"fig_{key}.png"
            fn(stages[key], out_path=out)
            saved[key] = out.name

    if "snr" in stages and best_model is not None:
        out = run_dir / "fig_snr_eval.png"
        fig_snr_eval(stages["snr"], best_model, out_path=out)
        saved["snr"] = out.name

    sweep = result.get("sweep")
    if sweep:
        fig_sweep_scatter(sweep["rows"], out_path=run_dir / "fig_sweep_scatter.png")
        saved["sweep_scatter"] = "fig_sweep_scatter.png"
        if best_model is not None and best_model.A is not None and best_model.A.shape[1] > 0:
            region_crop = sweep.get("region_crop")
            fig_sweep_footprints(best_model, cn, out_path=run_dir / "fig_sweep_footprints.png",
                                 region_crop=region_crop)
            saved["sweep_footprints"] = "fig_sweep_footprints.png"
            # Seed-coverage on the best candidate. The sweep extracts on a cutout
            # (crop-local coords) while cn/pnr are full-FOV, so crop them to the
            # region so detect_seeds + footprint peaks share one coordinate frame.
            if cn is not None and pnr is not None:
                cn_c, pnr_c = cn, pnr
                if region_crop is not None:
                    (y0, y1, x0, x1), _t = region_crop
                    cn_c, pnr_c = cn[y0:y1, x0:x1], pnr[y0:y1, x0:x1]
                bp = best_model.params
                fig_blob_coverage(best_model, cn_c, pnr_c, float(bp.sigma),
                                  min_corr=float(bp.min_corr), min_pnr=float(bp.min_pnr),
                                  out_path=run_dir / "fig_sweep_blob_coverage.png")
                saved["sweep_blob_coverage"] = "fig_sweep_blob_coverage.png"
            fig_sweep_traces(best_model, out_path=run_dir / "fig_sweep_traces.png")
            saved["sweep_traces"] = "fig_sweep_traces.png"

    # recommended_params.json + downsample.json. Serialize the merged CNMFeParams
    # (the long-recording base + the tuner's data-driven fields) so the JSON is a
    # complete param set — and the exact one validated on the full recording.
    # Fall back to the curated dict for older callers that don't pass it.
    _, filtered = _params_json(result["recommended"])
    params = result.get("recommended_params") or _params_json(result["recommended"])[0]
    params.to_json(run_dir / "recommended_params.json")
    (run_dir / "downsample.json").write_text(json.dumps(
        {"ssub": int(result.get("ssub", 1)), "tsub": int(result.get("tsub", 1))}, indent=2))

    (run_dir / "report.md").write_text(_render_md(result, saved, filtered))
    return run_dir


def _render_md(result, saved, filtered) -> str:
    cfg = result.get("config", {})
    src = result.get("sources", {})
    rat = result.get("rationale", {})
    L = [f"# CNMFe parameter-tuning report — {cfg.get('name', 'session')}", ""]

    L += ["## Run configuration", ""]
    for k in ("input", "input_kind", "mode", "region", "ssub", "tsub",
              "frame_rate_hz", "decay_time_ms", "n_jobs", "timestamp", "cli"):
        if k in cfg:
            L.append(f"- **{k}**: `{cfg[k]}`")
    L.append("")

    L += ["## Recommended parameters", "",
          "| param | value | source | rationale |", "|---|---|---|---|"]
    for k in sorted(filtered):
        L.append(f"| `{k}` | `{filtered[k]}` | {src.get(k, 'default')} | {rat.get(k, '')} |")
    L += ["", f"`ssub={result.get('ssub', 1)}` `tsub={result.get('tsub', 1)}` "
          "(written to `downsample.json`; not CNMFeParams fields).", "",
          "> [!TIP] Use these directly:", ">",
          "> ```bash",
          "> python run_mc.py <movie_or_zarr> -o mc/ --params recommended_params.json",
          "> python run_extract.py mc/mc.zarr -o results/ --params recommended_params.json",
          "> ```", ""]

    def section(title, *keys):
        present = [k for k in keys if k in saved]
        if not present:
            return
        L.append(f"## {title}")
        L.append("")
        for k in present:
            L.append(f"![]({saved[k]})")
            L.append("")

    section("Stage 1 — Motion correction", "mc_gsig", "max_shift", "downsample")
    section("Stage 3 — Initialisation", "sigma", "corr_pnr", "min_pixel")
    section("Stage 4 — Temporal / merge / evaluation", "decay", "g_prior", "merge", "snr")

    sweep = result.get("sweep")
    if sweep:
        L += ["## Sweep results", "",
              f"Region: **{sweep['region']}**"
              + (f" — crop {sweep.get('region_crop')}" if sweep.get("region_crop") else ""),
              ""]
        sgrid = (result.get("stages") or {}).get("sigma_grid")
        if sgrid:
            grid_s = ", ".join(f"{v:g}" for v in sgrid.get("values", []))
            L += [f"Resolved σ grid: **{grid_s}** "
                  f"(anchored on heuristic σ≈{sgrid.get('heuristic', float('nan')):.2f}).",
                  ""]
        rows = sweep["rows"]
        cols = ["idx", "sigma", "min_corr", "min_pnr", "merge_thr_corr",
                "global_bg_rank", "init_stride", "K", "K_accepted",
                "cprojcorr_median", "npix_median", "multipeak_frac", "npix_oversize",
                "snr_median", "score", "wall_s"]
        L.append("| " + " | ".join(cols) + " |")
        L.append("|" + "|".join("---" for _ in cols) + "|")
        for i, r in enumerate(rows):
            cells = []
            for c in cols:
                v = r.get(c)
                if isinstance(v, float):
                    v = f"{v:.3f}" if abs(v) < 1e6 else f"{v:.1e}"
                cells.append("—" if v is None else str(v))
            line = "| " + " | ".join(cells) + " |"
            L.append(f"**{line}**" if i == 0 else line)
        L.append("")
        for k in ("sweep_scatter", "sweep_footprints", "sweep_blob_coverage",
                  "sweep_traces"):
            if k in saved:
                L += [f"![]({saved[k]})", ""]

    L += ["## How to read these metrics", "", METRICS_BLURB, ""]
    return "\n".join(L)
