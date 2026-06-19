"""Ground-truth-free quality proxies for a fitted CNMFe model.

These are **proxies, not validation** — there is no ground truth on a real
recording. They are designed to (a) let the user eyeball candidate quality from
the report and (b) rank the sweep's candidates against each other. The roadmap's
C1 validation harness (``todo/future_improvements_roadmap.md``) would later add
real cross-method / paired-ephys validation; this module is the seam it could
reuse.

The central signal is ``cprojcorr_median`` — the median per-cell Pearson
correlation between the demixed trace ``C`` and the noisy projection ``C + YrA``.
Per CLAUDE.md, in a dense FOV this **falls** as the extracted cell count rises
(YrA cross-talk), so it doubles as a density↔purity knob for picking thresholds.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _multipeak_frac(A, dims, sigma: float, mask=None,
                    rel_thr: float = 0.3) -> float:
    """Fraction of footprints carrying >=2 well-separated bright peaks.

    A footprint with two or more distinct soma-scale maxima is the spatial
    signature of ``sigma`` set too large: the seed-suppression disk wipes a
    neighbour and the spatial LASSO merges both cells into one component (see
    ``tuning/report.py:SYMPTOM_CAUSE_KNOB``). Peaks are counted with
    ``peak_local_max`` at ``min_distance ~ sigma`` and ``threshold_abs =
    rel_thr * footprint.max()`` so only a substantial secondary soma counts, not
    jagged LASSO ripple. Restricted to ``mask`` (accepted cells) when given.
    Returns NaN if there are no footprints to judge.
    """
    from skimage.feature import peak_local_max

    A_csc = A.tocsc() if sp.issparse(A) else sp.csc_matrix(A)
    K = A_csc.shape[1]
    H, W = dims
    idx = range(K)
    if mask is not None and len(mask) == K:
        idx = [k for k in range(K) if mask[k]]
    if not len(list(idx)):
        return float("nan")
    min_distance = max(1, int(round(sigma)))
    multi = 0
    n = 0
    for k in idx:
        fp = np.asarray(A_csc[:, k].todense()).reshape(H, W)
        peak = float(fp.max())
        if peak <= 0:
            continue
        n += 1
        peaks = peak_local_max(fp, min_distance=min_distance,
                               threshold_abs=rel_thr * peak)
        if len(peaks) >= 2:
            multi += 1
    return float(multi / n) if n else float("nan")


def _per_cell_corr(C: np.ndarray, Cproj: np.ndarray) -> np.ndarray:
    """Per-row Pearson r between ``C`` and ``C + YrA``."""
    a = C - C.mean(axis=1, keepdims=True)
    b = Cproj - Cproj.mean(axis=1, keepdims=True)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    denom = na * nb
    out = np.zeros(C.shape[0], dtype=np.float64)
    good = denom > 1e-12
    out[good] = (a[good] * b[good]).sum(axis=1) / denom[good]
    return out


def per_cell_spatial_corr(A, cn: np.ndarray, dims, pad: int = 4) -> np.ndarray:
    """Per-component cn-patch proxy for spatial fidelity.

    For each footprint, Pearson r between the footprint values and the local
    correlation image, over a bounding box (footprint extent + ``pad`` surround).
    The surround is what gives a clean single-cell footprint a high score (its
    "bright blob on dark edge" matches the local CORR hotspot) while a merged /
    sprawled footprint spanning several CORR spots scores low. Cheap (no movie) —
    the per-candidate proxy for the faithful ``evaluate.spatial_r_values``.

    Args:
        A: (H*W, K) sparse/dense footprints (pixel order h*W + w).
        cn: (H, W) correlation image, SAME coords as ``A`` (crop it to the
            candidate's region first if A is crop-local).
        dims: (H, W).
        pad: surround in px added around each footprint's bbox.

    Returns: (K,) float r in [-1, 1] (0 if degenerate / empty footprint).
    """
    A_csc = A.tocsc() if sp.issparse(A) else sp.csc_matrix(A)
    H, W = dims
    K = A_csc.shape[1]
    cn = np.nan_to_num(np.asarray(cn, dtype=np.float64))
    r = np.full(K, np.nan)
    for k in range(K):
        s, e = A_csc.indptr[k], A_csc.indptr[k + 1]
        if s == e:
            continue
        rows = A_csc.indices[s:e]
        vals = A_csc.data[s:e].astype(np.float64)
        ys, xs = rows // W, rows % W
        y0, y1 = max(0, int(ys.min()) - pad), min(H, int(ys.max()) + 1 + pad)
        x0, x1 = max(0, int(xs.min()) - pad), min(W, int(xs.max()) + 1 + pad)
        fp = np.zeros((y1 - y0, x1 - x0), dtype=np.float64)
        fp[ys - y0, xs - x0] = vals
        cbox = cn[y0:y1, x0:x1].ravel()
        fpv = fp.ravel()
        if fpv.std() > 0 and cbox.std() > 0:
            r[k] = float(np.corrcoef(fpv, cbox)[0, 1])
        else:
            r[k] = 0.0
    return r


def model_quality(model, cn: "np.ndarray | None" = None) -> dict:
    """Flat dict of quality proxies for a fitted ``CNMFe`` model.

    Keys: ``K``, ``K_accepted``, ``accepted_frac``, ``cprojcorr_mean``,
    ``cprojcorr_median``, ``npix_median``, ``npix_iqr``, ``npix_p25``
    (footprint 25th-pct npix; the tuner derives ``min_pixel`` from it),
    ``multipeak_frac``, ``npix_oversize``, ``snr_mean``, ``snr_median``,
    ``trace_corr_median`` (median cross-component trace |corr| — redundancy
    proxy used in ``session_quality_verdict``), ``spatialcorr_median``.
    Guards the K==0 / unfitted cases (returns 0 / NaN).

    ``cn`` (optional, SAME coords as ``model.A``) enables the spatial-fidelity
    proxy ``spatialcorr_median`` (median footprint↔local-CORR correlation; see
    ``per_cell_spatial_corr``) — the term that lets the sweep prefer compact
    single-cell footprints over merged large-sigma blobs (which the temporal
    ``cprojcorr`` cannot distinguish). ``None`` → ``spatialcorr_median`` is NaN.
    """
    A = model.A
    K = int(A.shape[1]) if A is not None else 0
    q: dict = {"K": K}
    if K == 0:
        q.update(K_accepted=0, accepted_frac=0.0,
                 cprojcorr_mean=float("nan"), cprojcorr_median=float("nan"),
                 npix_median=0.0, npix_iqr=0.0, npix_p25=0.0,
                 multipeak_frac=float("nan"), npix_oversize=float("nan"),
                 snr_mean=float("nan"), snr_median=float("nan"),
                 trace_corr_median=float("nan"), spatialcorr_median=float("nan"))
        return q

    # Accepted fraction (auto-eval; may be unset).
    mask = getattr(model, "accepted_mask", None)
    if mask is not None and len(mask) == K:
        q["K_accepted"] = int(np.sum(mask))
        q["accepted_frac"] = float(np.mean(mask))
    else:
        q["K_accepted"] = K
        q["accepted_frac"] = 1.0

    # corr(C, C+YrA) per cell.
    if model.C is not None and model.YrA is not None:
        r = _per_cell_corr(np.asarray(model.C), np.asarray(model.C) + np.asarray(model.YrA))
        q["cprojcorr_mean"] = float(np.mean(r))
        q["cprojcorr_median"] = float(np.median(r))
    else:
        q["cprojcorr_mean"] = q["cprojcorr_median"] = float("nan")

    # Cross-component trace redundancy: median |pairwise Pearson r| among the
    # component traces. High => components share temporal activity — real
    # synchrony OR an over-split shared signal. Restricted to accepted cells
    # (noise/ghosts are uncorrelated and would dilute the number) and capped to
    # 200 cells for cost.
    if model.C is not None and K >= 2:
        C = np.asarray(model.C, dtype=np.float64)
        if mask is not None and len(mask) == K and int(np.sum(mask)) >= 2:
            C = C[np.asarray(mask, dtype=bool)]
        if C.shape[0] > 200:
            C = C[np.linspace(0, C.shape[0] - 1, 200).astype(int)]
        cc = np.corrcoef(C)
        iu = np.triu_indices(cc.shape[0], k=1)
        pair = np.abs(cc[iu])
        pair = pair[np.isfinite(pair)]
        q["trace_corr_median"] = float(np.median(pair)) if pair.size else float("nan")
    else:
        q["trace_corr_median"] = float("nan")

    # Footprint pixel-count distribution (nonzero pixels per column).
    A_csc = A.tocsc() if sp.issparse(A) else sp.csc_matrix(A)
    npix = np.diff(A_csc.indptr)
    q["npix_median"] = float(np.median(npix))
    q["npix_iqr"] = float(np.percentile(npix, 75) - np.percentile(npix, 25))
    # 25th-pct realized footprint area — the tuner uses this as min_pixel (a floor
    # that flags the smallest ~quarter of footprints), now measured on the actual
    # (nrg-thresholded) BCD footprints rather than greedy-init footprints.
    q["npix_p25"] = float(np.percentile(npix, 25))

    # Over-merge signals. ``multipeak_frac`` (fraction of accepted footprints
    # with >=2 distinct soma-scale peaks) is the direct signature of sigma too
    # large; it feeds the ranking. ``npix_oversize`` (median footprint area over
    # the expected single-soma area) is diagnostic only — shows roughly how many
    # cells the median footprint spans. Both need the spatial grid + sigma.
    sigma = float(getattr(getattr(model, "params", None), "sigma", 0.0) or 0.0)
    dims = getattr(model, "dims", None)
    if dims is not None and sigma > 0:
        q["multipeak_frac"] = _multipeak_frac(A, dims, sigma, mask=mask)
        cdf = float(getattr(model.params, "spatial_circular_max_dist_factor", 1.5))
        soma_area = np.pi * (cdf * sigma) ** 2
        q["npix_oversize"] = float(q["npix_median"] / soma_area) if soma_area > 0 else float("nan")
    else:
        q["multipeak_frac"] = float("nan")
        q["npix_oversize"] = float("nan")

    # Per-component SNR from auto-eval.
    eval_info = getattr(model, "eval_info", None)
    if eval_info is not None and "snr_amp" in eval_info:
        snr = np.asarray(eval_info["snr_amp"], dtype=float)
        q["snr_mean"] = float(np.mean(snr))
        q["snr_median"] = float(np.median(snr))
    else:
        q["snr_mean"] = q["snr_median"] = float("nan")

    # Spatial-fidelity proxy (footprint vs local CORR) — drives the ranking so a
    # merged large-sigma footprint (spanning several CORR spots) is penalised.
    if cn is not None and dims is not None:
        sc = per_cell_spatial_corr(A, cn, dims)
        q["spatialcorr_median"] = float(np.nanmedian(sc)) if sc.size else float("nan")
    else:
        q["spatialcorr_median"] = float("nan")
    return q


def mc_quality(shifts: "np.ndarray | None") -> dict:
    """Motion-correction quality proxies from the per-frame shift array.

    ``shift_smoothness`` = mean absolute frame-to-frame change (lower = smoother
    drift, no jitter); ``shift_p99`` / ``shift_max`` = percentile / max shift
    magnitude (sanity-check against ``max_shift``).
    """
    if shifts is None or len(shifts) == 0:
        return {"shift_smoothness": float("nan"), "shift_p99": float("nan"),
                "shift_max": float("nan")}
    shifts = np.asarray(shifts, dtype=np.float64)
    mag = np.linalg.norm(shifts, axis=1)
    dd = np.linalg.norm(np.diff(shifts, axis=0), axis=1) if len(shifts) > 1 else np.array([0.0])
    return {"shift_smoothness": float(np.mean(dd)),
            "shift_p99": float(np.percentile(mag, 99)),
            "shift_max": float(np.max(mag))}


def crispness(summary_img) -> float:
    """Frobenius norm of the spatial gradient of a 2-D summary image.

    The standard ground-truth-free registration-quality measure (CaImAn's
    mean-image crispness): a sharper, better-registered movie has a crisper
    time-average — residual motion smears edges and lowers the gradient energy.
    Only comparable **between candidates of the same clip** (same dims /
    intensity scale), which is exactly how the MC search uses it.
    """
    img = np.asarray(summary_img, dtype=np.float64)
    gy, gx = np.gradient(img)
    return float(np.sqrt((gx ** 2 + gy ** 2).sum()))


def correlation_image(movie, neighbors: int = 8) -> np.ndarray:
    """Local neighbour temporal-correlation image (CaImAn's ``Cn``).

    Each pixel's value is the mean Pearson correlation of its z-scored time
    series with its 4/8 nearest neighbours. High where temporally-active pixels
    (cells) are well co-registered; motion / misregistration decorrelates
    neighbours and lowers it. Static pixels have ~zero temporal variance, so
    they contribute ~0 — which is exactly why this tracks **cell** registration
    rather than the static background.
    """
    m = np.asarray(movie, dtype=np.float32)
    m = m - m.mean(axis=0, keepdims=True)
    s = m.std(axis=0, keepdims=True)
    m = np.divide(m, s, out=np.zeros_like(m), where=s > 1e-6)
    offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if neighbors == 8:
        offsets += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    acc = np.zeros(m.shape[1:], dtype=np.float32)
    for dy, dx in offsets:
        acc += (m * np.roll(m, (dy, dx), axis=(1, 2))).mean(axis=0)
    return acc / len(offsets)


def mc_registration_quality(corrected) -> dict:
    """Cell-focused registration-quality proxies for a corrected ``(T,H,W)`` stack.

    ``corr_mean`` / ``corr_p99`` summarise the local correlation image
    (``correlation_image``) — the **primary** MC-ranking signal: better cell
    co-registration raises neighbour correlation. ``std_crispness`` is the
    gradient energy of the temporal-std image (active pixels) — a secondary
    cell-focused sharpness measure that moves the same way.

    Note: mean-image crispness is deliberately NOT used to rank MC candidates on
    these 1p recordings — it is dominated by the bright *static* background,
    which smears (crispness *drops*) precisely when real tissue motion is
    correctly removed, so it rewards under-correction. ``mc_crispness`` is kept
    for diagnostics only.
    """
    arr = np.asarray(corrected, dtype=np.float32)
    cn = correlation_image(arr)
    return {"corr_mean": float(cn.mean()),
            "corr_p99": float(np.percentile(cn, 99)),
            "std_crispness": crispness(arr.std(axis=0))}


def mc_crispness(corrected) -> dict:
    """Mean/std-image crispness — DIAGNOSTIC ONLY (do not rank MC by this).

    ``crispness_mean`` is misleading on 1p data (static-background dominated);
    use ``mc_registration_quality`` to choose MC params. Retained for reports.
    """
    arr = np.asarray(corrected, dtype=np.float32)
    return {"crispness_mean": crispness(arr.mean(axis=0)),
            "crispness_std": crispness(arr.std(axis=0))}


# --------------------------------------------------------------------------
# Seed coverage + session quality verdict (the three by-eye checks the
# maintainer runs when reviewing a tuned extraction, turned into numbers)
# --------------------------------------------------------------------------

# Default thresholds for the PASS/WARN verdict. These are **proxies, not
# validation** (see the module docstring) — tune them per rig if needed.
QUALITY_THRESHOLDS = {
    "blob_recall_min": 0.80,         # < => bright CORR·PNR blobs with no footprint
    "footprint_precision_min": 0.80, # < => footprints sitting on no CORR·PNR blob
    "cprojcorr_min": 0.50,           # < => C+YrA diverges from the demixed C
    "trace_corr_max": 0.40,          # > => traces too correlated across cells
}


def detect_product_blobs(cn: np.ndarray, pnr: np.ndarray, sigma: float, *,
                         threshold: float = 0.05) -> np.ndarray:
    """``(N, 2)`` blob centres ``(row, col)`` on the normalised CORR·PNR image.

    Uses ``skimage.feature.blob_log`` on the normalised ``cn * pnr`` product —
    the same detector :func:`tuning.heuristics.suggest_sigma_extraction` uses to
    size neurons — so a "blob" is a visually-distinct cell, not the dense local
    maxima ``detect_seeds`` returns (which over-counts relative to the cells the
    eye picks out). Blob radius scales with ``sigma``. This is the unfiltered
    neuron set; :func:`detect_cell_blobs` applies the corr/pnr keep-filter on top,
    and the threshold-seed heuristics in :mod:`tuning.heuristics` read the
    corr/pnr values at these centres to *derive* those thresholds.
    """
    from skimage.feature import blob_log

    cn = np.asarray(cn, dtype=np.float64)
    pnr = np.asarray(pnr, dtype=np.float64)
    product = cn * pnr
    rng = product.max() - product.min()
    if rng <= 0:
        return np.zeros((0, 2), dtype=float)
    product_n = (product - product.min()) / (rng + 1e-8)
    min_s = max(1.0, 0.5 * sigma)
    max_s = max(min_s + 1.0, 3.0 * sigma)
    blobs = blob_log(product_n, min_sigma=min_s, max_sigma=max_s,
                     num_sigma=10, threshold=threshold)
    if len(blobs) == 0:
        return np.zeros((0, 2), dtype=float)
    return blobs[:, :2].astype(float)


def detect_cell_blobs(cn: np.ndarray, pnr: np.ndarray, sigma: float, *,
                      min_corr: float, min_pnr: float,
                      threshold: float = 0.05) -> np.ndarray:
    """``(N, 2)`` cell-like blob centres ``(row, col)`` on the CORR·PNR image.

    :func:`detect_product_blobs` finds the neuron blobs; here they are kept only
    where ``cn >= min_corr`` and ``pnr >= min_pnr`` at the centre, matching the
    thresholded CORR / min-corr image the user inspects.
    """
    blobs = detect_product_blobs(cn, pnr, sigma, threshold=threshold)
    if len(blobs) == 0:
        return blobs
    cn = np.asarray(cn, dtype=np.float64)
    pnr = np.asarray(pnr, dtype=np.float64)
    rc = blobs.astype(int)
    keep = (cn[rc[:, 0], rc[:, 1]] >= min_corr) & (pnr[rc[:, 0], rc[:, 1]] >= min_pnr)
    return blobs[keep]


def _footprint_peaks(model, *, accepted_only: bool = True) -> np.ndarray:
    """``(M, 2)`` array of footprint peak ``(row, col)`` locations.

    Uses :func:`minicnmfe._utils.footprint_center` (argmax of a lightly smoothed
    footprint — the same centre the algorithm uses for circular_constraint /
    merge, robust to sprawl/donuts). Columns whose footprint is all-zero are
    dropped. When ``accepted_only`` and ``model.accepted_mask`` is present, only
    accepted components contribute (the "circles" the user actually sees).
    """
    from minicnmfe._utils import footprint_center

    A = getattr(model, "A", None)
    dims = getattr(model, "dims", None)
    if A is None or dims is None or A.shape[1] == 0:
        return np.zeros((0, 2), dtype=float)
    K = A.shape[1]
    A_csc = A.tocsc() if sp.issparse(A) else sp.csc_matrix(A)
    mask = getattr(model, "accepted_mask", None)
    if accepted_only and mask is not None and len(mask) == K:
        idx = [k for k in range(K) if mask[k]]
    else:
        idx = list(range(K))
    H, W = dims
    pts = []
    for k in idx:
        fp = np.asarray(A_csc[:, k].todense()).reshape(H, W)
        if fp.max() <= 0:
            continue
        pts.append(footprint_center(fp))
    return np.asarray(pts, dtype=float) if pts else np.zeros((0, 2), dtype=float)


def _bidirectional_match(blobs: np.ndarray, peaks: np.ndarray, radius: float):
    """Count, for each side, how many points have a neighbour within ``radius``.

    Returns ``(n_blobs_covered, n_peaks_on_blob)``. Robust to either side being
    empty. Uses a cKDTree when both sides are non-trivial, else a direct
    distance matrix.
    """
    n_b, n_p = len(blobs), len(peaks)
    if n_b == 0 or n_p == 0:
        return 0, 0
    if n_b * n_p > 4096:
        from scipy.spatial import cKDTree
        tree_p = cKDTree(peaks)
        tree_b = cKDTree(blobs)
        covered = sum(1 for nb in tree_p.query_ball_point(blobs, radius) if nb)
        on_blob = sum(1 for nb in tree_b.query_ball_point(peaks, radius) if nb)
        return int(covered), int(on_blob)
    d = np.hypot(blobs[:, None, 0] - peaks[None, :, 0],
                 blobs[:, None, 1] - peaks[None, :, 1])
    within = d <= radius
    return int(np.count_nonzero(within.any(axis=1))), \
           int(np.count_nonzero(within.any(axis=0)))


def blob_coverage(model, cn: np.ndarray, pnr: np.ndarray, sigma: float, *,
                  min_corr: float, min_pnr: float,
                  radius_factor: float = 1.5) -> dict:
    """Two-way match between CORR·PNR cell blobs and extracted footprint peaks.

    Encodes the maintainer's "does every bright blob in the CORR / min-corr
    image have a circle, and does every circle sit on a blob?" check. Blobs come
    from :func:`detect_cell_blobs` (deduplicated, cell-scale, thresholded by the
    run's ``min_corr``/``min_pnr``); footprint peaks from :func:`_footprint_peaks`
    (accepted components). A blob and a peak match when within
    ``radius_factor * sigma`` pixels.

    Returns a flat dict: ``n_blobs``, ``n_blobs_covered``, ``blob_recall``
    (fraction of blobs with a footprint — "missing cells" when low),
    ``n_footprints``, ``n_footprints_on_blob``, ``footprint_precision``
    (fraction of footprints on a blob — possible ghosts when low),
    ``coverage_radius``. Recall/precision are NaN when their denominator is 0.
    """
    radius = float(radius_factor * sigma)
    blobs = detect_cell_blobs(cn, pnr, sigma, min_corr=min_corr, min_pnr=min_pnr)
    peaks = _footprint_peaks(model, accepted_only=True)
    n_blobs, n_peaks = len(blobs), len(peaks)
    n_cov, n_on = _bidirectional_match(blobs, peaks, radius)
    return {
        "n_blobs": int(n_blobs),
        "n_blobs_covered": int(n_cov),
        "blob_recall": float(n_cov / n_blobs) if n_blobs else float("nan"),
        "n_footprints": int(n_peaks),
        "n_footprints_on_blob": int(n_on),
        "footprint_precision": float(n_on / n_peaks) if n_peaks else float("nan"),
        "coverage_radius": radius,
    }


def session_quality_verdict(q: dict, coverage: dict,
                            thresholds: "dict | None" = None) -> dict:
    """Turn the three by-eye checks into a PASS/WARN verdict with reasons.

    Combines the blob-coverage metrics (:func:`blob_coverage`) with the two
    trace metrics already in :func:`model_quality` (``cprojcorr_median`` =
    agreement of ``C`` with ``C+YrA``; ``trace_corr_median`` = cross-cell trace
    correlation). A check that cannot be evaluated (NaN metric) is skipped, not
    failed. Returns ``{"status": "PASS"|"WARN", "warnings": [str, ...],
    "checks": {name: bool}}`` where each ``checks`` value is True when the check
    passes (or was skipped).
    """
    t = dict(QUALITY_THRESHOLDS)
    if thresholds:
        t.update(thresholds)
    checks: dict = {}
    warnings: list = []

    def _ok(name: str, passed: "bool | None", msg: str):
        # None => not evaluable; treat as passing but don't warn.
        checks[name] = True if passed is None else bool(passed)
        if passed is False:
            warnings.append(msg)

    recall = coverage.get("blob_recall")
    rev = None if recall is None or np.isnan(recall) else recall >= t["blob_recall_min"]
    _ok("blob_recall", rev,
        f"low blob coverage: only {recall:.2f} of CORR·PNR cell blobs have a footprint "
        f"(< {t['blob_recall_min']:.2f}) — bright blobs with no circle"
        if rev is False else "")

    prec = coverage.get("footprint_precision")
    pev = None if prec is None or np.isnan(prec) else prec >= t["footprint_precision_min"]
    _ok("footprint_precision", pev,
        f"low footprint precision: only {prec:.2f} of footprints sit on a CORR·PNR "
        f"blob (< {t['footprint_precision_min']:.2f}) — possible spurious/ghost components"
        if pev is False else "")

    cproj = q.get("cprojcorr_median")
    cev = None if cproj is None or np.isnan(cproj) else cproj >= t["cprojcorr_min"]
    _ok("cprojcorr", cev,
        f"C+YrA diverges from C: cprojcorr_median {cproj:.2f} < {t['cprojcorr_min']:.2f} "
        "— impure demixing / params off"
        if cev is False else "")

    tcorr = q.get("trace_corr_median")
    tev = None if tcorr is None or np.isnan(tcorr) else tcorr <= t["trace_corr_max"]
    _ok("trace_corr", tev,
        f"traces heavily correlated across cells: trace_corr_median {tcorr:.2f} > "
        f"{t['trace_corr_max']:.2f} — over-split or background bleed"
        if tev is False else "")

    status = "PASS" if all(checks.values()) else "WARN"
    return {"status": status, "warnings": warnings, "checks": checks}


def composite_score(q: dict, weights: "dict | None" = None) -> float:
    """Transparent per-candidate ranking score (NOT an absolute quality claim).

    ``score = w_corr · cprojcorr_median + w_spatial · spatialcorr_median
              + w_acc · accepted_frac
              − w_tight · (npix_iqr / npix_median) − w_merge · multipeak_frac``

    K==0 candidates score ``-inf`` (nothing extracted). The default weights
    favour per-trace purity (``cprojcorr``) AND per-footprint spatial fidelity
    (``spatialcorr`` — footprint vs local CORR) and a clean accepted set, while
    penalising inconsistent footprint sizes and over-merged (multi-peak)
    footprints. The spatial term is load-bearing: the temporal ``cprojcorr`` is
    nearly blind to an over-large ``sigma`` that fuses neighbours into bigger
    "clean"-looking blobs (validated: cprojcorr ≈ equal at sigma 3 vs 4 while
    spatialcorr / r_value roughly double), so without it the ranking over-picks
    the merged large-sigma candidate. ``spatialcorr_median`` is NaN when no
    ``cn`` was supplied to ``model_quality`` → the term contributes 0 (legacy
    behaviour). Fully re-derivable from the printed table.
    """
    w = {"corr": 1.0, "spatial": 1.0, "acc": 0.5, "tight": 0.25, "merge": 0.5}
    if weights:
        w.update(weights)
    if q.get("K", 0) == 0:
        return float("-inf")
    corr = q.get("cprojcorr_median", 0.0)
    corr = 0.0 if (corr is None or np.isnan(corr)) else corr
    spatial = q.get("spatialcorr_median", 0.0)
    spatial = 0.0 if (spatial is None or np.isnan(spatial)) else spatial
    acc = q.get("accepted_frac", 0.0) or 0.0
    npix_med = q.get("npix_median", 0.0) or 0.0
    tight = (q.get("npix_iqr", 0.0) or 0.0) / npix_med if npix_med > 0 else 0.0
    multipeak = q.get("multipeak_frac", 0.0)
    multipeak = 0.0 if (multipeak is None or np.isnan(multipeak)) else multipeak
    return float(w["corr"] * corr + w["spatial"] * spatial + w["acc"] * acc
                 - w["tight"] * tight - w["merge"] * multipeak)
