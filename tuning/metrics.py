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


def model_quality(model) -> dict:
    """Flat dict of quality proxies for a fitted ``CNMFe`` model.

    Keys: ``K``, ``K_accepted``, ``accepted_frac``, ``cprojcorr_mean``,
    ``cprojcorr_median``, ``npix_median``, ``npix_iqr``, ``multipeak_frac``,
    ``npix_oversize``, ``snr_mean``, ``snr_median``. Guards the K==0 / unfitted
    cases (returns 0 / NaN).
    """
    A = model.A
    K = int(A.shape[1]) if A is not None else 0
    q: dict = {"K": K}
    if K == 0:
        q.update(K_accepted=0, accepted_frac=0.0,
                 cprojcorr_mean=float("nan"), cprojcorr_median=float("nan"),
                 npix_median=0.0, npix_iqr=0.0,
                 multipeak_frac=float("nan"), npix_oversize=float("nan"),
                 snr_mean=float("nan"), snr_median=float("nan"),
                 trace_corr_median=float("nan"))
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


def composite_score(q: dict, weights: "dict | None" = None) -> float:
    """Transparent per-candidate ranking score (NOT an absolute quality claim).

    ``score = w_corr · cprojcorr_median + w_acc · accepted_frac
              − w_tight · (npix_iqr / npix_median) − w_merge · multipeak_frac``

    K==0 candidates score ``-inf`` (nothing extracted). The default weights
    favour per-trace purity and a clean accepted set while penalising
    inconsistent footprint sizes and over-merged (multi-peak) footprints — the
    latter stops the ranking preferring an over-large ``sigma`` that fuses
    neighbours into fewer, bigger, deceptively "clean" components. Fully
    re-derivable from the printed table, so the user can re-rank with their own
    weights.
    """
    w = {"corr": 1.0, "acc": 0.5, "tight": 0.25, "merge": 0.5}
    if weights:
        w.update(weights)
    if q.get("K", 0) == 0:
        return float("-inf")
    corr = q.get("cprojcorr_median", 0.0)
    corr = 0.0 if (corr is None or np.isnan(corr)) else corr
    acc = q.get("accepted_frac", 0.0) or 0.0
    npix_med = q.get("npix_median", 0.0) or 0.0
    tight = (q.get("npix_iqr", 0.0) or 0.0) / npix_med if npix_med > 0 else 0.0
    multipeak = q.get("multipeak_frac", 0.0)
    multipeak = 0.0 if (multipeak is None or np.isnan(multipeak)) else multipeak
    return float(w["corr"] * corr + w["acc"] * acc
                 - w["tight"] * tight - w["merge"] * multipeak)
