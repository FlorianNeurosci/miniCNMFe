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
    ``cprojcorr_median``, ``npix_median``, ``npix_iqr``, ``snr_mean``,
    ``snr_median``. Guards the K==0 / unfitted cases (returns 0 / NaN).
    """
    A = model.A
    K = int(A.shape[1]) if A is not None else 0
    q: dict = {"K": K}
    if K == 0:
        q.update(K_accepted=0, accepted_frac=0.0,
                 cprojcorr_mean=float("nan"), cprojcorr_median=float("nan"),
                 npix_median=0.0, npix_iqr=0.0,
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


def composite_score(q: dict, weights: "dict | None" = None) -> float:
    """Transparent per-candidate ranking score (NOT an absolute quality claim).

    ``score = w_corr · cprojcorr_median + w_acc · accepted_frac
              − w_tight · (npix_iqr / npix_median)``

    K==0 candidates score ``-inf`` (nothing extracted). The default weights
    favour per-trace purity and a clean accepted set while lightly penalising
    inconsistent footprint sizes. Fully re-derivable from the printed table, so
    the user can re-rank with their own weights.
    """
    w = {"corr": 1.0, "acc": 0.5, "tight": 0.25}
    if weights:
        w.update(weights)
    if q.get("K", 0) == 0:
        return float("-inf")
    corr = q.get("cprojcorr_median", 0.0)
    corr = 0.0 if (corr is None or np.isnan(corr)) else corr
    acc = q.get("accepted_frac", 0.0) or 0.0
    npix_med = q.get("npix_median", 0.0) or 0.0
    tight = (q.get("npix_iqr", 0.0) or 0.0) / npix_med if npix_med > 0 else 0.0
    return float(w["corr"] * corr + w["acc"] * acc - w["tight"] * tight)
