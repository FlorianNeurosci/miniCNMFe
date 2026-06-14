"""Per-parameter suggestion heuristics.

Each ``suggest_*`` function returns ``(value, evidence)`` where ``value`` is the
suggested parameter (or tuple) and ``evidence`` is a dict carrying the arrays a
figure needs. **No matplotlib here** — ``tuning.report`` draws from the
evidence dicts so the heuristics stay importable in headless / worker contexts.

The logic is lifted from ``live_runs/estimate_params.ipynb`` (the notebook is
now a thin viewer over these functions). It reuses the proven pipeline
primitives — ``correlation_pnr``, ``detect_seeds``, ``greedy_corr_pnr``,
``estimate_shifts``, ``estimate_ar_params`` — rather than reimplementing them.
"""

from __future__ import annotations

import numpy as np

from minicnmfe.initialization import greedy_corr_pnr
from minicnmfe.motion_correction import estimate_shifts
from minicnmfe.preprocess import correlation_pnr
from minicnmfe.temporal import estimate_ar_params

# ---------------------------------------------------------------------------
# Stage 1 — motion correction
# ---------------------------------------------------------------------------


def suggest_mc_gsig_and_sigma(
    sample: np.ndarray, *, min_sigma: float = 2, max_sigma: float = 15,
    num_sigma: int = 10, threshold: float = 0.05, top_n: int = 30,
    highpass_sigma: "float | None" = 8.0,
) -> "tuple[int, float, dict]":
    """Neuron radius from a temporal-std projection + ``blob_log``.

    Returns ``(mc_gSig_filt, sigma_native, evidence)``. The high-pass radius for
    MC should roughly match the neuron radius, so the same value seeds the
    extraction-grid ``sigma`` (stage 3).

    ``highpass_sigma`` (default 8 px) subtracts a wide Gaussian blur from the
    temporal-std image before ``blob_log``, removing broad out-of-focus haze /
    vignetting / gradients. Without it, on a hazy or out-of-focus FOV ``blob_log``
    latches onto large diffuse background structures and the median radius
    inflates (e.g. 6.3 px when the neurons are ~4 px), which then blows up
    ``sigma`` / ``ssub`` / ``min_pixel`` and makes footprints sprawl. The cutoff
    is set well above the neuron scale so real neurons survive; clean FOVs (no
    broad background) are left unchanged. Pass ``0`` / ``None`` to disable.
    """
    from skimage.feature import blob_log

    std_img = sample.std(axis=0).astype(np.float32)
    if highpass_sigma:
        import cv2

        bg = cv2.GaussianBlur(std_img, (0, 0), sigmaX=float(highpass_sigma))
        std_img = np.clip(std_img - bg, 0.0, None)
    std_norm = (std_img - std_img.min()) / (std_img.max() - std_img.min() + 1e-8)
    blobs = blob_log(
        std_norm, min_sigma=min_sigma, max_sigma=max_sigma,
        num_sigma=num_sigma, threshold=threshold,
    )
    evidence = {"std_img": std_img}
    if len(blobs) == 0:
        # Degenerate: fall back to a sane miniscope default.
        evidence.update(blobs_top=np.empty((0, 3)), sigmas_top=np.empty(0),
                        median_sigma=4.0, ok=False)
        return 4, 4.0, evidence

    scores = std_norm[blobs[:, 0].astype(int), blobs[:, 1].astype(int)]
    order = np.argsort(-scores)[:top_n]
    blobs_top = blobs[order]
    sigmas_top = blobs[order, 2]
    median_sigma = float(np.median(sigmas_top))
    evidence.update(blobs_top=blobs_top, sigmas_top=sigmas_top,
                    median_sigma=median_sigma, ok=True)
    return int(round(median_sigma)), median_sigma, evidence


def suggest_max_shift(
    sample: np.ndarray, median_img: np.ndarray, gSig_filt: "float | None", *,
    n_shift_frames: int = 200, upsample_factor: int = 10, margin: int = 2,
    probe_max_shift: "tuple[int, int]" = (50, 50),
) -> "tuple[tuple[int, int], int, dict]":
    """``max_shift`` from a histogram of per-frame shift estimates.

    Returns ``((dy, dx), border_px, evidence)`` — the 99th-percentile absolute
    shift on each axis plus a ``margin`` px safety cap; ``border_px`` is the max
    of the two (safe to trim ``warpAffine`` fill artefacts).
    """
    T_sample = sample.shape[0]
    if T_sample <= n_shift_frames:
        shift_sample = sample
    else:
        idx = np.linspace(0, T_sample - 1, n_shift_frames).astype(int)
        shift_sample = sample[idx]

    shifts = np.stack([
        estimate_shifts(f, median_img, upsample_factor=upsample_factor,
                        max_shift=probe_max_shift, gSig_filt=gSig_filt)
        for f in shift_sample
    ])
    abs_dy = np.abs(shifts[:, 0])
    abs_dx = np.abs(shifts[:, 1])
    p99_dy = float(np.percentile(abs_dy, 99))
    p99_dx = float(np.percentile(abs_dx, 99))
    max_shift_y = int(np.ceil(p99_dy)) + margin
    max_shift_x = int(np.ceil(p99_dx)) + margin
    border_px = max(max_shift_y, max_shift_x)
    evidence = {"abs_dy": abs_dy, "abs_dx": abs_dx,
                "p99_dy": p99_dy, "p99_dx": p99_dx, "gSig_filt": gSig_filt}
    return (max_shift_y, max_shift_x), border_px, evidence


def suggest_downsample(
    sigma_native: float, frame_rate_hz: float, decay_time_ms: float, *,
    min_fwhm: float = 4.0, max_ssub: int = 4, max_tsub: int = 5,
) -> "tuple[int, int, dict]":
    """``ssub`` / ``tsub`` from two simple rules.

    ``ssub``: keep neuron FWHM ≥ ``min_fwhm`` px on the binned grid.
    ``tsub``: keep the binned frame period ≤ ``decay_time_ms / 2`` (sample the
    rising edge ≥ twice). Returns ``(ssub, tsub, evidence)`` with per-candidate
    tables for the figure.
    """
    fwhm_native = 2.355 * sigma_native
    dt_ms = 1000.0 / frame_rate_hz

    ssub_rows, ssub_ok = [], []
    for ssub in range(1, max_ssub + 1):
        fwhm_ds = fwhm_native / ssub
        ok = fwhm_ds >= min_fwhm
        ssub_rows.append((ssub, fwhm_ds, ok))
        if ok:
            ssub_ok.append(ssub)
    ssub_suggested = max(ssub_ok) if ssub_ok else 1

    tsub_rows, tsub_ok = [], []
    for tsub in range(1, max_tsub + 1):
        dt_ds = dt_ms * tsub
        ok = dt_ds <= decay_time_ms / 2
        tsub_rows.append((tsub, dt_ds, dt_ds / decay_time_ms, ok))
        if ok:
            tsub_ok.append(tsub)
    tsub_suggested = max(tsub_ok) if tsub_ok else 1

    evidence = {
        "fwhm_native": fwhm_native, "dt_ms": dt_ms,
        "decay_time_ms": decay_time_ms, "frame_rate_hz": frame_rate_hz,
        "ssub_rows": ssub_rows, "tsub_rows": tsub_rows,
    }
    return int(ssub_suggested), int(tsub_suggested), evidence


# ---------------------------------------------------------------------------
# Stage 3 — initialisation (operates on an mc.zarr sample)
# ---------------------------------------------------------------------------


def suggest_sigma_extraction(
    mc_sample: np.ndarray, sigma_seed: float, *, center_psf: bool = True,
    n_jobs: int = 1, top_n: int = 30,
) -> "tuple[float, np.ndarray, np.ndarray, dict]":
    """Extraction-grid ``sigma`` from ``blob_log`` on the CORR·PNR product.

    Returns ``(sigma_refit, cn, pnr, evidence)``. ``cn``/``pnr`` are returned so
    the caller can reuse them for ``suggest_corr_pnr`` (avoids recomputing).
    """
    from skimage.feature import blob_log

    cn, pnr = correlation_pnr(mc_sample, sigma=sigma_seed,
                              center_psf=center_psf, n_jobs=n_jobs)
    product = cn * pnr
    product_n = (product - product.min()) / (product.max() - product.min() + 1e-8)
    blobs = blob_log(product_n, min_sigma=2, max_sigma=12, num_sigma=10,
                     threshold=0.05)
    if len(blobs) == 0:
        sigma_refit = float(sigma_seed)
        order = np.empty(0, dtype=int)
    else:
        scores = product_n[blobs[:, 0].astype(int), blobs[:, 1].astype(int)]
        order = np.argsort(-scores)[:top_n]
        sigma_refit = float(np.median(blobs[order, 2]))
    evidence = {"cn": cn, "pnr": pnr, "product": product,
                "blobs": blobs, "order": order, "sigma_refit": sigma_refit}
    return sigma_refit, cn, pnr, evidence


def _celllike_curve(img, thr_axis, a_min, a_max, min_solidity):
    """Per-threshold ``(#cell-like components, largest-CC fraction)`` for one image.

    A cell-like component is connected, soma-sized (``a_min ≤ area ≤ a_max``) and
    compact (``solidity > min_solidity``). ``largest_frac`` = largest component area
    ÷ total foreground — near 1 while one background mesh dominates, dropping once it
    fragments into blobs.
    """
    from skimage.measure import label, regionprops

    n_cell = np.zeros(len(thr_axis), dtype=int)
    largest_frac = np.zeros(len(thr_axis), dtype=float)
    for k, t in enumerate(thr_axis):
        mask = img > t
        fg = int(mask.sum())
        if fg == 0:
            continue
        props = regionprops(label(mask))
        areas = np.fromiter((p.area for p in props), dtype=float, count=len(props))
        largest_frac[k] = float(areas.max() / fg)
        n_cell[k] = sum(1 for p in props
                        if a_min <= p.area <= a_max and p.solidity > min_solidity)
    return n_cell, largest_frac


def suggest_corr_pnr(
    cn: np.ndarray, pnr: np.ndarray, sigma: float, *,
    corr_floor: float = 0.4, pnr_floor: float = 2.0, n_thr: int = 30,
    min_solidity: float = 0.85,
) -> "tuple[float, float, dict]":
    """``min_corr`` / ``min_pnr`` by image-threshold morphology.

    Mirrors manual CaImAn tuning: raise each image's threshold (``vmin``) until the
    diffuse background mesh fragments and only compact cell-blobs remain. For the CORR
    and PNR images independently, sweep the threshold and count **cell-like** connected
    components (soma-sized area from ``sigma``, ``solidity > min_solidity``); pick the
    threshold that maximises that count — most blobs visible = background gone, cells
    not yet lost. ``min_corr`` is read off the CORR image, ``min_pnr`` off the PNR
    image (independent, like the two sliders). Returns ``(min_corr, min_pnr, evidence)``.
    """
    import math

    a_min = max(3, int(0.5 * math.pi * sigma ** 2))
    a_max = max(a_min + 1, int(math.pi * (3.0 * sigma) ** 2))
    corr_axis = np.linspace(corr_floor, 0.95, n_thr)
    pnr_hi = float(np.percentile(pnr, 99.5))
    pnr_axis = np.linspace(pnr_floor, max(pnr_floor + 1.0, pnr_hi), n_thr)

    nc_corr, lf_corr = _celllike_curve(cn, corr_axis, a_min, a_max, min_solidity)
    nc_pnr, lf_pnr = _celllike_curve(pnr, pnr_axis, a_min, a_max, min_solidity)

    # argmax ties -> lowest threshold (more permissive). Fall back to safe defaults
    # when no cell-like blob is ever found (degenerate / empty image).
    min_corr = float(corr_axis[int(np.argmax(nc_corr))]) if nc_corr.max() > 0 else 0.8
    min_pnr = float(pnr_axis[int(np.argmax(nc_pnr))]) if nc_pnr.max() > 0 else 10.0

    evidence = {"cn": cn, "pnr": pnr, "sigma": float(sigma),
                "corr_axis": corr_axis, "pnr_axis": pnr_axis,
                "ncell_corr": nc_corr, "ncell_pnr": nc_pnr,
                "largest_frac_corr": lf_corr, "largest_frac_pnr": lf_pnr,
                "a_min": a_min, "a_max": a_max,
                "min_corr": min_corr, "min_pnr": min_pnr}
    return min_corr, min_pnr, evidence


def _neuron_bg_values(cn, pnr, sigma, *, bg_radius_factor=2.0, max_bg=5000):
    """Split CORR/PNR pixels into neuron-centre values and background values.

    Detects neuron blobs on the CORR·PNR product (the shared
    :func:`tuning.metrics.detect_product_blobs`); background = pixels at least
    ``bg_radius_factor*sigma`` px from every neuron. Returns
    ``(rc, corr_neuron, pnr_neuron, corr_bg, pnr_bg)`` with a deterministically
    strided background subsample for cheap thresholding.
    """
    from scipy.ndimage import binary_dilation
    from skimage.morphology import disk

    from tuning.metrics import detect_product_blobs

    cn = np.asarray(cn, dtype=np.float64)
    pnr = np.asarray(pnr, dtype=np.float64)
    blobs = detect_product_blobs(cn, pnr, sigma)
    rc = blobs.astype(int) if len(blobs) else np.zeros((0, 2), int)
    centre_mask = np.zeros(cn.shape, dtype=bool)
    if len(rc):
        centre_mask[rc[:, 0], rc[:, 1]] = True
    radius = max(1, int(round(bg_radius_factor * sigma)))
    bg_mask = ~binary_dilation(centre_mask, structure=disk(radius))
    corr_neuron = cn[rc[:, 0], rc[:, 1]] if len(rc) else np.empty(0)
    pnr_neuron = pnr[rc[:, 0], rc[:, 1]] if len(rc) else np.empty(0)
    corr_bg, pnr_bg = cn[bg_mask], pnr[bg_mask]
    if corr_bg.size > max_bg:
        step = corr_bg.size // max_bg
        corr_bg, pnr_bg = corr_bg[::step], pnr_bg[::step]
    return rc, corr_neuron, pnr_neuron, corr_bg, pnr_bg


def _separating_threshold(neuron_vals, bg_vals, thr_axis):
    """Threshold on ``thr_axis`` maximising Youden's J = TPR - FPR.

    TPR = fraction of neuron-centre values kept (``>= t``); FPR = fraction of
    background leaked. "Keep most neurons, filter background best." Returns
    ``(thr, j_curve)``.
    """
    nv = np.asarray(neuron_vals, dtype=np.float64)
    bv = np.asarray(bg_vals, dtype=np.float64)
    tpr = (nv[None, :] >= thr_axis[:, None]).mean(axis=1)
    fpr = (bv[None, :] >= thr_axis[:, None]).mean(axis=1)
    j = tpr - fpr
    return float(thr_axis[int(np.argmax(j))]), j


def suggest_corr_pnr_separation(
    cn: np.ndarray, pnr: np.ndarray, sigma: float, *,
    corr_floor: float = 0.4, pnr_floor: float = 2.0, n_thr: int = 60,
    min_blobs: int = 5,
) -> "tuple[float, float, dict]":
    """``min_corr`` / ``min_pnr`` from neuron-vs-background separation (Youden's J).

    Detects neuron blobs on the CORR·PNR image and, for the CORR and PNR images
    independently, picks the threshold that best separates the values at neuron
    centres from the background (max Youden's J). Falls back to safe defaults when
    too few neurons are detected. Returns ``(min_corr, min_pnr, evidence)``.
    """
    cn = np.asarray(cn, dtype=np.float64)
    pnr = np.asarray(pnr, dtype=np.float64)
    rc, corr_neuron, pnr_neuron, corr_bg, pnr_bg = _neuron_bg_values(cn, pnr, sigma)
    corr_axis = np.linspace(corr_floor, 0.95, n_thr)
    pnr_hi = float(np.percentile(pnr, 99.5))
    pnr_axis = np.linspace(pnr_floor, max(pnr_floor + 1.0, pnr_hi), n_thr)
    if len(rc) >= min_blobs and corr_bg.size and pnr_bg.size:
        min_corr, j_corr = _separating_threshold(corr_neuron, corr_bg, corr_axis)
        min_pnr, j_pnr = _separating_threshold(pnr_neuron, pnr_bg, pnr_axis)
        min_corr = float(max(corr_floor, min_corr))
        min_pnr = float(max(pnr_floor, min_pnr))
    else:
        min_corr, min_pnr = 0.8, 10.0
        j_corr = j_pnr = np.zeros(n_thr)
    evidence = {"cn": cn, "pnr": pnr, "sigma": float(sigma), "blob_rc": rc,
                "n_blobs": int(len(rc)), "corr_neuron": corr_neuron,
                "pnr_neuron": pnr_neuron, "corr_bg": corr_bg, "pnr_bg": pnr_bg,
                "thr_axis_corr": corr_axis, "thr_axis_pnr": pnr_axis,
                "j_corr": j_corr, "j_pnr": j_pnr,
                "min_corr": min_corr, "min_pnr": min_pnr}
    return min_corr, min_pnr, evidence


def suggest_corr_pnr_percentile(
    cn: np.ndarray, pnr: np.ndarray, sigma: float, *,
    pct: float = 25.0, corr_floor: float = 0.4, pnr_floor: float = 2.0,
    min_blobs: int = 5,
) -> "tuple[float, float, dict]":
    """``min_corr`` / ``min_pnr`` = the ``pct``-th percentile of CORR/PNR **at
    detected neuron-blob centres**.

    A simple, robust operating point: keep ~``(100-pct)%`` of detected neurons.
    Falls back to safe defaults when too few neurons are detected. Returns
    ``(min_corr, min_pnr, evidence)``.
    """
    cn = np.asarray(cn, dtype=np.float64)
    pnr = np.asarray(pnr, dtype=np.float64)
    rc, corr_neuron, pnr_neuron, _, _ = _neuron_bg_values(cn, pnr, sigma)
    if len(rc) >= min_blobs:
        min_corr = float(max(corr_floor, np.percentile(corr_neuron, pct)))
        min_pnr = float(max(pnr_floor, np.percentile(pnr_neuron, pct)))
    else:
        min_corr, min_pnr = 0.8, 10.0
    evidence = {"cn": cn, "pnr": pnr, "sigma": float(sigma), "blob_rc": rc,
                "n_blobs": int(len(rc)), "corr_neuron": corr_neuron,
                "pnr_neuron": pnr_neuron, "pct": float(pct),
                "min_corr": min_corr, "min_pnr": min_pnr}
    return min_corr, min_pnr, evidence


def suggest_min_pixel(
    mc_sample: np.ndarray, sigma: float, min_corr: float, min_pnr: float,
    dims: "tuple[int, int]", *, max_neurons: int = 200, n_jobs: int = 1,
    peak_frac: float = 0.05, pct: float = 25.0,
) -> "tuple[int, dict]":
    """``min_pixel`` from the footprint-area distribution of a fast greedy init.

    Pixel count per component = pixels above ``peak_frac`` of that footprint's
    peak (the spirit of ``threshold_footprint`` without running it). Suggested
    ``min_pixel`` = the ``pct``-th percentile. Returns ``(min_pixel, evidence)``.
    """
    H, W = dims
    A_init, _C, _C_raw, _centres = greedy_corr_pnr(
        mc_sample, sigma=float(sigma), min_corr=float(min_corr),
        min_pnr=float(min_pnr), max_neurons=max_neurons, n_jobs=n_jobs,
    )
    K = A_init.shape[1]
    if K == 0:
        return 3, {"pixel_counts": np.empty(0), "K": 0, "p25": 3, "p50": 3}
    A2 = np.asarray(A_init.todense()).reshape(H, W, K)
    pixel_counts = np.array([
        int((A2[..., k] > peak_frac * A2[..., k].max()).sum()) for k in range(K)
    ])
    p25 = int(np.percentile(pixel_counts, pct))
    p50 = int(np.percentile(pixel_counts, 50))
    evidence = {"pixel_counts": pixel_counts, "K": K, "p25": p25, "p50": p50}
    return max(1, p25), evidence


# ---------------------------------------------------------------------------
# Stage 4 — temporal / merge / eval (operates on a fitted model)
# ---------------------------------------------------------------------------


def _per_component_g_yw(model, *, ar_detrend_order: int = 0) -> np.ndarray:
    """Raw (no-prior, no-shrinkage) Yule-Walker AR(1) g per component."""
    C_raw = np.asarray(model.C_raw)
    K = C_raw.shape[0]
    g_yw = np.empty(K, dtype=np.float32)
    for k in range(K):
        g_arr, _ = estimate_ar_params(
            C_raw[k].astype(np.float64), p=1, g_prior=None, fudge_factor=1.0,
            detrend_order=ar_detrend_order,
        )
        g_yw[k] = float(g_arr[0])
    return g_yw


def suggest_decay_time(
    model, frame_rate_hz: float, *, ar_detrend_order: int = 0,
) -> "tuple[float, dict]":
    """``decay_time_ms`` = median per-component Yule-Walker τ.

    Returns ``(decay_time_ms, evidence)``; ``evidence`` carries ``g_yw`` so
    ``suggest_g_prior_weight`` can reuse it.
    """
    g_yw = _per_component_g_yw(model, ar_detrend_order=ar_detrend_order)
    g_safe = np.clip(g_yw, 1e-4, 0.9995)
    tau_frames = -1.0 / np.log(g_safe)
    tau_ms = tau_frames * 1000.0 / frame_rate_hz
    tau_med = float(np.median(tau_ms))
    evidence = {"g_yw": g_yw, "tau_ms": tau_ms, "tau_med": tau_med,
                "frame_rate_hz": frame_rate_hz}
    return float(round(tau_med)), evidence


def suggest_g_prior_weight(
    g_yw: np.ndarray, frame_rate_hz: float, decay_time_ms: float,
) -> "tuple[float, dict]":
    """``g_prior_weight`` from the spread of YW g around the physical target.

    Tight cluster (≤5% median deviation) -> 0.3; moderate (5–15%) -> 0.5;
    wide / drift-heavy -> 0.7. Returns ``(weight, evidence)``.
    """
    g_target = float(np.exp(-1.0 / (frame_rate_hz * decay_time_ms / 1000.0)))
    rel_dev = (g_yw - g_target) / g_target
    spread = float(np.median(np.abs(rel_dev)))
    if spread < 0.05:
        weight = 0.3
    elif spread < 0.15:
        weight = 0.5
    else:
        weight = 0.7
    evidence = {"g_yw": g_yw, "g_target": g_target, "spread": spread,
                "frame_rate_hz": frame_rate_hz, "decay_time_ms": decay_time_ms}
    return float(weight), evidence


def suggest_merge_thr(model) -> "tuple[float, dict]":
    """``merge_thr_corr`` from the pairwise C_raw correlation distribution.

    Suggests ``min(0.85, max(99th-pct, 0.7))`` so modestly-correlated real
    neighbours are not swept up. Returns ``(merge_thr_corr, evidence)``.
    """
    C_raw = np.asarray(model.C_raw)
    K = C_raw.shape[0]
    if K < 2:
        return 0.85, {"rs": np.empty(0), "p99": np.nan, "n_above": 0}
    C0 = C_raw - C_raw.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(C0, axis=1, keepdims=True) + 1e-8
    C0n = C0 / norms
    R = C0n @ C0n.T
    rs = R[np.triu_indices(K, k=1)]
    p99 = float(np.percentile(rs, 99))
    merge_thr = float(min(0.85, max(p99, 0.7)))
    evidence = {"rs": rs, "p99": p99, "n_above": int((rs > merge_thr).sum())}
    return merge_thr, evidence


def suggest_snr_thr(model) -> "tuple[float, dict]":
    """``auto_eval_snr_amp_thr`` from the largest gap in the low-SNR region.

    Real neurons score 10–70; ghosts cluster below ~2. The suggestion is the
    centre of the largest gap among components scoring < 10 (the ghost↔real
    boundary). Returns ``(snr_thr, evidence)``.
    """
    eval_info = getattr(model, "eval_info", None)
    current_thr = float(model.params.auto_eval_snr_amp_thr)
    if eval_info is None or "snr_amp" not in eval_info:
        return current_thr, {"snr": np.empty(0), "current_thr": current_thr}
    snr = np.asarray(eval_info["snr_amp"], dtype=float)
    low = np.sort(snr[snr < 10])
    if len(low) >= 2:
        gaps = np.diff(low)
        j = int(np.argmax(gaps))
        snr_suggested = float(round((low[j] + low[j + 1]) / 2, 1))
    else:
        snr_suggested = current_thr
    evidence = {"snr": snr, "current_thr": current_thr,
                "snr_suggested": snr_suggested}
    return snr_suggested, evidence
