"""Realistic synthetic 1-photon miniscope movie generator.

Compared to ``tests/conftest.py::make_synthetic_movie``, this simulator adds
the messy features you typically see in real 1p endoscopic recordings AND
targets visual fidelity to a real session (`demo_movies/demo_session`):

- A **circular GRIN-lens aperture** — black outside the lens with a soft edge,
  the single most recognisable 1p signature.
- **Branching dark vasculature** — sinuous trunks with side branches that
  darken the baseline; pulse subtly at the heart rate.
- **Discrete bright neurons** — anisotropic Gaussian cores (~σ5 px) with a dim
  out-of-focus halo; their transients are large enough relative to the
  background that they show up as discrete bright dots in a std projection.
- **Calcium dynamics from a real indicator** (GCaMP8m by default) via
  ``decay_time_ms``; physically meaningful AR(1) ``g`` from
  ``cnmfe.temporal.g_from_decay_time``.
- **Modest active neuropil** at fine spatial scale — adds the slow textured
  background of real data without swamping the cells.
- Multi-component slow drift + ghost cells (out-of-focus blurry distractors,
  the classic 1p false-positive source).
- A large positive uint8-like baseline (``F0``) with small ΔF/F on top.
- Radial vignette, exponential photobleaching, intensity-dependent (Poisson)
  shot noise + Gaussian read noise (optionally with a spatially-varying
  fixed-pattern gain), and 8-bit quantisation.

Realism is **on by default**, with a single ``difficulty ∈ [0, 1]`` knob and a
``realism=False`` kill-switch that restores the legacy low-DC fixture
(bit-for-bit: every new rng draw is guarded behind a flag).
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi


# ---------------------------------------------------------------------------
# Visual realism helpers (circular aperture + branching vasculature)
# ---------------------------------------------------------------------------

def _circular_aperture(H: int, W: int, radius_frac: float, edge_px: float) -> np.ndarray:
    """Soft-edge circular mask: 1 inside the GRIN lens, 0 outside, smooth at the
    rim over ``edge_px`` pixels."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    R = radius_frac * min(H, W) / 2.0
    # smoothstep from 1 (r<=R-edge) to 0 (r>=R)
    t = np.clip((R - r) / max(edge_px, 1e-3), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _make_branching_vasculature(
    H: int, W: int, rng: np.random.Generator,
    n_main: int = 3, max_depth: int = 2, base_width: float = 2.6,
) -> np.ndarray:
    """Build a (H, W) [0, 1] vessel mask: sinuous random-walk trunks with
    angled side branches, drawn as fading Gaussian blobs along the path so the
    edges look soft and natural (no aliased lines)."""
    mask = np.zeros((H, W), dtype=np.float32)

    # Reusable small Gaussian stamp (rebuilt per width).
    def _stamp(width: float) -> np.ndarray:
        rad = int(max(2, 2.5 * width))
        ay, ax = np.mgrid[-rad:rad + 1, -rad:rad + 1]
        return np.exp(-(ay ** 2 + ax ** 2) / (2.0 * width ** 2)).astype(np.float32)

    def _draw_segment(y: float, x: float, brightness: float, stamp: np.ndarray) -> None:
        yi, xi = int(round(y)), int(round(x))
        rad = stamp.shape[0] // 2
        y0, y1 = max(0, yi - rad), min(H, yi + rad + 1)
        x0, x1 = max(0, xi - rad), min(W, xi + rad + 1)
        if y0 >= y1 or x0 >= x1:
            return
        sy0 = rad - (yi - y0); sy1 = sy0 + (y1 - y0)
        sx0 = rad - (xi - x0); sx1 = sx0 + (x1 - x0)
        sub = mask[y0:y1, x0:x1]
        np.maximum(sub, brightness * stamp[sy0:sy1, sx0:sx1], out=sub)

    def _walk(y: float, x: float, dy: float, dx: float,
              length: int, width: float, depth: int) -> None:
        stamp = _stamp(width)
        brightness = float(np.clip(width / base_width, 0.4, 1.0))
        for s in range(length):
            if not (0 <= y < H and 0 <= x < W):
                return
            _draw_segment(y, x, brightness, stamp)
            # gently turn (smoothed random walk)
            ang = rng.normal(0.0, 0.18)
            ca, sa = np.cos(ang), np.sin(ang)
            ndy, ndx = ca * dy - sa * dx, sa * dy + ca * dx
            n = np.hypot(ndy, ndx)
            dy, dx = ndy / n, ndx / n
            y += dy; x += dx
            # spawn a side branch occasionally
            if depth < max_depth and length - s > 12 and rng.random() < 0.045:
                bang = rng.choice([-1.0, 1.0]) * float(np.pi / 3.0) * rng.uniform(0.6, 1.2)
                ca, sa = np.cos(bang), np.sin(bang)
                bdy, bdx = ca * dy - sa * dx, sa * dy + ca * dx
                _walk(y, x, bdy, bdx, max(8, (length - s) // 2),
                      width * rng.uniform(0.45, 0.7), depth + 1)

    for _ in range(n_main):
        # Start near a border, point inward.
        side = int(rng.integers(0, 4))
        if side == 0:
            y, x = 0.0, float(rng.integers(0, W)); dy, dx = 1.0, 0.0
        elif side == 1:
            y, x = float(H - 1), float(rng.integers(0, W)); dy, dx = -1.0, 0.0
        elif side == 2:
            y, x = float(rng.integers(0, H)), 0.0; dy, dx = 0.0, 1.0
        else:
            y, x = float(rng.integers(0, H)), float(W - 1); dy, dx = 0.0, -1.0
        # add a small initial angle off the inward normal
        dy += float(rng.normal(0.0, 0.35)); dx += float(rng.normal(0.0, 0.35))
        n = np.hypot(dy, dx); dy, dx = dy / n, dx / n
        length = int(min(H, W) * float(rng.uniform(0.6, 1.05)))
        width = float(rng.uniform(base_width * 0.7, base_width * 1.1))
        _walk(y, x, dy, dx, length, width, depth=0)

    # Soften further to avoid stairstepping at very thin widths.
    return ndi.gaussian_filter(mask, sigma=0.6).clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def make_miniscope_movie(
    n_neurons: int = 30,
    dims: tuple[int, int] = (128, 128),
    T: int = 600,
    fps: float = 20.0,
    # ---- master realism controls ----
    realism: bool = True,        # False => legacy behaviour (bit-for-bit)
    difficulty: float = 0.0,     # 0 = calibrated, 1 = harder (realism only)
    # ---- neurons ----
    sigma_neuron_range: tuple[float, float] | None = None,  # None => realism-dependent
    eccentricity_range: tuple[float, float] = (1.0, 1.5),
    irregularity: float = 0.07,
    fire_rate_range: tuple[float, float] = (0.5, 4.0),  # Hz
    spike_amp_range: tuple[float, float] = (0.6, 2.0),
    decay_time_ms: float | None = 180.0,  # indicator decay τ; default GCaMP8m
    decay_time_jitter: float = 0.10,      # per-neuron fractional spread on τ
    ar_decay_range: tuple[float, float] = (0.86, 0.96),  # legacy: used iff decay_time_ms is None
    burst_prob: float = 0.30,
    neuron_gain: float | None = None,     # ΔF amplitude of foreground cells (realism)
    # ---- defocused haloes ----
    neuron_halo: bool | None = None,      # None => realism
    halo_sigma_factor: float = 2.2,       # tighter halo so cells stay compact
    halo_amp: float = 0.04,
    # ---- subtle active neuropil (fine texture, not a dominant cloud) ----
    npil_active: bool | None = None,      # None => realism
    npil_n_components: int = 4,
    npil_field_sigma_range: tuple[float, float] = (1.5, 5.0),
    npil_fire_rate: float = 4.0,          # Hz per component
    npil_spike_amp_range: tuple[float, float] = (0.3, 1.0),
    npil_temporal_smooth: float = 2.5,    # frames
    npil_strength: float = 1.8,           # MUCH lower than before — texture, not cloud
    npil_neuron_coupling: float = 0.04,   # very small contamination skirt
    # ---- background ----
    bg_strength: float = 4.0,
    bg_n_components: int = 5,
    bg_spatial_sigma_range: tuple[float, float] = (15.0, 35.0),
    bg_temporal_sigma: float = 30.0,
    n_ghost_cells: int = 8,
    ghost_sigma_range: tuple[float, float] = (8.0, 15.0),
    # ---- baseline / scale (F0) ----
    realism_baseline: bool | None = None,
    f0_base: float = 42.0,
    # ---- GRIN aperture ----
    aperture: bool | None = None,         # None => realism
    aperture_radius_factor: float = 0.96,
    aperture_edge_px: float = 2.5,
    # ---- vasculature ----
    vasculature: bool = True,
    vasc_realism: bool | None = None,     # None => realism (branching) else legacy 2-line
    vasc_n_lines: int = 2,                # legacy param (only used when vasc_realism=False)
    vasc_n_main: int = 3,                 # realism branching trunks
    vasc_darkness: float = 0.55,          # fraction baseline is darkened by vessels
    vasc_pulse_amp: float = 0.07,         # heart-rate brightness modulation on vessels
    heartbeat_hz: float = 5.0,
    vasc_strength: float = 0.5,           # legacy path only
    # ---- imaging artefacts ----
    vignette_strength: float = 0.4,
    photobleach_tau_factor: float | None = 3.0,
    # ---- noise / quantisation ----
    shot_noise: bool = True,
    photon_scale: float | None = None,    # None => realism-dependent
    read_noise_std: float = 0.4,
    read_noise_fixed_pattern: bool | None = None,
    noise_cv: float = 0.4,
    noise_field_sigma: float = 8.0,
    quantize_8bit: bool = True,
    # ---- motion ----
    motion_max_shift: float = 0.0,
    motion_seed: int | None = None,
    # ---- misc ----
    seed: int = 0,
) -> dict:
    """Generate a realistic synthetic miniscope movie. See module docstring.

    Returns a dict with:
        movie     : (T, H, W) float32 — observed movie (uint8-scale in realism)
        A_true    : (H*W, K) float32 — true neuron CORES only (no haloes, no
                    neuropil, no ghosts)
        C_true    : (K, T) float32   — true calcium traces
        S_true    : (K, T) float32   — true spike trains (with amplitudes)
        centers   : (K, 2) int       — true neuron (row, col)
        g_true    : (K,) float32     — per-neuron AR(1) decay
        decay_time_ms : float | None — indicator decay τ used
        sn_true   : float            — representative read-noise std
        bleach    : (T,) float32     — photobleaching curve
        vignette  : (H, W) float32   — vignette mask
        aperture  : (H, W) float32   — circular GRIN mask applied (1 inside, 0
                    outside) — None on the legacy path
        vasc_mask : (H, W) float32   — vessel intensity map (1 = vessel) — None
                    on the legacy path
        background: (H*W, T) float32 — all nuisance ΔF (neuropil + haloes +
                    contamination + slow drift + ghosts + vasc pulsation)
        dims      : (H, W)
    """
    rng = np.random.default_rng(seed)
    H, W = dims
    K = n_neurons
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    # -------------------- resolve realism-dependent defaults --------------------
    def _resolve(flag: bool | None) -> bool:
        return realism if flag is None else flag

    npil_active = _resolve(npil_active)
    neuron_halo = _resolve(neuron_halo)
    realism_baseline = _resolve(realism_baseline)
    read_noise_fixed_pattern = _resolve(read_noise_fixed_pattern)
    aperture_on = _resolve(aperture)
    vasc_branching = _resolve(vasc_realism)
    if sigma_neuron_range is None:
        sigma_neuron_range = (4.0, 6.5) if realism else (2.5, 4.5)
    if photon_scale is None:
        photon_scale = 25.0 if realism else 80.0
    if neuron_gain is None:
        neuron_gain = 2.6 if realism else 1.0

    # Difficulty knob (realism only): make neurons dimmer, neuropil stronger,
    # photons fewer (more noise), and add more ghost distractors.
    d = float(np.clip(difficulty, 0.0, 1.0)) if realism else 0.0
    npil_strength *= (1.0 + 2.0 * d)
    npil_neuron_coupling *= (1.0 + 2.0 * d)
    neuron_gain *= (1.0 - 0.4 * d)
    photon_scale = photon_scale / (1.0 + 3.0 * d)
    n_ghost_cells = int(round(n_ghost_cells * (1.0 + d)))

    # -------------------- neuron centres (non-overlapping, inside aperture) ----
    sig_max = sigma_neuron_range[1]
    min_sep = int(3 * sig_max) + 2
    border = int((3 if realism else 4) * sig_max)
    # If the GRIN aperture is on, restrict placement to a disk just inside it
    # (a margin of sig_max so neuron footprints stay fully visible).
    cy_c, cx_c = (H - 1) / 2.0, (W - 1) / 2.0
    R_place = (aperture_radius_factor * min(H, W) / 2.0 - 1.5 * sig_max) if aperture_on else float("inf")
    centers: list[tuple[int, int]] = []
    attempts = 0
    while len(centers) < K and attempts < 20_000:
        r = int(rng.integers(border, H - border))
        c = int(rng.integers(border, W - border))
        if aperture_on and (r - cy_c) ** 2 + (c - cx_c) ** 2 > R_place ** 2:
            attempts += 1
            continue
        if all(abs(r - r2) + abs(c - c2) >= min_sep for r2, c2 in centers):
            centers.append((r, c))
        attempts += 1
    K = len(centers)
    centers_arr = np.array(centers, dtype=np.int32)

    # -------------------- footprints (cores + haloes) --------------------------
    A_true = np.zeros((H * W, K), dtype=np.float32)
    A_halo = np.zeros((H * W, K), dtype=np.float32) if (neuron_halo or npil_active) else None
    sigmas = rng.uniform(*sigma_neuron_range, size=K).astype(np.float32)
    eccs = rng.uniform(*eccentricity_range, size=K).astype(np.float32)
    angles = rng.uniform(0, np.pi, size=K).astype(np.float32)

    for k, ((r, c), sig, ecc, ang) in enumerate(zip(centers, sigmas, eccs, angles)):
        sx = sig * ecc
        sy = sig / np.sqrt(ecc)
        ca, sa = np.cos(ang), np.sin(ang)
        yc = yy - r; xc = xx - c
        yp =  ca * yc + sa * xc
        xp = -sa * yc + ca * xc
        blob = np.exp(-(yp ** 2 / (2 * sy ** 2) + xp ** 2 / (2 * sx ** 2)))
        if irregularity > 0:
            irreg = ndi.gaussian_filter(
                rng.standard_normal((H, W)).astype(np.float32) * irregularity, sigma=2.0,
            )
            blob = blob * (1.0 + irreg)
        blob = np.clip(blob, 0, None)
        blob /= blob.max() + 1e-10
        blob[blob < 0.05] = 0
        A_true[:, k] = blob.ravel()

        if A_halo is not None:
            hs = float(halo_sigma_factor * sig)
            halo = np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / (2 * hs ** 2))
            halo /= halo.max() + 1e-10
            A_halo[:, k] = halo.ravel()

    # -------------------- calcium traces (AR(1)) -------------------------------
    if decay_time_ms is not None:
        from cnmfe.temporal import g_from_decay_time
        tau_k = decay_time_ms * (1.0 + decay_time_jitter * rng.uniform(-1.0, 1.0, size=K))
        tau_k = np.clip(tau_k, 1.0, None)
        g_true = np.array(
            [g_from_decay_time(float(t), fps) for t in tau_k], dtype=np.float32
        )
    else:
        g_true = rng.uniform(*ar_decay_range, size=K).astype(np.float32)
    fire_rates = rng.uniform(*fire_rate_range, size=K) / fps   # spikes/frame

    S_true = np.zeros((K, T), dtype=np.float32)
    C_true = np.zeros((K, T), dtype=np.float32)
    for k in range(K):
        rate_k = float(fire_rates[k])
        events = rng.random(T) < rate_k
        events[0] = False
        amps = rng.uniform(*spike_amp_range, size=T).astype(np.float32)
        S_true[k] = events * amps
        spike_idx = np.where(events)[0]
        for t in spike_idx:
            if rng.random() < burst_prob:
                dt = int(rng.integers(1, 4))
                if t + dt < T and S_true[k, t + dt] == 0:
                    S_true[k, t + dt] = S_true[k, t] * float(rng.uniform(0.3, 0.8))
        gk = float(g_true[k])
        for t in range(1, T):
            C_true[k, t] = gk * C_true[k, t - 1] + S_true[k, t]

    # -------------------- background nuisance (ΔF, around 0) -------------------
    background = np.zeros((H * W, T), dtype=np.float32)

    # (a) Subtle active neuropil — fine texture, not a dominant cloud.
    if npil_active and npil_n_components > 0:
        from scipy.signal import lfilter
        from cnmfe.temporal import g_from_decay_time as _g_from_tau
        g_npil = (_g_from_tau(decay_time_ms, fps) if decay_time_ms is not None
                  else float(np.mean(g_true)))
        for _ in range(npil_n_components):
            sig_sp = float(rng.uniform(*npil_field_sigma_range))
            field = ndi.gaussian_filter(
                rng.standard_normal((H, W)).astype(np.float32), sigma=sig_sp
            )
            field -= field.min(); field /= field.max() + 1e-10
            ev = rng.random(T) < (npil_fire_rate / fps)
            amp = rng.uniform(*npil_spike_amp_range, size=T).astype(np.float32)
            trace = lfilter([1.0], [1.0, -g_npil], ev * amp).astype(np.float32)
            if npil_temporal_smooth > 0:
                trace = ndi.gaussian_filter1d(trace, sigma=npil_temporal_smooth)
            trace -= trace.mean()
            background += npil_strength * np.outer(field.ravel(), trace).astype(np.float32)

    # (b) Per-neuron surround (halo + light contamination, ΔF tracking C_true).
    if A_halo is not None:
        coef = (halo_amp if neuron_halo else 0.0) + \
               (npil_neuron_coupling if npil_active else 0.0)
        if coef > 0:
            background += coef * (A_halo @ C_true).astype(np.float32)

    # (c) Multi-component slow drifting background (zero-mean temporal).
    for _ in range(bg_n_components):
        sig_sp = float(rng.uniform(*bg_spatial_sigma_range))
        spatial = ndi.gaussian_filter(
            rng.standard_normal((H, W)).astype(np.float32), sigma=sig_sp
        )
        spatial -= spatial.min(); spatial /= spatial.max() + 1e-10
        temporal = ndi.gaussian_filter1d(
            rng.standard_normal(T).astype(np.float32), sigma=bg_temporal_sigma
        )
        temporal -= temporal.mean(); temporal /= temporal.std() + 1e-10
        temporal *= float(rng.uniform(0.3, 1.0))
        background += bg_strength * np.outer(spatial.ravel(), temporal).astype(np.float32)

    # (d) Ghost cells (slow blurry distractors).
    for _ in range(n_ghost_cells):
        cy = int(rng.integers(0, H))
        cx = int(rng.integers(0, W))
        sig_g = float(rng.uniform(*ghost_sigma_range))
        blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sig_g ** 2))
        blob /= blob.max() + 1e-10
        raw = rng.standard_normal(T).astype(np.float32)
        temp = ndi.gaussian_filter1d(raw, sigma=float(rng.uniform(15, 40)))
        temp -= temp.min(); temp /= temp.max() + 1e-10
        temp *= float(rng.uniform(0.3, 0.9))
        if realism_baseline:
            temp = temp - temp.mean()
        background += bg_strength * np.outer(blob.ravel(), temp).astype(np.float32)

    # -------------------- vasculature (realism = branching darkening) ----------
    vasc_mask_out: np.ndarray | None = None
    if vasculature and vasc_branching:
        vasc_mask = _make_branching_vasculature(H, W, rng, n_main=vasc_n_main)
        vasc_mask_out = vasc_mask
        # Heartbeat pulsation as a small zero-mean brightness modulation on vessels.
        t_arr = np.arange(T, dtype=np.float32)
        pulse = (np.sin(2 * np.pi * heartbeat_hz / fps * t_arr) * 0.5).astype(np.float32)
        # Sign: vessels get slightly less dark on a beat (more blood flow -> bright).
        background += vasc_pulse_amp * f0_base * np.outer(vasc_mask.ravel(), pulse).astype(np.float32)
    elif vasculature and vasc_n_lines > 0:
        # Legacy 2-line vasculature (only used when realism=False or explicit opt-out).
        vasc_spatial = np.zeros((H, W), dtype=np.float32)
        for _ in range(vasc_n_lines):
            ang = float(rng.uniform(0, np.pi))
            width = float(rng.uniform(1.5, 3.0))
            offset = float(rng.uniform(-min(H, W) / 3, min(H, W) / 3))
            perp = (np.cos(ang) * (xx - W / 2.0)
                    - np.sin(ang) * (yy - H / 2.0)
                    - offset)
            line = np.exp(-perp ** 2 / (2 * width ** 2))
            sign = 1.0 if rng.random() < 0.5 else -1.0
            vasc_spatial += sign * line * float(rng.uniform(0.6, 1.0))
        t_arr = np.arange(T, dtype=np.float32)
        pulse_legacy = np.sin(2 * np.pi * heartbeat_hz / fps * t_arr) * 0.5 + 1.0
        slow_legacy = 1.0 + 0.4 * ndi.gaussian_filter1d(
            rng.standard_normal(T).astype(np.float32), sigma=20.0
        )
        vasc_temporal = (pulse_legacy * slow_legacy).astype(np.float32)
        background += vasc_strength * np.outer(
            vasc_spatial.ravel(), vasc_temporal
        ).astype(np.float32)

    # -------------------- compose: F0 (baseline) + ΔF, vasc-darkened ------------
    if realism_baseline:
        F0_const = np.full((H * W,), f0_base, dtype=np.float32)
        if vasc_mask_out is not None:
            F0_const = F0_const * (1.0 - vasc_darkness * vasc_mask_out.ravel())
        Y = F0_const[:, None] + neuron_gain * (A_true @ C_true) + background
    else:
        Y = (A_true @ C_true) + background + 1.5

    # -------------------- bleach × vignette (multiplicative) -------------------
    if photobleach_tau_factor is not None:
        tau = float(photobleach_tau_factor) * T
        bleach = np.exp(-np.arange(T) / tau).astype(np.float32)
    else:
        bleach = np.ones(T, dtype=np.float32)
    Y = Y * bleach[np.newaxis, :]

    rr = np.sqrt(((yy - H / 2.0) ** 2 + (xx - W / 2.0) ** 2)
                 / (max(H, W) / 2.0) ** 2).astype(np.float32)
    vignette = (1.0 - vignette_strength * np.clip(rr, 0, 1) ** 2).astype(np.float32)
    Y = Y * vignette.ravel()[:, np.newaxis]

    # -------------------- Poisson + read noise ---------------------------------
    if shot_noise:
        photons = np.maximum(Y * photon_scale, 0)
        Y = rng.poisson(photons).astype(np.float32) / photon_scale

    if read_noise_fixed_pattern:
        gain = ndi.gaussian_filter(
            rng.standard_normal((H, W)).astype(np.float32), sigma=noise_field_sigma
        )
        gain = 1.0 + noise_cv * (gain - gain.mean()) / (gain.std() + 1e-10)
        gain = np.clip(gain, 0.1, None).ravel()[:, np.newaxis]
    else:
        gain = 1.0
    Y = Y + rng.standard_normal(Y.shape).astype(np.float32) * read_noise_std * gain

    # -------------------- circular GRIN aperture (last, before quant) ---------
    aperture_out: np.ndarray | None = None
    if aperture_on:
        ap = _circular_aperture(H, W, aperture_radius_factor, aperture_edge_px)
        aperture_out = ap
        Y = Y * ap.ravel()[:, np.newaxis]

    # -------------------- 8-bit quantisation ----------------------------------
    if quantize_8bit:
        if realism_baseline:
            Y = np.clip(np.round(Y), 0.0, 255.0).astype(np.float32)
        else:
            lo, hi = float(Y.min()), float(Y.max())
            Y_q = np.round((Y - lo) / (hi - lo + 1e-10) * 255.0)
            Y = (Y_q / 255.0 * (hi - lo) + lo).astype(np.float32)

    movie = Y.T.reshape(T, H, W).astype(np.float32)

    # -------------------- inter-frame motion (optional) -----------------------
    if motion_max_shift > 0:
        from scipy.ndimage import uniform_filter1d
        from cnmfe.motion_correction import apply_shift as _apply_shift
        rng_m = np.random.default_rng(seed + 1 if motion_seed is None else motion_seed)
        steps = rng_m.normal(0, motion_max_shift / 10, size=(T, 2)).astype(np.float64)
        drift = np.cumsum(steps, axis=0)
        drift = uniform_filter1d(drift, size=max(1, T // 20), axis=0)
        peak = np.abs(drift).max()
        if peak > 0:
            drift = drift * (motion_max_shift / peak)
        drift = drift.astype(np.float32)
        movie = np.stack([_apply_shift(movie[t], drift[t]) for t in range(T)],
                         axis=0).astype(np.float32)
        motion_shifts = drift
    else:
        motion_shifts = np.zeros((T, 2), dtype=np.float32)

    return {
        "movie":         movie,
        "A_true":        A_true,
        "C_true":        C_true,
        "S_true":        S_true,
        "centers":       centers_arr,
        "g_true":        g_true,
        "decay_time_ms": decay_time_ms,
        "sn_true":       float(read_noise_std),
        "ar_decay":      float(np.mean(g_true)),
        "bleach":        bleach,
        "vignette":      vignette,
        "aperture":      aperture_out,
        "vasc_mask":     vasc_mask_out,
        "background":    background,
        "motion_shifts": motion_shifts,
        "dims":          dims,
    }
