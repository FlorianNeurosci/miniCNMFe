"""Realistic synthetic 1-photon miniscope movie generator.

Compared to ``tests/conftest.py::make_synthetic_movie``, this simulator adds
the messy features you typically see in real 1p endoscopic recordings:

- Heterogeneous neuron sizes, eccentricities, orientations, and minor
  irregularity (not all identical Gaussians).
- Heterogeneous calcium dynamics: per-neuron AR(1) decay ``g``, firing rate,
  and per-spike amplitude. Optional bursting.
- Multi-component drifting background with several spatial scales and
  temporal correlation lengths.
- "Ghost" cells: large out-of-focus blurry blobs that are NOT real neurons
  but sit in the background and modulate slowly (the most common false-
  positive source on real data).
- Vasculature pulsation at the mouse heart rate.
- Radial vignetting (corners darker than centre).
- Photobleaching: gradual exponential decay of overall fluorescence.
- Photon shot noise (Poisson) on top of Gaussian read noise.
- Optional 8-bit quantisation to mimic camera output.

The return signature is a superset of ``make_synthetic_movie`` so this
function is a drop-in replacement.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi


def make_miniscope_movie(
    n_neurons: int = 30,
    dims: tuple[int, int] = (128, 128),
    T: int = 600,
    fps: float = 20.0,
    # ---- neurons ----
    sigma_neuron_range: tuple[float, float] = (2.5, 4.5),
    eccentricity_range: tuple[float, float] = (1.0, 1.5),
    irregularity: float = 0.15,
    fire_rate_range: tuple[float, float] = (0.5, 4.0),  # Hz
    spike_amp_range: tuple[float, float] = (0.6, 2.0),
    ar_decay_range: tuple[float, float] = (0.86, 0.96),
    burst_prob: float = 0.30,
    # ---- background ----
    bg_strength: float = 4.0,
    bg_n_components: int = 5,
    bg_spatial_sigma_range: tuple[float, float] = (15.0, 35.0),
    bg_temporal_sigma: float = 30.0,  # frames (smoothing kernel)
    n_ghost_cells: int = 10,
    ghost_sigma_range: tuple[float, float] = (8.0, 15.0),
    # ---- imaging artefacts ----
    vignette_strength: float = 0.4,
    vasculature: bool = True,
    vasc_n_lines: int = 2,
    vasc_strength: float = 0.5,
    heartbeat_hz: float = 5.0,
    photobleach_tau_factor: float | None = 3.0,  # tau = factor * T (None = no bleach)
    # ---- noise / quantisation ----
    shot_noise: bool = True,
    photon_scale: float = 80.0,
    read_noise_std: float = 0.4,
    quantize_8bit: bool = True,
    # ---- motion ----
    motion_max_shift: float = 0.0,  # peak drift amplitude (pixels); 0 = no motion
    motion_seed: int | None = None,
    # ---- misc ----
    seed: int = 0,
) -> dict:
    """Generate a realistic synthetic miniscope movie.

    Returns a dict with:
        movie     : (T, H, W) float32 — observed movie
        A_true    : (H*W, K) float32 — true spatial footprints (only true neurons,
                    NOT ghost cells)
        C_true    : (K, T) float32   — true calcium traces
        S_true    : (K, T) float32   — true spike trains (with amplitudes)
        centers   : (K, 2) int       — true neuron (row, col)
        g_true    : (K,) float32     — per-neuron AR(1) decay
        sn_true   : float            — combined read noise std
        bleach    : (T,) float32     — photobleaching curve applied
        vignette  : (H, W) float32   — vignette mask applied
        background: (H*W, T) float32 — true background contribution (no neurons)
        dims      : (H, W)
    """
    rng = np.random.default_rng(seed)
    H, W = dims
    K = n_neurons
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    # -----------------------------------------------------------------------
    # Neuron centres — non-overlapping
    # -----------------------------------------------------------------------
    sig_max = sigma_neuron_range[1]
    min_sep = int(3 * sig_max) + 2
    border = int(4 * sig_max)
    centers: list[tuple[int, int]] = []
    attempts = 0
    while len(centers) < K and attempts < 20_000:
        r = int(rng.integers(border, H - border))
        c = int(rng.integers(border, W - border))
        if all(abs(r - r2) + abs(c - c2) >= min_sep for r2, c2 in centers):
            centers.append((r, c))
        attempts += 1
    K = len(centers)
    centers_arr = np.array(centers, dtype=np.int32)

    # -----------------------------------------------------------------------
    # Heterogeneous footprints — anisotropic Gaussians + mild irregularity
    # -----------------------------------------------------------------------
    A_true = np.zeros((H * W, K), dtype=np.float32)
    sigmas = rng.uniform(*sigma_neuron_range, size=K).astype(np.float32)
    eccs = rng.uniform(*eccentricity_range, size=K).astype(np.float32)
    angles = rng.uniform(0, np.pi, size=K).astype(np.float32)

    for k, ((r, c), sig, ecc, ang) in enumerate(
        zip(centers, sigmas, eccs, angles)
    ):
        sx = sig * ecc
        sy = sig / np.sqrt(ecc)
        ca, sa = np.cos(ang), np.sin(ang)
        yc = yy - r
        xc = xx - c
        yp =  ca * yc + sa * xc
        xp = -sa * yc + ca * xc
        blob = np.exp(-(yp ** 2 / (2 * sy ** 2) + xp ** 2 / (2 * sx ** 2)))
        if irregularity > 0:
            irreg = ndi.gaussian_filter(
                rng.standard_normal((H, W)).astype(np.float32) * irregularity,
                sigma=2.0,
            )
            blob = blob * (1.0 + irreg)
        blob = np.clip(blob, 0, None)
        blob /= blob.max() + 1e-10
        blob[blob < 0.05] = 0          # soft support cutoff
        A_true[:, k] = blob.ravel()

    # -----------------------------------------------------------------------
    # Calcium traces — heterogeneous AR(1), variable spike amplitudes, bursts
    # -----------------------------------------------------------------------
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

        # Bursting: each event has a chance of a smaller follow-up within 1-3 frames
        spike_idx = np.where(events)[0]
        for t in spike_idx:
            if rng.random() < burst_prob:
                dt = int(rng.integers(1, 4))
                if t + dt < T and S_true[k, t + dt] == 0:
                    S_true[k, t + dt] = (
                        S_true[k, t] * float(rng.uniform(0.3, 0.8))
                    )

        # AR(1) integration
        gk = float(g_true[k])
        for t in range(1, T):
            C_true[k, t] = gk * C_true[k, t - 1] + S_true[k, t]

    # -----------------------------------------------------------------------
    # Multi-component drifting background (slow spatial × slow temporal)
    # -----------------------------------------------------------------------
    background = np.zeros((H * W, T), dtype=np.float32)
    for _ in range(bg_n_components):
        sig_sp = float(rng.uniform(*bg_spatial_sigma_range))
        spatial = ndi.gaussian_filter(
            rng.standard_normal((H, W)).astype(np.float32), sigma=sig_sp
        )
        spatial -= spatial.min()
        spatial /= spatial.max() + 1e-10

        temporal = ndi.gaussian_filter1d(
            rng.standard_normal(T).astype(np.float32), sigma=bg_temporal_sigma
        )
        # rescale temporal to roughly unit std then random amplitude
        temporal -= temporal.mean()
        temporal /= temporal.std() + 1e-10
        temporal *= float(rng.uniform(0.3, 1.0))

        background += np.outer(spatial.ravel(), temporal).astype(np.float32)

    # -----------------------------------------------------------------------
    # "Ghost" cells — out-of-focus blurry blobs, slow temporal modulation.
    # These are the typical 1p false-positive source: they look like neurons
    # but are out of focus, so their footprint is wider and their temporal
    # structure is slow (no sharp spikes).
    # -----------------------------------------------------------------------
    for _ in range(n_ghost_cells):
        cy = int(rng.integers(0, H))
        cx = int(rng.integers(0, W))
        sig_g = float(rng.uniform(*ghost_sigma_range))
        blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sig_g ** 2))
        blob /= blob.max() + 1e-10

        # Slow random walk modulation, no spikes
        raw = rng.standard_normal(T).astype(np.float32)
        temp = ndi.gaussian_filter1d(raw, sigma=float(rng.uniform(15, 40)))
        temp -= temp.min()
        temp /= temp.max() + 1e-10
        temp *= float(rng.uniform(0.3, 0.9))

        background += np.outer(blob.ravel(), temp).astype(np.float32)

    background = bg_strength * background

    # -----------------------------------------------------------------------
    # Vasculature — bright/dark elongated lines that pulse at heartbeat rate
    # -----------------------------------------------------------------------
    if vasculature and vasc_n_lines > 0:
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

        # Pulsation at heart rate plus slow drift
        t_arr = np.arange(T, dtype=np.float32)
        pulse = np.sin(2 * np.pi * heartbeat_hz / fps * t_arr) * 0.5 + 1.0
        slow = 1.0 + 0.4 * ndi.gaussian_filter1d(
            rng.standard_normal(T).astype(np.float32), sigma=20.0
        )
        vasc_temporal = (pulse * slow).astype(np.float32)
        background += vasc_strength * np.outer(
            vasc_spatial.ravel(), vasc_temporal
        ).astype(np.float32)

    # -----------------------------------------------------------------------
    # Compose: neurons + background, then bleach, then vignette, then noise
    # -----------------------------------------------------------------------
    Y = (A_true @ C_true) + background      # (H*W, T) — all neuron+bg signal

    # Add a uniform DC baseline so subsequent multiplicative ops stay positive
    baseline_offset = 1.5
    Y = Y + baseline_offset

    # Photobleaching — multiplicative exponential decay across time
    if photobleach_tau_factor is not None:
        tau = float(photobleach_tau_factor) * T
        bleach = np.exp(-np.arange(T) / tau).astype(np.float32)
    else:
        bleach = np.ones(T, dtype=np.float32)
    Y = Y * bleach[np.newaxis, :]

    # Vignetting — multiplicative radial fall-off
    rr = np.sqrt(((yy - H / 2.0) ** 2 + (xx - W / 2.0) ** 2)
                 / (max(H, W) / 2.0) ** 2).astype(np.float32)
    vignette = (1.0 - vignette_strength * np.clip(rr, 0, 1) ** 2).astype(np.float32)
    Y = Y * vignette.ravel()[:, np.newaxis]

    # Shot noise (Poisson) plus read noise (Gaussian)
    if shot_noise:
        photons = np.maximum(Y * photon_scale, 0)
        # Poisson on float-valued photon counts
        Y = rng.poisson(photons).astype(np.float32) / photon_scale
    Y = Y + rng.standard_normal(Y.shape).astype(np.float32) * read_noise_std

    # 8-bit quantisation (mimics camera output)
    if quantize_8bit:
        lo, hi = float(Y.min()), float(Y.max())
        Y_q = np.round((Y - lo) / (hi - lo + 1e-10) * 255.0)
        Y = (Y_q / 255.0 * (hi - lo) + lo).astype(np.float32)

    # Reshape to (T, H, W) movie
    movie = Y.T.reshape(T, H, W).astype(np.float32)

    # Simulate inter-frame rigid motion via a smoothed random walk drift
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

    result = {
        "movie":         movie,
        "A_true":        A_true,
        "C_true":        C_true,
        "S_true":        S_true,
        "centers":       centers_arr,
        "g_true":        g_true,
        "sn_true":       float(read_noise_std),
        "ar_decay":      float(np.mean(g_true)),
        "bleach":        bleach,
        "vignette":      vignette,
        "background":    background,
        "motion_shifts": motion_shifts,
        "dims":          dims,
    }
    return result
