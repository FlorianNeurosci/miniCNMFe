"""Synthetic ground-truth movie generator for all tests."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.ndimage as ndi
import scipy.sparse as sp


def make_synthetic_movie(
    n_neurons: int = 6,
    dims: tuple[int, int] = (64, 64),
    T: int = 300,
    noise_std: float = 0.5,
    ar_decay: float = 0.9,
    bg_strength: float = 1.5,
    sigma_neuron: float = 3.0,
    seed: int = 42,
) -> dict:
    """Generate a synthetic 1-photon calcium imaging movie with known ground truth.

    Returns a dict with:
        movie      : (T, H, W) float32 — raw movie with background and noise
        A_true     : (H*W, K) float32 — true spatial footprints
        C_true     : (K, T) float32   — true calcium traces (AR1 dynamics)
        S_true     : (K, T) float32   — true spike trains (Poisson)
        centers    : (K, 2) int       — true neuron centres (row, col)
        sn_true    : float            — true noise std per pixel
        ar_decay   : float            — AR(1) decay constant used
    """
    rng = np.random.default_rng(seed)
    H, W = dims
    K = n_neurons

    # --- Neuron centres (avoid borders) ---
    min_sep = int(3 * sigma_neuron) + 2
    border = int(4 * sigma_neuron)
    centers: list[tuple[int, int]] = []
    attempts = 0
    while len(centers) < K and attempts < 10_000:
        r = rng.integers(border, H - border)
        c = rng.integers(border, W - border)
        ok = all(abs(r - r2) + abs(c - c2) >= min_sep for r2, c2 in centers)
        if ok:
            centers.append((int(r), int(c)))
        attempts += 1
    K = len(centers)

    # --- Spatial footprints (Gaussian blobs) ---
    yy, xx = np.mgrid[0:H, 0:W]
    A_true = np.zeros((H * W, K), dtype=np.float32)
    for k, (r, c) in enumerate(centers):
        blob = np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / (2 * sigma_neuron ** 2))
        blob /= blob.max() + 1e-10
        A_true[:, k] = blob.ravel()

    # --- Spike trains (Poisson) ---
    fire_rate = 0.05  # spikes per frame
    S_true = (rng.random((K, T)) < fire_rate).astype(np.float32)
    S_true[:, 0] = 0

    # --- Calcium traces (AR1 decay) ---
    C_true = np.zeros((K, T), dtype=np.float32)
    for t in range(1, T):
        C_true[:, t] = ar_decay * C_true[:, t - 1] + S_true[:, t]

    # --- Ring-like background (spatially smooth, temporally correlated) ---
    bg_spatial = ndi.gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), sigma=10)
    bg_spatial = (bg_spatial - bg_spatial.min()) / (bg_spatial.max() - bg_spatial.min() + 1e-10)
    bg_temporal = np.cumsum(rng.standard_normal(T).astype(np.float32)) * 0.01
    bg_temporal -= bg_temporal.mean()
    background = np.outer(bg_spatial.ravel(), bg_temporal)  # (H*W, T)

    # --- Observed movie ---
    Y_flat = A_true @ C_true + bg_strength * background  # (H*W, T)
    Y_flat += rng.standard_normal(Y_flat.shape).astype(np.float32) * noise_std
    movie = Y_flat.T.reshape(T, H, W)  # (T, H, W)

    return {
        "movie": movie.astype(np.float32),
        "A_true": A_true,
        "C_true": C_true,
        "S_true": S_true,
        "centers": np.array(centers, dtype=np.int32),
        "sn_true": float(noise_std),
        "ar_decay": float(ar_decay),
        "dims": dims,
    }


@pytest.fixture
def synth():
    """Small synthetic movie (6 neurons, 64×64, 300 frames)."""
    return make_synthetic_movie()


@pytest.fixture
def synth_small():
    """Tiny synthetic movie for fast tests (3 neurons, 32×32, 150 frames)."""
    return make_synthetic_movie(n_neurons=3, dims=(32, 32), T=150, seed=0)
