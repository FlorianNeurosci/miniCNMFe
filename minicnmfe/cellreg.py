"""Cross-session cell registration (CellReg-style).

Clean Python reimplementation of the Ziv lab's CellReg algorithm
(Sheintuch et al., 2017, *Cell Reports* — "Tracking the Same Neurons across
Multiple Days in Ca2+ Imaging Data") for tracking the same neurons across
multiple imaging sessions. **No MATLAB / external algorithm code is used** —
everything is numpy / scipy / sklearn, matching this package's from-scratch
ethos.

Inputs are the spatial footprints minicnmfe already produces (``model.A``,
``model.dims``), detected *independently* per session. The pipeline:

1. **Load** footprints per session (a :class:`~minicnmfe.pipeline.CNMFe` model
   or a results dir written by ``model.save``) -> dense ``(K, H, W)`` stacks +
   centroids + a projection image for alignment.
2. **Align** each session's FOV to a reference with a rigid-body transform
   (translation, optionally + rotation), reusing
   :func:`minicnmfe.motion_correction.estimate_shifts`.
3. **Metrics** — for nearby cell pairs (KD-tree on aligned centroids within
   ``max_distance``) compute centroid distance + spatial (Pearson) correlation.
4a. **Phase 1 (default)** — threshold the metrics and resolve a one-to-one
    matching per session pair via the Hungarian algorithm.
4b. **Phase 2 (``probabilistic=True``)** — fit a ``P_same`` model (2-component
    Gaussian mixture over distance and/or correlation) and match on ``P_same``.
5. **Cluster** the pairwise matches across all sessions into a
   ``cell_to_index_map`` of shape ``(n_global_cells, n_sessions)`` (``-1`` =
   the cell is absent from that session), enforcing at most one cell per
   session per global cluster.

Top-level entry point: :func:`register_sessions`. Result container:
:class:`CellRegResult`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from minicnmfe._utils import footprint_center
from minicnmfe.motion_correction import apply_shift, estimate_shifts
from minicnmfe.spatial import threshold_footprint

if TYPE_CHECKING:
    from minicnmfe.pipeline import CNMFe


# ---------------------------------------------------------------------------
# Per-session footprint container
# ---------------------------------------------------------------------------

@dataclass
class _Session:
    """Normalised, in-memory view of one session's footprints."""

    footprints: np.ndarray          # (K, H, W) float32, unit-max normalised
    centroids: np.ndarray           # (K, 2) float32 — (cy, cx)
    projection: np.ndarray          # (H, W) float32 — max projection (alignment)
    dims: tuple[int, int]

    @property
    def K(self) -> int:
        return self.footprints.shape[0]


def _load_session(
    src: "CNMFe | str | Path",
    *,
    accepted_only: bool = False,
    apply_threshold: bool = True,
    max_thr: float = 0.2,
) -> _Session:
    """Build a :class:`_Session` from a CNMFe model or a results dir.

    Each footprint is reshaped to ``(H, W)``, optionally cleaned with
    :func:`~minicnmfe.spatial.threshold_footprint`, and unit-max normalised so
    spatial correlations and projections are amplitude-invariant.
    """
    # Lazy import avoids a circular import (pipeline imports nothing from here,
    # but keep the dependency direction one-way and cheap).
    from minicnmfe.pipeline import CNMFe

    if isinstance(src, (str, Path)):
        model = CNMFe.load(src)
    else:
        model = src

    if model.A is None or model.dims is None:
        raise ValueError("Session has no footprints (model.A / model.dims is None).")

    dims = (int(model.dims[0]), int(model.dims[1]))
    H, W = dims
    A = model.A.tocsc()

    cols = np.arange(A.shape[1])
    if accepted_only and getattr(model, "accepted_mask", None) is not None:
        cols = np.flatnonzero(np.asarray(model.accepted_mask, dtype=bool))

    K = len(cols)
    footprints = np.zeros((K, H, W), dtype=np.float32)
    centroids = np.zeros((K, 2), dtype=np.float32)
    for out_k, k in enumerate(cols):
        flat = np.asarray(A[:, k].todense(), dtype=np.float32).ravel()
        if apply_threshold:
            flat = threshold_footprint(flat, dims, max_thr=max_thr)
        peak = float(flat.max())
        if peak > 0:
            flat = flat / peak
        img = flat.reshape(H, W)
        footprints[out_k] = img
        cy, cx = footprint_center(img, smooth_sigma=1.0)
        centroids[out_k] = (cy, cx)

    projection = footprints.max(axis=0) if K > 0 else np.zeros(dims, np.float32)
    return _Session(footprints=footprints, centroids=centroids,
                    projection=projection, dims=dims)


# ---------------------------------------------------------------------------
# Stage 1 — rigid-body alignment
# ---------------------------------------------------------------------------

def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation of two images (scalar in [-1, 1])."""
    a = a.ravel().astype(np.float32)
    b = b.ravel().astype(np.float32)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom <= 0:
        return 0.0
    return float((a * b).sum() / denom)


def _rigid_affine(dims: tuple[int, int], dy: float, dx: float, theta: float) -> np.ndarray:
    """2x3 affine: rotate about the image centre by ``theta`` (deg, CCW) then
    translate by ``(dy, dx)``. Applying it to a point and warping an image with
    it move features identically (cv2 convention)."""
    H, W = dims
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), float(theta), 1.0)
    M[0, 2] += dx
    M[1, 2] += dy
    return M


def _warp_image(img: np.ndarray, M: np.ndarray, dims: tuple[int, int]) -> np.ndarray:
    H, W = dims
    return cv2.warpAffine(img.astype(np.float32), M, (W, H),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0.0)


def _warp_points(pts_yx: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply affine ``M`` to ``(N, 2)`` points given as ``(cy, cx)``."""
    if len(pts_yx) == 0:
        return pts_yx.copy()
    xy = pts_yx[:, ::-1]                       # (cx, cy)
    hom = np.column_stack([xy, np.ones(len(xy))])
    out = hom @ M.T                            # (N, 2) = (x', y')
    return out[:, ::-1].astype(np.float32)     # back to (cy', cx')


def _estimate_rigid(
    proj: np.ndarray,
    ref: np.ndarray,
    *,
    mode: str,
    max_shift_px: int,
    angle_range_deg: float,
    angle_step_deg: float,
) -> tuple[float, float, float]:
    """Estimate ``(dy, dx, theta)`` mapping ``proj`` onto ``ref``.

    ``mode`` is ``"none"``, ``"translation"`` or ``"rotation"``. Rotation does a
    coarse angle grid search (each angle followed by a phase-correlation
    translation), keeping the angle with the highest NCC.
    """
    if mode == "none":
        return 0.0, 0.0, 0.0

    dims = proj.shape
    if mode == "translation":
        dy, dx = estimate_shifts(proj, ref, max_shift=(max_shift_px, max_shift_px))
        return float(dy), float(dx), 0.0

    if mode == "rotation":
        angles = np.arange(-angle_range_deg, angle_range_deg + 1e-9, angle_step_deg)
        best = (-np.inf, 0.0, 0.0, 0.0)        # (score, dy, dx, theta)
        for theta in angles:
            R = _rigid_affine(dims, 0.0, 0.0, theta)
            rot = _warp_image(proj, R, dims)
            dy, dx = estimate_shifts(rot, ref, max_shift=(max_shift_px, max_shift_px))
            aligned = apply_shift(rot, (dy, dx))
            score = _ncc(aligned, ref)
            if score > best[0]:
                best = (score, float(dy), float(dx), float(theta))
        return best[1], best[2], best[3]

    raise ValueError(f"Unknown alignment mode: {mode!r}")


def align_sessions(
    sessions: list[_Session],
    *,
    reference: int = 0,
    mode: str = "translation",
    max_shift_px: int = 20,
    angle_range_deg: float = 10.0,
    angle_step_deg: float = 1.0,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Align every session to ``reference`` with a rigid transform.

    Returns ``(aligned_footprints, aligned_centroids, transforms)`` where
    ``transforms`` is ``(n_sessions, 3)`` of ``(dy, dx, theta)``. The reference
    keeps the identity transform. Footprint images are warped only when a
    non-identity transform is needed (so the common same-FOV case is cheap).
    """
    ref_proj = sessions[reference].projection
    aligned_fps: list[np.ndarray] = []
    aligned_cents: list[np.ndarray] = []
    transforms = np.zeros((len(sessions), 3), dtype=np.float32)

    for s, sess in enumerate(sessions):
        if s == reference or mode == "none":
            aligned_fps.append(sess.footprints)
            aligned_cents.append(sess.centroids.copy())
            continue
        dy, dx, theta = _estimate_rigid(
            sess.projection, ref_proj, mode=mode, max_shift_px=max_shift_px,
            angle_range_deg=angle_range_deg, angle_step_deg=angle_step_deg,
        )
        transforms[s] = (dy, dx, theta)
        M = _rigid_affine(sess.dims, dy, dx, theta)
        warped = np.stack([_warp_image(f, M, sess.dims) for f in sess.footprints]) \
            if sess.K > 0 else sess.footprints
        aligned_fps.append(warped.astype(np.float32))
        aligned_cents.append(_warp_points(sess.centroids, M))

    return aligned_fps, aligned_cents, transforms


# ---------------------------------------------------------------------------
# Stage 2 — pairwise candidate metrics
# ---------------------------------------------------------------------------

@dataclass
class _PairMetrics:
    """Candidate cell-pairs between two sessions and their similarity metrics."""

    i: np.ndarray          # (P,) cell index in session a
    j: np.ndarray          # (P,) cell index in session b
    dist: np.ndarray       # (P,) centroid distance (px)
    corr: np.ndarray       # (P,) spatial Pearson correlation


def _spatial_corr(fa: np.ndarray, fb: np.ndarray) -> float:
    """Pearson correlation of two footprint images over the union of supports."""
    mask = (fa > 0) | (fb > 0)
    n = int(mask.sum())
    if n < 2:
        return 0.0
    a = fa[mask]
    b = fb[mask]
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom <= 0:
        return 0.0
    return float((a * b).sum() / denom)


def pairwise_metrics(
    fps_a: np.ndarray, cents_a: np.ndarray,
    fps_b: np.ndarray, cents_b: np.ndarray,
    *,
    max_distance_px: float,
) -> _PairMetrics:
    """Centroid-distance + spatial-correlation metrics for nearby cell pairs.

    Candidate pairs are restricted to centroids within ``max_distance_px``
    (KD-tree) so the cost stays ``O(neighbours)`` rather than ``O(Ka*Kb)``.
    """
    Ka = len(cents_a)
    Kb = len(cents_b)
    if Ka == 0 or Kb == 0:
        empty_i = np.empty(0, dtype=int)
        return _PairMetrics(empty_i, empty_i.copy(),
                            np.empty(0), np.empty(0))

    tree = cKDTree(cents_b)
    neighbours = tree.query_ball_point(cents_a, r=max_distance_px)

    ii, jj, dd, cc = [], [], [], []
    for i, js in enumerate(neighbours):
        for j in js:
            d = float(np.hypot(*(cents_a[i] - cents_b[j])))
            c = _spatial_corr(fps_a[i], fps_b[j])
            ii.append(i)
            jj.append(j)
            dd.append(d)
            cc.append(c)

    return _PairMetrics(
        i=np.asarray(ii, dtype=int),
        j=np.asarray(jj, dtype=int),
        dist=np.asarray(dd, dtype=np.float32),
        corr=np.asarray(cc, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# CellReg-style NN / NNN candidate distributions
# ---------------------------------------------------------------------------

@dataclass
class NNDistributions:
    """Same-cell-candidate (nearest-neighbour) vs different-cell (non-nearest)
    metric distributions, the empirical basis for the CellReg P_same model.

    Mirrors ``compute_data_distribution.m``: for every cell, the cross-session
    neighbour with the **highest spatial correlation** (and, independently, the
    **smallest centroid distance**) is the same-cell candidate (NN); every other
    neighbour within ``max_distance`` is a different-cell example (NNN). The corr
    and distance NN/NNN arrays are kept separately because CellReg fits a 1-D
    model to each.
    """

    nn_corr: np.ndarray
    nnn_corr: np.ndarray
    nn_dist: np.ndarray
    nnn_dist: np.ndarray


def compute_nn_nnn_distributions(
    aligned_fps: list[np.ndarray],
    aligned_cents: list[np.ndarray],
    *,
    max_distance_px: float,
) -> NNDistributions:
    """Build NN/NNN correlation & distance distributions across all sessions.

    For each (session, cell), collect every cross-session neighbour within
    ``max_distance_px`` and its (centroid distance, spatial correlation). The
    nearest neighbour (max corr / min dist) seeds the NN arrays; the rest seed
    the NNN arrays. Pairs are gathered from each cell's perspective (so a pair is
    seen from both sides — fine for distribution fitting).
    """
    n = len(aligned_fps)
    nn_corr: list[float] = []
    nnn_corr: list[float] = []
    nn_dist: list[float] = []
    nnn_dist: list[float] = []

    for a in range(n):
        cents_a = aligned_cents[a]
        for i in range(len(cents_a)):
            ci = cents_a[i]
            fi = aligned_fps[a][i]
            corrs: list[float] = []
            dists: list[float] = []
            for b in range(n):
                if b == a:
                    continue
                cb = aligned_cents[b]
                if len(cb) == 0:
                    continue
                d = np.hypot(cb[:, 0] - ci[0], cb[:, 1] - ci[1])
                for j in np.flatnonzero(d <= max_distance_px):
                    corrs.append(_spatial_corr(fi, aligned_fps[b][j]))
                    dists.append(float(d[j]))
            if not corrs:
                continue
            ca = np.asarray(corrs)
            da = np.asarray(dists)
            kmax = int(np.argmax(ca))
            nn_corr.append(float(ca[kmax]))
            nnn_corr.extend(np.delete(ca, kmax).tolist())
            kmin = int(np.argmin(da))
            nn_dist.append(float(da[kmin]))
            nnn_dist.extend(np.delete(da, kmin).tolist())

    return NNDistributions(
        nn_corr=np.asarray(nn_corr, dtype=np.float64),
        nnn_corr=np.asarray(nnn_corr, dtype=np.float64),
        nn_dist=np.asarray(nn_dist, dtype=np.float64),
        nnn_dist=np.asarray(nnn_dist, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# CellReg-style P_same models (lognormal+beta on correlation; lognormal+
# linear*sigmoid on distance) with a DATA-DRIVEN threshold at the P_same=0.5
# crossing. Faithful to compute_spatial_correlations_model.m /
# compute_centroid_distances_model.m. Replaces the hand-set corr_thr/dist_thr.
# ---------------------------------------------------------------------------

def _lognpdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sigma = max(float(sigma), 1e-6)
    out = np.zeros_like(x)
    pos = x > 0
    out[pos] = (1.0 / (x[pos] * sigma * np.sqrt(2 * np.pi))) * \
        np.exp(-0.5 * ((np.log(x[pos]) - mu) / sigma) ** 2)
    return out


def _betapdf(x: np.ndarray, p: float, q: float) -> np.ndarray:
    from scipy.special import betaln
    x = np.clip(np.asarray(x, dtype=float), 1e-9, 1 - 1e-9)
    logpdf = (p - 1) * np.log(x) + (q - 1) * np.log(1 - x) - betaln(p, q)
    return np.exp(logpdf)


def _beta_mom(weights: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Weighted method-of-moments beta parameters."""
    w = np.asarray(weights, dtype=float)
    ws = w.sum() + 1e-12
    m = float((w * x).sum() / ws)
    m = min(max(m, 1e-4), 1 - 1e-4)
    v = float((w * (x - m) ** 2).sum() / ws) + 1e-9
    common = m * (1 - m) / v - 1.0
    common = max(common, 1e-3)
    return max(m * common, 1e-2), max((1 - m) * common, 1e-2)


@dataclass
class SpatialCorrelationModel:
    """Lognormal(same) + Beta(different) mixture over spatial correlation.

    Correlation in [-1, 1] is mapped to t = (corr+1)/2 in (0, 1) for the fit
    (CellReg's footprint correlations live in [0,1]; our union-support corr can
    go negative, so we rescale monotonically). ``p_same(corr)`` interpolates the
    fitted posterior; ``threshold`` is the corr where P_same crosses 0.5.
    """

    pi_same: float
    mu: float
    sigma: float
    p: float
    q: float
    same_is_lognormal: bool
    grid_corr: np.ndarray
    grid_psame: np.ndarray
    threshold: float
    cost: float = float("inf")        # FP+FN+MSE (choose_best_model)
    false_positive: float = float("nan")
    false_negative: float = float("nan")

    @staticmethod
    def _to_t(corr):
        return np.clip((np.asarray(corr, dtype=float) + 1.0) / 2.0, 1e-3, 1 - 1e-3)

    def p_same_from_corr(self, corr) -> np.ndarray:
        t = self._to_t(corr)
        return np.interp(t, self.grid_corr, self.grid_psame).astype(np.float32)


def fit_spatial_correlation_model(all_corr: np.ndarray, *, em_iters: int = 100
                                  ) -> SpatialCorrelationModel | None:
    """Fit the lognormal(same)+beta(different) correlation mixture by EM."""
    c = np.asarray(all_corr, dtype=float)
    c = c[np.isfinite(c)]
    if c.size < 20:
        return None
    t = SpatialCorrelationModel._to_t(c)
    logt = np.log(t)

    pi = 0.5
    mu, sigma = float(logt.mean()), float(logt.std() + 1e-3)
    p, q = 2.0, 5.0
    for _ in range(em_iters):
        ln = _lognpdf(t, mu, sigma)
        be = _betapdf(t, p, q)
        num = pi * ln
        r = num / (num + (1 - pi) * be + 1e-12)
        pi = float(r.mean())
        sw = r.sum() + 1e-12
        mu = float((r * logt).sum() / sw)
        sigma = float(np.sqrt((r * (logt - mu) ** 2).sum() / sw) + 1e-6)
        p, q = _beta_mom(1.0 - r, t)
        pi = min(max(pi, 1e-3), 1 - 1e-3)

    # Degenerate fit (single mode — e.g. a sparse field with only same-cell
    # candidates): one component carries everything. Return None so the caller
    # falls back to threshold matching instead of trusting a meaningless model.
    if pi < 0.02 or pi > 0.98:
        return None

    ln_mean = float(np.exp(mu + 0.5 * sigma ** 2))
    beta_mean = p / (p + q)
    same_is_lognormal = ln_mean >= beta_mean

    grid_t = np.linspace(1e-3, 1 - 1e-3, 1000)
    ln_g = _lognpdf(grid_t, mu, sigma)
    be_g = _betapdf(grid_t, p, q)
    same_g = pi * ln_g if same_is_lognormal else (1 - pi) * be_g
    diff_g = (1 - pi) * be_g if same_is_lognormal else pi * ln_g
    psame_g = same_g / (same_g + diff_g + 1e-12)

    # threshold = corr where P_same crosses 0.5 (the same/different crossing)
    thr_t = _threshold_from_psame(grid_t, psame_g)
    threshold = float(2.0 * thr_t - 1.0)

    # model cost = FP + FN + MSE (choose_best_model). Same-cell sits at high t.
    idx = int(np.argmin(np.abs(psame_g - 0.5)))
    fp = float(diff_g[idx:].sum() / (diff_g.sum() + 1e-12))
    fn = float(same_g[:idx].sum() / (same_g.sum() + 1e-12))
    edges = np.linspace(0, 1, 52)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist, _ = np.histogram(t, bins=edges, density=True)
    total = pi * _lognpdf(centers, mu, sigma) + (1 - pi) * _betapdf(centers, p, q)
    mse = float(np.mean((total - hist) ** 2))

    return SpatialCorrelationModel(
        pi_same=pi, mu=mu, sigma=sigma, p=p, q=q,
        same_is_lognormal=same_is_lognormal,
        grid_corr=grid_t, grid_psame=psame_g.astype(float), threshold=threshold,
        cost=fp + fn + mse, false_positive=fp, false_negative=fn,
    )


@dataclass
class CentroidDistanceModel:
    """Lognormal(same) + linear*sigmoid(different) mixture over centroid distance
    (in microns if ``microns_per_pixel`` given, else pixels)."""

    params: np.ndarray            # [p0, mu, sigma, a, c, b]
    microns_per_pixel: float
    grid_dist_px: np.ndarray
    grid_psame: np.ndarray
    threshold_px: float
    cost: float = float("inf")        # FP+FN+MSE (choose_best_model)
    false_positive: float = float("nan")
    false_negative: float = float("nan")

    def p_same_from_dist(self, dist_px) -> np.ndarray:
        return np.interp(np.asarray(dist_px, dtype=float),
                         self.grid_dist_px, self.grid_psame).astype(np.float32)


def _centroid_F(x, p0, mu, sigma, a, c, b):
    same = p0 * _lognpdf(x, mu, sigma)
    diff = (1 - p0) * b * x / (1.0 + np.exp(-a * (x - c)))
    return same + diff


def fit_centroid_distance_model(all_dist_px: np.ndarray, *,
                                microns_per_pixel: float | None = None,
                                max_distance_px: float,
                                n_bins: int = 51) -> CentroidDistanceModel | None:
    """Fit the lognormal(same)+linear*sigmoid(different) distance mixture to the
    pooled-distance histogram via nonlinear least squares (lsqcurvefit-style)."""
    from scipy.optimize import curve_fit

    mpp = float(microns_per_pixel) if microns_per_pixel else 1.0
    d = np.asarray(all_dist_px, dtype=float)
    d = d[np.isfinite(d)]
    if d.size < 20:
        return None
    d_u = d * mpp                                   # work in physical units
    hi = max_distance_px * mpp
    edges = np.linspace(0, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist, _ = np.histogram(d_u, bins=edges, density=True)

    med = float(np.median(d_u))
    p0 = [0.4, np.log(max(med * 0.4, 1e-2)), 0.6, 2.0 / max(mpp, 1e-3),
          0.5 * hi, hist.max() / max(hi, 1e-3)]
    bounds = ([1e-3, -5, 1e-2, 1e-3, 0.0, 0.0],
              [1 - 1e-3, np.log(hi + 1), 3.0, 50.0, hi, 1e3])
    try:
        popt, _ = curve_fit(_centroid_F, centers, hist, p0=p0, bounds=bounds,
                            maxfev=20000)
    except Exception:
        return None

    p0f, mu, sigma, a, c, b = popt
    if p0f < 0.02 or p0f > 0.98:        # degenerate single-mode fit -> fall back
        return None
    grid = np.linspace(1e-3, hi, 1000)
    same_g = p0f * _lognpdf(grid, mu, sigma)
    diff_g = (1 - p0f) * b * grid / (1.0 + np.exp(-a * (grid - c)))
    psame_g = same_g / (same_g + diff_g + 1e-12)
    thr_u = _threshold_from_psame(grid, psame_g, increasing=False)

    # model cost = FP + FN + MSE. Same-cell sits at small distance.
    idx = int(np.argmin(np.abs(psame_g - 0.5)))
    fp = float(diff_g[:idx].sum() / (diff_g.sum() + 1e-12))
    fn = float(same_g[idx:].sum() / (same_g.sum() + 1e-12))
    mse = float(np.mean((_centroid_F(centers, *popt) - hist) ** 2))

    return CentroidDistanceModel(
        params=np.asarray(popt), microns_per_pixel=mpp,
        grid_dist_px=(grid / mpp), grid_psame=psame_g.astype(float),
        threshold_px=float(thr_u / mpp),
        cost=fp + fn + mse, false_positive=fp, false_negative=fn,
    )


def _threshold_from_psame(grid: np.ndarray, psame: np.ndarray,
                          increasing: bool = True) -> float:
    """Locate where P_same crosses 0.5. ``increasing`` = P_same rises with the
    grid value (correlation); else it falls (distance)."""
    above = psame >= 0.5
    if increasing:
        idx = np.flatnonzero(~above[:-1] & above[1:])
        return float(grid[idx[0] + 1]) if idx.size else float(grid[int(np.argmax(psame))])
    idx = np.flatnonzero(above[:-1] & ~above[1:])
    return float(grid[idx[0]]) if idx.size else float(grid[int(np.argmax(psame))])


@dataclass
class CellRegModel:
    """Unified P_same model wrapping a spatial-correlation and/or centroid-
    distance sub-model, with the active ``feature`` exposed via ``p_same``."""

    feature: str                                   # "spatial" | "centroid"
    spatial: SpatialCorrelationModel | None = None
    centroid: CentroidDistanceModel | None = None

    def p_same(self, dist, corr) -> np.ndarray:
        if self.feature == "spatial":
            return self.spatial.p_same_from_corr(corr)
        return self.centroid.p_same_from_dist(dist)

    @property
    def threshold(self) -> float:
        return (self.spatial.threshold if self.feature == "spatial"
                else self.centroid.threshold_px)

    @property
    def _chosen(self):
        return self.spatial if self.feature == "spatial" else self.centroid

    @property
    def false_positive(self) -> float:
        return float(self._chosen.false_positive)

    @property
    def false_negative(self) -> float:
        return float(self._chosen.false_negative)


def compute_registration_scores(
    cell_to_index_map: np.ndarray,
    aligned_fps: list[np.ndarray],
    aligned_cents: list[np.ndarray],
    model: "CellRegModel",
    *,
    certainty: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-cluster confidence scores for a registration (CellReg compute_scores).

    Returns ``(cell_scores, p_same_registered_pairs, uncertain_fraction)``:
    - ``cell_scores`` ``(n_global,)`` = mean ``P_same`` of each cluster's
      within-cluster cross-session pairs (NaN for singletons);
    - ``p_same_registered_pairs`` = flat array of those pair P_same values;
    - ``uncertain_fraction`` = fraction of registered pairs with
      ``1-certainty < P_same < certainty`` (ambiguous matches).
    """
    n_global = int(cell_to_index_map.shape[0])
    cell_scores = np.full(n_global, np.nan, dtype=np.float32)
    pairs: list[float] = []
    for g, row in enumerate(cell_to_index_map):
        members = [(s, int(k)) for s, k in enumerate(row) if k >= 0]
        if len(members) < 2:
            continue
        vals: list[float] = []
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                sa, ka = members[a]
                sb, kb = members[b]
                d = float(np.hypot(aligned_cents[sa][ka][0] - aligned_cents[sb][kb][0],
                                   aligned_cents[sa][ka][1] - aligned_cents[sb][kb][1]))
                c = float(_spatial_corr(aligned_fps[sa][ka], aligned_fps[sb][kb]))
                ps = float(model.p_same(np.array([d]), np.array([c]))[0])
                vals.append(ps)
        cell_scores[g] = float(np.mean(vals))
        pairs.extend(vals)
    pp = np.asarray(pairs, dtype=np.float32)
    lo = 1.0 - certainty
    uncertain = (float(np.mean((pp > lo) & (pp < certainty))) if pp.size
                 else float("nan"))
    return cell_scores, pp, uncertain


def fit_cellreg_model(all_dist_px: np.ndarray, all_corr: np.ndarray, *,
                      feature: str = "spatial",
                      microns_per_pixel: float | None = None,
                      max_distance_px: float) -> CellRegModel | None:
    """Fit the CellReg P_same model(s) on pooled candidate metrics."""
    spatial = centroid = None
    if feature in ("spatial", "auto"):
        spatial = fit_spatial_correlation_model(all_corr)
    if feature in ("centroid", "auto"):
        centroid = fit_centroid_distance_model(
            all_dist_px, microns_per_pixel=microns_per_pixel,
            max_distance_px=max_distance_px)
    if feature == "spatial":
        m = CellRegModel("spatial", spatial=spatial) if spatial else None
    elif feature == "centroid":
        m = CellRegModel("centroid", centroid=centroid) if centroid else None
    else:  # auto: choose_best_model — pick the lower-cost (FP+FN+MSE) model.
        best = choose_best_model(spatial, centroid)
        m = CellRegModel(best, spatial=spatial, centroid=centroid) if best else None
    return _reject_degenerate(m, all_dist_px, all_corr)


def _reject_degenerate(model: "CellRegModel | None", all_dist_px, all_corr,
                       *, lo: float = 0.02, hi: float = 0.98) -> "CellRegModel | None":
    """Return ``None`` if the fitted model has no real two-population structure.

    A trustworthy P_same model labels *some* candidates "same" and *some*
    "different". If essentially all (or none) of the pooled candidates land on one
    side of P_same=0.5 — e.g. a sparse field with only same-cell neighbours and no
    different-cell population to calibrate against — the model is meaningless, so
    the caller should fall back to threshold matching instead."""
    if model is None:
        return None
    pred = model.p_same(np.asarray(all_dist_px), np.asarray(all_corr))
    if pred.size == 0:
        return None
    frac_same = float(np.mean(pred >= 0.5))
    return None if frac_same < lo or frac_same > hi else model


def choose_best_model(spatial: SpatialCorrelationModel | None,
                      centroid: CentroidDistanceModel | None) -> str | None:
    """Pick the lower-cost model (cost = false_positive + false_negative + MSE),
    matching CellReg's ``choose_best_model.m``. Returns ``"spatial"`` /
    ``"centroid"`` / ``None``."""
    if spatial is None and centroid is None:
        return None
    if centroid is None:
        return "spatial"
    if spatial is None:
        return "centroid"
    return "spatial" if spatial.cost <= centroid.cost else "centroid"


# ---------------------------------------------------------------------------
# Stage 3b — probabilistic P_same model (Phase 2)
# ---------------------------------------------------------------------------

@dataclass
class PSameModel:
    """Two-component Gaussian-mixture ``P_same`` model over pooled metrics.

    ``model`` selects the feature(s): ``"centroid"`` (distance), ``"spatial"``
    (correlation) or ``"joint"`` (both). The component judged to be the
    same-cell mode (small distance / high correlation) defines ``P_same`` as its
    posterior responsibility.
    """

    model: str
    gmm: object                     # fitted sklearn GaussianMixture
    same_component: int
    max_distance_px: float

    def _features(self, dist: np.ndarray, corr: np.ndarray) -> np.ndarray:
        if self.model == "centroid":
            return np.asarray(dist, dtype=float).reshape(-1, 1)
        if self.model == "spatial":
            return np.asarray(corr, dtype=float).reshape(-1, 1)
        return np.column_stack([np.asarray(dist, float), np.asarray(corr, float)])

    def p_same(self, dist: np.ndarray, corr: np.ndarray) -> np.ndarray:
        X = self._features(dist, corr)
        if len(X) == 0:
            return np.empty(0, dtype=np.float32)
        resp = self.gmm.predict_proba(X)
        return resp[:, self.same_component].astype(np.float32)


def fit_psame_model(
    all_metrics: list[_PairMetrics],
    *,
    model: str = "spatial",
    max_distance_px: float,
) -> PSameModel | None:
    """Fit a 2-component GMM over the pooled candidate metrics.

    Returns ``None`` when there are too few candidates to fit (the caller then
    falls back to threshold matching).
    """
    dist = np.concatenate([m.dist for m in all_metrics]) if all_metrics else np.empty(0)
    corr = np.concatenate([m.corr for m in all_metrics]) if all_metrics else np.empty(0)
    if len(dist) < 10:
        return None

    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        return None

    if model == "centroid":
        X = dist.reshape(-1, 1).astype(float)
    elif model == "spatial":
        X = corr.reshape(-1, 1).astype(float)
    elif model == "joint":
        X = np.column_stack([dist.astype(float), corr.astype(float)])
    else:
        raise ValueError(f"Unknown P_same model: {model!r}")

    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
    gmm.fit(X)

    # Identify the "same cell" component: smallest mean distance (centroid/joint)
    # or largest mean correlation (spatial).
    means = gmm.means_
    if model == "spatial":
        same = int(np.argmax(means[:, 0]))
    else:
        same = int(np.argmin(means[:, 0]))   # distance is column 0 for centroid/joint

    return PSameModel(model=model, gmm=gmm, same_component=same,
                      max_distance_px=max_distance_px)


# ---------------------------------------------------------------------------
# Stage 3a — one-to-one matching per session pair
# ---------------------------------------------------------------------------

@dataclass
class _Match:
    i: int
    j: int
    score: float
    dist: float
    corr: float


def match_pairwise(
    metrics: _PairMetrics,
    Ka: int,
    Kb: int,
    *,
    max_distance_px: float,
    dist_thr_px: float,
    corr_thr: float,
    corr_weight: float = 0.5,
    psame: PSameModel | None = None,
    p_same_thr: float = 0.5,
) -> list[_Match]:
    """Resolve a one-to-one matching for one session pair (Hungarian).

    Phase 1 (``psame is None``): candidates must satisfy ``dist <= dist_thr_px``
    AND ``corr >= corr_thr``; the score is a convex blend of correlation and
    (1 - normalised distance). Phase 2: candidates must have
    ``P_same >= p_same_thr`` and the score is ``P_same``.
    """
    if len(metrics.i) == 0 or Ka == 0 or Kb == 0:
        return []

    if psame is not None:
        score = psame.p_same(metrics.dist, metrics.corr)
        keep = score >= p_same_thr
    else:
        norm_d = np.clip(metrics.dist / max(max_distance_px, 1e-6), 0.0, 1.0)
        score = corr_weight * metrics.corr + (1.0 - corr_weight) * (1.0 - norm_d)
        keep = (metrics.dist <= dist_thr_px) & (metrics.corr >= corr_thr)

    if not keep.any():
        return []

    BIG = 1e6
    cost = np.full((Ka, Kb), BIG, dtype=float)
    for idx in np.flatnonzero(keep):
        i = int(metrics.i[idx])
        j = int(metrics.j[idx])
        c = -float(score[idx])
        if c < cost[i, j]:                     # keep best score per (i, j)
            cost[i, j] = c

    rows, cols = linear_sum_assignment(cost)
    out: list[_Match] = []
    for r, cc in zip(rows, cols):
        if cost[r, cc] < BIG / 2:              # a real (kept) candidate
            # recover metrics for this assigned pair
            sel = np.flatnonzero((metrics.i == r) & (metrics.j == cc))
            k = sel[np.argmax(score[sel])]
            out.append(_Match(i=int(r), j=int(cc), score=float(score[k]),
                              dist=float(metrics.dist[k]), corr=float(metrics.corr[k])))
    return out


# ---------------------------------------------------------------------------
# Stage 4 — greedy multi-session clustering -> cell_to_index_map
# ---------------------------------------------------------------------------

def build_cell_map(
    n_sessions: int,
    K_per_session: list[int],
    edges: list[tuple[int, int, int, int, float]],
) -> np.ndarray:
    """Cluster pairwise matches into a ``(n_global, n_sessions)`` index map.

    ``edges`` are ``(session_a, cell_a, session_b, cell_b, weight)``. Greedy by
    descending weight: strong matches form clusters first, and a node only joins
    a cluster if that cluster has no cell from the node's session yet (so each
    global cell has at most one cell per session). Unmatched cells become
    singleton rows. ``-1`` marks an absent session.
    """
    parent: dict[tuple[int, int], int] = {}    # (s, k) -> cluster id
    clusters: list[dict[int, int]] = []        # cluster id -> {session: cell}

    for s_a, k_a, s_b, k_b, _w in sorted(edges, key=lambda e: -e[4]):
        u = (s_a, k_a)
        v = (s_b, k_b)
        cu = parent.get(u)
        cv = parent.get(v)

        if cu is None and cv is None:
            cid = len(clusters)
            clusters.append({s_a: k_a, s_b: k_b})
            parent[u] = cid
            parent[v] = cid
        elif cu is not None and cv is None:
            c = clusters[cu]
            if s_b not in c:
                c[s_b] = k_b
                parent[v] = cu
        elif cu is None and cv is not None:
            c = clusters[cv]
            if s_a not in c:
                c[s_a] = k_a
                parent[u] = cv
        elif cu != cv:
            ca, cb = clusters[cu], clusters[cv]
            if set(ca).isdisjoint(cb):         # merge only if no session conflict
                ca.update(cb)
                for node, cid in list(parent.items()):
                    if cid == cv:
                        parent[node] = cu
                clusters[cv] = {}              # emptied; skipped when emitting

    rows: list[list[int]] = []
    for c in clusters:
        if not c:
            continue
        row = [-1] * n_sessions
        for s, k in c.items():
            row[s] = k
        rows.append(row)

    # singletons (cells in no cluster)
    for s in range(n_sessions):
        for k in range(K_per_session[s]):
            if (s, k) not in parent:
                row = [-1] * n_sessions
                row[s] = k
                rows.append(row)

    if not rows:
        return np.empty((0, n_sessions), dtype=int)
    return np.asarray(rows, dtype=int)


# ---------------------------------------------------------------------------
# Stage 4b — iterative joint clustering (CellReg cluster_cells.m)
# ---------------------------------------------------------------------------

def cluster_cells_iterative(
    cell_to_index_map: np.ndarray,
    aligned_fps: list[np.ndarray],
    aligned_cents: list[np.ndarray],
    model: "CellRegModel",
    *,
    max_distance_px: float,
    p_same_thr: float = 0.5,
    max_iters: int = 10,
    num_changes_thresh: int = 10,
    cluster_distance_factor: float = 1.7,
) -> np.ndarray:
    """Refine an initial ``cell_to_index_map`` by iterative reassignment.

    Mirrors ``cluster_cells.m``: each iteration recomputes cluster centroids
    (mean of member centroids) and reassigns every cell to the **maximal-
    similarity** cluster within ``cluster_distance_factor * max_distance``,
    enforcing at most one cell per session. Similarity = mean ``model.p_same`` of
    the cell to the cluster's members in *other* sessions (the calibrated P_same,
    not raw correlation). A cell whose best similarity is below ``p_same_thr`` is
    split off into its own singleton. Converges when the per-iteration change
    count drops below ``num_changes_thresh`` (or ``max_iters`` is reached).

    Unlike single-pass greedy ``build_cell_map``, this re-evaluates decisions, so
    an early wrong merge can be corrected. Deterministic: cells are visited in a
    fixed order and a move requires a strict similarity improvement.
    """
    n_sessions = int(cell_to_index_map.shape[1])
    clusters: list[dict[int, int]] = []
    parent: dict[tuple[int, int], int] = {}
    for row in cell_to_index_map:
        c = {s: int(k) for s, k in enumerate(row) if k >= 0}
        if not c:
            continue
        cid = len(clusters)
        clusters.append(c)
        for s, k in c.items():
            parent[(s, k)] = cid

    corr_cache: dict[tuple, float] = {}

    def pair_psame(s, k, sm, km) -> float:
        d = float(np.hypot(aligned_cents[s][k][0] - aligned_cents[sm][km][0],
                           aligned_cents[s][k][1] - aligned_cents[sm][km][1]))
        key = (s, k, sm, km) if (s, k) <= (sm, km) else (sm, km, s, k)
        cc = corr_cache.get(key)
        if cc is None:
            cc = float(_spatial_corr(aligned_fps[s][k], aligned_fps[sm][km]))
            corr_cache[key] = cc
        return float(model.p_same(np.array([d]), np.array([cc]))[0])

    def similarity(s, k, cluster) -> float:
        vals = [pair_psame(s, k, sm, km) for sm, km in cluster.items() if sm != s]
        return float(np.mean(vals)) if vals else -1.0

    radius = cluster_distance_factor * max_distance_px
    cells = [(s, k) for s in range(n_sessions)
             for k in range(len(aligned_cents[s]))]

    for _ in range(max_iters):
        cent = np.full((len(clusters), 2), np.nan)
        for ci, c in enumerate(clusters):
            if c:
                pts = np.array([aligned_cents[s][k] for s, k in c.items()])
                cent[ci] = pts.mean(axis=0)
        valid = np.flatnonzero(~np.isnan(cent[:, 0]))
        tree = cKDTree(cent[valid]) if valid.size else None

        changes = 0
        for (s, k) in cells:
            cu = parent[(s, k)]
            my = aligned_cents[s][k]
            cand = set()
            if tree is not None:
                for idx in tree.query_ball_point(my, r=radius):
                    cand.add(int(valid[idx]))
            cand.add(cu)
            best_cid, best_sim = cu, similarity(s, k, clusters[cu])
            for cid in cand:
                if cid == cu:
                    continue
                c = clusters[cid]
                if s in c and c[s] != k:           # one-per-session conflict
                    continue
                sim = similarity(s, k, c)
                if sim > best_sim + 1e-9:
                    best_cid, best_sim = cid, sim

            if best_sim < p_same_thr:
                if len(clusters[cu]) > 1:          # split off into a singleton
                    del clusters[cu][s]
                    clusters.append({s: k})
                    parent[(s, k)] = len(clusters) - 1
                    changes += 1
                continue
            if best_cid != cu:                     # move to a better cluster
                del clusters[cu][s]
                clusters[best_cid][s] = k
                parent[(s, k)] = best_cid
                changes += 1

        if changes < num_changes_thresh:
            break

    rows = [[c.get(s, -1) for s in range(n_sessions)] for c in clusters if c]
    if not rows:
        return np.empty((0, n_sessions), dtype=int)
    return np.asarray(rows, dtype=int)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CellRegResult:
    """Output of :func:`register_sessions`.

    Attributes:
        cell_to_index_map: ``(n_global, n_sessions)`` int. Entry ``[g, s]`` is
            the cell index in session ``s`` for global cell ``g`` (``-1`` =
            absent).
        n_sessions: number of registered sessions.
        transforms: ``(n_sessions, 3)`` rigid transforms ``(dy, dx, theta)``
            mapping each session onto the reference.
        aligned_centroids: per-session ``(K, 2)`` aligned centroids.
        dims: reference ``(H, W)``.
        p_same_threshold: ``P_same`` cutoff used (Phase 2 only; else ``None``).
        params: the registration parameters used.
    """

    cell_to_index_map: np.ndarray
    n_sessions: int
    transforms: np.ndarray
    aligned_centroids: list[np.ndarray]
    dims: tuple[int, int]
    p_same_threshold: float | None = None
    params: dict = field(default_factory=dict)
    # Registration-confidence outputs (probabilistic_model path only; else None).
    cell_scores: np.ndarray | None = None          # (n_global,) mean within-cluster P_same
    p_same_registered_pairs: np.ndarray | None = None  # flat P_same of registered pairs
    uncertain_fraction: float | None = None        # frac registered pairs w/ 0.05<P_same<0.95
    model_false_positive: float | None = None
    model_false_negative: float | None = None

    @property
    def n_global(self) -> int:
        return int(self.cell_to_index_map.shape[0])

    def n_registered(self, min_sessions: int = 2) -> int:
        """Number of global cells present in at least ``min_sessions`` sessions."""
        if self.n_global == 0:
            return 0
        present = (self.cell_to_index_map >= 0).sum(axis=1)
        return int((present >= min_sessions).sum())

    def registered_map(self, min_sessions: int = 2) -> np.ndarray:
        """Rows of ``cell_to_index_map`` present in >= ``min_sessions`` sessions."""
        if self.n_global == 0:
            return self.cell_to_index_map
        present = (self.cell_to_index_map >= 0).sum(axis=1)
        return self.cell_to_index_map[present >= min_sessions]

    def save(self, output_dir: "str | Path") -> None:
        """Persist the registration to ``output_dir``."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "cell_to_index_map.npy", self.cell_to_index_map)
        np.save(output_dir / "transforms.npy", self.transforms)
        np.savez(output_dir / "aligned_centroids.npz",
                 **{f"session_{s}": c for s, c in enumerate(self.aligned_centroids)})
        if self.cell_scores is not None:
            np.save(output_dir / "cell_scores.npy", self.cell_scores)
        if self.p_same_registered_pairs is not None:
            np.save(output_dir / "p_same_registered_pairs.npy",
                    self.p_same_registered_pairs)
        info = {
            "n_sessions": self.n_sessions,
            "n_global": self.n_global,
            "dims": list(self.dims),
            "p_same_threshold": self.p_same_threshold,
            "params": self.params,
            "K_per_session": [int(c.shape[0]) for c in self.aligned_centroids],
            "uncertain_fraction": self.uncertain_fraction,
            "model_false_positive": self.model_false_positive,
            "model_false_negative": self.model_false_negative,
        }
        (output_dir / "cellreg_info.json").write_text(json.dumps(info, indent=2))

    @classmethod
    def load(cls, output_dir: "str | Path") -> "CellRegResult":
        """Reconstruct a :class:`CellRegResult` written by :meth:`save`."""
        output_dir = Path(output_dir)
        info = json.loads((output_dir / "cellreg_info.json").read_text())
        cmap = np.load(output_dir / "cell_to_index_map.npy")
        transforms = np.load(output_dir / "transforms.npy")
        with np.load(output_dir / "aligned_centroids.npz") as npz:
            n = info["n_sessions"]
            cents = [npz[f"session_{s}"] for s in range(n)]
        cs_path = output_dir / "cell_scores.npy"
        pp_path = output_dir / "p_same_registered_pairs.npy"
        return cls(
            cell_to_index_map=cmap,
            n_sessions=int(info["n_sessions"]),
            transforms=transforms,
            aligned_centroids=cents,
            dims=tuple(info["dims"]),
            p_same_threshold=info.get("p_same_threshold"),
            params=info.get("params", {}),
            cell_scores=np.load(cs_path) if cs_path.exists() else None,
            p_same_registered_pairs=np.load(pp_path) if pp_path.exists() else None,
            uncertain_fraction=info.get("uncertain_fraction"),
            model_false_positive=info.get("model_false_positive"),
            model_false_negative=info.get("model_false_negative"),
        )


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def register_sessions(
    sessions: "list[CNMFe | str | Path]",
    *,
    microns_per_pixel: float | None = None,
    max_distance_um: float = 12.0,
    max_distance_px: float | None = None,
    align: str = "translation",
    reference: int = 0,
    accepted_only: bool = False,
    # Phase 1 thresholds
    dist_thr_um: float = 5.0,
    dist_thr_px: float | None = None,
    corr_thr: float = 0.65,
    corr_weight: float = 0.5,
    # Phase 2 probabilistic
    probabilistic: bool = False,
    model: str = "spatial",
    p_same_thr: float = 0.5,
    # CellReg-style probabilistic model with a data-driven threshold.
    # DEFAULT (like CellReg): fit the P_same model and auto-select corr vs
    # distance; falls back to threshold matching (corr_thr/dist_thr) when the
    # model can't be fit (too few candidates / degenerate single-mode field).
    registration_approach: str = "probabilistic_model",
    psame_feature: str = "auto",
    clustering: str = "iterative",
    # alignment knobs
    max_shift_px: int = 20,
    angle_range_deg: float = 10.0,
    angle_step_deg: float = 1.0,
) -> CellRegResult:
    """Register cells across multiple sessions (CellReg-style).

    Args:
        sessions: list of :class:`~minicnmfe.pipeline.CNMFe` models or results
            dirs (paths) written by ``model.save``. All sessions must share the
            same ``dims``.
        microns_per_pixel: physical scale. When given, distance thresholds are
            interpreted in microns; otherwise in pixels.
        max_distance_um / max_distance_px: neighbour search radius. ``_px`` (if
            given) overrides the micron value.
        align: ``"translation"`` (default), ``"rotation"`` or ``"none"``.
        reference: index of the reference session for alignment.
        accepted_only: use only ``accepted_mask`` components when available.
        dist_thr_um / dist_thr_px / corr_thr: Phase 1 acceptance thresholds.
        corr_weight: blend weight between correlation and distance in the
            Phase 1 match score (0 = distance only, 1 = correlation only).
        probabilistic: enable the Phase 2 ``P_same`` model.
        model: ``P_same`` feature(s) — ``"centroid"``, ``"spatial"``, ``"joint"``.
        p_same_thr: Phase 2 acceptance cutoff on ``P_same``.

    Returns:
        A :class:`CellRegResult`.
    """
    if len(sessions) < 2:
        raise ValueError("register_sessions needs at least 2 sessions.")

    loaded = [_load_session(s, accepted_only=accepted_only) for s in sessions]

    dims0 = loaded[0].dims
    for idx, sess in enumerate(loaded):
        if sess.dims != dims0:
            raise ValueError(
                f"Session {idx} dims {sess.dims} != reference dims {dims0}. "
                "All sessions must share the same FOV size."
            )

    # Resolve distance thresholds to pixels.
    if max_distance_px is None:
        max_distance_px = (max_distance_um / microns_per_pixel
                           if microns_per_pixel else max_distance_um)
    if dist_thr_px is None:
        dist_thr_px = (dist_thr_um / microns_per_pixel
                       if microns_per_pixel else dist_thr_um)

    # Stage 1 — align.
    aligned_fps, aligned_cents, transforms = align_sessions(
        loaded, reference=reference, mode=align, max_shift_px=max_shift_px,
        angle_range_deg=angle_range_deg, angle_step_deg=angle_step_deg,
    )
    K_per = [int(c.shape[0]) for c in aligned_cents]
    n = len(loaded)

    # Stage 2 — pairwise metrics for every session pair.
    pair_metrics: dict[tuple[int, int], _PairMetrics] = {}
    for a in range(n):
        for b in range(a + 1, n):
            pair_metrics[(a, b)] = pairwise_metrics(
                aligned_fps[a], aligned_cents[a],
                aligned_fps[b], aligned_cents[b],
                max_distance_px=max_distance_px,
            )

    # Stage 3b — optional P_same model.
    psame = None
    p_same_threshold = None
    cellreg_model = None
    if registration_approach == "probabilistic_model":
        # CellReg-style: fit lognormal/beta (corr) or lognormal/linear*sigmoid
        # (distance) on the pooled candidate metrics; the decision boundary is
        # the data-driven P_same=0.5 crossing (p_same_thr default 0.5), NOT a
        # hand-set corr_thr/dist_thr.
        all_dist = (np.concatenate([m.dist for m in pair_metrics.values()])
                    if pair_metrics else np.empty(0))
        all_corr = (np.concatenate([m.corr for m in pair_metrics.values()])
                    if pair_metrics else np.empty(0))
        cellreg_model = fit_cellreg_model(
            all_dist, all_corr, feature=psame_feature,
            microns_per_pixel=microns_per_pixel, max_distance_px=max_distance_px)
        if cellreg_model is not None:
            psame = cellreg_model
            p_same_threshold = p_same_thr
        # else: fall back to threshold matching (too few candidates / fit failed)
    elif probabilistic:
        psame = fit_psame_model(list(pair_metrics.values()), model=model,
                                max_distance_px=max_distance_px)
        if psame is not None:
            p_same_threshold = p_same_thr
        # else: silently fall back to threshold matching (too few candidates)

    # Stage 3a — match each pair, collect edges.
    edges: list[tuple[int, int, int, int, float]] = []
    for (a, b), m in pair_metrics.items():
        matches = match_pairwise(
            m, K_per[a], K_per[b],
            max_distance_px=max_distance_px, dist_thr_px=dist_thr_px,
            corr_thr=corr_thr, corr_weight=corr_weight,
            psame=psame, p_same_thr=p_same_thr,
        )
        for mt in matches:
            edges.append((a, mt.i, b, mt.j, mt.score))

    # Stage 4 — cluster into the map (greedy single-pass), then optionally
    # refine with the CellReg iterative joint clustering on the model path.
    cell_to_index_map = build_cell_map(n, K_per, edges)
    if (registration_approach == "probabilistic_model" and cellreg_model is not None
            and clustering == "iterative"):
        cell_to_index_map = cluster_cells_iterative(
            cell_to_index_map, aligned_fps, aligned_cents, cellreg_model,
            max_distance_px=max_distance_px, p_same_thr=p_same_thr)

    # Stage 5 — registration-confidence scores (model path only).
    cell_scores = p_same_pairs = uncertain_fraction = None
    model_fp = model_fn = None
    if cellreg_model is not None:
        cell_scores, p_same_pairs, uncertain_fraction = compute_registration_scores(
            cell_to_index_map, aligned_fps, aligned_cents, cellreg_model)
        model_fp = cellreg_model.false_positive
        model_fn = cellreg_model.false_negative

    params = {
        "microns_per_pixel": microns_per_pixel,
        "max_distance_px": float(max_distance_px),
        "align": align,
        "reference": reference,
        "accepted_only": accepted_only,
        "dist_thr_px": float(dist_thr_px),
        "corr_thr": corr_thr,
        "corr_weight": corr_weight,
        "probabilistic": probabilistic,
        "model": model if probabilistic else None,
        "p_same_thr": p_same_thr if probabilistic else None,
        "registration_approach": registration_approach,
        "clustering": (clustering if registration_approach == "probabilistic_model"
                       and cellreg_model is not None else "greedy"),
        "psame_feature": (cellreg_model.feature if cellreg_model is not None
                          else (psame_feature if registration_approach
                                == "probabilistic_model" else None)),
        "data_driven_threshold": (float(cellreg_model.threshold)
                                  if cellreg_model is not None else None),
    }

    return CellRegResult(
        cell_to_index_map=cell_to_index_map,
        n_sessions=n,
        transforms=transforms,
        aligned_centroids=aligned_cents,
        dims=dims0,
        p_same_threshold=p_same_threshold,
        params=params,
        cell_scores=cell_scores,
        p_same_registered_pairs=p_same_pairs,
        uncertain_fraction=uncertain_fraction,
        model_false_positive=model_fp,
        model_false_negative=model_fn,
    )
