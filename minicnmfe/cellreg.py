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
        info = {
            "n_sessions": self.n_sessions,
            "n_global": self.n_global,
            "dims": list(self.dims),
            "p_same_threshold": self.p_same_threshold,
            "params": self.params,
            "K_per_session": [int(c.shape[0]) for c in self.aligned_centroids],
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
        return cls(
            cell_to_index_map=cmap,
            n_sessions=int(info["n_sessions"]),
            transforms=transforms,
            aligned_centroids=cents,
            dims=tuple(info["dims"]),
            p_same_threshold=info.get("p_same_threshold"),
            params=info.get("params", {}),
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
    if probabilistic:
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

    # Stage 4 — cluster into the map.
    cell_to_index_map = build_cell_map(n, K_per, edges)

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
    }

    return CellRegResult(
        cell_to_index_map=cell_to_index_map,
        n_sessions=n,
        transforms=transforms,
        aligned_centroids=aligned_cents,
        dims=dims0,
        p_same_threshold=p_same_threshold,
        params=params,
    )
