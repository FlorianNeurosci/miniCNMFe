"""Tests for cross-session cell registration (minicnmfe.cellreg)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from minicnmfe.cellreg import (
    CellRegResult,
    _load_session,
    _PairMetrics,
    fit_psame_model,
    register_sessions,
)
from minicnmfe.pipeline import CNMFe


# ---------------------------------------------------------------------------
# Synthetic footprint helpers
# ---------------------------------------------------------------------------

def _gaussian(dims, cy, cx, sigma):
    H, W = dims
    yy, xx = np.mgrid[0:H, 0:W]
    g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma ** 2))
    g[g < 0.02] = 0.0
    return g.astype(np.float32)


def _make_A(centers, dims, sigma=3.0, rng=None, amp_noise=0.0):
    H, W = dims
    cols = []
    for cy, cx in centers:
        g = _gaussian(dims, cy, cx, sigma)
        if amp_noise and rng is not None:
            g = np.clip(g + rng.normal(0, amp_noise, g.shape).astype(np.float32), 0, None)
        cols.append(g.ravel())
    A = np.stack(cols, axis=1) if cols else np.zeros((H * W, 0), np.float32)
    return sp.csc_matrix(A.astype(np.float32))


def _make_model(centers, dims, sigma=3.0, rng=None, amp_noise=0.0, accepted_mask=None):
    m = CNMFe()
    m.A = _make_A(centers, dims, sigma, rng, amp_noise)
    m.dims = dims
    if accepted_mask is not None:
        m.accepted_mask = np.asarray(accepted_mask, dtype=bool)
    return m


def _sample_centers(n, dims, min_dist, rng, existing=None, margin=12):
    H, W = dims
    pts = list(existing) if existing else []
    base_len = len(pts)
    tries = 0
    while len(pts) - base_len < n and tries < 50000:
        p = (float(rng.uniform(margin, H - margin)),
             float(rng.uniform(margin, W - margin)))
        if all(np.hypot(p[0] - q[0], p[1] - q[1]) >= min_dist for q in pts):
            pts.append(p)
        tries += 1
    return pts[base_len:]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_load_session_recovers_centroids():
    dims = (64, 64)
    centers = [(20, 20), (20, 45), (45, 30)]
    sess = _load_session(_make_model(centers, dims))
    assert sess.K == 3
    # centroids land on the gaussian peaks (integer argmax)
    for (cy, cx), got in zip(centers, sess.centroids):
        assert abs(got[0] - cy) <= 1
        assert abs(got[1] - cx) <= 1


def test_accepted_only_filters_components():
    dims = (64, 64)
    centers = [(20, 20), (20, 45), (45, 30)]
    m = _make_model(centers, dims, accepted_mask=[True, False, True])
    sess = _load_session(m, accepted_only=True)
    assert sess.K == 2


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def test_alignment_recovers_translation():
    dims = (90, 90)
    centers = [(20, 20), (20, 60), (55, 30), (60, 65), (40, 45)]
    shift = (5, -4)  # (dy, dx) applied to make session 2
    centers2 = [(cy + shift[0], cx + shift[1]) for cy, cx in centers]
    m1 = _make_model(centers, dims)
    m2 = _make_model(centers2, dims)

    res = register_sessions([m1, m2], align="translation",
                            max_distance_px=10, dist_thr_px=3, corr_thr=0.5)

    # the recovered transform should undo the applied shift (magnitude match)
    dy, dx, theta = res.transforms[1]
    assert theta == 0.0
    assert abs(abs(dy) - abs(shift[0])) <= 1.5
    assert abs(abs(dx) - abs(shift[1])) <= 1.5

    # aligned session-2 centroids land back on the base centers
    for got in res.aligned_centroids[1]:
        nearest = min(np.hypot(got[0] - cy, got[1] - cx) for cy, cx in centers)
        assert nearest <= 1.5

    # all five cells register across both sessions, correct correspondence
    assert res.n_registered(2) == 5
    both = res.cell_to_index_map[(res.cell_to_index_map >= 0).all(axis=1)]
    assert np.array_equal(np.sort(both[:, 0]), np.sort(both[:, 1]))
    for i, j in both:
        assert i == j  # same construction order => identity correspondence


# ---------------------------------------------------------------------------
# Matching: drop / add / jitter / noise
# ---------------------------------------------------------------------------

def test_matching_precision_recall():
    dims = (110, 110)
    rng = np.random.default_rng(1)
    base = _sample_centers(20, dims, 14, rng)
    n_keep = 17

    m1 = _make_model(base, dims, rng=rng, amp_noise=0.02)

    kept = [(cy + rng.normal(0, 0.7), cx + rng.normal(0, 0.7))
            for cy, cx in base[:n_keep]]
    added = _sample_centers(2, dims, 14, rng, existing=base)
    centers2 = kept + added                          # 17 kept + 2 new
    m2 = _make_model(centers2, dims, rng=rng, amp_noise=0.02)

    res = register_sessions([m1, m2], align="none",
                            max_distance_px=8, dist_thr_px=4, corr_thr=0.5)

    cmap = res.cell_to_index_map
    both = cmap[(cmap >= 0).all(axis=1)]
    # correct = same construction index AND a kept cell
    correct = sum(1 for i, j in both if i == j and i < n_keep)
    wrong = sum(1 for i, j in both if not (i == j and i < n_keep))

    assert correct >= 15            # recall >= 15/17
    assert wrong == 0               # precision: no false matches
    # added cells (session-2 idx >= n_keep) never matched
    for row in cmap:
        if row[1] >= n_keep:
            assert row[0] == -1


# ---------------------------------------------------------------------------
# Conflict resolution (one-to-one per session pair)
# ---------------------------------------------------------------------------

def test_conflict_resolution_one_to_one():
    dims = (60, 60)
    # two session-1 cells, both overlapping one session-2 cell (so both are
    # genuine candidates) — the matcher must pick exactly one.
    m1 = _make_model([(30, 29), (30, 33)], dims)
    m2 = _make_model([(30, 31)], dims)

    res = register_sessions([m1, m2], align="none",
                            max_distance_px=10, dist_thr_px=10, corr_thr=-1.0)

    cmap = res.cell_to_index_map
    both = cmap[(cmap >= 0).all(axis=1)]
    assert both.shape[0] == 1                 # the session-2 cell matches exactly one
    assert res.n_global == 2                  # one pair + one singleton


# ---------------------------------------------------------------------------
# N=3 transitive clustering
# ---------------------------------------------------------------------------

def test_three_session_clustering():
    dims = (96, 96)
    rng = np.random.default_rng(2)
    base = _sample_centers(8, dims, 16, rng)
    models = [_make_model(base, dims) for _ in range(3)]

    res = register_sessions(models, align="none",
                            max_distance_px=6, dist_thr_px=3, corr_thr=0.6)

    assert res.n_global == 8
    assert res.n_registered(3) == 8
    # every row is fully populated and consistent
    assert (res.cell_to_index_map >= 0).all()


# ---------------------------------------------------------------------------
# Probabilistic P_same model (Phase 2)
# ---------------------------------------------------------------------------

def test_psame_model_separates_same_from_different():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(3)
    n = 300
    same_d = np.abs(rng.normal(1.0, 0.5, n))
    same_c = np.clip(rng.normal(0.9, 0.04, n), 0, 1)
    diff_d = rng.uniform(6, 10, n)
    diff_c = np.clip(rng.normal(0.2, 0.1, n), 0, 1)

    dist = np.concatenate([same_d, diff_d]).astype(np.float32)
    corr = np.concatenate([same_c, diff_c]).astype(np.float32)
    idx = np.zeros(len(dist), dtype=int)
    metrics = _PairMetrics(i=idx, j=idx.copy(), dist=dist, corr=corr)

    model = fit_psame_model([metrics], model="joint", max_distance_px=10)
    assert model is not None
    assert model.p_same(same_d, same_c).mean() > 0.8
    assert model.p_same(diff_d, diff_c).mean() < 0.2


def test_probabilistic_registration_end_to_end():
    pytest.importorskip("sklearn")
    # A dense field so the neighbour search captures both same-cell pairs and
    # different-cell pairs — the P_same mixture needs both modes to fit.
    dims = (120, 120)
    rng = np.random.default_rng(7)
    base = _sample_centers(40, dims, 9, rng)
    n_keep = 34

    m1 = _make_model(base, dims, rng=rng, amp_noise=0.02)
    kept = [(cy + rng.normal(0, 0.4), cx + rng.normal(0, 0.4))
            for cy, cx in base[:n_keep]]
    added = _sample_centers(4, dims, 9, rng, existing=base)
    m2 = _make_model(kept + added, dims, rng=rng, amp_noise=0.02)

    res = register_sessions([m1, m2], align="none", probabilistic=True,
                            model="joint", max_distance_px=13, p_same_thr=0.5)

    assert res.p_same_threshold == 0.5
    cmap = res.cell_to_index_map
    both = cmap[(cmap >= 0).all(axis=1)]
    correct = sum(1 for i, j in both if i == j and i < n_keep)
    wrong = sum(1 for i, j in both if not (i == j and i < n_keep))
    assert correct >= 30
    assert wrong == 0


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path):
    dims = (60, 60)
    base = [(20, 20), (20, 40), (40, 30)]
    res = register_sessions([_make_model(base, dims), _make_model(base, dims)],
                            align="none", max_distance_px=6, dist_thr_px=3, corr_thr=0.6)
    res.save(tmp_path / "reg")
    loaded = CellRegResult.load(tmp_path / "reg")

    assert np.array_equal(loaded.cell_to_index_map, res.cell_to_index_map)
    assert np.allclose(loaded.transforms, res.transforms)
    assert loaded.n_sessions == res.n_sessions
    assert loaded.dims == res.dims
    for a, b in zip(loaded.aligned_centroids, res.aligned_centroids):
        assert np.allclose(a, b)


def test_register_requires_two_sessions():
    dims = (40, 40)
    with pytest.raises(ValueError):
        register_sessions([_make_model([(20, 20)], dims)])


def test_mismatched_dims_raises():
    with pytest.raises(ValueError):
        register_sessions([_make_model([(20, 20)], (40, 40)),
                           _make_model([(20, 20)], (50, 50))], align="none")
