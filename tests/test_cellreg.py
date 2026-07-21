"""Tests for cross-session cell registration (minicnmfe.cellreg)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from minicnmfe.cellreg import (
    CellRegResult,
    _load_session,
    _PairMetrics,
    align_sessions,
    choose_best_model,
    cluster_cells_iterative,
    compute_nn_nnn_distributions,
    compute_registration_scores,
    fit_cellreg_model,
    fit_centroid_distance_model,
    fit_psame_model,
    fit_spatial_correlation_model,
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
# CellReg NN/NNN candidate distributions
# ---------------------------------------------------------------------------

def test_nn_nnn_distributions_separate_same_from_different():
    # a dense field shared across two sessions (small jitter) => the NN of each
    # cell is its true match (high corr / small dist); the NNN are neighbours.
    dims = (120, 120)
    rng = np.random.default_rng(11)
    base = _sample_centers(40, dims, 9, rng)
    m1 = _make_model(base, dims, rng=rng, amp_noise=0.02)
    jit = [(cy + rng.normal(0, 0.4), cx + rng.normal(0, 0.4)) for cy, cx in base]
    m2 = _make_model(jit, dims, rng=rng, amp_noise=0.02)

    loaded = [_load_session(m1), _load_session(m2)]
    fps, cents, _ = align_sessions(loaded, mode="none")
    nd = compute_nn_nnn_distributions(fps, cents, max_distance_px=10)

    assert nd.nn_corr.size > 0 and nd.nnn_corr.size > 0
    # same-cell candidates are more correlated and closer than different cells
    assert nd.nn_corr.mean() > nd.nnn_corr.mean() + 0.3
    assert nd.nn_dist.mean() < nd.nnn_dist.mean()


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

    res = register_sessions([m1, m2], align="none",
                            registration_approach="threshold", probabilistic=True,
                            model="joint", max_distance_px=13, p_same_thr=0.5)

    assert res.p_same_threshold == 0.5
    cmap = res.cell_to_index_map
    both = cmap[(cmap >= 0).all(axis=1)]
    correct = sum(1 for i, j in both if i == j and i < n_keep)
    wrong = sum(1 for i, j in both if not (i == j and i < n_keep))
    assert correct >= 30
    assert wrong == 0


# ---------------------------------------------------------------------------
# CellReg P_same models (lognormal+beta / lognormal+linear*sigmoid) + threshold
# ---------------------------------------------------------------------------

def test_spatial_correlation_model_threshold_between_clouds():
    rng = np.random.default_rng(5)
    same = np.clip(rng.normal(0.85, 0.06, 600), -1, 1)      # same-cell: high corr
    diff = np.clip(rng.normal(-0.6, 0.15, 6000), -1, 1)     # different: negative
    model = fit_spatial_correlation_model(np.concatenate([same, diff]))
    assert model is not None
    # data-driven threshold sits between the two clouds
    assert -0.6 < model.threshold < 0.85
    # P_same is high for same-cell corr, low for different-cell corr
    assert model.p_same_from_corr(same).mean() > 0.8
    assert model.p_same_from_corr(diff).mean() < 0.2


def test_centroid_distance_model_threshold_between_clouds():
    rng = np.random.default_rng(6)
    same = np.abs(rng.normal(0.0, 0.8, 600)) + 0.2          # same-cell: ~0-2 px
    diff = rng.uniform(6, 12, 6000)                          # different: far
    model = fit_centroid_distance_model(
        np.concatenate([same, diff]), max_distance_px=12.0)
    assert model is not None
    assert 2.0 < model.threshold_px < 6.0
    assert model.p_same_from_dist(same).mean() > 0.8
    assert model.p_same_from_dist(diff).mean() < 0.2


def test_choose_best_model_picks_lower_cost():
    rng = np.random.default_rng(8)
    # correlation barely separates (broad, overlapping); distance separates cleanly
    same_c = np.clip(rng.normal(0.3, 0.25, 600), -1, 1)
    diff_c = np.clip(rng.normal(0.0, 0.25, 4000), -1, 1)
    same_d = np.abs(rng.normal(0.0, 0.6, 600)) + 0.2
    diff_d = rng.uniform(7, 12, 4000)

    sm = fit_spatial_correlation_model(np.concatenate([same_c, diff_c]))
    cm = fit_centroid_distance_model(np.concatenate([same_d, diff_d]),
                                     max_distance_px=12.0)
    assert sm is not None and cm is not None
    # the cleanly-separating distance model should win on FP+FN+MSE cost
    assert choose_best_model(sm, cm) == "centroid"
    assert choose_best_model(sm, None) == "spatial"
    assert choose_best_model(None, cm) == "centroid"
    assert choose_best_model(None, None) is None


def test_auto_picks_the_more_separable_feature():
    rng = np.random.default_rng(9)
    n_same, n_diff = 600, 4000

    # Case A: distance separates cleanly, correlation overlaps -> auto: centroid
    dist_a = np.concatenate([np.abs(rng.normal(0, 0.6, n_same)) + 0.2,
                             rng.uniform(7, 12, n_diff)])
    corr_a = np.concatenate([np.clip(rng.normal(0.4, 0.3, n_same), -1, 1),
                             np.clip(rng.normal(0.2, 0.3, n_diff), -1, 1)])
    m_a = fit_cellreg_model(dist_a, corr_a, feature="auto", max_distance_px=12.0)
    assert m_a is not None and m_a.feature == "centroid"

    # Case B: correlation separates cleanly, distance overlaps -> auto: spatial
    dist_b = np.concatenate([rng.uniform(0, 6, n_same), rng.uniform(0, 6, n_diff)])
    corr_b = np.concatenate([np.clip(rng.normal(0.85, 0.05, n_same), -1, 1),
                             np.clip(rng.normal(-0.5, 0.1, n_diff), -1, 1)])
    m_b = fit_cellreg_model(dist_b, corr_b, feature="auto", max_distance_px=12.0)
    assert m_b is not None and m_b.feature == "spatial"


def test_auto_registration_no_worse_than_fixed_features():
    # end-to-end: auto should register at least as well as the worse fixed choice
    dims = (120, 120)
    rng = np.random.default_rng(7)
    base = _sample_centers(40, dims, 9, rng)
    n_keep = 34
    m1 = _make_model(base, dims, rng=rng, amp_noise=0.02)
    kept = [(cy + rng.normal(0, 0.4), cx + rng.normal(0, 0.4))
            for cy, cx in base[:n_keep]]
    added = _sample_centers(4, dims, 9, rng, existing=base)
    m2 = _make_model(kept + added, dims, rng=rng, amp_noise=0.02)

    def correct(feature):
        res = register_sessions([m1, m2], align="none",
                                registration_approach="probabilistic_model",
                                psame_feature=feature, max_distance_px=13)
        cmap = res.cell_to_index_map
        both = cmap[(cmap >= 0).all(axis=1)]
        c = sum(1 for i, j in both if i == j and i < n_keep)
        w = sum(1 for i, j in both if not (i == j and i < n_keep))
        return c, w, res.params["psame_feature"]

    c_auto, w_auto, feat = correct("auto")
    c_sp, _, _ = correct("spatial")
    c_ce, _, _ = correct("centroid")
    assert feat in ("spatial", "centroid")
    assert w_auto == 0
    assert c_auto >= min(c_sp, c_ce)        # never the worse choice
    assert c_auto >= 30


def test_default_uses_probabilistic_model():
    # the package default (no registration_approach) is now the CellReg model
    dims = (120, 120)
    rng = np.random.default_rng(7)
    base = _sample_centers(40, dims, 9, rng)
    models = [_make_model([(cy + rng.normal(0, 0.4), cx + rng.normal(0, 0.4))
                           for cy, cx in base], dims, rng=rng, amp_noise=0.02)
              for _ in range(2)]
    res = register_sessions(models, align="none", max_distance_px=13)
    assert res.params["registration_approach"] == "probabilistic_model"
    assert res.params["psame_feature"] in ("spatial", "centroid")
    assert res.cell_scores is not None


def test_probabilistic_model_registration_end_to_end():
    # same dense field as the GMM test, but using the CellReg data-driven model
    # (registration_approach="probabilistic_model") with NO hand-set corr_thr.
    dims = (120, 120)
    rng = np.random.default_rng(7)
    base = _sample_centers(40, dims, 9, rng)
    n_keep = 34
    m1 = _make_model(base, dims, rng=rng, amp_noise=0.02)
    kept = [(cy + rng.normal(0, 0.4), cx + rng.normal(0, 0.4))
            for cy, cx in base[:n_keep]]
    added = _sample_centers(4, dims, 9, rng, existing=base)
    m2 = _make_model(kept + added, dims, rng=rng, amp_noise=0.02)

    for feature in ("spatial", "centroid"):
        res = register_sessions([m1, m2], align="none",
                                registration_approach="probabilistic_model",
                                psame_feature=feature, max_distance_px=13)
        assert res.params["registration_approach"] == "probabilistic_model"
        assert res.params["data_driven_threshold"] is not None
        cmap = res.cell_to_index_map
        both = cmap[(cmap >= 0).all(axis=1)]
        correct = sum(1 for i, j in both if i == j and i < n_keep)
        wrong = sum(1 for i, j in both if not (i == j and i < n_keep))
        assert correct >= 30, (feature, correct)
        assert wrong == 0, (feature, wrong)


# ---------------------------------------------------------------------------
# Iterative joint clustering (Step 4)
# ---------------------------------------------------------------------------

class _StubModel:
    """Distance-only P_same: ~1 when close, ~0 when far."""
    feature = "centroid"

    def p_same(self, dist, corr):
        return (np.asarray(dist, dtype=float) < 4.0).astype(float)


def _aligned(models):
    loaded = [_load_session(m) for m in models]
    return align_sessions(loaded, mode="none")


def test_iterative_clustering_merges_singletons():
    dims = (96, 96)
    rng = np.random.default_rng(2)
    base = _sample_centers(8, dims, 16, rng)
    models = [_make_model(base, dims) for _ in range(3)]
    fps, cents, _ = _aligned(models)

    # start from a fully fragmented map (every cell its own singleton row)
    rows = []
    for s in range(3):
        for k in range(8):
            row = [-1, -1, -1]
            row[s] = k
            rows.append(row)
    broken = np.asarray(rows, dtype=int)

    out = cluster_cells_iterative(broken, fps, cents, _StubModel(),
                                  max_distance_px=6.0)
    # the 24 singletons collapse to 8 clusters, each present in all 3 sessions
    assert out.shape[0] == 8
    assert (out >= 0).all()


def test_iterative_clustering_keeps_clean_partition():
    dims = (96, 96)
    rng = np.random.default_rng(2)
    base = _sample_centers(8, dims, 16, rng)
    models = [_make_model(base, dims) for _ in range(3)]
    fps, cents, _ = _aligned(models)
    correct = np.tile(np.arange(8)[:, None], (1, 3))   # already-correct map
    out = cluster_cells_iterative(correct, fps, cents, _StubModel(),
                                  max_distance_px=6.0)
    assert out.shape[0] == 8
    assert (out >= 0).all()


def test_three_session_clustering_model_path():
    # the model + iterative-clustering path tracks a dense N=3 field (dense so the
    # P_same mixture sees both same- and different-cell candidates and calibrates)
    dims = (120, 120)
    rng = np.random.default_rng(2)
    base = _sample_centers(40, dims, 9, rng)
    models = [_make_model([(cy + rng.normal(0, 0.4), cx + rng.normal(0, 0.4))
                           for cy, cx in base], dims, rng=rng, amp_noise=0.02)
              for _ in range(3)]
    res = register_sessions(models, align="none",
                            registration_approach="probabilistic_model",
                            psame_feature="auto", max_distance_px=13)
    assert res.params["clustering"] == "iterative"
    # most of the 40 cells tracked across all 3 sessions, no gross over-merge
    assert res.n_registered(3) >= 33
    assert res.n_global <= 55


# ---------------------------------------------------------------------------
# Registration-confidence outputs (Step 5)
# ---------------------------------------------------------------------------

def test_registration_confidence_outputs_populated():
    dims = (120, 120)
    rng = np.random.default_rng(7)
    base = _sample_centers(40, dims, 9, rng)
    n_keep = 34
    m1 = _make_model(base, dims, rng=rng, amp_noise=0.02)
    kept = [(cy + rng.normal(0, 0.4), cx + rng.normal(0, 0.4))
            for cy, cx in base[:n_keep]]
    added = _sample_centers(4, dims, 9, rng, existing=base)
    m2 = _make_model(kept + added, dims, rng=rng, amp_noise=0.02)

    res = register_sessions([m1, m2], align="none",
                            registration_approach="probabilistic_model",
                            psame_feature="auto", max_distance_px=13)
    assert res.cell_scores is not None and res.cell_scores.shape[0] == res.n_global
    # registered (multi-session) cells have a score; singletons are NaN
    reg = (res.cell_to_index_map >= 0).sum(1) >= 2
    assert np.all(np.isfinite(res.cell_scores[reg]))
    assert np.all(np.isnan(res.cell_scores[~reg]))
    # registered matches are confident; few are ambiguous
    assert np.nanmean(res.cell_scores[reg]) > 0.7
    assert res.p_same_registered_pairs.min() >= 0.5      # all >= p_same_thr
    assert 0.0 <= res.uncertain_fraction <= 1.0
    assert 0.0 <= res.model_false_positive <= 1.0
    assert 0.0 <= res.model_false_negative <= 1.0


def test_confidence_none_on_threshold_path():
    dims = (60, 60)
    base = [(20, 20), (20, 40), (40, 30)]
    res = register_sessions([_make_model(base, dims), _make_model(base, dims)],
                            align="none", registration_approach="threshold",
                            max_distance_px=6, dist_thr_px=3, corr_thr=0.6)
    assert res.cell_scores is None
    assert res.uncertain_fraction is None
    assert res.model_false_positive is None


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


def test_save_load_roundtrip_with_confidence(tmp_path):
    dims = (120, 120)
    rng = np.random.default_rng(7)
    base = _sample_centers(40, dims, 9, rng)
    m1 = _make_model(base, dims, rng=rng, amp_noise=0.02)
    m2 = _make_model([(cy + rng.normal(0, 0.4), cx + rng.normal(0, 0.4))
                      for cy, cx in base], dims, rng=rng, amp_noise=0.02)
    res = register_sessions([m1, m2], align="none",
                            registration_approach="probabilistic_model",
                            psame_feature="auto", max_distance_px=13)
    res.save(tmp_path / "reg")
    loaded = CellRegResult.load(tmp_path / "reg")
    assert np.allclose(loaded.cell_scores, res.cell_scores, equal_nan=True)
    assert np.allclose(loaded.p_same_registered_pairs, res.p_same_registered_pairs)
    assert loaded.uncertain_fraction == res.uncertain_fraction
    assert loaded.model_false_positive == res.model_false_positive


def test_register_requires_two_sessions():
    dims = (40, 40)
    with pytest.raises(ValueError):
        register_sessions([_make_model([(20, 20)], dims)])


def test_mismatched_dims_raises():
    with pytest.raises(ValueError):
        register_sessions([_make_model([(20, 20)], (40, 40)),
                           _make_model([(20, 20)], (50, 50))], align="none")
