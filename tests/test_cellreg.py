"""Tests for the CellReg port (minicnmfe.cellreg): positive controls with known answers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from minicnmfe import cellreg as cr

H, W = 128, 140


def blob(cy, cx, sigma=1.8, dims=(H, W)):
    yy, xx = np.mgrid[:dims[0], :dims[1]]
    img = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    img[img < 0.05] = 0
    return img


def model_from(centres, dims=(H, W), sigma=1.8):
    cols = [blob(cy, cx, sigma, dims).ravel() for cy, cx in centres]
    A = sp.csc_matrix(np.stack(cols, axis=1)) if cols else sp.csc_matrix((dims[0] * dims[1], 0))
    return SimpleNamespace(A=A, dims=dims, accepted_mask=None)


def random_centres(rng, K, margin=14, min_sep=7.0):
    pts = []
    while len(pts) < K:
        p = rng.uniform([margin, margin], [H - margin, W - margin])
        if all(np.hypot(*(p - q)) >= min_sep for q in pts):
            pts.append(p)
    return np.asarray(pts)


def rotate_pts(pts, theta, center):
    t = np.deg2rad(theta)
    y, x = pts[:, 0] - center[0], pts[:, 1] - center[1]
    return np.column_stack([np.cos(t) * y + np.sin(t) * x + center[0],
                            -np.sin(t) * y + np.cos(t) * x + center[1]])


# --------------------------------------------------------------------------- primitives

def test_normxcorr2_matches_bruteforce():
    rng = np.random.default_rng(0)
    T = rng.random((5, 6))
    A = rng.random((7, 8))
    A[:3, :3] = 0                      # a constant region -> zero-variance windows
    C = cr.normxcorr2(T, A)
    Ht, Wt = T.shape
    Ap = np.pad(A, ((Ht - 1, Ht - 1), (Wt - 1, Wt - 1)))
    ref = np.zeros(C.shape)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            w = Ap[i:i + Ht, j:j + Wt]
            a, t = w - w.mean(), T - T.mean()
            d = np.sqrt((a * a).sum() * (t * t).sum())
            ref[i, j] = (a * t).sum() / d if d > 1e-12 else 0.0
    np.testing.assert_allclose(C, ref, atol=1e-9)


def test_normxcorr2_peak_position_is_the_shift():
    img = blob(40, 50, 3, (90, 100)) + blob(60, 30, 3, (90, 100))
    moved = cr.transform_image(img, 0.0, (-5, 7), (0, 0))     # content moved by (+5, -7)
    C = cr.normxcorr2(img, moved)
    i, j = np.unravel_index(np.argmax(C), C.shape)
    assert (i - (90 - 1), j - (100 - 1)) == (5, -7)


def test_gaussfit_recovers_mean():
    x = np.arange(-10, 11)
    y = np.exp(-(x - 1.3) ** 2 / (2 * 2.0 ** 2))
    sigma, mu = cr.gaussfit(x, y / y.sum(), 1.0, 0.0)
    assert abs(mu - 1.3) < 0.02 and abs(sigma - 2.0) < 0.05


def test_corr2_equals_pearson_over_whole_image():
    a, b = blob(30.2, 40.7), blob(31.5, 42.0)
    s = cr._load_session(model_from([(30.2, 40.7), (31.5, 42.0)]))
    r = cr._corr2(*[s.rows()[k] for k in (0, 1)], H * W)
    assert abs(r - np.corrcoef(a.ravel(), b.ravel())[0, 1]) < 1e-9


def test_corr2_of_non_overlapping_footprints_is_near_zero():
    s = cr._load_session(model_from([(30, 40), (30, 60)]))
    r = cr._corr2(s.rows()[0], s.rows()[1], H * W)
    assert -0.01 < r <= 0.0


def test_centroids_are_subpixel():
    pts = np.array([[30.3, 40.7], [70.6, 90.2], [100.0, 20.45]])
    s = cr._load_session(model_from(pts))
    c = cr.compute_centroids(s, 7.0)
    np.testing.assert_allclose(c, pts, atol=0.1)


# --------------------------------------------------------------------------- alignment

def _sessions(pts_list):
    S = [cr._load_session(model_from(p)) for p in pts_list]
    for s in S:
        s.centroids = cr.compute_centroids(s, 7.0)
    return S


def test_alignment_recovers_translation():
    rng = np.random.default_rng(1)
    pts = random_centres(rng, 60)
    shift = np.array([6.4, -4.2])
    keep = np.all((pts + shift > 3) & (pts + shift < [H - 3, W - 3]), axis=1)
    S = _sessions([pts, pts[keep] + shift])
    out, al = cr.align_sessions(S, reference=0, cell_px=7.0, rotations=False)
    assert al.bad_sessions == []
    np.testing.assert_allclose(al.shifts[1], shift, atol=0.3)
    np.testing.assert_allclose(out[1].centroids, pts[keep], atol=0.3)


def test_alignment_recovers_rotation():
    rng = np.random.default_rng(2)
    pts = random_centres(rng, 70)
    center = (H / 2 - 1, W / 2 - 1)
    rot = rotate_pts(pts, 4.0, center)
    keep = np.all((rot > 3) & (rot < [H - 3, W - 3]), axis=1)
    S = _sessions([pts, rot[keep]])
    for fix in (False, True):
        out, al = cr.align_sessions(S, reference=0, cell_px=7.0, rotations=True, maximal_rotation=8,
                                    fix_upstream_slips=fix)
        assert abs(al.rotations[1] + 4.0) < 0.3  # the correction undoes the +4 deg
    # with one rotation centre (the fix) the cells land back on their positions; the
    # source's swapped centre on this non-square field is absorbed only partly
    np.testing.assert_allclose(out[1].centroids, pts[keep], atol=0.5)


def _rotate_image_interp(img, theta, center):
    """Literal rotate_image_interp.m + interpolate_pixel_value.m (1-based, no translation)."""
    t = np.deg2rad(theta)
    N, M = img.shape
    Ti = np.linalg.inv(np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]).T)
    out = np.zeros(img.shape)

    def at(r, c):
        return img[r - 1, c - 1] if 1 <= r <= N and 1 <= c <= M else 0.0
    for p in range(1, N + 1):
        for q in range(1, M + 1):
            w = Ti @ np.array([p - center[0], q - center[1]]) + np.array(center)
            w[w < 0] = 0
            y0, x0, y1, x1 = int(np.floor(w[0])), int(np.floor(w[1])), int(np.ceil(w[0])), int(np.ceil(w[1]))
            a, b = w[1] - x0, w[0] - y0
            out[p - 1, q - 1] = ((1 - a) * ((1 - b) * at(y0, x0) + b * at(y1, x0))
                                 + a * ((1 - b) * at(y0, x1) + b * at(y1, x1)))
    return out


def test_rotation_centres_reproduce_the_source_on_a_non_square_field():
    img = np.random.default_rng(7).random((30, 41))
    c = (30 / 2, 41 / 2)                          # center_of_FOV = [Ny/2, Nx/2]
    search, apply_src = cr.rotation_centers(img.shape, fix_upstream_slips=False)
    # the search rotates the image as it is; the application rotates the transpose by -theta
    np.testing.assert_allclose(cr.transform_image(img, 3.0, (0, 0), search),
                               _rotate_image_interp(img, 3.0, c), atol=1e-12)
    np.testing.assert_allclose(cr.transform_image(img, 3.0, (0, 0), apply_src),
                               _rotate_image_interp(img.T, -3.0, c).T, atol=1e-12)
    assert cr.rotation_centers(img.shape, fix_upstream_slips=True)[1] == search


def test_unrelated_session_is_flagged():
    rng = np.random.default_rng(3)
    S = _sessions([random_centres(rng, 40), random_centres(np.random.default_rng(99), 40)])
    _, al = cr.align_sessions(S, reference=0, cell_px=7.0, rotations=False)
    assert al.bad_sessions == [1]


# --------------------------------------------------------------------------- all-pairs alignment

CENTER = (H / 2 - 1, W / 2 - 1)


def _chain(n=5, per=50, step=18, seed=4):
    """Sessions that see a sliding window of one cell population: neighbours share
    (per - step) cells, sessions >= per/step apart share none. Session k's content is
    the population rotated by angle[k] about CENTER and moved by shift[k]."""
    rng = np.random.default_rng(seed)
    pop = random_centres(rng, per + step * (n - 1), margin=18, min_sep=6.0)
    angle = np.r_[0.0, rng.uniform(-3, 3, n - 1)]
    shift = np.r_[[[0.0, 0.0]], rng.uniform(-6, 6, (n - 1, 2))]
    pts = []
    for k in range(n):
        p = rotate_pts(pop[k * step:k * step + per], angle[k], CENTER) + shift[k]
        pts.append(p[np.all((p > 4) & (p < [H - 4, W - 4]), axis=1)])
    return pop, angle, shift, pts


def test_estimate_pair_equals_align_sessions():
    rng = np.random.default_rng(2)
    pts = random_centres(rng, 70)
    rot = rotate_pts(pts, 4.0, CENTER) + [2.5, -3.0]
    S = _sessions([pts, rot[np.all((rot > 3) & (rot < [H - 3, W - 3]), axis=1)]])
    _, al = cr.align_sessions(S, reference=0, cell_px=7.0, maximal_rotation=8)
    fp = [cr.footprints_projection(s) for s in S]
    cp = [cr.centroids_projection(s.centroids, s.dims) for s in S]
    e = cr.estimate_pair(fp[0], cp[0], fp[1], cp[1], S[1].centroids, S[0].dims, 7.0,
                         maximal_rotation=8)
    assert e.passed and al.bad_sessions == []
    assert e.theta == al.rotations[1] and e.shift == tuple(al.shifts[1])


def test_fit_recovers_transforms_and_drops_an_inconsistent_edge():
    rng = np.random.default_rng(5)
    n = 5
    T = np.c_[np.r_[0.0, rng.uniform(-4, 4, n - 1)], np.r_[[[0.0, 0.0]], rng.uniform(-20, 20, (n - 1, 2))]]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    est = []
    for i, j in pairs:                 # exact edges: T_j = compose(T_i, e)
        t = T[j, 0] - T[i, 0]
        est.append(np.r_[t, np.linalg.solve(cr._Q(T[i, 0]), T[j, 1:] - T[i, 1:])])
    est = np.array(est)
    bad = pairs.index((1, 4))
    est[bad, 1] += 8.0
    Tf, inlier, res, anchor = cr.fit_session_transforms(n, pairs, est, np.ones(len(pairs), bool),
                                                        anchor=0)
    assert anchor == 0 and not inlier[bad] and inlier.sum() == len(pairs) - 1
    np.testing.assert_allclose(Tf, T, atol=1e-5)


def test_all_pairs_places_sessions_that_fail_against_the_reference():
    pop, angle, shift, pts = _chain()
    S = _sessions(pts)
    _, al_ref = cr.align_sessions(S, reference=0, cell_px=7.0, maximal_rotation=8)
    assert {3, 4} <= set(al_ref.bad_sessions)          # no cells shared with session 0
    al = cr.align_all_pairs(S, cell_size_px=7.0, maximal_rotation=8)
    assert al.anchor == 0 and al.unplaced == []
    k = len(S) - 1                     # warp_image: a spot at one cell of session k lands where
    c = S[k].centroids[0]              # that cell sits in session 0's frame
    back = al.warp_image(k, blob(*c, sigma=2.0))
    yy, xx = np.mgrid[:H, :W]
    w = np.where(back > 0.5 * back.max(), back, 0)
    got = np.array([(yy * w).sum(), (xx * w).sum()]) / w.sum()
    truth = rotate_pts((c - shift[k])[None], -angle[k], CENTER)[0]
    assert np.hypot(*(got - truth)) < 0.3
    for k in range(1, len(S)):
        back = cr._forward_points(S[k].centroids, al.transforms[k, 0], al.transforms[k, 1:], CENTER)
        truth = rotate_pts(S[k].centroids - shift[k], -angle[k], CENTER)
        assert np.median(np.hypot(*(back - truth).T)) < 0.3, k


def test_unrelated_session_is_unplaced_and_anchor_follows_the_largest_block():
    _, _, _, pts = _chain(n=4)
    rng = np.random.default_rng(99)
    other = random_centres(rng, 45)
    S = _sessions([other] + pts)                       # session 0 shares nothing
    al = cr.align_all_pairs(S, cell_size_px=7.0, maximal_rotation=8)
    assert al.anchor == 1 and al.unplaced == [0]
    assert np.isnan(al.transforms[0]).all() and (al.transforms[1] == 0).all()


def test_register_with_all_pairs_alignment_groups_cells_by_identity():
    rng = np.random.default_rng(11)
    n, per, step = 5, 50, 15
    pop = random_centres(rng, per + step * (n - 1), margin=18, min_sep=6.0)
    angle = np.r_[0.0, rng.uniform(-2, 2, n - 1)]
    shift = np.r_[[[0.0, 0.0]], rng.uniform(-5, 5, (n - 1, 2))]
    models, truth = [], []
    for k in range(n):
        ids = np.arange(k * step, k * step + per)
        p = rotate_pts(pop[ids], angle[k], CENTER) + shift[k] + rng.normal(0, 0.3, (per, 2))
        ok = np.all((p > 4) & (p < [H - 4, W - 4]), axis=1)
        models.append(model_from(p[ok]))
        truth.append(ids[ok])
    res = cr.register_sessions(models, cell_size_px=7.0, alignment="all_pairs", maximal_rotation=8)
    assert res.bad_sessions == []
    correct = _pairs(res.cell_to_index_map, truth)          # one entry per registered pair
    n_true = sum(len(set(truth[a]) & set(truth[b])) for a in range(n) for b in range(a + 1, n))
    assert correct.mean() >= 0.95 and correct.sum() >= 0.8 * n_true
    # every cell of every session has exactly one global row
    for s in range(n):
        col = res.cell_to_index_map[:, s]
        assert sorted(col[col >= 0]) == list(range(len(truth[s])))


# --------------------------------------------------------------------------- models

def test_choose_best_model_source_uses_the_correlation_index_for_both():
    c = np.linspace(0.05, 0.95, 10)
    mk = lambda f, ps, same, diff: cr.Model(f, c, ps, same, diff, same, 0.0, 0.5)
    spatial = mk("spatial", np.r_[np.zeros(7), 0.5, 1, 1], np.r_[np.zeros(7), np.ones(3)],
                 np.r_[np.ones(7), np.zeros(3)])                # P_same = 0.5 at index 7
    centroid = mk("centroid", np.r_[1, 0.5, np.zeros(8)], np.r_[np.ones(2), np.zeros(8)],
                  np.r_[np.zeros(2), np.ones(8)])               # P_same = 0.5 at index 1
    cr.choose_best_model(centroid, spatial, fix_upstream_slips=True)
    fp_fixed = centroid.false_positive
    cr.choose_best_model(centroid, spatial, fix_upstream_slips=False)
    ic = int(np.argmin(np.abs(0.5 - spatial.p_same)))
    assert centroid.false_positive == pytest.approx(centroid.different[:ic + 1].sum() / centroid.different.sum())
    assert centroid.false_positive > fp_fixed


def test_default_reproduces_the_source():
    import inspect
    for f in (cr.register_sessions, cr.align_sessions, cr.estimate_pair, cr.compute_centroids,
              cr.fit_spatial_correlation_model, cr.fit_centroid_distance_model, cr.choose_best_model):
        assert inspect.signature(f).parameters["fix_upstream_slips"].default is False, f.__name__

def test_centroid_model_psame_not_zero_at_zero_distance():
    rng = np.random.default_rng(4)
    same = rng.lognormal(np.log(1.0), 0.5, 3000)
    diff = rng.uniform(0, 8, 6000)
    d = np.concatenate([same, diff])
    d = d[d < 8.0]
    _, dc, _ = cr.number_of_bins(200, 5, 8.0)
    m = cr.fit_centroid_distance_model(d, dc, 7.0)
    assert m.p_same[0] == m.p_same[1] and m.p_same[0] > 0.5
    assert m.lookup([0.0])[0] > 0.5


def test_spatial_model_suppresses_low_correlations():
    rng = np.random.default_rng(5)
    same = 1 - rng.lognormal(np.log(0.15), 0.5, 3000)
    diff = rng.beta(1.2, 6.0, 6000)
    c = np.clip(np.concatenate([same, diff]), 1e-6, 1 - 1e-6)
    _, _, cc = cr.number_of_bins(200, 5, 8.0)
    m = cr.fit_spatial_correlation_model(c, cc)
    assert m.lookup([0.05])[0] < 0.05          # no spurious P_same ~ 1 at the left edge
    assert m.lookup([0.95])[0] > 0.95
    assert 0.3 < m.intersection < 0.9


# --------------------------------------------------------------------------- end to end

def _synthetic_days(seed, n_days=4, K=60, jitter=0.4, drop=0.15):
    rng = np.random.default_rng(seed)
    pts = random_centres(rng, K)
    models, truth = [], []
    for d in range(n_days):
        g = np.zeros(2) if d == 0 else rng.uniform(-6, 6, 2)
        ids = [k for k in range(K) if d == 0 or rng.random() > drop]
        p = pts[ids] + g + rng.normal(0, jitter, (len(ids), 2))
        ok = np.all((p > 4) & (p < [H - 4, W - 4]), axis=1)
        models.append(model_from(p[ok]))
        truth.append(np.asarray(ids)[ok])
    return models, truth


def _pairs(cmap, truth):
    out = []
    for row in cmap:
        pres = [(s, int(k)) for s, k in enumerate(row) if k >= 0]
        for a in range(len(pres)):
            for b in range(a + 1, len(pres)):
                (s1, k1), (s2, k2) = pres[a], pres[b]
                out.append(truth[s1][k1] == truth[s2][k2])
    return np.asarray(out)


def test_end_to_end_recovers_identities():
    models, truth = _synthetic_days(6)
    res = cr.register_sessions(models, cell_size_px=7.0, alignment="translations")
    assert res.bad_sessions == []
    ok = _pairs(res.cell_to_index_map, truth)
    n_true = sum(len(set(truth[a]) & set(truth[b]))
                 for a in range(len(truth)) for b in range(a + 1, len(truth)))
    assert ok.mean() >= 0.97                  # precision
    assert ok.sum() / n_true >= 0.95          # recall


def test_save_load_roundtrip(tmp_path):
    models, _ = _synthetic_days(7, n_days=3)
    res = cr.register_sessions(models, cell_size_px=7.0, alignment="translations")
    res.save(tmp_path)
    back = cr.CellRegResult.load(tmp_path)
    np.testing.assert_array_equal(back.cell_to_index_map, res.cell_to_index_map)
    np.testing.assert_allclose(back.transforms, res.transforms)
    assert back.model_feature == res.model_feature


def test_register_requires_two_sessions():
    with pytest.raises(ValueError):
        cr.register_sessions([model_from([(30, 30)])])


def test_keep_mode_registers_a_flagged_session():
    rng = np.random.default_rng(3)
    pts = random_centres(rng, 40)
    other = random_centres(np.random.default_rng(99), 40)
    models = [model_from(pts), model_from(pts + 0.2), model_from(other)]
    excl = cr.register_sessions(models, cell_size_px=7.0, alignment="translations")
    keep = cr.register_sessions(models, cell_size_px=7.0, alignment="translations",
                                on_bad_alignment="keep")
    assert excl.bad_sessions == [2] and keep.bad_sessions == [2]
    assert (excl.cell_to_index_map[:, 2] < 0).all()
    assert (keep.cell_to_index_map[:, 2] >= 0).any()


def test_old_api_arguments_go_to_cellreg_legacy():
    from minicnmfe import cellreg_legacy
    models, _ = _synthetic_days(7, n_days=3)
    old = dict(align="translation", accepted_only=False, max_distance_px=12.0, dist_thr_px=5.0)
    with pytest.warns(FutureWarning, match="cellreg_legacy"):
        res = cr.register_sessions(models, **old)
    ref = cellreg_legacy.register_sessions(models, **old)
    assert isinstance(res, cellreg_legacy.CellRegResult)
    np.testing.assert_array_equal(res.cell_to_index_map, ref.cell_to_index_map)
