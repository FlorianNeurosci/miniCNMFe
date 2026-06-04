"""Tests for the downsample-once front end.

Covers the streaming block-mean (``downsample_movie``), the parameter rescaling
(``CNMFeParams.downscaled``), and an end-to-end recovery check: downsample a
synthetic movie, then run the full extraction on the smaller movie and confirm
neurons are still recovered.
"""

import json

import numpy as np
import scipy.sparse as sp

from minicnmfe.downsample import downsample_movie, upsample_footprints, upsample_traces
from minicnmfe.io import open_zarr, save_zarr
from minicnmfe.pipeline import CNMFe, CNMFeParams


def _block_mean(movie: np.ndarray, ssub: int, tsub: int) -> np.ndarray:
    """Reference (non-streaming) block-mean for comparison."""
    T, H, W = movie.shape
    Tu, Hu, Wu = (T // tsub) * tsub, (H // ssub) * ssub, (W // ssub) * ssub
    m = movie[:Tu, :Hu, :Wu].astype(np.float32)
    m = m.reshape(Tu // tsub, tsub, Hu, Wu).mean(axis=1)
    m = m.reshape(Tu // tsub, Hu // ssub, ssub, Wu // ssub, ssub).mean(axis=(2, 4))
    return m


# --------------------------------------------------------------------------
# downsample_movie
# --------------------------------------------------------------------------

def test_block_mean_values_and_shape(tmp_path):
    rng = np.random.default_rng(0)
    movie = rng.standard_normal((8, 6, 6)).astype(np.float32)
    save_zarr(movie, tmp_path / "src.zarr")

    out = downsample_movie(tmp_path / "src.zarr", tmp_path / "ds.zarr",
                           ssub=2, tsub=2, verbose=False)
    expected = _block_mean(movie, ssub=2, tsub=2)
    assert out.shape == expected.shape == (4, 3, 3)
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-5, atol=1e-5)


def test_trims_non_divisible_dims(tmp_path):
    movie = np.ones((5, 5, 5), dtype=np.float32)
    save_zarr(movie, tmp_path / "src.zarr")
    out = downsample_movie(tmp_path / "src.zarr", tmp_path / "ds.zarr",
                           ssub=2, tsub=2, verbose=False)
    # 5 -> trim to 4 -> //2 = 2 on every axis.
    assert out.shape == (2, 2, 2)


def test_identity_when_factors_one(tmp_path):
    rng = np.random.default_rng(1)
    movie = rng.standard_normal((4, 4, 4)).astype(np.float32)
    save_zarr(movie, tmp_path / "src.zarr")
    out = downsample_movie(tmp_path / "src.zarr", tmp_path / "ds.zarr",
                           ssub=1, tsub=1, verbose=False)
    np.testing.assert_allclose(np.asarray(out), movie, rtol=1e-5, atol=1e-5)


def test_meta_sidecar_and_skip_if_exists(tmp_path):
    movie = np.ones((6, 4, 4), dtype=np.float32)
    save_zarr(movie, tmp_path / "src.zarr")
    downsample_movie(tmp_path / "src.zarr", tmp_path / "ds.zarr",
                     ssub=2, tsub=3, verbose=False)

    meta = json.loads((tmp_path / "ds_meta.json").read_text())
    assert meta["ssub"] == 2 and meta["tsub"] == 3
    assert meta["orig_dims"] == [4, 4] and meta["orig_T"] == 6
    assert meta["ds_dims"] == [2, 2] and meta["ds_T"] == 2

    # Second call with skip_if_exists returns the existing store unchanged.
    again = downsample_movie(tmp_path / "src.zarr", tmp_path / "ds.zarr",
                             ssub=2, tsub=3, skip_if_exists=True, verbose=False)
    assert again.shape == (2, 2, 2)


# --------------------------------------------------------------------------
# CNMFeParams.downscaled
# --------------------------------------------------------------------------

def test_downscaled_rescales_expected_fields():
    p = CNMFeParams(sigma=4.0, min_pixel=12, border_px=6, max_shift=(20, 20),
                    mc_gSig_filt=4.0, frame_rate_hz=30.0, decay_time_ms=140.0)
    d = p.downscaled(ssub=2, tsub=3)
    assert d.sigma == 2.0
    assert d.min_pixel == 3            # 12 // 2**2
    assert d.border_px == 3
    assert d.max_shift == (10, 10)
    assert d.mc_gSig_filt == 2.0
    assert d.frame_rate_hz == 10.0
    assert d.decay_time_ms == 140.0    # physical time, unchanged
    # Original is untouched.
    assert p.sigma == 4.0 and p.min_pixel == 12


def test_downscaled_handles_none_optionals():
    p = CNMFeParams(mc_gSig_filt=None, frame_rate_hz=None)
    d = p.downscaled(2, 2)
    assert d.mc_gSig_filt is None
    assert d.frame_rate_hz is None


# --------------------------------------------------------------------------
# End-to-end: downsample then extract
# --------------------------------------------------------------------------

def _spatial_downsample_A(A_true: np.ndarray, dims, ssub: int):
    """Block-mean each footprint column onto the downsampled grid."""
    H, W = dims
    Hu, Wu = (H // ssub) * ssub, (W // ssub) * ssub
    K = A_true.shape[1]
    cols = []
    for k in range(K):
        a = A_true[:, k].reshape(H, W)[:Hu, :Wu]
        a = a.reshape(Hu // ssub, ssub, Wu // ssub, ssub).mean(axis=(1, 3))
        cols.append(a.ravel())
    return np.stack(cols, axis=1)


def _best_match_corr(a_vec, B):
    """Max cosine similarity of column-vector a_vec against columns of B."""
    best = 0.0
    na = np.linalg.norm(a_vec)
    for j in range(B.shape[1]):
        nb = np.linalg.norm(B[:, j])
        if na > 0 and nb > 0:
            best = max(best, abs(float(a_vec @ B[:, j] / (na * nb))))
    return best


def test_end_to_end_downsample_then_extract(synth, tmp_path):
    """Downsample (ssub=2, tsub=2) then extract on the smaller movie; the
    ground-truth neurons must still be recoverable, both spatially and
    temporally, on the downsampled grid."""
    ssub, tsub = 2, 2
    movie = synth["movie"]
    dims = synth["dims"]
    save_zarr(movie, tmp_path / "src.zarr")

    ds = downsample_movie(tmp_path / "src.zarr", tmp_path / "ds.zarr",
                          ssub=ssub, tsub=tsub, verbose=False)
    H, W = dims
    assert ds.shape == (movie.shape[0] // tsub, H // ssub, W // ssub)

    params = CNMFeParams(
        sigma=3.0, min_corr=0.5, min_pnr=3.0,
        n_iter_main=1, n_iter_temporal=1, n_jobs=1,
    ).downscaled(ssub, tsub)

    model = CNMFe(params).fit(open_zarr(tmp_path / "ds.zarr"),
                              do_motion_correction=False)

    K_true = synth["A_true"].shape[1]
    assert 1 <= model.A.shape[1] <= 4 * K_true

    # Spatial: match downsampled ground-truth footprints to estimated ones.
    A_true_ds = _spatial_downsample_A(synth["A_true"], dims, ssub)
    A_est = np.asarray(model.A.todense())
    spatial_hits = sum(
        _best_match_corr(A_true_ds[:, k], A_est) > 0.5 for k in range(K_true)
    )
    assert spatial_hits >= (K_true + 1) // 2, (
        f"only {spatial_hits}/{K_true} footprints recovered after downsampling"
    )

    # Temporal: best-matched (C + YrA) trace vs temporally-binned ground truth.
    T = movie.shape[0]
    Tu = (T // tsub) * tsub
    C_true_ds = synth["C_true"][:, :Tu].reshape(K_true, Tu // tsub, tsub).mean(axis=2)
    C_est = model.C + model.YrA                       # (K_est, T_ds)
    best_trace = max(
        _best_match_corr(C_true_ds[k], C_est.T) for k in range(K_true)
    )
    assert best_trace > 0.6, f"best recovered trace corr only {best_trace:.3f}"


# --------------------------------------------------------------------------
# Upsampling helpers + CNMFe.upsample_to_native
# --------------------------------------------------------------------------

def _centroid(img):
    ys, xs = np.nonzero(img > img.max() * 0.5) if img.max() > 0 else (np.array([0]), np.array([0]))
    w = img[ys, xs]
    return np.array([(ys * w).sum() / w.sum(), (xs * w).sum() / w.sum()])


def test_upsample_footprints_nearest_is_block_inverse():
    # 2x2 footprint [[1,2],[3,4]] -> nearest 4x4 = each pixel replicated 2x2.
    A = sp.csc_matrix(np.array([[1.], [2.], [3.], [4.]], dtype=np.float32))
    up = upsample_footprints(A, (2, 2), (4, 4), order=0)
    assert up.shape == (16, 1)
    expected = np.array([[1, 1, 2, 2], [1, 1, 2, 2],
                         [3, 3, 4, 4], [3, 3, 4, 4]], dtype=np.float32)
    np.testing.assert_array_equal(np.asarray(up.todense()).reshape(4, 4), expected)


def test_upsample_footprints_bilinear_keeps_support_quadrant():
    # A blob in the top-left of a 4x4 footprint stays top-left after bilinear
    # upsample to 8x8.
    img = np.zeros((4, 4), dtype=np.float32)
    img[0:2, 0:2] = 1.0
    A = sp.csc_matrix(img.reshape(-1, 1))
    up = upsample_footprints(A, (4, 4), (8, 8), order=1)
    big = np.asarray(up.todense()).reshape(8, 8)
    assert big.shape == (8, 8)
    cy, cx = _centroid(big)
    assert cy < 4 and cx < 4    # centre of mass in the top-left quadrant


def test_upsample_traces_linear_ramp_and_endpoints():
    C = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)
    out = upsample_traces(C, 5)
    assert out.shape == (1, 5)
    np.testing.assert_allclose(out[0], [0.0, 0.5, 1.0, 1.5, 2.0], rtol=0, atol=1e-6)
    # Endpoints preserved for any target length.
    out2 = upsample_traces(C, 7)
    assert out2[0, 0] == 0.0 and out2[0, -1] == 2.0


def test_upsample_traces_identity_when_equal_length():
    C = np.random.default_rng(0).standard_normal((3, 10)).astype(np.float32)
    np.testing.assert_array_equal(upsample_traces(C, 10), C)


def test_upsample_to_native_shapes_and_non_destructive(synth_small):
    movie = synth_small["movie"]
    H, W = synth_small["dims"]
    T = movie.shape[0]
    params = CNMFeParams(sigma=3.0, min_corr=0.5, min_pnr=3.0,
                         n_iter_main=1, n_iter_temporal=1, n_jobs=1)
    model = CNMFe(params).fit_extract(movie, evaluate=True)
    K = model.A.shape[1]
    assert K >= 1

    nat_dims, nat_T = (2 * H, 2 * W), 2 * T
    up = model.upsample_to_native(orig_dims=nat_dims, orig_T=nat_T,
                                  ssub=2, tsub=2)

    # Upsampled shapes.
    assert up.A.shape == (nat_dims[0] * nat_dims[1], K)
    assert up.C.shape == (K, nat_T)
    assert up.YrA.shape == (K, nat_T)
    if up.C_raw is not None:
        assert up.C_raw.shape == (K, nat_T)
    assert up.dims == nat_dims
    # S stays at the downsampled rate; background/shifts dropped.
    assert up.S.shape == model.S.shape
    assert up.W is None and up.b0 is None and up.shifts is None
    # Per-component metadata carried over.
    if model.accepted_mask is not None:
        np.testing.assert_array_equal(up.accepted_mask, model.accepted_mask)

    # Non-destructive: original model untouched.
    assert model.A.shape == (H * W, K)
    assert model.C.shape == (K, T)


def test_upsample_to_native_from_ds_meta(tmp_path):
    meta = {"orig_dims": [40, 40], "orig_T": 60, "ssub": 2, "tsub": 2}
    (tmp_path / "ds_meta.json").write_text(json.dumps(meta))
    movie = np.random.default_rng(1).standard_normal((30, 20, 20)).astype(np.float32)
    params = CNMFeParams(sigma=3.0, min_corr=0.5, min_pnr=3.0,
                         n_iter_main=1, n_iter_temporal=1, n_jobs=1)
    model = CNMFe(params).fit_extract(movie, evaluate=False)
    if model.A.shape[1] == 0:
        return  # no components on pure noise; nothing to upsample
    up = model.upsample_to_native(ds_meta=tmp_path / "ds_meta.json")
    assert up.dims == (40, 40)
    assert up.C.shape[1] == 60


def test_upsample_roundtrip_recovers_centroids(synth, tmp_path):
    """Downsample -> fit -> upsample back to native; recovered footprint
    centroids land near the native ground-truth centers."""
    ssub, tsub = 2, 2
    movie = synth["movie"]
    H, W = synth["dims"]
    T = movie.shape[0]
    save_zarr(movie, tmp_path / "src.zarr")
    downsample_movie(tmp_path / "src.zarr", tmp_path / "ds.zarr",
                     ssub=ssub, tsub=tsub, verbose=False)

    params = CNMFeParams(sigma=3.0, min_corr=0.5, min_pnr=3.0,
                         n_iter_main=1, n_iter_temporal=1,
                         n_jobs=1).downscaled(ssub, tsub)
    model = CNMFe(params).fit(open_zarr(tmp_path / "ds.zarr"),
                              do_motion_correction=False)
    up = model.upsample_to_native(orig_dims=(H, W), orig_T=T, ssub=ssub, tsub=tsub)
    assert up.A.shape[0] == H * W
    assert up.C.shape[1] == T

    # Each native ground-truth centre should have an upsampled footprint
    # centroid within a few px.
    A_up = np.asarray(up.A.todense())
    centers = synth["centers"]                       # (K_true, 2) (row, col)
    up_centroids = [_centroid(A_up[:, k].reshape(H, W)) for k in range(up.A.shape[1])]
    hits = 0
    for cy, cx in centers:
        d = min(np.hypot(c[0] - cy, c[1] - cx) for c in up_centroids)
        if d <= 4.0:
            hits += 1
    assert hits >= (len(centers) + 1) // 2, (
        f"only {hits}/{len(centers)} native centers matched an upsampled footprint"
    )
