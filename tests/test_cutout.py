"""Tests for the spatial/temporal cutout (crop) feature.

Covers the resolution/apply helpers (``minicnmfe/cutout.py``), the
``CNMFeParams`` fields + serialisation, the ``fit`` wiring + ``self.cutout``
recording, the ``place_in_full_fov`` map-back, and the fused AVI->MC crop.
"""

import json

import numpy as np
import pytest
import scipy.sparse as sp

from minicnmfe.cutout import (
    apply_cutout,
    place_footprints_in_fov,
    place_traces_in_timeline,
    resolve_cutout,
)
from minicnmfe.io import open_zarr, save_zarr
from minicnmfe.pipeline import CNMFe, CNMFeParams


# --------------------------------------------------------------------------
# resolve_cutout
# --------------------------------------------------------------------------

def test_resolve_none_when_unset():
    assert resolve_cutout(CNMFeParams(), (64, 64), 100) is None


def test_resolve_rect_and_temporal():
    p = CNMFeParams(spatial_crop=(8, 40, 4, 36), temporal_crop=(10, 80))
    spec = resolve_cutout(p, (64, 64), 100)
    assert spec["bbox"] == [8, 40, 4, 36]
    assert spec["t_range"] == [10, 80]
    assert spec["orig_dims"] == [64, 64] and spec["orig_T"] == 100
    assert spec["masked"] is False and spec["mask_local"] is None


def test_resolve_clamps_to_bounds():
    p = CNMFeParams(spatial_crop=(-5, 200, 0, 50), temporal_crop=(-3, 999))
    spec = resolve_cutout(p, (64, 64), 100)
    assert spec["bbox"] == [0, 64, 0, 50]
    assert spec["t_range"] == [0, 100]


def test_resolve_mask_bbox_and_intersection(tmp_path):
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:30, 15:25] = True                 # bbox (20,30,15,25)
    np.save(tmp_path / "m.npy", mask)
    p = CNMFeParams(spatial_mask_path=str(tmp_path / "m.npy"))
    spec = resolve_cutout(p, (64, 64), 50)
    assert spec["bbox"] == [20, 30, 15, 25]
    assert spec["masked"] and spec["mask_local"].shape == (10, 10)
    assert spec["mask_local"].all()           # the bbox is exactly the mask here

    # Rect ∩ mask-bbox.
    p2 = CNMFeParams(spatial_mask_path=str(tmp_path / "m.npy"),
                     spatial_crop=(22, 64, 0, 64))
    spec2 = resolve_cutout(p2, (64, 64), 50)
    assert spec2["bbox"] == [22, 30, 15, 25]


def test_resolve_errors(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        resolve_cutout(CNMFeParams(temporal_crop=(50, 50)), (64, 64), 100)
    with pytest.raises(ValueError, match="empty"):
        resolve_cutout(CNMFeParams(spatial_crop=(10, 10, 0, 5)), (64, 64), 100)
    np.save(tmp_path / "empty.npy", np.zeros((64, 64), dtype=bool))
    with pytest.raises(ValueError, match="all-False"):
        resolve_cutout(CNMFeParams(spatial_mask_path=str(tmp_path / "empty.npy")),
                       (64, 64), 100)
    np.save(tmp_path / "bad.npy", np.ones((10, 10), dtype=bool))
    with pytest.raises(ValueError, match="match native dims"):
        resolve_cutout(CNMFeParams(spatial_mask_path=str(tmp_path / "bad.npy")),
                       (64, 64), 100)


# --------------------------------------------------------------------------
# apply_cutout
# --------------------------------------------------------------------------

def test_apply_cutout_numpy_and_zarr(tmp_path):
    rng = np.random.default_rng(0)
    movie = rng.standard_normal((20, 16, 16)).astype(np.float32)
    p = CNMFeParams(spatial_crop=(2, 10, 4, 12), temporal_crop=(5, 15))
    spec = resolve_cutout(p, (16, 16), 20)

    out = apply_cutout(movie, spec)
    assert out.shape == (10, 8, 8)
    np.testing.assert_array_equal(out, movie[5:15, 2:10, 4:12])

    save_zarr(movie, tmp_path / "m.zarr")
    out_z = apply_cutout(open_zarr(tmp_path / "m.zarr"), spec)
    np.testing.assert_allclose(out_z, out, rtol=0, atol=1e-5)


def test_apply_cutout_zeros_outside_mask(tmp_path):
    movie = np.ones((4, 8, 8), dtype=np.float32)
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    np.save(tmp_path / "m.npy", mask)
    spec = resolve_cutout(CNMFeParams(spatial_mask_path=str(tmp_path / "m.npy")),
                          (8, 8), 4)
    out = apply_cutout(movie, spec)            # bbox is (2,6,2,6) -> 4x4, all inside mask
    assert out.shape == (4, 4, 4)
    assert out.min() == 1.0                    # mask fully covers the bbox here


# --------------------------------------------------------------------------
# place-back helpers
# --------------------------------------------------------------------------

def test_place_footprints_in_fov():
    sub = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    A_crop = sp.csc_matrix(sub.reshape(-1, 1))      # 2x2 footprint
    full = place_footprints_in_fov(A_crop, [1, 3, 2, 4], (5, 6))
    img = np.asarray(full.todense()).reshape(5, 6)
    np.testing.assert_array_equal(img[1:3, 2:4], sub)
    img[1:3, 2:4] = 0
    assert img.sum() == 0                            # everything else zero


def test_place_traces_in_timeline():
    C = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = place_traces_in_timeline(C, [4, 7], 10)
    assert out.shape == (2, 10)
    np.testing.assert_array_equal(out[:, 4:7], C)
    assert out[:, :4].sum() == 0 and out[:, 7:].sum() == 0


# --------------------------------------------------------------------------
# CNMFeParams serialisation / downscaled
# --------------------------------------------------------------------------

def test_params_json_roundtrips_crop(tmp_path):
    p = CNMFeParams(spatial_crop=(1, 2, 3, 4), temporal_crop=(5, 6),
                    spatial_mask_path="m.npy")
    p.to_json(tmp_path / "p.json")
    q = CNMFeParams.from_json(tmp_path / "p.json")
    assert q.spatial_crop == (1, 2, 3, 4)        # restored to tuple
    assert q.temporal_crop == (5, 6)
    assert q.spatial_mask_path == "m.npy"


def test_downscaled_clears_crop():
    p = CNMFeParams(spatial_crop=(1, 2, 3, 4), temporal_crop=(5, 6),
                    spatial_mask_path="m.npy")
    d = p.downscaled(2, 2)
    assert d.spatial_crop is None
    assert d.temporal_crop is None
    assert d.spatial_mask_path is None


# --------------------------------------------------------------------------
# End-to-end fit on a cutout + place_in_full_fov
# --------------------------------------------------------------------------

def test_fit_on_cutout_and_place_back(synth):
    movie = synth["movie"]
    H, W = synth["dims"]
    T = movie.shape[0]
    y0, y1, x0, x1 = 8, 56, 8, 56
    t0, t1 = 20, T - 20
    p = CNMFeParams(sigma=3.0, min_corr=0.5, min_pnr=3.0,
                    n_iter_main=1, n_iter_temporal=1, n_jobs=1,
                    spatial_crop=(y0, y1, x0, x1), temporal_crop=(t0, t1))
    model = CNMFe(p).fit(movie, do_motion_correction=False)

    # Extraction ran on the cutout.
    assert model.dims == (y1 - y0, x1 - x0)
    assert model.C.shape[1] == t1 - t0
    assert model.A.shape[0] == (y1 - y0) * (x1 - x0)
    assert model.cutout["bbox"] == [y0, y1, x0, x1]
    assert model.cutout["t_range"] == [t0, t1]
    assert model.A.shape[1] >= 1

    # Map back to the full FOV / timeline.
    full = model.place_in_full_fov()
    assert full.dims == (H, W)
    assert full.A.shape[0] == H * W
    assert full.C.shape[1] == T
    # Traces are zero outside the window.
    assert full.C[:, :t0].sum() == 0 and full.C[:, t1:].sum() == 0
    # Every footprint lives strictly inside the spatial crop.
    A_full = np.asarray(full.A.todense())
    for k in range(full.A.shape[1]):
        ys, xs = np.nonzero(A_full[:, k].reshape(H, W))
        assert ys.min() >= y0 and ys.max() < y1
        assert xs.min() >= x0 and xs.max() < x1
    # Non-destructive.
    assert model.dims == (y1 - y0, x1 - x0)


def test_cutout_incompatible_with_y_flat_zarr(synth_small, tmp_path):
    movie = synth_small["movie"]
    save_zarr(movie, tmp_path / "m.zarr")
    from minicnmfe.io import transpose_zarr_to_pixel_major
    yf = transpose_zarr_to_pixel_major(tmp_path / "m.zarr", tmp_path / "yf.zarr",
                                       verbose=False)
    p = CNMFeParams(spatial_crop=(2, 20, 2, 20), n_jobs=1)
    with pytest.raises(ValueError, match="cannot.*be combined"):
        CNMFe(p).fit(open_zarr(tmp_path / "m.zarr"),
                     do_motion_correction=False, Y_flat_zarr=yf)


# --------------------------------------------------------------------------
# Fused AVI->MC with a cutout
# --------------------------------------------------------------------------

def test_fused_avi_mc_cutout(tmp_path):
    cv2 = pytest.importorskip("cv2")
    src = tmp_path / "session"
    src.mkdir()
    for i in range(3):
        w = cv2.VideoWriter(str(src / f"{i}.avi"),
                            cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (32, 32),
                            isColor=True)
        rng = np.random.default_rng(i)
        for _ in range(12):
            g = rng.integers(0, 256, (32, 32), np.uint8)
            w.write(np.stack([g, g, g], -1))
        w.release()

    # Temporal [10:30) spans files: 0->2, 1->12, 2->6 = 20; spatial 20x20.
    p = CNMFeParams(max_shift=(3, 3), mc_gSig_filt=2, mc_n_iter=1,
                    mc_batch_size=8, mc_template_max_frames=20, n_jobs=1,
                    spatial_crop=(4, 24, 8, 28), temporal_crop=(10, 30))
    model = CNMFe(p)
    mc = model.fit_mc_from_avis(src, tmp_path / "out", skip_if_exists=False)
    assert mc.shape == (20, 20, 20)
    assert model.shifts.shape == (20, 2)
    assert (tmp_path / "out" / "cutout.json").exists()
    meta = json.loads((tmp_path / "out" / "cutout.json").read_text())
    assert meta["bbox"] == [4, 24, 8, 28] and meta["t_range"] == [10, 30]
    assert model.cutout["orig_dims"] == [32, 32]
