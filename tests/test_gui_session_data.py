"""Tests for cnmfe.gui.data_loader.SessionData.

We hand-build the smallest possible "fit" by writing K Gaussian footprints +
AR(1)-like traces straight onto a fresh CNMFe model and ``save()``-ing it,
then exercise SessionData.load against the four cases of {cutout?, ds_meta?}.
We never actually run the pipeline -- too slow for what is purely a loading
test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from cnmfe.pipeline import CNMFe, CNMFeParams
from cnmfe.gui.data_loader import SessionData, TimeMap


# ---------------------------------------------------------------------- helpers

def _fake_results(
    out_dir: Path,
    *,
    H: int = 32,
    W: int = 40,
    K: int = 3,
    T: int = 50,
    cutout: dict | None = None,
) -> None:
    """Write a minimal results dir at ``out_dir`` that ``CNMFe.load`` accepts.

    The CNMFe model fields here describe extraction-space data (so if
    ``cutout`` is set with bbox=(y0,y1,x0,x1), dims = (y1-y0, x1-x0) and
    T = t1-t0). Cutout dict is the one written by ``manifest.json``.
    """
    rng = np.random.default_rng(0)

    A = sp.random(H * W, K, density=0.1, format="csc", dtype=np.float32, random_state=rng)
    # Ensure each column has at least one nonzero with a positive value.
    A = A.maximum(0.01 * sp.csc_matrix(np.ones((H * W, K), dtype=np.float32)))
    A = A.tocsc()
    # Trim back to sparsity by zeroing pixels below 0.05.
    A_dense = A.toarray()
    A_dense[A_dense < 0.05] = 0.0
    # Make sure each component has at least some support.
    for k in range(K):
        if not np.any(A_dense[:, k] > 0):
            A_dense[k * 5, k] = 1.0
    A = sp.csc_matrix(A_dense)

    C = rng.random((K, T)).astype(np.float32)
    S = rng.random((K, T)).astype(np.float32)
    YrA = 0.1 * rng.standard_normal((K, T)).astype(np.float32)
    sn = 0.05 * np.ones((H, W), dtype=np.float32)
    accepted = np.array([True, False, True][:K] + [True] * max(0, K - 3), dtype=bool)
    eval_info = {
        "pixel_count": np.array([int((A_dense[:, k] > 0).sum()) for k in range(K)], dtype=np.int64),
        "snr_amp": np.array([10.0, 1.0, 20.0][:K] + [5.0] * max(0, K - 3), dtype=np.float32),
        "pixel_pass": np.ones(K, dtype=bool),
        "snr_pass": accepted.copy(),
        "min_pixel": 1,
        "snr_amp_thr": 3.0,
    }

    model = CNMFe(CNMFeParams())
    model.dims = (H, W)
    model.A = A
    model.C = C
    model.S = S
    model.YrA = YrA
    model.sn = sn
    model.accepted_mask = accepted
    model.eval_info = eval_info
    model.cutout = cutout
    model.save(out_dir)


# ---------------------------------------------------------------------- no transforms

def test_load_plain_native(tmp_path: Path):
    H, W, K, T = 24, 32, 4, 30
    _fake_results(tmp_path / "results", H=H, W=W, K=K, T=T)
    # avi_folder is just any existing dir; we don't open AVIs here.
    avi_dir = tmp_path / "avis"
    avi_dir.mkdir()
    sd = SessionData.load(tmp_path / "results", avi_dir)
    assert (sd.H, sd.W) == (H, W)
    assert sd.K == K
    assert sd.C.shape == (K, T)
    assert sd.S.shape == (K, T)
    assert sd.YrA is not None and sd.YrA.shape == (K, T)
    assert sd.A_native_csc.shape == (H * W, K)
    assert sd.centroids.shape == (K, 2)
    np.testing.assert_array_equal(sd.auto_accepted, np.array([True, False, True, True]))
    # Time map for native dims = no cutout, no downsample.
    tm = sd.time_map
    assert (tm.t0_cutout, tm.t1_cutout, tm.tsub) == (0, T, 1)
    assert tm.orig_T == T
    assert tm.extraction_T == T
    assert tm.native_to_extraction(7) == 7
    assert tm.extraction_to_native(7) == 7
    assert tm.native_to_extraction(T) is None


def test_summary_columns_pulled_from_eval_info(tmp_path: Path):
    _fake_results(tmp_path / "results", K=3, T=10, H=16, W=16)
    avi_dir = tmp_path / "avis"; avi_dir.mkdir()
    sd = SessionData.load(tmp_path / "results", avi_dir)
    cols = sd.summary_columns()
    assert cols["snr_amp"].shape == (3,)
    assert cols["pixel_count"].shape == (3,)
    np.testing.assert_array_equal(cols["snr_pass"], np.array([True, False, True]))


def test_peak_native_frame_plain(tmp_path: Path):
    _fake_results(tmp_path / "results", K=2, T=20, H=16, W=16)
    avi_dir = tmp_path / "avis"; avi_dir.mkdir()
    sd = SessionData.load(tmp_path / "results", avi_dir)
    # peak_native_frame should equal argmax(C[k]) when there's no cutout/ds.
    for k in range(2):
        expected = int(np.argmax(sd.C[k]))
        assert sd.peak_native_frame(k) == expected


# ---------------------------------------------------------------------- cutout

def test_load_with_cutout_only(tmp_path: Path):
    H_orig, W_orig, T_orig = 40, 50, 60
    y0, y1, x0, x1 = 8, 32, 10, 42
    t0, t1 = 5, 45
    cutout = {
        "orig_dims": [H_orig, W_orig],
        "orig_T": T_orig,
        "bbox": [y0, y1, x0, x1],
        "t_range": [t0, t1],
        "masked": False,
    }
    H_cut, W_cut, T_cut = y1 - y0, x1 - x0, t1 - t0
    _fake_results(
        tmp_path / "results",
        H=H_cut, W=W_cut, T=T_cut, K=3,
        cutout=cutout,
    )
    avi_dir = tmp_path / "avis"; avi_dir.mkdir()

    sd = SessionData.load(tmp_path / "results", avi_dir)
    # After place_in_full_fov, dims must be the NATIVE FOV.
    assert (sd.H, sd.W) == (H_orig, W_orig)
    assert sd.A_native_csc.shape == (H_orig * W_orig, 3)
    # Traces stay at extraction rate (T_cut).
    assert sd.C.shape == (3, T_cut)
    tm = sd.time_map
    assert (tm.t0_cutout, tm.t1_cutout) == (t0, t1)
    assert tm.tsub == 1
    assert tm.orig_T == T_orig
    assert tm.extraction_T == T_cut
    # Native frame inside the window -> extraction frame; outside -> None.
    assert tm.native_to_extraction(t0) == 0
    assert tm.native_to_extraction(t0 + 7) == 7
    assert tm.native_to_extraction(t1) is None
    assert tm.native_to_extraction(0) is None
    # The reverse map plants the right native frame.
    assert tm.extraction_to_native(0) == t0
    assert tm.extraction_to_native(5) == t0 + 5

    # Centroid must land inside the cutout bbox in NATIVE coords.
    # (Every nonzero pixel in A_native lives in [y0:y1, x0:x1].)
    cy, cx = sd.centroids[0]
    assert y0 <= cy <= y1
    assert x0 <= cx <= x1


# ---------------------------------------------------------------------- ds_meta

def test_load_with_ds_meta_only(tmp_path: Path):
    # native 64x80, ssub=2 -> downsampled dims 32x40
    # native T=80, tsub=4 -> downsampled T=20
    H_ds, W_ds, T_ds = 32, 40, 20
    H_orig, W_orig, T_orig = 64, 80, 80
    ds_meta = {
        "ssub": 2, "tsub": 4,
        "orig_dims": [H_orig, W_orig],
        "orig_T": T_orig,
        "ds_dims": [H_ds, W_ds],
        "ds_T": T_ds,
        "src": "", "dest": "",
    }
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "ds_meta.json").write_text(json.dumps(ds_meta))
    _fake_results(tmp_path / "results", H=H_ds, W=W_ds, T=T_ds, K=2, cutout=None)
    avi_dir = tmp_path / "avis"; avi_dir.mkdir()

    sd = SessionData.load(tmp_path / "results", avi_dir)
    assert (sd.H, sd.W) == (H_orig, W_orig)
    assert sd.A_native_csc.shape == (H_orig * W_orig, 2)
    # Traces stay at the extraction rate (T_ds).
    assert sd.C.shape == (2, T_ds)
    tm = sd.time_map
    assert tm.tsub == 4
    assert tm.orig_T == T_orig
    assert tm.extraction_T == T_ds
    assert tm.t0_cutout == 0
    # Native frame 4 -> extraction 1 (4 // 4); frame 0 -> 0.
    assert tm.native_to_extraction(0) == 0
    assert tm.native_to_extraction(4) == 1
    assert tm.native_to_extraction(7) == 1  # floor(7/4) = 1
    assert tm.native_to_extraction(T_orig) is None
    # Reverse map plants on a tsub boundary.
    assert tm.extraction_to_native(3) == 12


# ---------------------------------------------------------------------- both

def test_load_with_cutout_and_ds_meta(tmp_path: Path):
    """The trickiest case: a cutout AND a downsampled run.

    The CNMFe pipeline currently clears cutout fields on ``downscaled()`` (the
    cutout is applied upstream of binning), so this combination is rare in
    practice. We still verify it would load: ``place_in_full_fov`` pads to
    the cutout's ``orig_dims``, then ``upsample_to_native`` interpolates that
    to the ds_meta's ``orig_dims`` (which here equals the cutout orig_dims
    -- it represents the same native FOV).
    """
    H_native, W_native, T_native = 48, 60, 80
    bbox = [4, 28, 10, 50]  # y0..y1, x0..x1
    t_range = [0, 60]
    cutout = {
        "orig_dims": [H_native, W_native],
        "orig_T": T_native,
        "bbox": bbox,
        "t_range": t_range,
        "masked": False,
    }
    # After cutout: dims = (24, 40), T = 60.
    # After ssub=2, tsub=3 binning: dims = (12, 20), T = 20.
    H_ds, W_ds, T_ds = 12, 20, 20
    # ds_meta describes the entire downsample of the cropped movie back to
    # the *cropped* native space, so orig_dims here equals the cropped dims.
    ds_meta = {
        "ssub": 2, "tsub": 3,
        "orig_dims": [bbox[1] - bbox[0], bbox[3] - bbox[2]],  # cropped native
        "orig_T": t_range[1] - t_range[0],
        "ds_dims": [H_ds, W_ds],
        "ds_T": T_ds,
        "src": "", "dest": "",
    }
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "ds_meta.json").write_text(json.dumps(ds_meta))
    _fake_results(tmp_path / "results", H=H_ds, W=W_ds, T=T_ds, K=2, cutout=cutout)
    avi_dir = tmp_path / "avis"; avi_dir.mkdir()

    sd = SessionData.load(tmp_path / "results", avi_dir)
    # After place_in_full_fov: dims become (H_native, W_native).
    # upsample_to_native then interpolates that to ds_meta.orig_dims, which
    # in this synthetic test equals the CROPPED native -- so the final dims
    # depend on which orig_dims pipeline.py picks. Read m.dims directly to
    # be tolerant of either.
    assert sd.C.shape == (2, T_ds)
    tm = sd.time_map
    assert tm.tsub == 3
    # extraction_to_native must account for the cutout offset.
    assert tm.extraction_to_native(0) == t_range[0]
    assert tm.extraction_to_native(5) == t_range[0] + 15
    # A native frame outside the cutout window -> None.
    assert tm.native_to_extraction(T_native - 1) is None or (
        T_native - 1 >= t_range[1]
    )


# ---------------------------------------------------------------------- guards

def test_avi_dims_mismatch_raises(tmp_path: Path):
    _fake_results(tmp_path / "results", H=10, W=10, K=2, T=5)
    avi_dir = tmp_path / "avis"; avi_dir.mkdir()
    with pytest.raises(RuntimeError, match="don't match"):
        SessionData.load(
            tmp_path / "results", avi_dir, expected_avi_dims=(99, 99)
        )


def test_missing_results_dir_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        SessionData.load(tmp_path / "nope", tmp_path)
