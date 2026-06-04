"""Tests for the per-pixel temporal detrend preprocessor.

The detrend removes slow per-pixel baseline drift (LED warm-up, photobleach,
step jumps) before extraction sees the movie. These tests check three
canonical scenarios:

- A flat baseline (no drift) should leave the residual ≈ -baseline.
- A linear baseline drift should be flattened to ~0.
- A step jump should be removed in interior frames after the window slides
  past the discontinuity.

We also confirm the rolling lower-percentile is robust to sparse spikes:
a clean baseline + sparse calcium-like transients should keep the transients
roughly intact while the baseline drift is removed.
"""

import numpy as np

from minicnmfe.detrend import detrend_movie
from minicnmfe.io import open_zarr, save_zarr


def _make_zarr(tmp_path, movie):
    src = tmp_path / "src.zarr"
    save_zarr(movie.astype(np.float32), src)
    return src


def test_flat_baseline_subtracts_to_constant(tmp_path):
    # Constant baseline -> rolling 10th-percentile == baseline, output ≈ 0.
    T, H, W = 200, 8, 8
    movie = np.full((T, H, W), 42.0, dtype=np.float32)
    src = _make_zarr(tmp_path, movie)

    out = detrend_movie(
        src, tmp_path / "out.zarr",
        window_s=2.0, percentile=10.0, frame_rate_hz=20.0,
        batch_t=64, verbose=False,
    )
    arr = np.asarray(out)
    assert arr.shape == movie.shape
    # Constant signal -> rolling percentile == 42 everywhere -> output is 0.
    assert np.allclose(arr, 0.0, atol=1e-5)


def test_linear_drift_is_flattened(tmp_path):
    # Per-pixel linear ramp; after detrend the interior of T should be ~flat.
    T, H, W = 300, 4, 4
    t = np.arange(T, dtype=np.float32)
    drift = (t * 0.05)[:, None, None]                         # 0..14.95
    movie = (50.0 + drift) * np.ones((T, H, W), dtype=np.float32)
    src = _make_zarr(tmp_path, movie)

    out = detrend_movie(
        src, tmp_path / "out.zarr",
        window_s=1.0, percentile=10.0, frame_rate_hz=30.0,  # 30-frame window
        batch_t=64, verbose=False,
    )
    arr = np.asarray(out)
    # Skip an edge margin equal to the window; the rolling filter near the
    # start sees only the rising-edge half-window so the baseline lags slightly.
    margin = 60
    interior = arr[margin:-margin]
    assert interior.std() < 1.0       # >>1 before detrend (drift spans ~15)
    assert abs(interior.mean()) < 1.0


def test_step_jump_is_removed(tmp_path):
    # Per-pixel step at t=120 from 30.0 to 50.0. After detrend, frames far
    # enough from the step (more than half a window away on each side)
    # should be near zero.
    T, H, W = 400, 4, 4
    movie = np.empty((T, H, W), dtype=np.float32)
    movie[:120] = 30.0
    movie[120:] = 50.0
    src = _make_zarr(tmp_path, movie)

    out = detrend_movie(
        src, tmp_path / "out.zarr",
        window_s=1.0, percentile=10.0, frame_rate_hz=40.0,  # 40-frame window
        batch_t=64, verbose=False,
    )
    arr = np.asarray(out)
    # Sufficiently far from the step on each side.
    assert abs(arr[:80].mean()) < 0.5
    assert abs(arr[200:].mean()) < 0.5
    # Total residual energy should be concentrated near the step.
    pre  = np.abs(arr[:80]).sum()
    post = np.abs(arr[200:]).sum()
    near = np.abs(arr[80:200]).sum()
    assert near > 10 * max(pre, post)


def test_sparse_spikes_preserved_on_top_of_drift(tmp_path):
    # Slow drift + a few impulsive "spikes" added on top of one pixel.
    # The lower-percentile baseline should ignore the spikes, so they
    # survive the subtraction at roughly full amplitude.
    T, H, W = 400, 4, 4
    rng = np.random.default_rng(7)
    t = np.arange(T, dtype=np.float32)
    drift = (t * 0.02)[:, None, None]
    movie = (40.0 + drift + 0.1 * rng.standard_normal((T, H, W))).astype(np.float32)
    spike_t = [80, 200, 320]
    for tt in spike_t:
        movie[tt, 2, 2] += 20.0

    src = _make_zarr(tmp_path, movie)
    out = detrend_movie(
        src, tmp_path / "out.zarr",
        window_s=2.0, percentile=10.0, frame_rate_hz=30.0,  # 60-frame window
        batch_t=128, verbose=False,
    )
    arr = np.asarray(out)
    # The spike pixel should still show large positive values at the spike
    # frames; non-spike samples should sit near zero.
    spike_vals = arr[spike_t, 2, 2]
    assert (spike_vals > 15.0).all()
    bg_pixel = arr[:, 0, 0]
    assert abs(bg_pixel.mean()) < 1.0
    assert bg_pixel.std() < 1.0


def test_skip_if_exists_returns_existing(tmp_path):
    movie = np.zeros((30, 4, 4), dtype=np.float32)
    src = _make_zarr(tmp_path, movie)
    dest = tmp_path / "out.zarr"
    detrend_movie(src, dest, window_s=0.3, frame_rate_hz=10.0,
                  batch_t=16, verbose=False)
    # Write a marker into the existing output; skip_if_exists should NOT
    # overwrite it.
    import zarr as _zarr
    z = _zarr.open_array(str(dest), mode="r+")
    z[0, 0, 0] = 999.0
    again = detrend_movie(src, dest, window_s=0.3, frame_rate_hz=10.0,
                          batch_t=16, verbose=False, skip_if_exists=True)
    assert float(np.asarray(again[0, 0, 0])) == 999.0


def test_anchor_baseline_matches_dense_for_slow_drift(tmp_path):
    # The anchor-and-interpolate baseline (default sparse anchors) should
    # agree with the dense per-frame baseline (anchor_stride=1) to within a
    # small tolerance on non-spike samples: the lower-percentile baseline is
    # slow by design, so sparse anchors + linear interpolation reproduce it.
    rng = np.random.default_rng(11)
    T, H, W = 600, 6, 6
    drift = (0.02 * np.arange(T, dtype=np.float32))[:, None, None]
    movie = (10.0 + drift + 0.1 * rng.standard_normal((T, H, W))).astype(np.float32)
    # Sparse impulsive spikes (lower percentile ignores them, so the two
    # baselines should agree on these frames too).
    for tt in (120, 300, 480):
        movie[tt, 3, 3] += 25.0

    src = _make_zarr(tmp_path, movie)

    dense = detrend_movie(
        src, tmp_path / "dense.zarr",
        window_s=2.0, percentile=10.0, frame_rate_hz=30.0,  # window=60 frames
        batch_t=128, anchor_stride=1, verbose=False,
    )
    sparse = detrend_movie(
        src, tmp_path / "sparse.zarr",
        window_s=2.0, percentile=10.0, frame_rate_hz=30.0,
        batch_t=128, verbose=False,  # default anchor_stride = window // 10 = 6
    )
    a = np.asarray(dense)
    b = np.asarray(sparse)
    # Tight mean error: the smoothing introduced by linear interpolation
    # between anchors is small relative to the noise floor (~0.1).
    assert np.abs(a - b).mean() < 0.05
    # Spike values must be preserved by both — drift removal doesn't touch
    # the impulsive part.
    for tt in (120, 300, 480):
        assert a[tt, 3, 3] > 15.0
        assert b[tt, 3, 3] > 15.0


def test_parallel_matches_serial(tmp_path):
    # Splitting the spatial frame into slabs must not change the result.
    rng = np.random.default_rng(3)
    T, H, W = 120, 8, 9
    movie = rng.standard_normal((T, H, W)).astype(np.float32) + 10.0
    src = _make_zarr(tmp_path, movie)

    a = detrend_movie(src, tmp_path / "serial.zarr",
                      window_s=0.5, frame_rate_hz=20.0,
                      batch_t=32, n_jobs=1, verbose=False)
    b = detrend_movie(src, tmp_path / "parallel.zarr",
                      window_s=0.5, frame_rate_hz=20.0,
                      batch_t=32, n_jobs=4, verbose=False)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-5)
