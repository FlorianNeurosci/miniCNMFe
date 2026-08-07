"""Tests for the outlier-frame rejection preprocessor."""

import numpy as np

from minicnmfe.io import save_zarr
from minicnmfe.reject_frames import reject_outlier_frames


def _make_zarr(tmp_path, movie):
    src = tmp_path / "src.zarr"
    save_zarr(movie.astype(np.float32), src)
    return src


def test_passthrough_when_no_outliers(tmp_path):
    # A clean recording: per-frame mean is Gaussian noise around a fixed
    # level. No frame should be flagged at k_mad=5.
    rng = np.random.default_rng(0)
    T, H, W = 200, 6, 6
    movie = (rng.standard_normal((T, H, W)) + 10.0).astype(np.float32)
    src = _make_zarr(tmp_path, movie)

    out, outliers = reject_outlier_frames(
        src, tmp_path / "out.zarr", k_mad=5.0, batch_t=64, verbose=False,
    )
    assert len(outliers) == 0
    arr = np.asarray(out)
    # Direct-copy path: dest matches src exactly.
    np.testing.assert_allclose(arr, movie, atol=1e-6)


def test_isolated_flash_is_interpolated(tmp_path):
    # Three flash frames at t=50, 100, 150 with mean ≈ 50 (vs baseline 10).
    rng = np.random.default_rng(0)
    T, H, W = 200, 6, 6
    movie = (rng.standard_normal((T, H, W)) + 10.0).astype(np.float32)
    flash_t = [50, 100, 150]
    for t in flash_t:
        movie[t] += 40.0  # uniform flash; mean spikes
    src = _make_zarr(tmp_path, movie)

    out, outliers = reject_outlier_frames(
        src, tmp_path / "out.zarr", k_mad=5.0, batch_t=64, verbose=False,
    )
    assert set(outliers.tolist()) == set(flash_t)
    arr = np.asarray(out)
    # Non-flash frames untouched.
    keep = np.setdiff1d(np.arange(T), flash_t)
    np.testing.assert_allclose(arr[keep], movie[keep], atol=1e-6)
    # Flash frames replaced — mean should be back near baseline, not ~50.
    for t in flash_t:
        assert abs(arr[t].mean() - 10.0) < 2.0


def test_consecutive_outlier_run(tmp_path):
    # A run of three consecutive outlier frames is fully interpolated
    # between the same two boundary frames.
    rng = np.random.default_rng(1)
    T, H, W = 100, 4, 4
    movie = (rng.standard_normal((T, H, W)) + 5.0).astype(np.float32)
    movie[40:43] += 50.0  # 3-frame flash run
    src = _make_zarr(tmp_path, movie)

    out, outliers = reject_outlier_frames(
        src, tmp_path / "out.zarr", k_mad=5.0, batch_t=32, verbose=False,
    )
    assert set(outliers.tolist()) == {40, 41, 42}
    arr = np.asarray(out)
    # Replacement values lie between the left/right boundary frames.
    left, right = movie[39], movie[43]
    np.testing.assert_allclose(arr[40], (3 / 4) * left + (1 / 4) * right, atol=1e-5)
    np.testing.assert_allclose(arr[41], (2 / 4) * left + (2 / 4) * right, atol=1e-5)
    np.testing.assert_allclose(arr[42], (1 / 4) * left + (3 / 4) * right, atol=1e-5)


def test_edge_outlier_falls_back_to_nearest(tmp_path):
    # If the outlier is at t=0 there is no "left" neighbour — replacement
    # uses the nearest non-outlier on the right.
    rng = np.random.default_rng(2)
    T, H, W = 60, 4, 4
    movie = (rng.standard_normal((T, H, W)) + 8.0).astype(np.float32)
    movie[0] += 30.0
    src = _make_zarr(tmp_path, movie)

    out, outliers = reject_outlier_frames(
        src, tmp_path / "out.zarr", k_mad=5.0, batch_t=32, verbose=False,
    )
    assert 0 in outliers.tolist()
    arr = np.asarray(out)
    # The nearest non-outlier on the right is frame 1 (assuming no other
    # outliers from the random noise — k_mad=5 makes that very unlikely).
    np.testing.assert_allclose(arr[0], movie[1], atol=1e-5)


def test_sidecar_persists_outliers(tmp_path):
    rng = np.random.default_rng(3)
    movie = (rng.standard_normal((50, 4, 4)) + 5.0).astype(np.float32)
    movie[25] += 40.0
    src = _make_zarr(tmp_path, movie)

    dest = tmp_path / "out.zarr"
    _, outliers_1 = reject_outlier_frames(
        src, dest, k_mad=5.0, batch_t=32, verbose=False,
    )
    sidecar = dest.parent / f"{dest.name}.outliers.npy"
    assert sidecar.exists()
    np.testing.assert_array_equal(np.load(sidecar), outliers_1)

    # Reload via skip_if_exists.
    _, outliers_2 = reject_outlier_frames(
        src, dest, k_mad=5.0, batch_t=32, verbose=False, skip_if_exists=True,
    )
    np.testing.assert_array_equal(outliers_2, outliers_1)


def test_plan_replacement_splits_runs_and_guards_long_ones():
    """Short runs may be interpolated; long ones must not be.

    A linear blend across many frames moves every pixel together in a smooth ramp
    — a globally correlated event, which is the artifact the rejection exists to
    remove, only harder to spot, and it reads as a slow transient. Measured over
    fifteen real recordings the longest run was 7 frames, so this guards a case
    that does not arise routinely.
    """
    import numpy as np
    from minicnmfe.reject_frames import plan_replacement

    rng = np.random.default_rng(0)
    means = rng.normal(10.0, 0.2, 400)
    means[5] = 500.0            # 1-frame run
    means[50:53] = 500.0        # 3-frame run
    means[100:130] = 500.0      # 30-frame run

    idx, run_id, interp_ok = plan_replacement(means, max_interp_run=10)
    assert idx.size == 1 + 3 + 30
    assert len(np.unique(run_id)) == 3
    assert interp_ok[idx == 5][0]                       # short -> interpolate
    assert interp_ok[idx == 51][0]
    assert not interp_ok[idx == 115][0]                 # long  -> do not
    assert int((~interp_ok).sum()) == 30


def test_plan_replacement_degenerate_and_clean_inputs():
    """A clean movie flags nothing; a zero-MAD movie does not divide by zero."""
    import numpy as np
    from minicnmfe.reject_frames import plan_replacement

    rng = np.random.default_rng(1)
    idx, _, _ = plan_replacement(rng.normal(10.0, 0.2, 300))
    assert idx.size == 0, f"clean movie should flag nothing, flagged {idx.size}"

    idx, _, _ = plan_replacement(np.full(100, 7.0))     # MAD == 0
    assert idx.size == 0
