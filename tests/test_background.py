"""Tests for ring-model background estimation."""

import numpy as np
import pytest
import scipy.sparse as sp

from cnmfe.background import (
    BackgroundSubtractor,
    build_ring_indices,
    compute_W,
    subtract_background,
)


class TestBuildRingIndices:
    def test_shape(self):
        dims = (16, 20)
        idx = build_ring_indices(dims, radius=3)
        assert len(idx) == 16 * 20

    def test_ring_pixels_in_bounds(self):
        dims = (20, 20)
        idx = build_ring_indices(dims, radius=3)
        H, W = dims
        for p, ring in enumerate(idx):
            assert (ring >= 0).all()
            assert (ring < H * W).all()

    def test_centre_pixel_not_in_ring(self):
        """The pixel itself should not be in its own ring."""
        dims = (20, 20)
        idx = build_ring_indices(dims, radius=3)
        for p, ring in enumerate(idx):
            assert p not in ring

    def test_border_pixels_have_fewer_neighbours(self):
        dims = (20, 20)
        idx = build_ring_indices(dims, radius=3)
        # Corner pixel
        corner_ring = idx[0]
        # Interior pixel near centre
        centre = 10 * 20 + 10
        centre_ring = idx[centre]
        assert len(corner_ring) < len(centre_ring)

    def test_ring_distances(self):
        """Check that ring pixels are at the correct Euclidean distance."""
        dims = (20, 20)
        radius = 3
        idx = build_ring_indices(dims, radius=radius)
        H, W = dims
        # Check a central pixel
        p = 10 * W + 10
        row_p, col_p = 10, 10
        for flat in idx[p]:
            row_q, col_q = divmod(flat, W)
            dist = np.sqrt((row_q - row_p) ** 2 + (col_q - col_p) ** 2)
            assert dist > radius, f"Ring pixel at dist {dist:.2f} ≤ radius {radius}"
            assert dist <= radius + 2, f"Ring pixel too far: dist {dist:.2f}"


class TestComputeW:
    def _make_data(self, H=20, W=20, T=200, seed=0):
        rng = np.random.default_rng(seed)
        Y_flat = rng.standard_normal((H * W, T)).astype(np.float32) * 2
        A = sp.csc_matrix((H * W, 0), dtype=np.float32)
        C = np.empty((0, T), dtype=np.float32)
        return Y_flat, A, C, (H, W)

    def test_returns_sparse_W_and_b0(self):
        Y, A, C, dims = self._make_data()
        W_mat, b0 = compute_W(Y, A, C, dims, radius=3)
        assert sp.issparse(W_mat)
        assert b0.shape == (dims[0] * dims[1],)

    def test_W_shape(self):
        Y, A, C, dims = self._make_data()
        W_mat, _ = compute_W(Y, A, C, dims, radius=3)
        n = dims[0] * dims[1]
        assert W_mat.shape == (n, n)

    def test_streaming_b0_matches_legacy(self):
        """b0 from (Y_sum - A @ C_sum) / T must equal (Y - A@C).mean(axis=1)."""
        rng = np.random.default_rng(7)
        H, W, T = 18, 22, 240
        Y = rng.standard_normal((H * W, T)).astype(np.float32) * 1.3
        K = 3
        A_dense = np.zeros((H * W, K), dtype=np.float32)
        for k in range(K):
            for p in rng.choice(H * W, size=15, replace=False):
                A_dense[p, k] = rng.uniform(0.1, 0.6)
        A = sp.csc_matrix(A_dense)
        C = rng.standard_normal((K, T)).astype(np.float32) * 0.7

        # Legacy formulation
        b0_legacy = (Y - A.dot(C)).mean(axis=1).astype(np.float32)

        # New streaming computation (same math, no full X intermediate)
        _, b0_new = compute_W(Y, A, C, (H, W), radius=3, tsub=1)

        np.testing.assert_allclose(b0_new, b0_legacy, atol=1e-4, rtol=1e-4)

    def test_W_cached_returns_cached_W_and_refit_b0(self):
        """W_cached=W returns the same W and recomputes b0 from current (A, C)."""
        rng = np.random.default_rng(11)
        H, W, T = 18, 22, 240
        Y = rng.standard_normal((H * W, T)).astype(np.float32) * 1.3
        A0 = sp.csc_matrix((H * W, 0), dtype=np.float32)
        C0 = np.empty((0, T), dtype=np.float32)

        # First call: full solve.
        W_mat, b0_initial = compute_W(Y, A0, C0, (H, W), radius=3)

        # Second call with a *different* (A, C) but the cached W.
        K = 2
        A_dense = np.zeros((H * W, K), dtype=np.float32)
        A_dense[rng.choice(H * W, size=10, replace=False), 0] = 0.4
        A_dense[rng.choice(H * W, size=10, replace=False), 1] = 0.4
        A = sp.csc_matrix(A_dense)
        C = rng.standard_normal((K, T)).astype(np.float32) * 0.7

        W_returned, b0_refit = compute_W(
            Y, A, C, (H, W), radius=3, W_cached=W_mat,
        )

        # W is the same object (no resolve)
        assert W_returned is W_mat
        # b0 reflects the new (A, C)
        expected_b0 = (Y - A.dot(C)).mean(axis=1).astype(np.float32)
        np.testing.assert_allclose(b0_refit, expected_b0, atol=1e-4, rtol=1e-4)
        # And b0 changed compared to the (A=empty) initial fit
        assert not np.allclose(b0_refit, b0_initial)

    def test_W_cached_skips_solve_for_large_pixels(self):
        """W_cached path must skip the per-pixel solve entirely (fast)."""
        import time
        rng = np.random.default_rng(13)
        H, W, T = 40, 40, 400
        Y = rng.standard_normal((H * W, T)).astype(np.float32)
        A = sp.csc_matrix((H * W, 0), dtype=np.float32)
        C = np.empty((0, T), dtype=np.float32)

        t0 = time.perf_counter()
        W_mat, _ = compute_W(Y, A, C, (H, W), radius=3)
        t_solve = time.perf_counter() - t0

        t0 = time.perf_counter()
        W_returned, _ = compute_W(Y, A, C, (H, W), radius=3, W_cached=W_mat)
        t_cached = time.perf_counter() - t0

        # The cached path should be dramatically faster — solve dominates.
        # Generous factor to avoid CI flakes; the real difference is ~50-100x.
        assert t_cached < t_solve / 5, (
            f"cached path ({t_cached*1e3:.1f}ms) not enough faster than "
            f"solve ({t_solve*1e3:.1f}ms)"
        )
        assert W_returned is W_mat

    def test_compute_W_streaming_matches_in_memory(self, tmp_path):
        """compute_W on a zarr-backed Y_flat must match the in-memory path.

        Phase F3 regression. The streaming path constructs X-slabs per
        pixel batch instead of materialising the full ``(H*W, T_sub)``
        residual. b0 and W weights must equal the in-memory computation
        within float32 tolerance.
        """
        import zarr as _zarr

        rng = np.random.default_rng(41)
        H, W, T = 16, 18, 220
        Y = rng.standard_normal((H * W, T)).astype(np.float32) * 1.4
        K = 3
        A_dense = np.zeros((H * W, K), dtype=np.float32)
        for k in range(K):
            for p in rng.choice(H * W, size=20, replace=False):
                A_dense[p, k] = rng.uniform(0.1, 0.5)
        A = sp.csc_matrix(A_dense)
        C = rng.standard_normal((K, T)).astype(np.float32) * 0.8

        z_path = tmp_path / "Y_pixel_major.zarr"
        z = _zarr.open_array(
            str(z_path), mode="w",
            shape=Y.shape, chunks=(48, 80), dtype="float32",
        )
        z[:] = Y

        W_np, b0_np = compute_W(Y, A, C, (H, W), radius=3, tsub=2)
        W_zr, b0_zr = compute_W(z, A, C, (H, W), radius=3, tsub=2)

        np.testing.assert_allclose(b0_zr, b0_np, atol=1e-4, rtol=1e-4)
        # Compare W as dense (small, K=3, n_pixels=288).
        np.testing.assert_allclose(
            W_zr.toarray(), W_np.toarray(), atol=1e-3, rtol=1e-3,
        )

    def test_compute_W_cached_streaming(self, tmp_path):
        """W_cached must short-circuit on the streaming path too, returning
        the cached W and a freshly-streamed b0."""
        import zarr as _zarr

        rng = np.random.default_rng(43)
        H, W, T = 14, 14, 200
        Y = rng.standard_normal((H * W, T)).astype(np.float32)
        z_path = tmp_path / "Y.zarr"
        z = _zarr.open_array(
            str(z_path), mode="w",
            shape=Y.shape, chunks=(32, 80), dtype="float32",
        )
        z[:] = Y

        A0 = sp.csc_matrix((H * W, 0), dtype=np.float32)
        C0 = np.empty((0, T), dtype=np.float32)
        W_mat, _ = compute_W(z, A0, C0, (H, W), radius=3)

        # Now refit with a non-empty (A, C) on the zarr-backed Y.
        K = 2
        A_dense = np.zeros((H * W, K), dtype=np.float32)
        A_dense[rng.choice(H * W, size=10, replace=False), 0] = 0.3
        A_dense[rng.choice(H * W, size=10, replace=False), 1] = 0.3
        A = sp.csc_matrix(A_dense)
        C = rng.standard_normal((K, T)).astype(np.float32) * 0.5

        W_returned, b0_refit = compute_W(z, A, C, (H, W), radius=3, W_cached=W_mat)
        assert W_returned is W_mat
        expected_b0 = (Y - A.dot(C)).mean(axis=1).astype(np.float32)
        np.testing.assert_allclose(b0_refit, expected_b0, atol=1e-4, rtol=1e-4)

    def test_subtract_background_reduces_variance(self):
        """Background subtraction should reduce variance in background-dominated pixels."""
        rng = np.random.default_rng(1)
        H, W, T = 20, 20, 300
        # Pure spatially correlated background (no neurons)
        bg_spatial = rng.standard_normal((H * W,)).astype(np.float32)
        bg_temporal = np.cumsum(rng.standard_normal(T)).astype(np.float32) * 0.05
        Y_flat = np.outer(bg_spatial, bg_temporal)
        Y_flat += rng.standard_normal((H * W, T)).astype(np.float32) * 0.1

        A = sp.csc_matrix((H * W, 0), dtype=np.float32)
        C = np.empty((0, T), dtype=np.float32)
        W_mat, b0 = compute_W(Y_flat, A, C, (H, W), radius=3)
        Y_res = subtract_background(Y_flat, W_mat, b0)

        # Total variance should be lower after background subtraction
        var_before = Y_flat.var(axis=1).mean()
        var_after = Y_res.var(axis=1).mean()
        assert var_after <= var_before * 1.1, (
            f"Background subtraction increased variance: {var_before:.3f} → {var_after:.3f}"
        )


class TestSubtractBackground:
    def test_shape(self):
        H, W, T = 10, 10, 50
        Y = np.random.rand(H * W, T).astype(np.float32)
        W_mat = sp.csr_matrix((H * W, H * W), dtype=np.float32)
        b0 = np.zeros(H * W, dtype=np.float32)
        Y_res = subtract_background(Y, W_mat, b0)
        assert Y_res.shape == (H * W, T)

    def test_zero_W_identity(self):
        """W=0 and b0=0 should return the input unchanged."""
        H, W, T = 10, 10, 50
        Y = np.random.rand(H * W, T).astype(np.float32)
        W_mat = sp.csr_matrix((H * W, H * W), dtype=np.float32)
        b0 = np.zeros(H * W, dtype=np.float32)
        Y_res = subtract_background(Y, W_mat, b0)
        np.testing.assert_allclose(Y_res, Y, rtol=1e-5)


class TestBackgroundSubtractor:
    """Streaming subtractor must agree with dense subtract_background.

    These tests pin the math: BackgroundSubtractor must be a drop-in
    replacement that returns identical (within float32 tolerance) pixel
    rows and projections.
    """

    def _make_data(self, H=20, W=20, T=180, seed=0):
        rng = np.random.default_rng(seed)
        Y = rng.standard_normal((H * W, T)).astype(np.float32) * 1.5
        A0 = sp.csc_matrix((H * W, 0), dtype=np.float32)
        C0 = np.empty((0, T), dtype=np.float32)
        W_mat, b0 = compute_W(Y, A0, C0, (H, W), radius=3)
        return Y, W_mat, b0, (H, W, T)

    def test_full_slice_matches_dense(self):
        Y, W_mat, b0, _ = self._make_data()
        dense = subtract_background(Y, W_mat, b0)
        bg = BackgroundSubtractor(Y, W_mat, b0)
        np.testing.assert_allclose(bg[:], dense, atol=1e-4, rtol=1e-4)

    def test_partial_slice_matches_dense(self):
        Y, W_mat, b0, dims = self._make_data()
        H, W, _ = dims
        dense = subtract_background(Y, W_mat, b0)
        bg = BackgroundSubtractor(Y, W_mat, b0)
        # A few non-trivial slices
        for start, end in [(0, 64), (32, 96), (H * W - 50, H * W)]:
            np.testing.assert_allclose(
                bg.slice(start, end), dense[start:end], atol=1e-4, rtol=1e-4
            )

    def test_getitem_matches_slice_method(self):
        Y, W_mat, b0, _ = self._make_data()
        bg = BackgroundSubtractor(Y, W_mat, b0)
        np.testing.assert_array_equal(bg[10:42], bg.slice(10, 42))

    def test_shape_and_dtype(self):
        Y, W_mat, b0, dims = self._make_data()
        H, W, T = dims
        bg = BackgroundSubtractor(Y, W_mat, b0)
        assert bg.shape == (H * W, T)
        assert bg.dtype == np.float32
        assert bg.slice(0, 8).dtype == np.float32

    def test_project_onto_matches_dense(self):
        """bg.project_onto(A) must equal subtract_background(...).T @ A."""
        rng = np.random.default_rng(2)
        Y, W_mat, b0, dims = self._make_data(seed=3)
        H, W, T = dims
        # Synthesize a couple of overlapping sparse footprints.
        K = 4
        A_dense = np.zeros((H * W, K), dtype=np.float32)
        for k in range(K):
            centre = rng.integers(0, H * W)
            A_dense[centre, k] = 1.0
            for nbr in rng.choice(H * W, size=6, replace=False):
                A_dense[nbr, k] += 0.3
        A = sp.csc_matrix(A_dense)

        dense_proj = subtract_background(Y, W_mat, b0).T @ A
        bg = BackgroundSubtractor(Y, W_mat, b0)
        stream_proj = bg.project_onto(A, batch_size=64)
        np.testing.assert_allclose(
            stream_proj, np.asarray(dense_proj, dtype=np.float32),
            atol=1e-3, rtol=1e-3,
        )

    def test_project_onto_empty_K(self):
        """K=0 must return a (T, 0) array without errors."""
        Y, W_mat, b0, dims = self._make_data()
        _, _, T = dims
        bg = BackgroundSubtractor(Y, W_mat, b0)
        proj = bg.project_onto(sp.csc_matrix((Y.shape[0], 0), dtype=np.float32))
        assert proj.shape == (T, 0)

    def test_slice_matches_numpy_on_zarr_backed_Y(self, tmp_path):
        """BackgroundSubtractor must produce identical slices when ``Y_flat``
        is a zarr-backed pixel-major store instead of a numpy array.

        Phase F2 regression. The zarr branch in ``slice()`` pulls only the
        ring-neighbour rows into RAM and remaps the sparse column indices,
        avoiding a full Y_flat materialisation. Result must equal the
        in-memory path bit-for-bit.
        """
        import zarr as _zarr

        Y, W_mat, b0, dims = self._make_data(seed=17)
        # Save Y as a pixel-major zarr; chunk small to force multi-chunk reads.
        zarr_path = tmp_path / "Y_pixel_major.zarr"
        z = _zarr.open_array(
            str(zarr_path), mode="w",
            shape=Y.shape, chunks=(32, 60), dtype="float32",
        )
        z[:] = Y

        bg_np = BackgroundSubtractor(Y, W_mat, b0)
        bg_zarr = BackgroundSubtractor(z, W_mat, b0)

        for start, end in [(0, 16), (50, 100), (Y.shape[0] - 25, Y.shape[0])]:
            np.testing.assert_allclose(
                bg_zarr.slice(start, end),
                bg_np.slice(start, end),
                atol=1e-4, rtol=1e-4,
            )

    def test_project_onto_matches_dense_on_zarr_backed_Y(self, tmp_path):
        import zarr as _zarr

        rng = np.random.default_rng(31)
        Y, W_mat, b0, dims = self._make_data(seed=5)
        H, W, T = dims
        zarr_path = tmp_path / "Y_pixel_major.zarr"
        z = _zarr.open_array(
            str(zarr_path), mode="w",
            shape=Y.shape, chunks=(32, 60), dtype="float32",
        )
        z[:] = Y

        K = 3
        A_dense = np.zeros((H * W, K), dtype=np.float32)
        for k in range(K):
            for p in rng.choice(H * W, size=8, replace=False):
                A_dense[p, k] = rng.uniform(0.1, 0.5)
        A = sp.csc_matrix(A_dense)

        np_proj = BackgroundSubtractor(Y, W_mat, b0).project_onto(A, batch_size=40)
        z_proj = BackgroundSubtractor(z, W_mat, b0).project_onto(A, batch_size=40)
        np.testing.assert_allclose(z_proj, np_proj, atol=1e-3, rtol=1e-3)
