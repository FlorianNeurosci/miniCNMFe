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
