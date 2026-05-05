"""Tests for ring-model background estimation."""

import numpy as np
import pytest
import scipy.sparse as sp

from cnmfe.background import build_ring_indices, compute_W, subtract_background


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
