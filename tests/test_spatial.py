"""Tests for spatial footprint update."""

import numpy as np
import pytest
import scipy.sparse as sp

from cnmfe.spatial import compute_support, threshold_footprint, update_spatial


class TestComputeSupport:
    def test_shape(self, synth_small):
        d = synth_small
        H, W = d["dims"]
        A = sp.csc_matrix(d["A_true"].astype(np.float32))
        support = compute_support(A, (H, W), dilation_radius=3)
        assert len(support) == H * W

    def test_nonzero_pixels_in_support(self, synth_small):
        """Every pixel that is nonzero in a footprint should be in its own support."""
        d = synth_small
        H, W = d["dims"]
        A = sp.csc_matrix(d["A_true"].astype(np.float32))
        support = compute_support(A, (H, W), dilation_radius=3)

        for k in range(A.shape[1]):
            col = np.asarray(A.getcol(k).todense()).ravel()
            nz_pixels = np.where(col > 0)[0]
            for p in nz_pixels:
                assert k in support[p], f"Component {k} missing from support of pixel {p}"

    def test_dilation_extends_support(self, synth_small):
        """Larger dilation radius should produce more or equal support pixels."""
        d = synth_small
        H, W = d["dims"]
        A = sp.csc_matrix(d["A_true"].astype(np.float32))
        support_small = compute_support(A, (H, W), dilation_radius=1)
        support_large = compute_support(A, (H, W), dilation_radius=5)
        total_small = sum(len(s) for s in support_small)
        total_large = sum(len(s) for s in support_large)
        assert total_large >= total_small


class TestThresholdFootprint:
    def test_output_shape(self):
        H, W = 20, 20
        ai = np.random.rand(H * W).astype(np.float32)
        result = threshold_footprint(ai, (H, W))
        assert result.shape == (H * W,)

    def test_single_connected_component(self):
        """After thresholding, footprint should have ≤ 1 connected component."""
        H, W = 20, 20
        # Two isolated blobs
        ai = np.zeros((H, W), dtype=np.float32)
        ai[3:6, 3:6] = 1.0
        ai[14:17, 14:17] = 0.5  # smaller blob, will be zeroed out
        result = threshold_footprint(ai.ravel(), (H, W)).reshape(H, W)
        # Should keep only one component
        import scipy.ndimage as ndi
        _, n = ndi.label(result > 0)
        assert n <= 1

    def test_zeros_return_zeros(self):
        ai = np.zeros(100, dtype=np.float32)
        result = threshold_footprint(ai, (10, 10))
        np.testing.assert_array_equal(result, 0)

    def test_non_negative(self):
        ai = np.random.rand(400).astype(np.float32)
        result = threshold_footprint(ai, (20, 20))
        assert (result >= 0).all()


class TestUpdateSpatial:
    def test_output_shape(self, synth_small):
        d = synth_small
        T = d["movie"].shape[0]
        H, W = d["dims"]
        K = d["A_true"].shape[1]
        Y_flat = d["movie"].reshape(T, H * W).T
        A = sp.csc_matrix(d["A_true"].astype(np.float32))
        C = d["C_true"].copy()
        sn = np.ones(H * W, dtype=np.float32) * d["sn_true"]

        A_new = update_spatial(Y_flat, C, A, sn, (H, W), dilation_radius=3)
        assert A_new.shape == (H * W, K)

    def test_non_negative_footprints(self, synth_small):
        d = synth_small
        T = d["movie"].shape[0]
        H, W = d["dims"]
        Y_flat = d["movie"].reshape(T, H * W).T
        A = sp.csc_matrix(d["A_true"].astype(np.float32))
        C = d["C_true"].copy()
        sn = np.ones(H * W, dtype=np.float32) * d["sn_true"]

        A_new = update_spatial(Y_flat, C, A, sn, (H, W))
        assert A_new.data.min() >= -1e-6
