"""Tests for cnmfe.gui.contours."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from cnmfe.gui.contours import (
    component_contour,
    footprint_image,
    precompute_centroids,
)


def test_centroid_single_pixel():
    H, W = 10, 12
    # Two components, one at (3, 4), one at (7, 2).
    A_dense = np.zeros((H * W, 2), dtype=np.float32)
    A_dense[3 * W + 4, 0] = 1.0
    A_dense[7 * W + 2, 1] = 1.0
    A = sp.csc_matrix(A_dense)
    centroids = precompute_centroids(A, H, W)
    np.testing.assert_allclose(centroids[0], [3.0, 4.0])
    np.testing.assert_allclose(centroids[1], [7.0, 2.0])


def test_centroid_weighted_mean():
    H, W = 8, 8
    A_dense = np.zeros((H * W, 1), dtype=np.float32)
    # Three pixels with different weights along x=4
    A_dense[2 * W + 4, 0] = 1.0  # y=2
    A_dense[4 * W + 4, 0] = 3.0  # y=4, heavier
    A = sp.csc_matrix(A_dense)
    cy, cx = precompute_centroids(A, H, W)[0]
    # weighted mean: (1*2 + 3*4) / 4 = 14/4 = 3.5
    assert abs(cy - 3.5) < 1e-5
    assert abs(cx - 4.0) < 1e-5


def test_centroid_empty_column_fallback():
    H, W = 6, 6
    A = sp.csc_matrix((H * W, 2), dtype=np.float32)  # all zero
    centroids = precompute_centroids(A, H, W)
    assert centroids.shape == (2, 2)
    np.testing.assert_allclose(centroids, [[2.5, 2.5], [2.5, 2.5]])


def test_footprint_image_round_trip():
    H, W = 5, 7
    img = np.zeros((H, W), dtype=np.float32)
    img[1, 2] = 0.5
    img[3, 5] = 1.0
    A = sp.csc_matrix(img.reshape(-1, 1))
    out = footprint_image(A, 0, H, W)
    np.testing.assert_array_equal(out, img)


def test_component_contour_nonempty_for_gaussian():
    H, W = 32, 32
    yy, xx = np.mgrid[:H, :W]
    blob = np.exp(-((yy - 16) ** 2 + (xx - 16) ** 2) / (2 * 3 ** 2)).astype(
        np.float32
    )
    contours = component_contour(blob, level_frac=0.3)
    assert len(contours) >= 1
    # Each contour is a (N, 2) array of floats.
    for c in contours:
        assert c.ndim == 2 and c.shape[1] == 2


def test_component_contour_empty_for_zero_footprint():
    z = np.zeros((10, 10), dtype=np.float32)
    assert component_contour(z) == []
