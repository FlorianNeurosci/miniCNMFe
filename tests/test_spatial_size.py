"""Dense-FOV footprint-size controls: ``spatial_lambda_scale`` (LASSO penalty
multiplier) and ``spatial_max_radius_factor`` (absolute circular-constraint cap).

Both default to no-op, so the bit-for-bit ``fit()`` path is pinned elsewhere
(``tests/test_stage_split.py``). Here we assert the *opt-in* behaviour: on a
dense field where footprints sprawl into neighbours, turning the knobs on
shrinks footprints (smaller median npix) **without losing real cells** (recall
against the simulator's ground-truth centres holds).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minicnmfe.pipeline import CNMFe, CNMFeParams  # noqa: E402
from minicnmfe.spatial import threshold_footprint  # noqa: E402
from tests.miniscope_simulator import make_miniscope_movie  # noqa: E402


def _diffuse_footprint(dims=(31, 31), sigma=5.0):
    """A low-contrast Gaussian blob with a broad skirt (sprawl-like)."""
    H, W = dims
    yy, xx = np.indices((H, W))
    r2 = (yy - H // 2) ** 2 + (xx - W // 2) ** 2
    return np.exp(-r2 / (2 * sigma ** 2)).astype(np.float32).ravel()


def test_nrg_thresholding_tightens_vs_max_and_is_optin():
    """thr_method='nrg' at a tight nrg_thr keeps fewer pixels than peak-relative
    'max'; the default ('max') is byte-identical to the legacy single-arg call."""
    dims = (31, 31)
    ai = _diffuse_footprint(dims)

    # Default path unchanged (bit-for-bit) vs explicit thr_method='max'.
    legacy = threshold_footprint(ai.copy(), dims)
    explicit_max = threshold_footprint(ai.copy(), dims, thr_method="max")
    assert np.array_equal(legacy, explicit_max)

    # Energy thresholding at 0.90 keeps fewer pixels than max@0.1 on a diffuse blob.
    out_max = threshold_footprint(ai.copy(), dims, max_thr=0.1, thr_method="max")
    out_nrg = threshold_footprint(ai.copy(), dims, thr_method="nrg", nrg_thr=0.90)
    assert (out_nrg > 0).sum() < (out_max > 0).sum()
    # And tighter nrg_thr keeps fewer pixels than looser.
    out_nrg_loose = threshold_footprint(ai.copy(), dims, thr_method="nrg", nrg_thr=0.999)
    assert (out_nrg > 0).sum() <= (out_nrg_loose > 0).sum()


def _footprint_npix(model) -> np.ndarray:
    """Per-component nonzero pixel count from the sparse footprint matrix."""
    A = model.A.tocsc()
    return np.diff(A.indptr)  # nnz per column


def _footprint_centroids(model, dims) -> np.ndarray:
    """(K, 2) intensity-weighted (row, col) centroid of each footprint."""
    H, W = dims
    A = model.A.tocsc()
    cents = []
    for k in range(A.shape[1]):
        col = A.getcol(k)
        rows = col.indices
        vals = col.data
        tot = vals.sum()
        if tot <= 0 or rows.size == 0:
            cents.append((np.nan, np.nan))
            continue
        yy, xx = rows // W, rows % W
        cents.append(((yy * vals).sum() / tot, (xx * vals).sum() / tot))
    return np.asarray(cents, dtype=np.float64)


def _recall(centroids: np.ndarray, gt_centers: np.ndarray, tol: float) -> int:
    """Number of ground-truth centres matched by some footprint within ``tol`` px."""
    valid = centroids[~np.isnan(centroids[:, 0])]
    if valid.size == 0:
        return 0
    matched = 0
    for gy, gx in gt_centers:
        d = np.sqrt((valid[:, 0] - gy) ** 2 + (valid[:, 1] - gx) ** 2)
        if d.min() <= tol:
            matched += 1
    return matched


@pytest.fixture(scope="module")
def dense_movie():
    """Small, densely-packed field: many compact cells so footprints crosstalk."""
    out = make_miniscope_movie(
        n_neurons=30,
        dims=(72, 72),
        T=300,
        sigma_neuron_range=(2.0, 3.0),   # small cells => small min_sep => dense
        seed=0,
    )
    return out["movie"].astype(np.float32), out["centers"].astype(np.float64)


def _fit(movie, *, lambda_scale, max_radius_factor):
    p = CNMFeParams(
        sigma=3.0,
        n_iter_main=2,
        n_jobs=1,
        # Pin peak-relative thresholding so this test isolates the lambda/radius
        # levers from the package default (now nrg, which already tightens both).
        spatial_thr_method="max",
        spatial_lambda_scale=lambda_scale,
        spatial_max_radius_factor=max_radius_factor,
    )
    model = CNMFe(p)
    model.fit(movie, do_motion_correction=False)
    return model


def test_knobs_shrink_footprints_without_losing_cells(dense_movie):
    movie, gt_centers = dense_movie
    dims = movie.shape[1:]

    base = _fit(movie, lambda_scale=1.0, max_radius_factor=0.0)
    tight = _fit(movie, lambda_scale=1.5, max_radius_factor=2.0)

    base_npix = _footprint_npix(base)
    tight_npix = _footprint_npix(tight)

    # (1) Footprints come out tighter.
    assert np.median(tight_npix) < np.median(base_npix), (
        f"median npix did not drop: base={np.median(base_npix)} "
        f"tight={np.median(tight_npix)}"
    )

    # (2) Recall holds: matched ground-truth centres do not fall (allow ±1).
    tol = 3.0  # px
    base_recall = _recall(_footprint_centroids(base, dims), gt_centers, tol)
    tight_recall = _recall(_footprint_centroids(tight, dims), gt_centers, tol)
    assert tight_recall >= base_recall - 1, (
        f"recall dropped: base={base_recall} tight={tight_recall} "
        f"(of {len(gt_centers)} GT cells)"
    )
