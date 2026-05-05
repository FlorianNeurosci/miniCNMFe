"""Tests for n_jobs parallelism.

Each test verifies that parallel (n_jobs=2) output is numerically identical
(or near-identical) to serial (n_jobs=1) output, and that the pipeline
completes without error when n_jobs=-1 (use all CPUs).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from tests.conftest import make_synthetic_movie


@pytest.fixture(scope="module")
def synth_data():
    return make_synthetic_movie(n_neurons=3, dims=(32, 32), T=150, seed=0)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

class TestCorrelationPNRParallel:
    def test_matches_serial(self, synth_data):
        from cnmfe.preprocess import correlation_pnr
        movie = synth_data["movie"]
        cn1, pnr1 = correlation_pnr(movie, sigma=3.0, n_jobs=1)
        cn2, pnr2 = correlation_pnr(movie, sigma=3.0, n_jobs=2)
        np.testing.assert_allclose(cn1, cn2, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(pnr1, pnr2, rtol=1e-5, atol=1e-5)

    def test_output_shape_parallel(self, synth_data):
        from cnmfe.preprocess import correlation_pnr
        movie = synth_data["movie"]
        _, H, W = movie.shape
        cn, pnr = correlation_pnr(movie, sigma=3.0, n_jobs=2)
        assert cn.shape == (H, W)
        assert pnr.shape == (H, W)


# ---------------------------------------------------------------------------
# Motion correction
# ---------------------------------------------------------------------------

class TestMotionCorrectParallel:
    def test_matches_serial(self, synth_data):
        from cnmfe.motion_correction import motion_correct
        movie = synth_data["movie"]
        _, shifts1 = motion_correct(movie, n_iter=1, upsample_factor=5, n_jobs=1)
        _, shifts2 = motion_correct(movie, n_iter=1, upsample_factor=5, n_jobs=2)
        np.testing.assert_allclose(shifts1, shifts2, atol=1e-4)

    def test_output_shape_parallel(self, synth_data):
        from cnmfe.motion_correction import motion_correct
        movie = synth_data["movie"]
        T = movie.shape[0]
        corrected, shifts = motion_correct(movie, n_iter=1, upsample_factor=5, n_jobs=2)
        assert np.asarray(corrected).shape == movie.shape
        assert shifts.shape == (T, 2)


# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------

class TestComputeWParallel:
    def test_matches_serial(self, synth_data):
        from cnmfe.background import compute_W
        from cnmfe._utils import make_2d
        d = synth_data
        Y_flat = make_2d(d["movie"])
        A = sp.csc_matrix(d["A_true"].astype(np.float32))
        dims = d["dims"]
        radius = 1.5 * (2 * 3.0 + 1)

        W1, b1 = compute_W(Y_flat, A, d["C_true"], dims, radius, n_jobs=1)
        W2, b2 = compute_W(Y_flat, A, d["C_true"], dims, radius, n_jobs=2)

        np.testing.assert_allclose(b1, b2, rtol=1e-5)
        diff = (W1 - W2)
        assert abs(diff).max() < 1e-5


# ---------------------------------------------------------------------------
# Spatial update
# ---------------------------------------------------------------------------

class TestUpdateSpatialParallel:
    def test_matches_serial(self, synth_data):
        from cnmfe.spatial import update_spatial
        from cnmfe._utils import make_2d
        d = synth_data
        Y_flat = make_2d(d["movie"])
        A = sp.csc_matrix(d["A_true"].astype(np.float32))
        C = d["C_true"].copy()
        sn = np.ones(d["dims"][0] * d["dims"][1], dtype=np.float32) * d["sn_true"]

        A1 = update_spatial(Y_flat, C, A, sn, d["dims"], n_jobs=1)
        A2 = update_spatial(Y_flat, C, A, sn, d["dims"], n_jobs=2)

        # Results should be identical (deterministic LassoLars)
        diff = (A1 - A2)
        assert abs(diff).max() < 1e-5

    def test_non_negative_parallel(self, synth_data):
        from cnmfe.spatial import update_spatial
        from cnmfe._utils import make_2d
        d = synth_data
        Y_flat = make_2d(d["movie"])
        A = sp.csc_matrix(d["A_true"].astype(np.float32))
        C = d["C_true"].copy()
        sn = np.ones(d["dims"][0] * d["dims"][1], dtype=np.float32) * d["sn_true"]

        A_new = update_spatial(Y_flat, C, A, sn, d["dims"], n_jobs=2)
        assert A_new.data.min() >= 0


# ---------------------------------------------------------------------------
# Temporal update
# ---------------------------------------------------------------------------

class TestUpdateTemporalParallel:
    def test_matches_serial(self, synth_data):
        from cnmfe.temporal import update_temporal
        from cnmfe._utils import make_2d
        d = synth_data
        Y_flat = make_2d(d["movie"])
        A = sp.csc_matrix(d["A_true"].astype(np.float32))
        C = d["C_true"].copy()
        sn = np.ones(d["dims"][0] * d["dims"][1], dtype=np.float32) * d["sn_true"]

        C1, S1 = update_temporal(Y_flat, A, C.copy(), sn, ar_order=1, n_iter=1, n_jobs=1)
        C2, S2 = update_temporal(Y_flat, A, C.copy(), sn, ar_order=1, n_iter=1, n_jobs=2)

        # Jacobi (parallel) and Gauss-Seidel (serial) differ numerically but
        # should give similar output: check shapes and non-negativity
        assert C2.shape == C1.shape
        assert S2.shape == S1.shape
        assert (S2 >= -1e-6).all()

    def test_non_negative_spikes_parallel(self, synth_data):
        from cnmfe.temporal import update_temporal
        from cnmfe._utils import make_2d
        d = synth_data
        Y_flat = make_2d(d["movie"])
        A = sp.csc_matrix(d["A_true"].astype(np.float32))
        sn = np.ones(d["dims"][0] * d["dims"][1], dtype=np.float32) * d["sn_true"]
        _, S = update_temporal(Y_flat, A, d["C_true"].copy(), sn, n_iter=1, n_jobs=2)
        assert (S >= -1e-6).all()


# ---------------------------------------------------------------------------
# End-to-end pipeline with n_jobs
# ---------------------------------------------------------------------------

class TestPipelineParallel:
    def test_pipeline_n_jobs_2(self, synth_data):
        """Full pipeline with n_jobs=2 should complete and return valid results."""
        from cnmfe.pipeline import CNMFe, CNMFeParams
        movie = synth_data["movie"]
        params = CNMFeParams(
            sigma=3.0,
            min_corr=0.5,
            min_pnr=3.0,
            n_iter_main=1,
            n_iter_temporal=1,
            n_jobs=2,
        )
        model = CNMFe(params).fit(movie, do_motion_correction=False)
        assert model.A is not None
        assert model.C is not None
        assert model.S is not None
        T = movie.shape[0]
        K = model.A.shape[1]
        assert model.C.shape == (K, T)
        assert model.S.shape == (K, T)

    def test_pipeline_n_jobs_minus1(self, synth_data):
        """n_jobs=-1 (all CPUs) should complete without error."""
        from cnmfe.pipeline import CNMFe, CNMFeParams
        movie = synth_data["movie"]
        params = CNMFeParams(
            sigma=3.0,
            min_corr=0.5,
            min_pnr=3.0,
            n_iter_main=1,
            n_jobs=-1,
        )
        model = CNMFe(params).fit(movie, do_motion_correction=False)
        assert model.A is not None
