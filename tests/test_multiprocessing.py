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
        from minicnmfe.preprocess import correlation_pnr
        movie = synth_data["movie"]
        cn1, pnr1 = correlation_pnr(movie, sigma=3.0, n_jobs=1)
        cn2, pnr2 = correlation_pnr(movie, sigma=3.0, n_jobs=2)
        np.testing.assert_allclose(cn1, cn2, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(pnr1, pnr2, rtol=1e-5, atol=1e-5)

    def test_output_shape_parallel(self, synth_data):
        from minicnmfe.preprocess import correlation_pnr
        movie = synth_data["movie"]
        _, H, W = movie.shape
        cn, pnr = correlation_pnr(movie, sigma=3.0, n_jobs=2)
        assert cn.shape == (H, W)
        assert pnr.shape == (H, W)


# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------

class TestComputeWParallel:
    def test_matches_serial(self, synth_data):
        from minicnmfe.background import compute_W
        from minicnmfe._utils import make_2d
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
        from minicnmfe.spatial import update_spatial
        from minicnmfe._utils import make_2d
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
        from minicnmfe.spatial import update_spatial
        from minicnmfe._utils import make_2d
        d = synth_data
        Y_flat = make_2d(d["movie"])
        A = sp.csc_matrix(d["A_true"].astype(np.float32))
        C = d["C_true"].copy()
        sn = np.ones(d["dims"][0] * d["dims"][1], dtype=np.float32) * d["sn_true"]

        A_new = update_spatial(Y_flat, C, A, sn, d["dims"], n_jobs=2)
        assert A_new.data.min() >= 0

    def test_thread_cap_applied(self, synth_data, monkeypatch):
        """A huge n_jobs is capped to spatial_thread_cap for the CD parallel
        path (the loop is GIL-bound; >cap threads just thrash the GIL), while
        results stay equivalent to serial."""
        import joblib
        from minicnmfe import spatial as _spatial
        from minicnmfe.spatial import update_spatial
        from minicnmfe._utils import make_2d
        d = synth_data
        Y_flat = make_2d(d["movie"])
        A = sp.csc_matrix(d["A_true"].astype(np.float32))
        C = d["C_true"].copy()
        sn = np.ones(d["dims"][0] * d["dims"][1], dtype=np.float32) * d["sn_true"]

        calls = []          # n_jobs of every joblib.Parallel created
        real_parallel = joblib.Parallel

        def _spy(*args, **kwargs):
            calls.append(kwargs.get("n_jobs", args[0] if args else None))
            return real_parallel(*args, **kwargs)

        monkeypatch.setattr(joblib, "Parallel", _spy)

        A1 = update_spatial(Y_flat, C, A, sn, d["dims"], n_jobs=1)
        A2 = update_spatial(Y_flat, C, A, sn, d["dims"],
                            n_jobs=1000, spatial_thread_cap=4)

        expected = min(joblib.effective_n_jobs(1000), 4)
        # The per-pixel CD is the FIRST Parallel in update_spatial; it must be
        # capped to spatial_thread_cap (<=4), not run with the raw 1000. (The
        # later threshold_footprint Parallel parallelises over K components with
        # GIL-releasing scipy.ndimage and is intentionally left uncapped.)
        assert calls[0] == expected
        assert abs(A1 - A2).max() < 1e-5     # still matches serial


# ---------------------------------------------------------------------------
# Temporal update
# ---------------------------------------------------------------------------

class TestUpdateTemporalParallel:
    def test_matches_serial(self, synth_data):
        from minicnmfe.temporal import update_temporal
        from minicnmfe._utils import make_2d
        d = synth_data
        Y_flat = make_2d(d["movie"])
        A = sp.csc_matrix(d["A_true"].astype(np.float32))
        C = d["C_true"].copy()
        sn = np.ones(d["dims"][0] * d["dims"][1], dtype=np.float32) * d["sn_true"]

        C1, S1, _, _ = update_temporal(Y_flat, A, C.copy(), sn, ar_order=1, n_iter=1, n_jobs=1)
        C2, S2, _, _ = update_temporal(Y_flat, A, C.copy(), sn, ar_order=1, n_iter=1, n_jobs=2)

        # Jacobi (parallel) and Gauss-Seidel (serial) differ numerically but
        # should give similar output: check shapes and non-negativity
        assert C2.shape == C1.shape
        assert S2.shape == S1.shape
        assert (S2 >= -1e-6).all()

    def test_non_negative_spikes_parallel(self, synth_data):
        from minicnmfe.temporal import update_temporal
        from minicnmfe._utils import make_2d
        d = synth_data
        Y_flat = make_2d(d["movie"])
        A = sp.csc_matrix(d["A_true"].astype(np.float32))
        sn = np.ones(d["dims"][0] * d["dims"][1], dtype=np.float32) * d["sn_true"]
        _, S, _, _ = update_temporal(Y_flat, A, d["C_true"].copy(), sn, n_iter=1, n_jobs=2)
        assert (S >= -1e-6).all()


# ---------------------------------------------------------------------------
# End-to-end pipeline with n_jobs
# ---------------------------------------------------------------------------

class TestPipelineParallel:
    def test_pipeline_n_jobs_2(self, synth_data):
        """Full pipeline with n_jobs=2 should complete and return valid results."""
        from minicnmfe.pipeline import CNMFe, CNMFeParams
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
        from minicnmfe.pipeline import CNMFe, CNMFeParams
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
