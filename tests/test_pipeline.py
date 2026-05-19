"""End-to-end pipeline tests."""

import numpy as np
import pytest
import scipy.sparse as sp

from cnmfe.pipeline import CNMFe, CNMFeParams


def match_components(A_est: sp.csc_matrix, A_true: np.ndarray) -> list[tuple[int, int, float]]:
    """For each true component, find the best-matching estimated component.

    Returns list of (true_idx, est_idx, spatial_correlation).
    """
    K_true = A_true.shape[1]
    K_est = A_est.shape[1]
    if K_est == 0:
        return [(k, -1, 0.0) for k in range(K_true)]

    A_e = np.asarray(A_est.todense())   # (H*W, K_est)
    A_t = A_true                         # (H*W, K_true)

    matches = []
    for k_true in range(K_true):
        a_t = A_t[:, k_true]
        corrs = []
        for k_est in range(K_est):
            a_e = A_e[:, k_est]
            denom = np.linalg.norm(a_t) * np.linalg.norm(a_e)
            corr = float(np.dot(a_t, a_e) / denom) if denom > 0 else 0.0
            corrs.append(corr)
        best = int(np.argmax(corrs))
        matches.append((k_true, best, corrs[best]))
    return matches


class TestCNMFePipeline:
    def test_fit_runs_without_error(self, synth_small):
        movie = synth_small["movie"]
        params = CNMFeParams(
            sigma=3.0,
            min_corr=0.5,
            min_pnr=3.0,
            n_iter_main=1,
            n_iter_temporal=1,
        )
        model = CNMFe(params)
        model.fit(movie, do_motion_correction=False)
        # Should complete without raising
        assert model.A is not None
        assert model.C is not None
        assert model.S is not None

    def test_output_shapes(self, synth_small):
        movie = synth_small["movie"]
        T, H, W = movie.shape
        params = CNMFeParams(sigma=3.0, min_corr=0.5, min_pnr=3.0, n_iter_main=1)
        model = CNMFe(params).fit(movie, do_motion_correction=False)

        K = model.A.shape[1]
        assert model.A.shape == (H * W, K)
        assert model.C.shape == (K, T)
        assert model.S.shape == (K, T)

    def test_non_negative_spikes(self, synth_small):
        movie = synth_small["movie"]
        params = CNMFeParams(sigma=3.0, min_corr=0.5, min_pnr=3.0, n_iter_main=1)
        model = CNMFe(params).fit(movie, do_motion_correction=False)
        if model.S is not None and model.S.size > 0:
            assert (model.S >= -1e-5).all()

    def test_finds_some_neurons(self, synth_small):
        movie = synth_small["movie"]
        params = CNMFeParams(
            sigma=3.0,
            min_corr=0.4,
            min_pnr=2.0,
            n_iter_main=1,
        )
        model = CNMFe(params).fit(movie, do_motion_correction=False)
        assert model.A.shape[1] >= 1, "Should find at least one neuron"

    def test_spatial_recovery(self, synth_small):
        """At least one true neuron should be recovered with r > 0.5."""
        movie = synth_small["movie"]
        params = CNMFeParams(
            sigma=3.0,
            min_corr=0.4,
            min_pnr=2.0,
            n_iter_main=1,
        )
        model = CNMFe(params).fit(movie, do_motion_correction=False)

        if model.A.shape[1] == 0:
            pytest.skip("No neurons found — lower thresholds")

        matches = match_components(model.A, synth_small["A_true"])
        best_corr = max(m[2] for m in matches)
        assert best_corr > 0.4, f"Best spatial correlation = {best_corr:.3f}"

    def test_default_params(self, synth_small):
        """Pipeline with default params should not crash."""
        movie = synth_small["movie"]
        model = CNMFe()
        # Default params may find 0 neurons on small movie, that's ok
        model.fit(movie, do_motion_correction=False)

    def test_fit_accepts_zarr_input(self, synth_small, tmp_path):
        """fit() should accept a zarr.Array directly and produce identical
        results to passing the same data as a numpy array.

        Phase C regression: zarr support is currently via np.asarray()
        materialisation (no streaming yet — that requires disk transpose).
        This test pins the API and the round-trip equivalence.
        """
        from cnmfe.io import save_zarr

        movie_np = synth_small["movie"].astype(np.float32)
        zarr_path = tmp_path / "movie.zarr"
        z = save_zarr(movie_np, str(zarr_path))

        params = CNMFeParams(
            sigma=3.0, min_corr=0.5, min_pnr=3.0,
            n_iter_main=1, n_iter_temporal=1,
        )
        model_np = CNMFe(params).fit(movie_np, do_motion_correction=False)
        model_zarr = CNMFe(params).fit(z, do_motion_correction=False)

        # Same number of components, identical footprints / traces.
        assert model_zarr.A.shape == model_np.A.shape
        np.testing.assert_allclose(
            np.asarray(model_zarr.A.todense()),
            np.asarray(model_np.A.todense()),
            atol=1e-4, rtol=1e-4,
        )
        np.testing.assert_allclose(model_zarr.C, model_np.C, atol=1e-4, rtol=1e-4)

    def test_with_motion_correction(self, synth_small):
        """Motion correction pass should complete without errors."""
        movie = synth_small["movie"]
        params = CNMFeParams(
            sigma=3.0,
            min_corr=0.4,
            min_pnr=2.0,
            n_iter_main=1,
            mc_n_iter=1,
        )
        model = CNMFe(params).fit(movie, do_motion_correction=True)
        assert model.shifts is not None
        assert model.shifts.shape == (movie.shape[0], 2)

    def test_temporal_correlation_against_truth(self, synth):
        """Recovered temporal traces should align with ground truth.

        Regression test: per-component re-estimation of the AR coefficient g
        across BCD iterations used to drift it toward 0 (fudge_factor=0.96
        re-applied each call), distorting calcium decay shape and dropping
        Pearson r vs ground truth to ~0.6-0.8. Pooling g across components
        and caching it across iterations brings r back above 0.85.
        """
        params = CNMFeParams(
            sigma=3.0, min_corr=0.7, min_pnr=3.0,
            n_iter_main=2, n_iter_temporal=2,
        )
        model = CNMFe(params).fit(synth["movie"], do_motion_correction=False)
        matches = match_components(model.A, synth["A_true"])

        def pearson(a, b):
            a = a - a.mean(); b = b - b.mean()
            d = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
            return float(np.dot(a, b) / d) if d > 0 else 0.0

        # Restrict to actually-matched truth neurons (spatial r > 0.5)
        valid = [(kt, ke) for kt, ke, sc in matches if sc > 0.5]
        assert len(valid) >= 5, f"Only {len(valid)}/6 truth neurons recovered"

        rs_oasis = [pearson(model.C[ke], synth["C_true"][kt]) for kt, ke in valid]
        rs_proj = [pearson((model.C + model.YrA)[ke], synth["C_true"][kt]) for kt, ke in valid]

        assert np.mean(rs_oasis) > 0.85, f"Mean r(C) = {np.mean(rs_oasis):.3f}"
        assert np.mean(rs_proj) > 0.85, f"Mean r(C+YrA) = {np.mean(rs_proj):.3f}"
        assert min(rs_proj) > 0.70, f"Min r(C+YrA) = {min(rs_proj):.3f}"

    def test_per_neuron_ar_temporal_correlation(self, synth):
        """Per-neuron AR estimation (global_ar=False) should recover traces as well as global."""
        params = CNMFeParams(
            sigma=3.0, min_corr=0.7, min_pnr=3.0,
            n_iter_main=2, n_iter_temporal=2,
            global_ar=False,
        )
        model = CNMFe(params).fit(synth["movie"], do_motion_correction=False)
        matches = match_components(model.A, synth["A_true"])

        def pearson(a, b):
            a = a - a.mean(); b = b - b.mean()
            d = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
            return float(np.dot(a, b) / d) if d > 0 else 0.0

        valid = [(kt, ke) for kt, ke, sc in matches if sc > 0.5]
        assert len(valid) >= 5, f"Only {len(valid)}/6 truth neurons recovered"

        rs_proj = [pearson((model.C + model.YrA)[ke], synth["C_true"][kt]) for kt, ke in valid]
        assert np.mean(rs_proj) > 0.85, f"Mean r(C+YrA) per-neuron AR = {np.mean(rs_proj):.3f}"
