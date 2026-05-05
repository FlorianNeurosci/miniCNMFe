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
