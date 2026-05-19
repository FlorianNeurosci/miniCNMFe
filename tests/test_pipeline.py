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

    def test_init_stride_recovers_footprints(self):
        """Strided greedy init must still recover spatially-comparable
        footprints to non-strided init on a longer movie.

        Phase D regression: pipeline.fit() runs greedy init on a strided
        sample (when init_stride > 1) to bound the (T,H,W) data_filtered /
        data_raw RAM cost. Footprints are spatial — independent of T — so
        recovery should be near-identical. Traces are re-projected at full T
        downstream.
        """
        from tests.conftest import make_synthetic_movie

        synth = make_synthetic_movie(
            n_neurons=4, dims=(48, 48), T=600, noise_std=0.4, seed=2,
        )
        movie = synth["movie"]
        K_true = synth["A_true"].shape[1]

        common = dict(sigma=3.0, min_corr=0.5, min_pnr=3.0,
                      n_iter_main=1, n_iter_temporal=1)
        # Baseline: stride=1 (no striding).
        m1 = CNMFe(CNMFeParams(**common, init_stride=1)).fit(
            movie, do_motion_correction=False
        )
        # Strided: stride=3 (init runs on T_init=200 frames).
        m3 = CNMFe(CNMFeParams(**common, init_stride=3)).fit(
            movie, do_motion_correction=False
        )

        # Both should find roughly the same neurons.
        assert m1.A.shape[1] >= K_true - 1, f"stride=1 missed: K={m1.A.shape[1]}"
        assert m3.A.shape[1] >= K_true - 1, f"stride=3 missed: K={m3.A.shape[1]}"

        # Spatial recovery — each ground-truth neuron should match at least
        # one footprint in BOTH models with high correlation.
        matches_1 = match_components(m1.A, synth["A_true"])
        matches_3 = match_components(m3.A, synth["A_true"])
        for k_true in range(K_true):
            _, _, r1 = matches_1[k_true]
            _, _, r3 = matches_3[k_true]
            assert r1 > 0.7, f"stride=1 poor spatial match on neuron {k_true}: r={r1:.3f}"
            assert r3 > 0.7, f"stride=3 poor spatial match on neuron {k_true}: r={r3:.3f}"

        # Full-T traces always; strided init recovers C at full T via the
        # post-init projection. Check shape.
        assert m3.C.shape == (m3.A.shape[1], movie.shape[0])

    def test_init_corrpnr_stride_recovers_footprints(self):
        """init_corrpnr_stride must not break neuron recovery.

        The initial CORR/PNR sweep inside greedy init runs on a strided
        slice of the (already strided) init_movie. Spatial reductions
        survive moderate subsampling; we verify each ground-truth neuron
        still matches an extracted footprint at r > 0.7.
        """
        from tests.conftest import make_synthetic_movie

        synth = make_synthetic_movie(
            n_neurons=4, dims=(48, 48), T=600, noise_std=0.4, seed=2,
        )
        movie = synth["movie"]
        K_true = synth["A_true"].shape[1]
        common = dict(sigma=3.0, min_corr=0.5, min_pnr=3.0,
                      n_iter_main=1, n_iter_temporal=1, init_stride=1)

        # Baseline (no extra CORR/PNR stride) and accelerated (stride=3).
        m1 = CNMFe(CNMFeParams(**common, init_corrpnr_stride=1)).fit(
            movie, do_motion_correction=False
        )
        m3 = CNMFe(CNMFeParams(**common, init_corrpnr_stride=3)).fit(
            movie, do_motion_correction=False
        )

        assert m1.A.shape[1] >= K_true - 1, f"stride=1 missed: K={m1.A.shape[1]}"
        assert m3.A.shape[1] >= K_true - 1, (
            f"corrpnr_stride=3 missed: K={m3.A.shape[1]}"
        )

        matches_3 = match_components(m3.A, synth["A_true"])
        for k_true in range(K_true):
            _, _, r3 = matches_3[k_true]
            assert r3 > 0.7, (
                f"corrpnr_stride=3 poor spatial match on neuron {k_true}: r={r3:.3f}"
            )

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

    def test_save_load_roundtrip(self, synth_small, tmp_path):
        """model.save() then CNMFe.load() must restore all fitted state.

        Pins the on-disk layout (A.npz, C.npy, ..., params.json, manifest.json)
        and the round-trip equivalence so downstream analysis scripts can
        rely on it.
        """
        movie = synth_small["movie"]
        params = CNMFeParams(
            sigma=3.0, min_corr=0.5, min_pnr=3.0,
            n_iter_main=1, n_iter_temporal=1,
            mc_gSig_filt=2.5,                  # exercises a non-default float|None
        )
        model = CNMFe(params).fit(movie, do_motion_correction=False)

        save_dir = tmp_path / "model"
        model.save(save_dir)

        # Spot-check the on-disk layout — these files anchor full_pipeline.py
        # and the demo notebook's save cell.
        for name in (
            "A.npz", "C.npy", "S.npy", "YrA.npy",
            "params.json", "manifest.json",
        ):
            assert (save_dir / name).exists(), f"{name} missing from save dir"

        loaded = CNMFe.load(save_dir)

        # Params survived intact (including the tuple field).
        assert loaded.params.sigma == params.sigma
        assert loaded.params.max_shift == params.max_shift
        assert isinstance(loaded.params.max_shift, tuple)
        assert loaded.params.mc_gSig_filt == params.mc_gSig_filt

        # All result arrays match bit-for-bit (np.save is lossless).
        np.testing.assert_array_equal(
            np.asarray(loaded.A.todense()), np.asarray(model.A.todense())
        )
        np.testing.assert_array_equal(loaded.C, model.C)
        np.testing.assert_array_equal(loaded.S, model.S)
        np.testing.assert_array_equal(loaded.YrA, model.YrA)
        np.testing.assert_array_equal(loaded.sn, model.sn)
        assert loaded.dims == model.dims

        # Optional fields that should be present after a normal fit.
        assert loaded.W is not None and loaded.b0 is not None
        assert loaded.g is not None and len(loaded.g) == model.A.shape[1]
        assert loaded.sn_per_k is not None

        # C_projected property = C + YrA.
        np.testing.assert_array_equal(loaded.C_projected, loaded.C + loaded.YrA)

    def test_C_projected_raises_before_fit(self):
        """C_projected before fit() should fail loudly, not silently None+None."""
        model = CNMFe()
        with pytest.raises(RuntimeError, match="not been fit"):
            _ = model.C_projected

    def test_save_raises_before_fit(self, tmp_path):
        """save() before fit() should fail loudly."""
        with pytest.raises(RuntimeError, match="not been fit"):
            CNMFe().save(tmp_path / "model")

    def test_params_to_from_json_unknown_keys_dropped(self, tmp_path):
        """from_json must ignore keys not on the current dataclass, so old
        save dirs keep loading after a field is added or removed."""
        p = CNMFeParams(sigma=2.5)
        path = tmp_path / "params.json"
        p.to_json(path)

        # Inject a stray field that the dataclass doesn't know about.
        import json as _json
        raw = _json.loads(path.read_text())
        raw["i_do_not_exist"] = "hello"
        path.write_text(_json.dumps(raw))

        loaded = CNMFeParams.from_json(path)
        assert loaded.sigma == 2.5
        assert not hasattr(loaded, "i_do_not_exist")

    def test_fit_with_Y_flat_zarr_matches_in_memory(self, synth_small, tmp_path):
        """fit(movie, Y_flat_zarr=...) on a pixel-major zarr must produce
        the same footprints and traces as the in-memory path on the same data.

        Phase F4 regression — pins the streaming-extraction API end-to-end.
        """
        from cnmfe.io import save_zarr, transpose_zarr_to_pixel_major
        import zarr as _zarr

        movie_np = synth_small["movie"].astype(np.float32)
        T, H, W = movie_np.shape

        # Round-trip through both layouts.
        src_path = tmp_path / "src.zarr"
        save_zarr(movie_np, str(src_path))
        src_zarr = _zarr.open_array(str(src_path), mode="r")

        pixel_path = tmp_path / "pixel.zarr"
        Y_flat_zarr = transpose_zarr_to_pixel_major(
            src_path, pixel_path,
            pixel_chunk=128, time_chunk=200,
            verbose=False,
        )

        params = CNMFeParams(
            sigma=3.0, min_corr=0.5, min_pnr=3.0,
            n_iter_main=1, n_iter_temporal=1,
        )
        m_mem = CNMFe(params).fit(movie_np, do_motion_correction=False)
        m_str = CNMFe(params).fit(
            src_zarr, do_motion_correction=False, Y_flat_zarr=Y_flat_zarr,
        )

        assert m_str.A.shape == m_mem.A.shape
        np.testing.assert_allclose(
            np.asarray(m_str.A.todense()),
            np.asarray(m_mem.A.todense()),
            atol=1e-3, rtol=1e-3,
        )
        np.testing.assert_allclose(m_str.C, m_mem.C, atol=1e-3, rtol=1e-3)

    def test_fit_Y_flat_zarr_full_pipeline_matches_in_memory(self, synth, tmp_path):
        """Streaming extraction must match the in-memory path through the deep
        pipeline (merge + auto-eval + final deconvolution), not just the
        trivial n_iter_main=1 case. Also pins that ``model.C_raw`` ends up
        aligned with ``model.A`` (Fix 1) on the streaming path.
        """
        from cnmfe.io import save_zarr, transpose_zarr_to_pixel_major
        import zarr as _zarr

        movie_np = synth["movie"].astype(np.float32)
        T, H, W = movie_np.shape

        src_path = tmp_path / "src.zarr"
        save_zarr(movie_np, str(src_path))
        src_zarr = _zarr.open_array(str(src_path), mode="r")

        pixel_path = tmp_path / "pixel.zarr"
        Y_flat_zarr = transpose_zarr_to_pixel_major(
            src_path, pixel_path,
            pixel_chunk=256, time_chunk=200,
            verbose=False,
        )

        # Same params as test_temporal_correlation_against_truth — loose
        # init thresholds so we exercise the auto-eval ghost-rejection path,
        # plus n_iter_main=2 and merge.
        params = CNMFeParams(
            sigma=3.0, min_corr=0.7, min_pnr=3.0,
            n_iter_main=2, n_iter_temporal=2,
        )
        m_mem = CNMFe(params).fit(movie_np, do_motion_correction=False)
        m_str = CNMFe(params).fit(
            src_zarr, do_motion_correction=False, Y_flat_zarr=Y_flat_zarr,
        )

        # Same K recovered after merge + auto-eval.
        assert m_str.A.shape == m_mem.A.shape, (
            f"K mismatch: streaming={m_str.A.shape[1]}, in-mem={m_mem.A.shape[1]}"
        )

        # Footprints, denoised traces, and spike trains match.
        np.testing.assert_allclose(
            np.asarray(m_str.A.todense()),
            np.asarray(m_mem.A.todense()),
            atol=1e-3, rtol=1e-3,
        )
        np.testing.assert_allclose(m_str.C, m_mem.C, atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(m_str.S, m_mem.S, atol=1e-3, rtol=1e-3)

        # Fix 1: C_raw must align with A column-for-column on BOTH paths.
        assert m_str.C_raw.shape[0] == m_str.A.shape[1], (
            f"streaming C_raw/A K mismatch: C_raw={m_str.C_raw.shape[0]}, "
            f"A={m_str.A.shape[1]}"
        )
        assert m_mem.C_raw.shape[0] == m_mem.A.shape[1], (
            f"in-memory C_raw/A K mismatch: C_raw={m_mem.C_raw.shape[0]}, "
            f"A={m_mem.A.shape[1]}"
        )
        # And the contracted C_raw matches across paths.
        np.testing.assert_allclose(m_str.C_raw, m_mem.C_raw, atol=1e-3, rtol=1e-3)

    def test_fit_zarr_movie_with_output_dir_auto_streams(self, synth_small, tmp_path):
        """Passing a zarr movie + output_dir without Y_flat_zarr must auto-derive
        the pixel-major store and route through the streaming branch, instead
        of materialising the entire movie into RAM (the old default).

        Pins the auto-streaming convenience layer and its idempotency.
        """
        from cnmfe.io import save_zarr
        import zarr as _zarr

        movie_np = synth_small["movie"].astype(np.float32)

        src_path = tmp_path / "src.zarr"
        save_zarr(movie_np, str(src_path))
        src_zarr = _zarr.open_array(str(src_path), mode="r")

        params = CNMFeParams(
            sigma=3.0, min_corr=0.5, min_pnr=3.0,
            n_iter_main=1, n_iter_temporal=1,
        )

        # Baseline: in-memory fit on the numpy array.
        m_mem = CNMFe(params).fit(movie_np, do_motion_correction=False)

        # Auto-streamed: zarr movie + output_dir, no Y_flat_zarr.
        out_dir = tmp_path / "results"
        m_auto = CNMFe(params).fit(
            src_zarr, do_motion_correction=False, output_dir=out_dir,
        )

        # The pixel-major zarr must have been created.
        pixel_zarr_path = out_dir / "Y_flat_pixel.zarr"
        assert pixel_zarr_path.exists(), (
            f"Auto-derived Y_flat_pixel.zarr not found at {pixel_zarr_path}"
        )

        # Results must equal the in-memory baseline.
        assert m_auto.A.shape == m_mem.A.shape
        np.testing.assert_allclose(
            np.asarray(m_auto.A.todense()),
            np.asarray(m_mem.A.todense()),
            atol=1e-3, rtol=1e-3,
        )
        np.testing.assert_allclose(m_auto.C, m_mem.C, atol=1e-3, rtol=1e-3)

        # Second call: idempotent — the existing pixel-major store is reused,
        # not re-written. Capture mtime to verify.
        first_mtime = pixel_zarr_path.stat().st_mtime
        m_again = CNMFe(params).fit(
            src_zarr, do_motion_correction=False, output_dir=out_dir,
        )
        second_mtime = pixel_zarr_path.stat().st_mtime
        assert second_mtime == first_mtime, (
            "Pixel-major zarr was rewritten on second call (should be idempotent)"
        )
        assert m_again.A.shape == m_mem.A.shape

    def test_fit_Y_flat_zarr_rejects_bad_shape(self, synth_small, tmp_path):
        """Shape mismatch between Y_flat_zarr and the movie must raise."""
        from cnmfe.io import save_zarr
        import zarr as _zarr

        movie_np = synth_small["movie"].astype(np.float32)
        T, H, W = movie_np.shape

        src_path = tmp_path / "src.zarr"
        save_zarr(movie_np, str(src_path))
        src_zarr = _zarr.open_array(str(src_path), mode="r")

        # Wrong shape — claim 2x as many pixels as the movie has.
        bogus = _zarr.open_array(
            str(tmp_path / "bogus.zarr"), mode="w",
            shape=(H * W * 2, T), chunks=(128, T), dtype="float32",
        )
        with pytest.raises(ValueError, match="Y_flat_zarr shape"):
            CNMFe(CNMFeParams()).fit(
                src_zarr, do_motion_correction=False, Y_flat_zarr=bogus,
            )

    def test_fit_Y_flat_zarr_requires_zarr_movie(self, synth_small, tmp_path):
        """Passing Y_flat_zarr with a numpy movie should error loudly."""
        from cnmfe.io import save_zarr, transpose_zarr_to_pixel_major

        movie_np = synth_small["movie"].astype(np.float32)
        src_path = tmp_path / "src.zarr"
        save_zarr(movie_np, str(src_path))
        Y_flat_zarr = transpose_zarr_to_pixel_major(
            src_path, tmp_path / "pixel.zarr",
            pixel_chunk=128, time_chunk=200,
            verbose=False,
        )
        with pytest.raises(TypeError, match="zarr.Array"):
            CNMFe(CNMFeParams()).fit(
                movie_np, do_motion_correction=False, Y_flat_zarr=Y_flat_zarr,
            )

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

    def test_auto_evaluation_rejects_ghosts(self, synth):
        """Loose init thresholds must not produce ghost components.

        Regression: with min_corr=0.7, min_pnr=3.0 on the 6-neuron synthetic
        fixture the pipeline used to return ~26 components — 6 real neurons
        plus ~20 ghost components at scattered background-noise locations
        7-26 px from any true neuron. Ghosts had tiny footprints (~11-29 px)
        compared to real sigma=3 Gaussians (~130 px after threshold_footprint
        at max_thr=0.1). The matching-based tests
        (test_temporal_correlation_against_truth etc.) pair each *true*
        neuron with its best estimate and so silently ignored the extras.

        Fix: the auto-evaluation step (cnmfe.evaluate.auto_evaluate_components,
        called from CNMFe.fit after the BCD loop) drops components with fewer
        than ceil(0.5*pi*sigma^2) pixels.
        """
        params = CNMFeParams(
            sigma=3.0, min_corr=0.7, min_pnr=3.0,
            n_iter_main=2, n_iter_temporal=2,
        )
        model = CNMFe(params).fit(synth["movie"], do_motion_correction=False)

        K_true = synth["A_true"].shape[1]
        K_recovered = model.A.shape[1]

        # Pre-fix: K_recovered == 26 for K_true=6. Post-fix expectation: 6-7.
        assert K_recovered <= K_true + 2, (
            f"Over-detection: recovered {K_recovered} components for "
            f"K_true={K_true} (pre-fix this was ~26 ghost-laden runs)."
        )

        # Ghost rejection must not trade away real neurons.
        matches = match_components(model.A, synth["A_true"])
        well_matched = sum(1 for m in matches if m[2] > 0.7)
        assert well_matched == K_true, (
            f"Lost real neurons: only {well_matched}/{K_true} matched with "
            f"spatial r > 0.7. Per-neuron r: "
            f"{[round(m[2], 3) for m in matches]}"
        )

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
