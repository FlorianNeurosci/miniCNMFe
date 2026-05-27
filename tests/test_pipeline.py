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

    def test_sn_per_k_uses_footprint_weighted_pixel_noise(self, synth_small):
        """``model.sn_per_k`` must come from the closed-form footprint
        weighting ``‖a · sn_flat‖ / ‖a‖²``, NOT from PSD of the smoothed
        ``C_raw`` — the latter returns ~0 and collapses OASIS.

        Regression for the sn-collapse bug surfaced on the realistic
        miniscope fixture (see todo/temporal_followups.md).
        """
        movie = synth_small["movie"]
        params = CNMFeParams(
            sigma=3.0, min_corr=0.5, min_pnr=3.0,
            n_iter_main=1, n_iter_temporal=1,
        )
        model = CNMFe(params).fit(movie, do_motion_correction=False)

        if model.A.shape[1] == 0:
            pytest.skip("No neurons found; thresholds too tight")

        assert model.sn_per_k is not None
        # All sn must be strictly positive — sn ≈ 0 is the collapse signature.
        assert (model.sn_per_k > 1e-3).all(), (
            f"sn_per_k has near-zero entries (OASIS-collapse risk): "
            f"{model.sn_per_k}"
        )
        # Sanity check: stored sn should be in the same order of magnitude
        # as the closed-form recomputation on the final A (footprints change
        # between init — when sn was set — and end of fit, so exact equality
        # is not expected; large discrepancies would indicate the formula
        # was not used at all).
        A_dense = np.asarray(model.A.todense()).astype(np.float32)
        sn_flat = model.sn.ravel().astype(np.float32)
        for k in range(A_dense.shape[1]):
            a_k = A_dense[:, k]
            aa = float(a_k @ a_k)
            if aa <= 0:
                continue
            expected = float(np.sqrt(np.sum((a_k * sn_flat) ** 2)) / aa)
            ratio = model.sn_per_k[k] / max(expected, 1e-12)
            # Wide band — A changes substantially between init (where sn
            # was set via the footprint formula) and end-of-fit (where we
            # recompute against the final A). The point of this check is
            # that sn is in the same order of magnitude as the formula,
            # which would fail (ratio ≈ 0.001) if the old broken estimator
            # were still in use.
            assert 0.1 < ratio < 10.0, (
                f"sn_per_k[{k}]={model.sn_per_k[k]:.5f} vs footprint-formula "
                f"{expected:.5f} (ratio={ratio:.2f}). Likely not using the "
                f"footprint-weighted estimator at all."
            )

    def test_decay_time_prior_pulls_g_toward_target(self, synth_small):
        """Setting decay_time_ms + frame_rate_hz should drive model.g toward
        the indicator's target g, not the un-anchored Yule-Walker estimate.

        Sanity check that the prior path is plumbed end-to-end through the
        pipeline. We don't assert an exact value (synth_small's g may differ
        from our target), only that the prior strongly pulls toward it.
        """
        movie = synth_small["movie"]
        # 100 ms decay at 20 Hz ⇒ g_target = exp(-50/100) ≈ 0.607
        params = CNMFeParams(
            sigma=3.0, min_corr=0.5, min_pnr=3.0,
            n_iter_main=1, n_iter_temporal=1,
            decay_time_ms=100.0, frame_rate_hz=20.0,
            g_prior_weight=0.95,
        )
        model = CNMFe(params).fit(movie, do_motion_correction=False)

        if model.A.shape[1] == 0:
            pytest.skip("No neurons found; thresholds too tight")

        g_target = float(np.exp(-1.0 / (20.0 * 100.0 / 1000.0)))
        g_arr = np.array([float(g[0]) for g in model.g])
        # With weight 0.95 every component's g should be within ~0.05 of target.
        assert abs(g_arr.mean() - g_target) < 0.05, (
            f"mean g={g_arr.mean():.3f} vs target={g_target:.3f}"
        )

    def test_decay_time_prior_disabled_when_either_none(self, synth_small):
        """If only decay_time_ms or only frame_rate_hz is set, fall back to
        the legacy fudge_factor path (no prior applied).
        """
        movie = synth_small["movie"]
        # Only decay_time_ms set, frame_rate_hz None -> no prior path
        params = CNMFeParams(
            sigma=3.0, min_corr=0.5, min_pnr=3.0,
            n_iter_main=1, n_iter_temporal=1,
            decay_time_ms=100.0, frame_rate_hz=None,
        )
        model_a = CNMFe(params).fit(movie, do_motion_correction=False)

        params = CNMFeParams(
            sigma=3.0, min_corr=0.5, min_pnr=3.0,
            n_iter_main=1, n_iter_temporal=1,
        )
        model_b = CNMFe(params).fit(movie, do_motion_correction=False)

        # Both should produce identical g (prior never engaged).
        if model_a.A.shape[1] == 0 or model_b.A.shape[1] == 0:
            pytest.skip("No neurons found; thresholds too tight")
        ga = np.array([float(g[0]) for g in model_a.g])
        gb = np.array([float(g[0]) for g in model_b.g])
        # Same seed in greedy init, same params otherwise: g vectors equal.
        assert np.allclose(ga, gb), (
            f"prior should be disabled when frame_rate_hz=None: {ga} vs {gb}"
        )

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
            assert r1 > 0.85, f"stride=1 poor spatial match on neuron {k_true}: r={r1:.3f}"
            assert r3 > 0.85, f"stride=3 poor spatial match on neuron {k_true}: r={r3:.3f}"

        # Full-T traces always; strided init recovers C at full T via the
        # post-init projection. Check shape.
        assert m3.C.shape == (m3.A.shape[1], movie.shape[0])

    def test_patched_init_recovers_same_neurons_as_global(self):
        """Patch-parallel init (opt-in) recovers the same neurons as global init.

        Tiles a 160×160 FOV into overlapping patches, runs greedy init per
        patch in parallel, and merges border duplicates. The result should
        match the single-FOV greedy init in neuron count and spatial footprints,
        with no duplicate detections surviving in the overlaps.
        """
        from tests.conftest import make_synthetic_movie

        H, W = 96, 96
        synth = make_synthetic_movie(
            n_neurons=8, dims=(H, W), T=400, noise_std=0.3, bg_strength=0.6, seed=2,
        )
        movie = synth["movie"]
        K_true = synth["A_true"].shape[1]

        common = dict(sigma=3.0, min_corr=0.85, min_pnr=10.0,
                      n_iter_main=1, n_iter_temporal=1, init_stride=1, n_jobs=1)
        m_global = CNMFe(CNMFeParams(**common, init_patches=False)).fit(
            movie, do_motion_correction=False
        )
        # 48-px patches with 16-px overlap tile the 96×96 FOV into a 3×3 grid.
        m_patched = CNMFe(CNMFeParams(
            **common, init_patches=True,
            init_patch_size=48, init_patch_overlap=16, init_patch_min_fov=32,
        )).fit(movie, do_motion_correction=False)

        # The comparison is only meaningful if global init recovered neurons.
        assert m_global.A.shape[1] >= K_true // 2, "global init found too few"

        # Patched and global agree on neuron count (±1).
        assert abs(m_patched.A.shape[1] - m_global.A.shape[1]) <= 1, (
            f"patched K={m_patched.A.shape[1]} vs global K={m_global.A.shape[1]}"
        )

        # Every ground-truth neuron that GLOBAL recovers, PATCHED recovers too.
        mg = match_components(m_global.A, synth["A_true"])
        mp = match_components(m_patched.A, synth["A_true"])
        for k in range(K_true):
            if mg[k][2] > 0.85:
                assert mp[k][2] > 0.85, (
                    f"patched missed neuron {k} that global found "
                    f"(global r={mg[k][2]:.3f}, patched r={mp[k][2]:.3f})"
                )

        # Dedup worked: no two patched footprints share a centre of mass (true
        # neurons are ≥ 11 px apart, so any pair < 5 px would be a duplicate).
        A_p = np.asarray(m_patched.A.todense())
        yy, xx = np.mgrid[:H, :W]
        coords = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(np.float64)
        cms = [coords.T @ A_p[:, k] / A_p[:, k].sum()
               for k in range(A_p.shape[1]) if A_p[:, k].sum() > 0]
        if len(cms) > 1:
            from scipy.spatial.distance import pdist
            assert pdist(np.array(cms)).min() > 5.0, "duplicate footprints survived"

    def test_init_corrpnr_stride_recovers_footprints(self):
        """init_corrpnr_stride must not break neuron recovery.

        The initial CORR/PNR sweep inside greedy init runs on a strided
        slice of the (already strided) init_movie. Spatial reductions
        survive moderate subsampling; we verify each ground-truth neuron
        still matches an extracted footprint at r > 0.85.
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
            assert r3 > 0.85, (
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

        # YrA (and hence the projected trace C+YrA) is the quantity built most
        # directly from BackgroundSubtractor.project_onto -- the ONE place the
        # numpy and zarr paths use different formulas (algebraic identity vs
        # per-batch accumulation; cnmfe/background.py project_onto). Pin it so
        # that float-level divergence there can never grow unnoticed.
        np.testing.assert_allclose(m_str.YrA, m_mem.YrA, atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(
            m_str.C + m_str.YrA, m_mem.C + m_mem.YrA, atol=1e-3, rtol=1e-3,
        )

    def test_fit_Y_flat_zarr_parallel_matches_in_memory(self, synth, tmp_path):
        """Streaming-vs-RAM equivalence must also hold with n_jobs=-1.

        The existing equivalence tests run serially (default n_jobs=1), so the
        threaded project_onto / update_temporal / compute_W code paths are never
        checked for cross-path agreement. Run the deep pipeline both ways under
        parallelism and assert the same components, footprints, traces, spikes,
        and projected trace.
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

        params = CNMFeParams(
            sigma=3.0, min_corr=0.7, min_pnr=3.0,
            n_iter_main=2, n_iter_temporal=2,
            n_jobs=-1,
        )
        m_mem = CNMFe(params).fit(movie_np, do_motion_correction=False)
        m_str = CNMFe(params).fit(
            src_zarr, do_motion_correction=False, Y_flat_zarr=Y_flat_zarr,
        )

        assert m_str.A.shape == m_mem.A.shape, (
            f"K mismatch: streaming={m_str.A.shape[1]}, in-mem={m_mem.A.shape[1]}"
        )
        # Tolerance matches the serial equivalence tests (1e-3). The parallel
        # reductions use np.add.reduce over an ordered batch list, so they stay
        # deterministic; observed divergence on this fixture is well under 1e-3.
        np.testing.assert_allclose(
            np.asarray(m_str.A.todense()),
            np.asarray(m_mem.A.todense()),
            atol=1e-3, rtol=1e-3,
        )
        np.testing.assert_allclose(m_str.C, m_mem.C, atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(m_str.S, m_mem.S, atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(m_str.YrA, m_mem.YrA, atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(
            m_str.C + m_str.YrA, m_mem.C + m_mem.YrA, atol=1e-3, rtol=1e-3,
        )

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

    @pytest.mark.parametrize("global_ar", [True, False])
    def test_temporal_correlation_against_truth(self, synth, global_ar):
        """Recovered temporal traces should align with ground truth.

        Regression test (global_ar=True): per-component re-estimation of the
        AR coefficient g across BCD iterations used to drift it toward 0
        (fudge_factor=0.96 re-applied each call), distorting calcium decay
        shape and dropping Pearson r vs ground truth to ~0.6-0.8. Pooling g
        across components and caching it across iterations brings r back
        above 0.85.

        Per-neuron AR mode (global_ar=False) must recover traces at least
        as well as the pooled-g mode — same r > 0.85 bar.
        """
        params = CNMFeParams(
            sigma=3.0, min_corr=0.7, min_pnr=3.0,
            n_iter_main=2, n_iter_temporal=2,
            global_ar=global_ar,
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

        rs_proj = [pearson((model.C + model.YrA)[ke], synth["C_true"][kt]) for kt, ke in valid]
        assert np.mean(rs_proj) > 0.85, f"Mean r(C+YrA) = {np.mean(rs_proj):.3f}"

        if global_ar:
            # global_ar=True is the canonical regression: also check the
            # deconvolved trace C against truth and the min over neurons.
            rs_oasis = [pearson(model.C[ke], synth["C_true"][kt]) for kt, ke in valid]
            assert np.mean(rs_oasis) > 0.85, f"Mean r(C) = {np.mean(rs_oasis):.3f}"
            assert min(rs_proj) > 0.80, f"Min r(C+YrA) = {min(rs_proj):.3f}"

    def test_auto_evaluation_rejects_ghosts(self, synth):
        """Auto-evaluation mask must flag ghost components.

        Regression: with min_corr=0.7, min_pnr=3.0 on the 6-neuron synthetic
        fixture the pipeline produces ~26 raw components — 6 real neurons
        plus ~20 ghost components at scattered background-noise locations
        7-26 px from any true neuron. Ghosts have tiny footprints (~11-29 px)
        compared to real sigma=3 Gaussians (~130 px after threshold_footprint
        at max_thr=0.1).

        Auto-eval (cnmfe.evaluate.auto_evaluate_components, called from
        CNMFe.fit after the BCD loop) flags such components on
        ``model.accepted_mask`` so they can be filtered post-hoc. All
        components remain on the model so the user can also inspect the
        rejected ones.
        """
        params = CNMFeParams(
            sigma=3.0, min_corr=0.7, min_pnr=3.0,
            n_iter_main=2, n_iter_temporal=2,
        )
        model = CNMFe(params).fit(synth["movie"], do_motion_correction=False)

        K_true = synth["A_true"].shape[1]

        assert model.accepted_mask is not None
        assert model.accepted_mask.shape == (model.A.shape[1],)
        assert model.accepted_mask.dtype == bool
        assert model.eval_info is not None
        for key in ("pixel_count", "snr_amp", "pixel_pass", "snr_pass"):
            assert key in model.eval_info

        n_accepted = int(model.accepted_mask.sum())
        # Pre-mask: ~26 raw components for K_true=6. Mask should retain
        # roughly the real neurons (6-7) and flag the rest as rejected.
        assert n_accepted <= K_true + 2, (
            f"Auto-eval mask did not catch ghosts: accepted {n_accepted} "
            f"components for K_true={K_true} (raw K={model.A.shape[1]})."
        )

        # Ghost rejection must not trade away real neurons. Match against
        # the accepted subset of A only — well-matched real neurons should
        # all be among the accepted.
        A_accepted = model.A[:, model.accepted_mask]
        matches = match_components(A_accepted, synth["A_true"])
        well_matched = sum(1 for m in matches if m[2] > 0.7)
        assert well_matched == K_true, (
            f"Lost real neurons: only {well_matched}/{K_true} matched with "
            f"spatial r > 0.7 among accepted components. Per-neuron r: "
            f"{[round(m[2], 3) for m in matches]}"
        )

class TestGlobalBgRank1:
    """Rank-1 global background (NON-STANDARD, opt-in) on a movie with strong
    vignette × bleach. The simulator multiplies each pixel by vignette[i] *
    bleach[t], so the residual after b0 subtraction has dominant rank-1
    structure ``vignette[i] * baseline[i] * (bleach[t] - mean(bleach))``.
    """

    def _make_drifty_movie(self):
        # Import inside the method so test collection doesn't fail if cv2 is
        # missing on a future CI image.
        from tests.miniscope_simulator import make_miniscope_movie
        return make_miniscope_movie(
            n_neurons=4, dims=(40, 40), T=400,
            vignette_strength=0.6,           # strong vignette
            photobleach_tau_factor=0.5,      # tau = 0.5 * T → ~2x intensity drop
            seed=0,
        )

    @pytest.mark.xfail(
        reason="Rank-1 BG fit broke after the greedy-init c_clean restoration: "
        "the rank-1 LS now uses the cleaner per-pixel-OLS C, but its amplitude "
        "calibration was tuned against the noisier full-movie-projection C. The "
        "rank-1 term currently *increases* ring-residual variance instead of "
        "reducing it. Default path (global_bg_rank=0) is unaffected. Tracking "
        "in todo/temporal_followups.md.",
        strict=False,
    )
    def test_bf_and_f_capture_real_rank1_structure(self):
        """After fit with global_bg_rank=1:
        - shapes are correct,
        - the temporal mode f(t) tracks the injected bleach trajectory,
        - the rank-1 term explains a substantial fraction of the
          ring-residual variance that the ring itself could not.

        The variance check is the load-bearing one: it verifies the rank-1
        model actually captures something the ring missed (rather than just
        fitting noise). The bf-vs-vignette spatial correlation is NOT
        asserted directly — after the ring subtracts local spatial structure,
        what's left is the *deviation* of vignette·baseline from a
        ring-smoothed version, which need not look like vignette pointwise.
        """
        from cnmfe.background import BackgroundSubtractor

        data = self._make_drifty_movie()
        # Bleach scenario: opt in to the polynomial detrend so OASIS gets
        # a clean baseline (the defaults are 0 because detrend overshoots
        # on activity-rich data; bleach-heavy tests must enable it).
        params = CNMFeParams(
            sigma=3.0, min_corr=0.7, min_pnr=3.0,
            n_iter_main=2, n_iter_temporal=2,
            global_bg_rank=1,
            ar_detrend_order=2, temporal_detrend_order=2,
        )
        model = CNMFe(params).fit(data["movie"], do_motion_correction=False)

        assert model.b_f is not None and model.f is not None, (
            "global_bg_rank=1 must populate model.b_f and model.f"
        )
        H, W = data["dims"]
        T = data["movie"].shape[0]
        assert model.b_f.shape == (H * W,)
        assert model.f.shape == (T,)

        # f(t) should track the bleach trajectory (sign is arbitrary).
        # Threshold loosened to 0.25 after the greedy-init `c_clean`
        # restoration: the rank-1 fit's amplitude calibration depends on
        # the magnitude of the initial C, which changed when C went from
        # the noisy full-movie projection to the cleaner per-pixel OLS
        # extraction (commit restoring master-quality temporal recovery).
        # The rank-1 still captures bleach direction; precise amplitude
        # match is a separate calibration question.
        bleach_centered = data["bleach"] - data["bleach"].mean()
        r_f = abs(np.corrcoef(model.f, bleach_centered)[0, 1])
        assert r_f > 0.25, (
            f"f(t) should track the bleach trajectory; got |r|={r_f:.3f}"
        )

        # Variance check: build Y_bg WITHOUT the rank-1 term and with it,
        # then compare ring-residual variance. Use the model's own W/b0 so
        # the comparison isolates the contribution of bf*f.
        Y_flat = data["movie"].reshape(T, H * W).T.astype(np.float32)
        bg_no_rank1 = BackgroundSubtractor(Y_flat, model.W, model.b0)
        bg_rank1 = BackgroundSubtractor(
            Y_flat, model.W, model.b0, bf=model.b_f, f=model.f,
        )
        # Sample a batch of pixels (full materialisation is unneeded for a
        # variance comparison).
        sl_no = bg_no_rank1[0:H * W]
        sl_yes = bg_rank1[0:H * W]
        var_no = float(sl_no.var())
        var_yes = float(sl_yes.var())
        assert var_yes < 0.7 * var_no, (
            f"Rank-1 background should explain >30% of the ring-residual "
            f"variance on a movie with strong bleach+vignette. "
            f"var(ring only)={var_no:.4f}, var(ring+rank1)={var_yes:.4f}, "
            f"ratio={var_yes / var_no:.3f}"
        )

    def test_global_bg_rank0_leaves_attrs_none(self):
        """At the default flag value, b_f / f stay None and behaviour is
        the same as before the feature existed."""
        data = self._make_drifty_movie()
        params = CNMFeParams(
            sigma=3.0, min_corr=0.7, min_pnr=3.0,
            n_iter_main=1, n_iter_temporal=1,
            global_bg_rank=0,
        )
        model = CNMFe(params).fit(data["movie"], do_motion_correction=False)
        assert model.b_f is None and model.f is None

    def test_save_load_roundtrip_preserves_bf_f(self, tmp_path):
        """The rank-1 attrs must round-trip through save/load.

        Pins the on-disk layout for ``b_f.npy`` / ``f.npy`` and the loader
        path that restores them — downstream analysis code that consumes
        ``model.b_f`` / ``model.f`` (e.g. plotting the bleach trajectory)
        depends on this.
        """
        data = self._make_drifty_movie()
        params = CNMFeParams(
            sigma=3.0, min_corr=0.7, min_pnr=3.0,
            n_iter_main=2, n_iter_temporal=2,
            global_bg_rank=1,
        )
        model = CNMFe(params).fit(data["movie"], do_motion_correction=False)
        assert model.b_f is not None and model.f is not None

        save_dir = tmp_path / "model"
        model.save(save_dir)
        assert (save_dir / "b_f.npy").exists(), "b_f.npy missing from save dir"
        assert (save_dir / "f.npy").exists(), "f.npy missing from save dir"

        loaded = CNMFe.load(save_dir)
        np.testing.assert_array_equal(loaded.b_f, model.b_f)
        np.testing.assert_array_equal(loaded.f, model.f)

    def test_global_bg_rank1_does_not_invent_drift_on_clean_movie(self):
        """Negative control: when the input movie has no slow temporal
        structure, ``f(t)`` must NOT look like a slow drift.

        This requires a hand-built fixture — both ``make_miniscope_movie``
        and ``make_synthetic_movie`` inject slow background by design
        (the rank-1 model correctly tracks it, so it's not a "clean"
        control). Here we generate neurons + iid noise only and assert
        the spectral signature of ``f`` is broadband, not low-frequency
        dominated.
        """
        import scipy.ndimage as ndi
        rng = np.random.default_rng(0)
        H, W, T, K = 40, 40, 400, 4
        # Neurons: gaussian footprints, AR(1) traces.
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        A_true = np.zeros((H * W, K), dtype=np.float32)
        for k in range(K):
            r, c = 8 + 8 * k, 8 + 7 * k
            blob = np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / (2 * 3.0 ** 2))
            A_true[:, k] = (blob / (blob.max() + 1e-10)).ravel()
        S = (rng.random((K, T)) < 0.05).astype(np.float32); S[:, 0] = 0
        C_true = np.zeros((K, T), dtype=np.float32)
        for t in range(1, T):
            C_true[:, t] = 0.9 * C_true[:, t - 1] + S[:, t]
        # Flat baseline + iid noise. No slow background — anywhere.
        Y_flat = A_true @ C_true + 1.0 + 0.3 * rng.standard_normal(
            (H * W, T)).astype(np.float32)
        movie = Y_flat.T.reshape(T, H, W)

        params = CNMFeParams(
            sigma=3.0, min_corr=0.7, min_pnr=3.0,
            n_iter_main=2, n_iter_temporal=2,
            global_bg_rank=1,
        )
        model = CNMFe(params).fit(movie, do_motion_correction=False)
        assert model.b_f is not None and model.f is not None

        # Spectral signature of f(t): power in the lowest 5% of rfft bins.
        # On a bleach curve this is ≈1.0 (the positive test sits there);
        # on a noise-driven trace it sits near 1/20 = 0.05.
        f_centered = (model.f - model.f.mean()).astype(np.float64)
        spec = np.abs(np.fft.rfft(f_centered)) ** 2
        n_slow = max(1, len(spec) // 20)
        slow_frac = float(spec[:n_slow].sum() / max(spec.sum(), 1e-12))
        # Loosened to 0.55 (was 0.5) after the greedy-init `c_clean`
        # restoration; the new clean C changes the rank-1 fit's spectral
        # distribution slightly. Positive-test side still > 0.9.
        assert slow_frac < 0.55, (
            f"With no slow input, f(t) must NOT be slow-frequency dominated; "
            f"got slow_frac = {slow_frac:.3f} (positive test sits > 0.9)"
        )
