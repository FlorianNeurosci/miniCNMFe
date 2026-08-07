"""Tests for the parameter-tuning workflow (``tuning/`` + ``tune.py``).

Fast, end-to-end-ish: a small miniscope-simulator movie drives the heuristics,
a tiny on-disk zarr drives the sweep, and a 2-file AVI folder drives the full
``run_tuning`` -> report-folder path. OASIS is not installed in the project env,
so every test here exercises the pure-Python AR(1) fallback (covers the
no-oasis requirement).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minicnmfe.pipeline import CNMFe, CNMFeParams  # noqa: E402
from tests.miniscope_simulator import make_miniscope_movie  # noqa: E402
from tuning import heuristics as H  # noqa: E402
from tuning import io_sample as S  # noqa: E402
from tuning import metrics as M  # noqa: E402
from tuning.sweep import SweepSpec, build_candidates, run_sweep  # noqa: E402

cv2 = pytest.importorskip("cv2")


@pytest.fixture(scope="module")
def sim_movie():
    out = make_miniscope_movie(n_neurons=12, dims=(96, 96), T=300, seed=0)
    return out["movie"].astype(np.float32)


@pytest.fixture(scope="module")
def sim_zarr(sim_movie, tmp_path_factory):
    import zarr

    path = tmp_path_factory.mktemp("z") / "mc.zarr"
    z = zarr.open(
        str(path), mode="w", shape=sim_movie.shape,
        chunks=(50, sim_movie.shape[1], sim_movie.shape[2]), dtype="float32",
    )
    z[:] = sim_movie
    return path


@pytest.fixture(scope="module")
def fitted_model(sim_movie):
    params = CNMFeParams(sigma=3.0, min_corr=0.7, min_pnr=6.0, n_iter_main=1,
                         frame_rate_hz=20.0, decay_time_ms=180.0)
    return CNMFe(params).fit_extract(sim_movie, evaluate=True)


def _write_avi(path, frames):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    H_, W_ = frames.shape[1], frames.shape[2]
    writer = cv2.VideoWriter(str(path), fourcc, 20.0, (W_, H_), isColor=True)
    assert writer.isOpened()
    try:
        u8 = np.clip(frames, 0, None)
        u8 = (255 * u8 / (u8.max() + 1e-8)).astype(np.uint8)
        for fr in u8:
            writer.write(cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR))
    finally:
        writer.release()


@pytest.fixture(scope="module")
def avi_folder(sim_movie, tmp_path_factory):
    folder = tmp_path_factory.mktemp("avis")
    half = sim_movie.shape[0] // 2
    _write_avi(folder / "0.avi", sim_movie[:half])
    _write_avi(folder / "1.avi", sim_movie[half:])
    return folder


# --------------------------------------------------------------------------


def test_heuristics_on_simulator(sim_movie):
    sample = sim_movie
    mc_gsig, sigma_native, ev = H.suggest_mc_gsig_and_sigma(sample)
    assert 1.0 <= sigma_native <= 15.0 and mc_gsig >= 1
    assert "std_img" in ev

    median_img = np.median(sample, axis=0)
    max_shift, border_px, ev = H.suggest_max_shift(sample, median_img, mc_gsig,
                                                   n_shift_frames=40)
    assert len(max_shift) == 2 and all(int(v) >= 2 for v in max_shift)
    assert border_px == max(max_shift)

    ssub, tsub, ev = H.suggest_downsample(sigma_native, 20.0, 180.0)
    assert ssub >= 1 and tsub >= 1 and ev["ssub_rows"]

    sigma_refit, cn, pnr, ev = H.suggest_sigma_extraction(sample, sigma_native)
    assert np.isfinite(sigma_refit) and cn.shape == sim_movie.shape[1:]

    min_corr, min_pnr, ev = H.suggest_corr_pnr(cn, pnr, sigma_refit)
    assert 0.0 < min_corr < 1.0 and min_pnr > 0.0
    assert {"ncell_corr", "ncell_pnr", "corr_axis", "pnr_axis"} <= set(ev)

    min_pixel, ev = H.suggest_min_pixel(sample, sigma_refit, min_corr, min_pnr,
                                        sim_movie.shape[1:])
    assert min_pixel >= 1


def test_sigma_heuristic_robust_to_background_haze():
    """The neuron-radius estimate must not be inflated by broad out-of-focus
    haze. A hazy FOV = small neurons riding on large diffuse background bumps;
    without the spatial high-pass blob_log latches onto the bumps and the median
    radius blows up (the real-recording sprawl bug). The high-pass must recover
    the neuron scale while a clean FOV stays unchanged.
    """
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(0)
    nh = nw = 200
    # A few small neurons (radius ~3 px) ...
    neurons = np.zeros((nh, nw), np.float32)
    for _ in range(10):
        y, x = rng.integers(20, nh - 20), rng.integers(20, nw - 20)
        neurons[y, x] = 1.0
    neurons = gaussian_filter(neurons, 3.0)
    neurons /= neurons.max()
    # ... riding on many broad, detectable-scale background bumps (radius ~12 px):
    # the out-of-focus-haze signature that dominates the blob set and inflates
    # the radius estimate when it isn't removed.
    haze = np.zeros((nh, nw), np.float32)
    for _ in range(40):
        y, x = rng.integers(15, nh - 15), rng.integers(15, nw - 15)
        haze[y, x] = 1.0
    haze = gaussian_filter(haze, 12.0)
    haze = 1.2 * haze / haze.max()

    # sample.std(axis=0) reproduces the target image exactly for [+img, -img].
    def sample_of(img):
        return np.stack([img, -img]).astype(np.float32)

    hazy = neurons + haze
    _g_on, sig_on, _ = H.suggest_mc_gsig_and_sigma(sample_of(hazy))           # default high-pass
    _g_off, sig_off, _ = H.suggest_mc_gsig_and_sigma(sample_of(hazy),
                                                     highpass_sigma=0)         # disabled
    # High-pass recovers the neuron scale; without it the haze inflates it badly.
    assert sig_on <= 5.0 < sig_off

    # A clean FOV (no haze) is left essentially unchanged by the high-pass.
    _g_c_on, sig_clean_on, _ = H.suggest_mc_gsig_and_sigma(sample_of(neurons))
    _g_c_off, sig_clean_off, _ = H.suggest_mc_gsig_and_sigma(sample_of(neurons),
                                                             highpass_sigma=0)
    assert abs(sig_clean_on - sig_clean_off) <= 1.5


def test_metrics_on_fitted_model(fitted_model):
    q = M.model_quality(fitted_model)
    for key in ("K", "K_accepted", "accepted_frac", "cprojcorr_mean",
                "cprojcorr_median", "npix_median", "npix_iqr", "multipeak_frac",
                "npix_oversize", "snr_mean", "snr_median"):
        assert key in q
    assert 0.0 <= q["accepted_frac"] <= 1.0
    if q["K"] > 0:
        assert -1.0 <= q["cprojcorr_median"] <= 1.0
    assert np.isfinite(M.composite_score(q))


def test_blob_coverage_metric(fitted_model, sim_movie):
    """blob_coverage returns the expected keys with recall/precision in [0, 1]
    and detects cell blobs on the simulator's CORR·PNR image."""
    from minicnmfe.preprocess import correlation_pnr

    cn, pnr = correlation_pnr(sim_movie, sigma=3.0)
    cov = M.blob_coverage(fitted_model, cn, pnr, sigma=3.0,
                          min_corr=0.7, min_pnr=6.0)
    for key in ("n_blobs", "n_blobs_covered", "blob_recall", "n_footprints",
                "n_footprints_on_blob", "footprint_precision", "coverage_radius"):
        assert key in cov, key
    if cov["n_blobs"]:
        assert 0.0 <= cov["blob_recall"] <= 1.0
    if cov["n_footprints"]:
        assert 0.0 <= cov["footprint_precision"] <= 1.0
    assert cov["n_footprints"] > 0 and cov["n_blobs"] > 0


def test_blob_coverage_empty_model():
    """K==0 model: NaN recall/precision, no crash."""
    empty = CNMFe(CNMFeParams(sigma=3.0))
    cn = np.zeros((32, 32), np.float32)
    pnr = np.zeros((32, 32), np.float32)
    cov = M.blob_coverage(empty, cn, pnr, sigma=3.0, min_corr=0.7, min_pnr=6.0)
    assert cov["n_footprints"] == 0
    assert np.isnan(cov["footprint_precision"])


def test_session_quality_verdict():
    """Each sub-check trips on the right metric; a clean set is PASS."""
    clean_q = {"cprojcorr_median": 0.9, "trace_corr_median": 0.1}
    clean_cov = {"blob_recall": 0.95, "footprint_precision": 0.95}
    v = M.session_quality_verdict(clean_q, clean_cov)
    assert v["status"] == "PASS" and not v["warnings"]

    bad = M.session_quality_verdict(
        {"cprojcorr_median": 0.2, "trace_corr_median": 0.8},
        {"blob_recall": 0.3, "footprint_precision": 0.4})
    assert bad["status"] == "WARN"
    assert bad["checks"] == {"blob_recall": False, "footprint_precision": False,
                             "cprojcorr": False, "trace_corr": False}
    assert len(bad["warnings"]) == 4

    # NaN metrics are skipped (treated as passing), not failed.
    skip = M.session_quality_verdict(
        {"cprojcorr_median": float("nan"), "trace_corr_median": float("nan")},
        {"blob_recall": float("nan"), "footprint_precision": float("nan")})
    assert skip["status"] == "PASS"


def _gauss_blob(dims, centre, sigma=3.0):
    from scipy.ndimage import gaussian_filter

    img = np.zeros(dims, np.float32)
    img[centre] = 1.0
    img = gaussian_filter(img, sigma)
    return img / (img.max() + 1e-8)


def _fake_model(footprints_2d, sigma=3.0):
    """Lightweight stand-in exposing exactly what ``model_quality`` reads."""
    import types

    import scipy.sparse as sp

    H_, W_ = footprints_2d[0].shape
    K = len(footprints_2d)
    A = sp.csc_matrix(np.stack([fp.ravel() for fp in footprints_2d], axis=1))
    rng = np.random.default_rng(0)
    C = rng.standard_normal((K, 60))  # non-constant so per-cell corr is defined
    return types.SimpleNamespace(
        A=A, C=C, YrA=np.zeros_like(C),          # YrA=0 -> cprojcorr == 1 for all
        accepted_mask=np.ones(K, bool), eval_info=None,
        dims=(H_, W_), params=CNMFeParams(sigma=sigma))


def test_multipeak_penalises_over_merge():
    """An over-merged candidate (two somata fused into each footprint) must score
    BELOW a compact one-cell-per-footprint candidate at matched purity/accepted
    fraction — the bug that picked the over-large-sigma footprints. The only
    score difference is the multi-peak penalty."""
    dims = (40, 40)
    # All peaks well inside the frame (>=10 px from any border) so peak_local_max's
    # border exclusion can't drop a genuine peak.
    centres = [(10, 10), (10, 26), (26, 10), (26, 26)]
    compact = _fake_model([_gauss_blob(dims, c) for c in centres])
    # Same cells, but each footprint fuses a second, 14-16 px separated soma.
    pairs = [((10, 10), (10, 26)), ((26, 10), (26, 26)),
             ((10, 12), (26, 12)), ((12, 26), (26, 26))]
    merged = _fake_model([_gauss_blob(dims, a) + _gauss_blob(dims, b)
                          for a, b in pairs])

    qc, qm = M.model_quality(compact), M.model_quality(merged)
    # Matched purity (corr==1) and accepted fraction; differ only on multipeak.
    assert qc["cprojcorr_median"] == pytest.approx(1.0)
    assert qm["cprojcorr_median"] == pytest.approx(1.0)
    assert qc["multipeak_frac"] == pytest.approx(0.0)
    assert qm["multipeak_frac"] >= 0.75
    assert M.composite_score(qc) > M.composite_score(qm)


def test_resolve_sigma_grid_always_includes_heuristic():
    from tuning.sweep import resolve_sigma_grid

    # None -> heuristic-centred {s-1, s, s+1}, floored at 2.
    assert resolve_sigma_grid(None, 5.0) == [4.0, 5.0, 6.0]
    assert resolve_sigma_grid(None, 2.0) == [2.0, 3.0]          # s-1 clips to 2
    # Offset DSL: around offsets + absolute extras, heuristic (offset 0) present.
    assert resolve_sigma_grid({"around": [-1, 0, 1], "extra": [9]}, 5.0) == [4.0, 5.0, 6.0, 9.0]
    # Back-compat absolute list: heuristic injected alongside the given values.
    assert resolve_sigma_grid([3, 4, 5], 2.0) == [2.0, 3.0, 4.0, 5.0]
    # Scalar + sub-floor extra dropped.
    assert resolve_sigma_grid(4, 2.0) == [2.0, 4.0]
    assert resolve_sigma_grid({"extra": [1]}, 2.0) == [2.0]


def test_resolve_offset_grid_thresholds():
    from tuning.sweep import resolve_offset_grid

    # min_pnr style: omitted -> just the detected value (no sweep).
    assert resolve_offset_grid(None, 12.0, floor=2.0) == [12.0]
    # around offsets, anchor always present.
    assert resolve_offset_grid({"around": [-3, 0, 3]}, 12.0, floor=2.0) == [9.0, 12.0, 15.0]
    # min_corr style: floor + clip_max applied to offsets.
    assert resolve_offset_grid({"around": [-0.05, 0, 0.05]}, 0.84, floor=0.3, clip_max=0.98) == \
        pytest.approx([0.79, 0.84, 0.89])
    assert resolve_offset_grid({"around": [0, 0.2]}, 0.9, floor=0.3, clip_max=0.98) == [0.9, 0.98]
    # sub-floor offset clamps to floor.
    assert resolve_offset_grid({"around": [-20, 0]}, 12.0, floor=2.0) == [2.0, 12.0]
    # back-compat absolute list injects the anchor.
    assert resolve_offset_grid([6, 10, 14], 12.0, floor=2.0) == [6.0, 10.0, 12.0, 14.0]


def test_suggest_corr_pnr_morphology():
    from scipy.ndimage import gaussian_filter

    # CORR-like image: compact cell blobs (radius ~3) on diffuse low-level haze.
    rng = np.random.default_rng(1)
    H_ = W_ = 120
    cells = np.zeros((H_, W_), np.float32)
    for _ in range(15):
        y, x = rng.integers(15, H_ - 15), rng.integers(15, W_ - 15)
        cells[y, x] = 1.0
    cells = gaussian_filter(cells, 3.0); cells /= cells.max()
    haze = gaussian_filter(rng.standard_normal((H_, W_)).astype(np.float32), 15.0)
    haze = 0.4 * (haze - haze.min()) / (np.ptp(haze) + 1e-8)
    cn_img = np.clip(cells + haze, 0, 1).astype(np.float32)
    pnr_img = (2 + 18 * cn_img).astype(np.float32)        # PNR tracks the blobs

    mc, mp, ev = H.suggest_corr_pnr(cn_img, pnr_img, sigma=3.0)
    # Threshold must sit above the haze floor (so background is gone) but below the
    # blob peaks (so cells survive), and the cell-like-count curve must be non-trivial.
    assert 0.3 < mc < 0.95 and mp > 2.0
    assert ev["ncell_corr"].max() >= 3
    assert ev["a_min"] >= 3 and ev["a_max"] > ev["a_min"]

    # K==0 guard.
    empty = M.model_quality(CNMFe(CNMFeParams()))
    assert empty["K"] == 0 and M.composite_score(empty) == float("-inf")


def test_temporal_heuristics_fallback(fitted_model):
    """Stage-4 heuristics (Yule-Walker; oasis-independent) complete."""
    if fitted_model.A.shape[1] == 0:
        pytest.skip("no components extracted on this fixture")
    decay, ev = H.suggest_decay_time(fitted_model, 20.0)
    assert decay > 0 and "g_yw" in ev
    gpw, _ = H.suggest_g_prior_weight(ev["g_yw"], 20.0, decay)
    assert gpw in (0.3, 0.5, 0.7)
    merge_thr, _ = H.suggest_merge_thr(fitted_model)
    assert 0.0 < merge_thr <= 0.85
    snr_thr, _ = H.suggest_snr_thr(fitted_model)
    assert np.isfinite(snr_thr)


def test_build_candidates_caps():
    base = CNMFeParams()
    spec = SweepSpec(sigma=[2, 3, 4, 5], min_corr=[0.6, 0.7, 0.8], min_pnr=[6, 10])
    full = build_candidates(base, spec, max_candidates=100)
    assert len(full) == 4 * 3 * 2
    capped = build_candidates(base, spec, max_candidates=5)
    assert len(capped) < len(full)  # one-knob-at-a-time fallback


def test_build_candidates_thr_seeds_coupled():
    """3 method seeds × sigma grid; each candidate sets min_corr AND min_pnr together."""
    from collections import Counter

    base = CNMFeParams()
    spec = SweepSpec(sigma=[2, 3, 4])
    seeds = [{"min_corr": 0.90, "min_pnr": 16.0, "thr_method": "morphology"},
             {"min_corr": 0.80, "min_pnr": 6.0, "thr_method": "separation"},
             {"min_corr": 0.85, "min_pnr": 9.0, "thr_method": "percentile"}]
    cands = build_candidates(base, spec, max_candidates=24, thr_seeds=seeds)
    assert len(cands) == 9  # 3 seeds × 3 sigma
    seed_pairs = {(s["min_corr"], s["min_pnr"]) for s in seeds}
    for p, snap in cands:
        # coupled: the (min_corr, min_pnr) pair is always one whole seed, never a
        # cross-product of independent grids.
        assert (p.min_corr, p.min_pnr) in seed_pairs
        assert snap["thr_method"] in {"morphology", "separation", "percentile"}
    assert Counter(s["thr_method"] for _, s in cands) == \
        {"morphology": 3, "separation": 3, "percentile": 3}


def test_build_candidates_per_sigma_seeds():
    """Seeds that carry their own sigma set (sigma, min_corr, min_pnr) together."""
    base = CNMFeParams()
    # 2 sigmas x 2 methods, each seed a full (sigma, min_corr, min_pnr) triple.
    seeds = [
        {"sigma": 3.0, "min_corr": 0.90, "min_pnr": 16.0, "thr_method": "morphology"},
        {"sigma": 3.0, "min_corr": 0.80, "min_pnr": 6.0, "thr_method": "separation"},
        {"sigma": 2.0, "min_corr": 0.70, "min_pnr": 5.0, "thr_method": "morphology"},
        {"sigma": 2.0, "min_corr": 0.60, "min_pnr": 4.0, "thr_method": "separation"},
    ]
    # sigma not in the spec grid (None) — it comes from the seeds.
    cands = build_candidates(base, SweepSpec(), max_candidates=24, thr_seeds=seeds)
    assert len(cands) == 4
    got = {(p.sigma, p.min_corr, p.min_pnr) for p, _ in cands}
    assert got == {(s["sigma"], s["min_corr"], s["min_pnr"]) for s in seeds}


def test_suggest_corr_pnr_methods_blobby_image():
    """separation + percentile recover a sane threshold on a blobby CORR/PNR image."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(2)
    H_ = W_ = 120
    cells = np.zeros((H_, W_), np.float32)
    for _ in range(20):
        y, x = rng.integers(15, H_ - 15), rng.integers(15, W_ - 15)
        cells[y, x] = 1.0
    cells = gaussian_filter(cells, 3.0); cells /= cells.max()
    cn_img = np.clip(cells, 0, 1).astype(np.float32)
    pnr_img = (2 + 18 * cn_img).astype(np.float32)
    for fn in (H.suggest_corr_pnr_separation, H.suggest_corr_pnr_percentile):
        mc, mp, ev = fn(cn_img, pnr_img, sigma=3.0)
        assert 0.3 <= mc <= 0.98 and mp >= 2.0
        assert ev["n_blobs"] >= 3


def test_suggest_corr_pnr_methods_fallback_on_flat():
    """Too few blobs -> safe defaults (0.8, 10) instead of collapsing to the floor."""
    flat = np.full((40, 40), 0.5, np.float32)
    for fn in (H.suggest_corr_pnr_separation, H.suggest_corr_pnr_percentile):
        mc, mp, _ = fn(flat, flat, sigma=3.0)
        assert mc == 0.8 and mp == 10.0


def test_load_mc_sample_chunk_aligned(sim_zarr):
    """Chunk-aligned sampler returns the right frame count with valid global indices."""
    import zarr

    from tuning.io_sample import load_mc_sample

    arr = zarr.open(str(sim_zarr), mode="r")
    T = arr.shape[0]
    sample, idx = load_mc_sample(arr, 40)
    assert sample.shape[0] == len(idx) <= 40
    assert sample.shape[1:] == arr.shape[1:]
    assert idx.min() >= 0 and idx.max() < T
    assert np.all(np.diff(idx) >= 0)  # sorted global indices
    # sampled frames match the movie at those indices (correct idx bookkeeping)
    assert np.allclose(sample[0], np.asarray(arr[int(idx[0])], dtype=np.float32))
    assert np.allclose(sample[-1], np.asarray(arr[int(idx[-1])], dtype=np.float32))


def test_fig_sweep_footprints_zero_components(corr_image, tmp_path):
    """A 0-component candidate (model.A is None) still writes a backdrop PNG, no crash."""
    import types

    from tuning.report import fig_sweep_footprints

    cn, _ = corr_image
    model = types.SimpleNamespace(
        dims=cn.shape, A=None, accepted_mask=None,
        params=types.SimpleNamespace(min_corr=0.8))
    out = tmp_path / "cand_empty_footprints.png"
    fig_sweep_footprints(model, cn, out_path=str(out), region_crop=None)
    assert out.exists() and out.stat().st_size > 0


def test_sweep_runs_and_ranks(sim_zarr, sim_movie):
    base = CNMFeParams(sigma=3.0, min_corr=0.7, min_pnr=6.0, n_iter_main=1,
                       frame_rate_hz=20.0, decay_time_ms=180.0)
    spec = SweepSpec(min_pnr=[6.0, 12.0])
    H_, W_ = sim_movie.shape[1:]
    region = ((0, H_, 0, W_), (0, sim_movie.shape[0]))
    rows, best_params, best_model = run_sweep(
        sim_zarr, base, spec, region_crop=region, n_jobs=1,
        workdir=sim_zarr.parent / "sweep")
    assert len(rows) >= 1
    scores = [r["score"] for r in rows]
    assert scores == sorted(scores, reverse=True)
    assert isinstance(best_params, CNMFeParams)
    assert best_model.A is not None


def test_pick_cutout(sim_movie):
    sample = sim_movie
    cn = sample.std(axis=0)
    crop, t_range = S.pick_cutout(cn, T=sample.shape[0], cutout_hw=(48, 48),
                                  window_t=100, sample=sample,
                                  sample_idx=np.arange(sample.shape[0]))
    y0, y1, x0, x1 = crop
    assert (y1 - y0, x1 - x0) == (48, 48)
    assert 0 <= t_range[0] < t_range[1] <= sample.shape[0]


def test_run_tuning_writes_report(avi_folder, tmp_path):
    from tuning.tuner import TunerConfig, run_tuning

    run_dir = tmp_path / "run"
    cfg = TunerConfig(
        input_path=avi_folder, output_dir=run_dir, mode="both", region="cutout",
        frame_rate_hz=20.0, decay_time_ms=180.0, ssub=2, tsub=1, max_avis=2,
        n_template_avis=2, stride_within_avi=5, n_init_frames=120,
        n_shift_frames=40, cutout_hw=(48, 48), window_t=200,
        sweep=SweepSpec(min_pnr=[6.0, 10.0]), max_candidates=4, n_jobs=1)
    result = run_tuning(cfg)

    assert (run_dir / "recommended_params.json").exists()
    assert (run_dir / "downsample.json").exists()
    assert (run_dir / "report.md").exists()
    assert list(run_dir.glob("fig_*.png")), "no figures written"
    # the sweep seed-coverage figure is produced when a best candidate is found
    if result.get("sweep") and (run_dir / "fig_sweep_footprints.png").exists():
        assert (run_dir / "fig_sweep_blob_coverage.png").exists()

    # Round-trips into the pipeline.
    params = CNMFeParams.from_json(run_dir / "recommended_params.json")
    assert params.sigma > 0 and params.min_corr > 0
    assert result["ssub"] == 2


def test_recommended_params_roundtrip_into_pipeline(sim_movie, tmp_path):
    """A written recommended_params.json is accepted by fit_extract."""
    p = CNMFeParams(sigma=3.0, min_corr=0.7, min_pnr=6.0, n_iter_main=1)
    jpath = tmp_path / "recommended_params.json"
    p.to_json(jpath)
    loaded = CNMFeParams.from_json(jpath)
    model = CNMFe(loaded).fit_extract(sim_movie, evaluate=True)
    assert model.A is not None


# --------------------------------------------------------------------------
# Packaged component diagnostics (Part 4) + HTML report (Part 3) + CLI (Part 2)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corr_image(sim_movie):
    from minicnmfe.preprocess import correlation_pnr

    cn, pnr = correlation_pnr(sim_movie, sigma=3.0)
    return cn, pnr


def test_diagnostic_figs_return_figure(fitted_model, corr_image, sim_movie, tmp_path):
    """Each packaged diagnostic returns a Figure (out_path=None) and writes a
    non-empty PNG (out_path given)."""
    from matplotlib.figure import Figure

    from tuning import report as R

    cn, pnr = corr_image
    calls = {
        "footprint_grid": lambda op: R.fig_footprint_grid(fitted_model, out_path=op),
        "eccentricity": lambda op: R.fig_eccentricity(fitted_model, out_path=op),
        "jaccard_merge": lambda op: R.fig_jaccard_merge(fitted_model, out_path=op),
        "centroid_drift": lambda op: R.fig_centroid_drift(fitted_model, cn, pnr=pnr, out_path=op),
        "blob_coverage": lambda op: R.fig_blob_coverage(
            fitted_model, cn, pnr, 3.0, min_corr=0.7, min_pnr=6.0, out_path=op),
        "mean_proj_activity": lambda op: R.fig_mean_proj_and_activity(sim_movie, out_path=op),
    }
    for name, fn in calls.items():
        assert isinstance(fn(None), Figure), name
        out = tmp_path / f"{name}.png"
        assert fn(out) is None and out.stat().st_size > 0, name


def test_diagnostic_figs_guard_empty(corr_image):
    """K==0 model: diagnostics still return a Figure (no crash)."""
    from matplotlib.figure import Figure

    from tuning import report as R

    cn, _pnr = corr_image
    empty = CNMFe(CNMFeParams())
    assert isinstance(R.fig_footprint_grid(empty), Figure)
    assert isinstance(R.fig_eccentricity(empty), Figure)
    assert isinstance(R.fig_jaccard_merge(empty), Figure)
    assert isinstance(R.fig_centroid_drift(empty, cn), Figure)
    assert isinstance(
        R.fig_blob_coverage(empty, cn, _pnr, 3.0, min_corr=0.7, min_pnr=6.0),
        Figure)


def test_mean_proj_and_activity_on_zarr(sim_zarr):
    """The streaming reduction works on a zarr (chunked) movie."""
    import zarr

    from tuning.report import mean_proj_and_activity

    z = zarr.open(str(sim_zarr), mode="r")
    mean_img, t_idx, activity = mean_proj_and_activity(z, max_pts=50)
    assert mean_img.shape == tuple(z.shape[1:])
    assert len(t_idx) == len(activity) and len(activity) > 0


def test_write_html_report(avi_folder, tmp_path):
    """A self-contained report.html with inlined figures + sortable table."""
    from tuning import report_html as RH
    from tuning.sweep import SweepSpec
    from tuning.tuner import TunerConfig, run_tuning

    run_dir = tmp_path / "run"
    cfg = TunerConfig(
        input_path=avi_folder, output_dir=run_dir, mode="both", region="cutout",
        frame_rate_hz=20.0, decay_time_ms=180.0, ssub=2, tsub=1, max_avis=2,
        n_template_avis=2, stride_within_avi=5, n_init_frames=120,
        n_shift_frames=40, cutout_hw=(48, 48), window_t=200,
        sweep=SweepSpec(min_pnr=[6.0, 10.0]), max_candidates=4, n_jobs=1)
    result = run_tuning(cfg)
    html_path = RH.write_html_report(run_dir, result)
    assert html_path.exists()
    txt = html_path.read_text()
    assert "data:image/png;base64," in txt          # figures inlined
    assert "Recommended parameters" in txt
    assert "onclick=\"sortTable(this)\"" in txt      # sortable candidate table
    assert 'class="best"' in txt                     # best row highlighted
    assert "How to read these metrics" in txt


def test_tune_cli_dry_run(tmp_path):
    """tune.py --dry-run exits 0, computes nothing, and defaults output to runs/."""
    import subprocess

    fake = tmp_path / "sess.zarr"
    fake.mkdir()
    proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "tune.py"), str(fake),
         "--dry-run", "--indicator", "gcamp8m"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    assert proc.returncode == 0, proc.stderr
    assert "[dry-run]" in proc.stdout
    assert "/runs/tune_" in proc.stdout              # gitignored default output


def test_tune_cli_batch_dry_run(tmp_path):
    """tune.py --sessions <list.txt> --dry-run resolves + prints the batch plan."""
    import subprocess

    sess = tmp_path / "sess.zarr"
    sess.mkdir()
    lst = tmp_path / "list.txt"
    lst.write_text(str(sess) + "\n")
    proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "tune.py"), "--sessions", str(lst),
         "-o", str(tmp_path / "out"), "--dry-run", "--indicator", "gcamp8m"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    assert proc.returncode == 0, proc.stderr
    assert "per-session plan" in proc.stdout
    assert "--validate" in proc.stdout               # consolidated tune+validate call


def test_per_cell_spatial_corr_and_score():
    """cn-proxy: a footprint matching the local CORR scores high, a footprint over
    flat/background CORR scores low; and composite_score rewards higher spatialcorr."""
    import scipy.sparse as sp
    from tuning.metrics import per_cell_spatial_corr, composite_score

    H = W = 24
    yy, xx = np.indices((H, W))
    g = np.exp(-(((yy - 12) ** 2 + (xx - 12) ** 2) / (2 * 3.0 ** 2)))
    cn = g.astype(np.float32)  # CORR hotspot exactly where the cell is
    # cell 0: footprint == the hotspot; cell 1: footprint off in a flat corner
    fp0 = (g > 0.2).astype(np.float32).ravel() * g.ravel()
    fp1 = np.zeros(H * W, np.float32)
    fp1[:9] = 1.0  # 3x3 block at (0,0) where cn≈0
    A = sp.csc_matrix(np.stack([fp0, fp1], axis=1))
    sc = per_cell_spatial_corr(A, cn, (H, W))
    assert sc[0] > 0.5 and sc[0] > sc[1]

    # composite_score: same q but higher spatialcorr_median -> higher score
    base = dict(K=10, cprojcorr_median=0.8, accepted_frac=0.7,
                npix_median=100.0, npix_iqr=20.0, multipeak_frac=0.3)
    hi = composite_score({**base, "spatialcorr_median": 0.6})
    lo = composite_score({**base, "spatialcorr_median": 0.2})
    assert hi > lo
    # missing spatialcorr (NaN) contributes 0 (legacy behaviour), not a crash
    none = composite_score({**base, "spatialcorr_median": float("nan")})
    assert none == composite_score({**base})


def test_edge_solution_detector_scales_with_axis_length():
    """An argmax in the bottom bins of the axis is an edge solution, and the band
    scales so a 30-point and a 60-point search agree on what "the edge" means."""
    assert H._is_edge_solution(0, 30) and H._is_edge_solution(1, 30)
    assert not H._is_edge_solution(2, 30)
    assert H._is_edge_solution(2, 60) and not H._is_edge_solution(3, 60)
    # a genuine interior optimum is never an edge solution
    assert not H._is_edge_solution(15, 30)


def test_morphology_collapsing_to_floor_falls_back_to_safe_default():
    """A CORR image whose cell-like count only ever falls must not return the
    floor: that is where the search stopped, not what the data says. This is the
    SOM/PV failure mode — a min_corr=0.4 seed over-segments by 10-50x."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(0)
    H_ = W_ = 120
    # Sparse, low-contrast field: faint texture only, no compact high-CORR blobs.
    # Squaring pushes most of the image well below the floor, so the few blobs that
    # survive at the most permissive threshold only ever thin out from there — the
    # cell-like count peaks at bin 0 and decays.
    noise = gaussian_filter(rng.standard_normal((H_, W_)).astype(np.float32), 3.0)
    noise = (noise - noise.min()) / np.ptp(noise)
    cn_img = (0.6 * noise ** 2).astype(np.float32)
    pnr_img = (2.0 + 4.0 * cn_img).astype(np.float32)

    mc, mp, ev = H.suggest_corr_pnr(cn_img, pnr_img, sigma=3.0)
    assert int(np.argmax(ev["ncell_corr"])) == 0 and ev["ncell_corr"].max() > 0
    assert ev["edge_corr"], "count curve peaking at the axis bottom must be flagged"
    assert mc == H.SAFE_MIN_CORR
    assert mc > 0.4, "must not hand back the corr_floor"
    # the un-substituted argmax is kept for the report
    assert ev["min_corr_raw"] is not None and ev["min_corr_raw"] < mc


def test_morphology_interior_optimum_is_not_flagged():
    """The healthy case (compact blobs on haze) keeps its data-driven threshold."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(1)
    H_ = W_ = 120
    cells = np.zeros((H_, W_), np.float32)
    for _ in range(15):
        y, x = rng.integers(15, H_ - 15), rng.integers(15, W_ - 15)
        cells[y, x] = 1.0
    cells = gaussian_filter(cells, 3.0); cells /= cells.max()
    haze = gaussian_filter(rng.standard_normal((H_, W_)).astype(np.float32), 15.0)
    haze = 0.4 * (haze - haze.min()) / (np.ptp(haze) + 1e-8)
    cn_img = np.clip(cells + haze, 0, 1).astype(np.float32)
    pnr_img = (2 + 18 * cn_img).astype(np.float32)

    mc, mp, ev = H.suggest_corr_pnr(cn_img, pnr_img, sigma=3.0)
    assert not ev["edge_corr"]
    assert mc == ev["min_corr_raw"], "interior optimum must be returned unchanged"


def test_percentile_and_separation_flag_floor_solutions():
    """Methods B and C report an edge when the floor, not the data, sets the
    threshold — previously they silently clamped to it."""
    flat = np.zeros((64, 64), np.float32)
    for fn in (H.suggest_corr_pnr_separation, H.suggest_corr_pnr_percentile):
        mc, mp, ev = fn(flat, flat, sigma=3.0)
        # too few blobs -> safe default, and the evidence dict still carries the keys
        assert mc == H.SAFE_MIN_CORR and mp == H.SAFE_MIN_PNR
        assert "edge" in ev and "min_corr_raw" in ev


def test_mixture_seeds_scales_with_cell_density():
    """Local maxima + a log-space mixture must track how many cells there are.

    A global threshold cannot: on a dense field it fuses everything into one blob,
    on a sparse one it only reveals the brightest few. That asymmetry is why the
    thresholded-component detector returned ~25 blobs on a WT field holding ~300.
    """
    from scipy.ndimage import gaussian_filter
    from minicnmfe.initialization import mixture_seeds

    rng = np.random.default_rng(0)
    H_ = W_ = 128
    sigma = 3.0

    def field(n_cells):
        cn = np.abs(gaussian_filter(rng.standard_normal((H_, W_)).astype(np.float32), 1.0))
        cn = 0.25 * cn / (cn.std() + 1e-8)
        pos = rng.integers(12, H_ - 12, size=(n_cells, 2))
        cells = np.zeros((H_, W_), np.float32)
        for y, x in pos:
            cells[y, x] = 1.0
        cells = gaussian_filter(cells, sigma)
        cells = 3.0 * cells / (cells.max() + 1e-8)
        return np.clip(cn + cells, 0, None), np.full((H_, W_), 8.0, np.float32)

    sparse = len(mixture_seeds(*field(8), sigma))
    dense = len(mixture_seeds(*field(80), sigma))
    assert dense > sparse, f"seed count must grow with density (sparse={sparse}, dense={dense})"
    assert sparse >= 3, f"sparse field should still yield seeds, got {sparse}"


def test_mixture_seeds_degenerate_fit_falls_back():
    """One dominating peak must not collapse the seed set.

    Fitting raw (not log) heights on a dim PV session put a single extreme peak in
    the 'high' component and returned ONE seed — seeding an extraction with one
    point is worse than any threshold, so the guard returns the brightest decile.
    """
    from minicnmfe.initialization import mixture_seeds

    H_ = W_ = 96
    sigma = 2.0
    cn = np.full((H_, W_), 0.05, np.float32)
    rng = np.random.default_rng(1)
    for y in range(6, H_ - 6, 7):          # a regular lattice of weak maxima
        for x in range(6, W_ - 6, 7):
            cn[y, x] = 0.06 + 0.01 * rng.random()
    cn[48, 48] = 50.0                       # one dominating peak
    out = mixture_seeds(cn, np.full((H_, W_), 5.0, np.float32), sigma)
    assert len(out) >= 3, f"degenerate fit must not collapse the seed set, got {len(out)}"


def test_seed_method_default_and_legacy_path_reachable():
    """Mixture seeding is the default; the legacy threshold gate stays reachable.

    The function-level defaults stay "threshold" so any direct caller of the
    initialization functions keeps its old behaviour; only the pipeline-level
    CNMFeParams default moves."""
    import inspect
    from minicnmfe.initialization import (greedy_corr_pnr, greedy_corr_pnr_patched,
                                          _greedy_patch_worker)

    assert CNMFeParams().seed_method == "mixture"
    assert CNMFeParams(seed_method="threshold").seed_method == "threshold"
    for fn in (greedy_corr_pnr, greedy_corr_pnr_patched, _greedy_patch_worker):
        p = inspect.signature(fn).parameters["seed_mode"]
        assert p.default == "threshold", f"{fn.__name__} default is {p.default!r}"


def test_sweep_searches_sigma_only_by_default(sim_zarr, tmp_path):
    """Default sweep varies sigma alone; the legacy setting restores threshold seeds.

    min_corr/min_pnr have no runtime effect under mixture seeding, so expanding the
    sweep by three threshold methods per sigma multiplies candidates for nothing.
    """
    from tuning.tuner import TunerConfig, run_tuning

    common = dict(input_path=sim_zarr, frame_rate_hz=20.0, decay_time_ms=180.0,
                  mode="sweep", region="full", n_init_frames=200, max_candidates=12)

    res_mix = run_tuning(TunerConfig(output_dir=tmp_path / "mix", **common))
    seeds_mix = res_mix["stages"]["thr_grid"]["seeds"]
    n_sigma = len({round(s["sigma"], 3) for s in seeds_mix})
    assert len(seeds_mix) == n_sigma, (
        f"sigma-only sweep should give one seed per sigma, got {len(seeds_mix)} "
        f"for {n_sigma} sigma value(s)"
    )
    assert {s["thr_method"] for s in seeds_mix} == {"morphology"}

    res_leg = run_tuning(TunerConfig(output_dir=tmp_path / "leg",
                                     seed_method="threshold", **common))
    seeds_leg = res_leg["stages"]["thr_grid"]["seeds"]
    assert len(seeds_leg) >= len(seeds_mix), "legacy path should not shrink the grid"
