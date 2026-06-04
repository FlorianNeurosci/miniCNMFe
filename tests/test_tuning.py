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
    assert 0.0 < min_corr < 1.0 and min_pnr > 0.0 and "counts" in ev

    min_pixel, ev = H.suggest_min_pixel(sample, sigma_refit, min_corr, min_pnr,
                                        sim_movie.shape[1:])
    assert min_pixel >= 1


def test_metrics_on_fitted_model(fitted_model):
    q = M.model_quality(fitted_model)
    for key in ("K", "K_accepted", "accepted_frac", "cprojcorr_mean",
                "cprojcorr_median", "npix_median", "npix_iqr", "snr_mean",
                "snr_median"):
        assert key in q
    assert 0.0 <= q["accepted_frac"] <= 1.0
    if q["K"] > 0:
        assert -1.0 <= q["cprojcorr_median"] <= 1.0
    assert np.isfinite(M.composite_score(q))

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
