"""Tests for the full-recording validation workflow (``tuning/validate.py``).

End-to-end-ish on a 2-file simulator AVI folder: metadata parsing, then a
two-threshold validation run that fuses MC + transposes Y_flat once and extracts
twice. OASIS is absent in the env so the pure-Python AR(1) fallback is exercised.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.miniscope_simulator import make_miniscope_movie  # noqa: E402
from tuning.validate import (  # noqa: E402
    good_defaults,
    read_session_meta,
    resolve_session_paths,
    validate_session,
)

cv2 = pytest.importorskip("cv2")


def _write_avi(path, frames):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20.0,
                         (frames.shape[2], frames.shape[1]), isColor=True)
    assert vw.isOpened()
    try:
        u8 = (255 * np.clip(frames, 0, None) / (frames.max() + 1e-8)).astype(np.uint8)
        for f in u8:
            vw.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
    finally:
        vw.release()


@pytest.fixture(scope="module")
def session_folder(tmp_path_factory):
    folder = tmp_path_factory.mktemp("session")
    mov = make_miniscope_movie(n_neurons=12, dims=(96, 96), T=240, seed=3)["movie"].astype(np.float32)
    half = mov.shape[0] // 2
    _write_avi(folder / "0.avi", mov[:half])
    _write_avi(folder / "1.avi", mov[half:])
    # synthetic metadata
    (folder / "metaData.json").write_text(json.dumps({
        "ROI": {"height": 96, "width": 96}, "frameRate": "20FPS",
        "deviceType": "Miniscope_V4_BNO"}))
    (folder / "timeStamps.csv").write_text(
        "Frame Number,Time Stamp (ms),Buffer Index\n"
        + "\n".join(f"{i},{i*50},0" for i in range(240)))
    return folder


def test_read_session_meta(session_folder):
    meta = read_session_meta(session_folder)
    assert meta["fps"] == 20.0
    assert meta["dims"] == (96, 96)
    assert meta["fps_measured"] is not None and 18 < meta["fps_measured"] < 22
    assert meta["n_frames_ts"] == 240


def test_resolve_session_paths(tmp_path):
    a = tmp_path / "sessA"; a.mkdir()
    b = tmp_path / "sessB"; b.mkdir()
    # direct list of paths (with a duplicate + a missing one)
    out = resolve_session_paths([str(a), str(b), str(a), str(tmp_path / "nope")])
    assert out == [a, b]
    # a .txt list with comments / blanks
    lst = tmp_path / "sessions.txt"
    lst.write_text(f"# my sessions\n{a}\n\n{b}\n")
    out2 = resolve_session_paths(str(lst))
    assert out2 == [a, b]
    # single string path
    assert resolve_session_paths(str(a)) == [a]
    # nothing valid -> raises
    import pytest as _pt
    with _pt.raises(FileNotFoundError):
        resolve_session_paths([str(tmp_path / "ghost")])


def test_good_defaults_overrides():
    p = good_defaults(frame_rate_hz=20.0, decay_time_ms=180.0)
    assert p.global_bg_rank == 1
    assert p.decay_time_ms == 180.0
    assert p.g_prior_weight == 0.6
    # PNR-safe init: full-T greedy (no peak loss); speed from the CORR-only stride.
    assert p.init_stride == 1
    assert p.init_corrpnr_stride == 3
    # Acceptance gate is OFF by default (report-only); ghosts handled upstream.
    assert p.auto_eval_snr_amp_thr == 0.0
    assert p.min_pixel == 10


def test_validate_two_thresholds(session_folder, tmp_path):
    out = tmp_path / "validation"
    native = good_defaults(frame_rate_hz=20.0, decay_time_ms=180.0, sigma=6.0,
                           min_corr=0.7, min_pnr=6.0, n_jobs=1)
    res = validate_session(
        session_folder, out, native_params=native, ssub=2, tsub=1,
        threshold_sets=[("recommended", 0.7, 6.0), ("lowthr", 0.6, 3.0)],
        n_template_avis=2, verbose=False)

    # MC + Y_flat (and the shared CORR/PNR images) produced exactly once.
    assert (out / "mc" / "mc.zarr").exists()
    assert (out / "Y_flat_pixel.zarr").exists()
    assert (out / "cn.npy").exists() and (out / "pnr.npy").exists()
    assert (out / "comparison.md").exists()
    # comparison table surfaces the new quality columns + verdict.
    comp = (out / "comparison.md").read_text()
    for col in ("blob_recall", "footprint_precision", "trace_corr_median", "status"):
        assert col in comp, col

    assert len(res["rows"]) == 2
    labels = {r["label"] for r in res["rows"]}
    assert labels == {"recommended", "lowthr"}
    for label in ("recommended", "lowthr"):
        rd = out / f"run_{label}"
        assert (rd / "A.npz").exists() and (rd / "C.npy").exists()
        assert (rd / "summary.txt").exists()
        row = next(r for r in res["rows"] if r["label"] == label)
        # quality metrics + verdict present on every row.
        for key in ("blob_recall", "footprint_precision", "n_blobs",
                    "n_footprints", "status"):
            assert key in row, key
        assert row["status"] in ("PASS", "WARN")
        figs = list((rd / "figs").glob("*.png"))
        if row["K"] > 0:
            assert figs, f"no figures for {label} despite K={row['K']}"
            assert (rd / "figs" / "blob_coverage.png").exists()


def test_validate_reuse_mc(session_folder, tmp_path):
    """A second validate run with reuse_mc skips fusing and still extracts."""
    out1 = tmp_path / "v1"
    native = good_defaults(frame_rate_hz=20.0, min_corr=0.7, min_pnr=6.0, n_jobs=1)
    validate_session(session_folder, out1, native_params=native, ssub=2,
                     threshold_sets=[("a", 0.7, 6.0)], n_template_avis=2, verbose=False)
    mc_path = out1 / "mc" / "mc.zarr"
    assert mc_path.exists()

    out2 = tmp_path / "v2"
    res = validate_session(session_folder, out2, native_params=native, ssub=2,
                           threshold_sets=[("b", 0.7, 6.0)], reuse_mc=mc_path,
                           verbose=False)
    assert not (out2 / "mc").exists()       # did not fuse a new mc
    assert (out2 / "run_b" / "A.npz").exists()
    assert len(res["rows"]) == 1
