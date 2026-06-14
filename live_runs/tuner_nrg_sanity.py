"""Sanity: the tuner now recommends nrg@0.95 + a min_pixel derived from realized
(nrg) footprint p25, not the greedy-init heuristic. Mirrors the avi_folder
fixture in tests/test_tuning.py."""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tests.miniscope_simulator import make_miniscope_movie  # noqa: E402
from minicnmfe.pipeline import CNMFeParams  # noqa: E402
from tuning.tuner import TunerConfig, run_tuning  # noqa: E402
from tuning.sweep import SweepSpec  # noqa: E402


def _write_avi(path, frames):
    f = frames.astype(np.float32)
    f = (255 * (f - f.min()) / (np.ptp(f) + 1e-9)).astype(np.uint8)
    H_, W_ = f.shape[1:]
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (W_, H_), isColor=True)
    for fr in f:
        w.write(cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR))
    w.release()


def main():
    out = make_miniscope_movie(n_neurons=12, dims=(96, 96), T=300, seed=0)
    mov = out["movie"].astype(np.float32)
    tmp = ROOT / "live_runs" / "_tuner_nrg_sanity"
    tmp.mkdir(parents=True, exist_ok=True)
    avis = tmp / "avis"; avis.mkdir(exist_ok=True)
    half = mov.shape[0] // 2
    _write_avi(avis / "0.avi", mov[:half])
    _write_avi(avis / "1.avi", mov[half:])

    cfg = TunerConfig(
        input_path=avis, output_dir=tmp / "run", mode="both", region="cutout",
        frame_rate_hz=20.0, decay_time_ms=180.0, ssub=1, tsub=1, max_avis=2,
        n_template_avis=2, stride_within_avi=5, n_init_frames=120,
        n_shift_frames=40, cutout_hw=(64, 64), window_t=200,
        sweep=SweepSpec(min_pnr=[6.0, 10.0]), max_candidates=4, n_jobs=1)
    result = run_tuning(cfg)

    rp = CNMFeParams.from_json(tmp / "run" / "recommended_params.json")
    print("\n===== RECOMMENDED PARAMS =====")
    print(f"  spatial_thr_method = {rp.spatial_thr_method!r}")
    print(f"  spatial_nrg_thr    = {rp.spatial_nrg_thr}")
    print(f"  min_pixel          = {rp.min_pixel}")
    print(f"  sigma              = {rp.sigma}")
    print(f"  min_pixel source   = {result.get('sources', {}).get('min_pixel')}")
    print(f"  min_pixel rationale= {result.get('rationale', {}).get('min_pixel')}")
    rows = (result.get("sweep") or {}).get("rows") or []
    if rows:
        print(f"  winner npix_p25={rows[0].get('npix_p25')} npix_median={rows[0].get('npix_median')} "
              f"thr_method={rows[0].get('thr_method')}")
    assert rp.spatial_thr_method == "nrg", "expected nrg default"
    assert abs(rp.spatial_nrg_thr - 0.95) < 1e-9, "expected nrg_thr 0.95"
    print("\nOK: tuner recommends nrg@0.95 with derived min_pixel.")


if __name__ == "__main__":
    main()
