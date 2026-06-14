"""Verify the sweep now prefers sigma=3 (spatial-aware score + min_pixel=1).
Reruns the tuner on the PICAST mc.zarr to a local output dir."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tuning.tuner import TunerConfig, run_tuning  # noqa: E402

MC = "/media/server/archive/projects/2023_intercontext/PICAST/data/1_preprocessed/" \
     "20260505_m0010800_wt_1557/miniscope_video/minicnmfe_mc_mcid_0/mc.zarr"
OUT = ROOT / "live_runs" / "tuner_rerun_out"

cfg = TunerConfig(input_path=Path(MC), output_dir=OUT, mode="both", region="cutout",
                  frame_rate_hz=20.0, decay_time_ms=180.0, ssub=1, tsub=1, n_jobs=-1)
res = run_tuning(cfg)
rp = res["recommended_params"]
print("\n==== RESULT ====", flush=True)
print(f"sigma={rp.sigma} min_pixel={rp.min_pixel} thr_method={rp.spatial_thr_method} "
      f"nrg={rp.spatial_nrg_thr} min_corr={rp.min_corr:.3f} min_pnr={rp.min_pnr:.3f}", flush=True)
rows = (res.get("sweep") or {}).get("rows") or []
print("idx sigma score K Kacc cproj spatialcorr npix multipeak", flush=True)
for r in rows:
    print(f"{r.get('idx')} {r.get('sigma')} {r.get('score'):.3f} {r.get('K')} "
          f"{r.get('K_accepted')} {r.get('cprojcorr_median'):.3f} "
          f"{r.get('spatialcorr_median', float('nan')):.3f} {r.get('npix_median'):.0f} "
          f"{r.get('multipeak_frac', float('nan')):.2f}", flush=True)
print(f"\nout: {OUT}", flush=True)
