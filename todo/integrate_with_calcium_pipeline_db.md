# Integrating simpler_cnmfe into CalciumImagingPipelineDB

Reference notes for the future integration work. Lives in this package
because the integration is *about* `simpler_cnmfe`; the DB code itself is at
`/home/fs539/code/CalciumImagingPipelineDB/` and is untouched here.

## Short answer

Almost no changes needed in `simpler_cnmfe`. The package already has the
right API surface — `CNMFe(params).fit_mc_from_avis(folder, output_dir)`
for MC and `CNMFe(params).fit(...)` for extraction. The work is integration
glue inside CalciumImagingPipelineDB (new method branches + adapters).

There's exactly **one tiny patch** worth making in `simpler_cnmfe` to
avoid friction (see #1 below); everything else is downstream.

## What the DB does (factual, with file:line)

All references in
`/home/fs539/code/CalciumImagingPipelineDB/calcium_pipeline_DB/schemas/calcium_imaging.py`:

- **Cropping** (line 430+) — temporal/spatial trim of source AVIs.
  Output: AVI chunks in `1_preprocessed/<session>/`.
- **Downsampling** (line 808–847) — `caiman_ds`: loads each cropped AVI,
  calls `cm.load(ele).resize(fx=1/spds, fy=1/spds, fz=1/tds).save(...)`,
  output is `<basename>_ds.avi` **in the same dir as the cropped AVIs**
  (line 837). Multiple files preserved.
- **MotionCorrection** (line ~1240–1365) — `caiman_mc`: globs `*_ds.avi`
  (line 1251), passes the list to `MotionCorrect(fname=fnames, ...)`
  (line 1283), then `cm.save_memmap(..., order="C", border_to_0=bord_px)`
  (line 1330) → produces `postmc_mcid_<hash>.mmap` in `1_preprocessed/`.
  Also auto-saves a `tds_factor=10` MC video (line 1352) and inserts
  rigid shifts + template into `RigidMotionCorrectionCaiman` (line 1314).
- **ComponentExtraction** (line 2497–2599) — currently only
  `caiman_CNMFe`: `cm.load_memmap` (line 2569) → reshape to `(T, H, W)`
  (line 2572) → `CNMF(params).fit(images)` (line 2573) →
  `cnm.save(<...>.hdf5)` (line 2584).

Method-enum tables to extend: `MotionCorrectionMethod` (line 933),
`ComponentExtractionMethod` (line 2211).

## Things to watch out for

### 1. Filename pattern — one-line patch worth making in simpler_cnmfe

`_numeric_key` in `concat_avis_to_zarr.py:73-76` matches `^\d+$` against
the file stem. The DB's Downsampling step writes `0_ds.avi`,
`1_ds.avi`, …; their stems (`0_ds`, `1_ds`, …) **don't match**, so
`fit_mc_from_avis` would skip every file. Two ways out:

- **Patch simpler_cnmfe** (~1 line): widen the regex to accept
  `^\d+(_ds)?$`, or capture the leading digits via a non-fullmatch.
  Cleanest, makes the package work out-of-the-box with DB outputs.
- **Or in the adapter** on the DB side: pass `pattern="*_ds.avi"` AND
  monkey-patch `_numeric_key`, OR rename the files at integration time.
  Worse.

### 2. MC output format — memmap vs zarr

DB's caiman path writes a memmap (`cm.save_memmap`, `order="C"`).
`simpler_cnmfe.fit_mc_from_avis` writes a zarr. Two clean choices for
the new MC method branch:

- **Native zarr** (recommended). New `MotionCorrectionMethod` row
  `"simpler_cnmfe_mc"` whose `make()` writes `mc.zarr` + `shifts.npy`
  to the per-session preprocessed dir. Then add a matching
  `extraction_method = "simpler_cnmfe"` branch on
  `ComponentExtraction.make()` that loads the zarr via
  `cnmfe.io.open_zarr(...)` and calls
  `CNMFe(params).fit(zarr, do_motion_correction=False)`. **Zarr only
  crosses the simpler_cnmfe boundary**; the rest of the DB schema stays
  untouched. The `MotionCorrectionTask.mc_output_dir` field stores the
  zarr path — `varchar(255)`, no schema change needed.
- **Memmap shim** (DON'T do this). Convert zarr → memmap so caiman-based
  extraction can still consume it. Loses the streaming benefits of zarr
  for no gain, since you'd also need the matching simpler_cnmfe
  extraction branch anyway.

### 3. Parameter dict — names and units mismatch

DB stores extraction params as a CaImAn-flavoured dict (line 2238).
`simpler_cnmfe.CNMFeParams` uses different field names and one different
unit. Translation the adapter needs to do (CaImAn → cnmfe):

| CaImAn key | simpler_cnmfe field | Notes |
|---|---|---|
| `gSig` (px) | `sigma` (px) | Same units; ~drop-in |
| `gSig_filt` (px) | `mc_gSig_filt` | MC high-pass filter sigma |
| `min_corr` | `min_corr` | Same |
| `min_pnr` | `min_pnr` | Same |
| `decay_time` (**s**) | `decay_time_ms` (**ms**) | **Units differ — multiply by 1000** |
| `fr` | `frame_rate_hz` | DB already divides by `temporal_ds_factor` at line 2541-2543 — reuse that |
| `K` (max components) | `max_neurons` | Mostly same role |
| `merge_thr` | `merge_thr_corr` | Same role |
| `ring_CNN`, `gnb`, `pw_rigid`, etc. | n/a | CaImAn-specific, ignore |

Build this as a `caiman_params_to_cnmfe_params(global_params,
ce_sess_specific_params, fr, decay_time_s) -> CNMFeParams` helper
inside the DB-side adapter. **Don't** change `CNMFeParams` to accept
the CaImAn names — that pollutes the simpler_cnmfe surface for standalone
notebook users.

### 4. `decay_time` is hardcoded at 0.4 s (line 2546)

The DB currently uses `decay_time = 0.4` with a
`# todo: read out the construct and get the decay time from that?`
comment. simpler_cnmfe's Bayesian-prior path on `g` (the whole reason
`decay_time_ms` exists — see CLAUDE.md *Bayesian prior on `g`* section)
wants the **actual indicator τ**.

- Honest fallback: pass `0.4 * 1000 = 400 ms` (≈ jGCaMP8m).
- Proper fix: per-recording indicator field + a τ table
  (GCaMP6f=140, jGCaMP7f=160, jGCaMP8f=70, jGCaMP8m=180, jGCaMP8s=350,
  GCaMP6s=1000 — all in ms). One-time table, then look up at
  extraction time.

### 5. Output format on disk

DB extraction writes one HDF5 file via `cnm.save()` (line 2584).
simpler_cnmfe's `model.save(out_dir)` writes a **directory** with
`A.npz`, `C.npy`, `S.npy`, `YrA.npy`, `g.npy`, `sn_per_k.npy`,
`accepted_mask.npy`, `params.json`, `manifest.json`, etc.

- DB's `extraction_output_dir` is already a *dir* path
  (`varchar(255)`) not a file path — storing the directory works
  without a schema change.
- **Anything downstream** of `ComponentExtraction` that loads via
  `cm.load_CNMF(hdf5_path)` will fail on the simpler_cnmfe row.
  Audit `CaimanComponentEvaluator.py` and any evaluation tables before
  flipping; branch on `extraction_method` there too.

### 6. `RigidMotionCorrectionCaiman` table (line 1314)

DB stores `x_shifts`, `y_shifts`, `x_std`, `y_std`,
`mc_total_template` after caiman MC. simpler_cnmfe gives us
`model.shifts` `(T, 2) float32` already; template is currently internal
to `_build_template_from_strided_avis` in `cnmfe/avi_mc.py`. Decide:

- Reuse `RigidMotionCorrectionCaiman` (slightly confusing name but
  zero schema change), OR
- New `RigidMotionCorrectionSimplerCnmfe` table with the same shape.

Optional simpler_cnmfe touch-up: have `concat_avis_to_mc_zarr` save
`mc_template.npy` alongside `mc.zarr` + `shifts.npy` so the adapter
doesn't have to recompute the template.

### 7. CaImAn cluster vs joblib

DB sets up a CaImAn cluster (`dview`, `n_processes`) and threads them
down. simpler_cnmfe uses joblib via `CNMFeParams.n_jobs`. They don't
interfere when run in separate processes, but if both run in the same
process watch for the "cluster is already running" error documented
in `simpler_cnmfe/CLAUDE.md` (Windows-specific caveats). The
workaround: between caiman and simpler_cnmfe calls, drain the loky
pool with

```python
from joblib.externals.loky import get_reusable_executor
get_reusable_executor().shutdown(wait=True)
```

### 8. `gSig` exploration step

DB has `CaimanGSigPreMCExplorationTask` (line 851) — runs MC at
multiple `g_sig` values to help pick one. No equivalent in
simpler_cnmfe. Options:

- Skip for the simpler_cnmfe path. Users pick `sigma` from CORR/PNR
  plots, which is the standard simpler_cnmfe workflow.
- Implement a parallel `SimplerCnmfeGSigPreMCExplorationTask` later
  if the tuning UX matters.

### 9. Cropping is a DB step; not in simpler_cnmfe

simpler_cnmfe doesn't have a cropping helper, but it doesn't need
one. `Cropping` writes AVIs that simpler_cnmfe can consume directly
(fused MC) or that go through downsampling first. Cropping happens
**upstream of simpler_cnmfe**.

### 10. The MC video is already produced by the DB

DB already saves a `tds_factor=10` MC video (line 1352). The current
`live_runs/run_session.ipynb`'s `mc_downsampled.mp4` cell duplicates
that. If the notebook becomes DB-aware, drop that cell. If it stays
standalone, keep it.

## Concrete integration sketch

**Tiny changes in `simpler_cnmfe`:**

1. Widen `_numeric_key` regex in `concat_avis_to_zarr.py:73-76` to
   accept `\d+(_ds)?`.
2. *(Optional)* Save `mc_template.npy` alongside `mc.zarr` +
   `shifts.npy` in `concat_avis_to_mc_zarr` (cnmfe/avi_mc.py) so the
   DB can populate the rigid-MC results table without an extra pass.

**Everything else inside `CalciumImagingPipelineDB`:**

- Add `"simpler_cnmfe_mc"` row to `MotionCorrectionMethod`; add a
  `make()` branch on `MotionCorrection` that calls
  `CNMFe(params).fit_mc_from_avis(folder, mc_output_dir)`, stores
  `mc.zarr` + `shifts.npy` path, and inserts shifts/template into the
  rigid-MC results table.
- Add `"simpler_cnmfe"` row to `ComponentExtractionMethod`; add a
  `make()` branch on `ComponentExtraction` that opens the zarr via
  `cnmfe.io.open_zarr(...)`, builds `CNMFeParams` via a
  `caiman_params_to_cnmfe_params(...)` helper (handles the rename +
  the `decay_time` s→ms conversion + the temporal-ds frame-rate
  adjustment), calls `CNMFe(params).fit(z, do_motion_correction=False,
  output_dir=<path>)`, and stores the results-directory path in
  `extraction_output_dir`.
- Audit anything downstream of `ComponentExtraction`
  (`CaimanComponentEvaluator`, any evaluation tables) and branch on
  `extraction_method` so callers don't assume a single HDF5 file.

## Sanity-check after integration

- Synthetic AVI test in `simpler_cnmfe`: rename the existing
  `tests/test_concat_avis.py` synthetic files to `0_ds.avi`,
  `1_ds.avi`, … and confirm `fit_mc_from_avis(pattern="*_ds.avi")`
  picks them up after the regex widening.
- End-to-end DataJoint smoke: pick one small session in the DB;
  populate Cropping → Downsampling → MotionCorrection
  (`simpler_cnmfe_mc`) → ComponentExtraction (`simpler_cnmfe`).
  Confirm `mc.zarr`, `shifts.npy`, and the results directory are
  written under `1_preprocessed/<session>/`.
- Cross-check K and a handful of footprint outlines against the
  `caiman_CNMFe` path on the same session.
