# TODO: Spatial/temporal "cutout" (crop) of the movie before extraction

**Status: DONE** (both phases). Implemented in `cnmfe/cutout.py`,
`CNMFe` (`_ingest_cutout` in `fit`/`fit_mc`, `place_in_full_fov`, save/load
`cutout` meta), `CNMFeParams` (`temporal_crop`/`spatial_crop`/`spatial_mask_path`,
`downscaled` clears them), and the fused decode path
(`_decode_avi_worker` crop args + `concat_avis_to_mc_zarr` → `cutout.json`).
Tests: `tests/test_cutout.py` (14). Plan retained below for reference.

## Context

Let the user restrict CNMFe to a sub-region and/or a time window of the
recording — a "cutout" specified up front on the `CNMFeParams` object. There is
**no crop/ROI support today** (the `select_roi` note in CLAUDE.md is an
unimplemented design sketch for shift estimation). Key enabler: `self.dims` is
set from whatever movie shape arrives and propagates cleanly (ring indices,
`make_3d`, init, background all treat their input as "the full FOV"), so
**cropping the movie at ingestion just works** — everything downstream runs on
the cutout with no further changes.

Decisions (from clarifying questions):
- **All paths**, including the fused AVI→MC path (crop during decode, single write).
- **Map back to full FOV** — helper to place footprints/traces back into the
  original FOV / timeline (parallels `CNMFe.upsample_to_native`).
- **Rectangle + arbitrary mask** — a boolean ROI mask is supported; its bounding
  box sets the crop rect and pixels outside the mask are zeroed.

Design choices baked in:
- **Cutout is applied FIRST, before MC**, in every path (uniform; required by the
  fused decode→crop→MC pass). Spatial-crop-before-MC means MC can't pull in
  content from outside the crop, so expect minor artifacts within ~`max_shift`
  px of the crop border — same accepted tradeoff as temporal-binning-before-MC.
  Suggest leaving a small margin around the ROI. Temporal crop before MC is clean.
- **Native coordinates.** Crop fields are always in native (full-res) units and
  consumed at ingestion, *before* any downsampling. `downscaled()` therefore
  **clears** the crop fields (already applied upstream).
- **Ownership (no double-apply):** crop is applied by the raw-movie entry points
  — `fit()` (top, before MC / before delegating) and `fit_mc`/`fit_mc_from_avis`
  (while producing `mc.zarr`). `fit_extract` does **not** crop (its input is
  already the ROI), so the staged `fit_mc_from_avis → fit_extract(mc.zarr)` flow
  crops exactly once.

## CNMFeParams fields (JSON/CLI-safe)

```
temporal_crop:     tuple[int, int] | None = None            # (t0, t1), t1 exclusive
spatial_crop:      tuple[int, int, int, int] | None = None  # (y0, y1, x0, x1)
spatial_mask_path: str | None = None                        # path to a bool .npy (H,W)
```
A mask array is not stored inline (keeps `to_json`/`from_json` clean); it lives
in a `.npy` referenced by path. `from_json` gets the same tuple-restore treatment
as `max_shift` (`pipeline.py:404`). `downscaled()` sets all three to `None`.

## New module `cnmfe/cutout.py`

- `resolve_cutout(params, native_dims, native_T) -> dict` — validate + normalize:
  intersect `spatial_crop` with the mask's bounding box → final `bbox
  (y0,y1,x0,x1)`; clamp `temporal_crop` to `(0, T)`; load the mask once. Returns
  `{bbox, t_range, mask_local (bbox-cropped bool or None), orig_dims, orig_T}`.
- `apply_cutout(movie, spec) -> np.ndarray` — temporal slice `[t0:t1]`, spatial
  bbox slice `[y0:y1, x0:x1]`, then zero pixels outside `mask_local`. Works on a
  numpy array or a `zarr.Array` (lazy slice → materialize the small crop).
- `place_footprints_in_fov(A_crop, bbox, orig_dims)` — pad sparse `(h·w, K)`
  footprints back to `(H·W, K)` at the `(y0,x0)` offset (zeros elsewhere).
- `place_traces_in_timeline(C_crop, t_range, orig_T)` — embed `(K, T_win)` into
  `(K, orig_T)` at `[t0:t1]` (zeros elsewhere).

## Pipeline wiring (`cnmfe/pipeline.py`)

- `fit()`: call `resolve_cutout` + `apply_cutout` on the movie at the top (both
  the `do_motion_correction` branch — before MC — and the delegate-to-extract
  branch). Store `self.cutout = spec`. When `Y_flat_zarr` is supplied
  (true-streaming), require the crop to be pre-baked and raise if crop fields
  are set (documented).
- `fit_mc`: when reading a raw zarr/numpy movie, apply the cutout before MC and
  record `self.cutout`.
- `place_in_full_fov(self, *, place_time=True) -> CNMFe` — **new method**,
  non-destructive, mirroring `upsample_to_native`: returns a new model with `A`
  padded to `orig_dims` and (if `place_time`) `C`/`YrA`/`C_raw`/`S` embedded in
  the full `orig_T` timeline at `[t0:t1]`; per-component metadata carried;
  `W`/`b0`/`b_f`/`f`/`shifts` dropped (inspection/overlay view). Reads
  `self.cutout`.
- `save()`/`load()`: persist `self.cutout` (add to `manifest.json`).

## Fused AVI→MC + inline concat (`cnmfe/avi_mc.py`, `concat_avis_to_zarr.py`)

Thread the cutout through the decode pipeline (reuse the ssub/tsub plumbing):
- `_decode_avi_worker`: after decoding each frame, crop to `bbox` and zero
  outside `mask_local` **before** the existing `_spatial_bin`/temporal grouping.
- Temporal crop: each decoder knows its file's global offset (pre-scan), so it
  emits only frames whose global index ∈ `[t0, t1)`; per-file output counts and
  zarr offsets are recomputed from the in-range counts (same shape-math as the
  tsub `out_counts`).
- `concat_avis_to_zarr` / `concat_avis_to_mc_zarr`: compute output `(T_out,
  H_bbox//ssub, W_bbox//ssub)` from the cutout + bin; `_build_template_*` crops
  its sample frames too. Per-frame order: **crop → mask → bin → MC**.
- `fit_mc_from_avis(..., temporal_crop=, spatial_crop=, spatial_mask_path=)` (or
  read from `self.params`) records `self.cutout`.

## Phasing
1. **Core (in-memory/zarr fit + map-back):** params fields, `cnmfe/cutout.py`,
   `fit`/`fit_mc` wiring, `place_in_full_fov`, save/load meta, `downscaled()`
   clears crop. Delivers most of the value.
2. **Fused decode crop:** thread the cutout through `_decode_avi_worker` /
   templates / `concat_avis_to_*` (heaviest; the temporal-range-across-files
   offset bookkeeping is the fiddly part).

## Critical files
- `cnmfe/pipeline.py` — `CNMFeParams` fields + `from_json`/`downscaled`; crop in
  `fit`/`fit_mc`; `place_in_full_fov`; save/load meta.
- `cnmfe/cutout.py` *(new)* — `resolve_cutout`, `apply_cutout`,
  `place_footprints_in_fov`, `place_traces_in_timeline`.
- `concat_avis_to_zarr.py` (`_decode_avi_worker`, `_binned_file_frames`, pre-scan
  counts) and `cnmfe/avi_mc.py` (`concat_avis_to_mc_zarr`, template builder) —
  Phase 2.
- Reuse: `cnmfe/_utils.py` `make_2d`/`make_3d`; `scipy.sparse` for padding; the
  `upsample_to_native` pattern as the template for `place_in_full_fov`.

## Verification
- `pytest tests/ -v` — all green; crop is opt-in (`None` defaults = no change).
- New `tests/test_cutout.py`:
  - `resolve_cutout`: rect, mask bbox, rect∩mask, temporal clamp, validation
    errors (out-of-range / empty).
  - `apply_cutout`: numpy + zarr inputs give the right shape; out-of-mask pixels
    zeroed; temporal window correct.
  - End-to-end on a `conftest`/`miniscope_simulator` movie: set a spatial+temporal
    cutout around a subset of ground-truth neurons, `fit()`, assert only the
    in-cutout neurons are found and `model.dims`/`T` equal the cutout size.
  - `place_in_full_fov`: footprints land at the correct `(y0,x0)` offset in the
    original FOV; traces embed at `[t0:t1]`; non-destructive; centroid round-trip
    matches the native location.
  - `CNMFeParams.downscaled` clears the crop fields; `to_json`/`from_json`
    round-trips the tuple fields.
- Phase 2: a fused `fit_mc_from_avis(..., spatial_crop=, temporal_crop=)` on the
  synthetic AVIs in `tests/test_avi_mc.py` yields a cropped `mc.zarr` of the
  expected shape and frame count.
- Manual: `CNMFeParams(spatial_crop=..., temporal_crop=...)` → `fit()` →
  `place_in_full_fov()` overlay on the full-FOV mean image.
