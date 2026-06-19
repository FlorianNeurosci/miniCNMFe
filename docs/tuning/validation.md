# Full-recording validation

Source: `tuning/validate.py` (`validate_session`, `tune_then_validate`,
`good_defaults`) and the `validate_session.py` CLI. The sweep tunes on a fast
cutout; validation confirms the recommendation on the **whole** recording and
produces the diagnostic figures + comparison table you actually judge by eye.

## Why a separate stage

The cutout sweep is a fast approximation — some wins (the rank-1 global
background, the right `init_stride`) only show up over a full long recording.
`tune_then_validate` runs the tuner, then re-extracts the **exact merged
`CNMFeParams` the tuner wrote** (long-recording base + data-driven fields, i.e.
`recommended_params.json`) on the full recording, so the `full/` figures reflect
what you'll apply downstream.

## Share the expensive prefix (`validate_session`)

The key idea: the motion-corrected movie and its pixel-major `Y_flat` store are
**threshold-independent**, so they are built **once** and reused across every
threshold set:

1. **MC once** — fuse the AVIs to `mc.zarr` (`fit_mc_from_avis`, `skip_if_exists`),
   or reuse a provided `mc.zarr`.
2. **`Y_flat` once** — `transpose_zarr_to_pixel_major` to a pixel-major store.
3. **CORR/PNR once** — a shared correlation image for all the footprint overlays.
4. **One extraction per threshold set** — for each `(label, min_corr, min_pnr)`,
   `fit_extract(mc, Y_flat_zarr=yf)` reusing the same `Y_flat` (so each extra
   candidate skips MC + transpose), then `model.save(run_<label>)`.

By default two threshold sets are compared: **`recommended`** (the tuned
thresholds) and **`lowthr`** (a lower-recall set: `min_corr − 0.1`, `min_pnr − 4`,
floored) — so you can see the density↔purity trade-off directly. `--no-lowthr`
runs a single set.

## Per-run diagnostics & comparison

Each run gets the standard figure set (`_diagnostics`): footprints on the CORR
image, traces, footprint-area distribution, cell-consistency, blob coverage, SNR
eval, and MC shifts. It also computes the **faithful** gold-standard per-cell
spatial r-value (`evaluate.spatial_r_values` — footprint vs. the data at the
cell's peak frames) on the full extraction, plus `model_quality`, `blob_coverage`,
and the `session_quality_verdict`. A `comparison.md` table lays the threshold sets
side by side (`K`, accepted, `cprojcorr`, `spatialcorr`, `rvalue`, recall,
precision, npix, SNR, PASS/WARN), and a `summary.txt` per run.

## The `good_defaults` base

`good_defaults(frame_rate_hz, decay_time_ms, …)` is the native-unit long-recording
starting point (rationale in `live_runs/tuning_picast/LEARNINGS.md`):
`global_bg_rank=1`, `min_pixel=60` (floor; SNR does ghost rejection),
`auto_eval_snr_amp_thr=20.0`, the physical-decay `g` prior (`g_prior_weight=0.6`),
`init_stride=2`, `n_iter_main=2`. Validation applies `downscaled(ssub, tsub)` to it
internally so native params run on the (possibly downsampled) grid.

> Note: `good_defaults` sets `auto_eval_snr_amp_thr=20.0`, which diverges from the
> `CNMFeParams` field default of `3.0` — an intentional long-recording override.
> See `todo/doc_comment_code_mismatches.md`.

## Batch & orchestration

`resolve_session_paths` expands a `.txt` list (or several paths) into deduped
session paths. `tune.py --sessions` / `batch_tune.run_batch` run one
`tune.py --validate` subprocess per session in a single BLAS-capped background
process (bounded concurrency), writing a `batch_summary.md`. The `/tune-session`
skill wraps the same flow.
