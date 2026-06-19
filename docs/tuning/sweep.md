# Extraction sweep

Source: `tuning/sweep.py`. The sweep actually runs `CNMFe.fit_extract` across a
small grid of the most impactful extraction knobs and scores each result with the
[ground-truth-free proxies](metrics.md), so the recommendation reflects real
output quality rather than image heuristics alone.

## Swept knobs

`SWEPT_FIELDS = (sigma, min_corr, min_pnr, merge_thr_corr, global_bg_rank,
init_stride)`. A `SweepSpec` holds a value list per knob (`None` = hold at the
base value).

## Two regions

Set by `region_crop`:

- **cutout** — a `(spatial_crop, temporal_crop)` window (`io_sample.pick_cutout`);
  each candidate runs on the cropped movie in RAM. Fast (seconds–minutes per
  candidate), the default.
- **full** — `None`; each candidate streams the whole `mc.zarr` through the
  pixel-major `Y_flat` path. Faithful, slow.

## Anchoring the grid to the data (the "Offset DSL")

A static grid (e.g. `sigma ∈ {3,4,5}`) can sit entirely away from the
data-driven value and never test it. `resolve_offset_grid` instead expresses the
grid **relative to a data anchor** (the measured radius, or a detected
threshold), so the anchor is always one of the candidates. A spec can be `None`
(offsets around the anchor), `{"around": [...], "extra": [...]}`, or absolute
values with the anchor injected; everything is clamped, deduped, sorted.

## Coupled threshold seeds

The CORR/PNR images depend on `sigma` (the PSF width), so `min_corr`/`min_pnr`
estimated at one `sigma` are wrong at another. The tuner therefore, **for each
`sigma` in the grid**, recomputes CORR/PNR and runs all three threshold methods
(morphology / separation / percentile), producing **coupled** seeds
`{sigma, min_corr, min_pnr, thr_method}`. `build_candidates` crosses these
`thr_seeds` with the remaining knob combos (`thr_seeds × base_combos`), so each
seed sets `min_corr` *and* `min_pnr` together — and the sweep scores the three
methods head-to-head. To keep cost bounded, the non-seed knobs use the full
Cartesian product only if it fits the budget, else a **one-knob-at-a-time** design
around the base.

## Scoring & ranking

Each candidate (`_run_one_candidate`, a module-level loky worker) is fit, scored
with `model_quality` + `composite_score`, and rendered to footprint/trace PNGs.
A failed candidate returns a row with `score = -inf` rather than aborting the
sweep. Rows are sorted by score (best first). `composite_score` is a transparent,
re-derivable combination (see [metrics](metrics.md#composite-score)):

```
score = w_corr·cprojcorr_median + w_spatial·spatialcorr_median + w_acc·accepted_frac
        − w_tight·(npix_iqr/npix_median) − w_merge·multipeak_frac
```

The **spatial** term is load-bearing: the temporal `cprojcorr` is nearly blind to
an over-large `sigma` that fuses neighbours into bigger "clean"-looking blobs, so
without `spatialcorr` the ranking would over-pick the merged large-`sigma`
candidate.

## Parallelism budget

`n_jobs` is the **total core budget**, split across two levels so their product
doesn't oversubscribe: candidates run as `cand_jobs` loky **processes**, each
`fit_extract` runs `inner_jobs` **threads** (`cand_jobs · inner_jobs ≈ budget`,
BLAS pinned to 1 per process). The tuner forces `init_patches=False` on the swept
candidates (loky-in-loky would serialize), except the single best-model re-fit in
the parent, which gets the full budget and patch-parallel init.

`run_sweep` returns `(rows, best_params, best_model)` — the best candidate is
re-fit once in the parent so the stage-4 heuristics have a live model to read.
