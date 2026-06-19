# Motion-correction parameter search

Source: `tuning/mc_tune.py` (orchestration) and `tuning/mc_search.py` (the search).
This runs **before** downsampling / MC, choosing `mc_gSig_filt`, `max_shift`, and
`mc_n_iter` for a raw AVI session by **scoring corrected clips**, not by heuristic
guess alone. It is the counterpart to the [extraction tuner](index.md): MC is
chosen on its own merits first, then extraction is tuned on the resulting
`mc.zarr`.

## Why scoring, not just a heuristic

`suggest_mc_gsig_and_sigma` / `suggest_max_shift` give a *seed* for the high-pass
width and shift range, but nothing checks that the resulting registration is
actually sharp. When `mc_gSig_filt` under-suppresses the slow 1p background, the
phase-correlation latches onto background structure instead of cells and the
corrected movie jitters. So the search turns the single guess into a
**seed + trial-and-score** loop.

## The search (`mc_search.search_mc_params`)

Run a handful of short rigid MCs on **one representative contiguous clip**
(`io_sample.decode_contiguous_clip` — motion is continuous, so quality can only be
judged on successive frames) over a grid, and keep the candidate whose corrected
clip has the best **cell-focused registration quality**.

- **Ranking metric** = mean local correlation image (`mc_registration_quality`'s
  `corr_mean`) — better cell co-registration raises neighbour correlation.
  Mean-image crispness is deliberately **not** used: on 1p data it is dominated by
  the static background and rewards under-correction (see [metrics](metrics.md)).
- **Absolute grid, not seed-multiples** (`McSearchSpec`): `gsig_values ∈ {3…16}`,
  `max_shift_values ∈ {4,8,16}`, `n_iter_values ∈ {1,2}`. The grid is absolute on
  purpose — an unreliable seed `max_shift` can collapse to a tiny value that
  *hard-constrains* phase-correlation (it zeros cross-correlation beyond
  `max_shift`) and blinds the search to real motion. The heuristic seed is still
  tried as one extra candidate (`include_seed`).
- **Coarse-to-fine**: sweep `gSig` at the most generous `max_shift` (so motion is
  visible while choosing the filter), fix the best, then sweep `max_shift`, then
  `n_iter`. Combos are memoized so shared trials run once.
- **Guardrails** (`_pick`): a candidate that **saturates** `max_shift` (99th-pct
  shift ≥ `saturation_frac·max_shift` — the clipping/jitter signature) or fails to
  **beat the raw clip** (`corr_mean` not above the uncorrected baseline) is never
  selected over a clean one.

Returns `(best_params, rows, evidence)` — `best_params` = `{mc_gSig_filt,
max_shift, mc_n_iter}`, `rows` = every candidate (scores + guardrail flags +
`is_best`) for storage and the report figure.

## Orchestration (`mc_tune.run_mc_tuning`)

1. Estimate the seeds in **native** units (`suggest_mc_gsig_and_sigma`,
   `suggest_max_shift`, `suggest_downsample`).
2. Decode one contiguous high-activity clip and block-mean it to the MC **grid**
   (`ssub`/`tsub`), since MC runs on the downsampled movie.
3. Convert the seeds to grid units and hand the clip to `search_mc_params`.
4. Write `mc_tuning.json` + a one-page before/after correlation-image PNG.

Returns `ssub`/`tsub` + grid-unit `mc_params` (`mc_gSig_filt`, `max_shift`,
`mc_n_iter`, `border_px`) plus per-candidate rows, sources, and rationale.
