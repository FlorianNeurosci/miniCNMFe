# Quality proxies

Source: `tuning/metrics.py`. These rank the sweep's candidates and feed the
report's PASS/WARN verdict. **They are proxies, not validation** — there is no
ground truth on a real recording, so they only compare candidates and let you
eyeball quality.

## Per-model quality (`model_quality`)

Returns a flat dict of proxies for a fitted model. The notable ones:

- **`cprojcorr_median` / `cprojcorr_mean`** — the central signal: per-cell Pearson
  `r` between the demixed trace `C` and the noisy projection `C + YrA`. In a dense
  FOV this **falls** as cell count rises (YrA cross-talk), so it doubles as a
  density↔purity knob.
- **`spatialcorr_median`** (`per_cell_spatial_corr`) — per-footprint Pearson `r`
  between the footprint values and the local CORR image over the footprint's
  bounding box + surround. A clean single-cell footprint scores high; a merged /
  sprawled footprint spanning several CORR spots scores low. This is the term that
  lets the ranking reject an over-large `sigma` the temporal metric is blind to.
- **`multipeak_frac`** (`_multipeak_frac`) — fraction of (accepted) footprints with
  ≥ 2 distinct soma-scale peaks — the direct spatial signature of `sigma` set too
  large (two cells fused into one component).
- **`npix_median` / `npix_iqr` / `npix_p25`** — footprint pixel-count distribution.
  `npix_p25` (25th percentile) is what the tuner uses to derive the final
  `min_pixel`, measured on the actual nrg-thresholded BCD footprints.
- **`accepted_frac` / `K` / `K_accepted`** — auto-eval accepted fraction and counts.
- **`trace_corr_median`** — median |pairwise Pearson| among accepted traces (high =
  over-split or background bleed across cells).
- **`snr_mean` / `snr_median`** — from the auto-eval `snr_amp`.

## Composite score

`composite_score(q)` is the transparent ranking key used by the sweep — fully
re-derivable from the printed table, **not** an absolute quality claim:

```
score = w_corr·cprojcorr_median + w_spatial·spatialcorr_median + w_acc·accepted_frac
        − w_tight·(npix_iqr/npix_median) − w_merge·multipeak_frac
```

Default weights `{corr:1, spatial:1, acc:0.5, tight:0.25, merge:0.5}`. A `K==0`
candidate scores `-inf`. NaN terms (e.g. no `cn` supplied → `spatialcorr_median`
NaN) contribute 0.

## Blob coverage (the by-eye check, as numbers)

`blob_coverage` encodes "does every bright CORR·PNR blob have a footprint, and
does every footprint sit on a blob?" It matches `detect_cell_blobs` centres
(CORR·PNR `blob_log`, kept where `cn ≥ min_corr` and `pnr ≥ min_pnr`) against
accepted footprint peaks (`footprint_center`), counting a match within
`radius_factor·sigma`:

- **`blob_recall`** — fraction of cell blobs that have a footprint (low → missing
  cells).
- **`footprint_precision`** — fraction of footprints sitting on a blob (low →
  possible ghosts).

## Session verdict

`session_quality_verdict(q, coverage)` turns four checks into `PASS`/`WARN` with
reasons (thresholds in `QUALITY_THRESHOLDS`): `blob_recall ≥ 0.80`,
`footprint_precision ≥ 0.80`, `cprojcorr_median ≥ 0.50`, `trace_corr_median ≤
0.40`. A check whose metric is NaN is skipped, not failed.

## Motion-correction proxies

For ranking MC candidates (see [MC search](mc-search.md)):

- **`mc_registration_quality`** — the **primary** MC signal: `corr_mean`/`corr_p99`
  of the local correlation image (`correlation_image`, CaImAn's `Cn`) — better cell
  co-registration raises neighbour correlation — plus `std_crispness` (gradient
  energy of the temporal-std image).
- **`mc_quality`** — shift-array stats: `shift_smoothness` (mean frame-to-frame
  change), `shift_p99`, `shift_max`.
- **`mc_crispness`** / `crispness` — mean/std-image gradient energy, **diagnostic
  only**: mean-image crispness is dominated by the bright static 1p background and
  *drops* precisely when real motion is removed, so it must **not** rank MC
  candidates.
