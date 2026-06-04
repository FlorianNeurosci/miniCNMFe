# Vignetting bias on greedy-init thresholds

## Observation

`pnr_corr_analysis.png` (run on
`tmp/cutout_analysis/` against `mc_detrended.zarr`, section H of
`live_runs/diagnostics.ipynb`) shows a strong **radial brightness
gradient** in both the CORR and PNR maps — bright central band, dim
periphery (bottom-right in particular). The brightness pattern matches
the miniscope LED illumination profile (vignetting), not the underlying
cell distribution.

This biases greedy init via a scalar `min_corr` / `min_pnr`:
- **Centre:** virtually everything clears threshold → over-seeding,
  duplicate seeds per cell, contributing to the 300+ component
  overcrowding seen in `tmp/comp_size/overcrowded.png`.
- **Periphery:** real cells sit below threshold → under-seeding,
  cells get missed.

A single global threshold can't satisfy both regions. The existing
`detrend_movie` (`cnmfe/detrend.py`) addresses *temporal* drift (bleach,
LED warm-up); vignetting is the *spatial* counterpart and is not
addressed anywhere in the pipeline.

## Options (in order of intrusiveness)

1. **Flat-field divide before greedy init.** Compute the per-pixel
   time-mean (already a vignette estimate), Gaussian-smooth it to remove
   cell signatures, divide the movie by it. Cheap (one O(T·H·W) pass,
   reuses the streaming pattern from `detrend_movie`). Acts like the
   per-pixel detrend in the spatial direction. No new pipeline knobs;
   could expose as `flatfield_movie(src, dest, ...)` in a new
   `cnmfe/flatfield.py` and wire as an optional preprocessing step in
   the cutout / live-run notebooks the same way as detrend.
2. **Spatial ROI mask** via the existing `spatial_mask_path` field of
   `CNMFeParams`. User restricts to the well-illuminated central region
   only. Zero code; loses peripheral cells on purpose.
3. **Local-adaptive `min_corr` / `min_pnr`.** Replace the scalar
   thresholds with per-pixel thresholds derived from local CORR/PNR
   quantiles inside `cnmfe/initialization.py`. Real fix, but a pipeline
   change with new params.

Option 1 is the cleanest first attempt. Option 3 is the principled
end-state.

## Verification when implemented

- Re-run `diagnostics.ipynb` section H against the flat-fielded
  movie: the CORR / PNR / CORR·PNR maps should look uniform (no
  bright-centre / dim-periphery gradient) before any seeds are taken.
- Component centroids overlaid in the third panel should distribute
  evenly across the FOV instead of clustering in the bright centre.
- Overall component count should fall to something matching the
  visible bright-soma count in the mean image.
