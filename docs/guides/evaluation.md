# Auto-evaluation

Source: `minicnmfe/evaluate.py` (`auto_evaluate_components`, `spatial_r_values`)
and `pipeline.py:CNMFe.evaluate`. The final extraction step tags each component
with a quality verdict.

**Nothing is ever dropped.** Evaluation writes a boolean `model.accepted_mask`
`(K,)` and a per-component `model.eval_info`; you filter downstream yourself
(`model.A[:, model.accepted_mask]`, and likewise for `C`/`S`/`YrA`/`C_raw`). This
keeps the step non-destructive and re-runnable on a loaded model to retune
thresholds without re-extracting.

## The two checks

A component is accepted only if it passes **both**
(`auto_evaluate_components`):

1. **Pixel-count floor** — footprint has at least `min_pixel` non-zero pixels.

2. **Mean-amplitude SNR** — scale-invariant, the real discriminator:

   ```
   snr_amp[k] = (‖a_k‖² / npix[k]) / mean( sn[support_k]² )
   ```

   i.e. the mean squared footprint amplitude over its support, divided by the
   mean pixel-noise variance over the same support, thresholded at
   `auto_eval_snr_amp_thr` (default 3.0). A real `sigma=3` Gaussian footprint
   scores ~10–70; a **ghost** component (born from a background-noise seed under
   loose init thresholds) sits at or below ~2 *even when it has many pixels* — so
   a pure pixel-count filter cannot separate them, but this SNR check can. Set the
   threshold to 0 to accept every component on the SNR check.

### The `a_norm` subtlety

`snr_amp ∝ ‖a_k‖²`. But the pipeline's final step relabels footprints to
**unit L2 norm** (amplitude moved into the traces — see
[the overview](index.md)), which would flatten `‖a_k‖²` to 1 and destroy the
discriminator. To avoid that, the original norms are cached on `model.A_norm` and
passed in as `a_norm`, so `‖a_k‖²` is reconstructed as `a_norm[k]²` instead of
read from the now-unit-norm `A`. Passing `a_norm=None` reads directly from `A`
(correct only for un-normalized footprints — old saved models / the unit tests).

## Spatial r-value (separate diagnostic)

`spatial_r_values` is a complementary **spatial** quality metric, not part of the
default accept/reject gate. For each component it builds the activity image
`ΔF = mean(movie over the trace's peak frames) − mean(movie over all frames)` on
the footprint's bounding box plus a `pad`-pixel surround, and Pearson-correlates
that with the footprint values. A clean "bright blob on dark surround" scores
high; a merged or sprawled footprint spanning several activity spots scores low —
catching shape problems the temporal SNR is blind to.
