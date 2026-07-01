# Auto-evaluation

Source: `minicnmfe/evaluate.py` (`auto_evaluate_components`, `spatial_r_values`)
and `pipeline.py:CNMFe.evaluate`. The final extraction step records per-component
quality metrics.

**Report-only by default; the gate is opt-in.** `evaluate()` always writes the
per-component `model.eval_info` (`snr_amp`, `pixel_count`) for inspection, but the
acceptance *gate* is **off by default** (`auto_eval_snr_amp_thr = 0.0`), so
`model.accepted_mask` is all-True — every extracted component is accepted. The
rationale: with seed thresholds (`min_corr`/`min_pnr`) set to the recording's noise
floor you don't get ghosts, so a default gate mostly produces false negatives
(rejecting real dim cells). Control ghosts upstream; raise `auto_eval_snr_amp_thr`
(~`3`) and/or `min_pixel` only to **opt in** to filtering on a noisy recording.

**Nothing is ever dropped.** Even with the gate on, evaluation only writes the
boolean `model.accepted_mask` `(K,)`; you filter downstream yourself
(`model.A[:, model.accepted_mask]`, and likewise for `C`/`S`/`YrA`/`C_raw`). This
keeps the step non-destructive and re-runnable on a loaded model to retune
thresholds without re-extracting.

## The two checks (when opted in)

With the gate enabled, a component is accepted only if it passes **both**
(`auto_evaluate_components`):

1. **Pixel-count floor** — footprint has at least `min_pixel` non-zero pixels.

2. **Mean-amplitude SNR** — scale-invariant, the real discriminator:

   ```
   snr_amp[k] = (‖a_k‖² / npix[k]) / mean( sn[support_k]² )
   ```

   i.e. the mean squared footprint amplitude over its support, divided by the
   mean pixel-noise variance over the same support, thresholded at
   `auto_eval_snr_amp_thr` (default `0.0` = off; use ~`3.0` when opting in). A real `sigma=3` Gaussian footprint
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
