# Per-knob heuristics

Source: `tuning/heuristics.py`. Each `suggest_*` returns `(value, evidence)` — the
value is the recommended parameter, the evidence dict carries the arrays a report
figure draws from. No matplotlib here, so the functions stay importable in
headless workers. They reuse the pipeline's own primitives (`correlation_pnr`,
`greedy_corr_pnr`, `estimate_shifts`, `estimate_ar_params`) rather than
reimplementing them.

## Motion-correction stage (raw AVI sample)

- **`suggest_mc_gsig_and_sigma`** — the neuron radius. Take the **temporal-std**
  projection of the sample (neurons flicker, background drifts), optionally
  subtract a wide Gaussian blur (`highpass_sigma=8`) to remove out-of-focus haze /
  vignetting, normalize, and run `skimage.feature.blob_log`. The **median radius**
  of the top-scoring blobs becomes `mc_gSig_filt` and seeds the extraction `sigma`.
  The high-pass step matters: on a hazy FOV `blob_log` otherwise latches onto
  broad background and the radius inflates, blowing up `sigma`/`ssub`/`min_pixel`
  and sprawling footprints.
- **`suggest_max_shift`** — register a sample of frames against the median image
  with a generous probe range, then set `max_shift` to the **99th-percentile**
  absolute shift per axis plus a 2 px margin. `border_px` = the max of the two
  (trims `warpAffine` fill artifacts).
- **`suggest_downsample`** — `ssub`/`tsub` from two rules: keep neuron FWHM
  (`2.355·sigma`) ≥ `min_fwhm` (4 px) on the binned grid; keep the binned frame
  period ≤ `decay_time_ms / 2` (sample the rising edge at least twice).

## Initialization stage (`mc.zarr` sample)

- **`suggest_sigma_extraction`** — refit `sigma` for extraction: `blob_log` on the
  normalized **CORR·PNR product** image, median radius of the top blobs. Returns
  the CORR/PNR images so the threshold heuristics can reuse them.
- **`min_corr` / `min_pnr`** — three independent methods, all scored against each
  other in the sweep:
    - **`suggest_corr_pnr`** (morphology) — sweep each image's threshold and count
      **cell-like** connected components (soma-sized area from `sigma`, solidity
      `> 0.85`); pick the threshold that **maximizes** that count — "most blobs
      visible = background mesh gone, cells not yet lost." Mirrors raising the
      `vmin` slider in CaImAn by eye.
    - **`suggest_corr_pnr_separation`** — detect neuron blobs on CORR·PNR, then for
      each image pick the threshold maximizing **Youden's J** (TPR − FPR) between
      values at neuron centres and at background pixels.
    - **`suggest_corr_pnr_percentile`** — the `pct`-th percentile (default 25) of
      CORR/PNR **at detected neuron centres** — a robust "keep ~75% of neurons"
      operating point.
  Each falls back to safe defaults (0.8 / 10.0) when too few neurons are detected.
- **`suggest_min_pixel`** — run a fast greedy init, count pixels above
  `peak_frac·peak` per footprint, take the `pct`-th percentile (default 25). (The
  tuner ultimately prefers the winning sweep candidate's realized `npix_p25`
  instead — greedy-init footprints don't see the nrg thresholding and over-estimate
  this.)

## Temporal / merge / eval stage (fitted model)

These read off the best fitted model:

- **`suggest_decay_time`** — median per-component Yule-Walker τ (no prior, no
  shrinkage). *Diagnostic only* — on long recordings this is drift-inflated, so the
  recommendation keeps the physical indicator τ.
- **`suggest_g_prior_weight`** — from the spread of the per-component YW `g` around
  the physical target `g_target`: tight cluster → 0.3, moderate → 0.5, wide /
  drift-heavy → 0.7.
- **`suggest_merge_thr`** — `min(0.85, max(99th-pct pairwise C_raw correlation,
  0.7))` so modestly-correlated real neighbours aren't swept up.
- **`suggest_snr_thr`** — `auto_eval_snr_amp_thr` as the centre of the **largest
  gap** among components scoring `< 10` (the ghost↔real boundary; real neurons
  score 10–70, ghosts cluster below ~2).
