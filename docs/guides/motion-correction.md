# Motion correction

Source: `minicnmfe/motion_correction.py`. Entry point:
`motion_correction_rigid(...)`; called from `CNMFe.fit_mc` / `CNMFe.fit`.

The only motion-correction algorithm in the package is **rigid** — whole-frame
translation `(dy, dx)`, no rotation or non-rigid warping. It is built to
reproduce CaImAn's shifts on real 1-photon data, so the specific OpenCV calls
below are load-bearing.

## Per-frame algorithm

Each frame is registered to a fixed **template** in three steps
(`_filter_estimate_apply`):

1. **High-pass filter** (`high_pass_filter_space`). For 1-photon data the broad
   background swamps cross-correlation, so the frame is convolved with a
   sum-to-zero Gaussian kernel of width `gSig_filt`
   (`cv2.getGaussianKernel` → outer product → subtract its mean on the high
   pixels, zero the rest, then `cv2.filter2D` with `BORDER_REFLECT`). The kernel
   side length is `int((3·gSig_filt)//2 · 2 + 1)`. Skipped when `gSig_filt` is
   `None` (2-photon data).

2. **Estimate the shift** (`register_translation_caiman`). Phase
   cross-correlation in the Fourier domain: `cv2`-based FFT of frame and
   template, normalized cross-power spectrum
   `image_product / |image_product|`, inverse FFT, then the peak of `|cross_corr|`
   gives the integer shift. The peak search is **constrained** to `±max_shift`
   pixels by zeroing the interior of the correlation surface. Sub-pixel
   refinement to `1/upsample_factor` px follows via an upsampled DFT around the
   integer peak (`_upsampled_dft`).

3. **Apply the shift** (`apply_shift_caiman`) to the **raw** (un-filtered) frame
   with `cv2.warpAffine` (an affine matrix with translation only,
   `INTER_LINEAR`, `BORDER_REFLECT`), then `nan_to_num` and clip to `≥ 0`.

The shift is estimated on the *filtered* frame but applied to the *raw* frame, so
the high-pass filtering never enters the corrected output.

## Template

The template is a **bin-median** of a strided sample of frames
(`_build_template_streaming` → `caiman_bin_median`): up to
`template_max_frames` evenly-spaced frames are read, high-pass filtered, grouped
into windows of `bin_window` (default 10), mean-reduced within each window, then
median-reduced across windows. Sampling caps template RAM at
`template_max_frames · H · W · 4` bytes regardless of `T`.

With `niter_rig > 1`, each pass rebuilds the template from the *previous pass's
corrected* frames and the per-pass shifts are summed
(`shifts_total += shifts_iter`).

### Template sharpening (`sharpen_template`, default on)

A plain bin-median over the strided sample is **smeared** when the drift is large
relative to the FOV: the sample spans every drift position, so the median lands
near the *mean* position and cross-correlating against that blur pulls each
frame's peak toward zero — a single pass then **under-tracks** (recovers the
right shift *shape* but a compressed *amplitude*). Historically you fixed this
with several `niter_rig` passes, each re-sharpening the template from the
corrected frames — but every pass re-reads and re-writes the whole movie.

`sharpen_template=True` (the **default**; `CNMFeParams.mc_sharpen_template`) does
the sharpening up front and cheaply (`_build_sharpened_template`): it aligns just
the in-RAM frame *sample* to convergence (the in-memory MC loop run on the
sample, no streaming IO) and builds the template from the aligned sample. A
**single** full-movie pass with that template then recovers the full amplitude —
matching a multi-pass run at ~one-pass cost. Because the sample is fixed-size
(`≤ template_max_frames`), the expensive full-movie sweep happens **once**
regardless of recording length, which is the dominant win on long sessions.

Set `sharpen_template=False` (`mc_sharpen_template=False`) for the legacy
smeared-median template — the exact CaImAn-equivalent behaviour. Sharpening is
skipped automatically when an explicit `template` is supplied.

### Early-stop on convergence (`converge_tol`)

When you do run multiple full passes, `converge_tol` (e.g. `0.01`;
`CNMFeParams.mc_converge_tol`) stops the loop early once a pass improves the
template's high-pass sharpness (`_template_sharpness`) by less than that relative
amount — a ground-truth-free convergence check, so `niter_rig` becomes just an
upper bound rather than an exact count. `None` (default) runs exactly
`niter_rig` passes. The sample-sharpening loop above always runs to its own
(fixed-tolerance) convergence, independent of this.

### Supplying a template (`fit_mc`)

`CNMFe.fit_mc(movie, ..., template=..., template_window=...)` lets you bypass the
auto-built template:

- `template` — a precomputed `(H, W)` array to register against (raw; MC
  high-pass filters it internally).
- `template_window=(t0, t1)` — build the template as the mean of frames
  `[t0:t1)`. Pick a *short, low-motion* window; only that slice is read. Mutually
  exclusive with `template`.

Both are also available on `motion_correction_rigid(..., template=...)`.

## Two execution paths (chosen automatically)

`motion_correction_rigid` dispatches on input type and `output_path`:

- **Streaming (`_motion_correction_streaming`)** — used when the input is a
  `zarr.Array` **or** an `output_path` is given. Reads/writes batches of
  `batch_size` frames; peak RAM is
  `(batch_size + template_max_frames) · H · W · 4` bytes, independent of `T`.
  The corrected movie is written to a zarr store. For `niter_rig ≥ 2` it
  ping-pongs between two scratch zarrs and `shutil.move`s the final result into
  place. **Zarr input requires `output_path`** (it refuses to silently load the
  store into RAM).
- **In-memory (`_motion_correction_in_memory`)** — used for a numpy input with
  no `output_path` (the small-movie / test path). Same algorithm, returns a
  numpy array. `in_place=True` warps frames back into the input buffer (~1× peak
  RAM) instead of allocating an output (~2×).

Both paths parallelize the per-frame work within a batch over `n_jobs` (the
frames are independent given a fixed template). Returns `(corrected, shifts)`
where `shifts` is `(T, 2)` float32 `(dy, dx)` per frame.

## Convenience wrappers

- `apply_shift(img, shift)` — alias for `apply_shift_caiman`.
- `estimate_shifts(frame, template, ...)` — `register_translation_caiman` with
  optional high-pass filtering of both inputs.
