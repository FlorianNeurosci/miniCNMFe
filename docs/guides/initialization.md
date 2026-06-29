# Initialization (greedy CORR-PNR)

Source: `minicnmfe/initialization.py`. Entry: `greedy_corr_pnr(...)` (and the
parallel `greedy_corr_pnr_patched`). Produces the first `A`, `C`, raw traces, and
neuron centres that the BCD loop then refines.

Initialization is **greedy**: find the best remaining seed, extract a neuron
there, subtract it from the movie, repeat. Each extraction changes the residual
the next seed sees, which is why this stage is inherently sequential.

## Setup (once)

1. **Filter** every frame with the [center-surround PSF](seeds-corr-pnr.md) via
   `cv2.filter2D` → `data_filtered`. Keep an unfiltered copy `data_raw`.
2. **Center** each pixel in time and estimate the per-pixel noise `noise_pixel`
   **once** (`estimate_noise`).
3. **Initial CORR and PNR**, and the search score `v_search = CORR · PNR`
   (zeroed where `CORR < min_corr` or `PNR < min_pnr`).
4. **`ind_search`** — a persistent `(H, W)` boolean mask of pixels that will not
   be tried as seeds. Initialized from sub-threshold pixels and the `border_px`
   margin.

## The greedy loop

Each outer pass rebuilds the sorted seed list with `peak_local_max` over the
current `v_search`. For each candidate seed (highest score first):

1. **Skip** if already in `ind_search`; otherwise mark it tried.
2. **Diff-noise guard** — reject pure-noise pixels whose
   `max(diff(trace)) < 3 · std(diff(trace))`.
3. **Extract footprint + trace** on a `gSiz = max(3·sigma, 5)` patch
   (`extract_spatial_temporal`, below). Reject if it fails or has fewer than
   `min_pixel` pixels.
4. **Deconvolve** the trace with OASIS (`estimate_ar_params` + `deconvolve`) to a
   clean calcium trace `c_clean`.
5. **Subtract** the component from both movies:
   - `data_raw`: subtract `ai · c_clean` on the extraction box.
   - `data_filtered`: subtract the **PSF-refiltered** footprint
     `cv2.filter2D(ai) · c_clean` on a `2·gSiz` halo — so the filtered residual
     stays clean and the soma's PSF sidelobes don't re-seed ghosts.
6. **Suppress** the neuron's own support: mark `ai > ai.max()/2` pixels in
   `ind_search` (CaImAn-style support mask). *(The `seed_suppress_factor` /
   suppression-disk parameter is deprecated and ignored.)*
7. **Refresh CORR/PNR locally** on the `2·gSiz` box against the cached
   `noise_pixel` (`_local_cn_pnr_box`) and update `v_search` there.

The loop stops when a pass makes no progress, no seeds remain, or `max_neurons`
is reached.

## Single-component extraction

`extract_spatial_temporal` turns one seed into one `(footprint, trace)` on its
patch:

1. Unit-normalize each pixel trace in the patch; correlate every pixel with the
   **seed pixel's** trace.
2. **Neuron pixels** = correlation `> min_corr_neuron` (default 0.8); their mean
   (filtered) trace is the temporal estimate `ci`. **Background pixels** =
   correlation `< max_corr_bg` (default 0.4); their median (raw) trace is a local
   background regressor.
3. Solve a **3-component OLS** `patch_raw ≈ [ci, y_bg, 1] · coef`; the footprint
   is `ai = coef[0]` clipped to `≥ 0`.
4. Apply the shape constraints: `circular_constraint` (zero pixels farther than
   `circular_max_dist_factor · √(area/π)` from the centroid) and
   `connectivity_constraint` (keep the largest connected component).
5. Baseline-subtract `ci` (median of its near-flat samples).

## Patch-parallel variant

`greedy_corr_pnr_patched` recovers parallelism that the serial greedy loop can't
offer: it tiles the FOV into **overlapping** patches (`_tile_grid`), runs the
unchanged `greedy_corr_pnr` on each in parallel **processes**, remaps each
patch's footprints/centres to global coordinates, concatenates them, and
**de-duplicates** neurons found in two overlapping patches with
`merge_components` (their copies have near-identical traces and close centres).
`border_px` and `max_neurons` are applied globally after the merge. The pipeline
uses this path for the single full-FOV init when the movie is in RAM, the FOV is
large enough (`init_patch_min_fov`), and the device is CPU.
