# CLAUDE.md — Project context for Claude Code

This file is read automatically at the start of every Claude Code session.
It captures project state, decisions, and caveats that are not obvious from the code alone.

---

## What this project is

A **clean Python reimplementation of CNMFe** (Constrained NMF for Endoscopic data) for
extracting neurons from 1-photon calcium imaging (miniscope) recordings.

**No CaImAn code is imported.** The CaImAn repository (`CaImAn-main/`) is present as an
algorithmic reference only. All math is reimplemented from scratch with numpy/scipy/sklearn.

---

## Current status

**Complete and working.** All 88 tests pass:

```bash
pytest tests/ -v
```

Major work done so far:
- Full pipeline: motion correction → CORR/PNR → greedy init → ring background → spatial/temporal refinement → merging
- `n_jobs` CPU parallelism via joblib (every bottleneck step)
- `device="cuda"` GPU acceleration via CuPy (opt-in, falls back gracefully)
- Algorithm bugs fixed: greedy-init over-detection, silent-merge failure, temporal-trace AR-coefficient drift, `avi_to_zarr` imageio v3 shape (see *Bugs already fixed* below)
- Two trace flavours exposed: `model.C` (OASIS-deconvolved) and `model.C + model.YrA` (noisy projected trace)
- Realistic miniscope simulator (`tests/miniscope_simulator.py`) with multi-component drifting bg, ghost cells, vasculature, vignetting, photobleaching, shot noise, 8-bit quantisation, and optional inter-frame motion drift (`motion_max_shift` param)
- Four tutorials: `tutorial.ipynb`, `tutorial2.ipynb` (clean rewrite), `tutorial_realistic.ipynb` (uses the realistic simulator), `tutorial_caiman_compare.ipynb` (CaImAn side-by-side, requires CaImAn), and `tutorial_demo.ipynb` (realistic lazy-load AVI workflow)
- Demo movie generation + AVI-to-zarr workflow: `generate_demo_movies.py`, `convert_to_zarr.py`, `concat_avis_to_zarr.py`
- CLI pipeline runner: `full_pipeline.py` (loads zarr lazily, runs full pipeline, saves all results to disk)
- Obsidian wiki (`wiki/`) updated for all of the above

---

## Tech stack

| Concern | Choice | Notes |
|---------|--------|-------|
| Movie storage | zarr v3 | Time-chunked, lazy, random-access |
| Deconvolution | `oasis-deconvolution` package | Pure-Python PAVA AR(1) fallback if not installed |
| CPU parallelism | `joblib` / `loky` | All parallel workers defined at module level (pickling requirement) |
| GPU | `cupy` (optional) | `get_xp(device)` in `_utils.py` returns numpy or cupy |
| Spatial LASSO | `sklearn.linear_model.LassoLars` | CPU-only, no GPU equivalent |
| OASIS | sequential PAVA | Cannot be GPU-accelerated (inherently sequential) |
| mp4 export | `opencv-python` (cv2) preferred, imageio with explicit codec as fallback | imageio's `pyav` plugin without an explicit codec fails on Windows envs |
| Tests | pytest | Synthetic ground-truth movies in `tests/conftest.py` and `tests/miniscope_simulator.py` |

---

## Key design decisions

### Motion correction — canonical implementation
`motion_correction_rigid` in `cnmfe/motion_correction.py` is the **only** motion
correction algorithm. It uses `cv2.filter2D` for high-pass filtering (matching
CaImAn's `border_reflect`) and `cv2.warpAffine` for applying shifts. Both choices
are critical for producing the same shifts as CaImAn on real data — using
`scipy.ndimage.convolve` or FFT-based shift application produces a ~4–5 px absolute
offset compared to CaImAn even when correlation is high. Do not replace these with
scipy equivalents.

The function has **two execution paths**, chosen automatically:

- **Streaming (zarr-backed)** — used when the input is a `zarr.Array` *or* an
  `output_path` is given. Reads/writes batches of frames; peak RAM is
  `(batch_size + template_max_frames) * H * W * 4` bytes, independent of T.
  Template is built from a strided sample of up to `template_max_frames` frames.
  For `niter_rig > 1` we ping-pong between two scratch zarrs
  (`<name>.scratch_a.zarr`, `<name>.scratch_b.zarr`) and `shutil.move` the final
  result to `output_path`. Zarr input **requires** `output_path` (we refuse to
  silently load it into RAM).
- **In-memory** — used when the input is a numpy array and no `output_path` is
  given. Same algorithm, returns a numpy corrected movie. Kept for the small-movie
  test path.

Both paths parallelize per-frame work via joblib (`n_jobs`). The worker
`_filter_estimate_apply` is module-level for `spawn`-based pickling on Windows.
Within a batch, frames are independent given the fixed template, so they
parallelize trivially.

CNMFeParams MC fields:
- `mc_batch_size: int = 200` — frames per streaming/parallel batch
- `mc_template_max_frames: int = 2000` — cap on frames sampled for the template
- `mc_output_chunk_t: int | None = None` — output zarr time chunk (None = match source)
- `mc_output_dtype: str = "float32"` — output zarr dtype

`fit_mc(movie, output_dir=...)` is the canonical entry for big movies: pass a
`zarr.Array` + `output_dir`, the corrected movie is written to
`<output_dir>/mc.zarr` without materializing T frames in RAM.

**Extraction RAM after the streaming refactor (Items 1A–1D, May 2026).**
Everything *on top of* `Y_flat` is streamed:
- `BackgroundSubtractor` (`background.py`) materialises pixel-row slices
  of `(I - W) @ (Y - b0)` on demand. The full `Y_bg` is never built.
  Works on numpy or zarr-backed `Y_flat` (the zarr branch extracts only
  ring-neighbour rows via advanced indexing).
- `compute_W` computes `b0` via streaming reductions
  (`(Y_sum - A @ C_sum) / T`) and builds X_fit only at the subsampled time
  resolution. After the first call it accepts `W_cached=W_mat` and only
  refreshes `b0` — saving the per-pixel BTB solve on subsequent BCD
  iterations. For zarr-backed `Y_flat` it builds X-slabs per pixel batch
  so the full `(H·W, T_sub)` residual is never materialised.
- Greedy init runs on a strided sample (`init_stride` field on `CNMFeParams`;
  auto = `max(1, T // 5000)`); full-T temporal traces are recovered by
  projecting `Y_flat` onto each footprint after init.

**Two RAM tiers (Item 5 / Phase F, May 2026):**

- **In-memory** — `fit(numpy_or_zarr)` materialises `Y_flat = make_2d(movie)`.
  Peak ≈ T·H·W·4 bytes. Good for 10k × 600 × 600 (~14 GB).
- **True T-streaming** — `transpose_zarr_to_pixel_major(mc.zarr, mc_pixel.zarr)`
  once, then `fit(mc_zarr, do_motion_correction=False, Y_flat_zarr=mc_pixel_zarr)`.
  `Y_flat` is the on-disk pixel-major store; the 3D zarr is read only for
  the strided init sample. Peak RAM independent of T; bounded by
  `K·T·4` (traces) + per-batch buffers. Unlocks 60k+ frame recordings.

The transpose is a one-time disk pass; pixel ordering matches `make_2d`
(pixel `(h, w)` → flat `h*W + w`).

Convenience wrappers added alongside `motion_correction_rigid`:
- `apply_shift(img, shift)` — alias for `apply_shift_caiman`
- `estimate_shifts(frame, template, ...)` — thin wrapper around `register_translation_caiman`

**ROI-based shift estimation (design note — not currently implemented):**
A `select_roi` function was prototyped that finds the neuron-dense rectangular
sub-region of a frame for use in shift estimation (shift estimated from ROI,
applied to full frame). Algorithm:
1. Temporal-std projection on a subsampled movie (neurons flicker; background drifts slowly).
2. Spatial high-pass (subtract Gaussian blur) to remove broad gradients.
3. Optional LoG blobness filter to favour blob-shaped structures over vessels.
4. Mask frame borders to avoid edge artefacts.
5. Find the crop of size `(frac_h·H, frac_w·W)` with the highest total score via an
   O(H·W) integral-image (vectorised sliding-window sum, no Python loop).
This approach can reduce the influence of vasculature and slow background on shift
estimation. If re-implementing, apply the filter to the crop before cross-correlation,
and apply the resulting shift to the full (uncropped) frame.

### `device` / `n_jobs` pattern
- `get_xp(device)` in `_utils.py` — returns `numpy` or `cupy`; call at function entry
- `to_numpy(arr)` in `_utils.py` — always returns numpy; handles cupy transparently
- GPU-capable functions take `device: str = "cpu"` parameter
- CPU-parallel functions take `n_jobs: int = 1` parameter
- Both are threaded through `CNMFeParams` → `CNMFe.fit()` → every downstream call
- `spatial.py` has no GPU path (LassoLars); `temporal.py` GPU only covers the projection, not OASIS

### Module-level worker functions
All functions dispatched by `joblib.Parallel` are defined at module top level (not as
lambdas or nested functions). Required for `spawn`-based pickling on Windows.
Workers: `_shift_and_correct_frame`, `_ring_pixel_batch`, `_spatial_pixel_batch`,
`_deconvolve_with` (in `temporal.py` — replaced the older `_deconvolve_one`; takes pre-computed `g`/`sn` so it doesn't re-estimate per call).

### Flat pixel representation
After initialization the movie is stored as `(H·W, T)` — pixels as rows, time as columns.
`make_2d` / `make_3d` in `_utils.py` handle reshaping.

### AR-coefficient `g` is estimated ONCE per pipeline run, not per BCD iteration
- `pipeline.fit()` estimates `g` from `C_raw.ravel()` (pooled across components for robustness on short traces) right after init, plus per-component `sn` from each `C_raw[k]`. Stored on `self.g` and `self.sn_per_k`.
- The cache is threaded into every `update_temporal(..., g_cached=..., sn_cached=...)` call.
- After `merge_components` reorders K, `_cache_after_merge(members_per_group)` updates the cache by inheriting from `members[0]` — no re-estimation, no drift.
- Re-estimating from a deconvolved trace re-applies `fudge_factor=0.96` and drifts `g` toward 0 across iterations. **Do not re-introduce per-iteration estimation.**

### Two trace flavours: `C` vs `C + YrA`
- `model.C` — OASIS-deconvolved (clean AR(1) shape). Use for spike-event detection.
- `model.YrA` — residual at each footprint after the final BCD pass.
- `model.C + model.YrA` — noisy *projected* trace; preserves the data's actual shape. Pearson r against ground truth typically ≥ 0.94 here vs ~0.6–0.85 for `C` alone, because OASIS's `c[t] >= g·c[t-1]` constraint introduces small spike-timing distortions.
- Use `C + YrA` for shape-faithful comparisons (cross-correlation with an external signal, regression, plotting raw fluorescence).

### Merge rule: temporal AND (spatial overlap OR centre proximity)
`merge_components` merges component pair (i, j) when:
```
|Pearson(C[i], C[j])| > thr_corr  AND
( Jaccard(i, j) > thr_overlap  OR  centre_dist(i, j) < merge_centre_dist_factor * sigma )
```
The centre-distance fallback catches duplicates whose post-`threshold_footprint` supports
have ended up disjoint despite tracking the same neuron. **Do not regress to AND-only.**

### `merge_components` does NOT re-deconvolve internally
It returns the mean trace (clipped non-negative) and lets the caller's next
`update_temporal` pass deconvolve with the cached `g`. Re-deconvolving inside
`merge_components` would re-introduce fudge-factor drift.

### `update_temporal` returns a 4-tuple
`(C, S, g_per_k, sn_per_k)`. Callers must unpack four values; tests have been
updated. **Do not regress to a 2-tuple.**

### `merge_components` returns a 4-tuple
`(A_merged, C_merged, n_merged, members_per_group)`. The pipeline uses
`members_per_group` to keep the AR cache aligned with the new component order.

### Pre-spatial merge on iteration 0
`pipeline.fit()` runs an extra `merge_components` pass *before* the first
`update_spatial`, on `C_raw`, to fuse duplicate seed detections while their
footprints still overlap (before `threshold_footprint` separates them).

### Auto-evaluation step (post-BCD quality filter)
`pipeline.fit()` runs `cnmfe.evaluate.auto_evaluate_components` between the
BCD loop and the final `update_temporal`. Two per-component checks must
both pass:
1. **Pixel-count floor:** `npix >= CNMFeParams.min_pixel`.
2. **Mean-amplitude SNR:** `(||a||² / npix) / mean(sn_pixel²) >= auto_eval_snr_amp_thr` (default `3.0`).

The SNR check is the real discriminator and is **scale-invariant** — real σ=3 Gaussian
footprints score 10–70, ghost components born from background-noise seeds (loose init
thresholds, e.g. `min_corr=0.7, min_pnr=3.0`) sit at or below 2 *even when their pixel
count is large* (ghosts can be wide, low-amplitude blobs that survive `threshold_footprint`
because they're a connected component of pixels each above 10 % of the ghost's own peak).
**Pure pixel-count filtering does not separate real from ghost components in this codebase**
— don't try to replace the SNR check with a fixed-area threshold. Set
`auto_eval_snr_amp_thr=0.0` to disable; `min_pixel` continues to apply.

Cache cleanup follows the existing `alive = nA > 0` pattern — `A`, `C`, `g_per_k`,
`sn_per_k` are filtered in lockstep before the final temporal update. The regression
test is `tests/test_pipeline.py::test_auto_evaluation_rejects_ghosts`.

### zarr v3 API
The project uses zarr v3 (`zarr >= 3.0`). The v3 API differs from v2 in chunk
specification and store opening. Do not regress to v2 patterns.

---

## Non-obvious bugs that were already fixed — do not re-introduce

### Floor division for odd image dimensions
`np.arange(-H // 2, ...)` is wrong for odd H because Python floors toward −∞
(`-31 // 2 = -16`, giving 32 elements instead of 31).
**Fix:** `np.arange(-(H // 2), H - H // 2)` — negate AFTER integer division.
Applied in `preprocess.py:local_correlations_fft` and `motion_correction.py:apply_shift`.

### Infinite loop in `greedy_corr_pnr`
After extracting a neuron and subtracting it, the local CORR/PNR update can regenerate
seeds at neighbouring pixels of the same neuron indefinitely.
**Fix:** after each local update, zero a disk of radius
`max(int(seed_suppress_factor * sigma), int(2*sigma + 1))` (was `max(2, int(sigma))`)
around every already-found centre. See `initialization.py` ~line 360.

### Greedy-init over-detection
Tutorial cell 25 used to extract 56 neurons from 6 ground-truth.
Two compounding causes:
1. Suppression disk was 3 px for σ=3, smaller than the neuron FWHM → residual halo re-seeded the same neuron.
2. `extract_spatial_temporal` thresholds were too tight (`min_corr_neuron=0.9`) → undershot footprint → incomplete subtraction → big halo.
**Fix:** suppression scales with `seed_suppress_factor` (default 2.0); thresholds loosened to `min_corr_neuron=0.8`, `max_corr_bg=0.4`; `circular_constraint` cutoff exposed as `circular_max_dist_factor` (default 2.5).

### Silent merge failure (`0 merged`)
`threshold_footprint(max_thr=0.1)` keeps only the largest connected component, which
made duplicate detections of the same neuron end up with disjoint supports. The merge
mask `(jaccard > 0.5) AND (|R| > 0.85)` always failed on the spatial side.
**Fix:** added the centre-distance fallback (see *Merge rule* above). Pre-merge on
iteration 0 also runs before footprints are separated.

### Temporal trace AR-coefficient drift
`update_temporal` used to call `estimate_ar_params(trace_k)` per component per BCD
iteration. `estimate_ar_params` always multiplies by `fudge_factor=0.96`, so re-estimating
from already-deconvolved traces shrunk `g` each call. With `n_iter_main=2` and
`n_iter_temporal=2`, true `g=0.9` drifted to ~0.76, dropping mean Pearson r against
ground truth from the expected ~0.94 to ~0.76.
**Fix:** estimate once from pooled `C_raw`, persist on `model.g`/`model.sn_per_k`,
thread cache through every `update_temporal` call. Inherited via `members[0]`
after merging.

### `avi_to_zarr` imageio v3 shape extraction
`iio.improps(src).shape` in imageio v3 returns `(T, H, W)` (full video shape), not `(H, W)`.
Using `props.shape[:2]` silently extracted `(T, H)` as the spatial dimensions, creating a zarr
with shape `(T, T, H)` and failing with a broadcast error when the first chunk was written.
**Fix** (in `cnmfe/io.py`): check whether `_s[0] == T` and extract H, W from `_s[1:]` if so,
otherwise fall back to `_s[0:2]`. The same fix applies to `concat_avis_to_zarr.py`.

---

## File structure

```
cnmfe/                         Main package
  _utils.py                    make_2d, make_3d, get_xp, to_numpy, iter_frames, ensure_float32
  io.py                        avi_to_zarr, open_zarr, save_zarr
  motion_correction.py         motion_correction_rigid, apply_shift, estimate_shifts
  preprocess.py                make_center_surround_psf, estimate_noise, correlation_pnr
  background.py                build_ring_indices, compute_W, subtract_background
  initialization.py            detect_seeds, extract_spatial_temporal, greedy_corr_pnr
  spatial.py                   compute_support, threshold_footprint, update_spatial
  temporal.py                  estimate_ar_params, deconvolve, update_temporal, _deconvolve_with
  merging.py                   merge_components  (4-tuple return)
  pipeline.py                  CNMFeParams (dataclass), CNMFe.fit()
tests/
  conftest.py                  make_synthetic_movie() — clean fixture; supports motion_max_shift
  miniscope_simulator.py       make_miniscope_movie() — realistic 1p movie with bg/ghosts/vasc/bleach/shot noise/8-bit/motion
  test_multiprocessing.py      n_jobs correctness tests
  test_pipeline.py             includes test_temporal_correlation_against_truth (regression for the AR drift fix)
wiki/                          Obsidian docs (math, eli5, architecture, api-reference, usage-guide)
demo_movies/                   Generated AVI files + _meta.npz sidecars + .zarr stores (created by scripts below)
generate_demo_movies.py        Generate demo_movies/*.avi with ground-truth NPZ sidecars (idempotent)
convert_to_zarr.py             Batch-convert demo_movies/*.avi -> *.zarr (idempotent)
concat_avis_to_zarr.py         Concatenate a folder of 0.avi ... N.avi into one zarr store (CLI)
full_pipeline.py               CLI: load any zarr lazily, run full CNMFe pipeline, save A/C/S/YrA/shifts/sn/params to disk
tutorial.ipynb                 Original walkthrough (preserved)
tutorial2.ipynb                Clean rewrite of the original tutorial
tutorial_realistic.ipynb       Tutorial on the realistic simulator + mp4 export of the simulated movie
tutorial_caiman_compare.ipynb  Side-by-side CaImAn vs our CNMFe (requires CaImAn installed separately)
tutorial_demo.ipynb            Realistic-use demo: AVI -> zarr -> lazy load -> full pipeline -> visualise
CaImAn-main/                   Reference source only — never import from here for production
todo/speedup.md                Implementation guide for future speed improvements (skip OASIS on first pass, cache W)
```

`CNMFeParams` fields (excerpt of the params added or made adjustable in this round):
- `init_min_corr_neuron: float = 0.8` (was hardcoded 0.9)
- `init_max_corr_bg: float = 0.4` (was hardcoded 0.3)
- `seed_suppress_factor: float = 2.0` — controls greedy-init suppression disk size
- `circular_max_dist_factor: float = 2.5` — `circular_constraint` cutoff
- `merge_centre_dist_factor: float = 2.0` — centre-distance fallback for `merge_components`
- `global_ar: bool = False` — `True` = one `g` estimated from pooled `C_raw`; `False` = per-neuron `g` from each `C_raw[k]`. Both modes estimate once from raw traces and cache; neither re-estimates from deconvolved traces.

`make_miniscope_movie` / `make_synthetic_movie` parameters added (in `tests/`):
- `motion_max_shift: float = 0.0` — peak drift amplitude in pixels; 0 = no motion (backward-compatible)
- `motion_seed: int | None = None` — RNG seed for drift (defaults to `seed + 1`); `make_synthetic_movie` always uses `seed + 1`
- Drift is a smoothed correlated random walk (cumsum of small Gaussian steps, uniform_filter1d, rescaled to peak = `motion_max_shift`)
- Applied frame-by-frame via `cnmfe.motion_correction.apply_shift`; stored as `result["motion_shifts"]` (T, 2) float32
- Sign convention: `motion_shifts[t]` is the `(dy, dx)` shift applied to generate frame t; motion correction's `model.shifts` is approximately the negative (the correction that undoes the drift)

`CNMFe` result attributes added:
- `model.YrA: (K, T)` — residual at each footprint; `C + YrA` = noisy projected trace
- `model.g: list[np.ndarray]` — per-component AR coefficients used for OASIS
- `model.sn_per_k: (K,)` — per-component noise std

---

## Running things

```bash
# Install (includes matplotlib in core deps)
pip install -e .

# Optional extras
pip install oasis-deconvolution          # faster deconvolution
pip install cupy-cuda12x                 # GPU support (match your CUDA version)

# Tests
pytest tests/ -v                         # all 88 tests
pytest tests/test_pipeline.py -v         # pipeline + temporal-correlation regression
pytest tests/test_multiprocessing.py -v  # parallelism only

# Demo movies (one-time setup)
python generate_demo_movies.py           # creates demo_movies/*.avi + *_meta.npz
python convert_to_zarr.py                # creates demo_movies/*.zarr

# CLI pipeline
python concat_avis_to_zarr.py /path/to/folder/   # concatenate 0.avi...N.avi -> movie.zarr
python full_pipeline.py /path/to/movie.zarr       # run full pipeline, save results/

# Tutorials
jupyter notebook tutorial.ipynb
jupyter notebook tutorial2.ipynb              # clean rewrite
jupyter notebook tutorial_realistic.ipynb     # realistic miniscope simulator
jupyter notebook tutorial_caiman_compare.ipynb  # needs CaImAn (see notebook intro)
jupyter notebook tutorial_demo.ipynb          # realistic lazy-load AVI workflow
```

---

## Windows-specific caveats

- Multiprocessing uses `spawn` (no `fork`). Scripts using `n_jobs != 1` must guard with
  `if __name__ == "__main__":`.
- `joblib` sometimes prints spurious warnings about worker processes on Windows — harmless.
- The test suite passes on Windows without the guard because pytest handles this correctly.
- For mp4 export from notebooks, prefer cv2 over imageio: imageio's pyav plugin requires an
  explicit `codec=` arg or it fails with "expected bytes, NoneType found" on Windows envs.
  `tutorial_realistic.ipynb` cell 8 uses cv2 first and falls back to imageio + tifffile.
- Building CaImAn from `CaImAn-main/` requires MSVC Build Tools (VS 2022 BuildTools edition is
  installed under `C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/`); use the
  Developer Command Prompt or run `vcvars64.bat` to get `cl` on PATH before
  `python setup.py build_ext --inplace`. CaImAn is not actually imported by `cnmfe/`.
- When using `n_jobs=-1` followed by CaImAn's `cm.cluster.setup_cluster()` (e.g. in
  `tutorial_caiman_compare.ipynb`), the loky worker pool from joblib persists after `CNMFe.fit()`.
  CaImAn's `setup_cluster` raises "A cluster is already running" when it detects live processes.
  **Fix:** call `from joblib.externals.loky import get_reusable_executor; get_reusable_executor().shutdown(wait=True)`
  before each `setup_cluster` call. This drains loky workers without affecting subsequent `CNMFe` calls.
