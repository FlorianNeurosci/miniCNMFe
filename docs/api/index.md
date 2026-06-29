---
tags: [minicnmfe, api, reference]
---

# CNMFe — API Reference

> See the [usage guide](../getting-started/index.md) for examples. See the [architecture overview](../concepts/architecture.md) for the module map.

---

## `minicnmfe.pipeline` — Top-level orchestrator

### `CNMFeParams`

Dataclass holding all algorithm parameters. Pass to `CNMFe()`.

```python
@dataclass
class CNMFeParams:
    # Motion correction
    max_shift: tuple[int, int] = (20, 20)     # Max (dy, dx) shift in pixels
    upsample_factor: int = 10                  # Subpixel precision = 1/upsample_factor
    mc_n_iter: int = 1                         # Number of correction passes (CaImAn default is 1)
    mc_gSig_filt: float | None = None          # High-pass Gaussian radius for MC (None = use sigma)
    mc_batch_size: int = 200                   # Frames per streaming/parallel MC batch
    mc_template_max_frames: int = 2000         # Cap on frames sampled to build the MC template
    mc_output_chunk_t: int | None = None       # Output zarr time chunk (None = match source)
    mc_output_dtype: str = "float32"           # Output zarr dtype for the corrected movie

    # Spatial filtering / PSF
    sigma: float = 3.0                         # Neuron Gaussian radius in pixels
    center_psf: bool = True                    # Use center-surround DoG filter

    # Initialization
    min_corr: float = 0.8                      # Min local correlation for a seed
    min_pnr: float = 10.0                      # Min PNR for a seed
    min_pixel: int = 3                         # Min nonzero pixels in a valid footprint
    border_px: int = 5                         # Ignore seeds within N border pixels
    max_neurons: int | None = None             # Stop early (None = no limit)
    init_min_corr_neuron: float = 0.8          # "Neuron pixel" threshold inside extract_spatial_temporal
    init_max_corr_bg: float = 0.4              # "Background pixel" threshold inside extract_spatial_temporal
    seed_suppress_factor: float = 2.0          # DEPRECATED no-op (kept for API stability); greedy init now suppresses via an ind_search support mask, not a disk
    circular_max_dist_factor: float = 2.5      # circular_constraint cutoff = factor * estimated_radius
    init_stride: int | None = None             # Temporal stride for greedy init (None = auto, max(1, T//5000))
    init_corrpnr_stride: int | None = None     # Temporal stride for the CORR/PNR summary images (None = auto)
    init_patches: bool = True                  # Patch-PARALLEL greedy init (CPU only); default ON. Auto-skips to single-FOV greedy for small FOV / GPU / streaming-zarr movies
    init_patch_size: int | None = None         # patch side px; None -> max(int(12*sigma), 48)
    init_patch_overlap: int | None = None      # overlap px between patches; None -> int(4*sigma)
    init_patch_min_fov: int = 128              # only tile when min(H, W) >= this; else fall back to global init
    init_patch_n_jobs: int | None = None       # patch workers (processes); None -> n_jobs
    init_patch_max_workers: int = 32           # upper bound on patch-init loky processes (caps process-count RAM on many-core boxes; results unchanged)

    # Background (ring model)
    ring_size_factor: float = 1.5              # ring_radius = factor * (2*sigma + 1)
    ring_lambda: float = 1e-5                  # Ridge regularisation for ring regression
    ring_constrain_sum: bool = False           # Constrain ring weights to sum to 1 (see ring-constrain-sum)
    global_bg_rank: int = 0                    # Extra low-rank global background components (0 = ring only)
    bg_tsub: int = 5                           # Temporal subsample factor when fitting W / b0

    # Spatial update (per-pixel positive elastic-net coordinate descent)
    dilation_radius: int = 2                   # Support dilation for the per-pixel CD solve
    spatial_max_thr: float = 0.1               # Zero footprint pixels < this fraction of peak (used when spatial_thr_method="max")
    spatial_close_radius: int = 1              # Morphological close radius on the support mask
    spatial_max_iter: int = 1000               # Per-pixel coordinate-descent iteration cap
    spatial_tol: float = 1e-4                  # Per-pixel coordinate-descent convergence tolerance
    spatial_circular_max_dist_factor: float = 1.5  # circular cutoff applied after the spatial update
    spatial_ridge: float = 1e-2                # Elastic-net L2 fraction (beta = spatial_ridge * max(diag(Gram))); bounds CD conditioning, 0 = pure LASSO
    spatial_thread_cap: int = 16               # Max BLAS/worker threads in the spatial CD
    spatial_lambda_scale: float = 1.0          # Multiplier on the per-pixel L1 penalty (>1 = tighter footprints; 1.0 = standard CNMF-E)
    spatial_max_radius_factor: float = 0.0     # Absolute clip radius cap = factor * sigma px (0.0 = off, area-derived radius only)
    spatial_thr_method: str = "nrg"            # Footprint thresholding: "nrg" (energy, default) or "max" (peak-relative, legacy)
    spatial_nrg_thr: float = 0.95              # Energy fraction kept when thr_method="nrg" (brightest pixels summing to this share of total a^2)
    spatial_tsub: int = 1                      # Time-subsample factor for the bandwidth-bound update_spatial slab

    # Temporal update / deconvolution
    ar_order: int = 1                          # AR model order (1 or 2)
    global_ar: bool = True                     # True = one g from pooled C_raw; False = per-neuron g
    n_iter_temporal: int = 2                   # BCD iterations per temporal update
    skip_first_deconv: bool = True             # Skip OASIS on the first temporal pass (speed)
    fudge_factor: float = 0.96                 # Yule-Walker shrinkage (legacy path; bypassed when prior is set)

    # Bayesian-prior on g (preferred over fudge_factor when known)
    decay_time_ms: float | None = None         # Indicator τ (single-AP, somatic); see indicator table
    frame_rate_hz: float | None = None         # Recording fps; both required to enable prior
    g_prior_weight: float = 0.5                # Shrinkage weight; 0=pure data, 1=pin at target

    # Trace detrending (NON-STANDARD knobs; default 0 = legacy CNMF-E)
    ar_detrend_order: int = 0                  # Polynomial order subtracted before Yule-Walker
    temporal_detrend_order: int = 0            # Polynomial order subtracted before OASIS

    # Merging
    merge_thr_corr: float = 0.85               # Min temporal correlation to merge
    merge_thr_overlap: float = 0.5             # Min Jaccard spatial overlap to merge
    merge_centre_dist_factor: float = 2.0      # Centre-distance fallback = factor * sigma (px)

    # Main loop
    n_iter_main: int = 2                       # Full spatial+temporal+merge cycles
    sample_frames: int = 1000                  # Frames sampled for noise/CORR-PNR summaries

    # Auto-evaluation (non-destructive tagging)
    auto_eval_snr_amp_thr: float = 3.0         # Mean-amplitude SNR threshold (0 = disable SNR check)

    # Parallelism
    n_jobs: int = 1                            # Workers (-1 = all CPUs, 1 = serial)
    device: str = "cpu"                        # 'cpu' or 'cuda' (requires CuPy)

    # Streaming-IO tuning (affects on-disk Y_flat IO speed only, never results)
    yflat_dir: str | None = None               # where auto-derived Y_flat_pixel.zarr is written (None = under output_dir)
    yflat_pixel_chunk: int = 512               # pixel-row chunk of the on-disk Y_flat store
    yflat_time_chunk: int | None = None        # time chunk (None = full T)
    yflat_compression: bool = True             # blosc lz4+bitshuffle on Y_flat (keep True on network mounts)

    # Cutout (crop the movie before extraction; NATIVE coords; all None = no cutout)
    temporal_crop: tuple[int, int] | None = None           # (t0, t1), t1 exclusive
    spatial_crop: tuple[int, int, int, int] | None = None  # (y0, y1, x0, x1)
    spatial_mask_path: str | None = None                   # path to a bool (H, W) .npy mask
```

**Cutout.** When any of `temporal_crop` / `spatial_crop` / `spatial_mask_path` is
set, the movie is cropped **once at ingestion, before motion correction**, in
native coordinates (see `minicnmfe.cutout`). The applied spec is recorded on
`model.cutout`. A cutout **cannot be combined with a pre-built `Y_flat_zarr`**
(bake the crop in upstream instead), and `.downscaled()` **clears** the cutout
fields since the crop is applied at native resolution. Use
`minicnmfe.cutout.place_footprints_in_fov` / `place_traces_in_timeline` to map cropped
results back onto the full FOV / timeline.

**Downsample-once rescale.** `params.downscaled(ssub, tsub)` returns a **copy**
with the unit-bearing fields rescaled for a movie binned by `ssub` (space) /
`tsub` (time): `sigma /= ssub`, `min_pixel //= ssub²`, `border_px //= ssub`,
`max_shift //= ssub`, `mc_gSig_filt /= ssub`, `frame_rate_hz /= tsub`.
`decay_time_ms` is a physical time and is left **unchanged**. Express params once
in native units and call `.downscaled(...)` for the binned pipeline.

**Bayesian g prior.** When both `decay_time_ms` and `frame_rate_hz` are set, the
pipeline derives `g_target = exp(-1 / (fps · τ_ms / 1000))` and shrinks every
Yule-Walker estimate toward it:

    g = (1 - g_prior_weight) · g_yw + g_prior_weight · g_target

`fudge_factor` is bypassed on this path (the prior already encodes the
physical bound). Suggested `decay_time_ms` (single-AP τ, somatic):

| Indicator | τ (ms) |
|---|---|
| GCaMP6f | ~140 |
| jGCaMP7f | ~160 |
| jGCaMP8f | ~70 |
| jGCaMP8m | ~180 |
| jGCaMP8s | ~350 |
| GCaMP6s / 7s | ~1000 |

Values vary 1.5–2× with cell type, AP count, expression level. Bump
`g_prior_weight` toward 1 on drift-heavy recordings where Yule-Walker is
upward-biased.

---

### `CNMFe`

Main pipeline class.

```python
class CNMFe:
    def __init__(self, params: CNMFeParams | None = None) -> None: ...

    def fit(
        self,
        movie: zarr.Array | np.ndarray,
        do_motion_correction: bool = True,
        output_dir: str | Path | None = None,
        Y_flat_zarr: zarr.Array | None = None,
        evaluate: bool = True,
    ) -> "CNMFe": ...
```

`fit()` is a **thin wrapper** that composes the standalone stages
`fit_mc` (optional) → `fit_extract` → `evaluate`. The decomposition is
bit-for-bit identical to the old monolith at `n_jobs=1`
(`tests/test_stage_split.py`).

**`fit()` parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `movie` | `zarr.Array` or `np.ndarray` | Input movie, shape `(T, H, W)` |
| `do_motion_correction` | `bool` | Run rigid motion correction first (default `True`) |
| `output_dir` | `str \| Path \| None` | Save motion-corrected movie here as zarr (optional) |
| `Y_flat_zarr` | `zarr.Array \| None` | Pre-transposed pixel-major store (see `transpose_zarr_to_pixel_major`) for true T-streaming; peak RAM then independent of T |
| `evaluate` | `bool` | Run the non-destructive auto-eval pass (default `True`) |

**Staged entry points** — `fit()` composes these; call them directly for a
disk handoff between stages (the four `run_*.py` CLIs wrap them):

```python
def fit_mc(self, movie, output_dir=None, in_place=False) -> zarr.Array | np.ndarray: ...
def fit_extract(self, movie, *, Y_flat_zarr=None, output_dir=None, evaluate=True) -> "CNMFe": ...
def evaluate(self) -> "CNMFe": ...
```

- `fit_mc` — in-memory rigid motion correction; returns the corrected movie (zarr handle if `output_dir` given, else numpy). Motion correction is **not** part of `fit_extract`.
- `fit_extract` — noise estimation through the BCD loop, final temporal pass, and `YrA`. **Resolution-agnostic**: runs on whatever movie it is handed (full or downsampled). Contains the streaming `Y_flat_zarr` auto-derive logic.
- `evaluate` — the non-destructive auto-eval. Reads **only** `self.A` + `self.sn`, so it can be re-run on a freshly `load()`-ed model to retune `min_pixel` / `auto_eval_snr_amp_thr` without re-extracting.

**Fused AVI → motion-corrected zarr** (wraps `minicnmfe.avi_mc.concat_avis_to_mc_zarr`, documented below):

```python
def fit_mc_from_avis(
    self,
    folder: str | Path,
    output_dir: str | Path,
    *,
    pattern: str = "*.avi",
    skip_if_exists: bool = False,
    ssub: int = 1,
    tsub: int = 1,
) -> zarr.Array: ...
```

Decode an AVI folder + apply rigid motion correction in one pass, writing only
`<output_dir>/mc.zarr` — no intermediate `session.zarr`. `ssub` / `tsub` bin
frames (block-mean) **before** MC; pass `params.downscaled(ssub, tsub)` so MC
knobs match the binned grid. Returns the open `mc.zarr` handle.

**Upsample a downsampled model back to native resolution:**

```python
def upsample_to_native(
    self,
    *,
    orig_dims: tuple[int, int] | None = None,
    orig_T: int | None = None,
    ssub: int | None = None,
    tsub: int | None = None,
    ds_meta: dict | str | Path | None = None,
    spatial_order: int = 1,
) -> "CNMFe": ...
```

Returns a **new, non-destructive** model with `A` (bilinear) and
`C`/`YrA`/`C_raw` (linear) interpolated to the native grid/rate — for overlaying
footprints on a native reference image and plotting against native-rate signals.
This is **interpolation, not recovery** (the native movie is gone in
downsample-once). The returned model is **inspection-only**: `S` stays at the
downsampled rate, and `W`/`b0`/`f`/`shifts` are dropped — do not re-run the BCD
on it. Supply native `(H, W)` / `T` via `orig_dims`+`orig_T`, the `ssub`/`tsub`
factors, or a `ds_meta` dict/path (only `downsample_movie` writes a
`ds_meta.json`).

**Persistence:**

```python
def save(self, output_dir: str | Path) -> None: ...
@staticmethod
def load(output_dir: str | Path) -> "CNMFe": ...
```

`save()` writes `A`/`C`/`S`/`C_raw`/`YrA`/`W`/`b0`/`sn`/`g`/`shifts`/`params`
plus `accepted_mask.npy` and `eval_info.npz`; `load()` restores them.

**Result attributes** (available after `fit()`)

| Attribute | Type | Shape | Description |
|-----------|------|-------|-------------|
| `A` | `sp.csc_matrix` | `(H·W, K)` | Spatial footprints |
| `C` | `np.ndarray` | `(K, T)` | OASIS-deconvolved calcium traces (clean AR(1) shape) |
| `S` | `np.ndarray` | `(K, T)` | Inferred spike trains |
| `C_raw` | `np.ndarray` | `(K_init, T)` | Raw traces from greedy init (pre-deconvolution); `K_init` may differ from final `K` after merging |
| `YrA` | `np.ndarray` | `(K, T)` | Residual at each footprint after the final BCD pass. `C + YrA` is the noisy projected trace — preserves the data's actual shape better than `C` alone (Pearson r vs ground truth typically > 0.9 vs ~0.6 for `C`). |
| `W` | `sp.csr_matrix` | `(H·W, H·W)` | Ring background weights |
| `b0` | `np.ndarray` | `(H·W,)` | Per-pixel baseline |
| `sn` | `np.ndarray` | `(H, W)` | Per-pixel noise std |
| `g` | `list[np.ndarray]` | `K × (p,)` | Per-component AR coefficients used for OASIS (pooled estimate from `C_raw` then cached, not re-estimated each iteration) |
| `sn_per_k` | `np.ndarray` | `(K,)` | Per-component noise std used for OASIS |
| `shifts` | `np.ndarray \| None` | `(T, 2)` | Per-frame (dy, dx) shifts, or `None` |
| `accepted_mask` | `np.ndarray` | `(K,)` bool | Components passing both auto-eval checks (pixel-count floor AND mean-amplitude SNR). **Nothing is dropped** — slice `A`/`C`/`S`/`YrA` by this mask to use accepted components only. |
| `eval_info` | `dict` | — | Per-component auto-eval stats: `pixel_count`, `snr_amp`, `pixel_pass`, `snr_pass`, plus the thresholds applied. |
| `cutout` | `dict \| None` | — | Applied cutout spec (`bbox`, `t_range`, mask path) when a cutout was used; `None` otherwise. Footprints/traces are in cropped coordinates — use `minicnmfe.cutout` helpers to place them back in the full FOV/timeline. |
| `dims` | `tuple[int, int]` | — | `(H, W)` image dimensions |

> [!TIP]
> Use `model.C` for analyses that want a clean denoised AR(1) trace (e.g. event detection); use `model.C + model.YrA` when you need the data's actual shape (e.g. correlation against an external reference, plotting raw fluorescence).

---

## `minicnmfe.io` — File I/O

### `avi_to_zarr`

```python
def avi_to_zarr(
    src: str | Path,
    dest: str | Path,
    chunk_t: int = 100,
    grayscale: bool = True,
    dtype: str = "uint8",
    compression: bool = True,
) -> zarr.Array
```

Stream an AVI/MP4 file to a zarr store. Never loads the full movie into memory.

**Returns:** open `zarr.Array` with shape `(T, H, W)`.

---

### `open_zarr`

```python
def open_zarr(path: str | Path, mode: str = "r") -> zarr.Array
```

Open an existing zarr store. `mode="r"` for read-only, `"r+"` to append.

---

### `save_zarr`

```python
def save_zarr(
    arr: np.ndarray,
    path: str | Path,
    chunk_t: int = 100,
    dtype: str = "float32",
    compression: bool = True,
) -> zarr.Array
```

Persist an in-memory `(T, H, W)` array to zarr. Useful for saving intermediate results.

---

### `transpose_zarr_to_pixel_major`

```python
def transpose_zarr_to_pixel_major(
    src: str | Path,
    dest: str | Path,
    *,
    pixel_chunk: int = 4096,
    time_chunk: int = 2000,
    dtype: str = "float32",
    compression: bool = True,
    src_batch_frames: int = 2000,
    skip_if_exists: bool = True,
    verbose: bool = True,
) -> zarr.Array
```

One-time on-disk transpose of a `(T, H, W)` zarr into a pixel-major `(H·W, T)`
store. Pixel ordering matches `make_2d` (pixel `(h, w)` → flat `h*W + w`). Pass
the result as `fit(..., Y_flat_zarr=...)` to run **true T-streaming** extraction —
peak RAM independent of T, bounded by `K·T·4` (traces) + per-batch buffers.
Open the result later with `open_zarr_pixel_major`.

---

### `open_zarr_pixel_major`

```python
def open_zarr_pixel_major(path: str | Path, mode: str = "r") -> zarr.Array
```

Open a pixel-major `(H·W, T)` store produced by `transpose_zarr_to_pixel_major`.

---

## `minicnmfe.motion_correction`

### `motion_correction_rigid`

```python
def motion_correction_rigid(
    movie: zarr.Array | np.ndarray,
    output_path: str | Path | None = None,
    max_shift: tuple[int, int] = (20, 20),
    gSig_filt: float = 7,
    upsample_factor: int = 10,
    niter_rig: int = 1,
    bin_window: int = 10,
    template: np.ndarray | None = None,
    batch_size: int = 200,
    n_jobs: int = 1,
    template_max_frames: int = 2000,
    output_chunk_t: int | None = None,
    output_dtype: str = "float32",
    compression: bool = True,
    verbose: bool = True,
) -> tuple[zarr.Array | np.ndarray, np.ndarray]
```

The **canonical** rigid motion correction. Two execution paths chosen
automatically:

- **Streaming (zarr-backed)** — used when the input is a `zarr.Array` *or*
  `output_path` is given. Reads/writes batches; peak RAM is
  `(batch_size + template_max_frames) · H · W · 4` bytes, independent of T.
  Zarr input **requires** `output_path`. For `niter_rig > 1` it ping-pongs
  between two scratch zarrs and moves the final result to `output_path`.
- **In-memory** — numpy input, no `output_path`. Same algorithm, returns a
  numpy corrected movie. Kept for the small-movie test path.

Uses `cv2.filter2D` high-pass filtering and `cv2.warpAffine` to apply shifts
(matching CaImAn; scipy/FFT equivalents introduce a ~4–5 px offset — do not
substitute). The template is built from a strided sample of up to
`template_max_frames` frames, bin-median-reduced over `bin_window`.

**Returns:** `(corrected, shifts)` — `corrected` is a `zarr.Array` handle (zarr
path) or `(T, H, W)` numpy array (in-memory path); `shifts` is `(T, 2)`.

> `CNMFe.fit_mc(movie, output_dir=...)` is the convenience entry for big movies;
> pass a `zarr.Array` + `output_dir` and the corrected movie is written to
> `<output_dir>/mc.zarr` without materialising T frames in RAM.

---

### `estimate_shifts`

```python
def estimate_shifts(
    frame: np.ndarray,
    template: np.ndarray,
    upsample_factor: int = 10,
    max_shift: tuple[int, int] = (20, 20),
    gSig_filt: float | None = None,
) -> np.ndarray
```

Thin wrapper around `register_translation_caiman`. Computes the subpixel
`(dy, dx)` shift between `frame` and `template`; if `gSig_filt` is set, both are
high-pass filtered first. Returns shape `(2,)`.

---

### `apply_shift`

```python
def apply_shift(img: np.ndarray, shift: np.ndarray) -> np.ndarray
```

Apply a `(dy, dx)` shift to `img` (alias for `apply_shift_caiman`, which uses
`cv2.warpAffine` with cubic interpolation — matches CaImAn's pixel grid).

---

## `minicnmfe.avi_mc` — Fused AVI → motion-corrected zarr

### `concat_avis_to_mc_zarr`

```python
def concat_avis_to_mc_zarr(
    folder: str | Path,
    output_path: str | Path,
    params: CNMFeParams,
    *,
    pattern: str = "*.avi",
    n_jobs: int | None = None,
    skip_if_exists: bool = False,
    n_template_avis: int = 10,
    ssub: int = 1,
    tsub: int = 1,
    verbose: bool = True,
) -> tuple[zarr.Array, np.ndarray]
```

Decode an AVI folder and apply rigid motion correction in a **single pass**,
writing only `mc.zarr` — **no intermediate `session.zarr`** is materialised
(saves ~5 min and ~6 GB on a network mount for a 100k-frame session vs. running
`concat_avis_to_zarr` + `fit_mc` separately). `CNMFe.fit_mc_from_avis` wraps this.

Pipeline: pre-scan each AVI for `(count, H, W)` → build the MC template from a
strided subset of `n_template_avis` files → re-decode every AVI in parallel and
motion-correct batches into `mc.zarr` + a `(T, 2)` shifts buffer.

**Inline downsampling** (`ssub` / `tsub`): decoders bin frames (block-mean) to
float32 **before** MC; the template and output shape come from the per-file
binned counts (`sum(n_i // tsub)`, `H//ssub`, `W//ssub`). Pass
`params.downscaled(ssub, tsub)` so `max_shift` / `mc_gSig_filt` match the binned
grid. Output uses heavyweight `clevel=5` + bitshuffle (mc.zarr is read many times
during extraction). `params.mc_n_iter > 1` is supported via a scratch-zarr handoff
to `motion_correction_rigid` for the remaining passes.

**Returns:** `(mc_zarr, shifts)`.

---

## `minicnmfe.downsample` — Spatial/temporal binning + re-upsampling

### `downsample_movie`

```python
def downsample_movie(
    src: str | Path,
    dest: str | Path,
    *,
    ssub: int = 1,
    tsub: int = 1,
    src_batch_frames: int = 2000,
    chunk_t: int = 500,
    dtype: str = "float32",
    compression: bool = True,
    skip_if_exists: bool = True,
    write_meta: bool = True,
    verbose: bool = True,
) -> zarr.Array
```

Streaming block-mean of an existing `(T, H, W)` zarr by `ssub` (space) / `tsub`
(time). Non-divisible dims are trimmed to a multiple of the factor before
binning; the trailing `< tsub` frames are dropped (output T = `T // tsub`). When
`write_meta`, a `ds_meta.json` sidecar (`ssub`, `tsub`, `orig_dims`, `orig_T`) is
written next to `dest` — feed it to `run_extract.py --ds-meta` or
`CNMFe.upsample_to_native(ds_meta=...)`.

**Returns:** open downsampled `zarr.Array`.

---

### `upsample_footprints`

```python
def upsample_footprints(A, ds_dims, native_dims, order: int = 1)
```

Interpolate footprints `A` `(ds_H·ds_W, K)` from the downsampled grid to
`native_dims` `(H, W)` via per-column `cv2.resize`. Returns `(H·W, K)`.

### `upsample_traces`

```python
def upsample_traces(C, native_T: int, kind: str = "linear")
```

Interpolate traces `C` `(K, T_ds)` to `native_T` columns via per-row `np.interp`.
Used by `CNMFe.upsample_to_native` — **interpolation, not recovery**.

---

## `minicnmfe.detrend` — Rolling-percentile baseline removal

### `detrend_movie`

```python
def detrend_movie(
    src: str | Path,
    dest: str | Path,
    *,
    window_s: float = 30.0,
    percentile: float = 10.0,
    frame_rate_hz: float,
    batch_t: int = 2000,
    anchor_stride: int | None = None,
    chunk_t: int | None = None,
    n_jobs: int = 1,
    skip_if_exists: bool = True,
    verbose: bool = True,
) -> zarr.Array
```

Standalone streaming zarr→zarr preprocessing. Per-pixel **rolling-percentile
temporal detrend**: estimates a slow F0 baseline (the `percentile`-th percentile
over a `window_s`-second sliding window) and subtracts it, removing slow drift
(bleach, scope warm-up) before extraction. `anchor_stride` subsamples the
baseline knots for speed. Returns the open detrended `zarr.Array`.

---

## `minicnmfe.reject_frames` — Outlier-frame replacement

### `reject_outlier_frames`

```python
def reject_outlier_frames(
    src: str | Path,
    dest: str | Path,
    *,
    k_mad: float = 5.0,
    batch_t: int = 1000,
    chunk_t: int | None = None,
    skip_if_exists: bool = True,
    verbose: bool = True,
) -> tuple[zarr.Array, np.ndarray]
```

Standalone streaming zarr→zarr preprocessing. Flags frames whose per-frame mean
deviates by more than `k_mad`·MAD from the running level and **replaces them with
neighbour interpolation** (e.g. dropped/saturated frames). Returns
`(dest_zarr, replaced_mask)` where `replaced_mask` is a `(T,)` bool array of
which frames were replaced.

---

## `minicnmfe.cutout` — Crop the movie before extraction

The cutout is normally **param-driven**: set `CNMFeParams.temporal_crop` /
`spatial_crop` / `spatial_mask_path` and `CNMFe.fit` / `fit_extract` apply it once
at ingestion via `resolve_cutout` + `apply_cutout` (internal). These helpers map
cropped results back onto the full FOV / timeline:

### `place_footprints_in_fov`

```python
def place_footprints_in_fov(A_crop, bbox, orig_dims)
```

Pad cropped sparse footprints `(h·w, K)` back to the full `(H·W, K)` grid, placing
them at `bbox = (y0, y1, x0, x1)` within `orig_dims = (H, W)`.

### `place_traces_in_timeline`

```python
def place_traces_in_timeline(C_crop, t_range, orig_T)
```

Embed cropped traces `(K, T_win)` into a full `(K, orig_T)` timeline at
`t_range = (t0, t1)` (zeros elsewhere).

---

## `minicnmfe.preprocess`

### `correlation_pnr`

```python
def correlation_pnr(
    movie: zarr.Array | np.ndarray,
    sigma: float | None = None,
    center_psf: bool = True,
    noise_range: tuple[float, float] = (0.25, 0.5),
    n_jobs: int = 1,
    device: str = "cpu",
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]
```

Compute CORR and PNR summary images.

- If `sigma` is given, apply center-surround PSF filtering first.
- If `center_psf=False` or `sigma=None`, skip filtering (use when movie is already filtered).
- `stride` subsamples frames in time for the summary (speed on long recordings).

**Returns:** `(cn, pnr)` both shape `(H, W)`.

---

### `estimate_noise`

```python
def estimate_noise(
    movie: zarr.Array | np.ndarray,
    noise_range: tuple[float, float] = (0.25, 0.5),
    method: str = "logmexp",
) -> np.ndarray
```

Per-pixel noise std estimated from high-frequency PSD. **Returns** `(H, W)` array.

---

### `make_center_surround_psf`

```python
def make_center_surround_psf(sigma: float, size: int | None = None) -> np.ndarray
```

Build a center-surround (DoG-like) kernel of shape `(2*half+1, 2*half+1)`. Sums approximately to zero.

---

### `local_correlations_fft`

```python
def local_correlations_fft(movie: np.ndarray) -> np.ndarray
```

8-neighbour local correlation image computed via FFT shift trick. Input `(T, H, W)`, time-mean already subtracted. **Returns** `(H, W)`.

---

## `minicnmfe.background`

### `compute_W`

```python
def compute_W(
    Y_flat: np.ndarray,
    A: sp.csc_matrix,
    C: np.ndarray,
    dims: tuple[int, int],
    radius: float,
    lambda_reg: float = 1e-5,
    n_jobs: int = 1,
) -> tuple[sp.csr_matrix, np.ndarray]
```

Fit ring-model background weights.

| Parameter | Description |
|-----------|-------------|
| `Y_flat` | `(H·W, T)` movie |
| `A` | `(H·W, K)` sparse spatial footprints |
| `C` | `(K, T)` temporal traces |
| `dims` | `(H, W)` |
| `radius` | Ring inner radius in pixels |
| `lambda_reg` | Ridge regularisation strength |

**Returns:** `(W, b0)` — sparse `(H·W, H·W)` weight matrix and `(H·W,)` baseline.

---

### `subtract_background`

```python
def subtract_background(
    Y_flat: np.ndarray,
    W: sp.csr_matrix,
    b0: np.ndarray,
) -> np.ndarray
```

Subtract ring background: `Y_res = Y_flat - b0 - W @ (Y_flat - b0)`. **Returns** `(H·W, T)`.

---

### `build_ring_indices`

```python
def build_ring_indices(dims: tuple[int, int], radius: float) -> list[np.ndarray]
```

For each pixel (flat index), return flat indices of pixels in its ring neighbourhood `[radius, radius+1]`. Geometry only — call once and cache.

---

## `minicnmfe.initialization`

### `greedy_corr_pnr`

```python
def greedy_corr_pnr(
    movie: zarr.Array | np.ndarray,
    sigma: float,
    min_corr: float = 0.8,
    min_pnr: float = 10.0,
    max_neurons: int | None = None,
    min_pixel: int = 3,
    border_px: int = 0,
    ar_order: int = 1,
    n_jobs: int = 1,
    device: str = "cpu",
    min_corr_neuron: float = 0.8,
    max_corr_bg: float = 0.4,
    seed_suppress_factor: float = 2.0,
    circular_max_dist_factor: float = 2.5,
) -> tuple[sp.csc_matrix, np.ndarray, np.ndarray, np.ndarray]
```

Greedy CORR-PNR neuron initialisation.

| Parameter | Description |
|-----------|-------------|
| `min_corr_neuron` | Threshold for "neuron pixel" set inside the patch — pixels whose normalised trace correlates above this with the seed are pooled into the trace estimate. |
| `max_corr_bg` | Threshold for "background pixel" set — pixels below this are used as the local-background regressor in the OLS extraction. |
| `seed_suppress_factor` | After extracting a component, the CORR/PNR around its centre is zeroed within a disk of radius `max(seed_suppress_factor * sigma, 2*sigma + 1)` to prevent re-seeding on the same neuron's residual halo. |
| `circular_max_dist_factor` | Footprint cleanup: zero pixels farther than `factor * sqrt(area / π)` from the centroid. |

**Returns:** `(A, C, C_raw, centers)`

| Output | Shape | Description |
|--------|-------|-------------|
| `A` | `(H·W, K)` sparse | Spatial footprints |
| `C` | `(K, T)` | Deconvolved traces |
| `C_raw` | `(K, T)` | Raw (pre-deconvolution) traces |
| `centers` | `(K, 2)` | Neuron `(row, col)` coordinates |

---

### `greedy_corr_pnr_patched`

```python
def greedy_corr_pnr_patched(
    movie, sigma, ...,                # same greedy knobs as greedy_corr_pnr
    patch_size: int = 64,
    patch_overlap: int = 16,
    n_jobs: int = 1,
    merge_thr_corr: float = 0.85,
    merge_thr_overlap: float = 0.5,
    merge_centre_dist_factor: float = 2.0,
) -> tuple[sp.csc_matrix, np.ndarray, np.ndarray, np.ndarray]
```

**Patch-parallel** drop-in for `greedy_corr_pnr` (same `(A, C, C_raw, centers)`
return contract). The greedy seed loop is inherently sequential (each extraction
mutates the residual the next seed reads), so it can't be threaded directly.
Instead this tiles the in-RAM `(T, H, W)` movie into **overlapping** spatial
patches, runs `greedy_corr_pnr` on each patch in parallel **processes** (the loop
is GIL-bound — the one place the codebase uses processes rather than threads),
remaps each patch's footprints/centres to global coordinates, concatenates, then
de-duplicates neurons detected in patch overlaps via `merge_components` (the
centre-distance fallback). Edge rejection (`border_px`) and `max_neurons` are
applied **globally** after dedup. Peak extra RAM ≈ `n_jobs × T × patch_size² × 4`
bytes. CPU only. Driven by `CNMFeParams.init_patches` (default **on**; auto-skips
to single-FOV greedy for small FOV / GPU / streaming-zarr movies) — see the
`init_patch_*` params and the `CNMFe.fit` init branch.

---

### `detect_seeds`

```python
def detect_seeds(
    cn: np.ndarray,
    pnr: np.ndarray,
    min_corr: float,
    min_pnr: float,
    min_distance: int = 5,
) -> np.ndarray
```

Find local maxima of `cn * pnr` above thresholds. **Returns** `(N, 2)` array of `(row, col)` sorted by score descending.

---

### `extract_spatial_temporal`

```python
def extract_spatial_temporal(
    data_filtered: np.ndarray,
    data_raw: np.ndarray,
    seed_rc: tuple[int, int],
    patch_radius: int,
    min_corr_neuron: float = 0.8,
    max_corr_bg: float = 0.4,
    circular_max_dist_factor: float = 2.5,
) -> tuple[np.ndarray, np.ndarray, bool]
```

Extract `(ai, ci, success)` for a single candidate neuron. Returns footprint patch, raw trace, and whether extraction succeeded.

---

## `minicnmfe.spatial`

### `update_spatial`

```python
def update_spatial(
    Y_flat: np.ndarray,
    C: np.ndarray,
    A: sp.csc_matrix,
    sn: np.ndarray,
    dims: tuple[int, int],
    dilation_radius: int = 3,
    n_jobs: int = 1,
    max_thr: float = 0.1,
    closing_radius: int = 1,
    max_iter: int = 1000,
    tol: float = 1e-4,
    circular_max_dist_factor: float = 1.5,
    spatial_ridge: float = 1e-2,
    spatial_thread_cap: int = 16,
    lambda_scale: float = 1.0,
    sigma: float | None = None,
    max_radius_factor: float = 0.0,
    thr_method: str = "max",
    nrg_thr: float = 0.9999,
    spatial_tsub: int = 1,
) -> sp.csc_matrix
```

Refine spatial footprints by per-pixel non-negative **elastic-net coordinate descent** (`sklearn.linear_model._cd_fast.enet_coordinate_descent_gram`, via `_spatial_pixel_batch` / the numba `_spatial_cd_kernel`) — not `LassoLars`. The solve carries an L1 penalty (`λ_p = lambda_scale · 0.5 · sn[p] · √(max diag Gram) / T`) plus an L2 ridge term (`beta = spatial_ridge · max(diag(Gram))`) that keeps the CD well-conditioned. **Returns** updated `(H·W, K)` sparse matrix.

`thr_method` / `nrg_thr`: footprint cleanup method. `"max"` zeroes pixels below `max_thr × peak`; `"nrg"` keeps the brightest pixels whose summed `a²` reaches `nrg_thr` of the footprint's total energy. (The pipeline passes `thr_method="nrg"`, `nrg_thr=0.95` from `CNMFeParams` by default; the standalone-function defaults shown here stay `"max"` / `0.9999` for backward compatibility.)

`max_thr`: when `thr_method="max"`, pixels whose value falls below `max_thr × peak` are zeroed. Lower values keep dim peripheral pixels; higher values produce tighter footprints.

---

### `compute_support`

```python
def compute_support(
    A: sp.csc_matrix,
    dims: tuple[int, int],
    dilation_radius: int = 3,
) -> list[np.ndarray]
```

For each pixel, return indices of components whose dilated footprint covers that pixel. **Returns** list of length `H·W`.

---

### `threshold_footprint`

```python
def threshold_footprint(
    ai: np.ndarray,
    dims: tuple[int, int],
    max_thr: float = 0.1,
    closing_radius: int = 1,
    circular_max_dist_factor: float = 1.5,
) -> np.ndarray
```

Clean a spatial footprint: median filter → zero pixels below `max_thr × max` → morphological close (`closing_radius`) → keep largest connected component → zero pixels beyond `circular_max_dist_factor × radius` from the centroid. Input is flat `(H·W,)`. **Returns** flat `(H·W,)`.

---

## `minicnmfe.temporal`

### `g_from_decay_time` / `decay_time_from_g`

```python
def g_from_decay_time(decay_time_ms: float, frame_rate_hz: float) -> float
def decay_time_from_g(g: float, frame_rate_hz: float) -> float
```

Convert between an AR(1) decay coefficient `g` and a physical indicator decay
time τ (ms): `g = exp(-1 / (fps · τ_ms / 1000))` and its inverse
`τ_ms = -1000 / (fps · ln g)` (`g` clipped to `(0, 1)` for safety). The forward
form is the same expression that builds the Bayesian `g` prior in
`pipeline.fit_extract` — exposed so the simulator can generate traces with a
settable decay, and so an estimated `g` can be reported in interpretable units.

Approximate single-AP somatic τ (ms): GCaMP6f ~140, jGCaMP7f ~160, jGCaMP8f ~70,
jGCaMP8m ~180, jGCaMP8s ~350, GCaMP6s/7s ~1000.

---

### `update_temporal`

```python
def update_temporal(
    Y_flat: np.ndarray,
    A: sp.csc_matrix,
    C: np.ndarray,
    sn: np.ndarray,
    ar_order: int = 1,
    n_iter: int = 2,
    n_jobs: int = 1,
    device: str = "cpu",
    g_cached: list[np.ndarray] | None = None,
    sn_cached: np.ndarray | None = None,
    deconvolve: bool = True,
    detrend_order: int = 0,
    g_prior: float | None = None,
    g_prior_weight: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]
```

Block coordinate descent temporal refinement. **Returns** `(C, S, g_per_k, sn_per_k)` — denoised traces `(K, T)`, spike trains `(K, T)`, per-component AR coefs (list of length K), and per-component noise std `(K,)`.

- `n_jobs=1`: Gauss-Seidel (sequential updates, slightly faster convergence)
- `n_jobs!=1`: Jacobi (parallel, all components updated simultaneously)

**`g_cached` / `sn_cached`**: pass pre-estimated AR coefficients and noise stds to skip per-call estimation. Critical for avoiding drift — without caching, `g` is re-estimated each call from the previously-deconvolved trace, and `estimate_ar_params` re-applies the `fudge_factor=0.96` shrinkage each time, drifting `g` toward 0 over iterations. The pipeline estimates `g` once after init from a pooled `C_raw.ravel()` trace and threads the cache through every call. If `None`, estimation runs once before the BCD loop on the input `C`.

**`g_prior` / `g_prior_weight`**: Bayesian shrinkage target on the dominant AR coefficient. The pipeline derives `g_prior` from `CNMFeParams.decay_time_ms` + `frame_rate_hz` and threads it here (alongside greedy init's call site) so the entire pipeline uses a consistent `g`. Used only on the fallback path when `g_cached` is `None`.

---

### `deconvolve`

```python
def deconvolve(trace: np.ndarray, g: np.ndarray, sn: float) -> tuple[np.ndarray, np.ndarray, float]
```

Deconvolve one fluorescence trace with OASIS (uses `oasis-deconv` package if installed, falls back to pure-Python PAVA for AR(1)).

**Returns:** `(c, s, bl)` — denoised trace, spike train, baseline offset.

---

### `estimate_ar_params`

```python
def estimate_ar_params(
    trace: np.ndarray,
    p: int = 1,
    noise_range: tuple[float, float] = (0.25, 0.5),
    fudge_factor: float = 0.96,
    lags: int = 5,
    detrend_order: int = 0,
    g_prior: float | None = None,
    g_prior_weight: float = 0.5,
) -> tuple[np.ndarray, float]
```

Estimate AR(p) decay constants `g` (shape `(p,)`) and noise std `sn` from a trace.

**Two shrinkage paths.** If `g_prior` is provided (set by the pipeline when `CNMFeParams.decay_time_ms` and `frame_rate_hz` are both non-None), the dominant AR coefficient `g[0]` is computed as a convex combination `(1 - g_prior_weight) · g_yw + g_prior_weight · g_prior` and `fudge_factor` is bypassed. For `p > 1`, higher-order coefficients still use the legacy `fudge_factor` multiplier — the prior is a single-scalar target meaningful only for the dominant decay. If `g_prior is None`, the entire `g` vector is multiplied by `fudge_factor` (legacy behaviour).

---

## `minicnmfe.merging`

### `merge_components`

```python
def merge_components(
    A: sp.csc_matrix,
    C: np.ndarray,
    thr_corr: float = 0.85,
    thr_overlap: float = 0.5,
    ar_order: int = 1,
    sigma: float | None = None,
    dims: tuple[int, int] | None = None,
    centre_dist_factor: float = 2.0,
    max_thr: float = 0.1,
) -> tuple[sp.csc_matrix, np.ndarray, int, list[np.ndarray]]
```

Merge spatially overlapping and temporally correlated components.

**Returns:** `(A_merged, C_merged, n_merged, members_per_group)` — merged footprints `(H·W, K_new)`, traces `(K_new, T)`, count of merge events, and a list of length `K_new` where `members_per_group[j]` gives the original-component indices that fused into output component `j` (singletons for unmerged, length > 1 for merged). Use this to update any per-component cache (e.g. `g_per_k`).

**Merge rule** (changed from earlier versions): two components merge if their traces are correlated **AND** they share spatial support **OR** sit close in centre-of-mass:

```
|Pearson(C[i], C[j])| > thr_corr  AND
( Jaccard(i, j) > thr_overlap  OR  centre_dist(i, j) < centre_dist_factor * sigma )
```

The centre-distance fallback catches duplicate detections of the same neuron whose footprints, after `threshold_footprint` keeps only the largest connected component around different peak pixels, end up with disjoint supports (`Jaccard ≈ 0`) despite tracking the same trace.

`sigma` and `dims` must be passed to enable the centre-distance fallback; otherwise only Jaccard is used. Merged footprints are re-thresholded via `threshold_footprint(max_thr=...)`. The merged trace is the mean of members (clipped non-negative) — re-deconvolution is **deferred** to the caller's next `update_temporal` pass, which uses the persistent per-component AR cache (re-estimating `g` here would re-introduce fudge-factor drift).

---

## `minicnmfe.evaluate`

### `auto_evaluate_components`

```python
def auto_evaluate_components(
    A: sp.csc_matrix,
    sn_flat: np.ndarray,
    min_pixel: int = 1,
    snr_amp_thr: float = 3.0,
) -> tuple[np.ndarray, dict]
```

Post-extraction quality filter. Called inside `CNMFe.fit()` between the BCD loop and the final temporal update; also usable standalone.

**Returns** `(keep_mask, info)` — `keep_mask` is a `(K,)` bool, `info` carries the per-component statistics (`pixel_count`, `snr_amp`, `pixel_pass`, `snr_pass`, plus the thresholds applied).

Two checks must both pass:

1. **Pixel-count floor.** `npix[k] >= min_pixel`.
2. **Mean-amplitude SNR.** A scale-invariant test against the per-pixel noise floor:

   ```
   snr_amp[k] = (||a_k||^2 / npix[k]) / mean(sn_flat[support_k]^2)
   ```

   Real σ=3 Gaussian footprints typically score 10–70 here; ghost components born from background-noise seeds (e.g. under loose `min_corr` / `min_pnr` thresholds) sit at or below 2 even when their pixel count is large. At `snr_amp_thr=3.0` the test cleanly separates real and ghost components.

`sn_flat` is the same per-pixel noise std produced by `minicnmfe.preprocess.estimate_noise(...).ravel()` — the pipeline reuses its own `self.sn` for this call.

Pipeline-level knob: `CNMFeParams.auto_eval_snr_amp_thr` (default `3.0`). Setting it to `0.0` disables the SNR check; `min_pixel` continues to apply.

---

## `minicnmfe._utils`

### `make_2d`

```python
def make_2d(movie: np.ndarray) -> np.ndarray
```

Reshape `(T, H, W)` → `(H·W, T)` (C-order). The standard flat representation for matrix algebra.

---

### `make_3d`

```python
def make_3d(Y_flat: np.ndarray, dims: tuple[int, int]) -> np.ndarray
```

Reshape `(H·W, T)` → `(T, H, W)`.

---

### `ensure_float32`

```python
def ensure_float32(arr: np.ndarray) -> np.ndarray
```

Cast to `float32` if not already. Returns view when possible.

---

### `iter_frames`

```python
def iter_frames(movie, batch_size: int = 200) -> Iterator[tuple[int, np.ndarray]]
```

Yield `(start_idx, batch)` for memory-efficient iteration over a zarr or numpy movie. `batch` is always float32.
