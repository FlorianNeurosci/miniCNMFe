---
tags: [cnmfe, api, reference]
---

# CNMFe — API Reference

> See [[usage-guide]] for examples. See [[architecture]] for the module map.

---

## `cnmfe.pipeline` — Top-level orchestrator

### `CNMFeParams`

Dataclass holding all algorithm parameters. Pass to `CNMFe()`.

```python
@dataclass
class CNMFeParams:
    # Motion correction
    max_shift: tuple[int, int] = (20, 20)     # Max (dy, dx) shift in pixels
    upsample_factor: int = 10                  # Subpixel precision = 1/upsample_factor
    mc_n_iter: int = 1                         # Number of correction passes (CaImAn default is 1)

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
    seed_suppress_factor: float = 2.0          # Suppression disk after extraction = factor * sigma
    circular_max_dist_factor: float = 2.5      # circular_constraint cutoff = factor * estimated_radius

    # Background (ring model)
    ring_size_factor: float = 1.5              # ring_radius = factor * (2*sigma + 1)
    ring_lambda: float = 1e-5                  # Ridge regularisation for ring regression

    # Spatial update
    dilation_radius: int = 3                   # Support dilation for LassoLars
    spatial_max_thr: float = 0.1               # Zero footprint pixels < this fraction of peak

    # Temporal update / deconvolution
    ar_order: int = 1                          # AR model order (1 or 2)
    n_iter_temporal: int = 2                   # BCD iterations per temporal update

    # Merging
    merge_thr_corr: float = 0.85               # Min temporal correlation to merge
    merge_thr_overlap: float = 0.5             # Min Jaccard spatial overlap to merge
    merge_centre_dist_factor: float = 2.0      # Centre-distance fallback = factor * sigma (px)

    # Main loop
    n_iter_main: int = 2                       # Full spatial+temporal+merge cycles

    # AR coefficient strategy
    global_ar: bool = False                    # True = one g from pooled C_raw; False = per-neuron g

    # Parallelism
    n_jobs: int = 1                            # Workers (-1 = all CPUs, 1 = serial)
    device: str = "cpu"                        # 'cpu' or 'cuda' (requires CuPy)
```

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
    ) -> "CNMFe": ...
```

**`fit()` parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `movie` | `zarr.Array` or `np.ndarray` | Input movie, shape `(T, H, W)` |
| `do_motion_correction` | `bool` | Run rigid motion correction first (default `True`) |
| `output_dir` | `str \| Path \| None` | Save motion-corrected movie here as zarr (optional) |

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
| `dims` | `tuple[int, int]` | — | `(H, W)` image dimensions |

> [!TIP]
> Use `model.C` for analyses that want a clean denoised AR(1) trace (e.g. event detection); use `model.C + model.YrA` when you need the data's actual shape (e.g. correlation against an external reference, plotting raw fluorescence).

---

## `cnmfe.io` — File I/O

### `avi_to_zarr`

```python
def avi_to_zarr(
    src: str | Path,
    dest: str | Path,
    chunk_t: int = 100,
    grayscale: bool = True,
    dtype: str = "float32",
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
def save_zarr(arr: np.ndarray, path: str | Path, chunk_t: int = 100) -> zarr.Array
```

Persist an in-memory `(T, H, W)` array to zarr. Useful for saving intermediate results.

---

## `cnmfe.motion_correction`

### `motion_correct`

```python
def motion_correct(
    movie: zarr.Array | np.ndarray,
    upsample_factor: int = 10,
    max_shift: tuple[int, int] = (20, 20),
    n_iter: int = 2,
    output_path: str | Path | None = None,
    template_frames: int = 200,
    update_interval: int = 100,
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray]
```

Rigid motion correction. Template initialised from the mean of the first `template_frames` frames; updated as a running mean every `update_interval` frames.

**Returns:** `(corrected_movie, shifts)` where `corrected_movie` has shape `(T, H, W)` and `shifts` has shape `(T, 2)`.

---

### `estimate_shifts`

```python
def estimate_shifts(
    frame: np.ndarray,
    template: np.ndarray,
    upsample_factor: int = 10,
    max_shift: tuple[int, int] = (20, 20),
) -> np.ndarray
```

Compute subpixel `(dy, dx)` shift between `frame` and `template` via phase cross-correlation. Returns shape `(2,)`.

---

### `apply_shift`

```python
def apply_shift(frame: np.ndarray, shift: np.ndarray) -> np.ndarray
```

Apply a `(dy, dx)` shift to `frame` via Fourier-domain phase multiplication. No spatial interpolation artifacts.

---

## `cnmfe.preprocess`

### `correlation_pnr`

```python
def correlation_pnr(
    movie: zarr.Array | np.ndarray,
    sigma: float | None = None,
    center_psf: bool = True,
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray]
```

Compute CORR and PNR summary images.

- If `sigma` is given, apply center-surround PSF filtering first.
- If `center_psf=False` or `sigma=None`, skip filtering (use when movie is already filtered).

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

## `cnmfe.background`

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

## `cnmfe.initialization`

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

## `cnmfe.spatial`

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
) -> sp.csc_matrix
```

Refine spatial footprints by per-pixel non-negative LassoLars regression. **Returns** updated `(H·W, K)` sparse matrix.

`max_thr`: after regression, pixels whose value falls below `max_thr × peak` are zeroed. Lower values keep dim peripheral pixels; higher values produce tighter footprints.

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
def threshold_footprint(ai: np.ndarray, dims: tuple[int, int], max_thr: float = 0.1) -> np.ndarray
```

Clean a spatial footprint: median filter → zero pixels below 10% of max → keep largest connected component. Input is flat `(H·W,)`. **Returns** flat `(H·W,)`.

---

## `cnmfe.temporal`

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
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]
```

Block coordinate descent temporal refinement. **Returns** `(C, S, g_per_k, sn_per_k)` — denoised traces `(K, T)`, spike trains `(K, T)`, per-component AR coefs (list of length K), and per-component noise std `(K,)`.

- `n_jobs=1`: Gauss-Seidel (sequential updates, slightly faster convergence)
- `n_jobs!=1`: Jacobi (parallel, all components updated simultaneously)

**`g_cached` / `sn_cached`**: pass pre-estimated AR coefficients and noise stds to skip per-call estimation. Critical for avoiding drift — without caching, `g` is re-estimated each call from the previously-deconvolved trace, and `estimate_ar_params` re-applies the `fudge_factor=0.96` shrinkage each time, drifting `g` toward 0 over iterations. The pipeline estimates `g` once after init from a pooled `C_raw.ravel()` trace and threads the cache through every call. If `None`, estimation runs once before the BCD loop on the input `C`.

---

### `deconvolve`

```python
def deconvolve(trace: np.ndarray, g: np.ndarray, sn: float) -> tuple[np.ndarray, np.ndarray, float]
```

Deconvolve one fluorescence trace with OASIS (uses `oasis-deconvolution` package if installed, falls back to pure-Python PAVA for AR(1)).

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
) -> tuple[np.ndarray, float]
```

Estimate AR(p) decay constants `g` (shape `(p,)`) and noise std `sn` from a trace.

---

## `cnmfe.merging`

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

## `cnmfe.evaluate`

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

`sn_flat` is the same per-pixel noise std produced by `cnmfe.preprocess.estimate_noise(...).ravel()` — the pipeline reuses its own `self.sn` for this call.

Pipeline-level knob: `CNMFeParams.auto_eval_snr_amp_thr` (default `3.0`). Setting it to `0.0` disables the SNR check; `min_pixel` continues to apply.

---

## `cnmfe._utils`

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
