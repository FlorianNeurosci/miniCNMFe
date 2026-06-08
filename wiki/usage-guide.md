---
tags: [minicnmfe, usage, tutorial, quickstart]
---

# CNMFe — Usage Guide

> See [[api-reference]] for full parameter docs. See [[algorithm-eli5]] for intuition.

---

## Installation

```bash
pip install -e ".[test]"
```

Requires Python ≥ 3.10. Key dependencies: `numpy`, `scipy`, `scikit-image`, `scikit-learn`, `zarr >= 3.0`, `joblib`, `tqdm`, `imageio-ffmpeg`.

Optional but recommended: `oasis-deconvolution` (faster AR deconvolution). Without it the pure-Python AR(1) PAVA fallback is used automatically.

---

## Quickstart

### From an AVI/MP4 file

```python
from minicnmfe import CNMFe, CNMFeParams
from minicnmfe.io import avi_to_zarr

# Convert video to zarr (streams frame by frame — never loads full movie)
movie = avi_to_zarr("recording.avi", "/tmp/movie.zarr")

params = CNMFeParams(
    sigma=3.0,       # neuron radius in pixels — most important parameter
    min_corr=0.8,    # lower = more neurons found (but more false positives)
    min_pnr=10.0,    # lower = more neurons found (but more noise)
)
model = CNMFe(params).fit(movie)

# Results
print(f"Found {model.A.shape[1]} neurons")
A = model.A       # (H*W, K) sparse — spatial footprints
C = model.C       # (K, T)   — calcium traces
S = model.S       # (K, T)   — spike trains
```

### From a numpy array

```python
import numpy as np
from minicnmfe import CNMFe, CNMFeParams

movie = np.load("movie.npy")           # (T, H, W) float32
model = CNMFe(CNMFeParams(sigma=3.0)).fit(movie, do_motion_correction=False)
```

---

## Common Workflows

### Skip motion correction (already corrected)

```python
model = CNMFe(params).fit(movie, do_motion_correction=False)
```

### Save motion-corrected output

```python
model = CNMFe(params).fit(movie, output_dir="/tmp/cnmfe_out/")
# Saves corrected movie to /tmp/cnmfe_out/mc.zarr
```

### Run on a large zarr (RAM-bounded extraction)

The extraction path is streaming-aware after the Phase 1 refactors (May 2026):

- `fit()` accepts a `zarr.Array` directly. The corrected movie is held once
  as a pixel-major `(H·W, T)` float32 array; downstream steps allocate
  only small per-batch buffers.
- `BackgroundSubtractor` (used internally by the BCD loop) materialises
  pixel-row slices of `(I - W) @ (Y - b0)` on demand — `Y_bg` is never
  built in full.
- `compute_W` computes `b0` via streaming reductions and reuses the ring
  weight matrix `W` across BCD iterations (`W_cached` arg).
- Greedy init runs on a strided sample of the movie
  (`CNMFeParams.init_stride`, auto = `max(1, T // 5000)`); full-T traces
  are recovered by projection after init.

```python
from minicnmfe.io import open_zarr
from minicnmfe import CNMFe, CNMFeParams

z = open_zarr("session/mc.zarr")            # already motion-corrected
params = CNMFeParams(
    sigma=3.0, min_corr=0.8, min_pnr=10.0,
    init_stride=None,                       # auto: max(1, T // 5000)
    n_jobs=-1,
)
model = CNMFe(params).fit(z, do_motion_correction=False)
```

Peak working RAM is `~T·H·W·4` bytes (the movie itself) plus `~K·T·4` for
traces. On a 10k × 600 × 600 movie that is ~14 GB; for longer recordings
use the true T-streaming path below.

### True T-streaming via pixel-major zarr (60k+ frames)

The in-RAM path above caps out around ~30k frames at 600 × 600 on a
typical 64 GB workstation. For longer recordings, transpose the corrected
movie to pixel-major chunks once (one disk pass) and pass it to `fit()`
as `Y_flat_zarr`. Extraction then never materialises the full
`(H·W, T)` array — the on-disk store IS `Y_flat`.

```python
from minicnmfe.io import (
    open_zarr,
    open_zarr_pixel_major,
    transpose_zarr_to_pixel_major,
)
from minicnmfe import CNMFe, CNMFeParams

# One-time preprocessing: rewrite mc.zarr with chunks (4096, 2000)
# so pixel-row reads are O(B·T) IO instead of O(H·W·T).
transpose_zarr_to_pixel_major(
    "session/mc.zarr",
    "session/mc_pixel.zarr",
    pixel_chunk=4096,
    time_chunk=2000,
    skip_if_exists=True,
)

# Open both layouts. The 3D zarr is read only for the strided greedy-init
# sample (~T / init_stride frames); the pixel-major zarr is Y_flat.
mc_3d   = open_zarr("session/mc.zarr")
Y_pixel = open_zarr_pixel_major("session/mc_pixel.zarr")

params = CNMFeParams(sigma=3.0, min_corr=0.8, min_pnr=10.0, n_jobs=-1)
model = CNMFe(params).fit(
    mc_3d,
    do_motion_correction=False,
    Y_flat_zarr=Y_pixel,
)
```

Peak working RAM in this mode is bounded by `K·T·4` (traces) plus small
per-batch buffers — independent of `T`. The 3D zarr is touched only once
per fit (for the strided init sample).

### Use multiple CPU cores

```python
params = CNMFeParams(sigma=3.0, n_jobs=4)    # 4 workers
# or
params = CNMFeParams(sigma=3.0, n_jobs=-1)   # all available CPUs
```

> [!NOTE]
> On Windows, `n_jobs != 1` uses the `loky` backend (spawn-based). First call has overhead (~1–2 s) to start workers. Worthwhile for movies with T > 500 or large H×W.

### Limit neuron count (fast preview)

```python
params = CNMFeParams(sigma=3.0, max_neurons=50)
```

### Access individual results

```python
H, W = model.dims
K = model.A.shape[1]

# Reshape one footprint to image space
footprint_k = np.array(model.A[:, 0].todense()).reshape(H, W)

# Get trace and spikes for neuron 0
trace_0 = model.C[0]    # (T,)  — OASIS-deconvolved (clean AR(1) shape)
spikes_0 = model.S[0]   # (T,)  — inferred spike train
```

### Two flavours of the calcium trace: `C` vs `C + YrA`

`model.C` is the OASIS-deconvolved trace — clean AR(1) shape, ideal for spike-event analyses. But OASIS imposes the strict shape constraint `c[t] >= g * c[t-1]` and small spike-timing distortions can drop its Pearson correlation with the underlying data to ~0.6 on synthetic ground truth.

`model.C + model.YrA` is the **noisy projected trace** — the residual at each footprint added back. It preserves the data's actual shape and typically correlates > 0.9 with ground truth. Use this when you need shape fidelity (correlation analyses, plotting raw fluorescence, regressing against an external reference signal).

```python
C_clean = model.C                  # denoised AR(1) — for spike detection, event analyses
C_raw_after = model.C + model.YrA  # noisy but shape-faithful — for correlation, plotting
```

> **Dense-FOV caveat.** "`C + YrA` is shape-faithful" only holds at **low footprint
> overlap**. `YrA_k` is the data projected onto footprint `k` after subtracting the
> other components; when footprints overlap, that subtraction leaks neighbours'
> transients into `YrA_k` (cross-talk). So as the extracted **cell count rises**
> (looser `min_corr`/`min_pnr` in a dense field), `corr(C, C+YrA)` falls — for the
> strong cells too, not just the weak ones (measured: K=221 → mean 0.88 / strong
> 0.77; K=722 → 0.74 / 0.45). In dense extractions prefer `C` (the demixed
> estimate); the `C`-vs-`C+YrA` gap is `YrA` contamination, not `C` being wrong.
> `corr(C, C+YrA)` itself is a good metric for choosing how aggressive to make your
> detection thresholds.
>
> If you need both high cell count *and* clean traces, `n_iter_main=2` plus tighter
> footprints (`spatial_max_thr=0.25`, `spatial_circular_max_dist_factor=1.2`)
> sharpen the demixing: verified at K≈600 to recover strong-cell `corr(C, C+YrA)`
> from **0.48 → 0.77** (footprint npix median 79→36), at ~30% more runtime.

The per-component AR coefficient and noise std used by OASIS are also exposed:

```python
g_per_neuron = model.g          # list of length K, each (p,) np.array
sn_per_neuron = model.sn_per_k  # (K,) np.array
```

### Tuning long or dense recordings

Behaviour observed on a real 37k-frame miniscope recording (not autotested):

- **Long recordings sprawl the footprints.** Slow drift / photobleaching over tens
  of thousands of frames makes the temporal traces collinear, and `update_spatial`
  then smears each footprint over its neighbours into big merged blobs. The fix is
  to remove the shared drift, *not* to change thresholds (thresholds only change
  cell count, not footprint size):

  ```python
  params = CNMFeParams(
      sigma=3.0, min_corr=0.8, min_pnr=12,
      global_bg_rank=1,                      # absorb slow drift as a rank-1 temporal background
      spatial_max_thr=0.25,                  # trim halo at 25% of peak
      spatial_circular_max_dist_factor=1.2,  # tighter radial clip
      merge_thr_corr=0.90,                   # do NOT over-merge dense neighbours...
      merge_centre_dist_factor=1.0,          # ...(0.75/2.0 fuses distinct co-active cells)
      init_stride=2,                         # auto = T//5000 can subsample transients away
  )
  ```
  (The per-pixel temporal detrend in `minicnmfe/detrend.py` is an equivalent
  alternative to `global_bg_rank=1`.)

- **Dense fields: keep the merge gentle.** With the drift handled by
  `global_bg_rank=1`, aggressive merging (`merge_thr_corr=0.75`,
  `merge_centre_dist_factor=2.0`) fuses genuinely distinct, co-active neighbours —
  on the test recording that collapsed 214 detected cells to 109. The gentler
  `0.90` / `1.0` kept 175 cells with *tighter* footprints.

- **Judge footprint positions on a correlation image, not the mean projection.**
  Over a long recording the mean projection is dominated by static
  background/vasculature; the transient neurons are dim in it, so good footprints
  look "off the bright spots." Overlay contours on a `correlation_pnr` `cn` image
  instead (the `live_runs/cutout_*` notebooks do this).

---

## Real-data CLI Workflow

### Automated parameter tuning (start here)

Before hand-picking parameters, let the tuner suggest them and show you graphs:

```bash
python tune.py /path/to/recording/ -o tuning/ \
    --frame-rate 20 --decay-time-ms 180 --mode both
```

It writes a `tune_*/` folder with `recommended_params.json` (+ `downsample.json`)
and a `report.md` of figures. Feed the JSON straight into the steps below
(`--params tuning/tune_*/recommended_params.json`). See [[parameter-tuning]] for
the full guide.

### From a folder of numbered AVI files

Miniscope recordings typically arrive as `0.avi`, `1.avi`, ..., `65.avi`. Use the included scripts for an end-to-end workflow without writing any Python:

```bash
# Step 1: concatenate all AVI files in the folder into one zarr store
python -m minicnmfe.concat_avis_to_zarr /path/to/recording/
# creates /path/to/recording/movie.zarr

# Step 2: run the full pipeline
python full_pipeline.py /path/to/recording/movie.zarr \
    --sigma 3.0 --min-corr 0.8 --min-pnr 8.0 \
    --n-jobs -1

# Results saved to /path/to/recording/results/:
#   A.npz         spatial footprints  (scipy CSC, H*W x K)
#   C.npy         OASIS-deconvolved traces  (K x T)
#   S.npy         spike trains  (K x T)
#   YrA.npy       residuals; C + YrA = noisy projected trace  (K x T)
#   shifts.npy    per-frame motion correction shifts  (T x 2)
#   sn.npy        per-pixel noise std  (H x W)
#   params.json   all pipeline parameters used
```

Loading the results in Python:

```python
import numpy as np
import scipy.sparse as sp

A   = sp.load_npz("results/A.npz")    # (H*W, K) sparse
C   = np.load("results/C.npy")        # (K, T)
YrA = np.load("results/YrA.npy")      # (K, T)
C_proj = C + YrA                      # noisy projected trace (shape-faithful)
```

### Demo movies

To generate synthetic demo recordings and try the full workflow:

```bash
python generate_demo_movies.py   # creates demo_movies/*.avi + *_meta.npz
python convert_to_zarr.py        # creates demo_movies/*.zarr
jupyter notebook tutorial_demo.ipynb
```

The `tutorial_demo.ipynb` notebook opens a zarr lazily, runs the full pipeline, and scores extraction quality against the ground-truth.

### `python -m minicnmfe.concat_avis_to_zarr` options

| Flag | Default | Description |
|------|---------|-------------|
| `--output PATH` | `<folder>/movie.zarr` | Output zarr path |
| `--pattern GLOB` | `*.avi` | Glob to select AVI files |
| `--chunk-t N` | `500` | Frames per time chunk |
| `--color` | off | Keep RGB channels (default: grayscale) |
| `--n-jobs N` | auto | Parallel decoder threads. Default: `min(cpu_count, len(avis))`; pass a smaller value to cap |
| `--clevel N` | `3` | blosc compression level for the output (3 = fast, 5 = balanced, 9 = small) |
| `--shuffle MODE` | `shuffle` | blosc shuffle filter: `shuffle` (byte, fast), `bitshuffle` (slower, ~10 % smaller), `noshuffle` |

#### Expected runtimes

For a typical 100k-frame miniscope session (100 AVIs × 1000 frames × 600×600 uint8):

- **Local SSD** source and output: 2–3 min.
- **Network mount** for both source AVIs and output zarr: 5–7 min.

If you don't need to inspect the raw (pre-MC) zarr separately, use the
fused AVI→MC entrypoint (`minicnmfe.avi_mc.concat_avis_to_mc_zarr`) instead —
it writes only the motion-corrected zarr in a single pass and saves ~5 min
on network mounts by skipping the intermediate.

### `full_pipeline.py` options

| Flag | Default | Description |
|------|---------|-------------|
| `--sigma` | `3.0` | Neuron radius in pixels |
| `--min-corr` | `0.8` | Min local correlation for seed detection |
| `--min-pnr` | `10.0` | Min peak-to-noise ratio for seed detection |
| `--n-iter` | `1` | Main refinement cycles |
| `--n-jobs` | `-1` | CPU workers (-1 = all cores) |
| `--no-mc` | off | Skip motion correction |
| `--mc-iter` | `2` | Motion correction passes |
| `--max-shift` | `20` | Max shift in pixels |
| `--merge-corr` | `0.85` | Temporal correlation threshold for merging |
| `--spatial-thr` | `0.1` | Footprint peak-fraction threshold |
| `--global-ar` | off | Use one pooled AR coefficient (default: per-neuron) |
| `--output PATH` | `<zarr_parent>/results/` | Output directory |

---

## Parameter Tuning Guide

> [!TIP]
> To get a starting point automatically — recommended values plus graphs to
> judge them — run the [[parameter-tuning]] workflow (`python tune.py <path>`)
> before hand-tuning the knobs below.

### `sigma` — most important parameter

Gaussian radius of a neuron in pixels. Determines the PSF kernel size and the ring background radius.

- Too small: footprints are clipped, traces noisy
- Too large: multiple neurons merged into one seed

**How to set:** look at the CORR/PNR images (see §Inspecting intermediate results below). Neurons should be visible as compact bright spots. Measure the radius of a typical spot in pixels.

Typical values: `2.0`–`5.0` for miniscope data at standard zoom.

---

### `min_corr` and `min_pnr` — initialisation thresholds

Control how many initial seed candidates are accepted.

| Symptom | Likely fix |
|---------|-----------|
| Too few neurons found | Lower `min_corr` (e.g. 0.6) and/or `min_pnr` (e.g. 5.0) |
| Too many false positives | Raise `min_corr` (e.g. 0.9) and/or `min_pnr` (e.g. 15.0) |
| Neurons at image border missed | Lower `border_px` (default 5) |

---

### `n_iter_main` — refinement iterations

How many full spatial+temporal+merge cycles to run.

- `1`: fast, lower quality — use for debugging or parameter search
- `2`: good balance (default)
- `3`–`4`: diminishing returns but useful for noisy data

---

### `ar_order` — calcium dynamics model

- `1`: single exponential decay (most neurons, simpler, faster)
- `2`: double exponential — needed if the GCaMP indicator has a visible rise time

Default `1` works for GCaMP6s/GCaMP7 at typical frame rates.

---

### `decay_time_ms` + `frame_rate_hz` — Bayesian prior on the calcium decay (`g`)

The AR(1) decay coefficient `g` is normally estimated from the data via
Yule-Walker. On clean recordings this works fine. On 1-photon miniscope data,
slow un-subtracted background inflates the lag-1 autocorrelation and pushes
`g_yw → 1` regardless of the indicator's true decay time — so the
deconvolved `C` becomes a sticky AR tail that eats subsequent spikes.

Fix: tell the pipeline what indicator and frame rate you actually used.

```python
params = CNMFeParams(
    ...,
    decay_time_ms=180,     # single-AP τ in ms (table below)
    frame_rate_hz=20,      # recording fps
    g_prior_weight=0.5,    # 0 = pure data, 1 = pin at target; 0.5 default
)
```

The pipeline derives `g_target = exp(-1 / (fps · τ_ms / 1000))` and shrinks
every Yule-Walker estimate toward it. `fudge_factor` is bypassed.

| Indicator | τ (single-AP, somatic) |
|---|---|
| GCaMP6f | ~140 ms |
| jGCaMP7f | ~160 ms |
| jGCaMP8f | ~70 ms |
| jGCaMP8m | ~180 ms |
| jGCaMP8s | ~350 ms |
| GCaMP6s / 7s | ~1000 ms |

Values vary 1.5–2× with cell type, AP count, expression level — these are
typical somatic numbers from the indicator papers. On drift-heavy
recordings or recordings where you trust the indicator value more than the
data, bump `g_prior_weight` toward 1 (e.g. 0.8). Leave either field as
`None` to fall back to the legacy `fudge_factor` shrinkage.

For mixed populations (e.g. pyramidal + PV interneurons) consider
`global_ar=False` so each neuron's `g_yw` shrinks toward the prior
independently, preserving per-cell variability.

---

### `merge_thr_corr` and `merge_thr_overlap` — merging thresholds

Pairs of components with temporal correlation above `merge_thr_corr` **and** (Jaccard overlap above `merge_thr_overlap` **or** centre-of-mass distance below `merge_centre_dist_factor * sigma`) are merged.

- `merge_thr_corr=0.85` (default): merges obvious duplicates
- `merge_thr_corr=0.95`: very conservative, keeps more separate components
- `merge_thr_corr=0.7`: aggressive merging (use if many duplicate components appear)

The centre-distance fallback (default `merge_centre_dist_factor=2.0`, i.e. ~2σ in pixels) catches duplicate detections of the same neuron whose post-thresholded footprints have ended up at slightly different peak pixels and so have zero Jaccard despite tracking the same trace. Raise this factor if real distinct neurons within ~2σ are being incorrectly merged; lower it (or set 0) to disable the fallback.

---

### `ring_size_factor` — background model

Ring radius = `ring_size_factor × (2σ + 1)`. Default `1.5` works for most 1p datasets.

- Increase if background extends further than expected
- Decrease if the ring overlaps nearby neurons

---

## Inspecting Intermediate Results

You can call individual steps manually to inspect intermediate outputs:

```python
import numpy as np
from minicnmfe.preprocess import correlation_pnr, estimate_noise
from minicnmfe._utils import make_2d

movie = np.load("movie.npy").astype(np.float32)

# Check noise level
sn = estimate_noise(movie)
print(f"Median noise std: {np.median(sn):.3f}")

# Inspect CORR/PNR images
cn, pnr = correlation_pnr(movie, sigma=3.0)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].imshow(cn, cmap="hot"); axes[0].set_title("CORR")
axes[1].imshow(pnr, cmap="hot"); axes[1].set_title("PNR")
axes[2].imshow(cn * pnr, cmap="hot"); axes[2].set_title("CORR × PNR")
plt.show()
```

```python
# Check initialization seeds
from minicnmfe.initialization import detect_seeds
seeds = detect_seeds(cn, pnr, min_corr=0.8, min_pnr=10.0)
print(f"Found {len(seeds)} seed candidates")
# Plot seeds on CORR image
plt.imshow(cn, cmap="gray")
plt.scatter(seeds[:, 1], seeds[:, 0], c="red", s=10)
plt.show()
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Single module
pytest tests/test_pipeline.py -v

# Parallelism tests only
pytest tests/test_multiprocessing.py -v

# With coverage
pytest tests/ --cov=minicnmfe --cov-report=html
```

> [!NOTE]
> Tests use a synthetic movie (32×32 pixels, 150 frames, 3 neurons) generated by `tests/conftest.make_synthetic_movie()`. Tests run in a few seconds on any machine.

---

## Performance Notes

### Bottlenecks by dataset size

| Stage | Time (32×32, T=150) | Time (128×128, T=1000) | Parallelisable? |
|-------|---------------------|------------------------|-----------------|
| Motion correction | < 1 s | ~15 s | Yes |
| CORR/PNR | < 1 s | ~5 s | Yes (PSF filtering) |
| Greedy init | 1–3 s | 10–30 s | Partial (init PSF only) |
| Background W | < 1 s | ~8 s | Yes |
| Spatial update | 1–2 s / iter | 15–30 s / iter | Yes |
| Temporal update | < 1 s / iter | ~5 s / iter | Yes |

### Memory usage

The most memory-intensive variable is `Y_flat` at shape `(H·W, T)` float32.

| H×W | T | `Y_flat` size |
|-----|---|--------------|
| 128×128 | 500 | ~33 MB |
| 256×256 | 1000 | ~262 MB |
| 512×512 | 2000 | ~2 GB |

For large movies, keep them on disk as zarr and avoid `np.asarray(movie)` before necessary.

---

## Output Export

```python
import h5py
import scipy.sparse as sp

# Save to HDF5
with h5py.File("results.h5", "w") as f:
    f.create_dataset("A_data", data=model.A.data)
    f.create_dataset("A_indices", data=model.A.indices)
    f.create_dataset("A_indptr", data=model.A.indptr)
    f.attrs["A_shape"] = model.A.shape
    f.create_dataset("C", data=model.C)
    f.create_dataset("S", data=model.S)
    f.create_dataset("YrA", data=model.YrA)             # C + YrA = noisy projected trace
    f.create_dataset("sn_per_k", data=model.sn_per_k)
    # model.g is a list of variable-length arrays — pad or save individually
    f.create_dataset("g", data=np.stack(model.g))       # works when ar_order is fixed
    f.create_dataset("shifts", data=model.shifts if model.shifts is not None else [])

# Load from HDF5
with h5py.File("results.h5", "r") as f:
    A = sp.csc_matrix(
        (f["A_data"][:], f["A_indices"][:], f["A_indptr"][:]),
        shape=tuple(f.attrs["A_shape"]),
    )
    C = f["C"][:]
    S = f["S"][:]
```

---

## Troubleshooting

### No neurons found

```
Found 0 initial components.
```

- Lower `min_corr` (try `0.5`) and `min_pnr` (try `3.0`)
- Check that `sigma` matches neuron size in your data
- Inspect the CORR × PNR image — are there any bright spots at all?

### Pipeline is slow

- Set `n_jobs=-1` to use all CPUs
- Reduce `n_iter_main` to `1` while tuning parameters
- For very large movies, ensure data is stored as zarr (not loaded into RAM)

### `ValueError` on odd-shaped movies

Should not occur in the current version (floor-division fix applied). If you see it, update to the latest version.

### Duplicate neurons (same cell found twice)

- Lower `merge_thr_corr` (try `0.7`)
- Raise `merge_centre_dist_factor` (try `3.0`) so duplicates within a wider radius merge even when their thresholded supports are disjoint
- Increase `seed_suppress_factor` (try `2.5`–`3.0`) so the post-extraction suppression disk in greedy init covers more of the residual halo and prevents reseeding on the same neuron in the first place

### Temporal traces look "snappy" / poorly correlated with ground truth

Use `model.C + model.YrA` instead of `model.C` for shape comparison. `model.C` is OASIS-deconvolved (strict AR(1) shape, may distort spike timing slightly); `C + YrA` is the noisy projected trace (preserves data shape).

### `model.C` flattens into one event with a long smooth tail ("shark fin")

The AR coefficient `g` is biased upward — typically because the ring background under-subtracted slow drift and Yule-Walker reads it as a long calcium tail. Check `model.g[0][0]`: if it's near 0.96 (the default `fudge_factor` ceiling), `g` has pinned. Fix by setting `decay_time_ms` + `frame_rate_hz` on `CNMFeParams` (see [§ Parameter Tuning Guide](#decay_time_ms--frame_rate_hz--bayesian-prior-on-the-calcium-decay-g)). For 8m at 20 Hz, `g_target ≈ 0.76`.

### `model.C + model.YrA` dips below zero

Expected — not a bug. `C + YrA` is the projection of the ring-background-*subtracted* movie. The ring is a linear regression; its residual is roughly mean-zero, so half of all pixel-frames sit below the fitted background after subtraction. Plus per-pixel shot noise is centred around the bg estimate, not around zero. Negative excursions are noise, not signal — they don't indicate the decomposition is wrong. For plotting, subtract a per-row percentile (e.g. `model.C_projected - np.percentile(model.C_projected, 5, axis=1, keepdims=True)`) to float the trace above zero. CaImAn's `Y_r` has the same property for the same reasons.

### Windows multiprocessing hangs

Ensure your script has a `if __name__ == "__main__":` guard:

```python
if __name__ == "__main__":
    model = CNMFe(CNMFeParams(n_jobs=4)).fit(movie)
```

This is required for `spawn`-based multiprocessing on Windows.
