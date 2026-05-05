# CNMFe

A clean Python implementation of **Constrained Non-negative Matrix Factorization for Endoscopic data** (CNMFe) — the standard algorithm for extracting neurons from 1-photon calcium imaging recordings.

No CaImAn code imported. The CaImAn source is used as an algorithmic reference only; all math is reimplemented from scratch using numpy / scipy / scikit-image / scikit-learn.

---

## What it does

Given a raw miniscope recording, CNMFe returns:

| Output | Shape | Description |
|--------|-------|-------------|
| `A` | `(H·W, K)` sparse | Spatial footprints — where each neuron lives |
| `C` | `(K, T)` | Denoised calcium traces |
| `S` | `(K, T)` | Inferred spike trains |

The pipeline handles the core challenge of 1-photon imaging: a large, spatially-correlated background that swamps individual neuron signals. It uses a **ring-model background** — per-pixel ridge regression on a ring of surrounding pixels — to separate neural signal from diffuse background before source extraction.

---

## Pipeline

```
AVI / zarr movie  (T × H × W)
        │
        ▼  motion_correct()       — FFT phase cross-correlation, subpixel shifts
        ▼  estimate_noise()       — per-pixel noise std from high-freq PSD
        ▼  correlation_pnr()      — CORR and PNR summary images
        ▼  greedy_corr_pnr()      — seed detection → extract → subtract → repeat
        ▼  compute_W()            — ring-model background weights
        │
        └─ for n_iter_main:
               subtract_background()
               update_spatial()   — per-pixel non-negative LassoLars
               update_temporal()  — block coordinate descent + OASIS AR deconvolution
               merge_components() — merge spatially overlapping + correlated pairs
               compute_W()        — refresh background
        │
        ▼  update_temporal()      — final deconvolution pass
        │
        A, C, S
```

---

## Installation

Requires Python ≥ 3.10.

```bash
git clone https://github.com/yourname/cnmfe.git
cd cnmfe
pip install -e .
```

**Optional — faster AR deconvolution:**

```bash
pip install oasis-deconvolution
```

Without it the package falls back to a pure-Python PAVA implementation for AR(1).

**Optional — GPU acceleration (CuPy):**

```bash
pip install cupy-cuda12x   # CUDA 12.x
pip install cupy-cuda11x   # CUDA 11.x
```

---

## Quick start

```python
from cnmfe import CNMFe, CNMFeParams
from cnmfe.io import avi_to_zarr

# Convert video to zarr (streams frame-by-frame, never loads full movie)
movie = avi_to_zarr("recording.avi", "/tmp/movie.zarr")

params = CNMFeParams(
    sigma=3.0,      # neuron radius in pixels — most important parameter
    min_corr=0.8,   # minimum local correlation for a seed
    min_pnr=10.0,   # minimum peak-to-noise ratio for a seed
)
model = CNMFe(params).fit(movie)

print(f"Found {model.A.shape[1]} neurons")
A = model.A   # (H*W, K) sparse — spatial footprints
C = model.C   # (K, T)   — calcium traces
S = model.S   # (K, T)   — spike trains
```

From a numpy array (no video file):

```python
import numpy as np
from cnmfe import CNMFe, CNMFeParams

movie = np.load("movie.npy")   # (T, H, W) float32
model = CNMFe(CNMFeParams(sigma=3.0)).fit(movie, do_motion_correction=False)
```

---

## Parallelism and GPU

```python
# Multi-core CPU
params = CNMFeParams(sigma=3.0, n_jobs=-1)   # all available cores

# GPU (requires CuPy + CUDA)
params = CNMFeParams(sigma=3.0, device="cuda")

# Both
params = CNMFeParams(sigma=3.0, n_jobs=4, device="cuda")
```

**What runs on GPU** (`device="cuda"`):

| Step | GPU operation |
|------|--------------|
| Motion correction | FFT2 + phase multiply (`apply_shift`) |
| CORR / PNR images | Per-frame PSF convolution + FFT correlation |
| Ring background | Batched `linalg.solve` grouped by ring size |
| Temporal projection | `(H·W × T) @ (H·W × K)` matrix multiply |
| Greedy init | Initial per-frame PSF convolution |

Steps that stay on CPU regardless: shift estimation (skimage), OASIS deconvolution, LassoLars, the greedy loop.

GPU speedup is most significant for large recordings (≥ 256 × 256, T ≥ 1000 frames). On small data the CPU↔GPU transfer overhead can outweigh the benefit.

> **Windows:** `n_jobs != 1` uses `spawn`-based multiprocessing. Wrap `CNMFe(...).fit(...)` calls inside `if __name__ == "__main__":` in scripts.

---

## Key parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `sigma` | `3.0` | Neuron Gaussian radius in pixels — **set this first** |
| `min_corr` | `0.8` | Minimum local correlation to accept a seed (lower → more neurons) |
| `min_pnr` | `10.0` | Minimum peak-to-noise ratio for a seed (lower → more neurons) |
| `n_iter_main` | `2` | Full spatial + temporal + merge cycles |
| `ar_order` | `1` | AR model order for calcium dynamics (1 or 2) |
| `ring_size_factor` | `1.5` | Ring radius = factor × (2σ + 1) |
| `n_jobs` | `1` | CPU workers (`-1` = all cores) |
| `device` | `"cpu"` | `"cpu"` or `"cuda"` |

---

## Running tests

```bash
pip install -e ".[test]"
pytest tests/ -v
```

76 tests covering every module plus end-to-end pipeline and multiprocessing correctness. All tests use a synthetic ground-truth movie generated in `tests/conftest.py`.

---

## Project structure

```
cnmfe/
├── __init__.py            # Public API: CNMFe, CNMFeParams, load_movie
├── _utils.py              # Shared helpers (make_2d, get_xp, to_numpy, …)
├── io.py                  # AVI/MP4 → zarr converter, open/save zarr
├── motion_correction.py   # Rigid motion correction (FFT phase correlation)
├── preprocess.py          # Noise estimation, center-surround PSF, CORR/PNR
├── background.py          # Ring-model background (compute_W, subtract_background)
├── initialization.py      # Greedy CORR-PNR seed detection and extraction
├── spatial.py             # Spatial footprint update (LassoLars per pixel)
├── temporal.py            # Temporal update + OASIS AR deconvolution
├── merging.py             # Component merging (overlap + correlation)
└── pipeline.py            # CNMFeParams dataclass + CNMFe.fit() orchestrator

tests/                     # pytest suite — synthetic ground-truth data
wiki/                      # Obsidian-optimised documentation
tutorial.ipynb             # End-to-end walkthrough with explanations
```

---

## Documentation

The `wiki/` directory contains Obsidian-optimised documentation:

| File | Contents |
|------|----------|
| `wiki/algorithm-math.md` | Full mathematical derivation (LaTeX) |
| `wiki/algorithm-eli5.md` | Plain-English explanation with analogies |
| `wiki/architecture.md` | Module map, dependency graph, data-flow diagram |
| `wiki/api-reference.md` | Every public function — signatures, parameters, returns |
| `wiki/usage-guide.md` | Quick-start, parameter tuning, troubleshooting |

The `tutorial.ipynb` notebook walks through each pipeline step interactively, explaining the *why* behind every decision.

---

## Algorithm sources

All algorithms are reimplemented from scratch. The CaImAn repository is used as an algorithmic reference only — no code is imported.

| Algorithm | CaImAn reference | Implementation |
|-----------|-----------------|----------------|
| FFT registration | `motion_correction.py:1442` | `skimage.registration.phase_cross_correlation` |
| Fourier shift | `motion_correction.py:1643` | `numpy.fft` + phase multiplication |
| CORR/PNR images | `summary_images.py:286` | `scipy.ndimage` + `numpy.fft.rfft` |
| Noise estimation | `pre_processing.py:128` | `numpy.fft.rfft` along time axis |
| Ring background | `initialization.py:1900` | vectorised ridge regression per pixel |
| GreedyCorr init | `initialization.py:1380` | reimplemented, sequential greedy loop |
| Spatial update | `spatial.py:29` | `sklearn.linear_model.LassoLars` |
| Temporal update | `temporal.py:64` | coordinate descent + OASIS package |
| Deconvolution | `deconvolution.py:16` | `oasis-deconvolution` + pure-Python AR(1) fallback |
| Merging | `merging.py:19` | `scipy.sparse` graph connected components |

---

## Dependencies

| Package | Used for |
|---------|---------|
| `numpy` | Array operations throughout |
| `scipy` | FFT, sparse matrices, ndimage filters |
| `scikit-image` | Phase cross-correlation, peak detection |
| `scikit-learn` | LassoLars for spatial update |
| `zarr` | Lazy, chunked movie storage |
| `joblib` | CPU multiprocessing |
| `imageio` / `imageio-ffmpeg` | AVI/MP4 reading |
| `matplotlib` | Plotting (tutorial notebook) |
| `tqdm` | Progress bars |
| `oasis-deconvolution` *(optional)* | Fast compiled OASIS AR deconvolution |
| `cupy` *(optional)* | GPU acceleration |
