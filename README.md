# minicnmfe

A clean Python implementation of **CNMF-E** (Constrained Non-negative Matrix Factorization for Endoscopic data) for 1-photon **miniscope** calcium imaging — the standard algorithm for extracting neurons from miniscope recordings.

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
        ▼  motion_correction_rigid() — cv2 filter2D high-pass + DFT shift estimation
        │                              + cv2 warpAffine apply (matches CaImAn)
        ▼  estimate_noise()         — per-pixel noise std from high-freq PSD
        ▼  greedy_corr_pnr()        — seed detection → extract → subtract → repeat
        ▼  compute_W()              — ring-model background weights
        │
        └─ for n_iter_main:
               BackgroundSubtractor  — lazy (Y − b0) − W(Y − b0), never materialised
               update_spatial()      — per-pixel non-negative elastic-net CD
               update_temporal()     — block coordinate descent + OASIS AR deconvolution
               merge_components()    — merge spatially overlapping + correlated pairs
               compute_W()           — refresh b0 (W cached)
        │
        ▼  auto_evaluate_components() — drop ghost components (pixel-count + amplitude SNR)
        ▼  update_temporal()        — final deconvolution pass
        │
        A, C, S
```

---

## Installation

Requires Python ≥ 3.10.

```bash
git clone https://github.com/yourname/minicnmfe.git
cd minicnmfe
pip install -e .
```

**Optional — faster AR deconvolution:**

```bash
pip install -e ".[oasis]"
```

Without it the package falls back to a pure-Python PAVA implementation for AR(1).

**Optional — GPU acceleration (CuPy):**

```bash
pip install -e ".[gpu]"        # installs cupy-cuda12x
# or for CUDA 11.x:
pip install cupy-cuda11x
```

**Optional — tests / tutorials / dev:**

```bash
pip install -e ".[test]"         # pytest + pytest-cov
pip install -e ".[tutorial]"     # jupyter + ipywidgets
pip install -e ".[dev]"          # test + tutorial + oasis + ruff
```

---

## Quick start

```python
from minicnmfe import CNMFe, CNMFeParams
from minicnmfe.io import avi_to_zarr

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
from minicnmfe import CNMFe, CNMFeParams

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
| CORR / PNR images | Per-frame PSF convolution + FFT correlation |
| Ring background | Batched `linalg.solve` grouped by ring size |
| Temporal projection | `(H·W × T) @ (H·W × K)` matrix multiply |
| Greedy init | Initial per-frame PSF convolution |

Steps that stay on CPU regardless: motion correction (cv2 — bit-identical to CaImAn), OASIS deconvolution, the spatial elastic-net coordinate descent, the greedy loop.

GPU speedup is most significant for large recordings (≥ 256 × 256, T ≥ 1000 frames). On small data the CPU↔GPU transfer overhead can outweigh the benefit.

> **Windows:** `n_jobs != 1` uses `spawn`-based multiprocessing. Wrap `CNMFe(...).fit(...)` calls inside `if __name__ == "__main__":` in scripts.

---

## Key parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `sigma` | `3.0` | Neuron Gaussian radius in pixels — **set this first** |
| `min_corr` | `0.8` | Minimum local correlation to accept a seed (lower → more neurons) |
| `min_pnr` | `10.0` | Minimum peak-to-noise ratio for a seed (lower → more neurons) |
| `min_pixel` | `3` | Hard floor on footprint pixel count (auto-eval check) |
| `auto_eval_snr_amp_thr` | `3.0` | Mean(a²)/mean(sn²) threshold; drops ghost components |
| `n_iter_main` | `2` | Full spatial + temporal + merge cycles |
| `ar_order` | `1` | AR model order for calcium dynamics (1 or 2) |
| `ring_size_factor` | `1.5` | Ring radius = factor × (2σ + 1) |
| `init_stride` | auto | Temporal stride for greedy init (None = `max(1, T // 5000)`) |
| `init_patches` | `True` | Parallelize the serial greedy seed loop across overlapping FOV patches (auto-skips for small FOV / GPU / streaming movies; see usage guide) |
| `spatial_ridge` | `1e-2` | Elastic-net L2 on the per-pixel `update_spatial` LASSO; keeps the CD converging when components are correlated (`0.0` = pure LASSO) |
| `spatial_max_iter` | `1000` | Per-pixel CD iteration cap (backstop; rarely hit with `spatial_ridge`) |
| `n_jobs` | `1` | CPU workers (`-1` = all cores) |
| `device` | `"cpu"` | `"cpu"` or `"cuda"` |

---

## Experimental features (use with care)

The following pass the automated test suite (on synthetic data) but have **not
yet been validated on real recordings**. They're off by default and don't change
standard behaviour unless you enable them — but treat their output as provisional
and sanity-check before relying on it:

- **Cutout analysis** — `temporal_crop` / `spatial_crop` / `spatial_mask_path`
  crop or mask the movie at ingestion (with `place_in_full_fov()` to map results
  back). Note: not supported on the streaming `Y_flat_zarr` path.
- **Detrending** — `ar_detrend_order` / `temporal_detrend_order` (polynomial
  detrend before AR estimation / OASIS) and the `detrend_movie` preprocessor.
  Non-standard; defaults are `0` (disabled = standard CNMF-E).
- **Running many sessions concurrently** — one process per session with capped
  BLAS threads (`OMP_NUM_THREADS=1`, etc.) and a unique scratch dir per session.
  This workflow is **not covered by automated tests** — validate on a small batch
  first.

---

## Running tests

```bash
pip install -e ".[test]"
pytest tests/ -v
```

A comprehensive pytest suite (127 tests at the time of writing) covers every module plus end-to-end pipeline, motion correction streaming, multiprocessing correctness, and auto-evaluation. All tests use synthetic ground-truth movies generated in `tests/conftest.py` and `tests/miniscope_simulator.py`.

---

## Project structure

```
minicnmfe/
├── __init__.py            # Public API: CNMFe, CNMFeParams, auto_evaluate_components, ...
├── _utils.py              # Shared helpers (make_2d, get_xp, to_numpy, ...)
├── io.py                  # AVI/MP4 -> zarr converter, open/save zarr
├── motion_correction.py   # Rigid motion correction (cv2 filter2D + DFT + warpAffine)
├── preprocess.py          # Noise estimation, center-surround PSF, CORR/PNR
├── background.py          # Ring-model background (compute_W, BackgroundSubtractor)
├── initialization.py      # Greedy CORR-PNR seed detection and extraction
├── spatial.py             # Spatial footprint update (elastic-net coordinate descent per pixel)
├── temporal.py            # Temporal update + OASIS AR deconvolution
├── merging.py             # Component merging (overlap + correlation)
├── evaluate.py            # Auto-evaluation: ghost-component quality filter
└── pipeline.py            # CNMFeParams dataclass + CNMFe.fit() orchestrator

tests/                     # pytest suite (~282 tests) — synthetic ground-truth data
docs/                      # Documentation (getting-started, concepts, api, guides, tuning)
demo_movies/               # Generated demo AVIs + zarr stores (created by scripts)
demo_notebooks/            # Tutorial notebooks (see below)
generate_demo_movies.py    # Generate demo_movies/*.avi with ground-truth sidecars
convert_to_zarr.py         # Batch-convert demo_movies/*.avi -> *.zarr
full_pipeline.py           # CLI: load zarr, run full pipeline, save results to disk
```

Tutorial notebooks live in `demo_notebooks/`:

- `01_load_and_motion_correct.ipynb` — load an AVI/zarr and run motion correction
- `02_extract_components.ipynb` — full extraction pipeline + result inspection
- `old_demos/` — earlier walkthroughs (CaImAn comparison, realistic simulator, etc.)

---

## Real-data workflow (AVI files)

Miniscope recordings are typically a folder of sequentially numbered AVI files. The recommended workflow:

```bash
# 1. Concatenate 0.avi ... 65.avi into one lazy zarr store
python -m minicnmfe.concat_avis_to_zarr /path/to/recording/

# 2. Run the full pipeline and save all results
python full_pipeline.py /path/to/recording/movie.zarr --sigma 3.0 --n-jobs -1

# Results written to /path/to/recording/results/:
#   A.npz         spatial footprints (scipy CSC, H*W x K)
#   C.npy         OASIS-deconvolved traces (K x T)
#   S.npy         spike trains (K x T)
#   YrA.npy       residuals; C + YrA is the noisy projected trace (K x T)
#   shifts.npy    per-frame motion correction shifts (T x 2)
#   sn.npy        per-pixel noise std (H x W)
#   params.json   all pipeline parameters
```

For more control, the pipeline is also split into **staged CLIs** with a disk
handoff between each step (a `--params p.json` carries a `CNMFeParams` between
stages): `run_preprocess.py` (downsample) → `run_mc.py` (motion correction) →
`run_extract.py` (extraction) → `run_evaluate.py` (re-run auto-eval / retune
thresholds without re-extracting). To pick good parameters for a session, use
`tune.py <path>` (heuristics + extraction sweep + full-recording validation →
`report.html`). See [`docs/getting-started/`](docs/getting-started/index.md) and
[`docs/tuning/`](docs/tuning/index.md).

To generate demo movies and try the pipeline end-to-end:

```bash
python generate_demo_movies.py   # creates demo_movies/*.avi
python convert_to_zarr.py        # creates demo_movies/*.zarr
# then open demo_notebooks/01_load_and_motion_correct.ipynb
# followed by   demo_notebooks/02_extract_components.ipynb
```

---

## Documentation

Full documentation lives in [`docs/`](docs/index.md) and renders directly on GitHub:

| Page | Contents |
|------|----------|
| [`docs/getting-started/`](docs/getting-started/index.md) | Install, quick-start, end-to-end workflow, CLI, troubleshooting |
| [`docs/api/`](docs/api/index.md) | Every public function and `CNMFeParams` field — signatures, parameters, returns |
| [`docs/concepts/`](docs/concepts/algorithm-math.md) | Algorithm math + ELI5, architecture, ring background, CaImAn comparison |
| [`docs/guides/`](docs/guides/index.md) | Per-stage implementation walkthroughs (motion correction → evaluation) |
| [`docs/tuning/`](docs/tuning/index.md) | Automated parameter-tuning workflow + the [tuning guide](docs/tuning/guide.md) |

The `demo_notebooks/01_load_and_motion_correct.ipynb` and `demo_notebooks/02_extract_components.ipynb` walk through the full pipeline end-to-end. Earlier walkthroughs are preserved in `demo_notebooks/old_demos/`.

---

## Algorithm sources

All algorithms are reimplemented from scratch. The CaImAn repository is used as an algorithmic reference only — no code is imported.

| Algorithm | CaImAn reference | Implementation |
|-----------|-----------------|----------------|
| Rigid registration | `motion_correction.py:1442` | `cv2.filter2D` (1p high-pass) + `cv2.dft` cross-correlation |
| Shift application | `motion_correction.py:1643` | `cv2.warpAffine` (bit-identical to CaImAn) |
| CORR/PNR images | `summary_images.py:286` | `scipy.ndimage` + `numpy.fft.rfft` |
| Noise estimation | `pre_processing.py:128` | `numpy.fft.rfft` along time axis |
| Ring background | `initialization.py:1900` | vectorised ridge regression per pixel |
| GreedyCorr init | `initialization.py:1380` | reimplemented, sequential greedy loop |
| Spatial update | `spatial.py:29` | `sklearn` `enet_coordinate_descent_gram` (positive elastic-net CD) |
| Temporal update | `temporal.py:64` | coordinate descent + OASIS package |
| Deconvolution | `deconvolution.py:16` | `oasis-deconv` + pure-Python AR(1) fallback |
| Merging | `merging.py:19` | `scipy.sparse` graph connected components |
| Component eval | `estimates.evaluate_components` | pixel-count + amplitude-SNR filter (`evaluate.py`) |

---

## Dependencies

| Package | Used for |
|---------|---------|
| `numpy` | Array operations throughout |
| `scipy` | FFT, sparse matrices, ndimage filters |
| `scikit-image` | Peak detection, blob feature filters |
| `scikit-learn` | Elastic-net coordinate descent (`enet_coordinate_descent_gram`) for spatial update |
| `opencv-python` | Motion correction (`filter2D`, `dft`, `warpAffine`) |
| `av` (PyAV) | AVI decoding via imageio's pyav plugin |
| `zarr` | Lazy, chunked movie storage |
| `joblib` | CPU multiprocessing |
| `imageio` / `imageio-ffmpeg` | AVI/MP4 reading and writing |
| `matplotlib` | Plotting (tutorial notebooks) |
| `tqdm` | Progress bars |
| `oasis-deconv` *(`[oasis]` extra)* | Fast compiled OASIS AR deconvolution |
| `cupy` *(`[gpu]` extra)* | GPU acceleration |
| `pytest`, `pytest-cov` *(`[test]` extra)* | Test runner |
| `jupyter`, `ipywidgets` *(`[tutorial]` extra)* | Demo notebooks |
