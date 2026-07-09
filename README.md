> ⚠️ **Disclaimer** — This pipeline is an experiment and written almost exclusively by an AI agent under human supervision. It is still under construction and the documentation so far has been written by AI. The results have been validated against Caiman and on synthetic data. However, use it under your own risk and assess results for plausibility. If you find errors please raise an issue so that they can be fixed! Enjoy and stay tuned for changes :)

# minicnmfe

Minicnmfe is a clean Python implementation of **CNMF-E** (Constrained Non-negative Matrix Factorization for Endoscopic data) for 1-photon **miniscope** calcium imaging — the standard algorithm for extracting neurons from miniscope recordings. It is heavily inspired by CaImAn (https://github.com/flatironinstitute/caiman) but does not import anything from the caiman library.

Apart from it being an experiment, the focus of this pipeline is on **automatic parameter estimation from raw videos** — the `tuning/` library in this repository.

> **The notebooks are the intended way to learn this pipeline.** Everything under
> [`docs/`](docs/index.md) is AI-written and has not been reviewed line-by-line — treat it
> as a convenience reference, not ground truth. The four numbered notebooks in
> [`demo_notebooks/`](demo_notebooks/) run end-to-end on generated demo movies and
> validate their own output, so they are the reliable place to understand how to use the
> pipeline. Read [Notebooks — start here](#notebooks--start-here) first.

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

## Installation

Requires Python ≥ 3.10.

```bash
git clone https://github.com/FlorianNeurosci/miniCNMFe.git
cd miniCNMFe
pip install -e .
```

**AR deconvolution (`oasis-deconv`)** ships as a core dependency, so a plain
`pip install -e .` already gets the fast compiled OASIS path. The `[oasis]` extra
is kept only for back-compat:

```bash
pip install -e ".[oasis]"   # equivalent — oasis-deconv is already a core dep
```

If `oasis-deconv` is unavailable, the package falls back to a pure-Python PAVA
implementation for AR(1).

**Optional — tests / tutorials / dev:**

```bash
pip install -e ".[test]"         # pytest + pytest-cov
pip install -e ".[tutorial]"     # jupyter + ipywidgets
pip install -e ".[dev]"          # test + tutorial + oasis + ruff
```

---

## Notebooks — start here

The notebooks in [`demo_notebooks/`](demo_notebooks/) are the primary way to learn the
pipeline. They run end-to-end on the reproducible simulated movies, so after a fresh clone
you can execute them from top to bottom without any data of your own:

```bash
python generate_demo_movies.py   # creates demo_movies/*.avi + ground-truth *_meta.npz
python convert_to_zarr.py        # creates demo_movies/*.zarr
```

Then work through them **in order** — this is the real-data workflow: load & motion-correct
→ tune → extract → advanced.

| # | Notebook | What it covers |
|---|----------|----------------|
| 1 | [`01_load_and_motion_correct.ipynb`](demo_notebooks/01_load_and_motion_correct.ipynb) | Ingest a session (numbered AVIs → one zarr) and run **streaming rigid motion correction**; verify the recovered shifts against the simulated ground-truth drift. |
| 2 | [`02_tuning.ipynb`](demo_notebooks/02_tuning.ipynb) | **Pick parameters** (`sigma`, seed thresholds, downsample factors) with the automated tuner, validated against ground truth on a calibrated movie. Run this *before* extraction. |
| 3 | [`03_extract_components.ipynb`](demo_notebooks/03_extract_components.ipynb) | The **full extraction** (seeding → ring background → spatial/temporal updates → merging → auto-eval) on a clean movie, scored against ground-truth footprints and traces. |
| 4 | [`04_advanced_features.ipynb`](demo_notebooks/04_advanced_features.ipynb) | **Production knobs** for real recordings: `nrg` footprint tightening, region cutouts, downsample-once + upsample, the rank-1 global background, and fused AVI → motion correction. |

---

## Analyze your own data

A miniscope session is typically a folder of sequentially numbered AVI files. The
recommended path mirrors the notebook order — **motion-correct → tune → extract → refine**:

1. **Load & motion-correct** (notebook 1) — concatenate the AVIs and run streaming rigid
   motion correction, producing a corrected `mc.zarr`. The fused
   `CNMFe.fit_mc_from_avis(folder, output_dir=...)` does decode + MC in one pass.
2. **Tune** (notebook 2) — let the automated tuner pick `sigma`, seed thresholds and
   downsample factors for *your* recording, rather than guessing. From the CLI:

   ```bash
   python tune.py /path/to/session   # heuristics + extraction sweep + validation -> report.html + recommended_params.json
   ```

   (The `/tune-session` skill wraps the same workflow.) **Set the seed thresholds from the
   recording's own CORR/PNR distribution** — the single most important habit on real data.
3. **Extract** (notebook 3) — run the full pipeline with those parameters and save `A`,
   `C`, `S`, `YrA`.
4. **Refine** (notebook 4) — reach for the production knobs (footprint tightening, rank-1
   global background, cutouts) on long, dense or drifting recordings.

### Scripted / CLI equivalents

For batch or headless runs the same workflow is available as scripts:

```bash
# One-shot: concatenate AVIs, run the full pipeline, save results/
python -m minicnmfe.concat_avis_to_zarr /path/to/recording/
python full_pipeline.py /path/to/recording/movie.zarr --sigma 3.0 --n-jobs -1
```

Results are written to `/path/to/recording/results/`:

```
A.npz         spatial footprints (scipy CSC, H*W x K)
C.npy         OASIS-deconvolved traces (K x T)
S.npy         spike trains (K x T)
YrA.npy       residuals; C + YrA is the noisy projected trace (K x T)
shifts.npy    per-frame motion correction shifts (T x 2)
sn.npy        per-pixel noise std (H x W)
params.json   all pipeline parameters
```

For finer control, the pipeline is split into **staged CLIs** with a disk handoff between
each step (`--params p.json` carries a `CNMFeParams` between stages):
`run_preprocess.py` (downsample) → `run_mc.py` (motion correction) → `run_extract.py`
(extraction) → `run_evaluate.py` (re-run auto-eval / retune thresholds without
re-extracting). See [`docs/getting-started/`](docs/getting-started/index.md) and
[`docs/tuning/`](docs/tuning/index.md).

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
        ▼  auto_evaluate_components() — record per-component quality (gate OFF by default; opt-in ghost tagging)
        ▼  update_temporal()        — final deconvolution pass
        │
        A, C, S
```

---

## Minimal API example

The notebooks above are the recommended, guided way to run the pipeline. This is the
minimal programmatic form:

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

## Parallelism

```python
params = CNMFeParams(sigma=3.0, n_jobs=-1)   # use all available CPU cores
```

Every bottleneck step that can be parallelised — the ring-background solve, the
per-pixel spatial elastic-net coordinate descent, temporal OASIS deconvolution,
and the per-frame motion-correction / CORR-PNR work — respects `n_jobs`. The
default is `1` (serial); `-1` uses all cores.

> **Windows:** `n_jobs != 1` uses `spawn`-based multiprocessing. Wrap `CNMFe(...).fit(...)` calls inside `if __name__ == "__main__":` in scripts.

---

## Key parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `sigma` | `3.0` | Neuron Gaussian radius in pixels — **set this first** |
| `min_corr` | `0.8` | Minimum local correlation to accept a seed (lower → more neurons) |
| `min_pnr` | `10.0` | Minimum peak-to-noise ratio for a seed (lower → more neurons) |
| `min_pixel` | `3` | Greedy-init footprint pixel floor (also the gate's pixel check when opted in) |
| `auto_eval_snr_amp_thr` | `0.0` | Acceptance-gate SNR threshold; **`0` = gate off** (report-only). Raise (~`3`) to opt in to ghost tagging |
| `n_iter_main` | `2` | Full spatial + temporal + merge cycles |
| `ar_order` | `1` | AR model order for calcium dynamics (1 or 2) |
| `ring_size_factor` | `1.5` | Ring radius = factor × (2σ + 1) |
| `init_stride` | auto | Temporal stride for greedy init (None = `max(1, T // 5000)`) |
| `init_patches` | `True` | Parallelize the serial greedy seed loop across overlapping FOV patches (auto-skips for small FOV / streaming movies; see usage guide) |
| `spatial_ridge` | `1e-2` | Elastic-net L2 on the per-pixel `update_spatial` LASSO; keeps the CD converging when components are correlated (`0.0` = pure LASSO) |
| `spatial_max_iter` | `1000` | Per-pixel CD iteration cap (backstop; rarely hit with `spatial_ridge`) |
| `n_jobs` | `1` | CPU workers (`-1` = all cores) |

Pick these automatically for a recording with `tune.py` (see [Analyze your own data](#analyze-your-own-data)).

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

A comprehensive pytest suite (300+ tests) covers every module plus end-to-end pipeline, motion correction streaming, multiprocessing correctness, and auto-evaluation. All tests use synthetic ground-truth movies generated in `tests/conftest.py` and `tests/miniscope_simulator.py`.

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
├── evaluate.py            # Auto-evaluation: per-component quality metrics (opt-in gate)
└── pipeline.py            # CNMFeParams dataclass + CNMFe.fit() orchestrator

tests/                     # pytest suite (300+ tests) — synthetic ground-truth data
docs/                      # AI-written documentation (getting-started, concepts, api, guides, tuning)
demo_movies/               # Generated demo AVIs + zarr stores (created by scripts)
demo_notebooks/            # Tutorial notebooks (see "Notebooks — start here" above)
generate_demo_movies.py    # Generate demo_movies/*.avi with ground-truth sidecars
convert_to_zarr.py         # Batch-convert demo_movies/*.avi -> *.zarr
full_pipeline.py           # CLI: load zarr, run full pipeline, save results to disk
```

---

## Documentation

> The prose docs below were written by AI and have not been reviewed line-by-line. They
> are a convenience reference — the [notebooks](#notebooks--start-here) are the trusted,
> executable source of truth for how to use the pipeline.

Documentation lives in [`docs/`](docs/index.md) and renders directly on GitHub:

| Page | Contents |
|------|----------|
| [`docs/getting-started/`](docs/getting-started/index.md) | Install, quick-start, end-to-end workflow, CLI, troubleshooting |
| [`docs/api/`](docs/api/index.md) | Every public function and `CNMFeParams` field — signatures, parameters, returns |
| [`docs/concepts/`](docs/concepts/algorithm-math.md) | Algorithm math + ELI5, architecture, ring background, CaImAn comparison |
| [`docs/guides/`](docs/guides/index.md) | Per-stage implementation walkthroughs (motion correction → evaluation) |
| [`docs/tuning/`](docs/tuning/index.md) | Automated parameter-tuning workflow + the [tuning guide](docs/tuning/guide.md) |

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
| `oasis-deconv` | Fast compiled OASIS AR deconvolution (core dep; pure-Python PAVA fallback if unavailable) |
| `numba` | GIL-free per-pixel coordinate descent in the spatial update |
| `pytest`, `pytest-cov` *(`[test]` extra)* | Test runner |
| `jupyter`, `ipywidgets` *(`[tutorial]` extra)* | Demo notebooks |
