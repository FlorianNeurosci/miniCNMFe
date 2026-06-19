# Algorithm overview

`minicnmfe` factorizes a motion-corrected 1-photon (miniscope) movie into a small
set of neurons. Every page in this section is written from the source in
`minicnmfe/`; the governing equations and thresholds are quoted from the code, not
from comments.

## The generative model

The movie is `Y`, a stack of `T` frames of `H × W` pixels. After initialization
the package reshapes it to a **flat pixel-major** matrix `Y ∈ ℝ^{(H·W) × T}` —
pixels are rows, time is columns (`minicnmfe/_utils.py:make_2d`, pixel `(h, w)` →
row `h·W + w`). The model is

```
Y  ≈  A · C  +  B
```

- **`A ∈ ℝ^{(H·W) × K}`** — the `K` spatial footprints, one sparse non-negative
  column per neuron.
- **`C ∈ ℝ^{K × T}`** — the `K` temporal traces (calcium fluorescence over time).
- **`B`** — the background. In CNMF-E this is the dominant term for 1-photon data
  and is modelled by a **ring**: each pixel's background is a weighted sum of the
  pixels on a ring around it, plus a per-pixel baseline `b0`
  (`B = b0 + W·(Y − b0)`, see [Ring background](background.md)). An optional
  rank-1 term `b_f · f(t)` can be added on top.

`A·C` is invariant under rescaling `A[:,k] *= s`, `C[k] /= s`. The pipeline fixes
this gauge at the very end by relabeling to **CaImAn's convention** — unit-L2-norm
footprints with the amplitude moved into the traces
(`pipeline.py:_normalize_to_trace_amplitude`).

## The pipeline

The orchestration lives in `minicnmfe/pipeline.py`. `CNMFe.fit()` is a thin
wrapper that runs motion correction in memory and then calls `fit_extract()`,
which performs every extraction step below.

1. **[Motion correction](motion-correction.md)** — rigid (translation-only)
   registration to a template (`motion_correction.py`). Run by `fit_mc` /
   `fit()`; `fit_extract` assumes its input is already corrected.
2. **Noise estimation** — per-pixel noise std `sn(H, W)` from the
   high-frequency power spectrum (`preprocess.py:estimate_noise`), computed on a
   strided frame sample (`sample_frames`).
3. **[Initialization](initialization.md)** — greedy CORR×PNR seeding finds
   neurons one at a time on a (PSF-filtered) sample of the movie
   (`initialization.py:greedy_corr_pnr`), producing the first `A`, `C`. The
   [CORR and PNR summary images](seeds-corr-pnr.md) that drive seeding are
   computed *inside* the greedy loop. (The separate global `correlation_pnr`
   call in `fit_extract` is commented out — it is a standalone diagnostic, not
   on the fit path.)
4. **[Ring background](background.md)** — fit the ring weight matrix `W` and
   baseline `b0` on the residual `X = Y − A·C − b0` (`background.py:compute_W`).
   Optional rank-1 global term (`global_bg_rank=1`).
5. **Block coordinate descent** (`n_iter_main` cycles), each iteration:
   - **[Merge](merging.md)** duplicate seeds (an extra pre-merge on iteration 0).
   - **[Spatial update](spatial-update.md)** — re-fit each footprint by a
     per-pixel non-negative coordinate-descent LASSO on the
     background-subtracted movie (`spatial.py:update_spatial`).
   - **[Temporal update](temporal-update.md)** — re-project each trace and
     deconvolve it with OASIS (`temporal.py:update_temporal`).
   - Refresh the baseline `b0` (the ring weights `W` are reused across
     iterations).
6. **Final temporal pass + residual** — one more deconvolution, then the
   residual-at-footprint `YrA` so that `C + YrA` is the shape-faithful "noisy
   projected" trace.
7. **[Auto-evaluation](evaluation.md)** — a non-destructive quality tag
   (`accepted_mask`); no component is ever dropped.

## The two trace flavours

Extraction returns two views of each neuron's activity (`pipeline.py`):

- **`model.C`** — OASIS-deconvolved, obeys the AR shape constraint
  `c[t] ≥ g·c[t−1]`. Use for spike-event timing.
- **`model.C + model.YrA`** — the data projected onto each footprint (residual
  added back). Same shape as the raw fluorescence; use for shape-faithful
  comparison and regression.
