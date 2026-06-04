# CaImAn vs this CNMFe — benchmarking notes

What we learned comparing this from-scratch CNMFe reimplementation against
CaImAn's CNMF-E (`caiman 1.13.1`). Read this before quoting any "vs CaImAn"
number — several early "striking differences" turned out to be **measurement
artifacts**, and the one real edge is partly a **home-field effect**.

Reproduce: `demo_notebooks/tutorial_caiman_compare_patches.ipynb` (the
orientation-correct, per-pipeline-tuned, patch-based comparison).

---

## Bottom line

- **Clean synthetic data (Gaussian footprints, white noise, rank-1 bg):** on par.
  After fixing the comparison artifacts below, footprints, detection, and both
  trace flavours match CaImAn within ~0.01–0.02 Pearson r; we are ~10× faster on
  these small movies (with the speed caveat below).
- **Realistic simulator (`make_miniscope_movie`):** we are *modestly* ahead on
  the temporal trace — but ablations show this is **largely because the
  simulator's background suits our model**, not a general CaImAn deficiency.
- **Real recordings:** unknown. This implementation has **only** been tested on
  synthetic data; CaImAn is validated on hundreds of real 1p recordings
  (some with paired electrophysiology). Do **not** generalise the synthetic
  results to real data.

---

## Comparison-fairness issues we found (and fixed)

Matching parameter *values* is not the same as a fair comparison. These all made
CaImAn look worse than it is until corrected:

1. **Footprint C-order vs Fortran-order mismatch.** Our `A_true`/`A` are
   C-flattened (`h*W+w`); CaImAn returns `A` Fortran-flattened (`h+w*H`).
   Comparing them directly transposes CaImAn's footprints — near-harmless for
   round Gaussians, but on the realistic movie's elliptical footprints it pinned
   CaImAn's mean spatial r at **0.30** (and "recovered" at **3/16**); the
   orientation fix raised it to **0.71** and **12/16**. *Fix:* map CaImAn's `A`
   into true `(h,w)` space and
   **auto-detect** the reshape order by centroid alignment to truth (also catches
   any memmap movie-feed transpose). Verified: order `F` aligns at ~1 px, `C` at
   ~7 px.
2. **Threshold operating-point mismatch.** `min_corr`/`min_pnr` copied from our
   params land at the wrong place on CaImAn's corr/pnr images (e.g. our
   `min_corr=0.8` sat *above* CaImAn's corr p99 of 0.78 → starved seeding).
   *Fix:* tune CaImAn's thresholds from **its own** `correlation_pnr`
   distribution; keep ours as-is (each algorithm at its own operating point).
3. **No-patch CaImAn never refines its ring background.** CaImAn's `compute_W`
   (ring-`W` recomputation, the analogue of our per-iteration `b0` refresh) runs
   **only in the patch path**. *Fix:* run CaImAn in patches (`rf`/`stride`) on a
   FOV large enough for a real patch grid (≥128 px); 64×64 is below its regime.
4. **(ours) PAVA deconvolution fallback bug.** When the `oasis-deconvolution`
   package is absent, we fell back to a pure-Python PAVA whose pool-merge test
   used bare `g` instead of `g**pool_length` — collapsing smooth decays. This
   dropped our **deconvolved `C`** vs truth to ~0.58 while CaImAn's was ~0.98,
   hidden behind the robust `C+YrA` (~0.96). *Fixed* in `minicnmfe/temporal.py`
   (commit `b90f032`); guarded by
   `tests/test_temporal.py::test_pava_fallback_reconstructs_clean_ar1`. After the
   fix our `C` is ~0.96, on par with CaImAn.

Still **intentionally not equalised:** `evaluate_components` (CaImAn's
CNN+SNR+spatial-correlation quality filter) is not called — the comparison is
raw output vs raw output. On real data that filter is CaImAn's main precision
advantage, so omitting it flatters whichever side has fewer safeguards.

---

## Results (after the fixes)

Synthetic, K=24, 128×128. `ours(2)` = `n_iter_main=2`; `caiman` = patch CNMF-E,
thresholds tuned to its own corr/pnr.

**Clean synthetic movie:**

| metric | ours(2) | caiman |
|---|---|---|
| K extracted | 24 | 24 |
| matched (>0.7 spatial) | 24/24 | 24/24 |
| mean spatial r | 0.982 | 0.994 |
| mean temporal r (`C`) | 0.970 | 0.982 |
| mean temporal r (`C+YrA`) | 0.970 | 0.978 |
| wall time | 16 s | 153 s |

**Realistic miniscope movie** (drifting multi-component bg, ghosts, vasculature,
vignette, photobleach, shot noise, 8-bit; `g_true ∈ [0.73,0.78]`):

| metric | ours(2) | caiman |
|---|---|---|
| spatial r | 0.943 | 0.925 |
| recovered | 24/24 | 23/24 |
| `C+YrA` r | **0.851** | **0.710** |

---

## Why CaImAn trails on the realistic movie (diagnosed by ablation)

The gap is **temporal (`C+YrA`), not spatial or detection.** Per-neuron, CaImAn's
footprints are good (spatial 0.84–0.99) but a *subset* of its projected traces
are contaminated despite good footprints — the signature of background leaking
into the residual at certain locations. Ablating one confound at a time
(`make_miniscope_movie`, median `C+YrA` gap ours−caiman):

| movie | median gap |
|---|---|
| baseline | 0.087 |
| no photobleach | 0.081 |
| no vasculature | 0.091 |
| **rank-1 background** | **0.020** |

Collapsing the background to rank-1 nearly closes the gap; removing bleach or
vasculature does not. **So the cause is the multi-component structured
background.** And it is largely a **home-field effect**: the simulator builds its
background as a sum of *separable rank-1 components*, structurally close to our
ring + per-iteration `b0`-refresh model (developed against this very simulator)
and simpler than real neuropil/hemodynamics. CaImAn's ring is tuned for *real* 1p
background. Expect this edge to shrink on real recordings.

(The *mean* `C+YrA` gap is further inflated by the one missed neuron being matched
to a wrong component; the median strips that out. A secondary wrinkle: the
per-movie threshold auto-tuner is sensitive — when the corr/pnr distribution
shifts it can pick a too-high `min_pnr` and hurt CaImAn's recall.)

---

## Genuine algorithmic differences (not bugs)

| aspect | this CNMFe | CaImAn CNMF-E |
|---|---|---|
| Background | ring + per-iteration `b0` refresh (+ optional rank-1 `b_f·f`) | ring (`compute_W`), refined in the patch path |
| Deconvolution | OASIS via `oasis-deconvolution`; pure-Python PAVA fallback | OASIS (`constrained_foopsi`) |
| AR `g` | estimated **once** from `C_raw`, cached (global or per-neuron); optional Bayesian `decay_time`/`fps` prior | per-neuron, per-iteration |
| Spatial solve | per-pixel non-neg LASSO on **raw** traces | LASSO on `StandardScaler`-normalised traces |
| Quality filter | non-destructive SNR-amplitude auto-eval (`accepted_mask`) | CNN + SNR + spatial-corr `evaluate_components` |
| Init | greedy corr/pnr (+ optional patch-parallel) | corr/pnr in patches |

Speed caveat: the ~10× wall-time advantage is partly because the comparison runs
CaImAn with `N_PROCESSES=1` + patches (sequential); CaImAn parallelises across
patches, so the gap narrows with more processes.

---

## How to read any "vs CaImAn" claim from this repo

Defensible statement: *"On synthetic benchmarks with a matched,
orientation-corrected, per-pipeline-tuned setup, this reimplementation achieves
recovery comparable to CaImAn's raw CNMF-E output (and is faster on small
movies). It has not been validated on real recordings or against CaImAn's full
evaluation pipeline."*

What would justify a stronger claim: real 1p recordings **not** produced by our
own simulator, with external ground truth (paired ephys or expert ROIs), running
CaImAn's **full** pipeline including `evaluate_components`.
