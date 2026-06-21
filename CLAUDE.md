# CLAUDE.md — Project context for Claude Code

This file is read automatically at the start of every Claude Code session.
It captures project state, decisions, and caveats that are not obvious from the code alone.

---

## What this project is

**minicnmfe** is a **clean Python reimplementation of CNMFe** (Constrained NMF for
Endoscopic data) for extracting neurons from 1-photon calcium imaging (miniscope)
recordings. The Python package is imported as `minicnmfe` (`from minicnmfe import
CNMFe, CNMFeParams`).

**No CaImAn code is imported.** The CaImAn repository (`CaImAn-main/`) is present as an
algorithmic reference only. All math is reimplemented from scratch with numpy/scipy/sklearn.

---

## Current status

**Complete and working.** All 282 tests pass:

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
- Documentation in `docs/` (getting-started, concepts, api, guides, tuning) — GitHub-renderable, updated for all of the above

### Experimental — passing automated tests, NOT yet validated on real data

These features work in the pytest suite (synthetic fixtures) but have **not been
exercised by the maintainer on a real recording**. Treat results as provisional
and sanity-check before relying on them. Defaults are chosen so none of these
change standard behaviour unless explicitly enabled.

1. **Cutout analysis** — `CNMFeParams.temporal_crop` / `spatial_crop` /
   `spatial_mask_path` (`minicnmfe/cutout.py`), applied at ingestion before MC, plus
   `place_in_full_fov()` to map results back. Autotested (`tests/test_cutout.py`,
   incl. end-to-end fit + fused AVI→MC). **Caveat:** incompatible with the
   streaming `Y_flat_zarr` path (raises by design); not validated on real data.
2. **Detrending** — the polynomial-detrend knobs `ar_detrend_order` (before
   Yule-Walker) and `temporal_detrend_order` (before OASIS), and the standalone
   `minicnmfe/detrend.py:detrend_movie` preprocessor. NON-STANDARD; defaults `0`
   (off) = standard CNMF-E. Autotested (`tests/test_detrend.py`,
   `tests/test_temporal.py`, a pipeline run with orders=2) but not validated on
   real recordings — tune cautiously.
3. **Parallel sessions** — the concurrent multi-session recipe (see *Running
   many sessions concurrently* below). **Not autotested AND not run end-to-end**
   — the BLAS-cap and unique-`yflat_dir` requirements are reasoned from the code,
   not verified. The weakest-tested of the three; validate on a small batch
   first.
4. **Parameter tuning** (`tuning/`, `tune.py`, `live_runs/tune.ipynb`,
   `docs/tuning/guide.md`) — one-path-in workflow that suggests MC +
   extraction params and writes a report folder (`recommended_params.json` +
   `report.md` + figures). Heuristics are lifted verbatim from the maintainer's
   `estimate_params.ipynb`; the graded extraction **sweep** + ground-truth-free
   quality proxies (`tuning/metrics.py`: `cprojcorr_median`, accepted frac,
   footprint npix, SNR) + report rendering are new. Autotested end-to-end
   (`tests/test_tuning.py`, 8 tests on the simulator + a 2-file AVI fixture).
   **Caveats:** the quality scores are *proxies, not validation* (the roadmap
   C1/C2 harness would add real validation, and could reuse `tuning/metrics.py`);
   the sweep's cutout values are a fast approximation to a full-recording run;
   not yet exercised on a real recording end-to-end.
   *Update (June 2026):* validated end-to-end on a real 60k-frame PICAST session
   — see `live_runs/tuning_picast/LEARNINGS.md` (confirmed `global_bg_rank=1` is
   the dominant long-recording win; surfaced the decay-time drift-inflation and
   full-vs-cutout recall caveats). The methodology is now packaged as the
   `/tune-session` skill + the reusable `validate_session.py` /
   `tuning.validate.validate_session` (full MC + Y_flat built once, reused across
   threshold sets). Autotested (`tests/test_validate.py`).
5. **Cross-session cell registration (CellReg)** — `minicnmfe/cellreg.py`
   (`register_sessions`, `CellRegResult`) + the `run_cellreg.py` CLI. A clean
   from-scratch reimplementation of the Ziv lab's CellReg (Sheintuch et al.,
   2017) that tracks the same neurons across sessions. Autotested end-to-end
   (`tests/test_cellreg.py`, 11 tests on synthetic Gaussian footprints:
   alignment recovery, matching precision/recall, conflict resolution, N=3
   clustering, the `P_same` model, save/load). **Caveat:** not yet validated on a
   real multi-session recording; defaults (`align="translation"`,
   threshold-matching) are conservative. See the design note below.

---

## Tech stack

| Concern | Choice | Notes |
|---------|--------|-------|
| Movie storage | zarr v3 | Time-chunked, lazy, random-access |
| Deconvolution | `oasis-deconv` package | Pure-Python PAVA AR(1) fallback if not installed |
| CPU parallelism | `joblib` / `loky` | All parallel workers defined at module level (pickling requirement) |
| GPU | `cupy` (optional) | `get_xp(device)` in `_utils.py` returns numpy or cupy |
| Spatial solve | `sklearn` `enet_coordinate_descent_gram` (positive elastic-net CD, L1 + `spatial_ridge` L2) | CPU-only, no GPU equivalent |
| OASIS | sequential PAVA | Cannot be GPU-accelerated (inherently sequential) |
| mp4 export | `opencv-python` (cv2) preferred, imageio with explicit codec as fallback | imageio's `pyav` plugin without an explicit codec fails on Windows envs |
| Tests | pytest | Synthetic ground-truth movies in `tests/conftest.py` and `tests/miniscope_simulator.py` |

---

## Key design decisions

### Motion correction — canonical implementation
`motion_correction_rigid` in `minicnmfe/motion_correction.py` is the **only** motion
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

**Streaming IO tuning (network vs SSD).** The streaming BCD loop makes ~5–6
full passes over the on-disk `Y_flat` store (`compute_W` ×2, `update_spatial`,
`update_temporal`/`project_onto` ×several, final YrA), so its cost is dominated
by *store IO*, not compute. Tuning knobs — all of which affect **only IO speed,
never the extracted results** (guarded by `tests/test_pipeline.py::test_fit_*Y_flat_zarr*`):

- **`CNMFeParams.yflat_dir`** — where the auto-derive path
  (`fit_extract(zarr, output_dir=...)` / `run_extract.py`) writes
  `Y_flat_pixel.zarr`. Default `None` = under `output_dir`. **On a network
  mount, point this at a local SSD/tmpfs:** the transpose reads `mc.zarr` from
  the network *once*, writes `Y_flat` locally, and all BCD passes then read
  locally. Usually the single biggest network win.
- **`yflat_pixel_chunk` (512)** / **`yflat_time_chunk` (None = full T)** — dest
  chunk shape, tuned for the contiguous-pixel-row / full-time read pattern. The
  old hardcoded `4096×2000` over-read ~16× because the read batches are 256/4096
  pixels. Smaller `pixel_chunk` = less pixel over-read; full-T time chunk = no
  time-axis amplification. Caveat: chunk bytes = `pixel_chunk·T·4` (cap
  `time_chunk` for very long T).
- **`yflat_compression` (True)** — keep `True` on a **network** mount (fewer
  bytes over the wire); try `False` on a **local SSD** (skips per-read
  decompression, IO-bound only; costs ~`H·W·T·4` bytes of disk).
- **`n_jobs=-1`** — parallelises the per-batch store reads + solves (default is
  `1`, serial). On a high-latency network this hides round-trip latency.
- **`minicnmfe.io.stage_zarr_to_local(src, local_dir)`** — copies any zarr store to
  local disk and returns the open handle, for users who already have a
  network-resident `Y_flat` (pass the local handle as `Y_flat_zarr=`). Equivalent
  to setting `yflat_dir` on the auto-derive path.
- **`run_extract.py`** exposes `--yflat-dir`, `--yflat-pixel-chunk`,
  `--yflat-time-chunk`, `--yflat-no-compress`.

`fit_extract` prints a **per-stage wall-clock summary** at the end (via
`minicnmfe._utils.StageTimer`): `transpose -> Y_flat`, `compute_W`, `update_spatial`,
`update_temporal`, `final YrA projection`, etc. — use it to see where the IO
time actually goes before/after tuning.

### Running many sessions concurrently (throughput)

To process a batch of sessions, prefer **one process per session, each pinned to
a small thread budget** over a single session on all cores. Independent sessions
are embarrassingly parallel: no within-session thread-coordination overhead, and
the serial-ish stages (greedy init is sequential; OASIS is inherently
per-component sequential) of one session overlap the IO/compute of another. For
total throughput (sessions/hour) this usually beats one session with `n_jobs=-1`.

**The non-obvious gotcha — `n_jobs=1` is NOT a single-threaded process.** The
serial code path does **not** cap BLAS (every `threadpool_limits(limits=1)` in
the package wraps a `n_jobs!=1` parallel branch only — see `spatial.py`,
`background.py`, `temporal.py`). So numpy/scipy (ring `BTB` solves, `project_onto`
matmuls, LASSO) still spawn **one thread per core** even at `n_jobs=1`. Launch N
such processes without capping BLAS and you get `N×cores` threads on `cores`
CPUs → oversubscription thrash, *slower* than a single session. **Fix:** export
the BLAS thread caps in each process's environment *before* numpy is imported:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
```

**Second gotcha — give each session a unique `yflat_dir`.** The auto-derive path
always names the store `Y_flat_pixel.zarr` under `yflat_dir` (or `output_dir`),
so concurrent sessions sharing one `--yflat-dir` would **clobber each other's
store** (a correctness bug). Each session needs its own scratch subdir (and its
own results dir).

Minimal launcher (one process per session, BLAS pinned, unique `yflat_dir`,
at most `$JOBS` in flight via bash ≥ 4.3 `wait -n`; reads `mc.zarr` paths from
stdin, forwards trailing args to `run_extract.py`):

```bash
#!/usr/bin/env bash
set -uo pipefail   # not -e: one failing session must not abort the batch
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
JOBS="${JOBS:-$(nproc)}"; YFLAT_ROOT="${YFLAT_ROOT:-/tmp/cnmfe_yflat}"
running=0
while IFS= read -r mc; do
  [[ -z "$mc" ]] && continue
  key="$(readlink -f "$mc" | md5sum | cut -c1-8)"          # unique per session
  ydir="$YFLAT_ROOT/$(basename "$(dirname "$mc")")_$key"
  ( python run_extract.py "$mc" --n-jobs 1 --yflat-dir "$ydir" "$@" \
      > "$ydir.log" 2>&1 && echo "OK   $mc" || echo "FAIL $mc ($ydir.log)" ) &
  if (( ++running >= JOBS )); then wait -n; ((running--)); fi
done
wait
# usage:  printf '%s\n' /net/data/*/mc.zarr | JOBS=8 YFLAT_ROOT=/local/ssd ./run_sessions.sh --params cfg.json
```

Notes:
- `--yflat-dir` on a **local SSD** keeps the ~5–6 BCD passes local even when
  `mc.zarr` lives on the network; only the one-time transpose hits the network.
- Pick **N ≈ min(physical cores, RAM_budget / per-session peak, IO headroom)**.
  Per-session peak RAM is bounded by streaming, but greedy init still
  materialises a strided sample (the spike — see `todo/greedy_init_streaming.md`),
  so budget for that × N. If you are **IO-bound on a network mount**, *fewer*
  concurrent sessions can be faster — N readers contend for the same bandwidth.
- A middle ground on many-core boxes: a few sessions each with a small `n_jobs`
  (e.g. 4 × `n_jobs=4` on 16 cores) — still needs the BLAS env caps so each
  session's BLAS doesn't fight its own joblib worker threads.

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
- `spatial.py` has no GPU path (elastic-net CD via `enet_coordinate_descent_gram`); `temporal.py` GPU only covers the projection, not OASIS

### Module-level worker functions
All functions dispatched by `joblib.Parallel` are defined at module top level (not as
lambdas or nested functions). Required for `spawn`-based pickling on Windows.
Workers: `_filter_estimate_apply` (motion correction, per batch via `_process_batch`),
`_ring_pixel_batch`, `_spatial_pixel_batch`, `_deconvolve_with` (in `temporal.py` —
replaced the older `_deconvolve_one`; takes pre-computed `g`/`sn` so it doesn't
re-estimate per call), `_project_onto_batch` (`background.py`) / `_yflat_proj_batch`
(`pipeline.py`) for threaded streaming `Y.T @ A`, and `_greedy_patch_worker`
(`initialization.py`) for patch-parallel init.

**Threads vs processes.** Every parallel step uses `prefer="threads"` (the inner
kernels — `cv2`/`ndi` convolve, BLAS matmul, sklearn CD — release the GIL) under
`threadpool_limits(limits=1, user_api="blas")`. The **one exception** is
patch-parallel init (`greedy_corr_pnr_patched`), which uses the default loky
**process** backend: the greedy seed loop is pure-Python and GIL-bound, so threads
wouldn't help. Process workers must set their own `threadpool_limits` inside the
worker body (the parent context does not cross the process boundary).

### Patch-based parallel initialization (default ON — `init_patches`)
The greedy seed loop (`greedy_corr_pnr`) is the dominant **serial** bottleneck in
extraction and can't be threaded (each extraction subtracts its component from the
residual the next seed reads). `CNMFeParams.init_patches=True` (**default on**)
runs `greedy_corr_pnr_patched`: tile the in-RAM `(T_init, H, W)` init sample into
**overlapping** patches, run the unchanged `greedy_corr_pnr` on each in parallel
processes (inner `n_jobs=1`, `border_px=0`, `max_neurons=None`), remap each patch's
footprints/centres to global coords, concatenate, and **dedup** border duplicates
with `merge_components` (the centre-distance fallback — duplicate copies of one
neuron have near-identical traces + close centres). `border_px` and `max_neurons`
are applied **globally** after dedup. Defaults derive from `sigma`:
`patch_size = max(int(12·sigma), 48)`, `patch_overlap = int(4·sigma)` (overlap >
`patch_radius ≈ 3·sigma` so a border neuron is fully captured in — and merged
across — both patches).

**Auto-skips back to the serial `greedy_corr_pnr`** when patching can't help or
would break (so the bit-for-bit serial path still runs there — that's why the
small-FOV `test_stage_split.py` regressions hold): `min(H,W) < init_patch_min_fov`
(default 128), `device != "cpu"`, or a streaming/zarr `init_movie` (the patched
driver does `np.asarray(movie)` → **in-RAM only**, no OOM). **The tuning sweep also
forces `init_patches=False` for its candidates** (`tuning/tuner.py`): they already
run in parallel loky processes, and nesting loky would serialise the inner patches.
So in practice patched init runs for the single full-FOV final extraction
(`MiniCnmfeExtraction`, wired to `init_patches=use_ram`) and standalone fits, **not**
inside the sweep. Validated equivalent to serial — seeds land on the same cells
(footprint corr ≈ 0.96, trace ≈ 0.99), ≈3× faster at `n_jobs=8` (more with more
cores). Set `init_patches=False` to force the bit-for-bit serial path. Regression
test: `test_pipeline.py::test_patched_init_recovers_same_neurons_as_global`.

**Worker count is capped — `init_patch_max_workers` (default 32).** Each patch
runs in a loky **process**, so on a many-core box the binding cost is *process
count*, not per-worker size: spawning one Python interpreter per requested job
(e.g. `n_jobs=128`) for a large movie multiplies per-process RSS far past the
useful number of patches, and most of that is pure interpreter overhead. `pipeline.fit`
therefore resolves the requested workers (`init_patch_n_jobs or n_jobs`, with
`-1`→all CPUs **before** clamping so `min(-1, cap)` can't bypass the cap) and
clamps to `init_patch_max_workers` via `initialization._resolve_patch_workers`;
`greedy_corr_pnr_patched` additionally clamps to `len(tiles)` (never more workers
than patches). At the default 32, init stays well-parallelized while bounding the
process count. **Results are unchanged** (patches are independent — worker count
is a pure scheduling choice). Raise the field to use more cores, or set it huge
to disable. The print line shows the effective (and capped) count. Regression:
`test_pipeline.py::test_resolve_patch_workers_caps_and_resolves` +
`test_patched_init_max_workers_does_not_change_results`.

### `update_spatial` per-pixel CD: ridge for convergence (`spatial_ridge`)
`update_spatial` solves a per-pixel non-negative LASSO via
`enet_coordinate_descent_gram`. On real data, active components often have
correlated/near-duplicate traces, so the per-pixel Gram `C_active @ C_active.T` is
near-singular and the **pure** LASSO CD crawls to 1000–10000 iters / fails to
converge. `CNMFeParams.spatial_ridge` (default `1e-2`) sets the solver's L2 `beta`
to `spatial_ridge · max(diag(Gram))`, bounding the condition number to ≈
`1/spatial_ridge` so the CD converges in tens of iters with ~1% coefficient
shrinkage (set `0.0` for pure LASSO). `spatial_max_iter` default lowered 10000→1000
(now just a backstop). `update_spatial` prints a one-line `update_spatial stats:`
diagnostic (mean/max CD iters, % hitting the cap, mean active-set size).

### Footprint-size knobs for dense FOVs (`spatial_lambda_scale`, `spatial_max_radius_factor`)
On dense / long FOVs footprints sprawl into their neighbours (see the dense-FOV
caveats in *Two trace flavours* + *Real-recording tuning*). Footprint size is
set in two places in `update_spatial`: the per-pixel LASSO threshold `λ_p` (which
pixels turn on), then the post-hoc `threshold_footprint` cleanup
(`spatial_max_thr`, `spatial_circular_max_dist_factor`). Two **opt-in** knobs were
added (June 2026) to tighten footprints *before* relying only on post-hoc
thresholding:

- **`spatial_lambda_scale`** (default `1.0` = unchanged): multiplies the per-pixel
  penalty, `λ_p = spatial_lambda_scale · 0.5 · sn_p · √(max_energy) / T`. `>1`
  raises the bar a pixel's `C_k·y` must clear to be nonzero → tighter footprints
  **at the regression source**, not just chopped after the fact. Threaded through
  all three CD paths (serial, numba kernel, threaded). ~1.5 is a good start.
- **`spatial_max_radius_factor`** (default `0.0` = off): caps the circular
  constraint's clip radius at `spatial_max_radius_factor · sigma` px. The default
  `circular_constraint` (`initialization.py`) derives its radius from the
  footprint's **own area**, so once a footprint sprawls its radius grows too and
  the constraint stops biting — self-defeating exactly when needed. This adds an
  absolute physical-radius cap (`circular_constraint(..., max_radius=…)`) that
  still clips bloated footprints. ~2.0 recommended for dense/long recordings.

Both default to no-op, so the bit-for-bit `n_jobs=1` path (`test_stage_split.py`)
is unchanged. `spatial_ridge` is **not** a size knob (it's L2 conditioning, does
not promote sparsity). Regression:
`tests/test_spatial_size.py::test_knobs_shrink_footprints_without_losing_cells`
(dense simulator: turning both on drops median footprint npix without losing
ground-truth cells).

**Empirical real-data result (June 2026, PICAST 5k-frame snippet, full 492×566
FOV, `sigma=3`, `global_bg_rank=1`, `n_iter_main=2`).** Three settings vs the
tuned recommended params:
- defaults: K=160, npix median **280**, `corr(C,C+YrA)` 0.862
- `+lambda_scale=1.5`: K=160, npix median **277**, corr 0.863 — **λ is nearly
  inert.** On real data the post-hoc `threshold_footprint` already bounds size
  *below* the LASSO support, so tightening λ only trims pixels that were going to
  be discarded anyway. λ would need to be pushed much higher (3–5×) to bite, at
  which point dim-cell-death risk climbs. (Mirrors CaImAn: the **size prior**
  does the work, the sparsity penalty is secondary.) **Prefer the radius cap over
  λ for footprint size.**
- `+lambda_scale=1.5, max_radius_factor=2.0`: npix median **280→109** (−61%),
  corr **0.862→0.875** (tighter footprints → cleaner demixing → higher purity),
  K even rose 160→173 (less overlap → fewer merges). The npix IQR collapsed to
  ≈π·(2·sigma)² — i.e. nearly *every* footprint exceeded `2·sigma` and was
  clamped (the whole field was sprawled).

**Energy (nrg) thresholding — `spatial_thr_method` / `spatial_nrg_thr`.** A third,
shape-aware size mechanism (CaImAn's `thr_method='nrg'`). `threshold_footprint`
step 2 normally zeroes pixels below `spatial_max_thr · peak` (**peak-relative** —
blind to concentration, so it keeps a sprawled footprint's broad low-contrast
skirt). `spatial_thr_method="nrg"` instead keeps the brightest pixels whose summed
`a²` reaches `spatial_nrg_thr` of the total; squaring discounts dim skirts, so they
drop cleanly. Complementary to the other levers: **λ** gates LASSO pixels
(sparsity), **nrg** trims dim skirts (intensity), the **radius cap** trims far
pixels (geometry). nrg *adapts* pixel count per footprint instead of pinning a
fixed circle, and (because footprints shrink) needs `min_pixel` to track the
smaller scale.

**nrg is now the package DEFAULT** (June 2026): `CNMFeParams.spatial_thr_method="nrg"`,
`spatial_nrg_thr=0.95` (the validated sweet spot). The `threshold_footprint()`
**function** default stays `thr_method="max"`, so direct callers / the
`tests/test_spatial.py` unit tests are unaffected — only the `CNMFeParams` field
default moved. Set `spatial_thr_method="max"` to restore legacy peak-relative
thresholding. Because the default flip shrinks footprints, the **tuner now derives
`min_pixel` from the winning sweep candidate's realized footprint 25th-pct
(`tuning/metrics.py:npix_p25` → `tuning/tuner.py`)** — measured on the actual nrg
footprints, not greedy-init footprints (which don't see the thresholding method
and would over-estimate `min_pixel`, causing the auto-eval to reject the smaller
footprints). No extra tuner runtime (reuses the sweep fits; `spatial_nrg_thr` is
fixed at 0.95, not swept). Synthetic recovery tests that need the old behaviour
pin `spatial_thr_method="max"` explicitly. Regression:
`tests/test_spatial_size.py::test_nrg_thresholding_tightens_vs_max_and_is_optin`.

**Empirical sweep (June 2026, PICAST 5k-frame snippet, full FOV, `sigma=3`,
`min_pixel=70` so the acceptance floor tracks the smaller footprints; K≈177
throughout). Decision metric = `corr(C, C+YrA)` mean/top30 (purity proxy) at
matched K:** `max@0.1` 0.850/0.890 (npix 288) · `nrg@0.99` 0.860/0.894 (243) ·
**`nrg@0.95` 0.871/0.899 (159) ← top30-purity peak** · `nrg@0.90` 0.875/0.897
(112) · `nrg@0.8` 0.873/0.875 (68) · `nrg@0.7` 0.868/0.885 (43) · `nrg@0.6`
0.862/0.882 (28). Conclusions: **nrg beats max** (smaller footprints AND higher
purity, no cell loss); the purity curve **peaks at nrg≈0.95** then declines
(below ~0.90 you over-tighten — footprints clip real signal, fall below
`min_pixel`, get rejected — accepted count collapsed 163→85→15→0 over
0.95→0.8→0.7→0.6); 0.95–0.90 brackets the physical soma area (≈π·(2σ)²≈113 px at
σ=3); and **nrg > the radius cap** (adaptive per-footprint IQR spread vs the
cap's pinned ≈[102,112], and higher purity). Recommended dense-FOV config:
`spatial_thr_method="nrg"`, `spatial_nrg_thr=0.95`, `min_pixel` retuned down.
Reproduce: `live_runs/nrg_compare.py [levels...]` (incremental panels +
side-by-side).

⚠️ **GOTCHA — `spatial_max_radius_factor` collides with `min_pixel`.** In that
run `K_accepted` went **116 → 0**: shrinking footprints to a 109-px median put
every one below `min_pixel=211` (which the tuner had calibrated for the *sprawled*
~280-px footprints), so the auto-eval rejected all of them. **If you enable the
radius cap, retune `min_pixel` down accordingly** (here ~60–90). This is an
expected interaction, not a bug — the acceptance floor must track the new
footprint scale. `max_radius_factor=2.0` is aggressive (binds on essentially all
footprints at `sigma=3`); try 2.5–3.0 to clip only the sprawled ones.
Reproduce: `live_runs/spatial_size_snippet_check.py`.

### Streaming `Y.T @ A` projections are threaded (`n_jobs`)
The two zarr/streaming projection loops — `BackgroundSubtractor.project_onto`
(final `YrA`) and the strided-init full-T trace recovery in `pipeline.fit_extract`
— are independent per-pixel-batch sums, now parallelized with
`Parallel(prefer="threads")` + `np.add.reduce`. The `n_jobs==1` path keeps the
exact serial accumulation order (so bit-for-bit tests hold); `n_jobs>1` reorders
the reduction (float32 drift ~1e-6). `project_onto` gained an `n_jobs` kwarg
(threaded from `pipeline.fit_extract` and `temporal.update_temporal`).

### Flat pixel representation
After initialization the movie is stored as `(H·W, T)` — pixels as rows, time as columns.
`make_2d` / `make_3d` in `_utils.py` handle reshaping.

### AR-coefficient `g` is estimated ONCE per pipeline run, not per BCD iteration
- `pipeline.fit()` estimates `g` from `C_raw.ravel()` (pooled across components for robustness on short traces) right after init, plus per-component `sn` from each `C_raw[k]`. Stored on `self.g` and `self.sn_per_k`.
- The cache is threaded into every `update_temporal(..., g_cached=..., sn_cached=...)` call.
- After `merge_components` reorders K, `_cache_after_merge(members_per_group)` updates the cache by inheriting from `members[0]` — no re-estimation, no drift.
- Re-estimating from a deconvolved trace re-applies `fudge_factor=0.96` and drifts `g` toward 0 across iterations. **Do not re-introduce per-iteration estimation.**

### Bayesian prior on `g` via `decay_time_ms` + `frame_rate_hz`
When **both** `CNMFeParams.decay_time_ms` and `CNMFeParams.frame_rate_hz` are
set, the pipeline derives `g_target = exp(-1 / (fps · τ_ms / 1000))` and
shrinks the Yule-Walker estimate toward it:

    g = (1 - g_prior_weight) · g_yw + g_prior_weight · g_target

`g_prior_weight` defaults to 0.5; bump toward 1 on drift-heavy recordings
(Yule-Walker is upward-biased there). `fudge_factor` is **bypassed** on the
prior path — the prior already encodes the physical bound. If either field
is `None`, the legacy `fudge_factor` shrinkage applies.

Suggested `decay_time_ms` values (single-AP τ, somatic, approximate; vary
1.5–2× with cell type / AP count / expression):
- GCaMP6f ~140, jGCaMP7f ~160
- jGCaMP8f ~70, jGCaMP8m ~180, jGCaMP8s ~350
- GCaMP6s/7s ~1000

The prior threads through every `estimate_ar_params` call site (pipeline
init, greedy init, `update_temporal` fallback) so `g` is consistent
end-to-end. See `todo/oasis_oversmoothing.md` for the diagnostic this fixes.

### Scale convention: CaImAn-style amplitude in the traces (unit-norm footprints)
CNMF-E factorizes `Y ≈ A·C`, invariant under `A[:,k] *= s` / `C[k] /= s`. The
extracted outputs are relabeled into **CaImAn's convention** as the final
canonicalization step of `fit_extract` (`pipeline.py:_normalize_to_trace_amplitude`,
applied unconditionally before the auto-eval gate): footprints are **unit-L2-norm**
(`‖A[:,k]‖₂ = 1`) and the per-component gain lives in the **traces**
(`C`/`S`/`YrA`/`C_raw`/`sn_per_k` all scaled by `s_k = ‖A[:,k]‖₂`). So `model.C`
peaks land in the tens–hundreds (like `caiman.estimates.C`), not ~0.1. `A·C` and
every scale-invariant quantity (correlations, SNR, spike timing) are **unchanged** —
this is a labeling choice, not a change to the extracted signal.
- The original per-component norms are kept on **`model.A_norm` (K,)** and persisted
  as `A_norm.npy`. This is **load-bearing for the auto-eval**: `evaluate.py`'s
  `snr_amp` discriminator is `∝ ‖a_k‖²`, which unit-norming would flatten to 1, so
  `auto_evaluate_components(..., a_norm=model.A_norm)` reconstructs the un-normalized
  SNR (`‖a_k‖² = A_norm[k]²`). `a_norm=None` keeps the historical direct-from-`A`
  path (old saved models with un-normalized `A`; the direct unit tests in
  `tests/test_evaluate.py`). **Do not** unit-norm `A` without threading `A_norm` into
  evaluate — it destroys the ghost filter and its tuned `auto_eval_snr_amp_thr=3.0`.
  (Note: `3.0` is the `CNMFeParams` package default. The tuner's long-recording
  base — `tuning/validate.py:good_defaults` — deliberately overrides it to `20.0`,
  a harder ghost cut given typical SNR spreads on long sessions. Both numbers are
  intentional; they apply to different entry points.)
- `upsample_to_native` / `place_in_full_fov` carry `A_norm` over (exact for the
  zero-padding map-back; approximate after bilinear upsampling — inspection-only views).
- Regression: `tests/test_stage_split.py` (bit-for-bit `fit()` == staged, and
  reproducible `accepted_mask` after `save`/`load`) pins both the unconditional-
  normalization ordering and the `A_norm` round-trip.

### Two trace flavours: `C` vs `C + YrA`
- `model.C` — OASIS-deconvolved (clean AR(1) shape). Use for spike-event detection.
- `model.YrA` — residual at each footprint after the final BCD pass.
- `model.C + model.YrA` — noisy *projected* trace; preserves the data's actual shape.
- Both `C` and `C + YrA` correlate ≳ 0.95 with ground truth on AR(1) synthetic data
  (on par with CaImAn). Use `C + YrA` for shape-faithful comparisons
  (cross-correlation with an external signal, regression, plotting raw fluorescence)
  and `C` for clean spike-event timing.
- **Dense-FOV caveat (empirical, June 2026 — real recording).** "`C + YrA` is
  shape-faithful" holds only at **low footprint overlap**. `YrA_k` is the data
  projected onto footprint `k` after subtracting the *other* components; when
  footprints overlap, that subtraction is imperfect and `YrA_k` soaks up
  neighbours' residual transients (cross-talk). So as the **extracted cell count
  rises** (e.g. loosening `min_corr`/`min_pnr` in a dense field), `corr(C, C+YrA)`
  **falls — for the strong cells too**, not just the weak ones. Measured on one
  180×180 cutout: K=221 → mean r 0.88 / top-30-amp r 0.77; K=722 → 0.74 / **0.45**.
  Implications: (1) in dense extractions, `C` (the demixed estimate) is the
  cleaner per-cell signal — the `C`-vs-`C+YrA` gap is `YrA` contamination, not `C`
  being wrong; (2) don't chase cell count with thresholds — there is a
  density↔purity sweet spot, and `corr(C, C+YrA)` is itself a good knob for
  picking thresholds; (3) `n_iter_main` ≥ 2 and tighter footprints
  (`spatial_max_thr` ↑, `spatial_circular_max_dist_factor` ↓) sharpen the demixing
  if you need both high K and clean traces — **verified**: at K≈600, `n_iter_main=2`
  + `spatial_max_thr=0.25` + `spatial_circular_max_dist_factor=1.2` recovered
  strong-cell `corr(C, C+YrA)` from **0.48 → 0.77** (≈ the low-K value) and shrank
  footprint npix median 79→36, at ~30% more runtime; each lever alone gives ~0.70.
- **Historical caveat (fixed May 2026):** `C` alone used to correlate only ~0.6
  when the `oasis-deconv` package was **not installed** — a bug in the
  pure-Python PAVA fallback (see *Bugs already fixed*), not an inherent OASIS
  limitation. If you see `C` ≈ 0.6 while `C + YrA` ≈ 0.96, you are on old code or
  a regressed fallback.

### Real-recording tuning: long & dense FOVs (empirical, June 2026)
Findings from a real 37398×300×300 miniscope recording (cutout extraction). All
validated by experiment; none are autotested.
- **Hazy / out-of-focus FOV inflates the neuron-radius estimate → sprawl
  (fixed June 2026).** On a hazy or out-of-focus recording, `blob_log` on the
  temporal-std projection latches onto broad background structure instead of the
  cells and the median radius blows up (measured **6.3 px** on one session vs
  **3.4 px** on an in-focus session of the same mouse). That cascades through
  `suggest_downsample` / `suggest_sigma_extraction` into a too-large `sigma`
  (15 vs 8), over-aggressive `ssub` (3 vs 2) and a huge `min_pixel` (810 vs 296)
  → giant sprawling footprints — **independent of recording length or thresholds**
  (a distinct cause from the length-driven sprawl below). **Fix:**
  `tuning/heuristics.py:suggest_mc_gsig_and_sigma` now subtracts a wide Gaussian
  blur (`highpass_sigma=8` px, default on; `0`/`None` disables) before `blob_log`,
  removing the haze so the radius reflects neurons. Verified: the hazy session
  dropped to `sigma=4`/`ssub=2`/`min_pixel≈396` with tight discrete footprints and
  *more* real neurons detected (the haze had masked them), while the in-focus
  session was unchanged. **Tell-tale:** `sigma`/`ssub`/`min_pixel` ~2× a
  comparable session — check `fig_mc_gsig.png`. Regression test:
  `tests/test_tuning.py::test_sigma_heuristic_robust_to_background_haze`.
- **Length, not thresholds, drives footprint sprawl.** On a long recording slow
  drift / photobleaching (~9% here) gives every trace a shared low-frequency
  trend → traces go collinear → `update_spatial`'s per-pixel LASSO can't separate
  neighbours and **smears each footprint over its neighbours into big merged
  blobs**. At matched thresholds, footprint area roughly *doubles* going from a
  4k-frame clip to the full 37k frames; thresholds only change the *count*, not
  the size. **Fix:** `global_bg_rank=1` (absorb the drift as a rank-1 temporal
  background `b_f·f(t)`) — or the temporal detrend (`detrend.py` / section-2b in
  the cutout notebooks) — plus cleanup `spatial_max_thr=0.25`,
  `spatial_circular_max_dist_factor=1.2`. Verified: footprint npix median
  209→113 (rank-1 bg) →71 (+cleanup).
- **`init_stride` under-detects on long movies (does NOT sprawl).** The auto value
  `max(1, T//5000)` is 7 for a 37k-frame movie, so greedy init runs on every 7th
  frame; this *subsamples calcium transients away* → lower CORR/PNR sensitivity →
  fewer seeds. Greedy-init footprint *size* is unaffected (the sprawl is a BCD
  effect, not an init one). Pin `init_stride` to 1–2 if a long movie is
  under-seeding.
- **Dense fields: don't over-merge.** `merge_thr_corr` / `merge_centre_dist_factor`
  that are fine for fusing drift-duplicates (e.g. 0.75 / 2.0) **fuse genuinely
  distinct, co-active neighbours** in a dense FOV: greedy found 214 cells, the
  0.75 merge collapsed them to 109. Once `global_bg_rank=1` handles the drift
  duplicates, use the gentler defaults (`merge_thr_corr≈0.90`,
  `merge_centre_dist_factor≈1.0`) → recovered 109→**175** cells *with tighter*
  footprints (npix median 92→62). The centre-distance fallback at 2σ merges any
  co-active pair within ~6 px — catastrophic when somata sit ~6–10 px apart.
- **Overlay footprints on a correlation image, not the mean projection.** Over
  tens of thousands of frames the mean projection is dominated by static
  background/vasculature, not the transient neurons (mean↔activity correlation
  ~0.35 long vs ~0.48 short). Footprints that sit correctly on the real cells then
  look "off the bright spots" of the mean. Judge positions on a `correlation_pnr`
  `cn` image (footprint *centres* land 2–3 px from `detect_seeds` peaks even when
  they look off the mean). `live_runs/cutout_extract.ipynb` and
  `cutout_analysis.ipynb` overlay on the correlation image for this reason.
- **Density ↔ per-trace purity is a hard tradeoff** — see the dense-FOV caveat in
  *Two trace flavours* above (cross-talk degrades `C` vs `C + YrA` as cell count
  rises).

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

### Auto-evaluation step (post-BCD quality tagging — non-destructive)
`pipeline.fit()` runs `minicnmfe.evaluate.auto_evaluate_components` between the
BCD loop and the final `update_temporal`. Two per-component checks are
recorded:
1. **Pixel-count floor:** `npix >= CNMFeParams.min_pixel`.
2. **Mean-amplitude SNR:** `(||a||² / npix) / mean(sn_pixel²) >= auto_eval_snr_amp_thr` (default `3.0`).

**Components are NEVER dropped.** The mask of components passing both checks is
exposed on `model.accepted_mask` (a `(K,)` bool array) and the full per-component
stats live on `model.eval_info` (keys: `pixel_count`, `snr_amp`, `pixel_pass`,
`snr_pass`, `min_pixel`, `snr_amp_thr`). Both are persisted by `model.save()`
(as `accepted_mask.npy` / `eval_info.npz`) and restored by `CNMFe.load()`.

To use only accepted components downstream:
```python
A_acc = model.A[:, model.accepted_mask]
C_acc = model.C[model.accepted_mask]
S_acc = model.S[model.accepted_mask]
YrA_acc = model.YrA[model.accepted_mask]
```

The SNR check is the real discriminator and is **scale-invariant** — real σ=3 Gaussian
footprints score 10–70, ghost components born from background-noise seeds (loose init
thresholds, e.g. `min_corr=0.7, min_pnr=3.0`) sit at or below 2 *even when their pixel
count is large* (ghosts can be wide, low-amplitude blobs that survive `threshold_footprint`
because they're a connected component of pixels each above 10 % of the ghost's own peak).
**Pure pixel-count filtering does not separate real from ghost components in this codebase**
— don't try to replace the SNR check with a fixed-area threshold. Set
`auto_eval_snr_amp_thr=0.0` to mark every component as accepted on the SNR check;
`min_pixel` continues to apply.

The regression test is `tests/test_pipeline.py::test_auto_evaluation_rejects_ghosts`.

### Cross-session cell registration (`minicnmfe/cellreg.py`)
Clean Python reimplementation of the Ziv lab's **CellReg** (Sheintuch et al.,
2017) — tracks the same neurons across multiple sessions. **No MATLAB / external
algorithm code**; numpy/scipy/sklearn only. Entry point
`register_sessions(sessions, ...)` (sessions = `CNMFe` models or results dirs);
returns a `CellRegResult` with a `cell_to_index_map` `(n_global, n_sessions)`
(`-1` = absent). CLI: `run_cellreg.py s1/ s2/ ... -o reg/`.

Pipeline stages (all in `cellreg.py`):
1. **Load** each session's footprints → dense `(K, H, W)` stacks. Footprints are
   cleaned with `spatial.threshold_footprint` and **unit-max normalised** (so
   correlations/projections are amplitude-invariant); centroids via
   `_utils.footprint_center`. `accepted_only=True` honours `model.accepted_mask`.
   **All sessions must share `dims`** (raises otherwise).
2. **Align** (`align="translation"` default | `"rotation"` | `"none"`) — rigid
   registration of each session's max-projection to a reference, reusing
   `motion_correction.estimate_shifts` (phase correlation). Rotation does a
   coarse angle grid search. Transforms are applied to footprint *images*
   (cv2.warpAffine) and *centroids* (analytic, same affine) consistently —
   `transforms` is `(n_sessions, 3)` of `(dy, dx, theta)`.
3. **Metrics** — KD-tree (`scipy.spatial.cKDTree`) finds candidate pairs within
   `max_distance` (µm if `microns_per_pixel` given, else px); per pair: centroid
   distance + **spatial Pearson correlation over the union of supports**. Note:
   this correlation falls off fast (~0 by a ~1.3·σ centroid offset) — that's
   expected/realistic, not a bug.
4. **Match** — *Phase 1 (default)*: threshold (`dist_thr`, `corr_thr`) + one-to-one
   Hungarian (`scipy.optimize.linear_sum_assignment`) per session pair, so two
   nearby cells can't both claim one. *Phase 2* (`probabilistic=True`): fit a
   2-component Gaussian-mixture `P_same` model (`model="centroid"|"spatial"|"joint"`)
   over the pooled metrics, match on `P_same >= p_same_thr`. **Gotcha:** the GMM
   needs *both* same-cell and different-cell candidate pairs to fit — a sparse FOV
   with a small `max_distance` yields only same-cell neighbours and the mixture
   degenerates. It falls back to threshold matching when there are <10 candidates.
5. **Cluster** — greedy-by-weight over the cross-session match graph
   (`build_cell_map`): strong matches form clusters first; a cell only joins if its
   session slot is free (≤1 cell/session per global cluster); unmatched cells
   become singleton rows.

Autotested in `tests/test_cellreg.py` (synthetic Gaussian footprints, no full
extraction needed). **Not yet validated on a real multi-session recording.**

### zarr v3 API
The project uses zarr v3 (`zarr >= 3.0`). The v3 API differs from v2 in chunk
specification and store opening. Do not regress to v2 patterns.

### Staged pipeline: `fit_extract` / `evaluate` + `fit` wrapper
`CNMFe.fit()` is now a **thin wrapper** that composes the standalone stages
`fit_mc` (optional, in-memory) → `fit_extract` → `evaluate`. Behaviour is
unchanged from the old monolith (regression: `tests/test_stage_split.py` asserts
`fit()` == the staged composition, bit-for-bit at `n_jobs=1`).

- `fit_extract(movie, *, Y_flat_zarr=None, output_dir=None, evaluate=True)` —
  everything from noise estimation through the BCD loop, final temporal pass,
  and YrA. **Resolution-agnostic**: runs on whatever movie it is handed (full
  or downsampled). Contains the streaming `Y_flat_zarr` auto-derive logic;
  motion correction is *not* part of it.
- `evaluate()` — the non-destructive auto-eval (moved out of the BCD body). Reads
  **only `self.A` + `self.sn`**, so it can be re-run on a freshly `load()`-ed
  model to retune `min_pixel` / `auto_eval_snr_amp_thr` without re-extracting.
  It is now invoked *after* the final temporal pass; order is irrelevant since
  it depends on neither `C` nor `S`. Pass `evaluate=False` to skip it.

Four root CLIs give each stage a disk handoff (mirror `full_pipeline.py`
conventions; `--params p.json` carries a `CNMFeParams` between stages):
`run_preprocess.py` → `run_mc.py` → `run_extract.py` → `run_evaluate.py`.

### Downsample-once front end
The design is **downsample once, then run MC + extraction + evaluation entirely
on the smaller movie** — outputs stay at downsampled resolution (no footprint
upsampling, no full-res finalization). There are **three ways to produce the
downsampled movie**, in increasing preference for the AVI workflow:

1. `minicnmfe/downsample.py::downsample_movie(src, dest, *, ssub, tsub, ...)` —
   streaming block-mean of an **existing `(T,H,W)` zarr** (template:
   `io.transpose_zarr_to_pixel_major`); writes a `ds_meta.json` sidecar. Use
   when you already have a raw zarr. `run_preprocess.py` wraps it.
2. `concat_avis_to_zarr(..., ssub=, tsub=)` — bins **inline while decoding the
   AVIs**, so only the downsampled zarr is written (no full-res intermediate).
3. **Fused (preferred for live sessions):** `concat_avis_to_mc_zarr` /
   `CNMFe.fit_mc_from_avis(..., ssub=, tsub=)` decode + bin + motion-correct in
   one pass, writing **only `mc.zarr`** — no `session.zarr`, no `ds.zarr`.

Binning is **per file** in the AVI paths (2, 3): a temporal group never spans
two AVIs, the trailing `< tsub` frames of each file are dropped, so the output
frame count is `sum(n_i // tsub)`. Binned frames are decoded straight to
**float32** (uint8 would round the means). For (2)/(3) pass MC params already in
downsampled units (`params.downscaled(ssub, tsub)`); `ssub`/`tsub` only describe
how the raw AVI frames are binned. Non-divisible dims are trimmed to a multiple
of the factor before binning.

`CNMFeParams.downscaled(ssub, tsub)` rescales the unit-bearing fields so params
can be expressed once in **native** units: `sigma /= ssub`, `min_pixel //= ssub²`,
`border_px //= ssub`, `max_shift //= ssub`, `mc_gSig_filt /= ssub`,
`frame_rate_hz /= tsub`. `decay_time_ms` is a physical time and is **unchanged**
(only the frame rate moves, which correctly raises the per-frame decay; the AR
`g` is then estimated on the binned traces, self-consistently). The ring radius
follows `sigma` automatically. `run_extract.py --ds-meta ds_meta.json` applies
this rescale automatically.

**Caveat (deliberate):** temporal binning happens *before* MC, so frames within
a `tsub` group are averaged prior to registration. Fine for slow drift / small
`tsub`; blurs neurons under large intra-bin motion.

**Opt-in re-upsampling to native res.** `CNMFe.upsample_to_native(*, orig_dims,
orig_T, ...)` returns a **new, non-destructive** model with `A` (bilinear) and
`C`/`YrA`/`C_raw` (linear) interpolated to the native grid/rate — for overlaying
footprints on a native reference image and plotting against native-rate signals.
Helpers `upsample_footprints` / `upsample_traces` live in `minicnmfe/downsample.py`
(per-column `cv2.resize` to the exact native `(H,W)`; per-row `np.interp` to the
exact native `T`, so trimming is handled). It is **interpolation, not recovery**
— the native movie is gone in downsample-once, so no discarded detail is
recovered. The returned model is for **inspection/overlay only**: `S` stays at
the downsampled rate, and `W`/`b0`/`b_f`/`f`/`shifts` are dropped (don't re-run
the BCD on it). Native `(H,W)`/`T` must be supplied (only `downsample_movie`
writes a `ds_meta.json`; the fused/concat paths don't), or passed via
`ds_meta=`.

### Cutout — crop the movie before extraction (`minicnmfe/cutout.py`)
Optional `CNMFeParams` fields restrict CNMFe to a sub-region/window, applied
**once at ingestion, before MC** (NATIVE coords):
- `temporal_crop=(t0,t1)`, `spatial_crop=(y0,y1,x0,x1)`, `spatial_mask_path`
  (a bool `.npy`; its bbox sets/narrows the rect, pixels outside are zeroed).
All `None` (default) = no cutout = bit-for-bit unchanged.

Because `self.dims`/`T` flow from the (cropped) movie shape, everything
downstream (ring indices, init, BCD) treats the cutout as "the movie" — no
other changes. `fit()` and `fit_mc`/`fit_mc_from_avis` apply the crop and record
`self.cutout`; **`fit_extract` does NOT crop** (its input is already the ROI),
so the staged `fit_mc_from_avis → fit_extract(mc.zarr)` flow crops exactly once.
`downscaled()` **clears** the crop fields (applied upstream of binning).

`minicnmfe/cutout.py`: `resolve_cutout` (rect ∩ mask-bbox, clamp, load mask),
`apply_cutout` (slice + zero-outside-mask; numpy or zarr), and the map-back
helpers used by **`CNMFe.place_in_full_fov(*, place_time=True)`** — a new model
(parallels `upsample_to_native`) with footprints padded to the original FOV at
the crop offset and traces embedded in the full timeline; `S`/traces zero
outside the window, background/`shifts` dropped (inspection view).

**Fused path:** `_decode_avi_worker` gained `crop_bbox`/`mask_local`/`frame_lo`/
`frame_hi`; `concat_avis_to_mc_zarr` resolves the cutout after the pre-scan,
computes per-file in-window frame ranges (temporal crop spans files via the
global offset), crops the template too, and writes a `cutout.json` sidecar that
`fit_mc_from_avis` loads onto `self.cutout`.

**Caveat (deliberate):** spatial-crop-before-MC means MC can't pull content from
outside the crop — minor edge artifacts within ~`max_shift` px of the border;
leave a small margin around the ROI.

### Fused AVI → MC (`minicnmfe/avi_mc.py`)
`concat_avis_to_mc_zarr` (and the convenience wrapper
`CNMFe.fit_mc_from_avis`) decode an AVI folder and apply rigid motion
correction in a **single pipeline**, writing only `mc.zarr` to disk. No
intermediate `session.zarr` is materialised — that saves ~5 min and ~6 GB
on a network mount for a 100k-frame session compared with running
`concat_avis_to_zarr` + `fit_mc` separately.

Pipeline shape:
1. Pre-scan every AVI for `(frame_count, H, W)` (same walk as the
   non-fused concat).
2. **Template phase**: decode a strided subset of `n_template_avis`
   (default 10) into a RAM buffer, stride-sample it down to
   `params.mc_template_max_frames`, high-pass filter the samples and
   bin-median them.
3. **Fused MC phase**: re-decode every AVI in parallel via
   `_decode_avi_worker` (reused from `concat_avis_to_zarr`). A
   queue-consuming writer pulls `(start, batch)` tuples, runs
   `_process_batch` (parallelised per-frame MC from
   `minicnmfe/motion_correction.py`), and writes the float32 corrected
   batch + shifts into `mc.zarr` / a `(T, 2)` shifts buffer.

**Inline downsampling** (`ssub` / `tsub` on `concat_avis_to_mc_zarr` /
`fit_mc_from_avis`): the decoders bin frames (block-mean) **before** MC, the
template is built from the binned frames, and offsets/output shape come from the
per-file binned counts (`sum(n_i // tsub)`, `H//ssub`, `W//ssub`). Decoders emit
float32 when binning. Pass `params.downscaled(ssub, tsub)` so `max_shift` /
`mc_gSig_filt` match the binned grid; `ssub`/`tsub` only drive the raw-frame
binning. This is the **single-write** downsampled path — only `mc.zarr` is
produced (`live_runs/run_session.ipynb` uses it).

**`mc_n_iter > 1`** is supported: the fused first pass writes a scratch
zarr at `<output_path>.parent / (".<output_path.name>.fused.zarr")`,
then `motion_correction_rigid` is called against that scratch with
`niter_rig = mc_n_iter - 1` to run the remaining iterations (its
ping-pong scratch handling kicks in for niter_rig ≥ 2). Shifts are
summed across the fused pass and the handoff (matching the existing
`shifts_total += shifts_iter` accumulation in
`_motion_correction_streaming`). The fused scratch is removed in a
`finally` block so a mid-iteration crash doesn't leak a multi-GB
orphan onto the network share. Even for `mc_n_iter > 1` the user
still skips the raw `session.zarr` intermediate.

Output zarr uses the heavyweight `clevel=5` + `bitshuffle` compression
(mc.zarr is read many times during extraction, so ratio matters).
This is the opposite of the concat output's `clevel=3` + byte-shuffle
choice, where the writer is the bottleneck.

Live-session example: `live_runs/run_session.ipynb` calls
`model.fit_mc_from_avis(folder, output_dir=...)`. The notebook keeps
the raw `before/after` diagnostic plots by pulling a strided sample
of frames directly from the AVIs (no zarr round-trip).

### `concat_avis_to_zarr` configuration
`concat_avis_to_zarr` does a per-file pre-scan via `_count_and_shape` to
get exact frame counts before allocating the output zarr — this lets the
parallel decoders write to known offsets and avoids any resize-at-end pass.
The pre-scan is reported in the timing line so users can see how long it
took on their setup (typically a few seconds per file over a network mount,
i.e. ~30–60 s for 100 AVIs).

Output zarr defaults: `clevel=3` + `shuffle="shuffle"` (byte shuffle) instead
of the project-wide `clevel=5` + `bitshuffle`. ~3× faster compress with
~10 % larger files on uint8 imaging data. The writer is single-threaded so
freeing it up is the only way to keep decoders un-stalled on
network-mounted output. `_open_array` in `minicnmfe/io.py` accepts `clevel` and
`shuffle` parameters to expose this — defaults remain the heavyweight
combination for callers (MC, temp stores) that prioritise ratio.

Expected runtimes for the typical 100k-frame miniscope session
(100 AVIs × 1000 frames × 600×600 uint8):
- Local SSD source and output: **~2–3 min**.
- Network mount (both source AVIs and output zarr): **~5–7 min**.
- If you want to skip the intermediate `session.zarr` entirely (saving ~5
  min on network mounts), use the fused AVI→MC entrypoint
  `concat_avis_to_mc_zarr` (see *Fused AVI→MC* below).

Defaults bumped:
- `chunk_t`: 100 → 500 (fewer chunk writes; halves the network round-trips
  on the writer side).
- `n_jobs` cap: removed (was `min(cpu_count, len(avis), 4)`; now
  `min(cpu_count, len(avis))`). More in-flight reads help on network mounts.

---

## Non-obvious bugs that were already fixed — do not re-introduce

### `high_pass_filter_space` float `gSig_filt` → cv2 ksize TypeError
`ksize` for `cv2.getGaussianKernel` must be an int. The historical expression
`(3*i)//2*2+1` stays a float when `gSig_filt` is a float — which happens as soon
as `CNMFeParams.downscaled(ssub, tsub)` scales `mc_gSig_filt` by `ssub`
(e.g. `7/2 = 3.5`). cv2 then raises `TypeError: Argument 'ksize' is required to
be an integer`. **Fix** (`motion_correction.py:high_pass_filter_space`): wrap the
ksize expression in `int(...)`. Identical for integer `gSig_filt`; only enables
the float (downsampled) case. Affects every MC path (`fit_mc`,
`motion_correction_rigid`, fused `avi_mc`).

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

### Pure-Python OASIS (PAVA) fallback merge condition
`temporal.py:_oasis_pava_run` is the deconvolution used when the
`oasis-deconv` package is **not** installed. Its pool-merge test was
`pool_val[i] * g > pool_val[i+1]`, but the OASIS boundary constraint
`c[t] >= g·c[t-1]` requires `pool_val[i] * g**pool_length[i] > pool_val[i+1]`.
Using bare `g` (correct only for length-1 pools) over-merged smooth exact-`g`
decays and collapsed the trace: deconvolving a **clean AR(1)** trace reconstructed
it at only r ≈ 0.4, and `model.C` vs ground truth sat at ~0.58 instead of ~0.96
(while `C + YrA` stayed ~0.96, masking it). Only bites without the compiled
package — which is why it slipped CI (the package path was exercised there) and
why it surfaced in the CaImAn comparison (run in an env without it).
**Fix:** use `g ** pool_length[i]` (plus a tiny tolerance for float-noise on
exact-`g` decays). Regression guard:
`tests/test_temporal.py::test_pava_fallback_reconstructs_clean_ar1` asserts the
fallback reconstructs a clean AR(1) at r > 0.95 (calls `_oasis_ar1_pava`
directly, so it tests the fallback regardless of whether the package is present).

### `avi_to_zarr` imageio v3 shape extraction
`iio.improps(src).shape` in imageio v3 returns `(T, H, W)` (full video shape), not `(H, W)`.
Using `props.shape[:2]` silently extracted `(T, H)` as the spatial dimensions, creating a zarr
with shape `(T, T, H)` and failing with a broadcast error when the first chunk was written.
**Fix** (in `minicnmfe/io.py`): check whether `_s[0] == T` and extract H, W from `_s[1:]` if so,
otherwise fall back to `_s[0:2]`. The same fix applies to `concat_avis_to_zarr.py`.

### Strided-init (`init_stride > 1`) leaked the 1p background into every trace
With `init_stride > 1`, `fit_extract` re-projected the full-T initial traces as a
**raw** footprint projection `C = (Y.T @ A) / ‖A‖²` (no background removal). That
injected the broad 1p background into `A·C` from frame 0, which then **blinded the
first `compute_W`** (ring is fit on the residual `X = Y − A·C − b0` — anything
already in `A·C` is invisible to it). The shared background then leaked into every
neuron trace for the whole BCD, a stable bad fixed point: the OASIS-deconvolved
traces shared an ~81 %-variance PC1 that was ≈ the global background (PICAST
cutout: median pairwise |r| ≈ 0.45, `corr(PC1, model.f)` ≈ 0.96), and **more BCD
iterations did NOT escape it** — only a clean init does. The `init_stride == 1`
path was fine because its first traces come from the center-surround-**filtered**
movie (`extract_spatial_temporal` uses `data_filtered`), so its first `compute_W`
saw clean traces. Since the tuner sets `init_stride = 2` by default (`good_defaults`),
this hit every long-recording run. **Fix** (`pipeline.py`, init re-projection):
for `init_stride > 1`, bootstrap a ring background (`compute_W`) from the clean
strided greedy traces (`C_init`) on the strided sample, then project the full-T
init traces through a `BackgroundSubtractor` so the first full `compute_W` is no
longer blinded. Verified on the PICAST cutout: median pairwise |r| 0.45 → 0.12,
PC1 var 81 % → 21 % (and *more* real cells, since cleaner traces clear the SNR
gate). Diagnostic: `live_runs/bg_leak_diag.py`. Regression:
`tests/test_pipeline.py::test_strided_init_does_not_leak_shared_background`
(event-like shared bg; pre-fix stride=3 median |r| ≈ 0.9, post-fix ≈ 0.06; the
bit-for-bit `init_stride == 1` path is untouched).

---

## File structure

```
minicnmfe/                         Main package
  _utils.py                    make_2d, make_3d, get_xp, to_numpy, iter_frames, ensure_float32
  io.py                        avi_to_zarr, open_zarr, save_zarr
  motion_correction.py         motion_correction_rigid, apply_shift, estimate_shifts
  preprocess.py                make_center_surround_psf, estimate_noise, correlation_pnr
  background.py                build_ring_indices, compute_W, subtract_background
  initialization.py            detect_seeds, extract_spatial_temporal, greedy_corr_pnr
  spatial.py                   compute_support, threshold_footprint, update_spatial
  temporal.py                  estimate_ar_params, deconvolve, update_temporal, _deconvolve_with
  merging.py                   merge_components  (4-tuple return)
  avi_mc.py                    Fused AVI -> mc.zarr in one pass (skips session.zarr)
  concat_avis_to_zarr.py       Concatenate a folder of 0.avi ... N.avi into one zarr store (importable + CLI: python -m minicnmfe.concat_avis_to_zarr)
  downsample.py                downsample_movie() — streaming spatial+temporal block-mean (+ ds_meta.json)
  cellreg.py                   register_sessions() / CellRegResult — cross-session cell registration (CellReg reimplementation)
  pipeline.py                  CNMFeParams (+ .downscaled()), CNMFe.fit()/fit_mc/fit_extract/evaluate
tests/
  conftest.py                  make_synthetic_movie() — clean fixture; supports motion_max_shift
  miniscope_simulator.py       make_miniscope_movie() — realistic 1p movie with bg/ghosts/vasc/bleach/shot noise/8-bit/motion
  test_multiprocessing.py      n_jobs correctness tests
  test_pipeline.py             includes test_temporal_correlation_against_truth (regression for the AR drift fix)
  test_stage_split.py          fit() == fit_mc -> fit_extract -> evaluate (staged decomposition)
  test_downsample.py           downsample_movie / downscaled / end-to-end downsampled recovery
  test_cellreg.py              cross-session registration: alignment / matching / clustering / P_same / save-load
docs/                          GitHub-renderable docs: getting-started/, concepts/ (algorithm-math, algorithm-eli5, architecture, ring-background, caiman-comparison), api/, guides/ (per-stage), tuning/ (index + guide)
demo_movies/                   Generated AVI files + _meta.npz sidecars + .zarr stores (created by scripts below)
generate_demo_movies.py        Generate demo_movies/*.avi with ground-truth NPZ sidecars (idempotent)
convert_to_zarr.py             Batch-convert demo_movies/*.avi -> *.zarr (idempotent)
full_pipeline.py               CLI: load any zarr lazily, run full CNMFe pipeline, save A/C/S/YrA/shifts/sn/params to disk
run_preprocess.py              Staged CLI 1/4: downsample a zarr (ssub/tsub) -> ds.zarr + ds_meta.json
run_mc.py                      Staged CLI 2/4: motion-correct a zarr -> mc.zarr + shifts.npy + params.json
run_extract.py                 Staged CLI 3/4: extract on mc.zarr -> results/ (--ds-meta auto-rescales params)
run_evaluate.py                Staged CLI 4/4: re-run auto-eval on a results dir (retune thresholds, no re-extract)
run_cellreg.py                 CLI: register cells across >=2 results dirs -> cell_to_index_map.npy + transforms.npy + cellreg_info.json
tune.py                        SINGLE front door: `tune.py <path>` = heuristics + sweep + full-recording validation + report.html (default output ./runs/, gitignored); `--sessions <list>` = batch; `--no-validate`/`--no-html`/`--no-lowthr`/`--dry-run` flags. Composes the stages below (calls tuning.validate.tune_then_validate).
validate_session.py            Internal stage / standalone CLI: fused MC once -> Y_flat once -> fit_extract per threshold set (reuses Y_flat) -> diagnostics + comparison.md. Use directly only to re-validate or add threshold sets.
batch_tune.py                  Batch stage: run_batch() runs one `tune.py --validate` subprocess per session in ONE background process (bounded concurrency, BLAS-capped) -> batch_summary.md. No sub-agents. `tune.py --sessions` delegates here.
tuning/                        Tuning package: io_sample, heuristics (per-knob suggest_*; neuron-radius estimate spatial-high-pass-filtered, highpass_sigma=8, for hazy/out-of-focus FOV robustness), metrics (GT-free quality proxies), sweep (graded fit_extract grid), report (figures + report.md + packaged diagnostics: fig_footprint_grid/eccentricity/jaccard_merge/centroid_drift/mean_proj_and_activity + DIAGNOSTIC_FIGS + METRICS_BLURB/SYMPTOM_CAUSE_KNOB), report_html (self-contained report.html: base64 figs + sortable candidate table), tuner, validate (read_session_meta + validate_session + tune_then_validate + good_defaults)
.claude/skills/tune-session/   User-invoked skill (/tune-session <path...>|<list.txt>): metadata -> tune.py (tune+validate+html) -> verdict + per-session LEARNINGS.md. Gotcha checklist lives in docs/tuning/guide.md (single source). Multiple paths / a .txt list run via batch_tune (one background process, NOT sub-agents). --figs/--no-figs gates end-of-run PNG viewing (the dominant token cost)
tutorial.ipynb                 Original walkthrough (preserved)
tutorial2.ipynb                Clean rewrite of the original tutorial
tutorial_realistic.ipynb       Tutorial on the realistic simulator + mp4 export of the simulated movie
tutorial_caiman_compare.ipynb  Side-by-side CaImAn vs our CNMFe (requires CaImAn installed separately)
tutorial_demo.ipynb            Realistic-use demo: AVI -> zarr -> lazy load -> full pipeline -> visualise
CaImAn-main/                   Reference source only — never import from here for production
```

(The former `todo/speedup.md` is gone — its two ideas are now implemented:
"skip OASIS on first pass" = `CNMFeParams.skip_first_deconv`, "cache W" =
`compute_W(..., W_cached=...)`.)

`CNMFeParams` fields (excerpt of the params added or made adjustable in this round):
- `init_min_corr_neuron: float = 0.8` (was hardcoded 0.9)
- `init_max_corr_bg: float = 0.4` (was hardcoded 0.3)
- `init_patch_max_workers: int = 32` — upper bound on patch-init loky processes (caps process-count RAM on many-core boxes; results unchanged). See *Patch-based parallel initialization*.
- `seed_suppress_factor: float = 2.0` — controls greedy-init suppression disk size
- `circular_max_dist_factor: float = 2.5` — `circular_constraint` cutoff
- `merge_centre_dist_factor: float = 2.0` — centre-distance fallback for `merge_components`
- `global_ar: bool = True` — `True` (default) = one `g` estimated from pooled `C_raw`; `False` = per-neuron `g` from each `C_raw[k]`. Both modes estimate once from raw traces and cache; neither re-estimates from deconvolved traces. With the prior path enabled (see below), `False` is the more defensible choice — each neuron's Yule-Walker estimate gets shrunk toward the same physical-units target independently, preserving real per-neuron variability.
- `fudge_factor: float = 0.96` — legacy Yule-Walker shrinkage. **Bypassed when the prior path is enabled.**
- `decay_time_ms: float | None = None`, `frame_rate_hz: float | None = None` — when both are set, enable the Bayesian prior on `g` (see *Bayesian prior on `g`* section above). Indicator τ table:
  - GCaMP6f ~140, jGCaMP7f ~160
  - jGCaMP8f ~70, jGCaMP8m ~180, jGCaMP8s ~350
  - GCaMP6s / 7s ~1000
- `g_prior_weight: float = 0.5` — shrinkage weight for the prior path. 0 = pure Yule-Walker, 1 = pin at target. Bump toward 1 on drift-heavy recordings.
- `ar_detrend_order: int = 0`, `temporal_detrend_order: int = 0` — NON-STANDARD polynomial detrend orders. Set ≥1 to strip slow drift before Yule-Walker (`ar_detrend_order`) and/or before OASIS (`temporal_detrend_order`). Defaults preserve standard CNMF-E behaviour.

`make_miniscope_movie` / `make_synthetic_movie` parameters added (in `tests/`):
- `motion_max_shift: float = 0.0` — peak drift amplitude in pixels; 0 = no motion (backward-compatible)
- `motion_seed: int | None = None` — RNG seed for drift (defaults to `seed + 1`); `make_synthetic_movie` always uses `seed + 1`
- Drift is a smoothed correlated random walk (cumsum of small Gaussian steps, uniform_filter1d, rescaled to peak = `motion_max_shift`)
- Applied frame-by-frame via `minicnmfe.motion_correction.apply_shift`; stored as `result["motion_shifts"]` (T, 2) float32
- Sign convention: `motion_shifts[t]` is the `(dy, dx)` shift applied to generate frame t; motion correction's `model.shifts` is approximately the negative (the correction that undoes the drift)

`CNMFe` result attributes added:
- `model.YrA: (K, T)` — residual at each footprint; `C + YrA` = noisy projected trace
- `model.g: list[np.ndarray]` — per-component AR coefficients used for OASIS
- `model.sn_per_k: (K,)` — per-component noise std
- `model.A_norm: (K,)` — original `‖a_k‖₂` before the CaImAn-scale unit-norm relabeling (see *Scale convention* above)

---

## Running things

```bash
# Install (includes matplotlib in core deps)
pip install -e .

# Optional extras
pip install oasis-deconv          # faster deconvolution
pip install cupy-cuda12x                 # GPU support (match your CUDA version)

# Tests
pytest tests/ -v                         # all 282 tests
pytest tests/test_pipeline.py -v         # pipeline + temporal-correlation regression
pytest tests/test_multiprocessing.py -v  # parallelism only

# Demo movies (one-time setup)
python generate_demo_movies.py           # creates demo_movies/*.avi + *_meta.npz
python convert_to_zarr.py                # creates demo_movies/*.zarr

# CLI pipeline
python -m minicnmfe.concat_avis_to_zarr /path/to/folder/   # concatenate 0.avi...N.avi -> movie.zarr
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
  `python setup.py build_ext --inplace`. CaImAn is not actually imported by `minicnmfe/`.
- When using `n_jobs=-1` followed by CaImAn's `cm.cluster.setup_cluster()` (e.g. in
  `tutorial_caiman_compare.ipynb`), the loky worker pool from joblib persists after `CNMFe.fit()`.
  CaImAn's `setup_cluster` raises "A cluster is already running" when it detects live processes.
  **Fix:** call `from joblib.externals.loky import get_reusable_executor; get_reusable_executor().shutdown(wait=True)`
  before each `setup_cluster` call. This drains loky workers without affecting subsequent `CNMFe` calls.
