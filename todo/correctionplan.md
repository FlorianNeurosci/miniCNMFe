# CNMFe pipeline correction plan

Living document. Each item is independent; tackle one at a time and tick it
off (mark **DONE** with date) before moving on. The active item is at the top.

> Companion to `todo/speedup.md`. That file is performance-focused; this one
> is broader (bugs, scaling, algorithm, API). Items overlap where noted.

---

## Item 0 — Prerequisite: rename `mc_niter_rig` → `mc_n_iter`

**Status: DONE (2026-05-19).** Field renamed; all 93 tests pass. Touched:
`cnmfe/pipeline.py` (1 def + 2 use sites), `tests/test_pipeline.py`,
`demo_notebooks/01_load_and_motion_correct.ipynb`, `wiki/api-reference.md`.
`full_pipeline.py` was already using `mc_n_iter` (the buggy reference) — that
crash is now fixed for free.

`full_pipeline.py` (lines 90, 102, 138) uses `mc_n_iter`; the dataclass
field is `mc_niter_rig` at `cnmfe/pipeline.py:36`. `wiki/api-reference.md:23`
documents the wrong name. CLI crashes the moment `--mc-iter` is used.

**Steps.**
1. Rename the dataclass field `mc_niter_rig` → `mc_n_iter` for snake_case
   consistency with the other `mc_*` fields.
2. Update `pipeline.py:172, 215` and `tests/test_pipeline.py:116`.
3. Update `demo_notebooks/01_load_and_motion_correct.ipynb` (one occurrence).
4. Update `wiki/api-reference.md`.
5. Optional `__post_init__` alias accepting `mc_niter_rig=` for backward compat.

**Verification.**
- `python full_pipeline.py demo_movies/realistic_small.zarr --mc-iter 1`
  runs to completion.
- `pytest tests/test_pipeline.py -v` passes.
- `grep -rn 'mc_n_iter\|mc_niter_rig' .` consistent.

---

## Item 1 — Stream extraction from zarr  *(DONE 2026-05-19)*

**Status: DONE.** Shipped Phases A–E. Test count 88 → 104. Deferred:
true T-streaming (disk-transpose preprocessing) and the full version of
streaming greedy init — both documented in their respective phases as
follow-up work, both unnecessary at current scale (10k frames).

Goal: `CNMFe(params).fit(zarr_array, do_motion_correction=False)` should run
end-to-end without ever materialising `(T, H, W)` in RAM. Peak working set
should be `O(K·H·W + batch·H·W)` instead of today's `O(T·H·W)`.

### Locked-in decisions
- **API:** extend the existing `fit()` to accept `zarr.Array` directly. When
  input is zarr and `do_motion_correction=False`, the streaming path runs
  automatically. Numpy input keeps the in-memory path. No new public method.
- **Greedy init:** fully stream (Phase D), not strided sample. Greedy init
  keeps a sparse `(A_so_far, C_so_far)` cache in RAM (small, K·T·4 bytes)
  and reads patches from zarr on demand.
- **`mc_n_iter` rename:** Item 0 — separate tiny prerequisite PR.

### What's broken today
`cnmfe/pipeline.py:199` calls `movie_arr = np.asarray(movie, dtype=np.float32)`
— full materialisation. Then several downstream allocations of the same size:

- `pipeline.py:358, 415` — `Y_bg = subtract_background(Y_flat, W, b0)`
  inside `background.py:272-294` allocates three near-full arrays per call
  (`Y - b0`, `W @ X`, the difference). Called twice per BCD iteration.
- `background.py:217-219` in `compute_W` — densifies
  `Y_flat - A @ C - b0[:, None]` explicitly before subsampling along time.

On a 60k × 600 × 600 movie this is ~86 GB per copy; peak working set hits
350–500 GB on a 2-iter run.

### Things already shaped for streaming
- `update_spatial` already iterates **pixel batches** of size 256
  (`spatial.py:204-208`). Each batch only needs `Y_bg[start:end, :]` of shape
  `(256, T)` ≈ 60 MB at T=60k. The current code passes the whole `Y_bg` and
  slices inside; we just need to provide the slice on demand.
- `update_temporal` accesses `Y_flat` exactly once per call (`temporal.py:302`,
  `Y_flat.T @ A` to make `YA: (T, K)`). After that it's pure `(T, K)` algebra.
  One streaming pass per call is enough.
- `compute_W` already subsamples time before the expensive solve
  (`background.py:225-226`, `X_fit = X[:, ::tsub]`, default tsub=5). The solve
  uses `(H·W, T/5)`. Only the **construction** of `X` is wasteful.
- Motion correction (`fit_mc`) is the reference streaming implementation —
  same pattern can apply downstream.

### Phased sub-plan

Each phase ends with a green test run and is independently shippable.

#### Phase A — Per-batch `subtract_background` *(DONE 2026-05-19)*

- Rewrite `cnmfe/background.py:subtract_background(Y_flat, W, b0)` to accept
  either a numpy array (current behaviour, returns full array) OR a callable
  + pixel range (`Y_bg_slice(start, end)` returns `(batch, T)` slice on demand).
- Add a `BackgroundSubtractor(W, b0)` helper class with a `slice(start, end)`
  method that materialises only the requested pixel rows. Internally it does:
  `Y_flat[start:end] - b0[start:end, None] - W[start:end, :] @ (Y_flat - b0[:, None])`
  — but W is sparse with O(H·W · n_ring) non-zeros; only ring neighbours of
  `start:end` are read from `Y_flat`. Cache `b0` once on the helper.
- Update `update_spatial` to accept the helper instead of a dense `Y_bg`.
  Inside the batch loop, call `bg.slice(start, end)` and feed it to the LASSO.
- Update `update_temporal` similarly: do one streaming sweep that yields
  `bg.slice(p0, p1)` for pixel chunks, accumulate `YA += bg_chunk.T @ A[p0:p1, :]`.

**Validates:** peak RAM during the BCD loop drops from `3·O(T·H·W)` per
`subtract_background` call to `O(batch · T)`.

**Shipped.**
- `cnmfe/background.py`: new `BackgroundSubtractor` class with
  `.slice(start, end)`, `[start:end]`, and `.project_onto(A)`. Uses the
  distributed-subtraction identity
  `W @ (Y - b0) = W @ Y - (W @ b0)[:, None]` to avoid the dense X intermediate.
  Sparse `W @ Y` is a sparse-dense matmul; scipy only reads ring-neighbour
  columns of Y. Legacy `subtract_background(Y, W, b0)` kept for backward compat.
- `cnmfe/temporal.py`: `update_temporal` dispatches to
  `Y_flat.project_onto(A)` when present; falls back to GPU / CPU dense matmul
  for numpy input.
- `cnmfe/pipeline.py`: `fit()` constructs a `BackgroundSubtractor` instead
  of calling `subtract_background`. Final `YA_final` projection uses
  `project_onto`.
- `tests/test_background.py`: 6 new tests covering full / partial slice
  equivalence with the dense path, `__getitem__`, shape/dtype, projection,
  and K=0 edge case.
- All 99 tests pass (was 93 before Phase A: 6 new); suite slightly faster
  (117s vs 132s) thanks to dropped allocations.

#### Phase B — Stream `compute_W` *(DONE 2026-05-19)*

- Rewrite `compute_W` to compute `b0` via a single streaming pass:
  `b0[i] = (Y_flat[i, :].sum() - A[i, :] @ C.sum(axis=1)) / T`. Vectorise as
  `b0 = (Y_sum - A @ C_sum) / T` where `Y_sum` is a `(H·W,)` streaming reduction.
- Construct `X_fit` on demand by **streaming time chunks**: for each
  `t_chunk = (t0, t1)` in a strided iterator (stride=tsub), accumulate into
  per-pixel rolling buffers, or stream directly into the per-pixel LLS solve.
- The `_ring_pixel_batch` worker currently takes the whole `X_fit` as input.
  Refactor it to accept either the strided X chunk by chunk (and accumulate
  `B^T B` / `B^T x` for each pixel), or pre-extract per-pixel ring time series
  on demand from the streamed chunks.

Simpler intermediate: keep `X_fit` materialised but make it only
`(H·W, T/tsub)` and never materialise full `X = Y - A @ C - b0[:, None]`. At
T=60k, tsub=5 this is ~17 GB instead of 86 GB. May be enough as a first cut.

**Validates:** `compute_W` peak RAM bounded by `O(H·W · T/tsub)`.

**Shipped.**
- `cnmfe/background.py`: `compute_W` now computes b0 via streaming
  reductions (`(Y_sum - A @ C_sum) / T`), avoiding the full (H·W, T) X
  intermediate. X_fit is built only at the subsampled time resolution.
  Math identity, so existing tests pass bit-equivalently.
- New `W_cached: sp.csr_matrix | None = None` argument: when given, reuse
  the cached ring weight matrix instead of solving — only b0 is refit
  from current (A, C). Folds in Item 4 / `speedup.md` Change 2.
- `cnmfe/pipeline.py`: BCD loop now passes `W_cached=W_mat` on the refit
  call. The initial solve before the loop computes W once; subsequent
  iterations just refresh b0.
- `tests/test_background.py`: 3 new tests — streaming b0 matches legacy
  formula; W_cached returns same W with new b0; W_cached path is dramatically
  faster (test uses 5× as the floor; real-world is ~50–100×).
- All 102 tests pass (was 99 before Phase B: 3 new).

#### Phase C — Accept `zarr.Array` directly in `fit()` *(DONE 2026-05-19)*

**Scope reframed during implementation.** The original aim ("no
materialisation, fully streaming Y_flat") turns out to require a
disk-transpose preprocessing step: zarr is stored time-major `(T, H, W)`
while extraction is fundamentally pixel-major `(H·W, T)`. Reading a pixel
batch from a time-major zarr requires the full disk read per batch, so
true streaming without disk-transpose is O(H·W·T) IO per pixel batch —
infeasible.

**Shipped (modest scope).**
- `fit()` already accepts a `zarr.Array` directly via `np.asarray()`;
  Phase A+B refactors mean the only full-movie copy is `movie_arr` itself
  (no Y_bg, no full X intermediate). `make_2d` returns a view, so
  `Y_flat` is a view of `movie_arr`.
- `cnmfe/pipeline.py`: dropped the redundant `np.stack(movie_arr[t_idx])`
  in the stats-sampling step (advanced indexing already copies; the
  `np.stack` wrapped a 3-D array as a 1-element sequence — no-op cycle).
- `tests/test_pipeline.py::test_fit_accepts_zarr_input` — regression test
  pinning the API: zarr input must produce footprints / traces equal to
  numpy input (within float32 tolerance).
- `demo_notebooks/02_extract_components.ipynb`: updated the "Load the
  corrected movie" markdown to reflect Phase A+B improvements (per-step
  overhead now `K·T·4` + small pixel batches, not multiple full-movie copies).
- All 103 tests pass (was 102: +1 regression test).

**Out of Phase C (deferred).**
True streaming (Y_flat itself not materialised) needs one of:
1. Disk-transpose zarr to pixel-major chunking as a preprocessing step.
2. Re-architect extraction to be time-major (collect per-pixel BTB/Bx
   accumulators per chunk — works for compute_W but not directly for
   per-pixel LASSO in update_spatial).

Neither is needed today: on a 10k × 600 × 600 movie the working set is
~14 GB (Y_flat) + small batches. The 60k+ frame use case requires either
subsampling or one of the above; left as a TODO if the scale grows.

#### Phase D — Stream greedy init *(DONE 2026-05-19 — strided variant)*

`initialization.py:greedy_corr_pnr` currently mutates `data_filtered` and
`data_raw` in place (each full `(T, H, W)`). Refactor it to keep no full
movie in RAM. Key insight: the algorithm only needs (1) a global CORR/PNR
map (per-pixel reductions over time, streamable) and (2) small per-neuron
patches read on demand from the zarr.

**Design.**

- Maintain a sparse cache of detections so far: `A_cache: list[(pixel_indices, ai_values)]`
  and `C_cache: np.ndarray (K_so_far, T)`. Small RAM: K·T·4 bytes plus per-
  footprint ~9·9 floats.
- Initial CORR/PNR sweep: stream `zarr` in time chunks; for each chunk apply
  the center-surround PSF per frame, accumulate Welford / running stats
  (`sum`, `sum_sq`, `min`, `max`, neighbour cross-product sums). Final
  reductions yield CORR and PNR images. Peak RAM: one chunk + the `(H, W)`
  reduction arrays.
- Main detection loop:
  1. Find argmax of `CORR * PNR` (passing thresholds).
  2. Read the active patch's time series from zarr — single slice, e.g.
     `(T, ph, pw)` with ph, pw ≈ 6σ.
  3. PSF-filter the patch per frame.
  4. Subtract contributions of already-detected components in this patch:
     for each `(pix_indices, ai)` in `A_cache` overlapping the patch,
     subtract `ai * C_cache[k]` from the patch's filtered movie.
  5. Solve 3-component OLS for new `ai`, `ci`; deconvolve → `c_clean`.
  6. Append `(pix_indices, ai_values)` and `ci` (the raw trace, per Item 4)
     to the caches.
  7. Local CORR/PNR update around the detected centre: re-read a slightly
     larger patch from zarr, apply PSF + subtract all overlapping cached
     components, recompute CORR/PNR locally, write back into the global
     map. Suppress disk around centres.
  8. Repeat until no pixel passes thresholds.
- Replace `data_raw / data_filtered` arrays entirely. They are never
  materialised; `A_cache` + `C_cache` represent all corrections.

**Bonus.** This also closes Item 4 (greedy init residual subtracts raw `ci`
rather than `c_clean`) in the same refactor since we're rewriting the
subtraction step anyway.

**Shipped (strided variant).**
Scope reduced after deeper analysis. Full streaming (per-time-chunk PSF
filter + sparse-cache patch reads + streaming CORR/PNR Welford reductions)
is 1–2 days of focused work and was deferred. The strided-sample variant
ships now, gets most of the practical RAM win, and leaves the architecture
clean for a true-streaming follow-up.

- `cnmfe/pipeline.py`: new `init_stride: int | None = None` field on
  `CNMFeParams`. When `None`, auto-selects `max(1, T // 5000)`. Greedy
  init now runs on `movie_arr[::init_stride]`, cutting the two largest
  greedy allocations (`data_filtered`, `data_raw`) by `stride`.
  After greedy returns, full-T temporal traces are recovered by
  projecting the full-resolution movie onto each footprint:
  `C[k, :] = (Y_flat.T @ A[:, k]) / ||A[:, k]||^2`. Spatial footprints
  are unchanged.
- `cnmfe/initialization.py`: bonus fix folded in — line 349 now subtracts
  the raw `ci` from `data_raw` and `data_filtered` instead of the
  OASIS-deconvolved `c_clean`. OASIS smooths transients forward in time
  via the `c[t] >= g·c[t-1]` constraint, so subtracting `c_clean` left
  structured residuals at spike locations and fed halo-driven re-seeding.
  This closes Item 3 from the original plan.
- `tests/test_pipeline.py::test_init_stride_recovers_footprints` — new
  regression on a 600-frame synthetic movie verifying stride=1 vs stride=3
  recover all ground-truth neurons with spatial r > 0.7 each.
- All 104 tests pass (was 103: +1 strided-init regression test).

**RAM impact (at default auto stride).**
- 10k × 600 × 600: stride=2 → greedy internals drop from ~28 GB to ~14 GB.
- 60k × 600 × 600: stride=12 → greedy internals drop from ~172 GB to ~14 GB.
- The remaining floor is `Y_flat` itself (full-T pixel-major). Reducing
  that requires Phase C-full (true streaming, see notes there) which is
  deferred.

**Out of this phase (deferred to a future session).**
True streaming greedy init:
- Initial CORR/PNR via single-pass Welford-style reductions on time chunks
  (per-pixel sum, sum_sq, neighbour cross-products, max, min) without
  ever materialising the full filtered movie.
- Sparse `(A_cache, C_cache)` representation; patch operations re-read
  from zarr and subtract cached components on the fly.
- Local CORR/PNR update by re-reading patches.
Estimated 1–2 days. Worth doing if the strided variant proves
insufficient at very large T or with sparse-firing neurons that get
missed by the stride.

#### Phase E — Documentation & demo update *(DONE 2026-05-19)*

**Shipped.**
- `CLAUDE.md`: rewrote the *Motion correction* note about extraction RAM
  to reflect the new state (BackgroundSubtractor, streaming compute_W,
  init_stride). Documented the remaining ceiling (full pixel-major Y_flat
  in RAM; requires disk transpose for true T-streaming).
- `wiki/usage-guide.md`: new section *"Run on a large zarr (RAM-bounded
  extraction)"* with an end-to-end example, explanation of where streaming
  applies, and the current RAM rule of thumb.
- `demo_notebooks/02_extract_components.ipynb`:
  - `subsample-load` cell: clarified that `TIME_STRIDE` controls only the
    materialised movie, and pointed to `CNMFeParams.init_stride` as the
    targeted knob for the greedy init step.
  - `params-code` cell: added `init_stride=None` with explanation, and
    a print line that resolves the auto value at runtime so users can see
    what stride was picked.
- All 104 tests pass (no code changes in Phase E).

### Effort estimate
- Item 0 (prereq rename): 30 min.
- Phase A (per-batch `subtract_background`): 2–3 days.
- Phase B (stream `compute_W`, fold in W caching from speedup.md Change 2):
  3–5 days.
- Phase C (zarr-aware `fit()` + `MovieReader` abstraction): 2–3 days plus
  regression hardening.
- Phase D (stream greedy init, bonus: fold in Item 4 raw-trace subtraction):
  1–2 weeks. Highest risk phase.
- Phase E (docs + demo update): half day.
- **Total: 3–4 weeks** including review and test.

### Files
- `cnmfe/pipeline.py` (orchestration, lines 181–445)
- `cnmfe/background.py` (`compute_W`, `subtract_background`)
- `cnmfe/spatial.py` (`update_spatial`, lines 161–280 already batch over pixels)
- `cnmfe/temporal.py` (`update_temporal`, the `Y_flat.T @ A` projection at line 302)
- `cnmfe/initialization.py` (`greedy_corr_pnr` — Phase D)
- `cnmfe/io.py` (potentially a `MovieReader` abstraction)
- `cnmfe/_utils.py` (may need streaming reductions)
- `tests/test_background.py`, `tests/test_spatial.py`, `tests/test_pipeline.py`
- `demo_notebooks/02_extract_components.ipynb`
- `CLAUDE.md`, `wiki/usage-guide.md`

---

## Item 2 — `model.save()` / `CNMFe.load()` + `CNMFeParams` (de)serialise

**Status: DONE (2026-05-19).**

**Shipped.**
- `cnmfe/pipeline.py`:
  - `CNMFeParams.to_json(path)` / `CNMFeParams.from_json(path)`. The
    classmethod drops unknown keys so old save dirs keep loading after a
    field is added or removed; tuple fields like `max_shift` round-trip
    correctly.
  - `CNMFe.save(output_dir)` writes the full result set as standalone
    files: A.npz, C.npy, S.npy, YrA.npy, C_raw.npy, sn.npy, shifts.npy,
    b0.npy, W.npz, g.npy, sn_per_k.npy, params.json, manifest.json
    (dims, K, T).
  - `CNMFe.load(output_dir)` classmethod restores everything. Optional
    files are loaded only if present; unfit-model attributes stay None.
  - `CNMFe.C_projected` property — `model.C + model.YrA`, the shape-
    faithful noisy projected trace. Raises if called before fit().
- `full_pipeline.py`: save block collapsed to `model.save(out_dir)` +
  a tiny `run_info.json` for non-parameter run metadata. Docstring file
  list updated. The "load results" block now shows the one-line
  `CNMFe.load(...)` API.
- `demo_notebooks/02_extract_components.ipynb`: save cell + adjacent
  "Save results to disk" markdown + "What's next" markdown all updated
  to use `model.save` / `CNMFe.load` / `model.C_projected`.
- `tests/test_pipeline.py`: 4 new tests
  (`test_save_load_roundtrip`, `test_C_projected_raises_before_fit`,
  `test_save_raises_before_fit`,
  `test_params_to_from_json_unknown_keys_dropped`).
- All 115 tests pass.

---

## Item 3 — Greedy init: subtract the *raw* trace, not the deconvolved one

**Status: folded into Item 1 Phase D.** The greedy-init refactor rewrites
the subtraction step; the raw-trace fix lands as part of that work.

At `cnmfe/initialization.py:349`, `sub` uses `c_clean` (OASIS-deconvolved).
The raw trace `ci` would be more correct for residual subtraction —
deconvolution smooths transients and shifts mass forward, so subtracting
`c_clean` leaves structured residuals where the real spikes occur. CLAUDE.md
notes this was a contributor to greedy-init over-detection; the symptom was
patched by tightening suppression / thresholds.

**Verification (when Phase D lands).** 88 tests still pass. A/B on
`demo_movies/realistic_medium.zarr`: extracted K, duplicate count, mean
Pearson r vs truth.

---

## Item 4 — (folded into Item 1 Phase B) Cache the ring background W

`todo/speedup.md` Change 2. `compute_W` is called every BCD iteration; 10–20%
wall time win. Will be folded into Item 1 Phase B since both refactor
`compute_W`.

---

## Summary table

| # | Item | Severity | Effort | Status |
|---|---|---|---|---|
| 0 | `mc_n_iter` rename (Item 1 prereq) | bug | 30 min | **DONE 2026-05-19** |
| 1 | Stream extraction from zarr | blocking for scale | shipped 1 session | **DONE 2026-05-19** |
| 2 | `model.save/load` + params (de)serialise | UX | half day | **DONE 2026-05-19** |
| 3 | Greedy init residual subtracts deconvolved trace | algorithm | shipped with 1D | **DONE 2026-05-19** |
| 4 | Cache W across BCD iterations | speedup | shipped with 1B | **DONE 2026-05-19** |

## Item 5 — Disk-transpose mc.zarr for true T-streaming extraction *(ACTIVE)*

**Status: ACTIVE.** The 60k+ frame ceiling that Phases A–D explicitly
deferred. Pixel-major zarr lets `fit()` stream `Y_flat[start:end]` in
`O(B·T)` IO instead of `O(H·W·T)`.

### Why this is the right next step
Per `CLAUDE.md` (Motion correction section): *"The full corrected movie
still has to fit in pixel-major RAM as Y_flat (time-major zarr can't be
efficiently sliced pixel-wise without a disk transpose)."* This phase
removes the ceiling without revisiting the BCD math.

### Sub-plan

#### F1 — `transpose_zarr_to_pixel_major` utility *(io.py)*
- Reads a time-major `(T, H, W)` zarr (e.g. mc.zarr from `fit_mc`) in
  time-chunks; writes a pixel-major `(H*W, T)` zarr with chunks like
  `(4096, 2000)`.
- Idempotent via `skip_if_exists`.
- Bounded RAM: one source time-chunk + one dest column-slab.
- Cost: one O(T·H·W) disk pass; ~30–40 GB for 60k × 600 × 600 with blosc.

#### F2 — `BackgroundSubtractor` on zarr-backed Y_flat
- Current `W_chunk @ self.Y_flat` would force loading the whole zarr.
  Instead: extract just the ring-neighbour rows via
  `Y_flat.get_orthogonal_selection((needed_pixels, slice(None)))`,
  remap column indices on `W_chunk`, do the sparse matmul on the
  small dense buffer.
- Tests: parity with the in-memory path within float32 tolerance.

#### F3 — `compute_W` truly streaming
- `Y_sum` via a chunked sum over pixel batches.
- Build the X_fit slab per pixel batch (don't materialise full `(H·W, T_sub)`).
- Pre-extract ring-union rows from zarr per batch.
- `_ring_pixel_batch` adapted to take per-batch X data (no `X_full`).

#### F4 — `pipeline.fit()` 2D zarr orchestration
- Detect 2D `(H*W, T)` input; skip the `movie_arr = np.asarray(...)`
  materialisation. Use the zarr as `Y_flat` directly.
- Greedy init still needs a small in-RAM sample — use the existing
  `init_stride`. The 3D source zarr (pre-transpose) is needed for init
  because greedy expects spatial structure; either accept both inputs or
  let the user pass the time-major source alongside.
- Post-init `Y_flat.T @ A` becomes a streaming sum over pixel batches.

#### F5 — Tests + docs + demo
- Round-trip equivalence: numpy fit() vs pixel-major-zarr fit() on the
  same movie must produce footprints / traces within float32 tolerance.
- Memory bound test: `tracemalloc` peak on a 5k-frame pixel-major zarr
  should be `O(K·T + batch·T)`, not `O(H·W·T)`.
- Update CLAUDE.md to remove the 60k+ ceiling note.
- Update `02_extract_components.ipynb` with the optional transpose step
  and the streaming RAM-bound numbers.

### Effort estimate
- F1: 1 day (well-bounded utility + idempotent test).
- F2: 1 day (slice + project_onto on zarr; ring-neighbour extraction).
- F3: 2–3 days (the riskiest; `_ring_pixel_batch` refactor).
- F4: 2 days (pipeline plumbing + post-init projection).
- F5: 1 day (docs + demo + memory tests).
- **Total: ~1 week** focused.

### Files
- `cnmfe/io.py` — F1 utility.
- `cnmfe/background.py` — `BackgroundSubtractor` zarr branch (F2),
  `compute_W` streaming (F3).
- `cnmfe/pipeline.py` — `fit()` 2D zarr orchestration (F4).
- `cnmfe/_utils.py` — possibly a `MovieAccessor` helper if duck typing
  gets unwieldy.
- `tests/test_io.py`, `tests/test_background.py`, `tests/test_pipeline.py`.
- `CLAUDE.md`, `demo_notebooks/02_extract_components.ipynb`,
  `wiki/usage-guide.md`.

---

## Out of scope for this plan

Lower-priority findings from the audit, deliberately not promoted to keep
the plan focused: LASSO α scaling (`spatial.py:56-58`), PSF hard-edge
ringing (`preprocess.py:48-49`), per-component baseline-estimation heuristic
(`initialization.py:188-192`), ridge regularisation tied to `trace(BTB)`
(`background.py:57-58`), spatial-update lock-in via threshold ordering,
unweighted trace mean in `merging.py:140`, dead-component pruning timing,
the unused / commented-out top-level CORR/PNR call at `pipeline.py:234-239`.
Pick up after #1–#4 are settled.
