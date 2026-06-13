# Tuning-sweep / extraction performance — findings & next steps

**Date:** 2026-06-14. **Hardware:** 128-core server (NUMBA_NUM_THREADS=128).
**How it's run:** `CalciumImagingPipelineDB` `server_script.py` → `MiniCnmfeTuning.populate`
→ `tuning.tuner.run_tuning` with `n_jobs = _n_processes` (server sets `N_PROCESSES=-1`).
Region = `cutout` (256×256 × 3000 frames per candidate).

**Original symptom:** only ~1 core at 100% during `greedy init` and `update_spatial`,
sweep very slow.

---

## What we changed (all merged to `master`; also on branch `numba-update-spatial`)

| commit | what |
|--------|------|
| `caf49d6` | **Decouple the sweep's two parallelism levels.** `--n-jobs` (= `_n_processes`) is now a *total core budget*, split in `run_sweep`: `cand_jobs = min(#candidates, budget)` loky **processes**, `inner_jobs = budget // cand_jobs` threads per fit. Previously one value drove both → 1 (serial) or N×N oversubscription. |
| `a4f5f89` | Resolve joblib's negative `n_jobs` (`-1`→all cores, `-2`→all-but-one) in the budget split (a bare `max(1,n_jobs)` turned `-1` into 1 = serial). |
| `b8b702e` | (user) print the candidate grid at sweep start. |
| `192482b` | Cap `update_spatial` CD threads to `spatial_thread_cap` (default 16). GIL-thrash hygiene — **necessary but not sufficient** (the CD path is GIL-bound; capping ≠ speedup). |
| `0c2cdbd` | **numba `@njit(parallel=True)` prange CD kernel** (`_spatial_cd_kernel`) — runs the non-negative elastic-net coordinate descent in compiled nogil code, escaping the GIL. Matches the sklearn serial path to <1e-6. numba added to `pyproject.toml` (pip-only wheels). |
| `64c4f0d` | Patch-parallel **init** for non-nested sweep fits (`cand_jobs==1` candidates + the best-model re-fit) + a `[...]` path label on the `update_spatial stats:` line. |
| `93d8185` | **Thread the background-subtraction slab** — the real `update_spatial` bottleneck. Adds `update_spatial timing: slab=.. cd=..` breakdown. |

---

## Key learnings (the why)

1. **Sweep ran serial** because `tune.py --n-jobs` defaulted to 1 and that single value drove
   both candidate-concurrency *and* each fit's inner `n_jobs`. → split into a budget (`caf49d6`).
2. **`update_spatial` per-pixel CD is GIL-bound** at `mean_active≈2` (mostly Python/numpy glue
   around a microsecond Cython solve). No thread count parallelizes it — confirmed: the thread
   cap didn't help, and numba (which made the CD truly parallel) *also* didn't move the stage,
   because **the CD was never the bottleneck for sparse supports**. `compute_W` scales only
   because its per-batch work is big GIL-released BLAS.
3. **The real `update_spatial` cost is the SERIAL ring-background slab:** every
   `Y_flat[start:end]` calls `BackgroundSubtractor.slice` (`background.py:664-705`) =
   sparse `W@Y` matmul + `Y_flat[needed]` copy. It releases the GIL, so threading it (waves of
   `n_workers`, `background.py:project_onto` pattern) scales. **This cut `update_spatial`
   112s → 35s.**
4. **numba is pip-only** (numba + llvmlite manylinux wheels); no conda. First call JIT-compiles
   (~0.5s warm, cached via `cache=True`); 16 candidate processes may compile concurrently on a
   cold cache → a one-time pause at the first "Updating spatial footprints…".
5. **Diagnostics now in the log:** `update_spatial stats: … [numba xN (slab-parallel)]` and
   `update_spatial timing: slab=Xs cd=Ys (xN threads, M blocks)`. Use these to see the path +
   split. If the label says `[threaded …]`, numba is **not** active in that env.

---

## Current profile (per sweep candidate, after all fixes)

```
Stage timings (total 231.4s, was 315.2s):
  noise estimation   11s   5%
  greedy init        96s  41%   <-- now #1
  compute_W          46s  20%   <-- #2
  update_spatial     35s  15%   (slab≈15s/iter on 16 threads, cd≈2.8s) — was 112s
  update_temporal    42s  19%
```

8 candidates run concurrently (`cand_jobs=8`, `inner=16`) → sweep wall ≈ slowest candidate.

---

## Open question / NEXT (resume here)

**Goal:** parallelize **greedy init** (96s, now the top cost). User does **NOT** want a
`max_neurons` cap (which would bound the over-seeding — loose-threshold candidates find
1857–2000+ seeds that merge down to ~242, inflating init *and* temporal).

**Constraint:** greedy init's only parallelization is **patch-init** (tile the FOV, run each
tile's serial seed loop in parallel loky **processes**; `greedy_corr_pnr_patched`). In the
concurrent-candidate sweep (`cand_jobs>1`) it **nests loky → joblib serializes it**, so it
can't run. It only engages at `cand_jobs==1` (already wired in `64c4f0d`).

**Experiment we were mid-implementing — a `--sweep-sequential` toggle (A/B test on real HW):**
run candidates one-at-a-time (`cand_jobs=1`) so each gets the full budget for patch-init.
- Estimate: **~break-even** on total wall (8 fast-sequential ≈ 1 slow-concurrent, because each
  candidate has irreducible serial bits: noise ~11s, temporal limited by component count).
  Worth measuring on the 128-core box rather than trusting the estimate.
- To implement: add `sequential: bool=False` to `run_sweep` (`tuning/sweep.py`) → set
  `cand_jobs=1` when true (patch-init then engages via the existing `cand_jobs==1` logic);
  `TunerConfig.sweep_sequential` (`tuning/tuner.py`); `--sweep-sequential` flag (`tune.py`).
  Test: `python tune.py <session> --sweep-sequential --n-jobs -1` vs without.

**Where patch-init clearly DOES win:** the single **full-recording validation/extraction** fit
(not the sweep) — one fit on all cores, so no nesting. Already wired (`good_defaults`
`init_patches=True` default + `init_stride=2` → numpy strided sample). **TODO: verify it
actually engages** there (log shows `Patch-parallel init: …`, FOV≥`init_patch_min_fov`=128).

**Secondary:** `compute_W` (46s) and `update_temporal` (42s) are next; both already threaded.
Temporal is also inflated by over-seeding (more components to deconvolve).

---

## Deployment notes

- `CalciumImagingPipelineDB` has **no submodule / vendored copy / version-pin** of
  `simpler_cnmfe` — it imports the **editable-installed** `minicnmfe`/`tuning`. So updating the
  server = pull **simpler_cnmfe** + `pip install -e .` (needed once, for the new `numba` dep);
  **no DB pull required** (`server_script.py:145` says as much). DB sets `n_jobs=_n_processes`
  unchanged → picks up the budget split automatically.
- Verify numba is live in the *running* env: `python -c "from minicnmfe.spatial import
  _HAS_NUMBA; print(_HAS_NUMBA)"` → must be `True`. (Server confirmed: numba 0.65.1, True.)
