# Extraction performance session — May 2026

Running log of the multi-session push to make `CNMFe.fit()` actually
parallelise and stream on real-data shapes (T = 10k–60k, H×W = 600×600,
K ≈ 100–300). Use this as the "what's the current state, what's next"
reference when picking the work back up.

---

## What was shipped (in commit order)

| Commit | What | File(s) |
|---|---|---|
| `cf2e1b3` | Auto-stream zarr movies via `output_dir`; log `Extraction config` line at fit start; parallelise streaming `compute_W` via threads | `pipeline.py`, `background.py` |
| (earlier in same set) | `model.C_raw` aligned with `model.A` through merge / alive / auto-eval contractions | `pipeline.py` |
| (earlier in same set) | Scratch zarr cleanup in motion-correction `try/finally` | `motion_correction.py` |
| (earlier in same set) | New E2E regression test for `fit(Y_flat_zarr=...)` deep path + C_raw alignment invariant | `tests/test_pipeline.py` |
| `985540f` | Notebook 02 sync: auto-streaming path, parallel compute_W, fit-start config line | `demo_notebooks/02_extract_components.ipynb` |
| `9caa60d` | Packaging: declare `cv2` + `pyav`, add `[oasis]` extra, README sync | `pyproject.toml`, `README.md` |
| `bce0069` | Switch four `Parallel(...)` sites from loky processes to `prefer="threads"` | `spatial.py`, `temporal.py`, `initialization.py`, `background.py` |
| `c6febed` | `update_spatial`: swap `LassoLars` (sklearn estimator) for `enet_coordinate_descent_gram` (Cython, GIL-releasing) | `spatial.py` |
| `9b60ef4` | Cap inner BLAS threads to 1 via `threadpool_limits` inside every parallel section (Linux OMP oversubscription fix) | `spatial.py`, `temporal.py`, `background.py`, `initialization.py`, `preprocess.py`, `pyproject.toml` |

Deferred-work notes (rationale captured so we don't lose context):
- `todo/b0_float64_accumulator.md` — float64 sum hygiene for streaming `b0`
- `todo/greedy_init_streaming.md` — stream `greedy_corr_pnr`'s `data_filtered` / `data_raw` materialisation

---

## Measured impact

### Synthetic (T=11k, H=W=192, K=60, Windows 16-core)

| Stage | Serial wall | Parallel wall (n_jobs=-1) | Parallelism factor |
|---|---|---|---|
| baseline (`LassoLars` + loky processes) | 13.1 s | 13.0 s | 0.99x |
| + threads | 13.0 s | 13.0 s | 1.18x |
| + Cython CD direct (`enet_coordinate_descent_gram`) | 10.4 s | **9.0 s** | 1.52x |

So at synthetic scale on Windows: ~1.4× wall speedup, but per-core efficiency
still poor (only 9% of theoretical 16x parallelism).

### Real data (T=11k, H=W=600, K=133, 16-core Ubuntu)

- Before `9b60ef4`: `update_spatial` ≈ 20+ minutes per BCD iteration.
- Root cause: OpenBLAS bound at 16 threads × 16 joblib worker threads =
  256-way oversubscription thrashing the Linux scheduler.
- After `9b60ef4`: should land in **2–3 minutes per iteration** based on
  synthetic-extrapolated prediction (n_pixels × per-pixel-time + per-pixel
  Python overhead). **TODO: confirm on the real machine.**

---

## What's left on the table

### High value, medium effort

1. **Vectorised projected-gradient LASSO** (option 2 from the earlier
   discussion). The current per-pixel CD has irreducible Python overhead
   per pixel (~0.25 ms × 360k pixels = ~90 s on its own). A solver that
   processes M pixels at once via batched gradient steps would
   amortise that overhead.
   - Risk: solutions match LARS only to within tolerance (~1e-3
     deviation per coefficient). Footprint masks could shift by 1 px
     at boundaries.
   - Effort: ~150 lines + careful tests.
   - Expected payoff: 5–10× on top of the current ~1.4×.

### Medium value, medium effort

2. **Stream `greedy_corr_pnr`'s filter pass**
   (`todo/greedy_init_streaming.md`). Currently materialises
   `data_filtered` + `data_raw` (~21 GB transient on 60k × 600 × 600
   strided init). Only matters when init RAM is the bottleneck — most
   workstation users have headroom.

3. **Parallel `update_spatial` per-component post-processing**. After
   the parallel LASSO finishes, the parent loops serially over K
   components to apply `threshold_footprint`. For K = 500 this can be
   measurable. Trivial to parallelise but small absolute win.

### Low value, low effort

4. **Float64 accumulators in streaming `b0`**
   (`todo/b0_float64_accumulator.md`). ~1e-4 absolute drift at T = 60k;
   not currently affecting correctness on any real data.

5. **Stricter shape invariants**: add a per-step `assert
   A.shape[1] == C.shape[0] == len(g_per_k) == len(sn_per_k)` after each
   major state-changing call. Pure defensive programming.

### Unknown until measured

6. **Confirm the `9b60ef4` Linux fix lands the predicted 2–3 min on the
   Ubuntu machine.** If it doesn't, the next-most-likely culprit is
   "real data has dense per-pixel support that makes CD do many
   iterations" — landing us at item 1 above.

---

## Diagnostic recipes that worked

Useful patterns to remember next time something looks slow:

### "Is parallelism actually happening?"

```python
import time, os
# inside the suspect call:
t_wall_0 = time.perf_counter()
t_cpu_0  = time.process_time()      # sums CPU across threads in this process
... do work ...
wall = time.perf_counter() - t_wall_0
cpu  = time.process_time() - t_cpu_0
print(f"wall={wall:.2f}s cpu={cpu:.2f}s parallelism={cpu/wall:.2f}x "
      f"(theoretical max {os.cpu_count()})")
```

`cpu/wall` is the effective parallelism factor. 1.0 = serial. n_cores
= perfectly parallel. Below 1.0 with `n_jobs > 1` = overhead is
*hurting*. 1–2 with high core count = GIL contention or oversubscription.

### "What's actually slow inside this function?"

```python
import cProfile, pstats
pr = cProfile.Profile()
pr.enable()
your_func(...)
pr.disable()
pstats.Stats(pr).sort_stats("tottime").print_stats(20)
```

**Sort by `tottime`, not `cumtime`** — cumtime double-counts work
across the call stack and misled me in this session. `tottime` is the
time spent *in the function itself*, excluding children. That's what
tells you where the actual hot spot is.

### "Is BLAS oversubscribed on Linux?"

```python
from threadpoolctl import threadpool_info
import os
print("CPU count:", os.cpu_count())
print("OMP/MKL/OPENBLAS env:", [os.environ.get(v, "<unset>")
      for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")])
for entry in threadpool_info():
    print(entry["prefix"], entry["user_api"], "threads=", entry["num_threads"])
```

If BLAS rows show `threads=N` where N is your core count AND you're
about to call code that uses joblib threads → expect 256-way thrash on
N=16. Fix: `with threadpool_limits(limits=1, user_api="blas"): ...`
around the parallel section.

### "Did my code change actually take effect in this notebook?"

Re-import doesn't fully reload C/Cython modules cached in
`sys.modules`. Restart the kernel after non-trivial source edits, or
use IPython's `%load_ext autoreload; %autoreload 2`. Confirm the
`Extraction config:` line prints the expected `n_jobs` / `streaming`
values — those are guaranteed-fresh from `params` and from runtime
detection, so if the line says something stale you're running old code.

---

## How to verify on the real machine

After pulling `9b60ef4`:

```bash
git pull
pip install -e .                 # picks up threadpoolctl as a direct dep
# in your notebook kernel: kernel restart
```

Then run the same pipeline you ran before. Watch for:

1. `Extraction config: n_jobs=-1 device='cpu' T=11000 H=600 W=600 streaming=...`
   confirms latest code.
2. `update_spatial` per-iteration wall time. Target: < 5 min, ideally 2–3
   min. If still > 10 min, paste the `tottime` cProfile output from a
   real-data `update_spatial` call and we look at item 1 (vectorised
   LASSO) next.

---

## File map

Production code touched this session:
- `minicnmfe/spatial.py` — Cython CD direct, threads, BLAS cap
- `minicnmfe/temporal.py` — threads, BLAS cap
- `minicnmfe/background.py` — threads (both compute_W branches), BLAS cap, lazy slicing
- `minicnmfe/initialization.py` — threads, BLAS cap
- `minicnmfe/preprocess.py` — threads, BLAS cap
- `minicnmfe/pipeline.py` — auto-streaming, config log, C_raw tracking
- `minicnmfe/motion_correction.py` — scratch zarr cleanup
- `pyproject.toml` — `threadpoolctl`, `opencv-python`, `av`, `[oasis]` extra
- `README.md` — doc sync
- `demo_notebooks/02_extract_components.ipynb` — auto-streaming, parallel updates
- `tests/test_pipeline.py` + `tests/test_background.py` — new streaming regressions

Plan / context files:
- `C:\Users\Florian\.claude\plans\luminous-wibbling-karp.md` — last plan
  (BLAS cap fix). Useful as a template for similar perf-investigation plans.
- `todo/b0_float64_accumulator.md`, `todo/greedy_init_streaming.md` —
  deferred items.
