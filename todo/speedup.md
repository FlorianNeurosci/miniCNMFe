# CNMFe pipeline speedup opportunities

**Context:** On a 64×64×6500 synthetic movie (10 neurons), CaImAn runs in ~58 s while
our pipeline takes ~137 s (1-iter) or ~197 s (2-iter). Parameters are already matched
(ssub=1, tsub=1, ssub_B=1). The gap is structural, not parametric.

This document is written for an implementing agent. Read `CLAUDE.md` first for project
architecture, key invariants, and bugs that must not be re-introduced.

---

## Change 1 — Skip OASIS on the first temporal pass (p=0 mode)

**Estimated saving: 25–40% of total wall time**

### What CaImAn does
CaImAn's single refinement cycle runs two temporal updates:
1. `update_temporal` with `p=0` → non-negative NNLS, no AR deconvolution at all
2. `update_temporal` with `p=1` → full OASIS (once, after merging)

We run full OASIS on every `update_temporal` call. Our 1-iter runs OASIS 2× (BCD pass +
final update), our 2-iter runs it 4×. CaImAn runs it 1×.

### Proposed implementation

Add a parameter to `CNMFeParams` (`minicnmfe/pipeline.py`):
```python
skip_first_deconv: bool = True   # use NNLS (p=0) for the first temporal pass
```

In `minicnmfe/temporal.py`, `update_temporal` already accepts `g_cached` / `sn_cached`.
Add an optional `deconvolve: bool = True` argument. When `False`, replace the
`_deconvolve_with(g, sn, trace)` call with a simple non-negative clip:
```python
C_k = np.maximum(trace, 0)   # p=0: non-negative NNLS, no AR shape
S_k = np.zeros_like(C_k)
```
The rest of the BCD (residual update, spatial solve) is unchanged.

In `minicnmfe/pipeline.py`, thread the flag through the first BCD temporal call:
```python
# iteration 0, first temporal pass:
C, S, g_per_k, sn_per_k = update_temporal(..., deconvolve=not p.skip_first_deconv)
# all subsequent temporal calls always deconvolve:
C, S, g_per_k, sn_per_k = update_temporal(..., deconvolve=True)
```

### Invariants to preserve
- `update_temporal` still returns the same 4-tuple `(C, S, g_per_k, sn_per_k)`.
- `g_cached` / `sn_cached` are still threaded in unchanged — they are just unused when
  `deconvolve=False`.
- Tests: `test_temporal_correlation_against_truth` must still pass. The final temporal
  update always deconvolves, so the AR fix (no re-estimation from deconvolved traces)
  is unaffected.

---

## Change 2 — Cache ring background across BCD iterations

**Estimated saving: 10–20% of total wall time**

### What we do now
`pipeline.fit()` calls `subtract_background` (which calls `compute_W`) before every
spatial update AND before every temporal update within each main iteration. For
T=6500, H×W=4096, each `compute_W` is an O(N × n_ring² × T) operation.

### Proposed implementation

Compute W once before entering the main BCD loop and reuse it. Background is
refit with the *current residual* each time, but the ring weights W do not need to
change unless A changes significantly.

In `minicnmfe/pipeline.py`, split `subtract_background` into two calls:
1. Compute W once (before the loop): `W, b0 = compute_W(Y_res, ring_indices)`
2. Inside the loop, only recompute the background *signal* `b0` (cheap: just
   `ring_indices @ Y_res.mean`), not the full LLS weight solve.

Alternatively, the simpler approach: compute W once before the loop, pass it into
`subtract_background` as an optional `W_cached` argument so it skips the solve.

Relevant functions:
- `minicnmfe/background.py` — `build_ring_indices`, `compute_W`, `subtract_background`
- `minicnmfe/pipeline.py` — the main BCD loop (~lines 270–330)

### Invariants to preserve
- The initial W computation (before any A/C refinement) uses the raw residual, same
  as now.
- Do not cache W across a full `fit()` call restart — it must be recomputed at the
  start of each `fit()`.

---

## Change 3 — OASIS speed (lower priority)

**Estimated saving: 10–15% of OASIS-related time**

The `oasis-deconvolution` package is pure Python PAVA. CaImAn's `constrained_foopsi`
is Cython-compiled. For K=10 neurons at T=6500 this is a few seconds.

Options (in order of effort):
1. **Already have**: `oasis-deconvolution` is faster than the pure-Python PAVA fallback
   in `temporal.py`. Ensure the package is installed (`pip install oasis-deconvolution`).
2. **Future**: Wrap CaImAn's Cython solver as an optional backend — but this would
   introduce a CaImAn dependency, which violates the project's "no CaImAn import" rule.
3. **Not recommended**: Write a Cython/Numba PAVA — high effort, marginal gain for K≤50.

No code change needed for option 1 (it's already the default when installed).

---

## What NOT to do

- Do not add `ssub`/`tsub` spatial/temporal downsampling during initialization. CaImAn's
  default (ssub=2, tsub=2) is its largest real-data speedup, but it degrades spatial
  resolution and the upsampling adds complexity. Our design principle is full-resolution.
- Do not reintroduce per-iteration AR re-estimation — see CLAUDE.md *Bugs already fixed*.
- Do not change `update_temporal`'s 4-tuple return signature.

---

## Verification

After implementing Changes 1 and 2, run:

```bash
pytest tests/ -v                          # all 77 tests must pass
pytest tests/test_pipeline.py -v          # especially test_temporal_correlation_against_truth
```

Then re-run `tutorial_caiman_compare.ipynb` cells 3–8 and check:
- `ours` (2-iter) wall time drops from ~197 s to ~100–120 s
- `ours` (1-iter) wall time drops from ~137 s to ~70–80 s
- Spatial r and temporal r scores are unchanged (or better due to better first-pass init)
- CaImAn metrics are unchanged
