# Deferred: float64 accumulator for streaming `b0` reductions

**Status:** deferred (audit finding; not load-bearing on current data).
**Source:** three-agent code audit (May 2026). Recorded for future reference.

## What

In `cnmfe/background.py:compute_W`, the streaming baseline reduction sums `Y_flat[s:e]` along the time axis in float32:

```python
C_sum = np.asarray(C.sum(axis=1), dtype=np.float32)
if Y_is_numpy:
    Y_sum = np.asarray(Y_flat.sum(axis=1), dtype=np.float32)
else:
    Y_sum = np.zeros(n_pixels, dtype=np.float32)
    sum_batch = 4096
    for s in range(0, n_pixels, sum_batch):
        e = min(s + sum_batch, n_pixels)
        Y_sum[s:e] = np.asarray(Y_flat[s:e], dtype=np.float32).sum(axis=1)
AC_sum = np.asarray(A @ C_sum, dtype=np.float32).ravel()
b0 = ((Y_sum - AC_sum) / float(T)).astype(np.float32)
```

NumPy uses pairwise summation for `.sum()` by default. For a float32 array of length `T` and per-element magnitude `M`, the absolute error in the sum is bounded by roughly `sqrt(log2(T)) * M * eps_fp32 ≈ 4 * M * 1e-7` at `T = 60000`. After dividing by `T`, the error in `b0` per pixel is around `1e-4` absolute, i.e. ~`2e-5` relative to a typical signal magnitude of 5.

**Tests are not failing.** The synthetic fixtures (`T = 300`) don't get anywhere near this regime, and on `T = 60000` recordings the bias is still well below the per-pixel noise floor.

## Proposed fix (when ready)

One-line change to force the reduction to accumulate in float64; store the result in float32 unchanged:

```python
C_sum = np.asarray(C.sum(axis=1, dtype=np.float64), dtype=np.float32)
if Y_is_numpy:
    Y_sum = Y_flat.sum(axis=1, dtype=np.float64).astype(np.float32)
else:
    Y_sum = np.zeros(n_pixels, dtype=np.float32)
    for s in range(0, n_pixels, sum_batch):
        e = min(s + sum_batch, n_pixels)
        Y_sum[s:e] = np.asarray(Y_flat[s:e]).sum(axis=1, dtype=np.float64).astype(np.float32)
```

Cost: ~zero (one temporary float64 per batch).
Benefit: defensive insurance for very long recordings where the accumulator dominates the per-pixel noise.

## Why deferred

The user judged this not useful right now — current correctness on real data is unaffected, and the change is opaque to anyone not staring at the numerical analysis. Revisit if extraction on `T > 100k` recordings shows mysterious b0 drift, or as part of a broader numerical-hygiene pass.
