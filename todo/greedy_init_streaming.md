# Deferred: stream `greedy_corr_pnr`'s filter pass

**Status:** deferred (audit finding, May 2026). Not currently fixed.
**Trigger to revisit:** init RAM becomes a real bottleneck on a real recording.

## What

`minicnmfe/initialization.py:greedy_corr_pnr` materialises both of the following at
the start (line 250 onwards):

```python
movie = np.asarray(movie, dtype=np.float32)        # the strided init sample
# ... PSF convolution ...
data_filtered = np.stack([...])                    # full (T_init, H, W) copy
data_raw      = movie.copy()                       # another full (T_init, H, W)
```

The strided sample size is `T_init * H * W * 4` bytes. On a typical 60k × 600 × 600
recording with `init_stride = T // 5000 = 12`, that's:
- `T_init ≈ 5000`
- one (T_init, H, W) copy ≈ **7.2 GB**
- the second copy ≈ **7.2 GB**
- transient `data_filtered` intermediate during the PSF stack ≈ another **7.2 GB**

Total: **~21 GB of transient RAM during init**, even when the user is on the
true T-streaming extraction path.

## Why deferred

Fixing it is invasive. Two viable shapes:

1. **Persist `data_filtered` to a temp zarr.** The PSF pass becomes a one-time
   disk write; the greedy loop then reads frames lazily. Adds a tempdir
   dependency and a cleanup story (when to delete the temp on success/failure).

2. **On-the-fly re-filter per seed.** Drop `data_filtered` entirely; recompute
   the PSF-filtered patch around each candidate seed inside
   `extract_spatial_temporal`. Costs more CPU (each seed extraction triggers a
   small local convolution) but keeps init RAM at zero overhead beyond the
   movie itself.

Both are substantial refactors that touch the greedy loop's invariants
(global cn/pnr updates, suppression disk semantics). The current 21 GB cost is
manageable on the kind of machine that runs 60k recordings (≥ 64 GB RAM
is typical), so the user judged this not worth doing right now.

## When to pick this back up

- A real recording fails with OOM during init while extraction succeeds.
- Init RAM becomes the binding constraint preventing larger recordings.
- We add a "init on streamed disk-only" mode for batch/cluster workflows where
  RAM is provisioned tightly.

## Related code

- `minicnmfe/initialization.py:250` — the asarray + filter pass.
- `minicnmfe/pipeline.py:fit` — `init_stride` already strides the input; this
  only reduces the constant, not the order.
- `minicnmfe/preprocess.py:make_center_surround_psf` — the PSF kernel used by
  the filter pass. Small (~`(2 * 4 * sigma + 1)^2`) and cheap to apply.
