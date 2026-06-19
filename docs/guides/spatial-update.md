# Spatial update

Source: `minicnmfe/spatial.py`. Entry: `update_spatial(...)`. Re-estimates the
footprints `A` given the current traces `C` and the background-subtracted movie.

## The per-pixel regression

Each pixel is modelled as a non-negative mix of the traces of the components that
reach it:

```
Y_bg[p, :]  ≈  Σ_{k ∈ active(p)}  A[p, k] · C[k, :]
```

So row `p` of `A` is found by regressing that pixel's (background-subtracted)
trace onto `C[active]`. `active(p)` is small — only components whose footprints
are near pixel `p` — which keeps every solve tiny and well-conditioned.
`compute_support` builds these sets by dilating each component's binary footprint
by `dilation_radius` pixels (default 2) and recording which pixels each dilated
footprint covers.

The solve is a **non-negative elastic-net coordinate descent**, calling
`sklearn`'s Cython kernel `enet_coordinate_descent_gram` directly (or an
equivalent numba `prange` kernel on the parallel path). For each pixel it builds
the small Gram `C_active · C_activeᵀ` and `C_active · y_p`, then runs cyclic
coordinate descent with:

- **L1 penalty** `α = λ_p · T`, where
  `λ_p = lambda_scale · 0.5 · sn_p · √(max_energy) / T`
  (`max_energy = max diag(Gram)`). This is the per-pixel sparsity threshold a
  pixel's `C_k·y` must clear to turn on. `lambda_scale > 1` tightens footprints at
  the source (default 1).
- **L2 penalty** `β = spatial_ridge · max_energy` (default `spatial_ridge = 1e-2`).
  This is **not** a size knob — it bounds the condition number of the per-pixel
  Gram to ≈ `1/spatial_ridge` so the descent converges in tens of iterations even
  when active traces are near-duplicates (a near-singular Gram otherwise crawls to
  the `max_iter` cap). Set `0` for pure LASSO.

> Note: the real solver is the elastic-net coordinate descent described above
> (`enet_coordinate_descent_gram`), not `LassoLars`.

The result prints a one-line `update_spatial stats:` diagnostic (mean/max CD
iterations, how many hit the cap, mean active-set size) so a slow update can be
attributed to non-convergence vs. dense active sets vs. sheer pixel count.

## Footprint cleanup

Each raw column is then cleaned by `threshold_footprint`, in order:

1. **3×3 median filter** (removes isolated pixels), clip to `≥ 0`.
2. **Threshold faint pixels**, one of two methods:
   - `"max"` — zero pixels below `max_thr · peak` (peak-relative; the legacy
     behaviour, the `threshold_footprint` function default).
   - `"nrg"` — energy thresholding: keep the brightest pixels whose summed `a²`
     reaches `nrg_thr` of the total. Squaring discounts a footprint's dim skirt,
     so sprawled low-contrast footprints get trimmed cleanly. **This is the
     `CNMFeParams` default** (`spatial_thr_method="nrg"`, `spatial_nrg_thr=0.95`).
3. **Binary closing** (`closing_radius`, default 1 = 3×3) to fuse one-pixel gaps,
   so a jagged LASSO support survives as one blob rather than fragmenting.
4. **Largest connected component** only.
5. **Circular constraint** (optional) — clip pixels farther than
   `circular_max_dist_factor · √(area/π)` from the centroid, trimming "tendril"
   extensions toward neighbours. With `max_radius_factor > 0`, also cap the clip
   distance at an absolute `max_radius_factor · sigma` px (the area-derived radius
   grows with a sprawled footprint and stops biting; the absolute cap doesn't).

Components whose footprint went all-zero are dropped by the caller after the
update. The cleanup runs per component and is parallelized over `n_jobs`.

## Sizing knobs (dense FOVs)

Footprint size is set in three complementary places, all exposed on
`CNMFeParams`: `spatial_lambda_scale` gates which pixels the LASSO turns on
(sparsity), `spatial_thr_method="nrg"` / `spatial_nrg_thr` trims dim skirts
(intensity), and `spatial_max_radius_factor` clips far pixels (geometry). They
all default to no-op or the validated `nrg@0.95`, so the bit-for-bit serial path
is preserved unless you opt in.
