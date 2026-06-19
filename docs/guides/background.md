# Ring background

Source: `minicnmfe/background.py` (`compute_W`, `BackgroundSubtractor`,
`build_ring_indices`) and `pipeline.py:_fit_global_bg_rank1`. This is the part of
CNMF-E that makes 1-photon data tractable.

## Why a ring

In a miniscope the out-of-focus background is large and **spatially correlated**:
neighbouring pixels share most of their background fluctuation. CNMF-E exploits
this — a pixel's background is predicted from a **ring** of pixels around it (far
enough out to exclude the pixel's own neuron, close enough to share the
background). The model is

```
X ≈ W · X ,    X = Y − A·C − b0
```

where `X` is the residual after removing the neural signal and a per-pixel
baseline, and `W` is a sparse weight matrix whose row `i` is nonzero only on the
ring around pixel `i`.

## Ring geometry

`build_ring_indices(dims, radius)` precomputes, for every pixel, the flat indices
of the pixels at Euclidean distance in `(radius, radius+1]` (a one-pixel-thick
annulus, built by subtracting an inner disk from an outer disk). Border pixels
simply have fewer ring neighbours. The radius used by the pipeline is
`ring_size_factor · (2·sigma + 1)` (default `ring_size_factor = 1.5`).

## Solving for W and b0

`compute_W` fits each pixel independently. The baseline is the temporal mean of
the residual, computed by streaming reductions so the full residual is never
materialized:

```
b0 = (Y_sum − A · C_sum) / T          # = (Y − A·C).mean(axis=1)
```

Then, for each pixel `i`, a **ridge-regularized least squares** over its ring
neighbours `B = X[ring_i]`:

```
w_i = argmin_w ‖X_i − w·B‖² + λ·‖w‖²
    = solve( B·Bᵀ + λ·trace(B·Bᵀ)·I ,  B·X_i )
```

with `λ = ring_lambda` (default `1e-5`, expressed as a fraction of
`trace(B·Bᵀ)`). The per-pixel solves are independent and parallelize over
`n_jobs`; `tsub` (`bg_tsub`, default 5) subsamples **time** for the expensive
`B·Bᵀ` solve only — `b0` always uses the full `T`.

Two performance options that don't change the math:

- **`W_cached`** — reuse a previously solved `W` and only refresh `b0`. The
  ring's spatial structure is a property of the data, not of `A`/`C`, so the BCD
  loop solves `W` once and reuses it across iterations.
- **Streaming path** — for a zarr-backed `Y_flat`, the residual rows for each
  pixel batch are pulled on demand (`_compute_w_streaming`) so the full
  `(H·W, T)` residual is never in RAM.

`constrain_sum=True` (param `ring_constrain_sum`, **non-standard**) adds the
equality constraint `Σ_j W[i,j] = 1` per row via a Lagrangian, so any spatially
uniform brightness change (LED flicker, uniform bleaching) cancels exactly in the
residual. Standard CNMF-E leaves it unconstrained (`False`).

## Subtracting the background lazily

The background-free movie is

```
Y_bg = (I − W)·(Y − b0)
```

(`subtract_background`). Materializing that is `(H·W) × T` — too big for long
recordings — so the BCD loop uses **`BackgroundSubtractor`** instead, which
computes pixel-row slices on demand from the identity

```
Y_bg[s:e] = Y[s:e] − b0[s:e] − W[s:e]·Y + (W[s:e]·b0)
```

For a zarr `Y_flat` it reads only the ring-neighbour rows it needs. It also offers
`project_onto(A)` = `Y_bgᵀ·A → (T, K)` without ever forming `Y_bg`, via
`Y_bgᵀ·A = Yᵀ·(I − Wᵀ)·A − b0·(...)`. Both spatial and temporal updates consume
the subtractor through these two methods, which is how extraction stays
RAM-bounded.

## Optional rank-1 global term

With `global_bg_rank=1` (non-standard), an extra term `b_f · f(t)` is fit on top
of the ring (`_fit_global_bg_rank1`). It captures **spatially non-uniform** slow
drift (vignetting-coupled photobleaching, scope warm-up) that the ring alone
cannot. It is fit by a few alternating least-squares sweeps on the residual
`R = (I − W)(Y − b0) − A·C`, both factors unconstrained (the drift is a signed
deviation from `b0`), warm-started across BCD iterations. When present,
`BackgroundSubtractor` also subtracts `b_f · f(t)`.
