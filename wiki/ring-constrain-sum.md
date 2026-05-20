---
tags: [cnmfe, math, algorithm, background, extension]
---

# Sum-to-1 Constrained Ring Background

> Opt-in extension to the standard CNMF-E ring model. Enabled with
> `CNMFeParams(ring_constrain_sum=True)`. Off by default — set explicitly
> to True for recordings with global brightness events.
>
> See also: [[algorithm-math]] §6 (ring background), [[api-reference]]
> for `compute_W`.

## Problem

The standard CNMF-E ring model (Zhou et al. 2018) predicts the background at
pixel $i$ from a ridge-regularised LS fit over its ring of neighbours:

$$\hat{b}_i(t) = b_{0,i} + \sum_{j \in \text{ring}(i)} W_{ij}\,\big(Y_j(t) - b_{0,j}\big)$$

with `compute_W` solving the **unconstrained** problem

$$\mathbf{w}_i = \arg\min_{\mathbf w} \; \|\mathbf{X}_i - \mathbf{w}^\top \mathbf{X}_{\text{ring}(i)}\|^2 + \lambda\|\mathbf{w}\|^2$$

This works well for *local* spatial structure (vasculature halos, slow gradients)
but fails on **spatially-uniform** brightness changes — LED flicker, sensor gain
shifts, uniform photobleaching, motion-coupled illumination.

If every pixel drops by a constant $D$ at frame $t$, the residual at pixel $i$ is

$$r_i(t) = \big(Y_i(t) - b_{0,i}\big) - \sum_j W_{ij}\big(Y_j(t) - b_{0,j}\big)
        = D \cdot \Big(1 - \textstyle\sum_j W_{ij}\Big)$$

With ridge regularisation, $\sum_j W_{ij} < 1$, so a fraction of $D$ leaks
through and is re-projected onto every extracted component's trace.

### Observed symptoms

`tmp/output.png` (real recording, 2000 frames) showed both modes:

1. **Synchronous sharp dips** at ~650, 950, 1100, 1450, 1700, 1900 across
   every extracted trace — global brightness events leaking through.
2. **A large slow-drifting "neuron"** (orange trace, ~2.0 → ~1.0) — a ghost
   component parked on residual photobleaching that the ring couldn't
   cancel, also collecting the leaked dips.

## Fix

Add the equality constraint $\sum_j W_{ij} = 1$ per row. Any spatially-uniform
mode then cancels **exactly**:

$$r_i(t) = D - D \cdot \sum_j W_{ij} = D - D \cdot 1 = 0$$

Neural signal is sparse and non-global, so it is unaffected.

### Solver

Solved via Lagrangian — one extra RHS in the already-factored ridge gram:

```python
# Gram (unchanged): M = B B^T + lambda * trace(B B^T) * I

# Solve two RHSs at once
sols   = solve(M, column_stack([B @ y, ones(n_ring)]))   # (n_ring, 2)
w_unc, w_corr = sols[:, 0], sols[:, 1]

# Lagrange correction to satisfy 1^T w = 1
mu = (1 - w_unc.sum()) / w_corr.sum()
w  = w_unc + mu * w_corr
```

Cost: one extra RHS in a per-pixel solve that was already happening — negligible.

## Code locations

| File | What |
|---|---|
| `cnmfe/background.py:30-65`   | `_ring_pixel_batch` — CPU serial/joblib path |
| `cnmfe/background.py:67-122`  | `_ring_pixel_batch_slab` — zarr-streaming path |
| `cnmfe/background.py:154-242` | `_compute_W_gpu` — batched CuPy path |
| `cnmfe/background.py:341-491` | `compute_W` — entry point + dispatcher |
| `cnmfe/pipeline.py:50-130`    | `CNMFeParams.ring_constrain_sum` (new field) |
| `cnmfe/pipeline.py:656, 760`  | Two `compute_W` call sites in `CNMFe.fit()` |
| `tests/test_background.py`    | `TestRingConstrainSum` — 4 tests |

## Verification

Four targeted tests in `tests/test_background.py::TestRingConstrainSum`:

1. **`test_row_sums_are_one`** — every non-empty row of $W$ sums to $1 \pm 10^{-4}$.
2. **`test_unconstrained_rows_do_not_sum_to_one`** — confirms the unconstrained
   ridge solution shrinks row sums visibly below 1 (the failure mode this fix targets).
3. **`test_global_pulse_cancels`** — injects a uniform $-5.0$ brightness drop at
   one frame, fits $W$ on the clean movie, applies to both versions. Constrained
   leak is at the float-precision floor ($< 10^{-4}$); unconstrained leak is
   ~$2.4$ RMS. Ratio > $10^4 \times$ better.
4. **`test_default_is_unconstrained`** — calling `compute_W` without the kwarg
   produces an identical matrix to `constrain_sum=False`; confirms the standard
   CNMF-E behaviour is preserved when the flag is off.

All four pass. All 115 pre-existing background/pipeline tests still pass.

## When to enable

Set `ring_constrain_sum=True` when the recording has:

- Visible LED flicker or sensor gain steps (synchronous brightness changes across the frame).
- Strong global photobleaching.
- Motion-coupled illumination changes the rigid MC can't compensate for.

Leave the default (`False`) for:

- Strict reproduction of the published CNMF-E algorithm.
- Recordings where the unconstrained ring already works (most lab-quality data
  without visible global artefacts).

## Limitations / what this does NOT fix

The constraint only cancels modes that are **spatially uniform**. If the global
event has spatial structure (e.g. one corner dims more than another, vignetting-
coupled bleach), partial leakage remains. The next-step extension would be a
per-frame scalar background term $f(t)$ (CaImAn-style `nb >= 1`) layered on top
of the ring — deferred until evidence shows the constrained ring is insufficient
on the recording of interest.

## Standard CNMF-E reminder

This is a **non-standard** extension. The original Zhou et al. 2018 algorithm uses
unconstrained ridge LS. `CNMFeParams` tags every non-standard knob with
`[NON-STANDARD]` in its inline comment so a reader can audit which defaults differ
from the published algorithm. See the `CNMFeParams` docstring for the full
convention.
