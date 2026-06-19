# Temporal update & deconvolution

Source: `minicnmfe/temporal.py`. Entry: `update_temporal(...)`; deconvolution in
`deconvolve` / `_oasis_ar1_pava`; AR estimation in `estimate_ar_params`.

Each trace is modelled as an **autoregressive** calcium signal driven by a
non-negative spike train:

```
C[k, t] = Σ_τ g^τ · S[k, t−τ] + baseline       (AR(1):  C[k,t] ≈ g·C[k,t−1] + S[k,t])
```

`g` is the per-frame decay (fraction of fluorescence retained each frame); `S ≥ 0`
is the inferred spikes.

## Estimating the decay g

`estimate_ar_params` recovers `g` from a trace by:

1. **Noise** `sn` from the high-frequency PSD (same band as
   [noise estimation](seeds-corr-pnr.md)).
2. **Detrend** the trace (mean-subtract, or subtract a degree-`detrend_order`
   polynomial — a slow bleach trend has lag-1 autocorrelation ≈ 1 and would
   otherwise dominate the fit).
3. **Yule-Walker** — solve the Toeplitz autocorrelation system `R·g = r` for the
   AR(`p`) coefficients.
4. **Shrink** the estimate, by one of two paths:
   - **Legacy** — multiply by `fudge_factor` (default 0.96) to avoid
     over-estimating the decay on clean traces.
   - **Bayesian prior** — when both `decay_time_ms` and `frame_rate_hz` are set,
     derive `g_target = exp(−1 / (fps · τ_s))` (`g_from_decay_time`) and shrink
     `g[0]` toward it: `g = (1−w)·g_yw + w·g_target` with `w = g_prior_weight`.
     The `fudge_factor` is **bypassed** on this path (the prior already encodes
     the physical bound).

In the pipeline `g` is estimated **once**, right after init, from the **pooled**
raw traces (`global_ar=True`) or per neuron (`False`), then cached on `model.g`
and reused in every BCD iteration. Re-estimating `g` from already-deconvolved
traces each iteration would re-apply the `fudge_factor` and drift `g` toward 0 —
so the cache (threaded as `g_cached`/`sn_cached`) is load-bearing, and merges
inherit `g` from the first member rather than re-estimating.

## OASIS deconvolution

Given `g` and `sn`, `deconvolve` infers `(c, s, baseline)`:

- **Fast path** — the compiled `oasis-deconv` package (constrained OASIS,
  `penalty=1`) for AR(1) or AR(2).
- **Fallback** — a pure-Python AR(1) PAVA solver (`_oasis_ar1_pava`) when the
  package is absent. It solves the L1-penalised noise-constrained problem
  `min ‖y − c‖² + λ·Σ s[t]` s.t. `c[t] ≥ g·c[t−1] ≥ 0`, choosing `λ` by bisection
  to meet the noise budget `‖y − c‖² ≈ T·sn²`. Its pool-merge test uses
  `g^{pool_length}` (not bare `g`) — using bare `g` over-merges smooth exact-`g`
  decays and collapses the trace.

## The BCD trace update

`update_temporal` first projects the data onto the footprints, computing
`YA = Y_bgᵀ·A` (via the lazy `BackgroundSubtractor.project_onto` so the movie
isn't materialized) and `AA = AᵀA`. The residual at each footprint is
`YrA = YA − (AA·C)ᵀ`. Then, for `n_iter` sweeps, each component's trace estimate

```
trace_k = YrA[:,k] / nA[k] + C[k]            (nA[k] = ‖a_k‖²)
```

is deconvolved (optionally polynomial-detrended first via
`temporal_detrend_order`), and `YrA` is updated with the change. Two scheduling
modes:

- **`n_jobs == 1`** — Gauss-Seidel: each `C[k]` update immediately feeds the next.
- **`n_jobs != 1`** — Jacobi: all `K` traces deconvolved in parallel from the
  previous iteration's residuals, then `YrA` updated. Slightly slower
  convergence, scales with cores.

Returns `(C, S, g_per_k, sn_per_k)` — note the **4-tuple**.

## C vs C + YrA

`C` is the deconvolved output (clean AR shape, for spike timing). The pipeline's
final pass also recomputes the residual `YrA`, so that **`C + YrA`** is the data
projected onto each footprint — same shape as the raw fluorescence, the
shape-faithful trace. The `skip_first_deconv` param runs NNLS (no OASIS) on the
first BCD pass for speed, deconvolving on later passes.
