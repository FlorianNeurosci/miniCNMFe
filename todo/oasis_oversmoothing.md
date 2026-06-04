# OASIS over-smoothing in `model.C`

**Status:** partially fixed 2026-05-21 by the indicator-prior path
(`decay_time_ms` + `frame_rate_hz` on `CNMFeParams`). The A/B knobs
(`fudge_factor`, `temporal_detrend_order`) below remain as fallbacks
when the indicator/fps aren't known.

**Preferred fix going forward.** Set `decay_time_ms` and `frame_rate_hz`
on `CNMFeParams` (see CLAUDE.md "Bayesian prior on `g`" section). The
pipeline then shrinks every Yule-Walker estimate toward
`g_target = exp(-1 / (fps · τ_ms / 1000))` with weight
`g_prior_weight` (default 0.5). This bypasses the `fudge_factor`
ceiling entirely and gives a principled physical-unit-grounded knob.

**Companion:** `todo/temporal_followups.md` section #1 ("g estimation from
the projected trace") — same root cause, the prior implementation is
the actionable resolution.

## Symptom

On `demo_notebooks/02_extract_components.ipynb` (and the realistic
miniscope fixture more generally), `model.C` (OASIS-deconvolved trace)
flattens visible spike-shaped transients that are clearly present in
`model.C + model.YrA`. Even on components that **pass auto-eval**, the
green `C` collapses most of the data into one event + smooth AR tail.

## Why OASIS smooths

OASIS in this codebase has two implementations behind a common API:

- `oasis-deconvolution` package, called with `penalty=1, g=(g_k,), sn=sn_k`
  (`minicnmfe/temporal.py:248-265`). Penalty=1 ⇒ L1-spike + noise-budget;
  `g` and baseline are **fixed** (no `optimize_g`, no `optimize_b`).
- Pure-Python fallback `_oasis_ar1_pava` (`minicnmfe/temporal.py:159-221`),
  which solves `min ‖y − c‖² + λ·Σ s[t]` s.t. AR(1) monotonicity, with
  **λ chosen by bisection so `‖y−c‖² ≈ T·sn²`**.

Three knobs control the smoothing:

| Knob | Where it's set | Failure mode (over-smoothing) |
|---|---|---|
| `g` (AR coefficient) | `estimate_ar_params` on pooled `C_raw.ravel()` (`minicnmfe/pipeline.py:906-916`), capped at `fudge_factor=0.96` | Too high → each spike's AR tail swallows the next spike → "shark fin" |
| `sn_per_k` | `_sn_from_footprint(A[:,k], sn_flat)` (`minicnmfe/pipeline.py:46-71, 919, 932`) | Too high → noise-budget target `T·sn²` is loose → λ pushed up → only the biggest spikes survive |
| What's in `y` | Projected trace `(YrA[:,k]/nA[k] + C[k])` (`minicnmfe/temporal.py:415`) | Slow drift in `y` → OASIS spends its noise budget on the drift residual → no budget left for small-spike resolution |

## What this run does specifically

1. **`g = 0.959` is pinned at the `fudge_factor=0.96` ceiling.** Per
   `temporal_followups.md` #1, the ideal-projection ground truth on the
   miniscope fixture is `g ≈ 0.90–0.93`. The pooled-`C_raw` estimate is
   biased upward by slow-drift residual the ring didn't subtract; the
   Yule-Walker autocorrelation sees the drift as a long calcium tail.
   τ at `g=0.959` is ~24 frames — wide enough that an event's tail
   covers the next event.

2. **`sn_per_k` from the footprint formula** is the projected
   *pixel-noise* std, `‖a·sn_flat‖₂ / ‖a‖²`. It does **not** include the
   variance of the slow-drift residual that OASIS can't fit. The noise
   budget `T·sn²` therefore implicitly assumes a clean AR(1)+iid-noise
   model — but the actual `y` is AR(1)+drift+noise.

3. **Net effect.** OASIS has to place the drift somewhere: model it as a
   long AR decay (eats the `g` budget — visible "shark fin"), or leave
   it in the residual (uses up the `T·sn²` budget). Either way, λ ends
   up too high for the small spikes; they get absorbed.

So "OASIS smooths too much" is exactly right. The OASIS code is
correct — it's being handed a `(g, sn, y)` triple that's inconsistent
with the data. Same underlying cause as the rest of `temporal_followups.md`:
slow background isn't fully subtracted by the ring.

## Cheapest next step — notebook A/B, no code changes

Both knobs are already on `CNMFeParams`. In notebook 02 cell
`fe7a623a126aa93d`:

| Param | From | To | Rationale |
|---|---|---|---|
| `fudge_factor` | 0.96 (default) | 0.85 | Cuts `g` from ~0.959 → ~0.85, τ from 24f → 6f. Less sticky AR ⇒ small spikes don't get swallowed by the prior decay. |
| `temporal_detrend_order` | 0 (default) | 1 | Subtracts a linear trend from each projected trace *just before* OASIS (`minicnmfe/temporal.py:417-420`). Removes the slow drift component that's eating the noise budget. Does **not** change `C + YrA` — the drift naturally flows into `YrA`. |

Run cells 11→17. Inspect:

1. **Visual:** the green `C` should now follow the peaks in blue
   `C + YrA` instead of flattening to one event + smooth decay.
2. **Numerical:** `np.array([g[0] for g in model.g]).min(), .max()`
   should drop below the old 0.959 ceiling.
3. **Regression:** `pytest tests/test_pipeline.py::test_temporal_correlation_against_truth`
   still passes (`r > 0.7` floor on the simple-synthetic fixture).

## If the A/B doesn't help

The upstream bottleneck is the ring background underfitting on a
2000-frame slice. Apply the previously-noted upstream fixes:

- Drop `movie = movie[:2000]` in cell 3 — run on the full 11000 frames.
- Restore the documented defaults: `n_iter_main=2`, `n_iter_temporal=2`,
  `skip_first_deconv=False`.

Better ring subtraction → less drift in `y` → both knobs above stop
mattering.

## Promotion plan

**Implemented 2026-05-21.** `decay_time_ms`, `frame_rate_hz`, and
`g_prior_weight` are exposed on `CNMFeParams` and threaded through every
`estimate_ar_params` call site (pipeline init, greedy init,
`update_temporal` fallback). When both `decay_time_ms` and
`frame_rate_hz` are set, the Yule-Walker estimate is shrunk toward
`g_target = exp(-1 / (fps · τ_ms / 1000))` and `fudge_factor` is
bypassed. The notebook params cell surfaces all three with the
indicator-τ table inline.

Tests:
- `tests/test_temporal.py::TestEstimateArParams::test_prior_pulls_g_toward_target`
- `tests/test_temporal.py::TestEstimateArParams::test_no_prior_unchanged`
- `tests/test_pipeline.py::TestCNMFePipeline::test_decay_time_prior_pulls_g_toward_target`
- `tests/test_pipeline.py::TestCNMFePipeline::test_decay_time_prior_disabled_when_either_none`

Remaining (opt-in) A/B knobs for users who don't know the indicator/fps:

- Bump `CNMFeParams.fudge_factor` default from `0.96` → `0.85` if the
  A/B run validates this on the realistic-miniscope fixture too.
- Bump `CNMFeParams.temporal_detrend_order` default from `0` → `1` if
  the A/B shows OASIS still over-smooths even with the prior applied.

These two stay as overrides because the prior is now the principled
path; `fudge_factor` only matters when the prior is disabled.

## Critical files (read-only references)

- `minicnmfe/temporal.py:159-221` — `_oasis_ar1_pava` (the actual smoother).
- `minicnmfe/temporal.py:308-464` — `update_temporal` (cache plumbing + call site).
- `minicnmfe/pipeline.py:46-71` — `_sn_from_footprint`.
- `minicnmfe/pipeline.py:906-932` — global vs per-component `g` + `sn` init.
- `todo/temporal_followups.md` — companion notes; section #1 documents
  the upstream g-bias the A/B is trying to compensate for.
