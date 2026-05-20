# Temporal-path follow-ups (after sn-collapse fix)

The OASIS-collapse bug was that `sn` was estimated from `C_raw[k]` —
the trace returned by greedy init, which runs on the
center-surround-*filtered* movie and therefore has its
high-frequency content stripped. The PSD-based noise estimator then
returned `sn ≈ 0.005`, OASIS had no noise budget, and `model.C`
collapsed to a near-flat tail.

Fixed by switching to the closed-form footprint-weighted formula:

    sn_k = ‖a · sn_flat‖₂ / ‖a‖²

`sn_flat` is already computed by `estimate_noise` on the raw movie,
so the new path is mathematically principled (assumes per-pixel
independent noise — the standard CNMF assumption) and cheap.

The items below were identified during the investigation but
deferred to keep that fix tightly scoped.

## 1. g estimation from the projected trace, not C_raw — TRIED, INSUFFICIENT

**Attempted.** Reordered `CNMFe.fit` so `compute_W` runs before AR
estimation and ran Yule-Walker on the ring-subtracted projection
`(A · Y_bg) / ‖a‖²`. Reverted: did not change `g` on the
realistic-miniscope fixture.

**Diagnostic.** Even the **IDEAL projection** `(a_true · Y) / ‖a_true‖²`
gives `g ≈ 0.957` for every component (true range 0.90–0.93). Running
Yule-Walker on `C_true` *directly* gives `g ≈ 0.87–0.91` — the
correct band. So the input trace IS what biases the estimator; the
ring just doesn't subtract enough.

**Why.** The simulator's slow background is a random-walk smoothed
with `bg_temporal_sigma=30` frames (5 components, summed) plus
vignette·bleach modulation. The ring removes *local* slow modes
inside a ~10-pixel radius; the simulator's slow modes span tens of
pixels and survive the subtraction. Polynomial detrend
(`ar_detrend_order=0..3`) doesn't catch them either — random walks
aren't polynomial-shaped. Same with detrend on the projection trace.

**What would actually work** (any one of):
- **High-pass / rolling-window detrend** with window ≫ calcium tau
  but ≪ drift correlation length (e.g. ~60 frames vs τ_cal ≈ 10).
  Subtracts the large-scale slow drift while preserving transients.
- **Decay-segment-only g estimator** — detect candidate transients,
  fit g on inter-spike decay intervals (CaImAn's
  `constrained_foopsi` path does this).
- **Noise-constrained OASIS** (item #2 below) — sidesteps the issue:
  with the correct sn budget, OASIS doesn't depend on a tight g.

## 2. CaImAn-style noise-constrained OASIS

**Why it matters.** Our current OASIS (in `cnmfe/temporal.py`) uses a
ridge-style shrinkage penalty. CaImAn uses
`constrained_foopsi`/`oasisAR1` with an explicit per-component noise
*budget* (the residual variance must equal `sn²`). The constrained
form is provably consistent and avoids both collapse and overfit
without param tuning. Likely the cleanest long-term fix.

**What to change.** Replace `_deconvolve_with` in `cnmfe/temporal.py`
with the noise-constrained OASIS solver from `oasis-deconvolution`'s
`foopsi` family. Keep the same return signature `(c, s)` so
`update_temporal` is untouched.

## 3. Robust (spike-aware) detrend → raise the detrend defaults

**Why it matters.** The polynomial detrend introduced in commit
`77a7adb` is a least-squares fit; sparse positive spikes pull the
polynomial upward and OASIS reconstructs inflated transients
(visible as overshoot in the original `tmp/compare_temporals.png`).
Reverted to default 0 in commit `e2ba67b`. A robust detrend (iterative
reweight, lower-envelope fit) would handle both bleach-heavy and
activity-rich data, enabling a defensible default ≥ 1.

**What to change.** Replace `_detrend_poly` in `cnmfe/temporal.py`
with an IRLS variant: fit polynomial → identify residuals above
the median → re-weight those points → repeat 2–3 times. The lower
envelope of the trace becomes the bleach trajectory; spikes don't
participate in the fit.

## 4. Realistic-miniscope `r(IDEAL_proj, truth) ≈ 0.20` ceiling

**Why it matters.** On the realistic-miniscope fixture, the noisy
projection `(a_true · Y) / ‖a_true‖²` (the *best* `C+YrA` any
algorithm could produce, given the data) only correlates ~0.2 with
the clean `C_true`. The signal is overwhelmed by vignette·bleach
modulation, ghost-cell temporal cross-talk, shot noise, and 8-bit
quantization. Optimizing `C+YrA` past this ceiling is mathematically
impossible — we should be optimising the denoised `C` instead.

**What to change.** Two things:
- Update `tmp/compare_to_truth.py` and the comparison notebook to
  report and visualise `r(C, truth)` as the primary metric, with
  `r(C+YrA, truth)` shown only for diagnostic context.
- Document in `wiki/` (or CLAUDE.md) that on miniscope-quality data
  the relevant metric for evaluating extracted traces is the
  AR(1)-denoised `C`, not the noisy projection.
