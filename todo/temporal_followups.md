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

## 1. g estimation from the projected trace, not C_raw

**Why it matters.** Diagnostic showed `g` pinned at the 0.96
fudge-factor ceiling for every component on the realistic miniscope
fixture (true range 0.86–0.96). Smoothing in `C_raw` inflates the
lag-1 autocorrelation, so Yule-Walker biases `g` upward, and the
`fudge_factor=0.96` clamp then makes every neuron look like the
slowest one. OASIS uses this for the AR(1) decay constraint.

**What to change.** In `cnmfe/pipeline.py`, replace the
`estimate_ar_params(C_raw[k], …)` call with one that runs on the
*ring-subtracted projection* `(a · (Y - W·b0)) / ‖a‖²`. Requires
the ring step (`compute_W`) to be moved before AR estimation (or
estimate on the unsubtracted projection — the high-pass effect of
the ring is what matters for the autocorrelation). Mirror the sn
helper structure: a `_g_from_projection(a_k, Y, W, b0, …) → np.ndarray`.

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
