# Temporal-path follow-ups

Live follow-up items on the OASIS / AR-estimation path. The original
`sn`-collapse bug (sn estimated from the centre-surround-filtered
`C_raw[k]` with stripped high-frequency content → near-zero `sn` →
OASIS collapse) was fixed in commit `4831330` by switching to the
closed-form footprint-weighted formula

    sn_k = ‖a · sn_flat‖₂ / ‖a‖²

with `sn_flat` from `estimate_noise(Y)`. Items below were identified
during that investigation and are still open. See also
`todo/oasis_oversmoothing.md` for the current actionable A/B on the
over-smoothing symptom.

## 0. Rank-1 BG (`global_bg_rank=1`) fit broken after Phase-D revert

After restoring greedy init's clean `c_clean` residual subtraction
(commit pending) and using greedy's per-pixel-OLS `C` as the initial
trace, the rank-1 BG alternating LS in `_fit_global_bg_rank1` no
longer matches its calibrated amplitude. On the bleach-heavy
fixture, `bf · f` now *increases* ring-residual variance instead of
reducing it; `r(f, bleach)` drops from 0.94 to 0.27.

The rank-1 feature is opt-in (default `global_bg_rank=0`), so the
default path is unaffected. The regression test
`test_bf_and_f_capture_real_rank1_structure` is marked xfail
pending a recalibration of the alternating-LS update with the
cleaner C as input. Likely needs a different normalisation in the
`bf` / `f` updates, or a fundamentally different initialisation.

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
- **Bayesian prior on g** — **implemented 2026-05-21.** Set
  `CNMFeParams.decay_time_ms` + `frame_rate_hz` (and optionally
  `g_prior_weight`); the Yule-Walker estimate is shrunk toward
  `g_target = exp(-1 / (fps · τ_ms / 1000))` and `fudge_factor` is
  bypassed. Plumbed through every `estimate_ar_params` call site
  (pipeline init, greedy init, `update_temporal` fallback). This is
  now the recommended fix for the upward-biased `g` problem.
- **High-pass / rolling-window detrend** with window ≫ calcium tau
  but ≪ drift correlation length (e.g. ~60 frames vs τ_cal ≈ 10).
  Subtracts the large-scale slow drift while preserving transients.
- **Decay-segment-only g estimator** — detect candidate transients,
  fit g on inter-spike decay intervals (CaImAn's
  `constrained_foopsi` path does this).
- **Noise-constrained OASIS** — already done; the fallback now uses
  L1 + λ bisection (commit `a8948d6`). With the correct sn budget,
  OASIS still depends on a tight g, but the symptom is now over-smoothing
  rather than collapse. See `todo/oasis_oversmoothing.md` for the
  concrete A/B that lowers `fudge_factor` and detrends the OASIS input.

## 2. Robust (spike-aware) detrend → raise the detrend defaults

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

## 3. Realistic-miniscope `r(IDEAL_proj, truth) ≈ 0.20` ceiling

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
