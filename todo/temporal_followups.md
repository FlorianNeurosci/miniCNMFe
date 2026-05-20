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
- **High-pass / rolling-window detrend** with window ≫ calcium tau
  but ≪ drift correlation length (e.g. ~60 frames vs τ_cal ≈ 10).
  Subtracts the large-scale slow drift while preserving transients.
- **Decay-segment-only g estimator** — detect candidate transients,
  fit g on inter-spike decay intervals (CaImAn's
  `constrained_foopsi` path does this).
- **Noise-constrained OASIS** (item #2 below) — sidesteps the issue:
  with the correct sn budget, OASIS doesn't depend on a tight g.

## 2. ~~CaImAn-style noise-constrained OASIS~~ — DONE (in the fallback)

**What was wrong.** The pure-Python `_oasis_ar1_pava` fallback accepted
`sn` in its signature but never referenced it — it was plain isotonic
LS under the AR constraint, no noise budget, no L1 penalty. With
`oasis-deconvolution` not installed in `claude_cnmfe` (the PyPI name
doesn't resolve, and the GitHub install is blocked by the auto-mode
classifier), every deconv call had silently been falling back to that
unconstrained PAVA — which is why the sn fix from commit `4831330`
had no visible effect on `model.C`.

**Fix.** Rewrote `_oasis_ar1_pava` as constrained foopsi (Friedrich
2017 §2.1): minimise `‖y − c‖² + lam·Σ s[t]` subject to the AR
constraint, with `lam` chosen by bisection so the residual variance
matches `T · sn²`. Pure Python, no new deps. The pool-value update
is `max(0, (num − lam/2) / den)` — only change from plain PAVA is the
`lam/2` shrinkage. Bisection runs ~10 PAVA sweeps per neuron.

**End-to-end effect (realistic miniscope fixture).** With the L1 fix
plus correct `sn` (commit 4831330), `r(C, truth)` rises from 0.02 to
0.08 with the wrong g, and to ~0.15 if `fudge_factor` is also lowered
(item #1). The realistic fixture's ideal-projection ceiling is
`r ≈ 0.20` (item #4) — most of the remaining gap is data-quality, not
algorithmic.

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
