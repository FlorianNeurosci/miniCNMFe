# Future-improvement roadmap

A prioritized list of algorithmic/scale/validation improvements, distilled from the
other `todo/*.md` notes plus a fresh read of the `minicnmfe/` modules. Three axes the
maintainer cares about: **accuracy on real data**, **scale & online**, **validation
infra**. (CaImAn feature-parity items — non-rigid MC, AR(2), multiday registration —
are deliberately *not* prioritized here.)

Through-line: **validation gates everything.** The `g`-over-smoothing fix (Bayesian
prior, May 2026) was reasoned + unit-tested but never confirmed on a real drift-heavy
recording. We can't tune accuracy knobs we can't measure — so build the measuring
stick first.

Tier tags: **[T1]** do-first · **[T2]** next · **[T3]** later/large R&D.

---

## Axis C — Validation infrastructure (the measuring stick)

### C1 — Real-recording ground-truth harness [T1]
All validation is synthetic; `wiki/caiman-comparison.md` flags a "home-field effect"
(the realistic simulator's separable rank-1 background suits our ring model). Nothing
is trustworthy on real data until this exists.

Add `validation/` (mirror `full_pipeline.py` conventions) that runs the staged pipeline
on a real movie and reports metrics. Three modes, increasing fidelity:
1. **Cross-method agreement** vs CaImAn on the *same* real movie — footprint IoU, count
   agreement, trace corr on matched pairs. Reuse `demo_notebooks/tutorial_caiman_compare.ipynb`.
2. **Expert-ROI recall/precision** — accept Suite2p/CaImAn-accepted ROI masks; centroid+IoU matching.
3. **Paired-ephys trace fidelity** — correlate `model.C` and `model.C+model.YrA` vs true
   spike train (the only mode that validates *deconvolution*, not just detection).

Files: new `validation/run_validation.py` + `validation/metrics.py`; reuse `fit_extract`,
`evaluate`, and the comparison-notebook footprint matching. Report must separate
`C` vs `C+YrA` correlation (per wiki guidance).

### C2 — A/B sweep harness for the experimental knobs [T2]
Cutout, detrending, prior weight, parallel-sessions are all "passing tests, never run on
real data" (CLAUDE.md *Experimental*). Thin sweep driver on C1: run a param grid on a
fixed real movie, tabulate C1 metrics, accept a knob only if it *measurably* helps.

---

## Axis A — Accuracy on real data

### A1 — Robust spike-aware *trace-level* detrend (IRLS) [T1]
(`todo/temporal_followups.md` #2, `todo/oasis_oversmoothing.md`.) **Already in place at
the movie level:** `minicnmfe/detrend.py` (`detrend_movie`) subtracts a rolling
lower-percentile baseline per pixel — spike-robust by construction — as a preprocessing
step. What remains is the *trace-level* detrend: `temporal.py` `_detrend_poly` (run just
before Yule-Walker / OASIS) is still plain least-squares (`np.polyfit`), so sparse spikes
pull the fit up — which is why `temporal_detrend_order` must default to 0 and slow drift
leaks into the AR fit (upward `g` bias, "shark-fin" transients). Fix: IRLS — fit →
down-weight residuals above median (spikes) → refit ×2–3 → tracks baseline not spikes.
Enables a defensible nonzero default. Files: `minicnmfe/temporal.py`; new test in
`tests/test_temporal.py` (recover baseline from drift+spikes without clipping spikes).

### A2 — Decay-segment-only `g` estimator [T1]
(`todo/temporal_followups.md` #1.) Even an *ideal* projection gives `g≈0.957` vs true
0.90–0.93 because slow background survives ring subtraction; the Bayesian prior masks but
doesn't fix the estimator. Fix: estimate `g` only on detected inter-spike *decay*
segments (genuinely AR), pool across components, keep the prior as a sanity bound. Files:
`minicnmfe/temporal.py` `estimate_ar_params` (new path, Yule-Walker+prior as fallback); thread
through the same call sites the prior uses. Verify via C1 paired-ephys + recovered-τ
scatter (the diagnostic `todo/gcamp8m_notebook_g_comparison.md` wants).

### A3 — Stronger component evaluation [T2]
`minicnmfe/evaluate.py` has only pixel-count + mean-amplitude SNR. Add two non-destructive
metrics (keep "never drop, tag `accepted_mask`" + multi-threshold pass-high-AND-low logic):
- **spatial r-value** — footprint vs. peak/mean activity-image patch correlation.
- **temporal exceptionality** — prob. that trace peaks exceed the noise distribution.
Deliberately **not** porting CaImAn's CNN (needs trained weights + labels). Files:
`minicnmfe/evaluate.py` (extend `auto_evaluate_components`/`eval_info`, persist in save/load),
new `CNMFeParams` thresholds, extend `test_auto_evaluation_rejects_ghosts`.

### A4 — Fix the regressed rank-1 global background [T2]
(`todo/temporal_followups.md` #0.) `global_bg_rank=1` is broken (xfail) after the Phase-D
revert. A working low-rank term captures slow global drift that per-pixel `b0` + single
ring miss — attacking the *root cause* of the `g` bias (A1/A2), not the symptom.
Recalibrate the alternating-LS `bf·f(t)` update (`pipeline.py` ~107–183) with cleaner `C`,
warm-start across BCD; keep opt-in (default 0 = byte-identical). Files: `minicnmfe/pipeline.py`,
`minicnmfe/background.py`; un-`xfail` `test_bf_and_f_capture_real_rank1_structure`.

### A5 — float64 accumulator for `b0` [T3]
(`todo/b0_float64_accumulator.md`.) Streaming `b0` sums in float32; per-pixel error ~1e-4
grows with T (risk on T>100k). ~1-line fix: accumulate reductions in float64, store
float32. Files: `minicnmfe/background.py` `compute_W`.

---

## Axis B — Scale & online

### B1 — Streaming greedy initialization [T1]
(`todo/greedy_init_streaming.md`.) Greedy init materializes `data_filtered`+`data_raw`
(~21 GB transient for 60k×600×600) — the **last** hard RAM ceiling (everything else is
streamed) and what caps concurrent-session count. Fix: persist `data_filtered` to a temp
zarr + lazy reads in the greedy loop (preferred, matches `Y_flat` pattern), or re-filter
per seed (CPU-for-RAM). Init sample is already strided. Files: `minicnmfe/initialization.py`
(~line 250, `greedy_corr_pnr`), `minicnmfe/pipeline.py` `fit_extract` init; keep in-RAM path
for small movies (dual-path like MC). Test: identical seeds vs in-RAM path (bit-for-bit at
`n_jobs=1`) + peak-RAM assertion.

### B2 — Incremental extraction MVP (toward OnACID-E) [T2]
Extraction is fully batch. Full OnACID-E is a big rewrite, but streaming MC + streaming
BCD already exist, so a pragmatic middle ground: *block-incremental* — fit on an initial
time block, then per subsequent block warm-start spatial/temporal from current `A`/`C`,
run greedy init on the *residual* to add new components, merge. Files: new `minicnmfe/online.py`
composing existing `fit_extract` building blocks. Test: incremental ≈ batch (IoU + trace
corr) on a synthetic movie. (Stage 2 [T3]: true frame-by-frame OnACID-E with online
deconv + online MC.)

### B3 — Patch-based streaming BCD for very large FOVs [T3]
(`wiki/architecture.md`.) Greedy *init* already has a patch-parallel variant
(`init_patches`); extend the overlapping-patch + border-dedup machinery to the
spatial/temporal/background BCD for very large FOVs. Low priority — only bites beyond
current FOV targets; B1+streaming cover the common large-T case. Files: `spatial.py`,
`temporal.py`, `background.py`, `pipeline.py`.

---

## Suggested execution order
1. **C1** (validation harness) — unblocks measuring everything.
2. **B1** (streaming greedy init) — cheapest big scale win, self-contained.
3. **A1 + A2** (robust detrend + decay-segment `g`) — core real-data accuracy gap, measured against C1.
4. **A3 + A4** (eval metrics + rank-1 bg) — second-order accuracy, measured via C1.
5. **C2**, **A5**, **B2**.
6. **B3** + full OnACID-E — only if scale demands it.

## Invariants to preserve (standing project rules)
- New accuracy knobs **default off / standard CNMF-E** → `test_stage_split.py` bit-for-bit holds.
- `n_jobs==1` path bit-identical; `n_jobs>1` may reorder float32 reductions (~1e-6).
- Module-level workers only (spawn-pickling); BLAS caps in parallel branches.
- `update_temporal`/`merge_components` stay 4-tuples; `g` estimated once + cached — **never**
  re-estimate from deconvolved traces (the AR-drift bug).

## Done-ness bar
No accuracy change is "done" until **C1 on ≥1 real recording** shows it helps (or is
neutral) — cross-method agreement at minimum, paired-ephys trace corr if a dataset exists.
