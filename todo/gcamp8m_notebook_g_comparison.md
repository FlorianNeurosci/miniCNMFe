# Remaining work: GCaMP8m decay in notebooks + head-to-head `g` comparison

## Done (this change)
- `minicnmfe/temporal.py`: added `g_from_decay_time(decay_time_ms, fps)` and
  `decay_time_from_g(g, fps)` (single source of truth for `g = exp(-1/(fps·τ_s))`).
- `tests/miniscope_simulator.py`: `make_miniscope_movie` now derives per-neuron `g`
  from `decay_time_ms` (default **180 ms = GCaMP8m**) at `fps`, with a `decay_time_jitter`
  (default ±10% on τ). `decay_time_ms=None` → legacy `ar_decay_range` draw. Returns a new
  `decay_time_ms` key; `g_true`/`ar_decay` unchanged in shape.
  Verified: default mean g≈0.755 (jitter band [0.736, 0.776] at 20 fps); 8f(70 ms)→0.50,
  8s(350 ms)→0.87; legacy still [0.86, 0.96].

## Remaining — `demo_notebooks/tutorial_caiman_compare.ipynb`

Variable names confirmed present: `cnm_r` (CaImAn realistic model), `ours_r2` (our 2-iter),
`A_true_r`, `g_true_r`, `K_r`, `H`, `W`, helpers `match_components_by_truth(A_est, A_true)`
→ list of `(kt, ke, r)`, `pearson`, `overlay`.

1. **Movie cell** (`data_r = make_miniscope_movie(...)`, src ~line 532): add explicit
   `decay_time_ms=180.0` (default already, but make it visible). `fps=20.0` already there.
2. **Printout** (~line 553): currently prints the raw g range. Replace with GCaMP8m framing,
   e.g. report `decay_time_from_g(g_true_r.mean(), 20.0)` ≈ 180 ms and the g range ≈[0.74, 0.78].
3. **Section 8 markdown** (~line 523): bullet "`g` sampled in [0.86, 0.96]" →
   "GCaMP8m τ=180 ms @ 20 fps (g≈0.76) with small per-neuron jitter".
4. **Section 9 markdown** point 3 (~line 715): "`g_true` spans ~0.86–0.96" → GCaMP8m wording.
5. **Add a prior-path fit.** New params `params_r_prior` = `params_r2` (keeps `global_ar=False`)
   **plus** `decay_time_ms=180.0, frame_rate_hz=20.0`, then
   `ours_r_prior = CNMFe(params_r_prior).fit(movie_r, do_motion_correction=False)`.
   This is the one extra full fit on the 6500-frame movie — adds runtime.
6. **Add the g-comparison cell** (after the metrics cell ~line 671, before Section 9):
   ```python
   from minicnmfe.temporal import decay_time_from_g
   FPS = 20.0
   def recovered_g(model_or_cnm, matches, is_caiman):
       out = []  # (true_g, est_g) per matched neuron
       for kt, ke, _ in matches:
           if is_caiman:
               g = float(np.atleast_1d(np.asarray(model_or_cnm.estimates.g[ke]).ravel())[0])
           else:
               g = float(model_or_cnm.g[ke][0])      # our g: list of (1,) arrays
           out.append((float(g_true_r[kt]), g))
       return np.array(out)
   pairs = {
       'ours (fudge)':  recovered_g(ours_r2,       match_components_by_truth(ours_r2.A,       A_true_r), False),
       'ours (prior)':  recovered_g(ours_r_prior,  match_components_by_truth(ours_r_prior.A,  A_true_r), False),
       'CaImAn':        recovered_g(cnm_r,          match_components_by_truth(cnm_r.estimates.A, A_true_r), True),
   }
   # table: mean true τ, mean est τ, mean signed bias (ms) per pipeline via decay_time_from_g(·, FPS)
   # scatter: x = true τ_ms, y = recovered τ_ms per neuron, 3 series + identity line
   ```
   - Index note: `estimates.g` is in the same component order as `estimates.A`, so `ke` from
     `match_components_by_truth(cnm_r.estimates.A, A_true_r)` indexes both consistently.
   - **Report τ at FPS=20 for all** — the *true* g corresponds to the simulator's fps=20.
     CaImAn's `opts` `fr=30` only feeds its quality metrics, not its (data-driven, frame-rate-
     independent) g estimate; optionally set `fr=20` cosmetically to avoid confusion.
   - Expected story: ours(fudge) and CaImAn estimate g purely from data (CaImAn per-neuron);
     ours(prior) is pulled toward 180 ms by `g_prior_weight=0.5`. The scatter shows whether the
     fudge path / pooled-vs-per-neuron choice biases τ away from the GCaMP8m truth, and whether
     the prior fixes it. This is the concrete "is g a problem" answer.

## Remaining — other notebooks (lower priority)
- `demo_notebooks/tutorial_caiman_compare_patches.ipynb`: same movie-gen + printout + markdown
  updates (Cell ~20 movie, ~line 598 printout). Mirror the g-comparison cell if desired.
- `demo_notebooks/tutorial_realistic.ipynb`: Cell ~20 prints the heterogeneous g range and
  compares estimated g vs `g_true`. Update narrative referencing 0.86–0.96 → GCaMP8m; consider
  setting `decay_time_ms=180, frame_rate_hz=20` on its `CNMFeParams` to demonstrate the
  recommended config, and report recovered τ via `decay_time_from_g`.

## Also remaining (not notebooks)
- Tests for the helper + simulator (`tests/test_temporal.py`):
  `g_from_decay_time(180,20)≈0.758` round-trips; `make_miniscope_movie(decay_time_ms=...)`
  controls mean g (180>70 etc.); `decay_time_ms=None` restores [0.86, 0.96].
- Optional: refactor `pipeline.py:1302-1305` to call `g_from_decay_time` (bit-for-bit identical
  float expression — safe for `test_stage_split.py` and the prior regression test).
