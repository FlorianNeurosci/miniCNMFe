---
tags: [minicnmfe, tuning, workflow, parameters]
---

# CNMFe — Automated Parameter-Tuning Workflow

> Point one command at a recording → it tests the recording, suggests
> motion-correction + extraction parameters, validates them on the full
> recording, and writes a self-contained `report.html` (recommended values +
> graphs) so you can judge quality **by eye, no AI needed**.
>
> For what each knob *does*, see [Parameter Tuning Guide](../getting-started/index.md#parameter-tuning-guide); for
> signatures see [API reference](../api/index.md).

`tune.py` is the **single front door** for the workflow in the `tuning/`
package: it runs the heuristics + graded sweep, then (by default, for AVI input)
full-recording validation, and writes both an HTML and a Markdown report.
`validate_session.py` (validate one session) and `batch_tune.py` (a list of
sessions in one background process) are internal stages it composes.

---

## What it produces

A new timestamped folder `runs/tune_<name>_<ts>/` (the default `runs/` parent is
gitignored) containing:

| File | Contents |
|------|----------|
| `report.html` | **Open this.** Self-contained (figures inlined): config + recommended-params table, a click-to-sort candidate table, every figure, and — when validated — the recommended-vs-lowthr comparison side by side. Judge candidates by eye, offline. |
| `recommended_params.json` | A real `CNMFeParams` JSON (NATIVE units) — feeds `run_mc.py` / `run_extract.py` directly |
| `downsample.json` | `ssub` / `tsub` (not `CNMFeParams` fields) — feeds `run_extract.py --ds-meta` |
| `report.md` | The same content as a Markdown file (terminal / GitHub view) |
| `fig_*.png` | One figure per suggested parameter + the sweep figures |
| `full/` | Full-recording validation (when validating): `comparison.md`, `cn.npy`, `run_<label>/` with the fitted model + diagnostic figures |

---

## Quick start

```bash
# tune + full-recording validate + HTML, all in one background run:
python tune.py /path/to/miniscope_video --indicator gcamp8m --n-jobs -1 &
# then open  runs/tune_<name>_<ts>/report.html  in a browser
```

> [!TIP]
> `--indicator gcamp8m` is shorthand for `--decay-time-ms 180`
> (table: 6f→140, 7f→160, 8f→70, 8m→180, 8s→350, 6s/7s→1000).
> `--no-validate` stops after the cutout sweep; `--no-html` skips the HTML;
> `--no-lowthr` drops the lower-recall validation set.

The recommended params then feed the staged pipeline directly:

```bash
python run_mc.py /path/to/movie -o mc/ --params runs/tune_*/recommended_params.json
python run_extract.py mc/mc.zarr -o results/ \
    --params runs/tune_*/recommended_params.json --ds-meta runs/tune_*/downsample.json
```

---

## Two depth modes (`--mode`)

| Mode | What it does | Speed |
|------|--------------|-------|
| `heuristic` | Image-based suggestions only — `blob_log` neuron radius, the 3-method `(min_corr, min_pnr)` proposals, shift histograms. No extraction (the morphology seed is reported). | Fast |
| `sweep` | Actually runs `fit_extract` across a grid of the key knobs and **scores** each candidate with quality proxies. | Slower |
| `both` *(default)* | Heuristics seed the grid; the sweep refines and adds the temporal/eval knobs. | Slower |

Only `sweep`/`both` (or `--existing-results <dir>`) produce the Stage-4 temporal
suggestions, because those need a fitted model to read.

## Two sweep regions (`--region`)

| Region | What it does | When |
|--------|--------------|------|
| `cutout` *(default)* | Runs the grid on a representative spatial+temporal window (auto-picked as the activity-dense region of the correlation image). | Long recordings — seconds-to-minutes per candidate |
| `full` | Runs the grid on the whole recording (streaming). | Short recordings, or when you want the faithful answer |

> [!NOTE]
> Cutout values are a **fast proxy**: the auto-picked window maximises summed
> CORR (activity-dense), not mean intensity (which chases vasculature /
> vignette). The report records the chosen crop so you can sanity-check it.

---

## Inputs

- **AVI folder** (`0.avi … N.avi`) — the live-session input. The tuner runs a
  quick fused MC on a subset (`--max-avis`) at the suggested `ssub`/`tsub` to
  produce a scratch `mc.zarr` for the extraction sweep.
- **`mc.zarr`** — an already-corrected movie. Skips the quick MC and the
  motion-correction heuristics; tunes extraction directly.
- `--reuse-mc-zarr PATH` — point the extraction sweep at an `mc.zarr` you
  already have (e.g. on a local SSD) regardless of the primary input.

---

## What gets suggested

**Motion correction** (`mc_gSig_filt`, `max_shift`, `border_px`) and the
downsample factors `ssub`/`tsub`.

**Initialisation** (`sigma`, `min_corr`, `min_pnr`, `min_pixel`) — `sigma` from
`blob_log` on the CORR·PNR image; `min_pixel` from the footprint-area distribution
of a fast greedy init. For the thresholds, three methods each propose a
`(min_corr, min_pnr)` operating point — **morphology** (threshold that maximises the
cell-like-blob count), **separation** (max Youden's J of CORR/PNR at detected neuron
centres vs background), and **percentile** (25th-pct of CORR/PNR at neuron centres) —
and the sweep tests all three as **coupled seeds**, picking the winner by quality
score (the winning method is named in `report.md`). The CORR/PNR images are computed
from a 2000-frame, chunk-aligned sample of `mc.zarr` (`n_init_frames`); the
chunk-aligned read avoids the strided-single-frame I/O blow-up on time-chunked zarrs.

**Sweep grid** (the most-impactful extraction knobs, per the real-recording
findings in [Tuning long or dense recordings](../getting-started/index.md#tuning-long-or-dense-recordings)): `sigma`,
`min_corr`, `min_pnr`, `merge_thr_corr`, `global_bg_rank` (0 vs 1), `init_stride`.
Pass comma-lists, e.g. `--grid-min-pnr 6,10,14 --grid-bg-rank 0,1`.

**Temporal / evaluation** (`decay_time_ms`, `g_prior_weight`, `merge_thr_corr`,
`auto_eval_snr_amp_thr`) — from the best swept model (or `--existing-results`).

---

## Reading the figures

- **Stage 1** — temporal-std with detected blobs (neuron radius); shift
  histograms (`max_shift`); ssub/tsub rule tables.
- **Stage 3** — CORR / PNR / CORR·PNR triptych (`sigma`); morphology curves +
  thresholded CORR/PNR images at the detected `(min_corr, min_pnr)`; footprint-area
  histogram (`min_pixel`).
- **Stage 4** — per-component τ and AR-`g` distributions (`decay_time_ms`,
  `g_prior_weight`); pairwise-correlation histogram (`merge_thr_corr`); SNR
  histogram + footprint montage at the threshold boundary (`auto_eval_snr_amp_thr`).
- **Sweep** — the **density↔purity scatter** (K vs median `corr(C, C+YrA)`,
  point size = accepted fraction, colour = score, best starred); footprints over
  the **correlation image thresholded at that candidate's `min_corr`** (sub-threshold
  background dropped to black so each footprint reads against the cell it covers; the
  mean projection is a vignette-dominated blob on 1p data — see [Tuning long or dense recordings](../getting-started/index.md#tuning-long-or-dense-recordings)); `C` vs `C+YrA` traces for the best candidate.

### Component diagnostics (`tuning.report`)

Packaged from the old `diagnostics.ipynb` / `cutout_analysis.ipynb` cells, these
take a fitted `model` (and import cleanly into a notebook —
`from tuning.report import fig_footprint_grid, …`; a registry is in
`report.DIAGNOSTIC_FIGS`):

- `fig_footprint_grid(model)` — top-N footprints cropped around their
  `footprint_center` (soma centred), titled `k n e` (id / npix / eccentricity),
  red title = auto-eval rejected. **The first thing to look at.**
- `fig_eccentricity(model)` — area / peak / eccentricity histograms; area ≫
  `π(2σ)²` = halo or multi-cell; eccentricity → 1 = elongated/vasculature.
- `fig_jaccard_merge(model)` — K×K spatial-Jaccard + trace-correlation matrices
  with the "should-have-merged" pair count (both above threshold).
- `fig_centroid_drift(model, cn, pnr=…)` — argmax (`footprint_center`, what the
  algorithm uses) vs binary-mask COM centres on CORR/PNR, quantifying COM drift.
- `fig_mean_proj_and_activity(movie)` — streamed mean projection + per-frame
  activity trace (drift / photobleaching / lockstep firing at a glance).

---

## The quality metrics

These are **ground-truth-free proxies**, not validation — there is no ground
truth on a real recording.

| Metric | Meaning | Direction |
|--------|---------|-----------|
| `K` / `K_accepted` / `accepted_frac` | components extracted / passing auto-eval | context |
| `cprojcorr_median` | median per-cell `corr(C, C+YrA)` — **primary purity signal** | higher = purer; *falls* as K rises in dense FOVs (YrA cross-talk) |
| `npix_median` / `npix_iqr` | footprint pixel-count distribution | tight, consistent = good |
| `snr_median` | per-component mean-amplitude SNR | higher = stronger |
| `composite_score` | `cprojcorr_median + 0.5·accepted_frac − 0.25·(npix_iqr/npix_median)` | ranking only |

> [!CAVEAT]
> Don't chase cell count with thresholds — there is a density↔purity sweet spot,
> and `cprojcorr_median` is itself a good knob for picking it. The `composite_score`
> is transparent and re-derivable from the table; re-rank with your own weights if
> you prefer.

These proxies are the seam a future validation harness could reuse for true
cross-method / paired-ephys validation.

---

## Gotcha checklist

**This is the single source of truth for interpreting a tuning run.** Apply every
item when reading a `comparison.md` / `report.html`:

- **Long recording → `global_bg_rank=1`.** On a long movie slow drift /
  photobleaching makes traces collinear and `update_spatial` smears footprints
  into merged blobs. Confirm rank-1 background wins in the sweep and footprints
  stay compact (tight `npix` IQR). `validate_session`'s `good_defaults` set it.
- **`decay_time_ms` from the data is drift-inflated.** Yule-Walker `g` is biased
  upward by slow background on long recordings, so the auto decay estimate (e.g.
  ~560 ms) is NOT the indicator. Set the physical τ from the indicator and
  `g_prior_weight ≈ 0.6`; a large auto estimate is *evidence of drift*, not a slow
  indicator.
- **`min_pixel` is a floor; SNR is the ghost discriminator.** Rejections in the
  auto-eval line should come from `snr_amp`, not `pixel_count`. If `min_pixel` is
  doing the rejecting (especially after downsampling), lower it (~60 native) and
  let `auto_eval_snr_amp_thr` (~20) cut ghosts.
- **Full-vs-cutout recall.** Thresholds tuned on a short cutout over-prune the
  full recording (PNR falls over long T). Judge recall on `footprints_on_corr.png`
  against the correlation image — count unclaimed bright blobs — **not** on raw
  `K`. Expect the lower-threshold run to recover more cells at a modest purity cost.
- **Density ↔ purity is a real tradeoff.** Report it in numbers (K / accepted vs
  `cprojcorr_median`) and recommend an operating point rather than "more is
  better"; `cprojcorr_median` *falls* as K rises in a dense FOV (YrA cross-talk).
- **Hazy / out-of-focus FOV → over-estimated neuron radius.** On a hazy or
  out-of-focus recording `blob_log` on the temporal-std latches onto broad
  background structure and reports a too-large radius (e.g. 6.3 px when the
  neurons are ~4 px), which cascades into a too-large `sigma`, an over-aggressive
  `ssub`, a huge `min_pixel`, and **sprawling footprints**. `suggest_mc_gsig_and_sigma`
  now applies a spatial high-pass (`highpass_sigma=8` px, default on) before
  `blob_log` to strip the haze, so this is handled automatically. **Tell-tale:**
  `sigma` / `ssub` / `min_pixel` come out ~2× a comparable in-focus session — check
  `fig_mc_gsig.png`; the detected-radius histogram should sit at the neuron scale,
  not be dragged up by a tail of large blobs. Bump `highpass_sigma` further only if
  a very hazy session still over-estimates.
- **Motion correction.** From `mc_shifts.png`: small shifts (≲ a few px) → one MC
  pass is fine; large/erratic shifts warrant a bigger `max_shift` or `mc_n_iter=2`.

### Symptom → cause → knob (by-eye)

The full by-eye troubleshooting table (multi-peak / amoeba / donut / crescent /
streaky / fragmented / ghosts / lockstep / square-wave traces / …) lives in code
as `tuning.report.SYMPTOM_CAUSE_KNOB` and is embedded in every `report.html`. The
most common few:

| Symptom | Knob |
|---------|------|
| Too many ghost components | raise `min_pnr` / `auto_eval_snr_amp_thr` |
| Too few neurons | lower `min_corr` / `min_pnr`; pin `init_stride` to 1–2 on long movies |
| Footprints sprawl / merge into blobs (long recording) | `global_bg_rank=1`, `spatial_max_thr↑`, `spatial_circular_max_dist_factor↓` |
| Footprints sprawl + `sigma`·`ssub`·`min_pixel` ~2× a comparable session (hazy/out-of-focus FOV) | radius over-estimated by `blob_log` on haze — auto-fixed by the `highpass_sigma` pre-filter; check `fig_mc_gsig.png`, raise `highpass_sigma` if it persists |
| Distinct neighbours fused | raise `merge_thr_corr`, lower `merge_centre_dist_factor` |
| "Shark-fin" / over-smoothed traces | set `decay_time_ms` + `frame_rate_hz` (Bayesian `g` prior) |

See [Parameter Tuning Guide](../getting-started/index.md#parameter-tuning-guide) for the full per-knob discussion, and
the **Component diagnostics** figures above to identify which symptom you have.

---

## Full-recording validation (`validate_session.py`)

The tuner sweeps on a cutout; to confirm the params on the **whole** recording —
where drift accumulates and PNR falls — use the validation CLI. It fuses motion
correction once, transposes the pixel-major `Y_flat` once, then runs
`fit_extract` for **each** `min_corr:min_pnr` set you ask for, **reusing** the
(threshold-independent) `Y_flat` so each extra candidate skips the expensive MC +
transpose. It writes per-run results, diagnostic figures, and a `comparison.md`.

```bash
python validate_session.py /path/to/miniscope_video -o out/ \
    --indicator gcamp8m --thresholds "0.8:10,0.7:6"
```

It bakes in the long-real-recording defaults learned on a real session
(`global_bg_rank=1`, low `min_pixel` floor with the SNR check doing ghost
rejection, the physical-decay prior, pinned `init_stride`) via
`tuning.validate.good_defaults`. Frame rate and dims are auto-read from the
session's `metaData.json`. `tune.py` already runs this stage for you (it calls
`tuning.validate.tune_then_validate`); use `validate_session.py` directly only to
re-validate an existing recommendation or to add threshold sets.

> [!CAVEAT]
> See the [Gotcha checklist](#gotcha-checklist) above — in particular the drift-inflated
> `decay_time_ms` and the full-vs-cutout recall caveats — when reading the
> comparison.

## Batch — many sessions in one background process

For several sessions, point the front door at a list:

```bash
python tune.py --sessions sessions.txt -o runs/batch \
    --indicator gcamp8m --jobs 2 --cores 6 &
```

This delegates to `batch_tune.run_batch`, which runs **one `tune.py --validate`
subprocess per session** under bounded concurrency (`--jobs`), each pinned to
`--cores` and BLAS-capped so `jobs·cores` threads don't oversubscribe (keep
`jobs·cores ≤ nproc−2`). Each
session is tuned **independently** (its own measured `sigma`/`ssub`/thresholds —
assume different animals/scopes). It writes per-session folders + a
`batch_summary.md`. `tuning.validate.resolve_session_paths` parses the list
(multiple args or a `.txt`; dedups, skips missing). `batch_tune.py` is the same
thing as a standalone CLI.

## Relation to other docs

- [usage guide](../getting-started/index.md) — what each parameter means and the long/dense-recording findings.
- [API reference](../api/index.md) — `CNMFeParams` fields and method signatures.
- [architecture](../concepts/architecture.md) — where the staged CLIs (`run_mc`/`run_extract`) fit.
