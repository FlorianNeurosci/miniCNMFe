# Parameter tuning

The `tuning/` package + the `tune.py` front door pick good CNMF-E parameters for a
recording you've never seen, and produce a report you can judge by eye. As with
the [algorithm guides](../guides/index.md), every page here is written from the
source; comment/code mismatches are recorded in
`todo/doc_comment_code_mismatches.md`.

The central honesty caveat (stated in `tuning/metrics.py`): on a real recording
there is **no ground truth**, so the quality numbers are **proxies, not
validation** — they rank candidates against each other and let you eyeball
quality, nothing more.

## The front door

```bash
python tune.py /path/to/avis --indicator gcamp8m --n-jobs -1 &
# then open  runs/tune_<name>_<ts>/report.html
```

`tune.py` accepts an **AVI folder** or an already-motion-corrected **`mc.zarr`**.
It resolves the frame rate (flag → `metaData.json` → 20 Hz default) and the
indicator decay τ (`--indicator` shortcut → `--decay-time-ms`), builds a
`TunerConfig`, and calls `tuning.validate.tune_then_validate`. Batch mode
(`--sessions list.txt`) delegates to `batch_tune` (one BLAS-capped background
process). The `/tune-session` skill wraps the same entry point.

## The workflow (`tuning.tuner.run_tuning`)

1. **Sampling** (`tuning/io_sample.py`) — cheap slices into RAM: a strided AVI
   sample for the MC heuristics, a chunk-aligned `mc.zarr` sample for the
   extraction heuristics, plus `pick_cutout` (a representative
   spatial+temporal window for the fast sweep).
2. **[Per-knob heuristics](heuristics.md)** (`tuning/heuristics.py`) — image-based
   suggestions for each parameter (neuron radius, `max_shift`, `ssub`/`tsub`,
   `sigma`, `min_corr`/`min_pnr`, `min_pixel`, and the temporal knobs). Each
   `suggest_*` returns `(value, evidence)`; the evidence feeds the report
   figures.
3. **[Extraction sweep](sweep.md)** (`tuning/sweep.py`) — actually run
   `fit_extract` across a small grid of the impactful knobs, scoring each
   candidate with the [GT-free proxies](metrics.md). Runs on a cutout (fast) or
   the full recording (faithful).
4. **Temporal/merge/eval heuristics** — read off the best fitted model:
   `decay_time_ms`, `g_prior_weight`, `merge_thr_corr`, `auto_eval_snr_amp_thr`.
5. **Report** — `recommended_params.json` (in **native** units),
   `downsample.json`, `report.md`, and a self-contained `report.html`.
6. **[Full-recording validation](validation.md)** (`tuning/validate.py`,
   default on for AVI input) — re-extract on the whole recording at the
   recommended (and a lower-recall) threshold set, with diagnostic figures and a
   `comparison.md` table.

Motion-correction parameters for a raw AVI session are searched separately by the
crispness-validated [MC search](mc-search.md) (`tuning/mc_tune.py` /
`mc_search.py`); the extraction tuner only tunes extraction on the corrected
movie.

## Units bookkeeping (the subtle part)

The sweep and heuristics run on the `mc.zarr` **grid** (possibly downsampled by
`ssub`/`tsub`), but `recommended_params.json` is written in **native** units so it
feeds `run_extract.py --ds-meta` directly: `sigma·ssub`, `min_pixel·ssub²`,
thresholds unchanged, frame rate `/tsub`. `ssub`/`tsub` are carried separately in
`downsample.json`. `CNMFeParams.downscaled(ssub, tsub)` does the native→grid
rescale at run time.

## The "good defaults" base

`tuning.validate.good_defaults` is the long-real-recording starting point the
tuner builds on (rationale in `live_runs/tuning_picast/LEARNINGS.md`):
`global_bg_rank=1` (absorb slow drift), `min_pixel=60` as a floor with the SNR
check doing ghost rejection, the physical-decay `g` prior, pinned `init_stride=2`,
`n_iter_main=2`. The tuner overwrites only the fields it has data-driven values
for, so the recommendation **contains** these wins and validation runs it
verbatim.
