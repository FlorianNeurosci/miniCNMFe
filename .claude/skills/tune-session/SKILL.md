---
name: tune-session
description: Check and recommend CNMFe parameters for one or many miniscope sessions. Use when the user gives a recording path (an AVI folder like .../miniscope_video, or an mc.zarr), several paths, or a .txt listing paths, and wants suggested motion-correction + extraction parameters validated by full-recording extraction with diagnostic figures and a written verdict. One session runs inline; multiple sessions fan out to parallel agents. Invoke as /tune-session <path> [more-paths...] | /tune-session <list.txt>.
argument-hint: <path...> | <list.txt> [--quick | --no-lowthr] [--figs | --no-figs] [--indicator <name>] [--jobs F] [--cores NJ]
allowed-tools: [Bash, Read, Edit, Write, Glob, Grep]
---

# tune-session — parameter check for CNMFe recording(s)

You are checking CNMFe parameters for the session(s) the user gave you, and
delivering a verdict they can act on. This reproduces the methodology in
`live_runs/tuning_picast/LEARNINGS.md` (read it once if you haven't this session).

The argument is **one or more session paths**, or a **`.txt`** listing one path
per line. A session path is an AVI folder (`0.avi … N.avi`, usually a
`.../miniscope_video` directory) or an existing `mc.zarr`. Optional flags:
`--quick` (tuner + feedback only, no full runs), `--no-lowthr` (single full run,
skip the lower-threshold comparison), `--figs`/`--no-figs` (view diagnostic PNGs
at the end — see *Reviewing figures*; default **on** for ≤2 sessions, **off**
for larger batches), `--indicator <name>` (e.g. `gcamp6f`), `--jobs F` (sessions
run concurrently in batch mode), `--cores NJ` (cores per session).

## Reviewing figures (token cost) — `--figs` / `--no-figs`

Viewing PNGs with the Read tool is by far the most token-expensive thing this
skill does (an image costs far more than the numeric `comparison.md`). So:
**defer all figure-viewing to ONE curated review at the very end**, and only
view the few that matter — per session: `full/run_<best>/figs/footprints_on_corr.png`
(spatial quality / recall) and `tuning/tune_*/fig_sweep_scatter.png`
(density↔purity). Never view every figure of every run.

`--figs` (default for 1–2 sessions) = do that final curated review and fold the
visual judgement into the verdict. `--no-figs` (default for larger batches) =
judge from the numeric outputs only; the PNGs are still written to disk for the
user to open. The user can always force either.

**First, resolve the input into a session list** and branch:
```bash
"$PY" -c "from tuning.validate import resolve_session_paths as r; import sys; print('\n'.join(map(str, r(sys.argv[1:]))))" <args...>
```
- **One session →** run the single-session procedure below inline.
- **Multiple sessions →** go to **Batch mode** (parallel agents). Do NOT process
  them one-by-one inline.

## Environment (do this right or runs hang/buffer)

- Resolve the project env's python **once** and use it directly (`mamba run`
  buffers stdout so you can't watch progress). Resolve it dynamically rather than
  hardcoding a path:
  ```bash
  PY=$(mamba run -n claude_cnmfe which python 2>/dev/null \
       || conda run -n claude_cnmfe which python 2>/dev/null)
  [ -x "$PY" ] || { echo "claude_cnmfe env python not found"; exit 1; }
  ```
  Then invoke scripts as `"$PY" -u <script> ...`. (`which python` is a quick
  one-shot, so the buffering caveat doesn't bite here — it only matters for the
  long-running scripts, which is why you run *those* via `"$PY"` directly.)
- Long stages (MC, extraction) take minutes to tens of minutes. Launch them with
  `nohup "$PY" -u <script> ... > <log> 2>&1 &` and **poll the log** with a bounded
  loop (`grep -q "ALL DONE" || pgrep -f <script>`), not a bare `sleep`.
- Write all outputs to **local disk** under `live_runs/tuning_<name>/` — session
  AVIs are typically on a network mount; reading them is fine, writing scratch
  there is slow.
- Before a full run, check headroom: `free -g` and `df -h`. The greedy-init
  sample on a long movie is the RAM spike (~10 GB at ssub 3 / `init_stride` 2).
- Reuse `mc.zarr` **and** `Y_flat_pixel.zarr` across threshold variants — they are
  threshold-independent; `validate_session.py` already does this.

## Batch mode (multiple sessions → one background orchestrator, NOT sub-agents)

When the input resolves to more than one session, do **not** spawn one
sub-agent per session and do **not** loop the single-session Procedure inline —
both burn tokens (each agent re-reads this skill, polls logs hundreds of times,
and views figures). Instead use the lean orchestrator **`batch_tune.py`**, which
tunes + validates every session in **one background process**, so the hours of
compute cost **zero model tokens**. Each session is still tuned **independently**
(its own measured `sigma`/`ssub`/thresholds) — assume every session may be a
different animal and/or miniscope; the per-session recommended params will
legitimately differ, and that is the point.

### B1. Plan resources
`batch_tune.py --jobs F --cores NJ` runs `F` sessions concurrently, `NJ` cores
each, BLAS-capped internally. Keep `F·NJ ≤ nproc−2`. Default `--jobs 2 --cores 6`
on a 16-core box (the earlier `4×3` over-contended → ~3 h/session; `2×6` finishes
each faster). On a **network mount** the MC-decode phase is IO-bound, so a
smaller `F` can be faster overall.

### B2. Launch in the background, then go idle
```bash
"$PY" -u batch_tune.py <resolved paths...|list.txt> -o live_runs/tuning_batch \
   --indicator <ind> --jobs <F> --cores <NJ> [--no-lowthr] > live_runs/tuning_batch/batch.log 2>&1 &
```
Run this with the Bash tool's **background** option (or `nohup ... &`) so you are
notified on completion rather than polling. (Add `--dry-run` first to print the
resolved sessions + per-session command plan and sanity-check it cheaply.)
`batch_tune.py` writes per-session outputs under
`live_runs/tuning_batch/<name>/` and a numeric `batch_summary.md`. It sets
`decay_time_ms` from `--indicator` (NOT the drift-inflated data estimate) and
never reuses one session's params for another.

### B3. Aggregate + (optional) review
When it finishes, **Read `live_runs/tuning_batch/batch_summary.md`** (text —
cheap) and give the verdict: present the per-session recommended params side by
side (expect them to differ; note any natural *clusters* like "A,C share
sigma/ssub → likely one scope" without forcing a shared set), the
density↔purity comparison per session, and list any FAILED sessions with their
log path (don't silently drop them).
**Figures:** follow *Reviewing figures* — with `--figs` (default for ≤2
sessions) view only the curated pair per session
(`full/run_<best>/figs/footprints_on_corr.png` + `tuning/tune_*/fig_sweep_scatter.png`);
with `--no-figs` (default for larger batches) judge from the text only and tell
the user the PNGs are on disk to open. Write per-session `LEARNINGS.md` only when
reviewing figures (it needs the visual judgement); otherwise the numeric
`comparison.md` per session stands on its own.

---

## Procedure (single session — default = tune → full run → low-thr compare)

*(For ONE session, run inline. Batch mode uses `batch_tune.py` instead, which
runs `tune.py` + `validate_session.py` per session under the
resource caps in B2.)*

### 1. Resolve the session + read acquisition metadata
- Confirm the path exists and is an AVI folder (`ls *.avi | head`) or an `mc.zarr`.
- Read settings from the session folder, don't guess:
  `"$PY" -c "from tuning.validate import read_session_meta; print(read_session_meta('<path>'))"`
  → gives `fps` (from `metaData.json` `frameRate`), `dims` (from `ROI`), and a
  `fps_measured` cross-check from `timeStamps.csv`. Use the measured fps if it
  differs from the nominal.
- **Indicator:** `metaData.json` does NOT record the GCaMP variant. Default to
  **jGCaMP8m (τ=180 ms)** and **say so explicitly** in your reply, inviting the
  user to override with `--indicator` or a τ. The decay knob materially changes
  `decay_time_ms`/`g_prior_weight`, and the data's own estimate is unreliable
  (see the gotcha checklist).

### 2. Tune (heuristics + cutout sweep)
Run the tuner to a local output dir (`OUT=live_runs/tuning_<sessionname>`).
Pass `--no-validate` here so this stays the fast cutout-only pass — step 3 does
the full-recording validation separately (so you can background it and poll). (A
direct `tune.py <path>` with no flags would tune **and** validate in one call —
that's the simpler path when you are not splitting the phases.)
```
"$PY" -u tune.py <path> -o $OUT/tuning --frame-rate <fps> --indicator gcamp8m \
   --mode both --region cutout --no-validate --max-avis 6 --grid-min-corr 0.7,0.8 \
   --grid-min-pnr 6,10,14 --grid-bg-rank 0,1 --n-jobs -1
```
(adjust `--max-avis`/grids to taste; this is the bounded quick pass). Then **read
the report text** (`Read $OUT/tuning/tune_*/report.md`) and pull the recommended
`sigma`, `min_corr`, `min_pnr`, `global_bg_rank`, `max_shift`, and `ssub`/`tsub`
(from `downsample.json`). Do **not** view figures here — figure-viewing is
deferred to the single curated review in step 4 (gated by `--figs`).

**If `--quick`:** stop here, give the verdict (step 4) on the tuner output, and
tell the user how to launch the full validation.

### 3. Validate on the full recording (two thresholds)
Run `validate_session.py` — it fuses MC once, transposes `Y_flat` once, and
extracts at each threshold set reusing `Y_flat`. Default to the tuner's
recommended thresholds **plus a lower-recall set** (`min_pnr−4`, `min_corr−0.1`):
```
nohup "$PY" -u validate_session.py <path> -o $OUT/full \
   --indicator gcamp8m --ssub <ssub> --tsub <tsub> --sigma <native_sigma> \
   --thresholds "<rec_corr>:<rec_pnr>,<low_corr>:<low_pnr>" --n-jobs -1 \
   > $OUT/full/run.log 2>&1 &
```
(With `--no-lowthr`, pass a single threshold set. With `--reuse-mc <mc.zarr>` you
can re-extract without re-fusing.) Launch it with the Bash **background** option
(or `nohup`) so you're notified on completion instead of polling. When done,
**Read `$OUT/full/comparison.md`** (text). Defer figures to step 4.

### 4. Interpret — the gotcha checklist (apply every item)
**Figures (gated by `--figs`, default on for a single session):** view the
curated pair only — `$OUT/full/run_<best>/figs/footprints_on_corr.png` (recall /
spatial quality) and `$OUT/tuning/tune_*/fig_sweep_scatter.png` (density↔purity).
Look at `mc_shifts.png` too if MC looks suspect. With `--no-figs`, skip and judge
from `comparison.md` + the auto-eval log line.

Then **apply every item in the Gotcha checklist** — the single source of truth is
the "Gotcha checklist" section of `docs/tuning/guide.md` (long recording →
`global_bg_rank=1`; drift-inflated `decay_time_ms`; `min_pixel` floor vs SNR ghost
cut; full-vs-cutout recall; density↔purity; MC shifts). Do not re-derive it here —
read that section and apply each bullet to this session's numbers/figures.

### 5. Deliver
- Write a per-session `$OUT/LEARNINGS.md` mirroring
  `live_runs/tuning_picast/LEARNINGS.md`: recording characteristics, the
  recommended-params table with a verdict per knob, the full-vs-cutout /
  threshold comparison, and the recommended operating point.
- In your reply: state the indicator assumption, give the recommended parameter
  set (call out any override vs the tuner's raw recommendation and why), show the
  comparison table, reference the figures you looked at, and name the output
  folder. Be honest about proxy-vs-ground-truth and any under/over-seeding you saw.

## Notes
- The good-default overrides (rank-1 bg, low `min_pixel`, SNR cut, decay prior,
  pinned `init_stride`) live in `tuning.validate.good_defaults` — `validate_session.py`
  applies them, so you only pass thresholds + indicator + ssub/tsub.
- Entry points: **`tune.py` is the single front door** — `tune.py <path>` runs
  tune + full-recording validation + `report.html` in one call (default output
  `runs/`, gitignored); `tune.py --sessions <list>` runs the batch. It composes
  the internal stages `validate_session.py` (validate one session; use directly
  only to re-validate / add threshold sets) and `batch_tune.py` (the
  one-background-process batch — `tune.py --sessions` delegates to it). The PICAST
  `live_runs/tuning_picast/run_full*.py` are historical one-offs superseded by
  `validate_session.py`.
- Token discipline: figures are the dominant token cost — view them only in the
  single curated `--figs` review at the end, never per-stage or per-run, and
  never during a `--no-figs` / large-batch run. Compute (the CLIs) costs no model
  tokens, so prefer background CLIs + one text aggregation over interpreting
  sub-agents.
- This is a long, billable run by default. If the user seems to want a quick
  answer, offer `--quick` first.
