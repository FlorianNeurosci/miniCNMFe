"""Extraction parameter sweep.

Runs ``CNMFe.fit_extract`` across a small grid of the most-impactful extraction
knobs (per CLAUDE.md: ``sigma``, ``min_corr``, ``min_pnr``, ``merge_thr_corr``,
``global_bg_rank``, ``init_stride``) and scores each candidate with the
ground-truth-free proxies in ``tuning.metrics``. Candidates run in parallel
processes via joblib; the worker ``_run_one_candidate`` is module-level for
spawn-pickling.

Two regions (set by ``region_crop``):

- **cutout** — a ``(spatial_crop, temporal_crop)`` tuple; each candidate runs on
  the cropped movie held in RAM (fast; seconds-to-minutes per candidate).
- **full** — ``None``; each candidate streams the whole ``mc.zarr`` (faithful,
  slow) via the pixel-major ``Y_flat`` path under ``workdir``.
"""

from __future__ import annotations

import itertools
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from minicnmfe.io import open_zarr
from minicnmfe.pipeline import CNMFe, CNMFeParams
from tuning.metrics import composite_score, model_quality

# Swept fields, in display order. Each maps to a SweepSpec attribute.
SWEPT_FIELDS = ("sigma", "min_corr", "min_pnr", "merge_thr_corr",
                "global_bg_rank", "init_stride")


@dataclass
class SweepSpec:
    """Per-knob value lists for the sweep. ``None`` = hold at the base value."""

    sigma: "list[float] | None" = None
    min_corr: "list[float] | None" = None
    min_pnr: "list[float] | None" = None
    merge_thr_corr: "list[float] | None" = None
    global_bg_rank: "list[int] | None" = None
    init_stride: "list[int] | None" = None

    def active(self) -> "dict[str, list]":
        """Mapping of {field: values} for fields that were given a list."""
        out = {}
        for f in SWEPT_FIELDS:
            v = getattr(self, f)
            if v:
                out[f] = list(v)
        return out


def resolve_offset_grid(
    spec, anchor: float, *, floor: float, default_around=(0,),
    round_anchor: bool = False, clip_max: "float | None" = None,
) -> "list[float]":
    """Resolve a heuristic-relative sweep spec into concrete grid values ("Offset DSL").

    A data-driven ``anchor`` (sigma radius, or a detected ``min_corr``/``min_pnr``) is
    only known at sweep runtime, so a stored spec expresses the grid *relative* to it.
    Resolving here guarantees the anchor is always one of the tested candidates — the
    failure mode it fixes is a static grid (e.g. sigma 3,4,5 or pnr 6,10,14) that sits
    away from the data-driven value, which then can never be a candidate.

    Accepted ``spec`` forms:
    - ``None`` — ``{anchor + o for o in default_around}``.
    - ``{"around": [...], "extra": [...]}`` — offsets around the anchor plus absolute
      ``extra`` values (either key optional; ``around`` defaults to ``[0]`` so the
      anchor is always present).
    - ``list``/``tuple`` — back-compat absolute values, with the anchor injected.
    - scalar — a single absolute value, anchor injected.

    The anchor is optionally ``round``-ed (``round_anchor``), all values are clamped to
    ``[floor, clip_max]``, deduped and sorted.
    """
    base = float(anchor)
    if round_anchor:
        base = float(int(round(base)))
    base = max(floor, base)
    if spec is None:
        around, extra = list(default_around), []
    elif isinstance(spec, dict):
        around = list(spec.get("around", [0]))
        extra = list(spec.get("extra", []))
    elif isinstance(spec, (list, tuple)):
        around, extra = [0], list(spec)
    else:
        around, extra = [0], [spec]
    vals = {base + float(o) for o in around} | {float(v) for v in extra}
    out = set()
    for v in vals:
        v = max(floor, v)
        if clip_max is not None:
            v = min(clip_max, v)
        out.add(v)
    return sorted(out)


def resolve_sigma_grid(spec_sigma, sigma_ds: float) -> "list[float]":
    """Sigma grid via the Offset DSL, anchored on the ``blob_log`` radius ``sigma_ds``.

    Omitted -> heuristic-centred ``{s-1, s, s+1}``; floored at 2. Thin wrapper over
    :func:`resolve_offset_grid` (see it for the full spec form)."""
    return resolve_offset_grid(spec_sigma, sigma_ds, floor=2.0,
                               default_around=(-1, 0, 1), round_anchor=True)


def build_candidates(
    base_params: CNMFeParams, spec: SweepSpec, max_candidates: int = 24,
    thr_seeds: "list[dict] | None" = None,
) -> "list[tuple[CNMFeParams, dict]]":
    """Expand ``spec`` (and optional ``thr_seeds``) into ``(params, swept)`` candidates.

    The non-seed knobs in ``spec`` form a base set of combos: the full Cartesian
    product if it fits the budget, else a **one-knob-at-a-time** design around the
    base params (linear in the number of values), so cost stays bounded.

    ``thr_seeds`` is an optional list of **coupled** ``{"min_corr", "min_pnr",
    "thr_method"}`` dicts — each a candidate operating point from a different
    threshold-selection method. When given, candidates are
    ``thr_seeds × base_combos`` (each seed sets ``min_corr`` *and* ``min_pnr``
    **together**, not as an independent-grid cross-product), so the sweep can
    score the methods against each other. ``thr_method`` is carried into the
    swept snapshot. The per-seed budget is ``max_candidates // len(thr_seeds)``.
    """
    active = spec.active()
    seeds = list(thr_seeds) if thr_seeds else [None]
    per_seed_budget = max(1, max_candidates // len(seeds))

    if not active:
        base_combos: "list[dict]" = [{}]
    else:
        fields_ = list(active)
        grid_size = int(np.prod([len(active[f]) for f in fields_]))
        base_combos = []
        if grid_size <= per_seed_budget:
            for values in itertools.product(*(active[f] for f in fields_)):
                base_combos.append(dict(zip(fields_, values)))
        else:
            base_combos.append({})  # base
            for f in fields_:
                base_v = getattr(base_params, f)
                for v in active[f]:
                    if v != base_v:
                        base_combos.append({f: v})

    candidates = []
    for seed in seeds:
        for combo in base_combos:
            full = dict(combo)
            label = None
            if seed is not None:
                full["min_corr"] = float(seed["min_corr"])
                full["min_pnr"] = float(seed["min_pnr"])
                label = seed.get("thr_method")
            p = replace(base_params, **full)
            snap = {f: getattr(p, f) for f in SWEPT_FIELDS}
            if label is not None:
                snap["thr_method"] = label
            candidates.append((p, snap))
    return candidates


def _fit_candidate(params, mc_zarr_path, region_crop, workdir):
    """Fit one candidate; return the fitted model (with auto-eval)."""
    mc = open_zarr(mc_zarr_path)
    model = CNMFe(params)
    if region_crop is not None:
        (y0, y1, x0, x1), (t0, t1) = region_crop
        arr = np.asarray(mc[t0:t1, y0:y1, x0:x1], dtype=np.float32)
        model.fit_extract(arr, evaluate=True)
    else:
        Path(workdir).mkdir(parents=True, exist_ok=True)
        model.fit_extract(mc, output_dir=Path(workdir), evaluate=True)
    return model


def _render_candidate_figs(model, cn, fp_out, tr_out):
    """Render this candidate's footprints + traces PNGs (best-effort).

    Reuses the same renderers as the best-candidate report figures. The model and
    the (already cutout-sliced) ``cn`` are both in crop-local coords, so no offset
    (``region_crop=None``). A plotting failure must NOT fail the candidate, so all
    errors are swallowed and simply leave the fig fields unset.
    """
    import matplotlib
    matplotlib.use("Agg")
    from tuning.report import fig_sweep_footprints, fig_sweep_traces

    out = {}
    try:
        if model.A is not None and model.A.shape[1] > 0:
            Path(fp_out).parent.mkdir(parents=True, exist_ok=True)
            fig_sweep_footprints(model, cn, out_path=fp_out, region_crop=None)
            out["footprints_fig"] = Path(fp_out).name
            fig_sweep_traces(model, out_path=tr_out)
            out["traces_fig"] = Path(tr_out).name
    except Exception:  # noqa: BLE001 — figures are optional, never abort
        pass
    return out


def _run_one_candidate(args) -> dict:
    """Module-level sweep worker (spawn-pickling). Returns a metrics row.

    A failed candidate returns a row with ``error`` set and a ``-inf`` score
    rather than aborting the whole sweep.
    """
    idx, params, mc_zarr_path, region_crop, workdir, swept, cn_fig, fp_out, tr_out = args
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        threadpool_limits = None

    t0 = time.time()
    try:
        if threadpool_limits is not None:
            with threadpool_limits(limits=1, user_api="blas"):
                model = _fit_candidate(params, mc_zarr_path, region_crop, workdir)
        else:
            model = _fit_candidate(params, mc_zarr_path, region_crop, workdir)
        q = model_quality(model)
        row = {"idx": idx, "error": None, "wall_s": time.time() - t0}
        row.update(swept)
        row.update(q)
        row["score"] = composite_score(q)
        row.update(_render_candidate_figs(model, cn_fig, fp_out, tr_out))
        return row
    except Exception as exc:  # noqa: BLE001 — surface, don't abort the batch
        row = {"idx": idx, "error": repr(exc), "wall_s": time.time() - t0,
               "score": float("-inf")}
        row.update(swept)
        return row


def run_sweep(
    mc_zarr_path: "str | Path", base_params: CNMFeParams, spec: SweepSpec,
    *, region_crop=None, n_jobs: int = 1, workdir: "str | Path",
    max_candidates: int = 24, cn=None, thr_seeds: "list[dict] | None" = None,
) -> "tuple[list[dict], CNMFeParams, CNMFe]":
    """Run the sweep and return ``(rows, best_params, best_model)``.

    ``rows`` are sorted by ``composite_score`` (best first). ``best_model`` is
    the best candidate **re-fit once in the parent** on the same region (so the
    stage-4 heuristics have a live model to read).

    ``n_jobs`` is the **total core budget**, split internally between candidate-level
    concurrency (loky processes) and each candidate's inner ``fit_extract`` threads
    so the product never exceeds the budget (avoids N*N thread oversubscription).

    ``cn`` is the FOV correlation image; when given, each candidate also gets a
    footprints + traces PNG written to the tuner output dir (``workdir.parent``)
    as ``fig_cand_<i>_footprints.png`` / ``fig_cand_<i>_traces.png``, and the
    basenames are recorded in the candidate row.
    """
    from joblib import Parallel, delayed

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    candidates = build_candidates(base_params, spec, max_candidates, thr_seeds=thr_seeds)

    # Print the candidate grid up front (in the parent, before dispatch) so each
    # sweep logs exactly which parameter combinations it is about to extract.
    region_name = "full" if region_crop is None else "cutout"
    print(f"[sweep] {len(candidates)} candidate(s) on {region_name} region:", flush=True)
    for i, (_p, swept) in enumerate(candidates):
        vals = ", ".join(
            f"{f}={swept[f]:g}" if isinstance(swept.get(f), float) else f"{f}={swept[f]}"
            for f in SWEPT_FIELDS if swept.get(f) is not None
        )
        if swept.get("thr_method"):
            vals = f"[{swept['thr_method']}] " + vals
        print(f"[sweep]   cand {i}: {vals}", flush=True)

    # Split the `n_jobs` core budget across the two parallelism levels so their
    # product ~= budget (no N*N thread oversubscription). Candidates run as loky
    # *processes* (cand_jobs of them); each fit_extract then runs cand-internal
    # *threads* (inner_jobs). BLAS is pinned to 1 per process in _run_one_candidate,
    # so total OS threads ~= cand_jobs * inner_jobs ~= budget. With the default
    # few-candidate sweep this puts the surplus cores into each fit (e.g. 3 cands x
    # ~10 inner on a 32-core box); with a large grid it goes candidate-parallel,
    # inner-serial. Candidate level must stay processes: with init_patches=False the
    # per-candidate greedy seed loop is pure-Python/GIL-bound, so threads wouldn't help.
    #
    # The split arithmetic (min / //) needs a concrete core count, so resolve
    # joblib's negative sentinels here: -1 -> all cores, -2 -> all but one, etc.
    # (a bare `max(1, n_jobs)` would turn -1 into 1 = fully serial).
    if n_jobs < 0:
        budget = max(1, (os.cpu_count() or 1) + 1 + n_jobs)
    else:
        budget = max(1, n_jobs)
    cand_jobs = min(len(candidates), budget)
    inner_jobs = max(1, budget // cand_jobs)

    # Patch-parallel greedy init (loky processes) is the only way to parallelise
    # the otherwise-serial greedy seed loop — the dominant cost on long/large
    # fits. It must NOT nest under candidate-level loky, so enable it only when
    # there's no candidate-level fan-out (cand_jobs == 1: candidates run inline in
    # the parent). Then a candidate's init uses the full inner_jobs across patches.
    # With cand_jobs > 1, init parallelism comes from candidates running
    # concurrently instead, and per-candidate patches stay off (tuner set
    # init_patches=False on the base).
    cand_init_patches = (cand_jobs == 1 and inner_jobs > 1)
    candidates = [(replace(params, n_jobs=inner_jobs,
                           init_patches=cand_init_patches), swept)
                  for params, swept in candidates]

    # candidate figures use a crop-local correlation image; slice the FOV cn to
    # the cutout so contours land on the right pixels (region_crop=None at render).
    if cn is not None and region_crop is not None:
        (y0, y1, x0, x1), _t = region_crop
        cn_fig = cn[y0:y1, x0:x1]
    else:
        cn_fig = cn
    # figs go at the top level of the tuner output dir so the DB durable-copy
    # mirrors them into 2_processed alongside the other report figures.
    fig_dir = workdir.parent

    tasks = [
        (i, params, str(mc_zarr_path), region_crop, str(workdir / f"cand_{i}"),
         swept, cn_fig, str(fig_dir / f"fig_cand_{i}_footprints.png"),
         str(fig_dir / f"fig_cand_{i}_traces.png"))
        for i, (params, swept) in enumerate(candidates)
    ]
    rows = Parallel(n_jobs=cand_jobs)(delayed(_run_one_candidate)(t) for t in tasks)
    rows = sorted(rows, key=lambda r: r.get("score", float("-inf")), reverse=True)

    best_idx = rows[0]["idx"]
    # The best-model re-fit runs alone in the parent (no candidate-level fan-out),
    # so give it the full core budget and patch-parallel init (top-level loky,
    # nothing to nest under).
    best_params = replace(candidates[best_idx][0], n_jobs=budget,
                          init_patches=(budget > 1))
    best_model = _fit_candidate(best_params, str(mc_zarr_path), region_crop,
                                str(workdir / "best"))
    return rows, best_params, best_model
