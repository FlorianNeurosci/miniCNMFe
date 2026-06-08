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


def build_candidates(
    base_params: CNMFeParams, spec: SweepSpec, max_candidates: int = 24,
) -> "list[tuple[CNMFeParams, dict]]":
    """Expand ``spec`` into ``(params, swept_values)`` candidates.

    If the full Cartesian product fits within ``max_candidates``, use it.
    Otherwise fall back to a **one-knob-at-a-time** design around the base
    params (linear in the number of values), so cost stays bounded. The base
    params themselves are always included as the first candidate.
    """
    active = spec.active()
    if not active:
        snap = {f: getattr(base_params, f) for f in SWEPT_FIELDS}
        return [(replace(base_params), snap)]

    fields_ = list(active)
    grid_size = int(np.prod([len(active[f]) for f in fields_]))

    combos: "list[dict]" = []
    if grid_size <= max_candidates:
        for values in itertools.product(*(active[f] for f in fields_)):
            combos.append(dict(zip(fields_, values)))
    else:
        combos.append({})  # base
        for f in fields_:
            base_v = getattr(base_params, f)
            for v in active[f]:
                if v != base_v:
                    combos.append({f: v})

    candidates = []
    for combo in combos:
        p = replace(base_params, **combo)
        snap = {f: getattr(p, f) for f in SWEPT_FIELDS}
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
    max_candidates: int = 24, cn=None,
) -> "tuple[list[dict], CNMFeParams, CNMFe]":
    """Run the sweep and return ``(rows, best_params, best_model)``.

    ``rows`` are sorted by ``composite_score`` (best first). ``best_model`` is
    the best candidate **re-fit once in the parent** on the same region (so the
    stage-4 heuristics have a live model to read).

    ``cn`` is the FOV correlation image; when given, each candidate also gets a
    footprints + traces PNG written to the tuner output dir (``workdir.parent``)
    as ``fig_cand_<i>_footprints.png`` / ``fig_cand_<i>_traces.png``, and the
    basenames are recorded in the candidate row.
    """
    from joblib import Parallel, delayed

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    candidates = build_candidates(base_params, spec, max_candidates)

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
    rows = Parallel(n_jobs=n_jobs)(delayed(_run_one_candidate)(t) for t in tasks)
    rows = sorted(rows, key=lambda r: r.get("score", float("-inf")), reverse=True)

    best_idx = rows[0]["idx"]
    best_params = candidates[best_idx][0]
    best_model = _fit_candidate(best_params, str(mc_zarr_path), region_crop,
                                str(workdir / "best"))
    return rows, best_params, best_model
