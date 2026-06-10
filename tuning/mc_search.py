"""Crispness-validated motion-correction parameter search.

The per-knob heuristics (``tuning.heuristics.suggest_mc_gsig_and_sigma`` /
``suggest_max_shift``) give a *seed* for the 1p high-pass ``mc_gSig_filt`` and
the ``max_shift`` search range, but nothing checks that the resulting rigid
registration is actually sharp. When ``mc_gSig_filt`` under-suppresses the slow
1p background, the phase-correlation latches onto background structure instead
of cells and the corrected movie jitters.

This module turns that single guess into a small **seed + trial-and-score**
search: run a handful of short rigid MCs on one representative clip over a grid
seeded from the heuristics, and keep the candidate whose corrected clip is
crispest (``tuning.metrics.mc_crispness``). Two guardrails reject pathological
picks — a candidate that **saturates** ``max_shift`` (the jitter signature: the
search hit its range limit) or that fails to **beat the raw clip** (MC made
things worse) is never selected over a clean one.

Pure numpy + the in-RAM path of ``minicnmfe.motion_correction_rigid``; no zarr,
no DB. ``tuning.mc_tune`` (and the DB ``MiniCnmfeMcTuning`` table) consume it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from minicnmfe.motion_correction import motion_correction_rigid
from tuning.metrics import (
    correlation_image, mc_crispness, mc_quality, mc_registration_quality,
)


@dataclass
class McSearchSpec:
    """Grid for the MC parameter search — **absolute** grid-pixel values.

    These are intentionally absolute, not multiples of the heuristic seed: the
    automatic neuron-radius (sigma) detection is unreliable, and seeding
    ``max_shift`` from it can collapse to a tiny value that *hard-constrains* the
    phase-correlation (it zeros cross-correlation beyond ``max_shift``), blinding
    the search to real motion. An absolute ladder always spans a regime where
    genuine drift is visible; the quality metric then arbitrates. The heuristic
    seed is still tried as one extra candidate when ``include_seed`` (hybrid).

    Candidates are ranked by ``score`` = mean local correlation image
    (``tuning.metrics.mc_registration_quality``), NOT mean-image crispness — on
    1p data the latter is dominated by the static background and rewards
    under-correction (see that module's docstring).

    ``gsig_values`` are high-pass sigmas (grid px); the useful 1p range is wide
    because the static background must be suppressed before the correlation can
    track cells (the optimum sits well above the neuron radius). ``max_shift_values``
    are per-axis caps (grid px); ``n_iter_values`` are rigid-pass counts.
    ``coarse_to_fine`` sweeps gSig at the most generous ``max_shift`` (so motion
    is visible while choosing the filter), fixes the best, then sweeps
    ``max_shift`` and ``n_iter``. ``saturation_frac`` is the fraction of
    ``max_shift`` the 99th-pct shift may reach before the candidate is flagged
    saturated (clipping).
    """

    gsig_values: list = field(default_factory=lambda: [3.0, 5.0, 7.0, 9.0, 12.0, 16.0])
    max_shift_values: list = field(default_factory=lambda: [4, 8, 16])
    n_iter_values: list = field(default_factory=lambda: [1, 2])
    coarse_to_fine: bool = True
    include_seed: bool = True
    saturation_frac: float = 0.95
    upsample_factor: int = 10


def _trial(clip, *, gsig, max_shift, n_iter, upsample, raw_score, sat_frac, n_jobs):
    """Run one short rigid MC on ``clip`` and score it. Never raises.

    ``score`` (the ranking key) is the mean local correlation image of the
    corrected clip — higher = better cell co-registration. ``crispness_mean`` is
    recorded for diagnostics only.
    """
    ms = (int(max_shift[0]), int(max_shift[1]))
    row = {"mc_gSig_filt": float(gsig), "max_shift": ms, "mc_n_iter": int(n_iter)}
    try:
        corrected, shifts = motion_correction_rigid(
            np.asarray(clip, dtype=np.float32), output_path=None,
            max_shift=ms, gSig_filt=float(gsig), upsample_factor=int(upsample),
            niter_rig=int(n_iter), n_jobs=n_jobs, verbose=False)
        reg = mc_registration_quality(corrected)
        cr = mc_crispness(corrected)
        q = mc_quality(shifts)
        ms_lim = float(min(ms))
        saturated = bool(q["shift_p99"] >= sat_frac * ms_lim) if ms_lim > 0 else False
        # "beats raw" = corrected cells at least as well co-registered as raw
        # (small tolerance so a genuinely-still movie's do-nothing candidate
        # still counts as acceptable, not "MC failed").
        beats_raw = bool(reg["corr_mean"] >= raw_score * (1.0 - 1e-3))
        row.update(score=reg["corr_mean"], corr_mean=reg["corr_mean"],
                   corr_p99=reg["corr_p99"], std_crispness=reg["std_crispness"],
                   crispness_mean=cr["crispness_mean"],
                   shift_smoothness=q["shift_smoothness"], shift_p99=q["shift_p99"],
                   shift_max=q["shift_max"], saturated=saturated,
                   beats_raw=beats_raw, error=None)
    except Exception as e:  # a bad param combo must not kill the whole search
        row.update(score=float("-inf"), corr_mean=float("nan"), corr_p99=float("nan"),
                   std_crispness=float("nan"), crispness_mean=float("nan"),
                   shift_smoothness=float("nan"), shift_p99=float("nan"),
                   shift_max=float("nan"), saturated=True, beats_raw=False,
                   error=str(e)[:255])
    return row


def _pick(rows):
    """Best row: prefer clean (no error, not saturated, beats raw) by score."""
    clean = [r for r in rows
             if r["error"] is None and not r["saturated"] and r["beats_raw"]]
    pool = clean or [r for r in rows
                     if r["error"] is None and np.isfinite(r["score"])]
    pool = pool or rows
    return max(pool, key=lambda r: r["score"]
               if np.isfinite(r["score"]) else float("-inf"))


def search_mc_params(clip, base_params, seed_gsig, seed_max_shift,
                     spec: "McSearchSpec | None" = None, *, n_jobs: int = 1):
    """Search MC params on ``clip`` (a (T, H, W) numpy stack at the MC grid).

    ``seed_gsig`` / ``seed_max_shift`` are the heuristic seeds **in the same
    (downsampled) units as the clip**; they are tried as one extra candidate
    (when ``spec.include_seed``) but do NOT define the grid — the absolute
    ladders in ``spec`` do, so an unreliable seed can't blind the search.
    Returns ``(best_params, rows, evidence)`` where ``best_params`` is
    ``{mc_gSig_filt, max_shift, mc_n_iter}`` and ``rows`` is every candidate
    (crispness / shift / guardrail fields + an ``is_best`` flag) for storage and
    plotting. ``base_params`` is accepted for interface symmetry with the tuner.
    """
    spec = spec or McSearchSpec()
    clip = np.asarray(clip, dtype=np.float32)
    raw_score = float(correlation_image(clip).mean())

    # Absolute candidate value sets, with the heuristic seed appended (hybrid).
    gsig_vals = [max(1.0, float(g)) for g in spec.gsig_values]
    ms_vals = [(int(v), int(v)) for v in spec.max_shift_values]
    if spec.include_seed:
        gsig_vals.append(max(1.0, float(seed_gsig)))
        ms_vals.append((max(1, int(seed_max_shift[0])), max(1, int(seed_max_shift[1]))))
    gsig_vals = list(dict.fromkeys(round(g, 3) for g in gsig_vals))
    ms_vals = list(dict.fromkeys(ms_vals))
    # Most generous max_shift — use it while choosing gSig so real motion is
    # within range (a tight cap would hide it and bias the filter choice).
    probe_ms = max(ms_vals, key=lambda m: min(m))

    # Memoize on the exact (gSig, max_shift, n_iter) so combos shared across the
    # coarse-to-fine sweeps run once.
    seen: dict = {}

    def trial(gsig, max_shift, n_iter):
        g = max(1.0, float(gsig))
        ms = (max(1, int(round(max_shift[0]))), max(1, int(round(max_shift[1]))))
        key = (round(g, 3), ms, int(n_iter))
        if key not in seen:
            seen[key] = _trial(clip, gsig=g, max_shift=ms, n_iter=int(n_iter),
                               upsample=spec.upsample_factor, raw_score=raw_score,
                               sat_frac=spec.saturation_frac, n_jobs=n_jobs)
        return seen[key]

    if spec.coarse_to_fine:
        # 1) sweep gSig at the most generous max_shift (motion visible)
        gsig_rows = [trial(g, probe_ms, 1) for g in gsig_vals]
        best_gsig = _pick(gsig_rows)["mc_gSig_filt"]
        # 2) sweep max_shift at the winning gSig
        ms_rows = [trial(best_gsig, ms, 1) for ms in ms_vals]
        best_ms = _pick(ms_rows)["max_shift"]
        # 3) sweep n_iter (>1; n_iter=1 already covered) at the winning gSig+max_shift
        for n in spec.n_iter_values:
            if n != 1:
                trial(best_gsig, best_ms, n)
    else:
        for g in gsig_vals:
            for ms in ms_vals:
                for n in spec.n_iter_values:
                    trial(g, ms, n)

    rows = list(seen.values())  # unique candidates, insertion order
    best = _pick(rows)
    for r in rows:
        r["is_best"] = r is best
    best_params = {"mc_gSig_filt": best["mc_gSig_filt"],
                   "max_shift": best["max_shift"], "mc_n_iter": best["mc_n_iter"]}
    evidence = {"raw_score": float(raw_score), "seed_gsig": float(seed_gsig),
                "seed_max_shift": (int(seed_max_shift[0]), int(seed_max_shift[1])),
                "n_candidates": len(rows),
                "best_beats_raw": bool(best["beats_raw"]),
                "best_saturated": bool(best["saturated"])}
    return best_params, rows, evidence
