"""Spatial/temporal cutout (crop) of the movie before extraction.

A cutout restricts CNMFe to a rectangular sub-region (optionally further
narrowed by a boolean ROI mask) and/or a frame window. It is applied **once at
ingestion** — before motion correction and everything downstream — so the rest
of the pipeline simply treats the cutout as "the movie" (``self.dims`` and the
trace length follow automatically).

Cutout is specified on ``CNMFeParams`` (``temporal_crop`` / ``spatial_crop`` /
``spatial_mask_path``) and resolved here into a concrete spec; the resulting
footprints/traces can be mapped back onto the original FOV/timeline with
``place_footprints_in_fov`` / ``place_traces_in_timeline`` (see
``CNMFe.place_in_full_fov``).

Coordinates are NATIVE (full-resolution) and consumed before any downsampling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp


def resolve_cutout(params, native_dims, native_T) -> "dict | None":
    """Normalize the cutout fields on ``params`` against a movie ``(T, H, W)``.

    Returns ``None`` when no cutout is set, else a spec dict:
    ``{orig_dims:[H,W], orig_T, bbox:[y0,y1,x0,x1], t_range:[t0,t1],
       masked:bool, mask_local:(h,w) bool|None}``. ``mask_local`` is the mask
    cropped to ``bbox`` (not JSON-serialisable; the rest is).

    The final ``bbox`` is ``spatial_crop`` intersected with the mask's bounding
    box; pixels inside ``bbox`` but outside the mask are flagged via
    ``mask_local`` for zeroing in ``apply_cutout``.
    """
    H, W = int(native_dims[0]), int(native_dims[1])
    T = int(native_T)
    tc = getattr(params, "temporal_crop", None)
    sc = getattr(params, "spatial_crop", None)
    mp = getattr(params, "spatial_mask_path", None)
    if tc is None and sc is None and mp is None:
        return None

    # --- temporal window ---
    if tc is None:
        t0, t1 = 0, T
    else:
        t0, t1 = int(tc[0]), int(tc[1])
        t0, t1 = max(0, t0), min(T, t1)
        if t1 <= t0:
            raise ValueError(f"temporal_crop {tuple(tc)} empty after clamp to (0, {T})")

    # --- spatial rectangle ---
    if sc is None:
        y0, y1, x0, x1 = 0, H, 0, W
    else:
        y0, y1, x0, x1 = (int(v) for v in sc)
        y0, x0 = max(0, y0), max(0, x0)
        y1, x1 = min(H, y1), min(W, x1)
        if y1 <= y0 or x1 <= x0:
            raise ValueError(f"spatial_crop {tuple(sc)} empty after clamp to {(H, W)}")

    # --- mask: intersect bbox with mask extent ---
    mask_local = None
    if mp is not None:
        mask = np.load(mp)
        if mask.shape != (H, W):
            raise ValueError(
                f"spatial_mask {mask.shape} must match native dims {(H, W)}"
            )
        mask = mask.astype(bool)
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            raise ValueError(f"spatial_mask {mp} is all-False")
        y0, y1 = max(y0, int(ys.min())), min(y1, int(ys.max()) + 1)
        x0, x1 = max(x0, int(xs.min())), min(x1, int(xs.max()) + 1)
        if y1 <= y0 or x1 <= x0:
            raise ValueError("spatial_crop and spatial_mask do not overlap")
        mask_local = mask[y0:y1, x0:x1]

    return {
        "orig_dims": [H, W],
        "orig_T": T,
        "bbox": [int(y0), int(y1), int(x0), int(x1)],
        "t_range": [int(t0), int(t1)],
        "masked": mask_local is not None,
        "mask_local": mask_local,
    }


def apply_cutout(movie, spec: dict) -> np.ndarray:
    """Slice ``movie`` (numpy or zarr) to the cutout and zero outside the mask.

    Returns a contiguous float32 ``(t1-t0, y1-y0, x1-x0)`` array.
    """
    y0, y1, x0, x1 = spec["bbox"]
    t0, t1 = spec["t_range"]
    sub = np.asarray(movie[t0:t1, y0:y1, x0:x1], dtype=np.float32)
    mask_local = spec.get("mask_local")
    if mask_local is not None:
        sub = sub * mask_local[None, :, :].astype(np.float32)
    return np.ascontiguousarray(sub)


def public_spec(spec: dict) -> dict:
    """JSON-serialisable subset of a cutout spec (drops the mask array)."""
    return {k: spec[k] for k in ("orig_dims", "orig_T", "bbox", "t_range", "masked")}


def place_footprints_in_fov(A_crop, bbox, orig_dims):
    """Pad cropped sparse footprints ``(h·w, K)`` back to ``(H·W, K)``.

    Each footprint is written at the ``(y0, x0)`` offset of ``bbox``; pixels
    outside the crop are zero. Pixel order is ``h*W + w`` (matches ``make_3d``).
    """
    y0, y1, x0, x1 = bbox
    H, W = int(orig_dims[0]), int(orig_dims[1])
    h, w = y1 - y0, x1 - x0
    A_crop = A_crop.tocsc()
    K = A_crop.shape[1]
    if A_crop.shape[0] != h * w:
        raise ValueError(
            f"A_crop has {A_crop.shape[0]} rows but bbox implies {h * w}."
        )
    if K == 0:
        return sp.csc_matrix((H * W, 0), dtype=np.float32)
    cols = []
    for k in range(K):
        sub = np.asarray(A_crop[:, k].todense(), dtype=np.float32).reshape(h, w)
        full = np.zeros((H, W), dtype=np.float32)
        full[y0:y1, x0:x1] = sub
        cols.append(sp.csc_matrix(full.reshape(-1, 1)))
    return sp.hstack(cols, format="csc")


def place_traces_in_timeline(C_crop, t_range, orig_T):
    """Embed cropped traces ``(K, T_win)`` into ``(K, orig_T)`` at ``[t0:t1]``.

    Frames outside the window are zero.
    """
    if C_crop is None:
        return None
    t0, t1 = int(t_range[0]), int(t_range[1])
    C_crop = np.asarray(C_crop, dtype=np.float32)
    K, T_win = C_crop.shape
    out = np.zeros((K, int(orig_T)), dtype=np.float32)
    out[:, t0:t0 + T_win] = C_crop
    return out
