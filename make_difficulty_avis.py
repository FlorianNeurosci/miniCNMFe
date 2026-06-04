#!/usr/bin/env python
"""Render the realistic miniscope simulator at several difficulty levels to AVIs.

Generates ``make_miniscope_movie`` (tests/miniscope_simulator.py) at a sweep of
``difficulty`` values with the SAME seed/geometry, so only the difficulty changes
between clips — easy to eyeball how the active neuropil, noise, and neuron
contrast evolve from a clean recording (difficulty=0, calibrated to a real clean
session: CORR≈0.92, PNR≈10) toward a hard, low-SNR, neuropil-heavy regime
(difficulty=1).

Each clip is written as 8-bit grayscale (the realism movie is already on a
uint8 0–255 scale, so frames are clipped + cast directly — no per-clip restretch,
so brightness is comparable across difficulties).

Examples
--------
    python make_difficulty_avis.py
    python make_difficulty_avis.py --difficulties 0 0.5 1 --dims 256 256 --T 600
    python make_difficulty_avis.py --out-dir /tmp/sim_avis --fps 20 --stats
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Make the simulator importable whether run from the repo root or elsewhere.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))
from miniscope_simulator import make_miniscope_movie  # noqa: E402


def _to_uint8(movie: np.ndarray) -> np.ndarray:
    """Clip to the uint8 camera range and cast. The realism movie already lives
    on a 0–255 scale, so this preserves cross-difficulty brightness comparison.
    If a (legacy) movie is off-scale, fall back to a per-clip min–max stretch."""
    lo, hi = float(movie.min()), float(movie.max())
    if lo < -1.0 or hi > 300.0:  # not on a uint8 scale (e.g. realism=False) -> stretch
        movie = (movie - lo) / (hi - lo + 1e-10) * 255.0
    return np.clip(np.round(movie), 0, 255).astype(np.uint8)


def _write_avi(path: str, frames_u8: np.ndarray, fps: float) -> str:
    """Write (T, H, W) uint8 frames to an AVI. Prefer cv2 (MJPG), fall back to
    imageio with an explicit codec (cv2 is the project's preferred encoder)."""
    T, H, W = frames_u8.shape
    try:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        vw = cv2.VideoWriter(path, fourcc, float(fps), (W, H), isColor=True)
        if not vw.isOpened():
            raise RuntimeError("cv2.VideoWriter failed to open")
        for t in range(T):
            vw.write(cv2.cvtColor(frames_u8[t], cv2.COLOR_GRAY2BGR))
        vw.release()
        return "cv2/MJPG"
    except Exception as e_cv2:  # noqa: BLE001
        try:
            import imageio.v2 as imageio
            imageio.mimwrite(path, list(frames_u8), fps=float(fps), codec="mjpeg")
            return "imageio/mjpeg"
        except Exception as e_iio:  # noqa: BLE001
            raise RuntimeError(
                f"Could not write AVI via cv2 ({e_cv2}) or imageio ({e_iio}). "
                "Install opencv-python or imageio[ffmpeg]."
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--difficulties", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0],
                    help="difficulty levels in [0,1] (default: 0 .25 .5 .75 1)")
    ap.add_argument("--dims", type=int, nargs=2, default=[256, 256], metavar=("H", "W"))
    ap.add_argument("--T", type=int, default=600, help="frames (default 600)")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--n-neurons", type=int, default=100)
    ap.add_argument("--decay-time-ms", type=float, default=180.0, help="indicator τ (GCaMP8m)")
    ap.add_argument("--seed", type=int, default=0,
                    help="shared seed: same geometry across difficulties")
    ap.add_argument("--out-dir", type=str, default="sim_difficulty_avis")
    ap.add_argument("--legacy", action="store_true",
                    help="also render a realism=False clip for contrast")
    ap.add_argument("--stats", action="store_true",
                    help="also print CORR/PNR per clip (slower; needs minicnmfe)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    H, W = args.dims

    runs = [("difficulty", d, {"difficulty": float(d)}) for d in args.difficulties]
    if args.legacy:
        runs.append(("legacy", None, {"realism": False}))

    print(f"Rendering {len(runs)} clip(s) at {H}x{W}, T={args.T}, fps={args.fps}, "
          f"seed={args.seed} -> {args.out_dir}/")
    for kind, val, kw in runs:
        data = make_miniscope_movie(
            n_neurons=args.n_neurons, dims=(H, W), T=args.T, fps=args.fps,
            decay_time_ms=args.decay_time_ms, seed=args.seed, **kw,
        )
        mov = data["movie"]
        frames = _to_uint8(mov)
        tag = "legacy" if kind == "legacy" else f"d{val:.2f}"
        path = os.path.join(args.out_dir, f"sim_{tag}.avi")
        backend = _write_avi(path, frames, args.fps)

        line = (f"  {path}  [{backend}]  K={data['A_true'].shape[1]}  "
                f"median={float(np.median(mov)):.0f}  range=[{float(mov.min()):.0f},"
                f"{float(mov.max()):.0f}]")
        if args.stats:
            from minicnmfe.preprocess import correlation_pnr
            cn, pnr = correlation_pnr(mov, sigma=5.0, center_psf=True)
            line += (f"  CORR_p50={np.nanpercentile(cn, 50):.2f}  "
                     f"PNR_p50={np.nanpercentile(pnr, 50):.1f}")
        print(line, flush=True)

    print("Done. Open the AVIs in any viewer to inspect.")


if __name__ == "__main__":
    main()
