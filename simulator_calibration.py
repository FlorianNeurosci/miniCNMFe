"""Calibrate the realistic miniscope simulator against real recordings.

Computes a panel of ground-truth-free statistics that drive cell *detectability*
(CORR/PNR distributions, intensity range, background coherence, seed density) on
a real recording and on the simulator, side by side, so the simulator can be
tuned until it matches reality.

Usage:
    python simulator_calibration.py --real tests/data/real_0.zarr --sim generate
    python simulator_calibration.py --real tests/data/real_0.zarr --sim path/to/movie.avi
    python simulator_calibration.py --real tests/data/real_0.zarr --sweep-difficulty

The stat functions are importable (reused by tests/test_simulator_realism.py).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from minicnmfe.preprocess import correlation_pnr
from minicnmfe.initialization import detect_seeds


# --------------------------------------------------------------------------- IO
def load_movie(path: str | Path, max_frames: int | None = 2000) -> np.ndarray:
    """Load a (T, H, W) float32 movie from a zarr store or an AVI file."""
    path = Path(path)
    if path.suffix == ".zarr" or path.is_dir():
        from minicnmfe.io import open_zarr
        z = open_zarr(path)
        T = z.shape[0] if max_frames is None else min(z.shape[0], max_frames)
        return np.asarray(z[:T], dtype=np.float32)
    # AVI
    import imageio.v3 as iio
    frames = iio.imread(path, plugin="pyav")  # (T, H, W[, C])
    arr = np.asarray(frames)
    if arr.ndim == 4:
        arr = arr[..., 0]
    if max_frames is not None:
        arr = arr[:max_frames]
    return arr.astype(np.float32)


# ------------------------------------------------------------------- statistics
def movie_stats(mov: np.ndarray, sigma: float = 5.0, n_jobs: int = 4) -> dict:
    """GT-free detectability statistics for a (T, H, W) movie."""
    T, H, W = mov.shape
    cn, pnr = correlation_pnr(mov, sigma=sigma, center_psf=True, n_jobs=n_jobs)
    fr = mov.reshape(T, -1)
    pc = lambda x, q: float(np.percentile(x, q))

    # shared-background dominance via temporal-PC variance fractions
    Yd = fr - fr.mean(0, keepdims=True)
    s = np.linalg.svd(Yd, compute_uv=False)
    var = s ** 2
    pc1 = float(var[0] / var.sum())
    top5 = float(var[:5].sum() / var.sum())

    seeds = detect_seeds(cn, pnr, min_corr=0.8, min_pnr=10, min_distance=5)
    nseed = 0 if seeds is None else len(seeds)

    # photobleach proxy: relative drop in per-frame mean (smoothed) over time
    fmean = fr.mean(1)
    bleach = float((fmean[: T // 10].mean() - fmean[-T // 10:].mean())
                   / max(fmean[: T // 10].mean(), 1e-6))

    return dict(
        shape=(T, H, W),
        int_mean=float(mov.mean()), int_p1=pc(mov, 1), int_p99=pc(mov, 99),
        int_max=float(mov.max()), clip_lo=float((mov <= 0).mean()),
        clip_hi=float((mov >= 255).mean()),
        cn_med=float(np.median(cn)), cn_p90=pc(cn, 90), cn_p99=pc(cn, 99),
        pnr_med=float(np.median(pnr)), pnr_p90=pc(pnr, 90), pnr_p99=pc(pnr, 99),
        pc1=pc1, top5=top5,
        seed_density=nseed / (H * W) * 1e4, bleach_frac=bleach,
    )


def print_panel(label_stats: list[tuple[str, dict]]) -> None:
    rows = [
        ("shape", lambda s: f"{s['shape'][0]}x{s['shape'][1]}x{s['shape'][2]}"),
        ("intensity mean", lambda s: f"{s['int_mean']:.0f}"),
        ("intensity p1/p99/max", lambda s: f"{s['int_p1']:.0f}/{s['int_p99']:.0f}/{s['int_max']:.0f}"),
        ("8-bit clip lo/hi %", lambda s: f"{100*s['clip_lo']:.1f}/{100*s['clip_hi']:.1f}"),
        ("CORR med/p90/p99", lambda s: f"{s['cn_med']:.2f}/{s['cn_p90']:.2f}/{s['cn_p99']:.2f}"),
        ("PNR med/p90/p99", lambda s: f"{s['pnr_med']:.1f}/{s['pnr_p90']:.1f}/{s['pnr_p99']:.1f}"),
        ("bg PC1 / top5 frac", lambda s: f"{s['pc1']:.2f}/{s['top5']:.2f}"),
        ("seeds /1e4 px", lambda s: f"{s['seed_density']:.1f}"),
        ("bleach frac", lambda s: f"{s['bleach_frac']:+.2f}"),
    ]
    labels = [lab for lab, _ in label_stats]
    w = max(len(r[0]) for r in rows) + 1
    head = "  ".join(f"{lab:>16}" for lab in labels)
    print(f"{'stat':<{w}}  {head}")
    for name, fn in rows:
        cells = "  ".join(f"{fn(s):>16}" for _, s in label_stats)
        print(f"{name:<{w}}  {cells}")


def gen_sim(difficulty: float = 0.0, seed: int = 21, **kw) -> np.ndarray:
    sys.path.insert(0, str(Path(__file__).parent / "tests"))
    from miniscope_simulator import make_miniscope_movie
    res = make_miniscope_movie(
        n_neurons=15, dims=(128, 128), T=600, difficulty=difficulty, seed=seed, **kw)
    return np.asarray(res["movie"], dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--sim", default="generate")
    ap.add_argument("--difficulty", type=float, default=0.0)
    ap.add_argument("--sweep-difficulty", action="store_true")
    args = ap.parse_args()

    real = load_movie(args.real)
    panel = [("REAL", movie_stats(real))]
    if args.sweep_difficulty:
        for d in (0.0, 0.25, 0.5, 0.75, 1.0):
            panel.append((f"sim d={d}", movie_stats(gen_sim(difficulty=d))))
    elif args.sim == "generate":
        panel.append((f"sim d={args.difficulty}", movie_stats(gen_sim(difficulty=args.difficulty))))
    else:
        panel.append(("sim", movie_stats(load_movie(args.sim))))
    print_panel(panel)


if __name__ == "__main__":
    main()
