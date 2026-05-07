"""Generate demo movies in demo_movies/ as AVI files with ground-truth NPZ sidecars.

Usage (from project root):
    python generate_demo_movies.py

Movies produced:
    simple_small     — 64×64, T=300,   6 neurons, clean synthetic, no motion
    simple_medium    — 128×128, T=1000, 12 neurons, clean synthetic, no motion
    realistic_small  — 64×64, T=300,   8 neurons, all artefacts, 5 px drift
    realistic_medium — 128×128, T=600, 15 neurons, all artefacts, 8 px drift
    realistic_large  — 256×256, T=1000, 25 neurons, all artefacts, 10 px drift

Each movie gets:
    <name>.avi          — grayscale 8-bit AVI (uint8, normalised to [0, 255])
    <name>_meta.npz     — A_true, C_true, S_true, centers, dims (+ g_true,
                          motion_shifts for realistic movies)

Re-running skips existing files (idempotent).
Run convert_to_zarr.py afterwards to produce lazy-loadable zarr stores.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

CONFIGS = [
    dict(
        name="simple_small",
        kind="synthetic",
        kwargs=dict(n_neurons=6, dims=(64, 64), T=300,
                    noise_std=0.5, bg_strength=1.5, seed=10),
        fps=20.0,
    ),
    dict(
        name="simple_medium",
        kind="synthetic",
        kwargs=dict(n_neurons=12, dims=(128, 128), T=1000,
                    noise_std=0.4, bg_strength=2.0, seed=11),
        fps=20.0,
    ),
    dict(
        name="realistic_small",
        kind="realistic",
        kwargs=dict(n_neurons=8, dims=(64, 64), T=300,
                    n_ghost_cells=4, vasculature=True,
                    vignette_strength=0.4, photobleach_tau_factor=3.0,
                    shot_noise=True, quantize_8bit=True,
                    motion_max_shift=5.0, seed=20),
        fps=20.0,
    ),
    dict(
        name="realistic_medium",
        kind="realistic",
        kwargs=dict(n_neurons=15, dims=(128, 128), T=600,
                    n_ghost_cells=8, bg_n_components=5,
                    vasculature=True, vignette_strength=0.4,
                    photobleach_tau_factor=3.0,
                    shot_noise=True, quantize_8bit=True,
                    motion_max_shift=8.0, seed=21),
        fps=20.0,
    ),
    dict(
        name="realistic_large",
        kind="realistic",
        kwargs=dict(n_neurons=25, dims=(256, 256), T=1000,
                    n_ghost_cells=12, bg_n_components=5,
                    vasculature=True, vignette_strength=0.4,
                    photobleach_tau_factor=3.0,
                    shot_noise=True, quantize_8bit=True,
                    motion_max_shift=10.0, seed=22),
        fps=20.0,
    ),
dict(
        name="realistic_medium_long",
    kind="realistic",
    kwargs=dict(n_neurons=15, dims=(128, 128), T=6000,
                n_ghost_cells=8, bg_n_components=5,
                vasculature=True, vignette_strength=0.4,
                photobleach_tau_factor=3.0,
                shot_noise=True, quantize_8bit=True,
                motion_max_shift=8.0, seed=32),
    fps=20.0,
),
]


def to_uint8(movie: np.ndarray) -> np.ndarray:
    """Normalise (T, H, W) float32 → uint8 using 99.5th-percentile clipping."""
    lo = float(movie.min())
    hi = float(np.percentile(movie, 99.5))
    clipped = np.clip(movie, lo, hi)
    return ((clipped - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)


def save_avi(movie_u8: np.ndarray, path: Path, fps: float = 20.0) -> None:
    """Write (T, H, W) uint8 grayscale array as AVI. Tries cv2, falls back to imageio."""
    T, H, W = movie_u8.shape
    try:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(str(path), fourcc, fps, (W, H), isColor=False)
        if not out.isOpened():
            raise RuntimeError("VideoWriter failed to open")
        for frame in movie_u8:
            out.write(frame)
        out.release()
        return
    except Exception as e:
        print(f"    cv2 failed ({e}), falling back to imageio …")

    import imageio
    imageio.mimwrite(
        str(path),
        [frame for frame in movie_u8],
        fps=fps,
        format="FFMPEG",
        codec="mjpeg",
        output_params=["-pix_fmt", "yuvj420p"],
    )


def main() -> None:
    from tests.conftest import make_synthetic_movie
    from tests.miniscope_simulator import make_miniscope_movie

    out_dir = Path("demo_movies")
    out_dir.mkdir(exist_ok=True)

    for cfg in CONFIGS:
        name: str = cfg["name"]
        avi_path = out_dir / f"{name}.avi"
        meta_path = out_dir / f"{name}_meta.npz"

        if avi_path.exists():
            print(f"  {name}: already exists, skipping")
            continue

        print(f"  {name}: generating …", flush=True)
        if cfg["kind"] == "synthetic":
            data = make_synthetic_movie(**cfg["kwargs"])
        else:
            data = make_miniscope_movie(**cfg["kwargs"])

        movie = data["movie"]          # (T, H, W) float32
        print(f"    shape={movie.shape}  "
              f"range=[{movie.min():.2f}, {movie.max():.2f}]")

        save_avi(to_uint8(movie), avi_path, fps=cfg["fps"])
        print(f"    saved {avi_path}  ({avi_path.stat().st_size / 1e6:.1f} MB)")

        meta_keys = ["A_true", "C_true", "S_true", "centers", "dims"]
        if "g_true" in data:
            meta_keys.append("g_true")
        if "motion_shifts" in data:
            meta_keys.append("motion_shifts")
        np.savez(meta_path, **{k: np.asarray(data[k]) for k in meta_keys if k in data})
        print(f"    saved {meta_path}")

    print("\nDone. Run  python convert_to_zarr.py  to convert AVIs to zarr.")


if __name__ == "__main__":
    main()
