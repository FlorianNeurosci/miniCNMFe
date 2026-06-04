"""Convert AVI files in demo_movies/ to zarr stores for lazy loading.

Usage (from project root):
    python convert_to_zarr.py

Skips AVIs whose zarr already exists (idempotent).
The zarr stores are time-chunked (100 frames per chunk) and can be opened
lazily with minicnmfe.io.open_zarr() without loading the full movie into RAM.
"""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    from minicnmfe.io import avi_to_zarr

    demo_dir = Path("demo_movies")
    demo_dir = Path('real_vids')
    if not demo_dir.exists():
        print("demo_movies/ not found. Run generate_demo_movies.py first.")
        return

    avis = sorted(demo_dir.glob("*.avi"))
    if not avis:
        print("No AVI files found in demo_movies/. Run generate_demo_movies.py first.")
        return

    for avi_path in avis:
        zarr_path = avi_path.with_suffix(".zarr")
        if zarr_path.exists():
            print(f"  {avi_path.name}: zarr already exists, skipping")
            continue
        print(f"  {avi_path.name} → {zarr_path.name} …", end=" ", flush=True)
        z = avi_to_zarr(avi_path, zarr_path, chunk_t=100, grayscale=True)
        print(f"shape={z.shape}  chunks={z.chunks}")

    print("\nDone. Open zarrs lazily with  minicnmfe.io.open_zarr('demo_movies/<name>.zarr')")


if __name__ == "__main__":
    main()
