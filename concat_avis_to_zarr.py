"""Concatenate sequentially numbered AVI files into a single zarr store.

Usage (from project root or anywhere):
    python concat_avis_to_zarr.py /path/to/folder/
    python concat_avis_to_zarr.py /path/to/folder/ --output /path/to/movie.zarr
    python concat_avis_to_zarr.py /path/to/folder/ --pattern "*.avi"

The AVI files are sorted by the integer embedded in their filename:
    0.avi, 1.avi, ..., 65.avi   (numeric order, not lexicographic)

Files whose names are not purely numeric (e.g. "preview.avi") are skipped
unless you pass --pattern to change the glob.



Output zarr is time-chunked (100 frames/chunk), uint8, shape (T_total, H, W),
with lossless blosc lz4+bitshuffle compression. Use --dtype float32 for
float-valued intermediates.
It can be opened lazily with  cnmfe.io.open_zarr(output_path).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


def _iter_frames(path: Path, grayscale: bool = True):
    """Yield frames from a single AVI file in their natural dtype."""
    import imageio.v3 as iio
    for frame in iio.imiter(str(path), plugin="pyav"):
        frame = np.asarray(frame)  # keep natural dtype (uint8 for 8-bit AVIs)
        if grayscale and frame.ndim == 3:
            frame = frame.mean(axis=-1)  # store assignment handles final cast
        yield frame


def _count_and_shape(path: Path) -> tuple[int, int, int]:
    """Return (n_frames, H, W) for a single AVI without loading pixel data."""
    import imageio.v3 as iio
    props = iio.improps(str(path), plugin="pyav")
    n = int(props.n_images)
    _s = props.shape
    if len(_s) >= 3 and _s[0] == n:
        H, W = int(_s[1]), int(_s[2])
    else:
        H, W = int(_s[0]), int(_s[1])
    return n, H, W


def _numeric_key(path: Path) -> int:
    """Sort key: the integer in the filename stem, or -1 if not purely numeric."""
    m = re.fullmatch(r"\d+", path.stem)
    return int(m.group()) if m else -1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path,
                        help="Directory containing numbered AVI files")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output zarr path (default: <folder>/movie.zarr)")
    parser.add_argument("--pattern", default="*.avi",
                        help="Glob pattern for AVI files (default: *.avi)")
    parser.add_argument("--chunk-t", type=int, default=100,
                        help="Frames per time chunk in zarr (default: 100)")
    parser.add_argument("--dtype", default="uint8",
                        help="On-disk dtype (default: uint8; use float32 for float intermediates)")
    parser.add_argument("--color", action="store_true",
                        help="Keep colour channels (default: convert to grayscale)")
    args = parser.parse_args()

    folder: Path = args.folder.resolve()
    if not folder.is_dir():
        parser.error(f"Not a directory: {folder}")

    grayscale = not args.color
    dtype: str = args.dtype

    # --- Collect AVI files, sort numerically --------------------------------
    candidates = sorted(folder.glob(args.pattern), key=_numeric_key)
    avis = [p for p in candidates if _numeric_key(p) >= 0]
    if not avis:
        parser.error(
            f"No numerically-named AVI files found in {folder} "
            f"matching '{args.pattern}'. Expected files like 0.avi, 1.avi, ..."
        )

    print(f"Found {len(avis)} AVI files: "
          f"{avis[0].name} ... {avis[-1].name}")

    # --- Pre-scan: count frames and validate spatial dimensions -------------
    print("Scanning frame counts ...", flush=True)
    counts: list[int] = []
    ref_H = ref_W = None
    for avi in avis:
        n, H, W = _count_and_shape(avi)
        if ref_H is None:
            ref_H, ref_W = H, W
        elif (H, W) != (ref_H, ref_W):
            parser.error(
                f"Spatial mismatch: {avi.name} is {H}x{W} "
                f"but first file is {ref_H}x{ref_W}"
            )
        counts.append(n)
        print(f"  {avi.name}: {n} frames  ({H}x{W})", flush=True)

    T_total = sum(counts)
    print(f"\nTotal: {T_total} frames  x  {ref_H}x{ref_W} px")

    # --- Create zarr --------------------------------------------------------
    out_path: Path = args.output if args.output else folder / "movie.zarr"
    if out_path.exists():
        print(f"\nOutput already exists: {out_path}")
        print("Delete it first if you want to overwrite.")
        return

    from cnmfe.io import _open_array
    store = _open_array(out_path, "w",
                        shape=(T_total, ref_H, ref_W),
                        chunks=(args.chunk_t, ref_H, ref_W),
                        dtype=dtype,
                        compression=True)
    print(f"\nWriting -> {out_path}")
    print(f"  shape={store.shape}  chunks={store.chunks}  dtype={dtype}", flush=True)

    # --- Stream each AVI into the zarr -------------------------------------
    write_start = 0
    buf: list[np.ndarray] = []

    def _flush(buf: list, start: int) -> int:
        if not buf:
            return start
        batch = np.stack(buf, axis=0).astype(dtype)
        end = start + len(batch)
        store[start:end] = batch
        return end

    for avi_idx, avi in enumerate(avis):
        print(f"  [{avi_idx + 1}/{len(avis)}] {avi.name} ...", end=" ", flush=True)
        n_written = 0
        for frame in _iter_frames(avi, grayscale=grayscale):
            buf.append(frame)
            if len(buf) == args.chunk_t:
                write_start = _flush(buf, write_start)
                buf = []
                n_written += args.chunk_t
        print(f"{counts[avi_idx]} frames", flush=True)

    # flush any remaining frames
    write_start = _flush(buf, write_start)

    print(f"\nDone. Zarr written to: {out_path}")
    print(f"  Total frames written: {write_start}")
    print(f"\nLoad lazily with:")
    print(f"  from cnmfe.io import open_zarr")
    print(f"  z = open_zarr('{out_path}')")


if __name__ == "__main__":
    main()
