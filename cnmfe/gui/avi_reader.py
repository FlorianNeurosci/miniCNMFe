"""Random-access AVI frame reader for the curation GUI.

The codebase elsewhere decodes AVIs sequentially (imageio v3 / PyAV).
The curation GUI scrubs by frame index, so we need O(1)-ish access.

Strategy
--------
1.  Pre-scan every AVI in the folder for ``(n_frames, H, W)`` and cache the
    table to ``avi_index.json`` next to the AVIs (or in a user-supplied
    location). Pre-scan uses ``_count_and_shape`` from
    ``concat_avis_to_zarr.py`` so frame counts match the rest of the pipeline.
2.  On ``get(t)``, ``np.searchsorted`` on the cumulative-count array maps
    ``t`` -> ``(file_idx, local_frame)``.
3.  Decode the frame via ``cv2.VideoCapture`` with ``CAP_PROP_POS_FRAMES``.
    On init we self-test each file (read 0, read 5, re-read 0; assert
    byte-equal). If a file fails the test, mark it ``backend="pyav"`` and
    use a sequential PyAV decoder + a per-file forward cursor.
4.  An LRU cache around the decoded ``(H, W) uint8`` frame smooths
    short-range scrubbing.

The reader always returns a single-channel ``(H, W) uint8`` frame (averages
3-channel BGR if cv2 hands one back).
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# File listing + pre-scan
# ----------------------------------------------------------------------

def _list_avis(folder: Path, pattern: str = "*.avi") -> list[Path]:
    """Sort AVIs in ``folder`` by the integer in their stem (matches
    ``concat_avis_to_zarr._numeric_key``).
    """
    # Import here so a stale top-level run_curation.py doesn't fail at import
    # time if concat_avis_to_zarr is moved one day.
    from concat_avis_to_zarr import _numeric_key  # type: ignore

    candidates = sorted(folder.glob(pattern), key=_numeric_key)
    return [p for p in candidates if _numeric_key(p) >= 0]


def _count_and_shape_safe(path: Path) -> tuple[int, int, int]:
    """Wrap ``concat_avis_to_zarr._count_and_shape``; raise a clearer error."""
    from concat_avis_to_zarr import _count_and_shape  # type: ignore

    try:
        return _count_and_shape(path)
    except Exception as e:  # pragma: no cover - upstream PyAV is robust
        raise RuntimeError(f"Failed to probe {path}: {e}") from e


# ----------------------------------------------------------------------
# Index file (path -> n_frames + backend choice)
# ----------------------------------------------------------------------

@dataclass
class AviIndexEntry:
    path: str
    n_frames: int
    H: int
    W: int
    mtime: float
    backend: str = "cv2"  # "cv2" or "pyav"


@dataclass
class AviIndex:
    """Per-folder index of AVIs and cumulative frame offsets."""

    files: list[AviIndexEntry] = field(default_factory=list)
    pattern: str = "*.avi"
    version: int = 1

    @property
    def n_frames(self) -> int:
        return sum(f.n_frames for f in self.files)

    @property
    def dims(self) -> tuple[int, int]:
        if not self.files:
            raise RuntimeError("AviIndex is empty.")
        return self.files[0].H, self.files[0].W

    @property
    def offsets(self) -> np.ndarray:
        """Cumulative start frame of each AVI; shape ``(N+1,)``."""
        ns = [f.n_frames for f in self.files]
        return np.concatenate([[0], np.cumsum(ns)]).astype(np.int64)

    def locate(self, t: int) -> tuple[int, int]:
        """Map a global frame ``t`` to ``(file_idx, local_t)``."""
        if t < 0 or t >= self.n_frames:
            raise IndexError(f"frame {t} out of range [0, {self.n_frames})")
        offs = self.offsets
        idx = int(np.searchsorted(offs, t, side="right") - 1)
        return idx, t - int(offs[idx])

    # ---- serialisation -----------------------------------------------

    def to_json(self, path: Path) -> None:
        d = {
            "version": self.version,
            "pattern": self.pattern,
            "files": [asdict(f) for f in self.files],
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(d, indent=2))
        tmp.replace(path)

    @classmethod
    def from_json(cls, path: Path) -> "AviIndex":
        d = json.loads(path.read_text())
        return cls(
            files=[AviIndexEntry(**f) for f in d["files"]],
            pattern=d.get("pattern", "*.avi"),
            version=int(d.get("version", 1)),
        )

    def matches(self, folder: Path, pattern: str) -> bool:
        """True if this index still reflects the current state on disk."""
        current = _list_avis(folder, pattern)
        if len(current) != len(self.files):
            return False
        for p, entry in zip(current, self.files):
            if str(p) != entry.path:
                return False
            try:
                if p.stat().st_mtime != entry.mtime:
                    return False
            except OSError:
                return False
        return True


def build_index(folder: Path, pattern: str = "*.avi") -> AviIndex:
    """Walk the folder, probe each AVI, and return a fresh index."""
    files = _list_avis(folder, pattern)
    if not files:
        raise FileNotFoundError(
            f"No AVIs matching {pattern!r} found in {folder}"
        )
    entries: list[AviIndexEntry] = []
    for p in files:
        n, H, W = _count_and_shape_safe(p)
        entries.append(
            AviIndexEntry(
                path=str(p),
                n_frames=int(n),
                H=int(H),
                W=int(W),
                mtime=p.stat().st_mtime,
            )
        )
    # All files must share the same (H, W); refuse otherwise.
    H, W = entries[0].H, entries[0].W
    for e in entries[1:]:
        if (e.H, e.W) != (H, W):
            raise RuntimeError(
                f"AVI dimension mismatch: {e.path} is ({e.H}, {e.W}), "
                f"expected ({H}, {W})."
            )
    return AviIndex(files=entries, pattern=pattern)


# ----------------------------------------------------------------------
# Per-file decoder backends
# ----------------------------------------------------------------------

class _Cv2Backend:
    """cv2.VideoCapture with random seek; rebuilds on backward jumps."""

    def __init__(self, path: str, n_frames: int):
        import cv2

        self.cv2 = cv2
        self.path = path
        self.n_frames = n_frames
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"cv2 failed to open {path}")

    def read(self, local_t: int) -> np.ndarray:
        cap, cv2 = self.cap, self.cv2
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(local_t))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(
                f"cv2 read failed at local_t={local_t} in {self.path}"
            )
        return _to_gray_uint8(frame)

    def close(self) -> None:
        try:
            self.cap.release()
        except Exception:  # pragma: no cover
            pass


class _PyAvBackend:
    """Sequential PyAV decoder with a forward cursor; restart on backward."""

    def __init__(self, path: str, n_frames: int):
        import av  # type: ignore

        self.av = av
        self.path = path
        self.n_frames = n_frames
        self._open()

    def _open(self) -> None:
        self.container = self.av.open(self.path)
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "FRAME"
        self._iter = self.container.decode(self.stream)
        self._next = 0  # local index of the next frame yielded by _iter

    def read(self, local_t: int) -> np.ndarray:
        if local_t < self._next:
            # Backward seek: PyAV's seek is keyframe-aligned and unreliable
            # on MJPEG-in-AVI; cheaper to restart.
            try:
                self.container.close()
            except Exception:  # pragma: no cover
                pass
            self._open()
        while True:
            try:
                frame = next(self._iter)
            except StopIteration as e:
                raise RuntimeError(
                    f"pyav exhausted before local_t={local_t} in {self.path}"
                ) from e
            cur = self._next
            self._next += 1
            if cur == local_t:
                arr = frame.to_ndarray(format="gray8")
                return np.ascontiguousarray(arr, dtype=np.uint8)

    def close(self) -> None:
        try:
            self.container.close()
        except Exception:  # pragma: no cover
            pass


def _to_gray_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        # BGR or RGB -- mean is fine for visualisation.
        frame = frame.mean(axis=2)
    if frame.dtype != np.uint8:
        # Clip then cast. cv2 occasionally returns int16 for 10-bit AVIs.
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _self_test_cv2(path: str, n_frames: int) -> bool:
    """Read frame 0, frame ``min(5, n-1)``, frame 0 again. True iff stable."""
    try:
        be = _Cv2Backend(path, n_frames)
    except Exception:
        return False
    try:
        f0a = be.read(0)
        if n_frames >= 6:
            be.read(5)
        f0b = be.read(0)
        return np.array_equal(f0a, f0b)
    except Exception:
        return False
    finally:
        be.close()


# ----------------------------------------------------------------------
# Public reader
# ----------------------------------------------------------------------

class AviReader:
    """Random-access reader over a folder of AVIs.

    Frames are addressed by a global index ``[0, n_frames)`` defined by the
    sorted-by-stem-int order of the folder (matching
    ``concat_avis_to_zarr._numeric_key``).
    """

    def __init__(
        self,
        folder: "str | Path",
        *,
        pattern: str = "*.avi",
        index_path: "str | Path | None" = None,
        cache_size: int = 64,
        force_rebuild: bool = False,
    ):
        self.folder = Path(folder)
        self.pattern = pattern
        self.index_path = (
            Path(index_path)
            if index_path is not None
            else self.folder / "avi_index.json"
        )

        idx: AviIndex | None = None
        if not force_rebuild and self.index_path.exists():
            try:
                cached = AviIndex.from_json(self.index_path)
                if cached.matches(self.folder, pattern):
                    idx = cached
            except Exception as e:
                logger.warning("Ignoring stale %s: %s", self.index_path, e)
        if idx is None:
            idx = build_index(self.folder, pattern=pattern)
            # Self-test each file for cv2 random seek and stamp the backend.
            for entry in idx.files:
                if _self_test_cv2(entry.path, entry.n_frames):
                    entry.backend = "cv2"
                else:
                    logger.info(
                        "cv2 seek failed for %s; falling back to pyav", entry.path
                    )
                    entry.backend = "pyav"
            try:
                idx.to_json(self.index_path)
            except OSError as e:  # read-only mount, etc.
                logger.warning("Could not write %s: %s", self.index_path, e)
        self.index = idx

        # Lazy per-file backend, opened on first access.
        self._backends: dict[int, _Cv2Backend | _PyAvBackend] = {}
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_size = int(cache_size)

    # ---- properties --------------------------------------------------

    @property
    def n_frames(self) -> int:
        return self.index.n_frames

    @property
    def dims(self) -> tuple[int, int]:
        return self.index.dims

    @property
    def n_files(self) -> int:
        return len(self.index.files)

    # ---- frame access ------------------------------------------------

    def get(self, t: int) -> np.ndarray:
        """Decode global frame ``t`` -> ``(H, W) uint8``."""
        t = int(t)
        cached = self._cache.get(t)
        if cached is not None:
            # Touch -> move to MRU end.
            self._cache.move_to_end(t)
            return cached
        file_idx, local_t = self.index.locate(t)
        be = self._get_backend(file_idx)
        frame = be.read(local_t)
        # Sanity: dims should match the index's declared dims.
        H, W = self.dims
        if frame.shape != (H, W):
            raise RuntimeError(
                f"Decoded frame at t={t} has shape {frame.shape}, expected {(H, W)}"
            )
        self._cache[t] = frame
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return frame

    def get_range(self, t0: int, t1: int) -> np.ndarray:
        """Decode ``[t0, t1)``; returns ``(t1-t0, H, W) uint8``."""
        if t1 <= t0:
            raise ValueError(f"t1 ({t1}) must be > t0 ({t0})")
        H, W = self.dims
        out = np.empty((t1 - t0, H, W), dtype=np.uint8)
        for i, t in enumerate(range(t0, t1)):
            out[i] = self.get(t)
        return out

    def iter_frames(self, ts: Iterable[int]) -> Iterable[np.ndarray]:
        for t in ts:
            yield self.get(t)

    # ---- backend management ------------------------------------------

    def _get_backend(self, file_idx: int):
        be = self._backends.get(file_idx)
        if be is not None:
            return be
        entry = self.index.files[file_idx]
        if entry.backend == "cv2":
            be = _Cv2Backend(entry.path, entry.n_frames)
        else:
            be = _PyAvBackend(entry.path, entry.n_frames)
        self._backends[file_idx] = be
        return be

    def close(self) -> None:
        for be in self._backends.values():
            try:
                be.close()
            except Exception:  # pragma: no cover
                pass
        self._backends.clear()
        self._cache.clear()

    def __enter__(self) -> "AviReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"AviReader(folder={self.folder!s}, n_files={self.n_files}, "
            f"n_frames={self.n_frames}, dims={self.dims})"
        )
