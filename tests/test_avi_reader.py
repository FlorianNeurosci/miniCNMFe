"""Tests for cnmfe.gui.avi_reader.AviReader.

We write small synthetic MJPEG AVIs with cv2.VideoWriter and verify random
access matches a numpy ground-truth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


cv2 = pytest.importorskip("cv2")

from cnmfe.gui.avi_reader import AviReader, build_index


def _make_synthetic_avi(
    path: Path,
    n_frames: int,
    H: int = 32,
    W: int = 48,
    fps: int = 30,
    base: int = 0,
) -> np.ndarray:
    """Write an MJPEG AVI where frame t has a distinctive pattern.

    Returns the (n_frames, H, W) uint8 array we wrote (for comparison).
    """
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (W, H), isColor=False)
    assert writer.isOpened(), f"VideoWriter could not open {path}"

    frames = np.zeros((n_frames, H, W), dtype=np.uint8)
    for t in range(n_frames):
        # Strong, smooth pattern that survives MJPEG quantisation well:
        # large block of constant intensity that varies frame-to-frame.
        val = (base + t) % 200 + 30  # avoid black; MJPEG washes near 0
        frames[t, :, :] = val
        # Add a corner marker so we can detect frame-ordering bugs even if
        # mean intensity collides.
        frames[t, :8, :8] = (val + 50) % 256
        writer.write(frames[t])
    writer.release()
    return frames


def _approx_equal(a: np.ndarray, b: np.ndarray, tol: int = 8) -> bool:
    """MJPEG is lossy; compare with a small tolerance on absolute pixel diff."""
    if a.shape != b.shape:
        return False
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return float(diff.mean()) <= tol


# ---------------------------------------------------------------------- index

def test_build_index_single_file(tmp_path: Path):
    avi = tmp_path / "0.avi"
    truth = _make_synthetic_avi(avi, n_frames=12, H=32, W=48)
    idx = build_index(tmp_path)
    assert idx.n_frames == 12
    assert idx.dims == (32, 48)
    assert idx.files[0].path == str(avi)
    # locate is correct
    assert idx.locate(0) == (0, 0)
    assert idx.locate(11) == (0, 11)
    with pytest.raises(IndexError):
        idx.locate(12)
    # Truth array used only to silence unused-var lint; presence checked above.
    assert truth.shape == (12, 32, 48)


def test_build_index_multi_file_offsets(tmp_path: Path):
    _make_synthetic_avi(tmp_path / "0.avi", n_frames=5)
    _make_synthetic_avi(tmp_path / "1.avi", n_frames=7)
    _make_synthetic_avi(tmp_path / "2.avi", n_frames=3)
    idx = build_index(tmp_path)
    assert idx.n_frames == 15
    assert idx.locate(0) == (0, 0)
    assert idx.locate(4) == (0, 4)
    assert idx.locate(5) == (1, 0)
    assert idx.locate(11) == (1, 6)
    assert idx.locate(12) == (2, 0)
    assert idx.locate(14) == (2, 2)


def test_build_index_skips_non_numeric_names(tmp_path: Path):
    _make_synthetic_avi(tmp_path / "0.avi", n_frames=4)
    _make_synthetic_avi(tmp_path / "not_a_number.avi", n_frames=4)
    idx = build_index(tmp_path)
    # Only 0.avi has a purely-numeric stem -> kept.
    assert idx.n_frames == 4


# ---------------------------------------------------------------------- reads

def test_reader_random_access_single_file(tmp_path: Path):
    truth = _make_synthetic_avi(tmp_path / "0.avi", n_frames=20)
    with AviReader(tmp_path) as r:
        assert r.n_frames == 20
        assert r.dims == (32, 48)
        # Random order to defeat any "lucky sequential" implementation.
        for t in [17, 0, 5, 19, 3, 0, 5, 17]:
            frame = r.get(t)
            assert frame.dtype == np.uint8
            assert _approx_equal(frame, truth[t]), (
                f"frame {t} mismatch: mean diff "
                f"{np.abs(frame.astype(int) - truth[t].astype(int)).mean():.2f}"
            )


def test_reader_crosses_file_boundary(tmp_path: Path):
    truth0 = _make_synthetic_avi(tmp_path / "0.avi", n_frames=10)
    truth1 = _make_synthetic_avi(tmp_path / "1.avi", n_frames=10, base=100)
    truth = np.concatenate([truth0, truth1], axis=0)
    with AviReader(tmp_path) as r:
        # Walk back and forth across the boundary.
        for t in [0, 9, 10, 11, 9, 10, 0, 19]:
            assert _approx_equal(r.get(t), truth[t])


def test_get_range(tmp_path: Path):
    truth = _make_synthetic_avi(tmp_path / "0.avi", n_frames=8)
    with AviReader(tmp_path) as r:
        out = r.get_range(2, 6)
        assert out.shape == (4, *truth.shape[1:])
        for i in range(4):
            assert _approx_equal(out[i], truth[2 + i])


def test_out_of_range_raises(tmp_path: Path):
    _make_synthetic_avi(tmp_path / "0.avi", n_frames=4)
    with AviReader(tmp_path) as r:
        with pytest.raises(IndexError):
            r.get(4)
        with pytest.raises(IndexError):
            r.get(-1)


# ---------------------------------------------------------------------- index file

def test_index_persists_and_reused(tmp_path: Path):
    _make_synthetic_avi(tmp_path / "0.avi", n_frames=6)
    r1 = AviReader(tmp_path)
    r1.close()
    index_path = tmp_path / "avi_index.json"
    assert index_path.exists()
    mtime_before = index_path.stat().st_mtime
    # Touch the json explicitly so we can detect a rewrite.
    import time

    time.sleep(0.05)
    r2 = AviReader(tmp_path)
    assert r2.n_frames == 6
    r2.close()
    assert index_path.stat().st_mtime == mtime_before, (
        "index was rewritten despite matching folder state"
    )


def test_index_invalidated_when_files_change(tmp_path: Path):
    _make_synthetic_avi(tmp_path / "0.avi", n_frames=5)
    r1 = AviReader(tmp_path)
    r1.close()
    # Add a second file -> index should be rebuilt.
    _make_synthetic_avi(tmp_path / "1.avi", n_frames=3)
    r2 = AviReader(tmp_path)
    assert r2.n_frames == 8
    r2.close()


def test_empty_folder_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        AviReader(tmp_path)


def test_cv2_self_test_returns_consistent_frames(tmp_path: Path):
    """The 0-5-0 self-test in __init__ must run without error on plain MJPEG."""
    _make_synthetic_avi(tmp_path / "0.avi", n_frames=10)
    r = AviReader(tmp_path)
    # All files should be marked cv2 backend on a healthy MJPEG.
    assert all(e.backend == "cv2" for e in r.index.files)
    r.close()
