"""Seek-based sampling of long single-file recordings (tuning/io_sample.py).

The FFV1 acquisition writes one file per recording with a keyframe every ~12
frames. The tuner used to decode every frame of it to keep every 50th; the seek
path must return exactly the same frames, and the MC clip must start
``start_frac`` into the recording rather than at frame 0.
"""

from __future__ import annotations

import numpy as np
import pytest

av = pytest.importorskip("av")

from tuning import io_sample as S  # noqa: E402

T, H, W, GOP = 300, 16, 16, 12


@pytest.fixture
def ffv1_file(tmp_path):
    """One FFV1 AVI whose frame i has every pixel == i % 256 (so frames are identifiable)."""
    path = tmp_path / "ffv12024-09-17T18_34_45.avi"
    with av.open(str(path), "w") as c:
        s = c.add_stream("ffv1", rate=30)
        s.width, s.height, s.pix_fmt = W, H, "gray"
        s.codec_context.gop_size = GOP
        for i in range(T):
            f = av.VideoFrame.from_ndarray(np.full((H, W), i % 256, np.uint8), format="gray")
            for pkt in s.encode(f):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)
    return path


def _ids(stack):
    return [int(round(float(fr.mean()))) for fr in stack]


def test_seek_sample_equals_full_decode(ffv1_file, monkeypatch):
    monkeypatch.setattr(S, "_SEEK_MIN_FRAMES", 10**9)          # force the old full decode
    full = S.decode_strided_sample([ffv1_file], 8, 7)
    monkeypatch.setattr(S, "_SEEK_MIN_FRAMES", 100)            # force the seek path
    seek = S.decode_strided_sample([ffv1_file], 8, 7)
    np.testing.assert_array_equal(full, seek)
    assert _ids(seek) == [i % 256 for i in range(0, T, 7)]


def test_clip_starts_start_frac_into_a_single_long_file(ffv1_file, monkeypatch):
    monkeypatch.setattr(S, "_SEEK_MIN_FRAMES", 100)
    clip = S.decode_contiguous_clip([ffv1_file], 50, start_frac=0.4)
    assert _ids(clip) == list(range(120, 170))                 # not 0..49 (LED warm-up)


def test_short_chunked_files_keep_the_file_based_start(ffv1_file, monkeypatch):
    monkeypatch.setattr(S, "_SEEK_MIN_FRAMES", 10**9)          # files count as chunks
    clip = S.decode_contiguous_clip([ffv1_file], 50, start_frac=0.4)
    assert _ids(clip) == list(range(0, 50))                    # int(0.4 * 1 file) = file 0
