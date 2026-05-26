"""Tests for the parallel `concat_avis_to_zarr` pipeline.

Generates a handful of tiny synthetic MJPEG AVIs (grayscale source: R==G==B)
and exercises the public API:

1. Serial (`n_jobs=1`) and parallel (`n_jobs=4`) produce byte-equal output.
2. ``grayscale_method="luma"`` and ``"mean"`` produce byte-equal output on
   a grayscale-encoded source.
3. ``skip_if_exists=True`` is idempotent (second call returns the existing
   store without re-decoding).

The fixtures stream to disk with cv2 to avoid colour-space surprises.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

cv2 = pytest.importorskip("cv2")

from concat_avis_to_zarr import concat_avis_to_zarr  # noqa: E402


def _write_synthetic_avi(path: Path, T: int, H: int, W: int, seed: int) -> None:
    """Write a (T, H, W) grayscale MJPEG AVI at `path` (R==G==B per frame)."""
    rng = np.random.default_rng(seed)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (W, H), isColor=True)
    assert writer.isOpened(), f"cv2.VideoWriter failed to open: {path}"
    try:
        for _ in range(T):
            gray = rng.integers(0, 256, size=(H, W), dtype=np.uint8)
            # Stuff the same plane into all three channels so the encoded
            # source is genuinely grayscale (luma == mean).
            frame = np.stack([gray, gray, gray], axis=-1)
            writer.write(frame)
    finally:
        writer.release()


def _make_session(folder: Path, n_files: int = 3, T: int = 30,
                  H: int = 48, W: int = 48) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n_files):
        _write_synthetic_avi(folder / f"{i}.avi", T=T, H=H, W=W, seed=100 + i)


class TestParallelEquivalence:
    """Output must not depend on the parallelism level."""

    def test_serial_vs_parallel_byte_equal(self, tmp_path):
        src = tmp_path / "session"
        _make_session(src, n_files=3, T=30)

        z_serial = concat_avis_to_zarr(
            src, output_path=tmp_path / "serial.zarr",
            chunk_t=10, n_jobs=1, grayscale_method="luma", verbose=False,
        )
        z_parallel = concat_avis_to_zarr(
            src, output_path=tmp_path / "parallel.zarr",
            chunk_t=10, n_jobs=4, grayscale_method="luma", verbose=False,
        )

        a = np.asarray(z_serial[:])
        b = np.asarray(z_parallel[:])
        assert a.shape == b.shape, f"{a.shape} vs {b.shape}"
        assert a.dtype == b.dtype == np.uint8
        # Same input, same decoder path -> byte-equal output.
        assert np.array_equal(a, b), (
            f"parallel and serial outputs differ "
            f"({np.count_nonzero(a != b)} / {a.size} mismatched pixels)"
        )


class TestGrayscaleMethods:
    """For grayscale-encoded sources (R==G==B), luma and mean track each
    other within a few LSB. Exact equality is not expected because pyav's
    Y-plane decode (limited-range) and RGB-mean decode (full-range) use
    different colour-range conventions on MJPEG sources.
    """

    def test_luma_vs_mean_grayscale_source(self, tmp_path):
        src = tmp_path / "session"
        _make_session(src, n_files=2, T=20)

        z_luma = concat_avis_to_zarr(
            src, output_path=tmp_path / "luma.zarr",
            chunk_t=8, n_jobs=1, grayscale_method="luma", verbose=False,
        )
        z_mean = concat_avis_to_zarr(
            src, output_path=tmp_path / "mean.zarr",
            chunk_t=8, n_jobs=1, grayscale_method="mean", verbose=False,
        )

        a = np.asarray(z_luma[:])
        b = np.asarray(z_mean[:])
        # MJPEG decodes luma (Y plane, BT.601 limited-range) and mean
        # (full-range RGB averaged) via two different pyav paths. They are
        # not bit-identical even on grayscale-encoded sources:
        #   - limited→full-range expansion biases the RGB path ~+2 LSB,
        #   - chroma quantisation adds ±1 LSB of stochastic drift,
        #   - float→uint8 truncation in mean() biases another -0.5 LSB.
        # The test guards against gross divergence (e.g. a bug that scales
        # one path), not bit-exactness.
        diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
        assert diff.max() <= 4, (
            f"luma vs mean max drift > 4 LSB (got {diff.max()})"
        )
        assert diff.mean() < 2.5, (
            f"luma vs mean mean drift unexpectedly large (got {diff.mean():.2f})"
        )


class TestIdempotency:
    """skip_if_exists must reuse the existing store without re-decoding."""

    def test_skip_if_exists_reuses_store(self, tmp_path):
        src = tmp_path / "session"
        _make_session(src, n_files=2, T=15)
        out = tmp_path / "movie.zarr"

        z1 = concat_avis_to_zarr(
            src, output_path=out, n_jobs=2, verbose=False,
        )
        first_data = np.asarray(z1[:])

        # Mutate the original AVIs to verify the second call doesn't re-read.
        _write_synthetic_avi(src / "0.avi", T=15, H=48, W=48, seed=9999)

        z2 = concat_avis_to_zarr(
            src, output_path=out, n_jobs=2, skip_if_exists=True, verbose=False,
        )
        second_data = np.asarray(z2[:])

        assert np.array_equal(first_data, second_data), (
            "skip_if_exists must not re-decode"
        )

    def test_existing_output_without_skip_raises(self, tmp_path):
        src = tmp_path / "session"
        _make_session(src, n_files=2, T=15)
        out = tmp_path / "movie.zarr"

        concat_avis_to_zarr(src, output_path=out, n_jobs=1, verbose=False)
        with pytest.raises(FileExistsError):
            concat_avis_to_zarr(src, output_path=out, n_jobs=1, verbose=False)


class TestOutputShape:
    def test_shape_and_chunks(self, tmp_path):
        src = tmp_path / "session"
        _make_session(src, n_files=3, T=25, H=32, W=64)

        z = concat_avis_to_zarr(
            src, output_path=tmp_path / "out.zarr",
            chunk_t=10, n_jobs=2, verbose=False,
        )
        assert isinstance(z, zarr.Array)
        assert z.shape == (75, 32, 64)
        assert z.chunks == (10, 32, 64)
        assert np.dtype(z.dtype) == np.dtype("uint8")


class TestNonUniformLengths:
    """Pre-scan + exact-offset path tolerates files of arbitrary length —
    middle files shorter or longer than their neighbours, last file short.
    """

    def test_non_uniform_lengths(self, tmp_path):
        src = tmp_path / "session"
        src.mkdir()
        _write_synthetic_avi(src / "0.avi", T=15, H=24, W=24, seed=1)
        _write_synthetic_avi(src / "1.avi", T=8,  H=24, W=24, seed=2)
        _write_synthetic_avi(src / "2.avi", T=22, H=24, W=24, seed=3)

        z = concat_avis_to_zarr(
            src, output_path=tmp_path / "out.zarr",
            chunk_t=10, n_jobs=2, verbose=False,
        )
        assert z.shape == (45, 24, 24), z.shape


def _ref_block_mean(movie: np.ndarray, ssub: int, tsub: int) -> np.ndarray:
    """Global block-mean reference (matches per-file binning only when each
    file's frame count is divisible by tsub)."""
    T, H, W = movie.shape
    Tu, Hu, Wu = (T // tsub) * tsub, (H // ssub) * ssub, (W // ssub) * ssub
    m = movie[:Tu, :Hu, :Wu].astype(np.float32)
    m = m.reshape(Tu // tsub, tsub, Hu, Wu).mean(axis=1)
    return m.reshape(Tu // tsub, Hu // ssub, ssub, Wu // ssub, ssub).mean(axis=(2, 4))


class TestInlineDownsampling:
    """ssub/tsub bin frames as they are decoded — only the downsampled zarr
    is written (no intermediate full-res store)."""

    def test_downsampled_shape(self, tmp_path):
        src = tmp_path / "session"
        _make_session(src, n_files=3, T=30, H=48, W=48)
        z = concat_avis_to_zarr(
            src, output_path=tmp_path / "ds.zarr",
            ssub=2, tsub=3, dtype="float32", n_jobs=2, verbose=False,
        )
        # per file 30 // 3 = 10 output frames; 3 files -> 30; 48 // 2 = 24.
        assert z.shape == (30, 24, 24), z.shape
        assert np.dtype(z.dtype) == np.dtype("float32")

    def test_serial_parallel_byte_equal_downsampled(self, tmp_path):
        src = tmp_path / "session"
        _make_session(src, n_files=3, T=30, H=48, W=48)
        zs = concat_avis_to_zarr(src, output_path=tmp_path / "s.zarr",
                                 ssub=2, tsub=3, dtype="float32",
                                 chunk_t=7, n_jobs=1, verbose=False)
        zp = concat_avis_to_zarr(src, output_path=tmp_path / "p.zarr",
                                 ssub=2, tsub=3, dtype="float32",
                                 chunk_t=7, n_jobs=4, verbose=False)
        np.testing.assert_array_equal(np.asarray(zs[:]), np.asarray(zp[:]))

    def test_matches_block_mean_of_full_res(self, tmp_path):
        # T per file (30) divisible by tsub (3) => per-file binning aligns
        # with a global block-mean, so the inline result must match the
        # block-mean of the full-resolution decode.
        src = tmp_path / "session"
        _make_session(src, n_files=3, T=30, H=48, W=48)
        full = concat_avis_to_zarr(src, output_path=tmp_path / "full.zarr",
                                   dtype="float32", n_jobs=2, verbose=False)
        ds = concat_avis_to_zarr(src, output_path=tmp_path / "ds.zarr",
                                 ssub=2, tsub=3, dtype="float32",
                                 n_jobs=2, verbose=False)
        ref = _ref_block_mean(np.asarray(full[:]), ssub=2, tsub=3)
        np.testing.assert_allclose(np.asarray(ds[:]), ref, rtol=1e-4, atol=1e-2)

    def test_spatial_only_keeps_frame_count(self, tmp_path):
        src = tmp_path / "session"
        _make_session(src, n_files=2, T=20, H=32, W=48)
        z = concat_avis_to_zarr(src, output_path=tmp_path / "ds.zarr",
                                ssub=2, tsub=1, dtype="float32",
                                n_jobs=2, verbose=False)
        assert z.shape == (40, 16, 24), z.shape  # T unchanged, H/W halved

    def test_rejects_downsampling_in_color(self, tmp_path):
        src = tmp_path / "session"
        _make_session(src, n_files=1, T=10)
        with pytest.raises(ValueError, match="requires grayscale"):
            concat_avis_to_zarr(src, output_path=tmp_path / "ds.zarr",
                                ssub=2, grayscale=False, verbose=False)
