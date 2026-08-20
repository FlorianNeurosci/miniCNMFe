"""Tests for the streaming / zarr-backed motion correction path.

These tests verify that:
1. The streaming path (zarr in, zarr out) and the in-memory path produce the
   same shifts on the same movie.
2. Parallel (n_jobs > 1) and serial (n_jobs == 1) produce the same shifts.
3. The output zarr is created with the correct shape, chunks, and dtype, and
   the corrected frames match the in-memory result.
4. Multi-pass (niter_rig > 1) ping-pong works end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from minicnmfe.motion_correction import apply_shift, motion_correction_rigid
from minicnmfe.pipeline import CNMFe, CNMFeParams


def _make_movie(T=40, H=48, W=48, seed=0):
    """A small synthetic movie with known per-frame shifts.

    Builds a clean frame with a few bright blobs, then applies a smooth
    drifting shift per frame. Good enough for shift recovery + identity checks.
    """
    rng = np.random.default_rng(seed)
    frame = np.zeros((H, W), dtype=np.float32)
    for _ in range(6):
        r = rng.integers(8, H - 8)
        c = rng.integers(8, W - 8)
        frame[r - 3:r + 4, c - 3:c + 4] = rng.random()
    shifts = rng.uniform(-3, 3, (T, 2)).astype(np.float32)
    movie = np.stack([apply_shift(frame, shifts[t]) for t in range(T)]).astype(np.float32)
    return movie, shifts


def _smooth_drift_movie(T=60, H=64, W=64, peak=6.0, seed=0):
    """A movie with a smooth, large-amplitude drift spanning the whole clip.

    This is the regime where a smeared full-movie template under-tracks: the
    drift covers ±peak px, so a median over all frames blurs the template.
    """
    rng = np.random.default_rng(seed)
    frame = np.zeros((H, W), dtype=np.float32)
    for _ in range(10):
        r, c = rng.integers(10, H - 10), rng.integers(10, W - 10)
        frame[r - 3:r + 4, c - 3:c + 4] = rng.random() + 0.3
    drift = np.cumsum(rng.normal(0, peak / 8, (T, 2)), axis=0)
    drift -= drift.mean(0)
    drift *= peak / np.abs(drift).max()
    movie = np.stack([apply_shift(frame, drift[t]) for t in range(T)]).astype(np.float32)
    return movie, drift.astype(np.float32)


def _save_zarr(arr: np.ndarray, path: Path, chunk_t: int = 10) -> zarr.Array:
    """Persist a (T, H, W) numpy array as a zarr store for streaming tests."""
    T, H, W = arr.shape
    z = zarr.open_array(
        str(path), mode="w", shape=(T, H, W),
        chunks=(chunk_t, H, W), dtype="float32",
    )
    z[:] = arr
    return z


class TestStreamingPath:
    def test_zarr_in_zarr_out_matches_in_memory(self, tmp_path):
        """Streaming MC (zarr→zarr) must produce the same shifts as in-memory MC."""
        movie, _true = _make_movie(T=30, H=48, W=48, seed=1)

        # In-memory reference
        _, shifts_mem = motion_correction_rigid(
            movie, max_shift=(8, 8), gSig_filt=None,
            upsample_factor=10, niter_rig=1, verbose=False,
        )

        # Streaming via zarr
        src_path = tmp_path / "src.zarr"
        out_path = tmp_path / "mc.zarr"
        src = _save_zarr(movie, src_path, chunk_t=10)

        out, shifts_str = motion_correction_rigid(
            src, output_path=out_path, max_shift=(8, 8), gSig_filt=None,
            upsample_factor=10, niter_rig=1, batch_size=7,
            template_max_frames=1000, verbose=False,
        )

        # Shifts must match exactly (same algorithm, same template)
        np.testing.assert_allclose(shifts_str, shifts_mem, atol=1e-4)

        # Output zarr exists, has correct shape and dtype
        assert isinstance(out, zarr.Array)
        assert out.shape == (30, 48, 48)
        assert np.dtype(out.dtype) == np.dtype("float32")

    def test_streaming_corrected_frames_match_in_memory(self, tmp_path):
        """Corrected frames written to zarr must equal the in-memory result."""
        movie, _ = _make_movie(T=20, H=40, W=40, seed=2)

        corrected_mem, _ = motion_correction_rigid(
            movie, max_shift=(6, 6), gSig_filt=None,
            upsample_factor=5, niter_rig=1, verbose=False,
        )

        src = _save_zarr(movie, tmp_path / "src.zarr", chunk_t=8)
        out_path = tmp_path / "mc.zarr"
        out, _ = motion_correction_rigid(
            src, output_path=out_path, max_shift=(6, 6), gSig_filt=None,
            upsample_factor=5, niter_rig=1, batch_size=8, verbose=False,
        )

        corrected_streamed = np.asarray(out[:])
        np.testing.assert_allclose(corrected_streamed, corrected_mem, atol=1e-4)

    def test_zarr_input_without_output_path_raises(self, tmp_path):
        """A zarr input with no output_path must raise: forces explicit RAM control."""
        movie, _ = _make_movie(T=10, H=32, W=32, seed=3)
        src = _save_zarr(movie, tmp_path / "src.zarr", chunk_t=5)

        with pytest.raises(ValueError, match="output_path"):
            motion_correction_rigid(src, verbose=False)

    def test_numpy_input_with_output_path_uses_streaming(self, tmp_path):
        """Numpy input + output_path should also route to streaming and write zarr."""
        movie, _ = _make_movie(T=20, H=40, W=40, seed=4)
        out_path = tmp_path / "mc.zarr"
        out, shifts = motion_correction_rigid(
            movie, output_path=out_path, max_shift=(6, 6), gSig_filt=None,
            upsample_factor=5, niter_rig=1, batch_size=8, verbose=False,
        )
        assert isinstance(out, zarr.Array)
        assert out_path.exists()
        assert shifts.shape == (20, 2)

    def test_multi_pass_ping_pong(self, tmp_path):
        """niter_rig=2 should run two passes via scratch zarrs and land at output_path."""
        movie, _ = _make_movie(T=25, H=40, W=40, seed=5)

        # Reference: in-memory niter_rig=2
        _, shifts_mem = motion_correction_rigid(
            movie, max_shift=(6, 6), gSig_filt=None,
            upsample_factor=5, niter_rig=2, verbose=False,
        )

        src = _save_zarr(movie, tmp_path / "src.zarr", chunk_t=8)
        out_path = tmp_path / "mc.zarr"
        out, shifts_str = motion_correction_rigid(
            src, output_path=out_path, max_shift=(6, 6), gSig_filt=None,
            upsample_factor=5, niter_rig=2, batch_size=8, verbose=False,
        )

        # Final output landed at the requested path
        assert isinstance(out, zarr.Array)
        assert out_path.exists()

        # Scratch directories are cleaned up
        assert not (out_path.parent / f".{out_path.name}.scratch_a.zarr").exists()
        assert not (out_path.parent / f".{out_path.name}.scratch_b.zarr").exists()

        # Shifts match in-memory result
        np.testing.assert_allclose(shifts_str, shifts_mem, atol=1e-4)

    def test_output_chunk_t_override(self, tmp_path):
        """User-specified output_chunk_t controls the output zarr chunking."""
        movie, _ = _make_movie(T=20, H=32, W=32, seed=6)
        src = _save_zarr(movie, tmp_path / "src.zarr", chunk_t=10)
        out, _ = motion_correction_rigid(
            src, output_path=tmp_path / "mc.zarr",
            max_shift=(5, 5), gSig_filt=None, upsample_factor=5, niter_rig=1,
            batch_size=8, output_chunk_t=4, verbose=False,
        )
        assert out.chunks[0] == 4


class TestParallelism:
    def test_n_jobs_equals_one_vs_serial(self):
        """n_jobs=1 path is the explicit serial branch; sanity-check it works."""
        movie, _ = _make_movie(T=12, H=32, W=32, seed=7)
        _, shifts = motion_correction_rigid(
            movie, max_shift=(5, 5), gSig_filt=None,
            upsample_factor=5, niter_rig=1, n_jobs=1, verbose=False,
        )
        assert shifts.shape == (12, 2)

    @pytest.mark.parametrize("input_mode", ["numpy", "zarr"])
    def test_parallel_matches_serial(self, tmp_path, input_mode):
        """n_jobs=2 must produce the same shifts as n_jobs=1, on both the
        in-memory (``numpy``) and streaming (``zarr``) input paths."""
        movie, _ = _make_movie(T=20, H=40, W=40, seed=8)

        if input_mode == "numpy":
            _, shifts_s = motion_correction_rigid(
                movie, max_shift=(6, 6), gSig_filt=None,
                upsample_factor=5, niter_rig=1, n_jobs=1, verbose=False,
            )
            _, shifts_p = motion_correction_rigid(
                movie, max_shift=(6, 6), gSig_filt=None,
                upsample_factor=5, niter_rig=1, n_jobs=2, verbose=False,
            )
        else:
            src = _save_zarr(movie, tmp_path / "src.zarr", chunk_t=8)
            _, shifts_s = motion_correction_rigid(
                src, output_path=tmp_path / "mc_s.zarr",
                max_shift=(6, 6), gSig_filt=None, upsample_factor=5,
                niter_rig=1, n_jobs=1, batch_size=8, verbose=False,
            )
            _, shifts_p = motion_correction_rigid(
                src, output_path=tmp_path / "mc_p.zarr",
                max_shift=(6, 6), gSig_filt=None, upsample_factor=5,
                niter_rig=1, n_jobs=2, batch_size=8, verbose=False,
            )
        np.testing.assert_allclose(shifts_p, shifts_s, atol=1e-4)


class TestWithHighPass:
    """The 1-photon path (gSig_filt set) is the common production case — exercise it."""

    def test_streaming_matches_in_memory_with_gsig(self, tmp_path):
        movie, _ = _make_movie(T=25, H=48, W=48, seed=10)
        _, shifts_mem = motion_correction_rigid(
            movie, max_shift=(6, 6), gSig_filt=3,
            upsample_factor=10, niter_rig=1, verbose=False,
        )
        src = _save_zarr(movie, tmp_path / "src.zarr", chunk_t=10)
        _, shifts_str = motion_correction_rigid(
            src, output_path=tmp_path / "mc.zarr",
            max_shift=(6, 6), gSig_filt=3, upsample_factor=10,
            niter_rig=1, batch_size=8, verbose=False,
        )
        np.testing.assert_allclose(shifts_str, shifts_mem, atol=1e-3)


class TestConvergeEarlyStop:
    """``converge_tol`` stops multi-pass MC once the template stops sharpening.

    It must be a pure scheduling choice: ``converge_tol=None`` is bit-for-bit
    unchanged, and a run that early-stops at K passes must equal a fixed
    ``niter_rig=K`` run (the early-stop only decides *when* to stop, never what
    each pass computes).
    """

    def test_converge_tol_none_is_bit_for_bit_unchanged(self):
        movie, _ = _make_movie(T=30, H=48, W=48, seed=3)
        _, a = motion_correction_rigid(
            movie, max_shift=(6, 6), gSig_filt=3, upsample_factor=10,
            niter_rig=5, converge_tol=None, verbose=False,
        )
        _, b = motion_correction_rigid(
            movie, max_shift=(6, 6), gSig_filt=3, upsample_factor=10,
            niter_rig=5, verbose=False,
        )
        assert np.array_equal(a, b)

    def test_huge_tol_stops_after_one_pass_and_matches_niter1(self):
        # A tolerance no pass can clear must stop right after the first pass,
        # giving exactly the niter_rig=1 result.
        movie, _ = _make_movie(T=30, H=48, W=48, seed=4)
        _, capped = motion_correction_rigid(
            movie, max_shift=(6, 6), gSig_filt=3, upsample_factor=10,
            niter_rig=10, converge_tol=1e9, verbose=False,
        )
        _, one = motion_correction_rigid(
            movie, max_shift=(6, 6), gSig_filt=3, upsample_factor=10,
            niter_rig=1, verbose=False,
        )
        np.testing.assert_array_equal(capped, one)

    def test_early_stop_streaming_matches_in_memory(self, tmp_path):
        # Same early-stop decision + result on both execution paths.
        movie, _ = _make_movie(T=30, H=48, W=48, seed=5)
        _, mem = motion_correction_rigid(
            movie, max_shift=(6, 6), gSig_filt=3, upsample_factor=10,
            niter_rig=8, converge_tol=0.05, verbose=False,
        )
        src = _save_zarr(movie, tmp_path / "src.zarr", chunk_t=10)
        _, strm = motion_correction_rigid(
            src, output_path=tmp_path / "mc.zarr",
            max_shift=(6, 6), gSig_filt=3, upsample_factor=10,
            niter_rig=8, converge_tol=0.05, batch_size=8, verbose=False,
        )
        np.testing.assert_allclose(strm, mem, atol=1e-3)


class TestFitMcTemplate:
    """``CNMFe.fit_mc`` can take a precomputed template / build one from a window."""

    def _params(self):
        # mc_converge_tol pinned to None: these tests compare fit_mc against a
        # direct motion_correction_rigid call with a fixed niter_rig, so the
        # early-stop (on by default) must be off on both sides.
        return CNMFeParams(max_shift=(6, 6), upsample_factor=10, mc_gSig_filt=3,
                           mc_n_iter=2, mc_converge_tol=None, n_jobs=1)

    def test_fit_mc_template_matches_direct_call(self, tmp_path):
        movie, _ = _make_movie(T=30, H=48, W=48, seed=6)
        tmpl = movie[:8].mean(axis=0).astype(np.float32)
        src = _save_zarr(movie, tmp_path / "src.zarr", chunk_t=10)
        m = CNMFe(self._params())
        m.fit_mc(src, output_dir=tmp_path / "out", template=tmpl)
        _, ref = motion_correction_rigid(
            src, output_path=tmp_path / "ref.zarr", max_shift=(6, 6),
            gSig_filt=3, upsample_factor=10, niter_rig=2, template=tmpl,
            batch_size=200, n_jobs=1, verbose=False,
        )
        np.testing.assert_allclose(m.shifts, ref, atol=1e-4)

    def test_template_window_equals_manual_mean(self, tmp_path):
        movie, _ = _make_movie(T=30, H=48, W=48, seed=7)
        src = _save_zarr(movie, tmp_path / "src.zarr", chunk_t=10)
        a = CNMFe(self._params())
        a.fit_mc(src, output_dir=tmp_path / "a", template_window=(0, 8))
        b = CNMFe(self._params())
        b.fit_mc(src, output_dir=tmp_path / "b",
                 template=movie[:8].mean(axis=0).astype(np.float32))
        np.testing.assert_allclose(a.shifts, b.shifts, atol=1e-4)

    def test_invalid_template_args_raise(self):
        movie, _ = _make_movie(T=20, H=48, W=48, seed=8)
        m = CNMFe(self._params())
        with pytest.raises(ValueError):  # both given
            m.fit_mc(movie, template=movie[0], template_window=(0, 4))
        with pytest.raises(ValueError):  # wrong shape
            m.fit_mc(movie, template=np.zeros((8, 8), np.float32))
        with pytest.raises(ValueError):  # window out of range
            m.fit_mc(movie, template_window=(0, 9999))


class TestSharpenTemplate:
    """Auto template sharpening (default) recovers full drift amplitude at one pass.

    The smeared full-movie median template under-tracks large drift; aligning the
    in-RAM sample first fixes it without extra full-movie passes.
    """

    _MC = dict(gSig_filt=3, max_shift=(10, 10), upsample_factor=10, verbose=False)

    @staticmethod
    def _slope(sh, tgt):
        return float(np.mean([np.cov(sh[:, j], tgt[:, j])[0, 1] / np.var(tgt[:, j])
                              for j in range(2)]))

    def test_one_sharpened_pass_matches_legacy_multipass(self):
        # The core claim: aligning the sample to build a sharp template means a
        # single full pass recovers the amplitude a multi-pass legacy run needs.
        movie, drift = _smooth_drift_movie(seed=1)
        tgt = -drift
        _, sharp1 = motion_correction_rigid(movie, niter_rig=1, **self._MC)
        _, legacy5 = motion_correction_rigid(movie, niter_rig=5,
                                             sharpen_template=False, **self._MC)
        assert self._slope(sharp1, tgt) > 0.85                       # near-unity gain
        assert abs(self._slope(sharp1, tgt) - self._slope(legacy5, tgt)) < 0.1

    def test_streaming_matches_in_memory_with_sharpen(self, tmp_path):
        movie, _ = _smooth_drift_movie(seed=2)
        _, mem = motion_correction_rigid(movie, niter_rig=1, **self._MC)
        src = _save_zarr(movie, tmp_path / "src.zarr", chunk_t=10)
        _, strm = motion_correction_rigid(src, output_path=tmp_path / "mc.zarr",
                                          niter_rig=1, batch_size=8, **self._MC)
        np.testing.assert_allclose(strm, mem, atol=1e-3)

    def test_explicit_template_skips_sharpen(self):
        # When a template is supplied, the sharpen flag is a no-op.
        movie, _ = _smooth_drift_movie(seed=3)
        tmpl = movie[:10].mean(0).astype(np.float32)
        _, a = motion_correction_rigid(movie, niter_rig=1, template=tmpl,
                                       sharpen_template=True, **self._MC)
        _, b = motion_correction_rigid(movie, niter_rig=1, template=tmpl,
                                       sharpen_template=False, **self._MC)
        np.testing.assert_array_equal(a, b)
