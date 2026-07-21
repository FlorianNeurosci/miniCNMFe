"""Unit tests for minicnmfe.evaluate."""

import numpy as np
import scipy.sparse as sp

from minicnmfe.evaluate import (
    auto_evaluate_components,
    component_metrics,
    compute_peak_snr,
    compute_trace_skew,
    footprint_morphology,
    spatial_r_values,
)


def _disk_footprint(H: int, W: int, cy: int, cx: int,
                    radius: float, peak: float = 1.0) -> np.ndarray:
    """Flat (H*W,) footprint that's `peak` inside a disk and 0 outside."""
    yy, xx = np.mgrid[0:H, 0:W]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    return (peak * mask.astype(np.float32)).ravel()


class TestAutoEvaluateComponents:
    def test_drops_low_amplitude_footprint(self):
        """A weak-amplitude ghost footprint near the noise floor must be rejected,
        a bright real footprint must be kept — even if the ghost has more pixels."""
        H, W = 32, 32
        sn_flat = np.full(H * W, 0.5, dtype=np.float32)

        # Bright real: ~50 pixels, peak 5.0
        bright = _disk_footprint(H, W, 16, 16, radius=4.0, peak=5.0)
        # Wide weak ghost: ~150 pixels, peak 0.4 (below the noise floor)
        weak = _disk_footprint(H, W, 5, 5, radius=7.0, peak=0.4)

        A = sp.csc_matrix(np.column_stack([bright, weak]))
        keep, info = auto_evaluate_components(A, sn_flat=sn_flat, snr_amp_thr=3.0)

        assert keep.tolist() == [True, False]
        assert info["snr_amp"][0] > 3.0
        assert info["snr_amp"][1] < 3.0

    def test_snr_threshold_zero_keeps_all_nonempty(self):
        """snr_amp_thr=0 disables the amplitude check (only min_pixel applies)."""
        H, W = 32, 32
        sn_flat = np.full(H * W, 0.5, dtype=np.float32)

        bright = _disk_footprint(H, W, 16, 16, radius=4.0, peak=5.0)
        weak = _disk_footprint(H, W, 5, 5, radius=7.0, peak=0.4)
        A = sp.csc_matrix(np.column_stack([bright, weak]))

        keep, _ = auto_evaluate_components(
            A, sn_flat=sn_flat, min_pixel=1, snr_amp_thr=0.0
        )
        assert keep.tolist() == [True, True]

    def test_min_pixel_floor_rejects_tiny_footprint(self):
        """A footprint with fewer than min_pixel pixels must fail regardless of SNR."""
        H, W = 32, 32
        sn_flat = np.full(H * W, 0.5, dtype=np.float32)

        big = _disk_footprint(H, W, 16, 16, radius=4.0, peak=5.0)   # ~50 px
        tiny = _disk_footprint(H, W, 5, 5, radius=1.0, peak=5.0)    # ~5 px, also bright
        A = sp.csc_matrix(np.column_stack([big, tiny]))

        keep, info = auto_evaluate_components(
            A, sn_flat=sn_flat, min_pixel=10, snr_amp_thr=0.0
        )
        assert keep.tolist() == [True, False]
        assert info["pixel_pass"].tolist() == [True, False]

    def test_empty_column_fails(self):
        """A literally empty footprint must always fail."""
        H, W = 16, 16
        sn_flat = np.full(H * W, 0.5, dtype=np.float32)
        A = sp.csc_matrix(np.zeros((H * W, 1), dtype=np.float32))
        keep, info = auto_evaluate_components(A, sn_flat=sn_flat)
        assert keep.tolist() == [False]
        assert info["pixel_count"][0] == 0
        assert info["snr_amp"][0] == 0.0

    def test_zero_components(self):
        """An empty A (K=0) returns empty mask without crashing."""
        H, W = 16, 16
        sn_flat = np.full(H * W, 0.5, dtype=np.float32)
        A = sp.csc_matrix((H * W, 0), dtype=np.float32)
        keep, info = auto_evaluate_components(A, sn_flat=sn_flat)
        assert keep.shape == (0,)
        assert info["snr_amp"].shape == (0,)

    def test_scale_invariance(self):
        """Doubling both the footprint amplitude and the noise std should not
        change the keep decision — the SNR statistic is scale-invariant."""
        H, W = 32, 32
        a = _disk_footprint(H, W, 16, 16, radius=4.0, peak=1.0)
        A = sp.csc_matrix(a.reshape(-1, 1))

        for scale in [0.1, 1.0, 10.0, 100.0]:
            A_s = A * scale
            sn_s = np.full(H * W, 0.3 * scale, dtype=np.float32)
            keep, info = auto_evaluate_components(A_s, sn_flat=sn_s, snr_amp_thr=3.0)
            # SNR statistic should be ~constant across scales
            np.testing.assert_allclose(info["snr_amp"][0], (1.0 / 0.3) ** 2, rtol=1e-4)
            assert keep.tolist() == [True]


class TestSpatialRValues:
    def test_real_cell_high_noise_cell_low(self):
        """A footprint whose pixels co-fluctuate with its trace scores high r;
        a footprint over noise (uncorrelated with its trace) scores ~0."""
        H = W = 24
        T = 400
        rng = np.random.default_rng(0)
        yy, xx = np.indices((H, W))
        g = np.exp(-(((yy - 12) ** 2 + (xx - 12) ** 2) / (2 * 3.0 ** 2))).ravel()
        trace = np.clip(rng.standard_normal(T), 0, None)
        trace[::20] += 5.0  # transients
        movie = np.outer(g, trace) + 0.1 * rng.standard_normal((H * W, T))
        noise_fp = (rng.random(H * W) < 0.02).astype(np.float32)
        A = sp.csc_matrix(np.stack([g.astype(np.float32), noise_fp], axis=1))
        C = np.stack([trace, rng.standard_normal(T)])

        r = spatial_r_values(A, C, movie, (H, W))
        assert r[0] > 0.5
        assert r[0] > r[1]
        assert abs(r[1]) < 0.3

    def test_empty_footprint_is_nan(self):
        H = W = 16
        A = sp.csc_matrix((H * W, 1), dtype=np.float32)  # no pixels
        C = np.ones((1, 50), dtype=np.float32)
        movie = np.zeros((H * W, 50), dtype=np.float32)
        r = spatial_r_values(A, C, movie, (H, W))
        assert np.isnan(r[0])


def _make_traces(T=400, seed=0):
    """A right-skewed transient trace + a symmetric-noise trace, with YrA noise."""
    rng = np.random.default_rng(seed)
    transient = np.clip(rng.standard_normal(T), 0, None)
    transient[::25] += 6.0  # sparse positive spikes -> right-skewed, tall peaks
    noise = rng.standard_normal(T)  # symmetric, no transients
    C = np.stack([transient, np.zeros(T)])           # demixed
    YrA = 0.2 * rng.standard_normal((2, T))          # residual ~ noise
    YrA[1] = noise                                   # cell 1's "trace" is pure noise
    return C, YrA


class TestComputePeakSnr:
    def test_transient_beats_noise(self):
        C, YrA = _make_traces()
        ps = compute_peak_snr(C, YrA)
        assert ps[0] > ps[1]
        assert ps[0] > 5.0          # a real transient towers over the noise floor
        assert ps.shape == (2,)

    def test_dead_trace_no_div_by_zero(self):
        C = np.zeros((1, 100))
        YrA = np.zeros((1, 100))    # degenerate noise floor
        ps = compute_peak_snr(C, YrA)
        assert ps.shape == (1,)
        assert np.isfinite(ps[0]) and ps[0] == 0.0


class TestComputeTraceSkew:
    def test_right_skew_beats_symmetric(self):
        C, YrA = _make_traces()
        sk = compute_trace_skew(C, YrA)
        assert sk[0] > sk[1]        # transient trace is right-skewed
        assert sk[0] > 0.5
        assert abs(sk[1]) < 0.5     # noise ~ symmetric


class TestFootprintMorphology:
    def _disk(self, H, W, cy, cx, r):
        yy, xx = np.mgrid[0:H, 0:W]
        return ((yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2).astype(np.float32).ravel()

    def _line(self, H, W, cy, cx, length):
        m = np.zeros((H, W), dtype=np.float32)
        m[cy, cx:cx + length] = 1.0
        return m.ravel()

    def test_disk_compact_line_eccentric(self):
        H = W = 40
        disk = self._disk(H, W, 20, 20, 5.0)        # compact, round
        line = self._line(H, W, 10, 5, 20)          # 1px-tall elongated streak
        A = sp.csc_matrix(np.column_stack([disk, line]))
        m = footprint_morphology(A, (H, W))
        assert m["eccentricity"][0] < m["eccentricity"][1]
        assert m["eccentricity"][1] > 0.9           # line is highly eccentric
        assert m["solidity"][0] > 0.8               # disk ~ convex
        assert m["aspect_ratio"][1] > m["aspect_ratio"][0]

    def test_empty_is_nan(self):
        H = W = 16
        A = sp.csc_matrix((H * W, 1), dtype=np.float32)
        m = footprint_morphology(A, (H, W))
        for key in ("eccentricity", "solidity", "compactness", "aspect_ratio"):
            assert np.isnan(m[key][0])


class TestComponentMetrics:
    EXPECTED_KEYS = {
        "npix", "snr_amp", "r_value", "cn_corr",
        "eccentricity", "solidity", "compactness", "aspect_ratio",
        "peak_snr", "skew", "kurtosis", "cc_purity",
        "autocorr_lag1", "max_amp", "mean_amp",
    }

    def _model_arrays(self, H=24, W=24, T=300, seed=1):
        rng = np.random.default_rng(seed)
        yy, xx = np.indices((H, W))
        g = np.exp(-(((yy - 12) ** 2 + (xx - 12) ** 2) / (2 * 3.0 ** 2))).ravel()
        g = g.astype(np.float32)
        trace = np.clip(rng.standard_normal(T), 0, None)
        trace[::20] += 5.0
        movie = (np.outer(g, trace) + 0.1 * rng.standard_normal((H * W, T))).astype(np.float32)
        A = sp.csc_matrix(g.reshape(-1, 1))
        C = trace.reshape(1, -1).astype(np.float64)
        YrA = (0.1 * rng.standard_normal((1, T))).astype(np.float64)
        sn_flat = np.full(H * W, 0.1, dtype=np.float32)
        return A, C, YrA, movie, sn_flat, (H, W)

    def test_all_keys_present_with_movie(self):
        A, C, YrA, movie, sn_flat, dims = self._model_arrays()
        cn = np.zeros(dims, dtype=np.float32)
        m = component_metrics(A, C, YrA, None, sn_flat, dims, movie=movie, cn=cn)
        assert set(m.keys()) == self.EXPECTED_KEYS
        for key, arr in m.items():
            assert arr.shape == (1,), key
        assert np.isfinite(m["r_value"][0])     # movie supplied
        assert np.isfinite(m["cn_corr"][0])     # cn supplied

    def test_movie_dependent_metrics_nan_without_movie(self):
        A, C, YrA, _movie, sn_flat, dims = self._model_arrays()
        m = component_metrics(A, C, YrA, None, sn_flat, dims, movie=None, cn=None)
        assert set(m.keys()) == self.EXPECTED_KEYS
        assert np.isnan(m["r_value"][0])
        assert np.isnan(m["cn_corr"][0])
        # Movie-free metrics still finite.
        assert np.isfinite(m["peak_snr"][0])
        assert np.isfinite(m["skew"][0])

    def test_zero_components(self):
        H = W = 16
        A = sp.csc_matrix((H * W, 0), dtype=np.float32)
        C = np.zeros((0, 50)); YrA = np.zeros((0, 50))
        sn_flat = np.full(H * W, 0.1, dtype=np.float32)
        m = component_metrics(A, C, YrA, None, sn_flat, (H, W))
        assert set(m.keys()) == self.EXPECTED_KEYS
        for arr in m.values():
            assert arr.shape == (0,)
