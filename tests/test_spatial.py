"""Tests for spatial footprint update."""

import numpy as np
import pytest
import scipy.sparse as sp

from cnmfe.spatial import compute_support, threshold_footprint, update_spatial


class TestComputeSupport:
    def test_shape(self, synth_small):
        d = synth_small
        H, W = d["dims"]
        A = sp.csc_matrix(d["A_true"].astype(np.float32))
        support = compute_support(A, (H, W), dilation_radius=3)
        assert len(support) == H * W

    def test_nonzero_pixels_in_support(self, synth_small):
        """Every pixel that is nonzero in a footprint should be in its own support."""
        d = synth_small
        H, W = d["dims"]
        A = sp.csc_matrix(d["A_true"].astype(np.float32))
        support = compute_support(A, (H, W), dilation_radius=3)

        for k in range(A.shape[1]):
            col = np.asarray(A.getcol(k).todense()).ravel()
            nz_pixels = np.where(col > 0)[0]
            for p in nz_pixels:
                assert k in support[p], f"Component {k} missing from support of pixel {p}"

    def test_dilation_extends_support(self, synth_small):
        """Larger dilation radius should produce more or equal support pixels."""
        d = synth_small
        H, W = d["dims"]
        A = sp.csc_matrix(d["A_true"].astype(np.float32))
        support_small = compute_support(A, (H, W), dilation_radius=1)
        support_large = compute_support(A, (H, W), dilation_radius=5)
        total_small = sum(len(s) for s in support_small)
        total_large = sum(len(s) for s in support_large)
        assert total_large >= total_small


class TestThresholdFootprint:
    def test_output_shape(self):
        H, W = 20, 20
        ai = np.random.rand(H * W).astype(np.float32)
        result = threshold_footprint(ai, (H, W))
        assert result.shape == (H * W,)

    def test_single_connected_component(self):
        """After thresholding, footprint should have ≤ 1 connected component."""
        H, W = 20, 20
        # Two isolated blobs
        ai = np.zeros((H, W), dtype=np.float32)
        ai[3:6, 3:6] = 1.0
        ai[14:17, 14:17] = 0.5  # smaller blob, will be zeroed out
        result = threshold_footprint(ai.ravel(), (H, W)).reshape(H, W)
        # Should keep only one component
        import scipy.ndimage as ndi
        _, n = ndi.label(result > 0)
        assert n <= 1

    def test_zeros_return_zeros(self):
        ai = np.zeros(100, dtype=np.float32)
        result = threshold_footprint(ai, (10, 10))
        np.testing.assert_array_equal(result, 0)

    def test_non_negative(self):
        ai = np.random.rand(400).astype(np.float32)
        result = threshold_footprint(ai, (20, 20))
        assert (result >= 0).all()


class TestUpdateSpatial:
    def test_output_shape(self, synth_small):
        d = synth_small
        T = d["movie"].shape[0]
        H, W = d["dims"]
        K = d["A_true"].shape[1]
        Y_flat = d["movie"].reshape(T, H * W).T
        A = sp.csc_matrix(d["A_true"].astype(np.float32))
        C = d["C_true"].copy()
        sn = np.ones(H * W, dtype=np.float32) * d["sn_true"]

        A_new = update_spatial(Y_flat, C, A, sn, (H, W), dilation_radius=3)
        assert A_new.shape == (H * W, K)

    def test_non_negative_footprints(self, synth_small):
        d = synth_small
        T = d["movie"].shape[0]
        H, W = d["dims"]
        Y_flat = d["movie"].reshape(T, H * W).T
        A = sp.csc_matrix(d["A_true"].astype(np.float32))
        C = d["C_true"].copy()
        sn = np.ones(H * W, dtype=np.float32) * d["sn_true"]

        A_new = update_spatial(Y_flat, C, A, sn, (H, W))
        assert A_new.data.min() >= -1e-6


class TestThresholdFootprintClosing:
    """Binary closing inside threshold_footprint.

    Two complementary tests:

    1. **Legacy invariance**: ``closing_radius=0`` must reproduce the
       pre-change algorithm bit-for-bit. Recovers the old behaviour exactly.

    2. **Closing bridges a 2-pixel gap**: construct a footprint with two
       solid clusters separated by a 2-pixel-wide zero valley wider than
       what the 3×3 median filter can heal. Without closing the largest-CC
       step keeps only the larger cluster; with ``closing_radius=2`` the
       supports are reunified.

       Note: the failure mode in tmp/spatial_traces.png is fragmentation
       caused by post-median, post-threshold fractures of varying width.
       The 3×3 median filter and the default 3×3 closing have overlapping
       1-px reach, so the *default* ``closing_radius=1`` mostly catches
       boundary roughness rather than full splits; ``closing_radius=2`` is
       what we exercise here because it is the regime where closing
       provably adds new behaviour vs the existing median filter.
    """

    def _two_block_footprint(self, H=21, W=21, gap_width=2):
        """5×5 cluster A (value 1.0) and 5×5 cluster B (value 0.5) separated
        by ``gap_width`` columns of zero. With gap_width=2 the 3×3 median
        filter cannot bridge the gap (the median pixel in the centre of the
        gap sees 6 zeros + 3 non-zeros → median = 0).
        """
        ai2d = np.zeros((H, W), dtype=np.float32)
        ay, ax = 5, 5
        ai2d[ay : ay + 5, ax : ax + 5] = 1.0
        bx = ax + 5 + gap_width
        ai2d[ay : ay + 5, bx : bx + 5] = 0.5
        return ai2d, (H, W)

    def test_closing_default_off_reproduces_legacy_behaviour(self):
        """``closing_radius=0`` must reproduce the pre-change pipeline
        (median filter + threshold + largest-CC, no closing) bit-for-bit
        on a representative footprint, so users can opt out and get the
        exact prior behaviour.
        """
        import scipy.ndimage as ndi

        ai2d, dims = self._two_block_footprint()
        H, W = dims

        # Hand-roll the pre-change pipeline.
        legacy_2d = ai2d.copy()
        legacy_2d = ndi.median_filter(legacy_2d, size=3).clip(0)
        legacy_2d[legacy_2d < 0.1 * legacy_2d.max()] = 0.0
        labeled, n = ndi.label(legacy_2d > 0)
        if n > 1:
            sizes = ndi.sum(legacy_2d > 0, labeled, range(1, n + 1))
            largest = int(np.argmax(sizes)) + 1
            legacy_2d[labeled != largest] = 0.0
        legacy = legacy_2d.ravel().astype(np.float32)

        # Pass circular_max_dist_factor=0.0 so the new circular constraint
        # default doesn't perturb the legacy comparison.
        out_off = threshold_footprint(
            ai2d.ravel(), dims, max_thr=0.1, closing_radius=0,
            circular_max_dist_factor=0.0,
        )
        np.testing.assert_array_equal(out_off, legacy)

    def test_closing_bridges_2px_gap_and_keeps_both_clusters(self):
        """With closing_radius=0 the smaller cluster is discarded (largest-CC
        keeps only the larger one). With closing_radius=2 the gap is bridged
        and both clusters are kept. Disable the circular constraint so we
        isolate the closing-only behaviour.
        """
        import scipy.ndimage as ndi

        ai2d, dims = self._two_block_footprint(gap_width=2)
        H, W = dims
        ai_flat = ai2d.ravel()

        out_off = threshold_footprint(
            ai_flat, dims, max_thr=0.1, closing_radius=0,
            circular_max_dist_factor=0.0,
        )
        out_r2 = threshold_footprint(
            ai_flat, dims, max_thr=0.1, closing_radius=2,
            circular_max_dist_factor=0.0,
        )

        n_off = int((out_off > 0).sum())
        n_r2 = int((out_r2 > 0).sum())
        # Without closing: largest-CC picks one cluster only.
        # With closing radius=2: closing unifies the two CC's for the
        # selection step, so both clusters are kept in the output.
        # (The bridge pixels themselves stay 0 — closing affects the
        # binary mask used by ndi.label, not the grayscale values.)
        assert n_r2 >= 1.8 * n_off, (
            f"Closing radius=2 should keep both clusters (~2x pixels). "
            f"Got n_off={n_off}, n_r2={n_r2} (ratio {n_r2 / max(n_off, 1):.2f})"
        )
        # Both pre-existing intensity levels (1.0 from cluster A, 0.5 from
        # cluster B) must survive in the closed output — i.e. neither cluster
        # was dropped.
        assert float(out_r2.max()) == pytest.approx(1.0), "Cluster A (peak 1.0) missing"
        assert ((out_r2 > 0.4) & (out_r2 < 0.6)).any(), "Cluster B (peak 0.5) missing"
        # Without closing: only the larger cluster (peak 1.0) survives;
        # the smaller cluster (peak 0.5) is absent.
        assert not ((out_off > 0.4) & (out_off < 0.6)).any(), (
            "Cluster B (peak 0.5) should be dropped without closing"
        )


class TestSpatialSolverBudget:
    """`spatial_max_iter` / `spatial_tol` knobs and the single-line summary
    that replaces sklearn's per-pixel ConvergenceWarning."""

    def test_max_iter_honored(self, synth_small, capsys):
        """With max_iter=1 (essentially no convergence budget), update_spatial
        should report a non-zero unconverged count in a single summary line.
        """
        d = synth_small
        T = d["movie"].shape[0]
        H, W = d["dims"]
        Y_flat = d["movie"].reshape(T, H * W).T
        A = sp.csc_matrix(d["A_true"].astype(np.float32))
        C = d["C_true"].copy()
        sn = np.ones(H * W, dtype=np.float32) * d["sn_true"]

        update_spatial(
            Y_flat, C, A, sn, (H, W),
            max_iter=1, tol=1e-9,
        )

        out = capsys.readouterr().out
        assert "update_spatial:" in out and "hit max_iter=1" in out, (
            f"Expected single summary line about unconverged pixels; got:\n{out}"
        )
        # Exactly one summary line, not a flood of per-pixel warnings.
        n_summary_lines = sum(
            1 for line in out.splitlines() if "hit max_iter" in line
        )
        assert n_summary_lines == 1, (
            f"Expected exactly 1 summary line; got {n_summary_lines}"
        )

    def test_default_no_summary_when_converged(self, synth_small, capsys):
        """At default budget, the toy fixture converges on every pixel and
        the summary line is NOT printed."""
        d = synth_small
        T = d["movie"].shape[0]
        H, W = d["dims"]
        Y_flat = d["movie"].reshape(T, H * W).T
        A = sp.csc_matrix(d["A_true"].astype(np.float32))
        C = d["C_true"].copy()
        sn = np.ones(H * W, dtype=np.float32) * d["sn_true"]

        update_spatial(Y_flat, C, A, sn, (H, W))  # default max_iter/tol

        out = capsys.readouterr().out
        assert "hit max_iter" not in out, (
            f"Summary line should not appear on a fully-converging fixture; "
            f"got:\n{out}"
        )


class TestDilationRadiusTendrils:
    """Regression: smaller `dilation_radius` reduces LASSO crosstalk between
    adjacent neurons, suppressing the thin "tendril" filaments that were
    visible in extracted footprints when the active set extended too far.
    """

    def _two_adjacent_neuron_movie(self, gap=9, sigma=2.0, T=200, seed=0):
        """Two Gaussian neurons with independent AR(1) traces, centres
        ``gap`` pixels apart. With gap=9 and sigma=2 the footprints are
        well-separated (FWHM≈4.7 px each) but their LASSO active sets
        overlap if `dilation_radius` is large enough.

        Returns (movie (T, H, W), A_true (HW, 2), C_true (2, T), centres).
        """
        H = W = 31
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:H, 0:W]
        c0 = (H // 2, W // 2 - gap // 2)
        c1 = (H // 2, W // 2 + (gap - gap // 2))

        A_true = np.zeros((H * W, 2), dtype=np.float32)
        for k, (cy, cx) in enumerate([c0, c1]):
            blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2)).astype(np.float32)
            blob /= blob.max()
            A_true[:, k] = blob.ravel()

        # Independent AR(1) traces.
        g = 0.9
        S = (rng.random((2, T)) < 0.05).astype(np.float32)
        S[:, 0] = 0
        C_true = np.zeros((2, T), dtype=np.float32)
        for t in range(1, T):
            C_true[:, t] = g * C_true[:, t - 1] + S[:, t]

        # Build the movie + small noise.
        Y_flat = (A_true @ C_true).astype(np.float32)
        Y_flat += rng.standard_normal(Y_flat.shape).astype(np.float32) * 0.05
        movie = Y_flat.T.reshape(T, H, W)
        return movie, A_true, C_true, [c0, c1]

    def test_smaller_dilation_reduces_crosstalk(self):
        """Leaked spatial mass at the *other* neuron's centre must be
        smaller (or equal, since both can be 0 after thresholding) at
        dilation_radius=2 than at dilation_radius=3.

        Compares the worst-case leakage across both directions
        (A[centre_B, k_A] and A[centre_A, k_B]) so we don't accidentally
        pass on a single-direction lucky case.
        """
        movie, A_true, C_true, centres = self._two_adjacent_neuron_movie()
        T = movie.shape[0]
        H, W = movie.shape[1], movie.shape[2]
        Y_flat = movie.reshape(T, H * W).T

        # Use the ground-truth A as starting support, true C as regressors.
        sn = np.full(H * W, 0.05, dtype=np.float32)
        A0 = sp.csc_matrix(A_true)

        def leakage(dilation_radius):
            A_new = update_spatial(
                Y_flat, C_true, A0, sn, (H, W),
                dilation_radius=dilation_radius,
            )
            A_dense = np.asarray(A_new.todense())
            (cy0, cx0), (cy1, cx1) = centres
            # A's value at the OTHER neuron's centre — the tendril precursor.
            leak_a_at_b = float(A_dense[cy1 * W + cx1, 0])
            leak_b_at_a = float(A_dense[cy0 * W + cx0, 1])
            return max(leak_a_at_b, leak_b_at_a)

        leak2 = leakage(2)
        leak3 = leakage(3)
        # We require strict improvement: smaller dilation must reduce the
        # worst-case crosstalk leakage. Using a 0.7x factor leaves room for
        # noise / floating-point variation while still meaningfully
        # demonstrating the suppression.
        assert leak2 < 0.7 * leak3 + 1e-6, (
            f"Smaller dilation should reduce crosstalk leakage at the "
            f"neighbour's centre. leak(d=2)={leak2:.4f}, leak(d=3)={leak3:.4f}"
        )


class TestCircularConstraintPostUpdate:
    """Bandaid prior: clip non-disk-shaped extensions inside
    threshold_footprint. Same prior already used at greedy init.
    """

    def _blob_with_tendril(self, H=31, W=31):
        """A Gaussian blob at (15, 10) plus a 2-px-tall, 8-px-long tendril
        extending toward (15, 22). Tendril is two rows thick so the 3×3
        median filter (which would kill an isolated 1-row line) preserves
        it, and it's long enough that the far end sits well outside the
        circular cutoff for any reasonable factor (≤2).
        """
        ai2d = np.zeros((H, W), dtype=np.float32)
        yy, xx = np.mgrid[0:H, 0:W]
        # Main blob (centroid at (15, 10), sigma 2).
        blob = np.exp(-((yy - 15) ** 2 + (xx - 10) ** 2) / (2 * 4.0)).astype(np.float32)
        ai2d += blob
        # Two-row tendril at rows 14-15, cols 14..22, value 0.2 (above the
        # 0.1 * 1.0 max_thr cutoff).
        for row in (14, 15):
            for col in range(14, 23):
                ai2d[row, col] = 0.2
        return ai2d, (H, W)

    def test_constraint_kills_tendril(self):
        """At the new default factor=1.5 the long tendril's far end gets
        clipped; at factor=0.0 the entire tendril survives.
        """
        ai2d, dims = self._blob_with_tendril()
        ai_flat = ai2d.ravel()

        out_on = threshold_footprint(
            ai_flat, dims, max_thr=0.1, closing_radius=1,
            circular_max_dist_factor=1.5,
        )
        out_off = threshold_footprint(
            ai_flat, dims, max_thr=0.1, closing_radius=1,
            circular_max_dist_factor=0.0,
        )

        H, W = dims
        out_on_2d = out_on.reshape(H, W)
        out_off_2d = out_off.reshape(H, W)

        # Tendril spans cols 14..22 at rows 14..15. At factor=1.5 the far
        # end (cols ≥ ~18) should be clipped; at factor=0.0 all tendril
        # pixels survive.
        tendril_rows = slice(14, 16)
        far_cols = slice(20, 23)  # far end of tendril
        mass_far_on = float(out_on_2d[tendril_rows, far_cols].sum())
        mass_far_off = float(out_off_2d[tendril_rows, far_cols].sum())
        assert mass_far_off > 0, "Tendril must survive when constraint is off"
        assert mass_far_on < 0.2 * mass_far_off, (
            f"Far-end tendril mass with constraint ({mass_far_on:.3f}) "
            f"should be <20% of the unconstrained mass ({mass_far_off:.3f})"
        )
        # The main blob peak must be unaffected by the constraint (it's
        # at the centre of mass, well inside the cutoff radius).
        blob_peak_on = float(out_on_2d[15, 10])
        blob_peak_off = float(out_off_2d[15, 10])
        assert blob_peak_on == pytest.approx(blob_peak_off, rel=1e-3), (
            "Main blob peak should be unaffected by the constraint"
        )

    def test_factor_zero_disables(self):
        """`circular_max_dist_factor=0.0` must be bit-identical to a
        pre-change call (no constraint applied). Use a footprint that
        has no tendrils so the only difference would come from the
        constraint step.
        """
        H = W = 21
        yy, xx = np.mgrid[0:H, 0:W]
        blob = np.exp(-((yy - 10) ** 2 + (xx - 10) ** 2) / (2 * 4.0)).astype(np.float32)
        dims = (H, W)

        out_zero = threshold_footprint(
            blob.ravel(), dims, max_thr=0.1, closing_radius=1,
            circular_max_dist_factor=0.0,
        )

        # Hand-roll the pre-change behaviour: median + threshold + closing
        # + largest-CC, no circular constraint.
        import scipy.ndimage as ndi
        legacy_2d = ndi.median_filter(blob, size=3).clip(0)
        legacy_2d[legacy_2d < 0.1 * legacy_2d.max()] = 0.0
        se = ndi.generate_binary_structure(2, 2)
        closed = ndi.binary_closing(legacy_2d > 0, structure=se)
        labeled, n = ndi.label(closed)
        if n > 1:
            sizes = ndi.sum(closed, labeled, range(1, n + 1))
            largest = int(np.argmax(sizes)) + 1
            keep = labeled == largest
        else:
            keep = closed
        legacy = (legacy_2d * keep.astype(legacy_2d.dtype)).ravel().astype(np.float32)

        np.testing.assert_array_equal(out_zero, legacy)
