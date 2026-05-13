"""Functional tests: compare our rigid MC to CaImAn's output on real miniscope data.

Reference shifts were produced by CaImAn's rigid motion correction on all 1000
frames of ``tests/data/real_0.avi`` (600×600 px) with the following parameters:

    gSig_filt=(7, 7), max_shifts=(10, 10), border_nan='copy', niter_rig=1

The saved array (shape (1000, 2)) contains CaImAn's ``shifts_rig`` — the
(dy, dx) correction applied to each frame (i.e. negative of motion).

``motion_correction_rigid`` (cnmfe/motion_correction.py) is the validated
implementation that reproduces CaImAn's results.

On first run the AVI is converted to ``tests/data/real_0.zarr`` automatically;
subsequent runs reuse the existing zarr.  All tests are skipped automatically
when neither source is available, so the regular pytest suite is unaffected.

Required files (local-only, not committed to git):
  tests/data/real_0.avi   OR   tests/data/real_0.zarr  (auto-generated)
  tests/fixtures/caiman_shifts_0avi.npy
"""

import pathlib

import numpy as np
import pytest

_TESTS_DIR = pathlib.Path(__file__).parent
VIDEO_PATH = _TESTS_DIR / "data" / "real_0.avi"
ZARR_PATH = _TESTS_DIR / "data" / "real_0.zarr"
CAIMAN_SHIFTS_PATH = _TESTS_DIR / "fixtures" / "caiman_shifts_0avi.npy"

_FILES_PRESENT = (
    (VIDEO_PATH.exists() or ZARR_PATH.exists())
    and CAIMAN_SHIFTS_PATH.exists()
)
requires_real_data = pytest.mark.skipif(
    not _FILES_PRESENT,
    reason="real video/zarr or CaImAn reference shifts not available locally",
)


# Module-scope fixture so the MC run happens once for the whole test class.
@pytest.fixture(scope="module")
def mc_result():
    """Load zarr (converting from AVI on first run), run motion_correction_rigid."""
    from cnmfe.motion_correction import motion_correction_rigid

    if ZARR_PATH.exists():
        from cnmfe.io import open_zarr
        zarr_movie = open_zarr(ZARR_PATH)
    elif VIDEO_PATH.exists():
        from cnmfe.io import avi_to_zarr
        zarr_movie = avi_to_zarr(VIDEO_PATH, ZARR_PATH)
    else:
        pytest.skip("neither zarr nor AVI source available")

    # motion_correction_rigid requires a numpy array
    movie = np.asarray(zarr_movie, dtype=np.float32)

    _, shifts_ours = motion_correction_rigid(
        movie,
        max_shift=(15, 15),
        gSig_filt=7,
        upsample_factor=10,
        niter_rig=1,    # matches CaImAn's default
    )
    shifts_caiman = np.load(CAIMAN_SHIFTS_PATH)
    return shifts_ours, shifts_caiman


@requires_real_data
class TestMCvsCaiman:
    """Compare motion_correction_rigid shifts to CaImAn's reference on real 1p data."""

    def test_shift_correlation(self, mc_result):
        """Pearson |r| vs CaImAn must exceed 0.90 on both dy and dx axes."""
        shifts_ours, shifts_caiman = mc_result
        T = min(len(shifts_ours), len(shifts_caiman))

        for axis, name in enumerate(["dy", "dx"]):
            r = np.corrcoef(shifts_ours[:T, axis], shifts_caiman[:T, axis])[0, 1]
            assert abs(r) > 0.90, (
                f"Shift axis {name}: |Pearson r| = {abs(r):.3f} < 0.90 vs CaImAn"
            )

    def test_shift_mae(self, mc_result):
        """Mean absolute error vs CaImAn shifts must be below 1.5 px on each axis."""
        shifts_ours, shifts_caiman = mc_result
        T = min(len(shifts_ours), len(shifts_caiman))

        for axis, name in enumerate(["dy", "dx"]):
            mae = float(np.abs(shifts_ours[:T, axis] - shifts_caiman[:T, axis]).mean())
            assert mae < 1.5, (
                f"Shift axis {name}: MAE = {mae:.3f} px vs CaImAn (threshold 1.5 px)"
            )

    def test_shift_range_plausible(self, mc_result):
        """Our shifts must stay within the allowed max_shift bounds."""
        shifts_ours, _ = mc_result
        assert np.abs(shifts_ours[:, 0]).max() <= 15.0, "dy shift exceeds max_shift=15"
        assert np.abs(shifts_ours[:, 1]).max() <= 15.0, "dx shift exceeds max_shift=15"
