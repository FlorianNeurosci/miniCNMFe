"""Realism regression tests for the miniscope simulator.

These guard the calibration that puts ``make_miniscope_movie`` into the
*detectability* regime measured on real 1p recordings (``tests/data/real_0.zarr``
plus two PICAST sessions; see ``simulator_calibration.py``): the CORR/PNR seed
maps, intensity range, and photobleach all fall inside the real envelope —
CORR median ~0.5–0.75, PNR median ~3–4 with a detectable tail to ~10, no 8-bit
clipping, ~flat (no photobleach). The OLD sim sat far outside this at CORR ≈ 0.95
/ PNR ≈ 18 with 32% bleaching — a clean, unrealistically easy regime. They also
pin the F0+ΔF baseline invariants and the ``realism=False`` escape hatch.
"""

from __future__ import annotations

import numpy as np

from miniscope_simulator import make_miniscope_movie
from minicnmfe.preprocess import correlation_pnr


def test_realism_corr_pnr_in_real_like_band():
    """Default (realism, difficulty=0) sim lands in the detectability band
    measured on real 1p data, not the old clean/oversaturated regime.

    Real envelope (real_0 + two PICAST sessions): CORR median 0.47–0.76, PNR
    median 3.2–4.3 with p99 6.7–19.2, intensity mean ~52–75 with no 8-bit
    clipping, photobleach ~0. The old sim was CORR 0.95 / PNR 18 / 32% bleach.
    """
    d = make_miniscope_movie(n_neurons=15, dims=(128, 128), T=1000, fps=20.0, seed=0)
    mov = d["movie"]
    cn, pnr = correlation_pnr(mov, sigma=5.0, center_psf=True)
    corr_p50 = float(np.nanpercentile(cn, 50))
    pnr_p50 = float(np.nanpercentile(pnr, 50))
    pnr_p99 = float(np.nanpercentile(pnr, 99))
    # CORR seed map must discriminate (not saturated ~1) yet stay in the real band.
    assert 0.45 < corr_p50 < 0.80, f"CORR p50 {corr_p50:.3f} outside real band [0.45, 0.80]"
    # Background noise floor realistic, with a tail that keeps cells detectable.
    assert 2.5 < pnr_p50 < 5.5, f"PNR p50 {pnr_p50:.2f} outside real band [2.5, 5.5]"
    assert 6.0 < pnr_p99 < 22.0, f"PNR p99 {pnr_p99:.2f} outside real band (cells un/over-detectable)"
    # No 8-bit saturation; intensity in the real range.
    assert (mov >= 255).mean() < 1e-3, "8-bit clipping at 255 (unrealistic)"
    assert 40.0 < float(mov.mean()) < 90.0, f"intensity mean {mov.mean():.0f} outside real range"
    # Real recordings are ~flat (no strong photobleach).
    fm = mov.reshape(mov.shape[0], -1).mean(1)
    drop = float((fm[:100].mean() - fm[-100:].mean()) / max(fm[:100].mean(), 1e-6))
    assert abs(drop) < 0.12, f"photobleach {drop:+.2f} too strong vs ~0 in real data"


def test_realism_baseline_positive_and_uint8_like():
    """F0+ΔF model: a large positive uint8-like baseline, no negatives, with a
    realistic small ΔF/F (background ≫ transient)."""
    d = make_miniscope_movie(n_neurons=20, dims=(96, 96), T=600, fps=20.0, seed=1)
    mov = d["movie"]
    assert mov.min() >= 0.0, "realism baseline must be non-negative (uint8-like)"
    assert 15.0 < float(np.median(mov)) < 80.0, "median baseline should be uint8-scale"


def test_contract_a_true_is_foreground_only():
    """A_true/C_true/S_true describe only the true foreground neurons; all
    nuisance structure (neuropil, haloes, contamination, ghosts) lives in
    ``background``."""
    d = make_miniscope_movie(n_neurons=12, dims=(96, 96), T=400, fps=20.0, seed=2)
    K = len(d["centers"])
    assert d["A_true"].shape[1] == K
    assert d["C_true"].shape[0] == K
    assert d["S_true"].shape[0] == K
    # Neuropil + haloes give the background substantial dynamic energy.
    assert d["background"].std() > 0


def test_difficulty_makes_it_harder():
    """Higher difficulty raises neuropil/noise (the distractor + noise that make
    true-neuron recovery harder)."""
    d0 = make_miniscope_movie(n_neurons=12, dims=(96, 96), T=400, fps=20.0,
                              seed=3, difficulty=0.0)
    d1 = make_miniscope_movie(n_neurons=12, dims=(96, 96), T=400, fps=20.0,
                              seed=3, difficulty=1.0)
    assert d1["background"].std() > d0["background"].std()


def test_realism_false_reproduces_legacy_regime():
    """The kill-switch restores the legacy low-DC composition (a small uniform
    offset, so noise drives values negative) and the smaller legacy neurons."""
    d = make_miniscope_movie(n_neurons=20, dims=(128, 128), T=400, fps=20.0,
                             seed=0, realism=False)
    mov = d["movie"]
    assert mov.min() < 0.0, "legacy path should produce negative values (low DC)"
    assert float(np.median(mov)) < 5.0, "legacy DC baseline is ~1.5, not uint8-scale"
