"""Realism regression tests for the miniscope simulator.

These guard the calibration that brings ``make_miniscope_movie`` into the
detectability regime of a real clean 1p recording (demo_movies/demo_session):
median local-correlation CORR ≈ 0.92 and median PNR ≈ 10 (vs the old
noise-dominated 0.53 / 3.5), using the same metric the pipeline keys on
(``cnmfe.preprocess.correlation_pnr``). They also pin the F0+ΔF baseline
invariants and the ``realism=False`` escape hatch.
"""

from __future__ import annotations

import numpy as np

from miniscope_simulator import make_miniscope_movie
from cnmfe.preprocess import correlation_pnr


def test_realism_corr_pnr_in_real_like_band():
    """Default (realism, difficulty=0) sim lands in the real clean-recording
    CORR/PNR band, not the old noise-dominated regime."""
    d = make_miniscope_movie(n_neurons=20, dims=(128, 128), T=1000, fps=20.0, seed=0)
    cn, pnr = correlation_pnr(d["movie"], sigma=5.0, center_psf=True)
    corr_p50 = float(np.nanpercentile(cn, 50))
    pnr_p50 = float(np.nanpercentile(pnr, 50))
    # Real target: CORR p50 ≈ 0.92, PNR p50 ≈ 10.2 (old sim was 0.53 / 3.5).
    assert corr_p50 > 0.85, f"CORR p50 {corr_p50:.3f} too low (noise-dominated regime?)"
    assert corr_p50 < 0.999, f"CORR p50 {corr_p50:.3f} pathologically uniform"
    assert pnr_p50 > 7.0, f"PNR p50 {pnr_p50:.2f} too low vs real ~10"


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
