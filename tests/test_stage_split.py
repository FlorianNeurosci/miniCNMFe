"""Regression tests for the staged decomposition of ``CNMFe.fit``.

``fit`` is now a thin wrapper that composes ``fit_mc`` (optional, in-memory) ->
``fit_extract`` -> ``evaluate``. These tests pin the contract that running the
stages individually reproduces the monolithic ``fit`` exactly, so the
disk-handoff CLI workflow (run_mc / run_extract / run_evaluate) is faithful.
"""

import numpy as np

from minicnmfe.pipeline import CNMFe, CNMFeParams


def _params() -> CNMFeParams:
    # n_jobs=1 keeps everything deterministic so array equality is meaningful.
    return CNMFeParams(
        sigma=3.0,
        min_corr=0.5,
        min_pnr=3.0,
        n_iter_main=1,
        n_iter_temporal=1,
        n_jobs=1,
    )


def _assert_same_model(a: CNMFe, b: CNMFe) -> None:
    assert a.A.shape == b.A.shape
    np.testing.assert_array_equal(a.A.toarray(), b.A.toarray())
    np.testing.assert_allclose(a.C, b.C, rtol=0, atol=0)
    np.testing.assert_allclose(a.S, b.S, rtol=0, atol=0)
    np.testing.assert_allclose(a.YrA, b.YrA, rtol=0, atol=0)
    np.testing.assert_array_equal(a.accepted_mask, b.accepted_mask)


def test_extract_then_evaluate_matches_fit(synth_small):
    """fit(do_motion_correction=False) == fit_extract(evaluate=False) + evaluate()."""
    movie = synth_small["movie"]

    full = CNMFe(_params()).fit(movie, do_motion_correction=False)

    staged = CNMFe(_params())
    staged.fit_extract(movie, evaluate=False)
    # Deferred evaluation: the mask is unset until evaluate() is called.
    assert staged.accepted_mask is None
    staged.evaluate()

    _assert_same_model(full, staged)


def test_mc_extract_evaluate_matches_fit(synth_small):
    """fit(do_motion_correction=True) == fit_mc -> fit_extract -> evaluate."""
    movie = synth_small["movie"]

    full = CNMFe(_params()).fit(movie, do_motion_correction=True)

    staged = CNMFe(_params())
    mc = staged.fit_mc(movie)                 # in-memory numpy corrected movie
    staged.fit_extract(mc, evaluate=False)
    staged.evaluate()

    np.testing.assert_allclose(full.shifts, staged.shifts, rtol=0, atol=0)
    _assert_same_model(full, staged)


def test_evaluate_is_reproducible_after_reload(synth_small, tmp_path):
    """evaluate() depends only on A + sn, so it reruns identically post-load."""
    movie = synth_small["movie"]
    model = CNMFe(_params()).fit(movie, do_motion_correction=False)
    mask0 = model.accepted_mask.copy()

    model.save(tmp_path / "results")
    reloaded = CNMFe.load(tmp_path / "results")
    reloaded.evaluate()

    np.testing.assert_array_equal(mask0, reloaded.accepted_mask)
