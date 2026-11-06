"""The ring background must follow the published CNMF-E model (Zhou et al. 2018).

Three properties, each of which the pipeline once got wrong:

* the ring is applied to neuron-free data, ``B = W (Y - A C - b0)`` (Eq. 8, 11);
* ``W`` is re-estimated inside the loop, not solved once from the init traces;
* the ring radius is sized from the neuron size ``l ~ 4 sigma``, not ``2 sigma + 1``.

The formula itself is pinned in ``test_background.py``; these tests pin how the
pipeline drives it, plus the end-to-end effect on the realistic simulator.
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import minicnmfe.pipeline as P  # noqa: E402
from minicnmfe.pipeline import CNMFe, CNMFeParams  # noqa: E402
from tests.miniscope_simulator import make_miniscope_movie  # noqa: E402

N_ITER = 2


def _params(**kw) -> CNMFeParams:
    base = dict(sigma=3.0, n_iter_main=N_ITER, n_jobs=1, init_patches=False,
                init_stride=1, seed_method="threshold", min_corr=0.7, min_pnr=5.0)
    base.update(kw)
    return CNMFeParams(**base)


def _fit(movie, params) -> CNMFe:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CNMFe(params).fit_extract(movie, evaluate=False)


class _Spies:
    """Record what ``fit_extract`` hands to the background stages."""

    def __init__(self):
        self.compute_W = []        # (radius, W_cached is None, A, W_returned)
        self.subtractors = []      # kwargs of each BackgroundSubtractor
        self.global_bg = []        # (A argument, W argument)
        self._o_cw = P.compute_W
        self._o_bs = P.BackgroundSubtractor
        self._o_gb = P._fit_global_bg_rank

    def cw(self, Y, A, C, dims, radius, *a, **k):
        out = self._o_cw(Y, A, C, dims, radius, *a, **k)
        self.compute_W.append((radius, k.get("W_cached") is None, A, out[0]))
        return out

    def bs(self, *a, **k):
        self.subtractors.append(k)
        return self._o_bs(*a, **k)

    def gb(self, Y, A, C, W, b0, rank, **k):
        self.global_bg.append((A, W))
        return self._o_gb(Y, A, C, W, b0, rank, **k)

    def run(self, movie, params):
        with mock.patch.multiple(P, compute_W=self.cw, BackgroundSubtractor=self.bs,
                                 _fit_global_bg_rank=self.gb):
            return _fit(movie, params)


@pytest.fixture(scope="module")
def movie(request):
    from tests.conftest import make_synthetic_movie
    return make_synthetic_movie(n_neurons=4, dims=(48, 48), T=240, seed=3)["movie"]


class TestRingDriving:
    def test_W_is_refit_every_iteration_by_default(self, movie):
        s = _Spies()
        s.run(movie, _params())
        assert sum(fresh for _, fresh, _, _ in s.compute_W) == N_ITER + 1

    def test_ring_fit_once_solves_W_once(self, movie):
        s = _Spies()
        s.run(movie, _params(ring_fit_once=True))
        assert sum(fresh for _, fresh, _, _ in s.compute_W) == 1
        assert len(s.compute_W) == N_ITER + 1     # b0 is still refreshed

    def test_ring_radius_is_sized_from_4_sigma(self, movie):
        s = _Spies()
        s.run(movie, _params(sigma=3.0, ring_size_factor=1.5))
        assert s.compute_W[0][0] == pytest.approx(1.5 * (4 * 3.0 + 1))

    def test_subtractor_gets_the_A_C_that_W_was_fit_with(self, movie):
        s = _Spies()
        s.run(movie, _params())
        assert s.subtractors, "pipeline built no BackgroundSubtractor"
        assert all(k.get("A") is not None and k.get("C") is not None
                   for k in s.subtractors)
        # Iteration 0 merges between compute_W and the subtractor, so the term
        # must come from the fit, not from the current A.
        assert s.subtractors[0]["A"] is s.compute_W[0][2]

    def test_C_snapshot_survives_the_temporal_update(self, movie):
        """The term must keep the traces ``W`` was fit with, unchanged by the
        spatial/temporal updates that run while the subtractor is alive."""
        s = _Spies()
        seen = []
        o_ut = P.update_temporal

        def ut(Y_bg, *a, **k):
            before = None if Y_bg.C_bg is None else Y_bg.C_bg.copy()
            out = o_ut(Y_bg, *a, **k)
            seen.append(before is None or np.array_equal(before, Y_bg.C_bg))
            return out

        with mock.patch.object(P, "update_temporal", ut):
            s.run(movie, _params())
        assert seen and all(seen)

    def test_raw_flag_drops_the_neuron_term(self, movie):
        s = _Spies()
        s.run(movie, _params(ring_apply_to_raw=True))
        assert all(k.get("A") is None and k.get("C") is None for k in s.subtractors)

    def test_global_background_is_fit_on_the_eq8_residual(self, movie):
        """With Eq. 8 the residual is ``(I-W)(Y-b0-AC) - b_f f``, so the global
        term must see ``(I-W)A``, not ``A``."""
        s = _Spies()
        s.run(movie, _params(global_bg_rank=1))
        A_arg, W_arg = s.global_bg[0]
        A_fit = s.compute_W[0][2]
        want = (A_fit - W_arg @ A_fit).toarray()
        np.testing.assert_allclose(A_arg.toarray(), want, atol=1e-6)

    def test_raw_flag_fits_global_background_on_A(self, movie):
        s = _Spies()
        s.run(movie, _params(global_bg_rank=1, ring_apply_to_raw=True))
        A_arg, _ = s.global_bg[0]
        assert A_arg is s.compute_W[0][2]


class TestRingWarnings:
    def test_raw_flag_warns_deprecated(self, movie):
        with pytest.warns(FutureWarning, match="deprecated"):
            CNMFe(_params(ring_apply_to_raw=True)).fit_extract(movie, evaluate=False)

    def test_fit_once_warns_but_is_not_deprecated(self, movie):
        with pytest.warns(UserWarning, match="ring_fit_once") as rec:
            CNMFe(_params(ring_fit_once=True)).fit_extract(movie, evaluate=False)
        assert not any(issubclass(w.category, FutureWarning)
                       and "ring_fit_once" in str(w.message) for w in rec)

    def test_no_ring_warning_at_defaults(self, movie):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            CNMFe(_params()).fit_extract(movie, evaluate=False)
        assert not [w for w in rec if "ring_" in str(w.message)]


def _zc(X):
    X = X - X.mean(0)
    return X / (np.linalg.norm(X, axis=0) + 1e-12)


def _score(model, sim):
    """Mean spatial r and far-pair trace r over reciprocal-best matched cells."""
    A = model.A.toarray()
    Y = np.asarray(model.C) + np.asarray(model.YrA)
    At = np.asarray(sim["A_true"], dtype=np.float64)
    cen = np.asarray(sim["centers"], dtype=float)
    R = _zc(A).T @ _zc(At)
    bt, bk = R.argmax(1), R.argmax(0)
    M = [(k, bt[k]) for k in range(A.shape[1]) if bk[bt[k]] == k and R[k, bt[k]] > 0.5]
    ks, ts = [k for k, _ in M], [t for _, t in M]
    pair_r = np.corrcoef(Y[ks])
    dist = np.linalg.norm(cen[ts][:, None] - cen[ts][None], axis=2)
    iu = np.triu_indices(len(M), 1)
    return float(np.mean([R[k, t] for k, t in M])), float(pair_r[iu][dist[iu] >= 48].mean())


class TestEndToEnd:
    """Effect on the realistic simulator, against ground truth."""

    @pytest.fixture(scope="class")
    def sim(self):
        return make_miniscope_movie(n_neurons=60, dims=(128, 128), T=2000, seed=1)

    @pytest.fixture(scope="class")
    def base(self):
        return CNMFeParams(sigma=3.0, n_iter_main=2, n_jobs=4, init_patches=False,
                           frame_rate_hz=20.0, decay_time_ms=180.0, g_prior_weight=0.6,
                           auto_eval_snr_amp_thr=0.0)

    @pytest.mark.parametrize("rank", [0, 1])
    def test_published_model_recovers_footprints_without_shared_signal(self, sim, base, rank):
        movie = np.asarray(sim["movie"], dtype=np.float32)
        old_factor = 1.5 * (2 * 3.0 + 1) / (4 * 3.0 + 1)
        legacy = replace(base, global_bg_rank=rank, ring_apply_to_raw=True,
                         ring_fit_once=True, ring_size_factor=old_factor)
        spatial_old, _ = _score(_fit(movie, legacy), sim)
        spatial_new, far_new = _score(_fit(movie, replace(base, global_bg_rank=rank)), sim)

        assert spatial_new > spatial_old + 0.05
        # A rank-1 term fit on the wrong residual pushes shared signal into every
        # trace, which shows up as correlation between cells far apart.
        assert abs(far_new) < 0.02
