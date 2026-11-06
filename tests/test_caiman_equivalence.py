"""Our stages against CaImAn's, on identical inputs (needs CaImAn; skipped without it).

Agreement here shows equivalence to CaImAn, not correctness. See
``validation/caiman_swap`` for the harness and the full swap report.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

pytest.importorskip("caiman")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import minicnmfe.pipeline as P  # noqa: E402
from minicnmfe.pipeline import CNMFe, CNMFeParams  # noqa: E402
from tests.miniscope_simulator import make_miniscope_movie  # noqa: E402
from validation.caiman_swap.adapters import STAGES, CaimanStages  # noqa: E402
from validation.caiman_swap.run_swap import aligned_ring_factor, same_input_report  # noqa: E402

SIGMA = 3.0


@pytest.fixture(scope="module")
def movie():
    sim = make_miniscope_movie(n_neurons=20, dims=(96, 96), T=800, seed=1)
    return np.asarray(sim["movie"], dtype=np.float32)


@pytest.fixture(scope="module")
def params():
    return CNMFeParams(sigma=SIGMA, n_iter_main=2, n_jobs=2, init_patches=False, bg_tsub=1,
                       ring_size_factor=aligned_ring_factor(SIGMA), min_corr=0.6, min_pnr=5.0,
                       spatial_thr_method="max", frame_rate_hz=20.0, decay_time_ms=180.0,
                       g_prior_weight=0.6, auto_eval_snr_amp_thr=0.0)


@pytest.fixture(scope="module")
def stages(movie):
    return CaimanStages(movie.shape[1:], sigma=SIGMA, frame_rate_hz=20.0, decay_time_ms=180.0)


@pytest.fixture(scope="module")
def report(movie, params, stages):
    return same_input_report(movie, params, stages)


def test_ring_solve_is_equivalent(report):
    assert report["W_support_jaccard"] == 1.0
    assert report["W_rel_l2"] < 1e-4
    assert report["b0_rel_diff"] < 1e-5


def test_spatial_update_agrees(report):
    assert report["spatial_r_median"] > 0.95


def test_temporal_update_agrees(report):
    assert report["temporal_r_median"] > 0.95


@pytest.mark.parametrize("stage", STAGES)
def test_each_stage_runs_inside_our_loop(movie, params, stages, stage):
    model = CNMFe(params)
    with mock.patch.multiple(P, **stages.patches(stage)):
        model.fit_extract(movie, evaluate=False)
    assert model.A.shape[1] > 0
    assert model.C.shape == (model.A.shape[1], movie.shape[0])
