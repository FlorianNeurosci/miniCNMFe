"""CaImAn's stage functions behind minicnmfe's stage signatures.

``minicnmfe.pipeline`` binds every stage as a module-level name, so a stage can be
replaced from outside with ``unittest.mock.patch.multiple``: our loop keeps running
and CaImAn does the maths for the patched stages. Nothing in ``minicnmfe/`` imports
CaImAn; this directory is the only place that does.

    stages = CaimanStages(dims, sigma=3.0, frame_rate_hz=20.0, decay_time_ms=180.0)
    with mock.patch.multiple(P, **stages.patches("compute_W", "update_temporal")):
        CNMFe(params).fit_extract(movie)

Agreement here means "equivalent to CaImAn", not "correct". Use a stage swap to find
where two implementations differ, then settle who is right against the paper.

Set ``init_patches=False`` in the params when swapping ``greedy_corr_pnr``: patched
init runs in loky workers, which do not see the patch. CaImAn's seeding uses the
params' ``min_corr``/``min_pnr`` on its own corr/PNR scale, so they usually need to be
lower than a value that suits ours (and ``seed_method='mixture'`` ignores them).
"""

from __future__ import annotations

import logging
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")   # CaImAn pulls in TF/CUDA
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import scipy.sparse as sp
from caiman.source_extraction.cnmf import initialization as c_init
from caiman.source_extraction.cnmf import merging as c_merging
from caiman.source_extraction.cnmf import params as c_params
from caiman.source_extraction.cnmf import spatial as c_spatial
from caiman.source_extraction.cnmf import temporal as c_temporal

import minicnmfe.pipeline as P

logging.getLogger("caiman").setLevel(logging.ERROR)

STAGES = ("greedy_corr_pnr", "compute_W", "update_spatial", "update_temporal",
          "merge_components")


class CaimanStages:
    """One FOV's adapters. Ours is C-ordered ``h*W + w``, CaImAn's F-ordered ``h + w*H``."""

    def __init__(self, dims, *, sigma, frame_rate_hz, decay_time_ms, merge_thr=0.85):
        self.dims = tuple(dims)
        self.n_pix = int(np.prod(dims))
        self.perm = np.arange(self.n_pix).reshape(self.dims).ravel(order="F")  # Y_F = Y_C[perm]
        self.inv = np.argsort(self.perm)                                       # Y_C = Y_F[inv]
        g_sig = max(2, int(round(sigma)))
        self.g_sig, self.g_siz = g_sig, 4 * g_sig + 1
        opts = c_params.CNMFParams(params_dict=dict(
            dims=self.dims, fr=frame_rate_hz, decay_time=decay_time_ms / 1000.0,
            method_init="corr_pnr", gSig=(g_sig, g_sig), gSiz=(self.g_siz, self.g_siz),
            p=1, nb=0, only_init=True, center_psf=True, ring_size_factor=1.5,
            method_deconvolution="oasis", low_rank_background=None, normalize_init=False,
            update_background_components=False, ssub_B=1, thr_method="max", maxthr=0.1,
            merge_thr=merge_thr,
        ))
        self.spatial_opts = dict(opts.get_group("spatial"))
        self.spatial_opts.update(n_pixels_per_process=2048, nb=0,
                                 update_background_components=False)
        self.temporal_opts = dict(opts.get_group("temporal"))
        self.temporal_opts.update(nb=0)
        self._dense = (None, None)

    # -- layout -------------------------------------------------------------

    def _A_to_F(self, A):
        return sp.csr_matrix(A)[self.perm].tocsc()

    def _A_to_C(self, A):
        return sp.csr_matrix(A)[self.inv].tocsc()

    def _Y_bg_F(self, Y_bg):
        """Dense F-ordered background-subtracted movie, kept for the current ``Y_bg``."""
        if self._dense[0] is not Y_bg:
            Y = Y_bg.slice(0, Y_bg.shape[0]) if hasattr(Y_bg, "slice") else np.asarray(Y_bg)
            self._dense = (Y_bg, np.ascontiguousarray(Y[self.perm], dtype=np.float32))
        return self._dense[1]

    def _no_bg(self, n_rows=None):
        return (np.zeros((self.n_pix, 0), np.float32) if n_rows is None
                else np.zeros((0, n_rows), np.float32))

    # -- stages, with our signatures ------------------------------------------

    def greedy_corr_pnr(self, movie, sigma, min_corr=0.8, min_pnr=10.0, max_neurons=None,
                        min_pixel=3, border_px=0, **_ignored):
        o = dict(self.temporal_opts)
        o.update(s_min=None, p=min(o["p"], 1))
        A, C, C_raw, _, center = c_init.init_neurons_corr_pnr(
            np.asarray(movie, dtype=np.float32), max_number=max_neurons, gSiz=self.g_siz,
            gSig=self.g_sig, center_psf=True, min_corr=min_corr, min_pnr=min_pnr,
            seed_method="auto", deconvolve_options=o, min_pixel=min_pixel, bd=border_px,
            swap_dim=False,
        )
        A = self._A_to_C(sp.csc_matrix(A, dtype=np.float32)).astype(np.float32)
        return A, np.asarray(C, np.float32), np.asarray(C_raw, np.float32), np.asarray(center).T

    def compute_W(self, Y, A, C, dims, radius, lambda_reg=1e-5, n_jobs=1, device="cpu",
                  tsub=1, W_cached=None, constrain_sum=False):
        if W_cached is not None:                  # the b0-only refresh is not a CaImAn stage
            return _OURS["compute_W"](Y, A, C, dims, radius, lambda_reg=lambda_reg,
                                      n_jobs=n_jobs, device=device, tsub=tsub,
                                      W_cached=W_cached, constrain_sum=constrain_sum)
        W, b0 = c_init.compute_W(
            np.ascontiguousarray(np.asarray(Y)[self.perm]), self._A_to_F(A).toarray(),
            np.asarray(C), dims, radius, ssub=1, tsub=tsub,
        )
        W = sp.csr_matrix(W)[self.inv][:, self.inv].tocsr().astype(np.float32)
        return W, np.asarray(b0, dtype=np.float32).ravel()[self.inv]

    def update_spatial(self, Y_bg, C, A, sn, dims, *_args, **_kwargs):
        B = self._Y_bg_F(Y_bg)
        C = np.asarray(C, dtype=np.float32)
        norm = np.sqrt((C.astype(np.float64) ** 2).sum(1)) + 1e-12
        A_out, _, C_out, _ = c_spatial.update_spatial_components(
            B, C=C, f=self._no_bg(B.shape[1]), A_in=self._A_to_F(A),
            sn=np.asarray(sn)[self.perm], b_in=self._no_bg(), dview=None, dims=dims,
            **self.spatial_opts,
        )
        A_out = self._A_to_C(sp.csc_matrix(A_out)[:, :C_out.shape[0]])
        # CaImAn unit-norms C and may drop components. Our loop keeps its own C and
        # filters empty columns itself, so undo the scaling and restore the columns.
        rows = (np.asarray(C_out, np.float64) @ (C / norm[:, None]).T).argmax(1)
        out = sp.lil_matrix((self.n_pix, C.shape[0]), dtype=np.float32)
        for j, i in enumerate(rows):
            out[:, i] = A_out[:, j] / norm[i]
        return out.tocsc()

    def update_temporal(self, Y_bg, A, C, sn, ar_order=1, n_iter=2, n_jobs=1, device="cpu",
                        g_cached=None, sn_cached=None, deconvolve=True, **_kwargs):
        B = self._Y_bg_F(Y_bg)
        opts = dict(self.temporal_opts)
        opts.update(p=ar_order if deconvolve else 0, ITER=n_iter)
        res = c_temporal.update_temporal_components(
            B, self._A_to_F(A).astype(np.float32), self._no_bg(),
            np.asarray(C, dtype=np.float32), self._no_bg(B.shape[1]),
            dview=None, bl=None, c1=None, sn=None, g=None, **opts,
        )
        C_out, S_out = np.asarray(res[0], np.float32), np.asarray(res[4], np.float32)
        if C_out.shape[0] != C.shape[0]:
            raise RuntimeError("CaImAn's temporal update changed K; the loop cannot follow")
        return C_out, S_out, g_cached, sn_cached

    def merge_components(self, A, C, thr_corr=0.85, **_kwargs):
        K = A.shape[1]
        C = np.asarray(C, dtype=np.float32)
        Y_shape_only = np.broadcast_to(np.float32(0), (self.n_pix, C.shape[1]))  # fast_merge reads only its shape
        res = c_merging.merge_components(
            Y_shape_only, self._A_to_F(A), [], C, None, [], C, [], self.temporal_opts,
            self.spatial_opts, dview=None, thr=thr_corr, mx=np.inf, fast_merge=True,
        )
        A_out, C_out, merged = sp.csc_matrix(res[0]), np.asarray(res[1], np.float32), res[3]
        grouped = np.concatenate(merged).astype(int) if len(merged) else np.array([], int)
        members = [np.array([i]) for i in np.setdiff1d(np.arange(K), grouped)]
        members += [np.asarray(m, dtype=int) for m in merged]
        return self._A_to_C(A_out).astype(np.float32), C_out, len(merged), members

    # -- patching ---------------------------------------------------------------

    def patches(self, *names):
        """Keyword arguments for ``mock.patch.multiple(minicnmfe.pipeline, ...)``."""
        unknown = set(names) - set(STAGES)
        if unknown:
            raise ValueError(f"unknown stages {sorted(unknown)}; choose from {STAGES}")
        return {name: getattr(self, name) for name in names}


_OURS = {name: getattr(P, name) for name in STAGES}
