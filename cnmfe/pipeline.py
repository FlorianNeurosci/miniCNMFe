"""High-level CNMFe pipeline.

Orchestrates motion correction → preprocessing → initialization →
ring background → iterative spatial/temporal refinement.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

from cnmfe._utils import make_2d
from cnmfe.background import BackgroundSubtractor, compute_W
from cnmfe.evaluate import auto_evaluate_components
from cnmfe.initialization import greedy_corr_pnr
from cnmfe.io import transpose_zarr_to_pixel_major
from cnmfe.merging import merge_components
from cnmfe.motion_correction import motion_correction_rigid
from cnmfe.preprocess import correlation_pnr, estimate_noise
from cnmfe.spatial import update_spatial
from cnmfe.temporal import estimate_ar_params, update_temporal


def _zarr_store_path(arr) -> "str | None":
    """Best-effort: return the on-disk path backing ``arr`` (a zarr.Array).

    Returns None if the array is in-memory or has no resolvable path —
    callers should fall back to materialisation in that case.
    """
    store = getattr(arr, "store", None)
    if store is None:
        return None
    # zarr v3 LocalStore exposes .root; FsspecStore exposes .path.
    for attr in ("root", "path"):
        val = getattr(store, attr, None)
        if val is not None:
            return str(val)
    return None


def _Y_times_vec(Y_flat, v, axis: int) -> np.ndarray:
    """Compute ``Y_flat @ v`` (axis=1, output shape (H*W,)) or ``Y_flat.T @ v``
    (axis=0, output shape (T,)) without materialising a zarr ``Y_flat`` in full.
    """
    if isinstance(Y_flat, np.ndarray):
        return (Y_flat @ v if axis == 1 else Y_flat.T @ v).astype(np.float32)
    n_pix = int(Y_flat.shape[0])
    T = int(Y_flat.shape[1])
    batch = 4096
    if axis == 1:
        out = np.zeros(n_pix, dtype=np.float32)
        for s in range(0, n_pix, batch):
            e = min(s + batch, n_pix)
            out[s:e] = np.asarray(Y_flat[s:e], dtype=np.float32) @ v
        return out
    out = np.zeros(T, dtype=np.float32)
    for s in range(0, n_pix, batch):
        e = min(s + batch, n_pix)
        out += np.asarray(Y_flat[s:e], dtype=np.float32).T @ v[s:e]
    return out


def _fit_global_bg_rank1(
    Y_flat,
    A: "sp.csc_matrix",
    C: np.ndarray,
    W_mat: "sp.csr_matrix",
    b0: np.ndarray,
    bf: "np.ndarray | None",
    f: "np.ndarray | None",
    n_iter: int = 2,
) -> "tuple[np.ndarray, np.ndarray]":
    """Alternating LS for the rank-1 temporal background ``b_f · f(t)``.

    Fits ``R ≈ b_f · f(t)`` where
        R = (I − W)(Y − b0) − A C
    is the ring-subtracted, neural-signal-subtracted residual. ``bf`` and ``f``
    are unconstrained (drift can have either sign per pixel relative to b0);
    standard CNMF non-negativity would force a monotonic-only background and
    is not appropriate when modelling a *deviation* from b0.

    On first call pass ``bf=None, f=None`` for init via the per-frame mean of
    ``Y_bg``; subsequent calls warm-start from the previous values.

    Algebra (avoids materialising R):

        u  = (I − W^T) bf
        f  = (Y^T u  −  u·b0  −  (bf·A) C) / (bf·bf)

        v  = Y f − b0·(f.sum())
        bf = ((I − W) v  −  A (C f)) / (f·f)

    All terms use sparse matvec + dense matvec; cost per iteration is
    O(H·W·T) (the ``Y @ f`` and ``Y.T @ u`` calls) plus O(H·W·K + K·T) for
    the A/C terms. Streams cleanly on zarr Y_flat via ``_Y_times_vec``.
    """
    n_pix = int(Y_flat.shape[0])
    T = int(Y_flat.shape[1])
    A_csr = A.tocsr() if sp.issparse(A) else sp.csr_matrix(A)

    if bf is None or f is None:
        # Init: f_0 = per-frame mean of Y_bg (drift signal).
        # mean_p (I-W)(Y-b0) = (1/HW) · (1 − W^T 1)^T · (Y − b0)
        ones = np.ones(n_pix, dtype=np.float32) / float(n_pix)
        u0 = ones - (W_mat.T @ ones)
        f = _Y_times_vec(Y_flat, u0, axis=0) - float(u0 @ b0)
        f = (f - f.mean()).astype(np.float32)
        if not np.any(f):
            return (np.zeros(n_pix, dtype=np.float32),
                    np.zeros(T, dtype=np.float32))
        # bf_0: LS projection of Y_bg onto f.
        # bf = ((I-W)(Y - b0) f) / (f·f) — neural term skipped on init.
        v = _Y_times_vec(Y_flat, f, axis=1) - b0 * float(f.sum())
        bf = (v - (W_mat @ v)).astype(np.float32) / max(float(f @ f), 1e-10)

    for _ in range(n_iter):
        # Update f given bf
        u = (bf - (W_mat.T @ bf)).astype(np.float32)
        yTu = _Y_times_vec(Y_flat, u, axis=0)
        ub0 = float(u @ b0)
        if A_csr.shape[1] > 0:
            bfA = np.asarray(bf @ A_csr, dtype=np.float32)
            bfA_C = bfA @ C
        else:
            bfA_C = np.zeros(T, dtype=np.float32)
        denom_f = max(float(bf @ bf), 1e-10)
        f = ((yTu - ub0 - bfA_C) / denom_f).astype(np.float32)

        # Update bf given f
        v = _Y_times_vec(Y_flat, f, axis=1) - b0 * float(f.sum())
        Iv_minus_Wv = v - (W_mat @ v)
        if A_csr.shape[1] > 0:
            ACf = np.asarray(A_csr @ (C @ f), dtype=np.float32)
        else:
            ACf = np.zeros(n_pix, dtype=np.float32)
        denom_b = max(float(f @ f), 1e-10)
        bf = ((Iv_minus_Wv - ACf) / denom_b).astype(np.float32)

    return bf, f

if TYPE_CHECKING:
    import zarr


@dataclass
class CNMFeParams:
    """All CNMFe algorithm parameters.

    Deviations from standard CNMF-E (Zhou et al. 2018) are tagged
    ``[NON-STANDARD]`` next to the field. With every ``[NON-STANDARD]``
    *algorithm* flag at its default, the math matches the published CNMF-E.
    Fields tagged ``[NON-STANDARD speed]`` are speed/memory trade-offs
    (subsampling, parallelism, etc.) that converge to the standard
    algorithm at their slow-but-faithful settings (``tsub=1``,
    ``stride=1``, ``skip_first_deconv=False``, ``sample_frames=T``).
    Implementation mechanics like ``n_jobs``, ``device``, and ``mc_*``
    are not tagged.
    """

    # --- Motion correction ---
    max_shift: tuple[int, int] = (20, 20)
    upsample_factor: int = 10
    mc_n_iter: int = 1                 # number of rigid MC passes (CaImAn default is 1)
    mc_gSig_filt: float | None = None  # 1p high-pass sigma; set ≈ sigma to enable
    mc_batch_size: int = 200           # frames per streaming/parallel batch in MC
    mc_template_max_frames: int = 2000 # cap on frames sampled to build the MC template
    mc_output_chunk_t: "int | None" = None   # output zarr chunk on time axis (None = match source)
    mc_output_dtype: str = "float32"   # output zarr dtype for the corrected movie

    # --- Spatial filtering / PSF ---
    sigma: float = 3.0        # Gaussian sigma in pixels (neuron size)
    center_psf: bool = True   # Use center-surround kernel for 1p background rejection

    # --- Initialization (GreedyCorr) ---
    min_corr: float = 0.8
    min_pnr: float = 10.0
    min_pixel: int = 3        # Minimum nonzero pixels in a valid footprint
    border_px: int = 5        # Ignore seeds within this many border pixels
    max_neurons: int | None = None  # Stop early (None = no limit)
    init_min_corr_neuron: float = 0.8         # "Neuron pixel" threshold inside extract_spatial_temporal
    init_max_corr_bg: float = 0.4             # "Background pixel" threshold inside extract_spatial_temporal
    seed_suppress_factor: float = 2.0         # Suppression disk radius after extraction = factor * sigma
    circular_max_dist_factor: float = 2.5     # circular_constraint cutoff = factor * estimated_radius
    # [NON-STANDARD speed] Temporal stride for greedy init (None = auto: max(1, T//5000)).
    init_stride: "int | None" = None
    # [NON-STANDARD speed] Extra stride applied to the *initial* CORR/PNR sweep inside greedy init.
    # CORR/PNR are per-pixel reductions, so a strided slice gives a
    # near-identical seed map at a fraction of the cost. None auto-selects
    # max(1, T_init // 2000) so the sweep sees ~2000 frames regardless of T.
    init_corrpnr_stride: "int | None" = None

    # --- Background (ring model) ---
    ring_size_factor: float = 1.5  # ring radius = ring_size_factor * (2*sigma+1)
    ring_lambda: float = 1e-5      # Ridge regularization for ring regression
    # [NON-STANDARD] Enforce Σ_j W[i, j] = 1 per row of the ring weight matrix
    # (Lagrangian-constrained ridge LS). Cancels spatially-uniform brightness
    # events (LED flicker, uniform photobleaching) exactly in the residual —
    # the unconstrained ring shrinks row sums below 1, so a fraction of every
    # global event leaks into the extracted traces. Standard CNMF-E uses
    # unconstrained ridge LS (i.e. False). Negligible cost (one extra RHS in
    # the same per-pixel solve).
    ring_constrain_sum: bool = False
    # [NON-STANDARD] Add a rank-1 temporal background ``b_f · f(t)`` on top of
    # the ring (CNMF-style ``nb=1``). Captures *spatially-non-uniform* slow
    # drift (vignetting-coupled photobleaching, scope warmup) that the ring
    # cannot — the constrained ring only cancels spatially-uniform modes.
    # ``0`` = standard CNMF-E (ring only). ``1`` = one rank-1 term, fit via
    # alternating non-negative LS in each BCD iteration. Higher ranks are
    # deferred — ``1`` covers the typical bleach pattern.
    global_bg_rank: int = 0

    # --- Spatial update ---
    dilation_radius: int = 3
    spatial_max_thr: float = 0.1    # Zero footprint pixels below this fraction of the peak

    # --- Temporal update / deconvolution ---
    ar_order: int = 1
    global_ar: bool = True  # True = one g estimated from pooled C_raw; False = per-neuron
    n_iter_temporal: int = 2
    # [NON-STANDARD] Polynomial order subtracted from each trace before the
    # Yule-Walker autocorrelation that estimates `g`. A slow bleach trend
    # has lag-1 autocorrelation ≈ 1 and, when not detrended, pushes `g`
    # toward 1.0 — OASIS then explains the whole trace as one decaying tail
    # and the deconvolved C collapses to ~0 after the first transient.
    # Default 0 matches standard CNMF-E (mean-only centring). Set ≥1 when
    # traces carry a slow bleach component (order 2 absorbs typical
    # exponential photobleaching).
    ar_detrend_order: int = 0
    # [NON-STANDARD] Polynomial order subtracted from each component's trace
    # `YrA[:,k]/nA[k] + C[k]` immediately before OASIS, inside the BCD loop.
    # Standard CNMF-E feeds OASIS the raw projection and lets OASIS fit a
    # single-scalar median baseline; that cannot track a long slow drift,
    # so deconvolved spikes get suppressed. Set ≥1 to opt in. Default 0:
    # on activity-rich recordings the least-squares polynomial gets pulled
    # upward by the spike envelope, depressing the inter-spike baseline
    # below truth so OASIS reconstructs inflated transients, and the BCD
    # spatial update then propagates that distortion into A — visible as
    # overshoot in `C + YrA` vs ground truth.
    temporal_detrend_order: int = 0

    # --- Merging ---
    merge_thr_corr: float = 0.85
    merge_thr_overlap: float = 0.5  # Min Jaccard spatial overlap to consider merging
    # [NON-STANDARD] Centre-distance fallback for merging: pairs whose footprints
    # have ended up disjoint after threshold_footprint can still be merged if
    # their centres are within `factor * sigma`. Standard CNMF-E merges on spatial
    # overlap only. Disable by setting factor=0.
    merge_centre_dist_factor: float = 2.0

    # --- Main loop ---
    n_iter_main: int = 2  # Full spatial + temporal + merge cycles

    # --- Parallelism ---
    n_jobs: int = 1      # Workers for pixel-parallel steps (-1 = all CPUs)
    device: str = "cpu"  # 'cpu' or 'cuda' (requires CuPy + CUDA GPU)

    # --- Statistics sampling ---
    # [NON-STANDARD speed] Max frames used for noise + CORR/PNR (evenly sampled
    # when T > this). Standard CNMF-E uses all frames.
    sample_frames: int = 1000

    # --- Speed / accuracy trade-offs ---
    # [NON-STANDARD speed] Use NNLS (p=0) for the first temporal pass; OASIS on all
    # others. Saves the dominant cost on iteration 0 with no measurable accuracy hit.
    skip_first_deconv: bool = True
    # [NON-STANDARD speed] Temporal subsampling factor for the ring W solve. The b0
    # baseline still uses the full T; only the per-pixel BTB regression sees a strided
    # slice. Standard CNMF-E uses tsub=1.
    bg_tsub: int = 5

    # --- Auto evaluation (post-BCD quality tagging — non-destructive) ---
    # [NON-STANDARD] `min_pixel` above is reused as a hard floor on footprint extent.
    # `auto_eval_snr_amp_thr` is the threshold on the mean-amplitude SNR
    # mean(a^2) / mean(sn_pixel^2) — a scale-invariant check that catches
    # ghost components whose footprint sits at the pixel-noise floor.
    # Components below this score (or below `min_pixel` non-zero pixels) are
    # flagged in ``model.accepted_mask`` but are NOT removed from the model;
    # filter post-hoc with ``model.A[:, model.accepted_mask]`` (and similarly
    # for C, S, YrA, C_raw). Per-component stats live in ``model.eval_info``.
    # Set to 0 to mark every component as accepted on the SNR check.
    # Standard CNMF-E has no comparable post-BCD quality filter.
    auto_eval_snr_amp_thr: float = 3.0

    # --- Serialisation ---------------------------------------------------------

    def to_json(self, path: "str | Path") -> None:
        """Write all parameters to a JSON file."""
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: "str | Path") -> "CNMFeParams":
        """Construct ``CNMFeParams`` from a JSON file written by ``to_json``.

        Tuple fields (e.g. ``max_shift``) come back as lists from JSON; this
        method restores their declared types. Unknown keys are dropped so old
        save dirs remain loadable after fields are added or removed.
        """
        raw = json.loads(Path(path).read_text())
        valid_names = {f.name for f in fields(cls)}
        data = {k: v for k, v in raw.items() if k in valid_names}
        if "max_shift" in data and isinstance(data["max_shift"], list):
            data["max_shift"] = tuple(data["max_shift"])
        return cls(**data)


class CNMFe:
    """Clean CNMFe for 1-photon calcium imaging.

    Example::

        from cnmfe import CNMFe, CNMFeParams
        from cnmfe.io import avi_to_zarr

        movie = avi_to_zarr("recording.avi", "/tmp/movie.zarr")
        model = CNMFe(CNMFeParams(sigma=3.0, min_corr=0.8, min_pnr=10))
        model.fit(movie)

        A = model.A   # (H*W, K) sparse spatial footprints
        C = model.C   # (K, T)   calcium traces
        S = model.S   # (K, T)   spike trains
    """

    def __init__(self, params: CNMFeParams | None = None) -> None:
        self.params = params or CNMFeParams()

        # Results — populated by fit()
        self.A: sp.csc_matrix | None = None
        self.C: np.ndarray | None = None
        self.S: np.ndarray | None = None
        self.C_raw: np.ndarray | None = None
        self.YrA: np.ndarray | None = None    # residual at each footprint; C + YrA = noisy projection
        self.W: sp.csr_matrix | None = None
        self.b0: np.ndarray | None = None
        self.b_f: np.ndarray | None = None     # (H*W,) — rank-1 spatial bg
        self.f: np.ndarray | None = None       # (T,)   — rank-1 temporal bg
        self.sn: np.ndarray | None = None
        self.shifts: np.ndarray | None = None
        self.dims: tuple[int, int] | None = None
        self.g: list[np.ndarray] | None = None    # per-component AR coefs
        self.sn_per_k: np.ndarray | None = None   # per-component noise std
        self.mc_roi: "tuple[slice, slice] | None" = None  # ROI used for shift estimation
        self.accepted_mask: np.ndarray | None = None    # (K,) bool — passed auto-eval
        self.eval_info: dict | None = None              # full dict from auto_evaluate_components

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def C_projected(self) -> np.ndarray:
        """The shape-faithful "noisy projected trace" ``C + YrA``.

        - ``self.C`` is OASIS-deconvolved (clean AR(1) shape); use it for
          spike-event detection or smooth amplitude features where the
          shape constraint ``c[t] >= g·c[t-1]`` is acceptable.
        - ``self.C + self.YrA`` is the residual at each footprint added
          back to ``C``; it has the same shape as the underlying data
          and typically correlates >0.9 with ground truth, vs ~0.6–0.85
          for ``C`` alone. Use it for cross-correlation, regression
          against external signals, or any shape-sensitive analysis.

        Raises:
            RuntimeError: if ``fit()`` has not been called yet.
        """
        if self.C is None or self.YrA is None:
            raise RuntimeError("Model has not been fit; C and YrA are unset.")
        return self.C + self.YrA

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, output_dir: "str | Path") -> None:
        """Persist results + params to ``output_dir`` as standalone files.

        Layout (mirrors ``full_pipeline.py`` so existing analysis scripts
        keep working):

        - ``A.npz``        sparse CSC footprints ``(H*W, K)``
        - ``C.npy``        OASIS-deconvolved traces ``(K, T)``
        - ``S.npy``        spike trains ``(K, T)``
        - ``YrA.npy``      residual; ``C + YrA`` is the noisy projection
        - ``C_raw.npy``    raw init traces (when available)
        - ``sn.npy``       per-pixel noise std ``(H, W)``
        - ``shifts.npy``   per-frame motion shifts ``(T, 2)`` (when available)
        - ``b0.npy``       per-pixel ring-bg baseline (when available)
        - ``W.npz``        sparse CSR ring weights (when available)
        - ``g.npy``        ``(K, ar_order)`` stacked AR coefs (when available)
        - ``sn_per_k.npy`` ``(K,)`` per-component noise std (when available)
        - ``accepted_mask.npy`` ``(K,)`` bool from auto-eval (when available)
        - ``eval_info.npz`` per-component auto-eval stats (when available)
        - ``params.json``  the ``CNMFeParams`` dataclass
        - ``manifest.json`` non-parameter metadata: ``dims``, ``K``, ``T``.

        Raises:
            RuntimeError: if ``fit()`` has not been called yet.
        """
        if self.A is None or self.C is None:
            raise RuntimeError("Cannot save a model that has not been fit.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        sp.save_npz(output_dir / "A.npz", self.A.tocsc())
        np.save(output_dir / "C.npy", self.C)
        np.save(output_dir / "S.npy", self.S)
        np.save(output_dir / "YrA.npy", self.YrA)
        if self.C_raw is not None:
            np.save(output_dir / "C_raw.npy", self.C_raw)
        if self.sn is not None:
            np.save(output_dir / "sn.npy", self.sn)
        if self.shifts is not None:
            np.save(output_dir / "shifts.npy", self.shifts)
        if self.b0 is not None:
            np.save(output_dir / "b0.npy", self.b0)
        if self.b_f is not None:
            np.save(output_dir / "b_f.npy", self.b_f)
        if self.f is not None:
            np.save(output_dir / "f.npy", self.f)
        if self.W is not None:
            sp.save_npz(output_dir / "W.npz", self.W.tocsr())
        if self.g is not None and len(self.g) > 0:
            np.save(output_dir / "g.npy", np.stack(self.g))
        if self.sn_per_k is not None:
            np.save(output_dir / "sn_per_k.npy", self.sn_per_k)
        if self.accepted_mask is not None:
            np.save(output_dir / "accepted_mask.npy", self.accepted_mask)
        if self.eval_info is not None:
            # eval_info has arrays (pixel_count, snr_amp, pixel_pass, snr_pass)
            # plus scalars (min_pixel, snr_amp_thr); npz stores both fine.
            np.savez(output_dir / "eval_info.npz", **self.eval_info)

        self.params.to_json(output_dir / "params.json")

        manifest = {
            "dims": list(self.dims) if self.dims is not None else None,
            "K": int(self.A.shape[1]),
            "T": int(self.C.shape[1]),
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    @classmethod
    def load(cls, output_dir: "str | Path") -> "CNMFe":
        """Reconstruct a ``CNMFe`` from a directory written by ``save``.

        All result attributes (``A``, ``C``, ``S``, ``YrA``, ``sn``, etc.)
        are restored. Optional files (``shifts``, ``b0``, ``W``, ``g``,
        ``sn_per_k``, ``C_raw``, ``accepted_mask``, ``eval_info``) are
        loaded if present; otherwise the corresponding attribute stays
        ``None``.
        """
        output_dir = Path(output_dir)
        if not output_dir.is_dir():
            raise FileNotFoundError(f"Save directory not found: {output_dir}")

        params = CNMFeParams.from_json(output_dir / "params.json")
        model = cls(params)

        manifest = json.loads((output_dir / "manifest.json").read_text())
        if manifest.get("dims") is not None:
            model.dims = tuple(manifest["dims"])

        model.A = sp.load_npz(output_dir / "A.npz").tocsc()
        model.C = np.load(output_dir / "C.npy")
        model.S = np.load(output_dir / "S.npy")
        model.YrA = np.load(output_dir / "YrA.npy")

        if (output_dir / "C_raw.npy").exists():
            model.C_raw = np.load(output_dir / "C_raw.npy")
        if (output_dir / "sn.npy").exists():
            model.sn = np.load(output_dir / "sn.npy")
        if (output_dir / "shifts.npy").exists():
            model.shifts = np.load(output_dir / "shifts.npy")
        if (output_dir / "b0.npy").exists():
            model.b0 = np.load(output_dir / "b0.npy")
        if (output_dir / "b_f.npy").exists():
            model.b_f = np.load(output_dir / "b_f.npy")
        if (output_dir / "f.npy").exists():
            model.f = np.load(output_dir / "f.npy")
        if (output_dir / "W.npz").exists():
            model.W = sp.load_npz(output_dir / "W.npz").tocsr()
        if (output_dir / "g.npy").exists():
            g_arr = np.load(output_dir / "g.npy")
            model.g = [g_arr[k] for k in range(g_arr.shape[0])]
        if (output_dir / "sn_per_k.npy").exists():
            model.sn_per_k = np.load(output_dir / "sn_per_k.npy")
        if (output_dir / "accepted_mask.npy").exists():
            model.accepted_mask = np.load(output_dir / "accepted_mask.npy")
        if (output_dir / "eval_info.npz").exists():
            with np.load(output_dir / "eval_info.npz") as f:
                model.eval_info = {
                    k: f[k].item() if f[k].ndim == 0 else f[k] for k in f.files
                }

        return model

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def fit_mc(
        self,
        movie: "zarr.Array | np.ndarray",
        output_dir: str | Path | None = None,
    ) -> "zarr.Array | np.ndarray":
        """Run only the motion correction step.

        Stores self.shifts and self.dims.

        - If ``output_dir`` is given **or** the input is a zarr.Array, runs the
          streaming MC path: reads/writes batches of frames, peak RAM is
          ``(mc_batch_size + mc_template_max_frames) * H * W * 4`` bytes
          regardless of T. The corrected movie is written to
          ``<output_dir>/mc.zarr`` and the zarr handle is returned.
        - Otherwise (numpy input, no output_dir): returns a float32 numpy array
          in memory. Per-frame work is parallelized over ``params.n_jobs``.

        Call ``model.fit(mc, do_motion_correction=False)`` afterward to run
        extraction without re-running correction.

        Args:
            movie: Input movie, shape (T, H, W). zarr or numpy array.
            output_dir: If given, write ``mc.zarr`` here and return zarr handle.

        Returns:
            Corrected movie — zarr handle (if output_dir given / zarr input)
            or np.ndarray (if numpy input + no output_dir).
        """
        p = self.params

        # Read shape without materializing the full movie when it's a zarr.
        T, H, W = int(movie.shape[0]), int(movie.shape[1]), int(movie.shape[2])
        self.dims = (H, W)

        output_path: "Path | None" = None
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "mc.zarr"

        corrected, self.shifts = motion_correction_rigid(
            movie,
            output_path=output_path,
            max_shift=p.max_shift,
            gSig_filt=p.mc_gSig_filt,
            upsample_factor=p.upsample_factor,
            niter_rig=p.mc_n_iter,
            batch_size=p.mc_batch_size,
            n_jobs=p.n_jobs,
            template_max_frames=p.mc_template_max_frames,
            output_chunk_t=p.mc_output_chunk_t,
            output_dtype=p.mc_output_dtype,
        )
        return corrected

    def fit(
        self,
        movie: "zarr.Array | np.ndarray",
        do_motion_correction: bool = True,
        output_dir: str | Path | None = None,
        Y_flat_zarr: "zarr.Array | None" = None,
    ) -> "CNMFe":
        """Run the full CNMFe pipeline on a (T, H, W) movie.

        Args:
            movie: Input movie. zarr.Array or numpy array, shape (T, H, W).
                Always required — used for motion correction (if enabled),
                noise estimation, and greedy init (the latter on a strided
                sample so the 3D movie need not fully materialise).
            do_motion_correction: Run rigid motion correction before extraction.
            output_dir: If given, save motion-corrected movie and intermediate
                results here as zarr stores.
            Y_flat_zarr: Optional pixel-major ``(H*W, T)`` zarr produced by
                ``transpose_zarr_to_pixel_major``. When provided, extraction
                (compute_W, BackgroundSubtractor, update_spatial,
                update_temporal) runs directly against this on-disk array
                so the full ``(H*W, T)`` movie is never held in RAM.
                ``movie`` must be a zarr.Array in this mode (we need
                strided 3D access for init); it is **not** materialised.
                Shape must match the spatial dims of ``movie``.

        Returns:
            self (for chaining).
        """
        p = self.params

        # --- Auto-derive Y_flat_zarr from a zarr movie + output_dir -----------
        # Passing a zarr.Array `movie` without Y_flat_zarr used to silently
        # asarray() the whole thing into RAM (86 GB for 60k×600×600). When
        # the user also gives an `output_dir` we know where to put a derived
        # pixel-major store: transpose once (idempotent — skip_if_exists),
        # then route through the existing streaming branch below.
        auto_derived_y_flat = False
        if Y_flat_zarr is None and not do_motion_correction:
            try:
                import zarr as _zarr_pkg
                is_zarr_movie = isinstance(movie, _zarr_pkg.Array)
            except ImportError:
                is_zarr_movie = False
            if is_zarr_movie and output_dir is not None:
                src_path = _zarr_store_path(movie)
                if src_path is None:
                    raise ValueError(
                        "Cannot auto-derive Y_flat_zarr: the input zarr has no "
                        "resolvable on-disk path. Pass Y_flat_zarr= explicitly "
                        "(e.g. via cnmfe.io.transpose_zarr_to_pixel_major)."
                    )
                pixel_path = Path(output_dir) / "Y_flat_pixel.zarr"
                pixel_path.parent.mkdir(parents=True, exist_ok=True)
                Y_flat_zarr = transpose_zarr_to_pixel_major(
                    src_path, pixel_path, verbose=True,
                )
                auto_derived_y_flat = True

        # --- Streaming mode (Y_flat_zarr supplied or auto-derived) ------------
        streaming = Y_flat_zarr is not None
        if streaming:
            try:
                import zarr as _zarr_pkg
                if not isinstance(movie, _zarr_pkg.Array):
                    raise TypeError(
                        "Y_flat_zarr requires `movie` to be a zarr.Array (3D) "
                        "so the strided init sample can be read lazily."
                    )
            except ImportError as exc:
                raise RuntimeError("zarr is required for Y_flat_zarr") from exc
            if do_motion_correction:
                raise ValueError(
                    "Y_flat_zarr is the already-corrected movie; pass "
                    "do_motion_correction=False (and run fit_mc separately if needed)."
                )
            T, H, W = int(movie.shape[0]), int(movie.shape[1]), int(movie.shape[2])
            if Y_flat_zarr.shape != (H * W, T):
                raise ValueError(
                    f"Y_flat_zarr shape {Y_flat_zarr.shape} must equal "
                    f"(H*W, T) = ({H * W}, {T}) derived from movie {movie.shape}."
                )
            movie_arr = movie    # zarr handle; never asarray'd in full
        else:
            movie_arr = np.asarray(movie, dtype=np.float32)
            T, H, W = movie_arr.shape

        dims = (H, W)
        self.dims = dims

        # --- Log effective config so users can see what they got ---
        if streaming:
            stream_str = f"yes (Y_flat_zarr={'auto-derived' if auto_derived_y_flat else 'user-supplied'})"
        else:
            stream_str = "no (movie materialised in RAM)"
        print(
            f"Extraction config: n_jobs={p.n_jobs} device={p.device} "
            f"T={T} H={H} W={W} streaming={stream_str}"
        )
        if p.n_jobs == 1:
            print(
                "  Note: n_jobs=1 (serial). "
                "Set CNMFeParams(n_jobs=-1) to use all CPU cores."
            )

        # --- Step 1: Motion correction ---
        if do_motion_correction:
            # In-memory path here: extraction steps below need the full movie
            # in RAM anyway, so streaming-to-zarr would only add disk IO.
            # Use fit_mc(zarr, output_dir=...) directly when the input is too
            # big to materialize.
            movie_arr, self.shifts = motion_correction_rigid(
                movie_arr,
                max_shift=p.max_shift,
                gSig_filt=p.mc_gSig_filt,
                upsample_factor=p.upsample_factor,
                niter_rig=p.mc_n_iter,
                batch_size=p.mc_batch_size,
                n_jobs=p.n_jobs,
                template_max_frames=p.mc_template_max_frames,
            )

        # --- Steps 2-3: sample frames for statistics if T > sample_frames ---
        # T can come from a zarr handle (streaming) or a numpy array (in-memory).
        T = int(movie_arr.shape[0])
        stride = max(1, T // p.sample_frames)
        if stride > 1:
            t_idx = np.arange(0, T, stride)
            stats_movie = movie_arr[t_idx]  # advanced indexing already copies
        else:
            stats_movie = movie_arr

        # --- Step 2: Noise estimation ---
        print("Estimating noise...")
        self.sn = estimate_noise(stats_movie)   # (H, W)

        # # --- Step 3: Summary images ---
        # print("Computing CORR and PNR images...")
        # cn, pnr = correlation_pnr(
        #     stats_movie, sigma=p.sigma, center_psf=p.center_psf,
        #     n_jobs=p.n_jobs, device=p.device,
        # )

        # --- Step 4: Initialization ---
        # Greedy init allocates two full (T, H, W) float32 copies inside
        # (`data_filtered` after PSF + `data_raw`). On a 10k × 600 × 600
        # movie that is ~28 GB of transient overhead. Running init on a
        # strided sample cuts this by `stride`. The footprints A are
        # spatial — independent of T — so spatial recovery is unaffected;
        # the temporal traces are then re-projected at full T below.
        init_stride = p.init_stride
        if init_stride is None:
            init_stride = max(1, T // 5000)

        if init_stride > 1:
            init_movie = movie_arr[::init_stride]
        else:
            init_movie = movie_arr
        T_init = int(init_movie.shape[0])

        # Resolve the secondary stride for the initial CORR/PNR sweep.
        # Independent of init_stride: the greedy extraction loop reads the
        # full T_init array, but the global CORR/PNR map only needs a
        # representative slice. Default cap ~2000 frames.
        corrpnr_stride = p.init_corrpnr_stride
        if corrpnr_stride is None:
            corrpnr_stride = max(1, T_init // 2000)

        if init_stride > 1 or corrpnr_stride > 1:
            print(
                f"Running greedy CORR-PNR initialization "
                f"(init_stride={init_stride}, T_init={T_init}; "
                f"corrpnr_stride={corrpnr_stride}, T_corrpnr={T_init // corrpnr_stride})..."
            )
        else:
            print("Running greedy CORR-PNR initialization...")

        A, _, _, centers = greedy_corr_pnr(
            init_movie,
            sigma=p.sigma,
            min_corr=p.min_corr,
            min_pnr=p.min_pnr,
            max_neurons=p.max_neurons,
            min_pixel=p.min_pixel,
            border_px=p.border_px,
            ar_order=p.ar_order,
            n_jobs=p.n_jobs,
            device=p.device,
            min_corr_neuron=p.init_min_corr_neuron,
            max_corr_bg=p.init_max_corr_bg,
            seed_suppress_factor=p.seed_suppress_factor,
            circular_max_dist_factor=p.circular_max_dist_factor,
            corrpnr_stride=corrpnr_stride,
        )
        # Free the strided init movie before allocating C_raw at full T.
        if init_stride > 1:
            del init_movie
        print(f"  Found {A.shape[1]} initial components.")

        if A.shape[1] == 0:
            print("No neurons found. Try lowering min_corr or min_pnr.")
            self.A = A
            self.C = np.empty((0, T), dtype=np.float32)
            self.S = np.empty((0, T), dtype=np.float32)
            self.C_raw = np.empty((0, T), dtype=np.float32)
            return self

        # Flatten movie to (H*W, T) for all subsequent steps. In streaming
        # mode the user-supplied pixel-major zarr IS our Y_flat — no
        # materialisation.
        if streaming:
            Y_flat = Y_flat_zarr
        else:
            Y_flat = make_2d(movie_arr)     # (H*W, T) view
        sn_flat = self.sn.ravel()           # (H*W,)

        # Recover full-T temporal traces by projecting the full-resolution
        # movie onto each footprint:
        #     C[k, t] = (Y[:, t] @ A[:, k]) / ||A[:, k]||^2
        # Greedy returned C at the strided resolution (stride > 1) or full T
        # (stride == 1). Re-projecting at full T uniformly handles both.
        AA_init = (A.T @ A).toarray()
        nA_init = np.maximum(np.diag(AA_init), 1e-10).astype(np.float32)
        if isinstance(Y_flat, np.ndarray):
            YA_init = np.asarray(Y_flat.T @ A, dtype=np.float32)      # (T, K)
        else:
            # Streaming Y_flat.T @ A: sum pixel-batched contributions.
            YA_init = np.zeros((T, A.shape[1]), dtype=np.float32)
            A_csr = A.tocsr()
            proj_batch = 4096
            for s in range(0, H * W, proj_batch):
                e = min(s + proj_batch, H * W)
                Y_chunk = np.asarray(Y_flat[s:e], dtype=np.float32)
                YA_init += np.asarray(Y_chunk.T @ A_csr[s:e], dtype=np.float32)
        C_raw = (YA_init / nA_init[None, :]).T.astype(np.float32)     # (K, T)
        C = C_raw.copy()

        # Estimate the AR coefficient `g` ONCE from the pooled raw traces.
        #
        # Per-component Yule-Walker on T~300 traces has high variance (~0.1
        # spread even on clean ground truth) because the lag-1 / lag-2
        # autocorrelations are noisy. Pooling all components into one long
        # concatenated trace gives an effective sample length of K*T and a
        # much more accurate estimate. We assume all neurons share the same
        # calcium indicator dynamics — the common case for one recording.
        #
        # The estimate is persisted and passed into every update_temporal
        # call instead of being re-estimated each iteration, which would
        # re-apply the fudge_factor 0.96 multiplier to already-deconvolved
        # traces and drift g toward 0.
        K_init = A.shape[1]
        g_per_k: list[np.ndarray] = []
        sn_per_k = np.zeros(K_init, dtype=np.float32)

        if p.global_ar:
            try:
                g_global, _ = estimate_ar_params(
                    C_raw.ravel().astype(np.float32),
                    p=p.ar_order,
                    detrend_order=p.ar_detrend_order,
                )
            except Exception:
                g_global = np.array([0.9 ** (1.0 / max(p.ar_order, 1))] * p.ar_order,
                                    dtype=np.float32)
            for k in range(K_init):
                g_per_k.append(g_global.copy())
                try:
                    _, sn_k = estimate_ar_params(
                        C_raw[k], p=p.ar_order,
                        detrend_order=p.ar_detrend_order,
                    )
                except Exception:
                    sn_k = float(np.std(C_raw[k])) if np.std(C_raw[k]) > 0 else 1.0
                sn_per_k[k] = sn_k
        else:
            for k in range(K_init):
                try:
                    g_k, sn_k = estimate_ar_params(
                        C_raw[k], p=p.ar_order,
                        detrend_order=p.ar_detrend_order,
                    )
                except Exception:
                    g_k = np.array([0.9 ** (1.0 / max(p.ar_order, 1))] * p.ar_order,
                                   dtype=np.float32)
                    sn_k = float(np.std(C_raw[k])) if np.std(C_raw[k]) > 0 else 1.0
                g_per_k.append(g_k)
                sn_per_k[k] = sn_k

        # --- Step 5: Initial ring background ---
        ring_radius = p.ring_size_factor * (2 * p.sigma + 1)
        print(f"Fitting ring-model background (radius={ring_radius:.1f}px, tsub={p.bg_tsub})...")
        W_mat, b0 = compute_W(
            Y_flat, A, C, dims, ring_radius,
            lambda_reg=p.ring_lambda, n_jobs=p.n_jobs, device=p.device,
            tsub=p.bg_tsub,
            constrain_sum=p.ring_constrain_sum,
        )

        # --- Step 5b: Rank-1 global background bf · f(t) (opt-in, NON-STANDARD) ---
        bf: np.ndarray | None = None
        f_bg: np.ndarray | None = None
        if p.global_bg_rank == 1:
            print("Fitting rank-1 global background b_f · f(t) (initial)...")
            bf, f_bg = _fit_global_bg_rank1(
                Y_flat, A, C, W_mat, b0, bf=None, f=None, n_iter=2,
            )

        def _cache_after_merge(members_per_group: list[np.ndarray]) -> None:
            """Update g_per_k / sn_per_k after merge_components reorders K."""
            nonlocal g_per_k, sn_per_k
            new_g: list[np.ndarray] = []
            new_sn = np.zeros(len(members_per_group), dtype=np.float32)
            for j, members in enumerate(members_per_group):
                # Inherit from the first (typically strongest-seeded) member.
                # No re-estimation -> no fudge_factor drift.
                m0 = int(members[0])
                new_g.append(g_per_k[m0])
                new_sn[j] = sn_per_k[m0]
            g_per_k = new_g
            sn_per_k = new_sn

        # --- Step 6: Main refinement loop ---
        for iteration in range(p.n_iter_main):
            print(f"Refinement iteration {iteration + 1}/{p.n_iter_main}...")

            # Early merge: catch duplicates from greedy init while their footprints
            # still overlap, before threshold_footprint() in update_spatial separates them.
            if iteration == 0 and A.shape[1] >= 2:
                print("  Pre-merging duplicate seeds...")
                A, C, n_pre_merged, members_per_group = merge_components(
                    A, C_raw,
                    thr_corr=p.merge_thr_corr,
                    thr_overlap=p.merge_thr_overlap,
                    ar_order=p.ar_order,
                    sigma=p.sigma,
                    dims=dims,
                    centre_dist_factor=p.merge_centre_dist_factor,
                )
                _cache_after_merge(members_per_group)
                # Keep C_raw aligned with A's new column order: merge groups
                # become the mean of their members' rows (matches what
                # merge_components does for C).
                C_raw = np.vstack([
                    C_raw[m].mean(axis=0).clip(0) for m in members_per_group
                ]).astype(np.float32)
                if n_pre_merged:
                    print(f"  {A.shape[1]} components ({n_pre_merged} pre-merged).")

            Y_bg = BackgroundSubtractor(Y_flat, W_mat, b0, bf=bf, f=f_bg)  # lazy (H*W, T)

            print("  Updating spatial footprints...")
            A = update_spatial(Y_bg, C, A, sn_flat, dims, p.dilation_radius, p.n_jobs, p.spatial_max_thr)

            # Remove dead components (all-zero footprints)
            nA = np.asarray(A.power(2).sum(axis=0)).ravel()
            alive = nA > 0
            if not alive.all():
                A = A[:, alive]
                C = C[alive]
                C_raw = C_raw[alive]
                alive_idx = np.where(alive)[0]
                g_per_k = [g_per_k[i] for i in alive_idx]
                sn_per_k = sn_per_k[alive_idx]

            if A.shape[1] == 0:
                print("  All components died. Stopping.")
                break

            print("  Updating temporal traces...")
            _deconvolve = (iteration > 0) or (not p.skip_first_deconv)
            C, S, g_per_k, sn_per_k = update_temporal(
                Y_bg, A, C, sn_flat, p.ar_order, p.n_iter_temporal,
                n_jobs=p.n_jobs, device=p.device,
                g_cached=g_per_k, sn_cached=sn_per_k,
                deconvolve=_deconvolve,
                detrend_order=p.temporal_detrend_order,
            )

            print("  Merging correlated components...")
            A, C, n_merged, members_per_group = merge_components(
                A, C,
                thr_corr=p.merge_thr_corr,
                thr_overlap=p.merge_thr_overlap,
                ar_order=p.ar_order,
                sigma=p.sigma,
                dims=dims,
                centre_dist_factor=p.merge_centre_dist_factor,
            )
            _cache_after_merge(members_per_group)
            C_raw = np.vstack([
                C_raw[m].mean(axis=0).clip(0) for m in members_per_group
            ]).astype(np.float32)
            if n_merged:
                C, S, g_per_k, sn_per_k = update_temporal(
                    Y_bg, A, C, sn_flat, p.ar_order, 1,
                    n_jobs=p.n_jobs, device=p.device,
                    g_cached=g_per_k, sn_cached=sn_per_k,
                    deconvolve=True,
                    detrend_order=p.temporal_detrend_order,
                )
            print(f"  {A.shape[1]} components ({n_merged} merged).")

            # Refresh the per-pixel baseline b0 from the refined (A, C).
            # Reuse the ring weight matrix W from the initial solve — the
            # ring's spatial structure is a property of the data, not of A/C,
            # so it remains valid across BCD iterations. Saves the expensive
            # per-pixel BTB solve every iteration (speedup.md Change 2).
            W_mat, b0 = compute_W(
                Y_flat, A, C, dims, ring_radius,
                lambda_reg=p.ring_lambda, n_jobs=p.n_jobs, device=p.device,
                tsub=p.bg_tsub,
                W_cached=W_mat,
                constrain_sum=p.ring_constrain_sum,
            )

            # Refresh the rank-1 global background warm-started from the
            # previous iteration's (bf, f). Two alternating-LS sweeps converge
            # fast since A, C are nearly converged here.
            if p.global_bg_rank == 1:
                bf, f_bg = _fit_global_bg_rank1(
                    Y_flat, A, C, W_mat, b0, bf=bf, f=f_bg, n_iter=2,
                )

        # --- Step 7: Auto-evaluation (informational, non-destructive) ---
        # Tag components with per-component quality results, but do NOT drop
        # them. Combines a pixel-count floor with a scale-invariant
        # mean-amplitude SNR check (mean(a^2)/mean(sn^2)); see
        # cnmfe/evaluate.py for the rationale. The mask catches ghost
        # components born from background-noise seeds under loose init
        # thresholds — their footprints can be wide but sit at the
        # pixel-noise floor.
        #
        # All components are retained on the model. Callers can filter
        # post-hoc via ``model.A[:, model.accepted_mask]`` (and likewise
        # for C, S, YrA, C_raw, g, sn_per_k).
        if A.shape[1] > 0:
            keep, eval_info = auto_evaluate_components(
                A,
                sn_flat=sn_flat,
                min_pixel=p.min_pixel,
                snr_amp_thr=p.auto_eval_snr_amp_thr,
            )
            self.accepted_mask = keep
            self.eval_info = eval_info
            n_drop = int((~keep).sum())
            n_px_fail = int((~eval_info["pixel_pass"]).sum())
            n_snr_fail = int((~eval_info["snr_pass"]).sum())
            print(
                f"Auto-evaluation: {int(keep.sum())}/{A.shape[1]} accepted "
                f"(flagged {n_drop}: {n_px_fail} fail pixel_count<{p.min_pixel}, "
                f"{n_snr_fail} fail snr_amp<{p.auto_eval_snr_amp_thr}). "
                f"All components retained; filter via model.accepted_mask."
            )
        else:
            # BCD ended with no surviving components — skip the final
            # temporal pass + YrA recomputation, return empty arrays.
            print("No components survived refinement.")
            self.A = A
            self.C = np.empty((0, T), dtype=np.float32)
            self.S = np.empty((0, T), dtype=np.float32)
            self.C_raw = C_raw
            self.YrA = np.empty((0, T), dtype=np.float32)
            self.W = W_mat
            self.b0 = b0
            self.b_f = bf
            self.f = f_bg
            self.g = g_per_k
            self.sn_per_k = sn_per_k
            return self

        # Final deconvolution pass to get spike trains
        print("Final temporal update...")
        Y_bg = BackgroundSubtractor(Y_flat, W_mat, b0, bf=bf, f=f_bg)
        C, S, g_per_k, sn_per_k = update_temporal(
            Y_bg, A, C, sn_flat, p.ar_order, p.n_iter_temporal,
            n_jobs=p.n_jobs, device=p.device,
            g_cached=g_per_k, sn_cached=sn_per_k,
            detrend_order=p.temporal_detrend_order,
        )

        # Compute the residual projected onto each footprint:
        #   YrA[k, t] = (a_k . (Y_bg - A @ C)[:, t]) / ||a_k||^2
        # The "noisy projected trace" with the same shape as the underlying
        # data is C + YrA. OASIS-deconvolved C alone correlates only ~0.6
        # with ground truth on synthetic data because the shape constraint
        # c[t] >= g * c[t-1] introduces small spike-timing distortions; the
        # noisy projection preserves shape and typically correlates > 0.9.
        AA_final = (A.T @ A).toarray()
        nA_final = np.maximum(np.diag(AA_final), 1e-10)
        YA_final = Y_bg.project_onto(A)                                   # (T, K)
        crosstalk = AA_final @ C - np.diag(AA_final)[:, None] * C        # (K, T)
        YrA = (YA_final.T - crosstalk) / nA_final[:, None] - C           # (K, T)

        assert C_raw.shape[0] == A.shape[1], (
            f"C_raw/A K mismatch: C_raw has {C_raw.shape[0]} rows, A has {A.shape[1]} cols"
        )
        self.A = A
        self.C = C
        self.S = S
        self.C_raw = C_raw
        self.YrA = YrA
        self.W = W_mat
        self.b0 = b0
        self.b_f = bf
        self.f = f_bg
        self.g = g_per_k
        self.sn_per_k = sn_per_k
        print(f"Done. Extracted {A.shape[1]} neurons.")
        return self
