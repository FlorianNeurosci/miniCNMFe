"""Component quality evaluation for CNMFe.

Post-extraction filter that drops components failing per-component quality
checks. The API returns an info dict so additional checks (temporal SNR,
spatial coherence) can be slotted in without breaking callers.

Reference (algorithmic only): CaImAn estimates.evaluate_components.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def auto_evaluate_components(
    A: sp.csc_matrix,
    sn_flat: np.ndarray,
    min_pixel: int = 1,
    snr_amp_thr: float = 3.0,
    a_norm: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Return ``(keep_mask, info)`` for the components in ``A``.

    Checks (both must pass; a component fails if either is violated):

    1. **Minimum pixel count.** Footprint must have at least ``min_pixel``
       non-zero pixels (a hard floor on extent).
    2. **Mean-amplitude SNR.** Mean squared amplitude over the footprint
       support, divided by mean pixel-noise variance over the same support,
       must exceed ``snr_amp_thr``::

           snr_amp[k] = (||a_k||^2 / npix[k]) / mean(sn_flat[support_k]^2)

       This is a scale-invariant test: a real neuron has mean(a^2) several
       standard deviations above the local pixel-noise variance, while a
       ghost component (born from a background-noise seed under loose init
       thresholds) sits at or near it. At ``snr_amp_thr=3.0`` a real sigma=3
       Gaussian footprint typically scores 10-70 while ghosts score below 2.

    Args:
        A: (H*W, K) sparse spatial footprints (post-threshold_footprint).
        sn_flat: (H*W,) per-pixel noise std (e.g. minicnmfe.preprocess.estimate_noise(...).ravel()).
        min_pixel: Hard floor on the per-component pixel count.
        snr_amp_thr: Threshold on the mean-amplitude SNR (dimensionless).
        a_norm: Optional (K,) original per-component footprint L2 norms. The
            pipeline relabels ``A`` to CaImAn scale (unit-L2-norm footprints,
            amplitude moved into the traces), which would otherwise flatten
            ``||a_k||^2`` to 1 and destroy this discriminator. When supplied,
            ``||a_k||^2`` is taken as ``a_norm[k]**2`` (exact for unit-norm
            footprints) so ``snr_amp`` reproduces the un-normalized value.
            When ``None``, ``||a_k||^2`` is read directly from ``A`` (the
            historical path, correct for un-normalized footprints).

    Returns:
        keep: (K,) bool, ``True`` for components that pass both checks.
        info: dict with keys
            ``'pixel_count'``  — (K,) int, non-zero pixel count;
            ``'snr_amp'``      — (K,) float32, the SNR statistic above;
            ``'pixel_pass'``   — (K,) bool, pixel-count check pass mask;
            ``'snr_pass'``     — (K,) bool, SNR check pass mask;
            ``'min_pixel'``    — int, threshold actually applied;
            ``'snr_amp_thr'``  — float, threshold actually applied.
    """
    A_csc = A.tocsc() if not sp.isspmatrix_csc(A) else A
    K = A_csc.shape[1]

    pixel_count = np.diff(A_csc.indptr).astype(np.int64)
    snr_amp = np.zeros(K, dtype=np.float32)

    sn_sq = np.asarray(sn_flat, dtype=np.float64) ** 2
    a_norm = None if a_norm is None else np.asarray(a_norm, dtype=np.float64)

    for k in range(K):
        start, end = A_csc.indptr[k], A_csc.indptr[k + 1]
        if start == end:
            continue
        rows = A_csc.indices[start:end]
        # ||a_k||^2: from the stored (CaImAn-scale unit-norm) footprint via the
        # cached original norm when available, else directly from the values.
        if a_norm is not None:
            a_sq = float(a_norm[k]) ** 2
        else:
            vals = A_csc.data[start:end].astype(np.float64)
            a_sq = float(np.dot(vals, vals))
        mean_a_sq = a_sq / len(rows)
        mean_sn_sq = float(np.mean(sn_sq[rows]))
        snr_amp[k] = mean_a_sq / max(mean_sn_sq, 1e-12)

    pixel_pass = pixel_count >= int(min_pixel)
    snr_pass = snr_amp >= float(snr_amp_thr)
    keep = pixel_pass & snr_pass

    info = {
        "pixel_count": pixel_count,
        "snr_amp": snr_amp,
        "pixel_pass": pixel_pass,
        "snr_pass": snr_pass,
        "min_pixel": int(min_pixel),
        "snr_amp_thr": float(snr_amp_thr),
    }
    return keep, info
