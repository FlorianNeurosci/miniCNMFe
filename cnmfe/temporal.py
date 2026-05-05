"""Temporal trace update and AR deconvolution.

For each component k, the fluorescence trace C[k] is modeled as:
    C[k, t] = sum_τ g^τ * S[k, t-τ] + baseline
where S is the (non-negative) spike train and g is the AR decay constant.

Deconvolution uses OASIS (Online Active Set method to Infer Spikes):
    - Fast, exact solution for AR(1) and AR(2) models.
    - Implemented via the `oasis-deconvolution` PyPI package if available,
      with a pure-Python AR(1) fallback (PAVA algorithm).

Reference (algorithmic only): CaImAn temporal.py:update_temporal_components (line 64)
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from cnmfe._utils import get_xp, to_numpy


# ---------------------------------------------------------------------------
# AR parameter estimation
# ---------------------------------------------------------------------------

def estimate_ar_params(
    trace: np.ndarray,
    p: int = 1,
    noise_range: tuple[float, float] = (0.25, 0.5),
    fudge_factor: float = 0.96,
    lags: int = 5,
) -> tuple[np.ndarray, float]:
    """Estimate AR(p) decay constants and noise std from a fluorescence trace.

    Algorithm:
    1. Estimate noise via power in high-frequency bins (rfft).
    2. Fit AR(p) by solving the Yule-Walker equations on the autocorrelation.
    3. Apply fudge_factor to prevent over-estimating the decay (slight bias).

    Args:
        trace: (T,) fluorescence trace.
        p: AR model order (1 or 2).
        noise_range: Frequency band [f_low, f_high] * Nyquist for noise estimate.
        fudge_factor: Shrinkage applied to g (< 1 avoids over-estimated decay).
        lags: Number of autocorrelation lags used.

    Returns:
        g: AR coefficients, shape (p,).
        sn: Noise standard deviation.
    """
    T = len(trace)
    # Noise via high-frequency PSD
    Xf = np.fft.rfft(trace)
    freqs = np.fft.rfftfreq(T)
    noise_mask = (freqs >= noise_range[0]) & (freqs <= noise_range[1])
    if noise_mask.sum() == 0:
        noise_mask = freqs >= noise_range[0]
    psd = np.abs(Xf[noise_mask]) ** 2 / T * 2
    sn = float(np.sqrt(np.exp(np.log(psd + 1e-10).mean())))

    # Yule-Walker: build autocorrelation matrix
    trace_centered = trace - trace.mean()
    ac = np.array([
        np.dot(trace_centered[: T - k], trace_centered[k:]) / (T - k)
        for k in range(lags + 1)
    ])
    # R @ g = r  (Toeplitz system)
    R = np.array([[ac[abs(i - j)] for j in range(p)] for i in range(p)])
    r = ac[1 : p + 1]
    try:
        g = np.linalg.solve(R, r)
    except np.linalg.LinAlgError:
        g = np.array([fudge_factor ** (1.0 / max(p, 1))] * p)

    g = g * fudge_factor
    # Clip to (0, 1) for stability
    g = np.clip(g, 0.0, 0.9999)
    return g.astype(np.float32), float(sn)


# ---------------------------------------------------------------------------
# Deconvolution (OASIS with pure-Python AR1 fallback)
# ---------------------------------------------------------------------------

def _oasis_ar1_pava(y: np.ndarray, g: float, sn: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Pure-Python OASIS AR(1) deconvolution using the pool-adjacent violators algorithm.

    Solves:  min_c  ||y - c||^2   s.t.  c[t] >= g * c[t-1],  c[t] >= 0
    which is equivalent to constrained AR(1) deconvolution.

    This is the PAVA (pool-adjacent-violators) implementation for the
    non-negative constrained LS problem.

    Returns:
        c: Denoised calcium trace.
        s: Spike train (s[t] = c[t] - g * c[t-1]).
        bl: Estimated baseline.
    """
    T = len(y)
    # Estimate and subtract baseline
    bl = float(np.median(y))
    y_bl = (y - bl).astype(np.float64)

    # PAVA pools
    # Each pool i represents a segment [start_i, start_i + length_i)
    # where the constrained optimal value is pool_val[i] = max(0, w[i] / v[i])
    pool_start = list(range(T))
    pool_length = [1] * T
    pool_num = [y_bl[t] for t in range(T)]   # numerator  = sum of weighted y
    pool_den = [1.0] * T                      # denominator = sum of weights
    pool_val = [max(0.0, y_bl[t]) for t in range(T)]

    def merge(i: int, j: int) -> None:
        """Merge pool j into pool i."""
        g_pow = g ** pool_length[i]
        pool_num[i] += g_pow * pool_num[j]
        pool_den[i] += (g_pow ** 2) * pool_den[j]
        pool_length[i] += pool_length[j]
        pool_val[i] = max(0.0, pool_num[i] / pool_den[i])

    i = 0
    while i < len(pool_start) - 1:
        # Check if constraint c[t] >= g * c[t-1] is violated between pools i and i+1
        if pool_val[i] * g > pool_val[i + 1]:
            merge(i, i + 1)
            pool_start.pop(i + 1)
            pool_length.pop(i + 1)
            pool_num.pop(i + 1)
            pool_den.pop(i + 1)
            pool_val.pop(i + 1)
            # Re-check previous merge
            if i > 0:
                i -= 1
        else:
            i += 1

    # Reconstruct c
    c = np.zeros(T, dtype=np.float64)
    for k in range(len(pool_start)):
        start = pool_start[k]
        length = pool_length[k]
        val = pool_val[k]
        for t in range(length):
            c[start + t] = val * (g ** t)

    # Spike train
    s = np.zeros(T, dtype=np.float64)
    s[0] = c[0]
    s[1:] = np.maximum(c[1:] - g * c[:-1], 0)

    return c.astype(np.float32), s.astype(np.float32), float(bl)


def deconvolve(
    trace: np.ndarray,
    g: np.ndarray,
    sn: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Deconvolve a fluorescence trace to infer the spike train.

    Uses the `oasis-deconvolution` package (constrained OASIS).
    Falls back to pure-Python PAVA for AR(1) if oasis is not installed.

    Args:
        trace: (T,) raw fluorescence trace.
        g: AR coefficients shape (p,) where p=1 or 2.
        sn: Noise standard deviation.

    Returns:
        c: (T,) denoised calcium trace.
        s: (T,) non-negative spike train.
        bl: Estimated baseline offset.
    """
    trace = ensure_float32_1d(trace)
    p = len(g)

    try:
        if p == 1:
            from oasis.functions import deconvolve as oasis_deconvolve
            c, s, bl, g_out, _ = oasis_deconvolve(
                trace,
                penalty=1,
                g=(float(g[0]),),
                sn=sn,
            )
            return c.astype(np.float32), s.astype(np.float32), float(bl)
        else:
            from oasis.functions import deconvolve as oasis_deconvolve
            c, s, bl, g_out, _ = oasis_deconvolve(
                trace,
                penalty=1,
                g=tuple(float(gi) for gi in g),
                sn=sn,
            )
            return c.astype(np.float32), s.astype(np.float32), float(bl)
    except ImportError:
        # Pure-Python fallback (AR1 only)
        if p == 1:
            return _oasis_ar1_pava(trace, float(g[0]), sn)
        else:
            # Approximate AR2 as AR1 with the dominant time constant
            g1_approx = float(g[0] + g[1] ** 0.5) if len(g) > 1 else float(g[0])
            g1_approx = min(g1_approx, 0.9999)
            return _oasis_ar1_pava(trace, g1_approx, sn)


def ensure_float32_1d(x: np.ndarray) -> np.ndarray:
    """Cast to float32 and ensure 1-D."""
    x = np.asarray(x, dtype=np.float32)
    return x.ravel()


# ---------------------------------------------------------------------------
# Temporal component update
# ---------------------------------------------------------------------------

def _deconvolve_one(trace_k: np.ndarray, ar_order: int) -> tuple[np.ndarray, np.ndarray]:
    """Deconvolve a single trace (module-level so it is picklable)."""
    try:
        g_k, sn_k = estimate_ar_params(trace_k, p=ar_order)
        c_k, s_k, _ = deconvolve(trace_k, g_k, sn_k)
    except Exception:
        c_k = trace_k.clip(0)
        s_k = np.zeros_like(c_k)
    return c_k, s_k


def update_temporal(
    Y_flat: np.ndarray,
    A: sp.csc_matrix,
    C: np.ndarray,
    sn: np.ndarray,
    ar_order: int = 1,
    n_iter: int = 2,
    n_jobs: int = 1,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Refine temporal traces by block coordinate descent.

    For each component k, the residual data after subtracting all other
    components is projected onto A[:,k], then deconvolved with OASIS.

    When n_jobs != 1 a Jacobi (parallel) update is used within each iteration:
    all K traces are deconvolved simultaneously with the residuals from the
    previous iteration. This converges slightly slower than Gauss-Seidel but
    scales linearly with available cores.

    When device='cuda' the initial projections (Y_flat.T @ A and A.T @ A) run
    on GPU. OASIS deconvolution always runs on CPU (no GPU implementation).

    Args:
        Y_flat: (H*W, T) background-subtracted movie.
        A: (H*W, K) sparse spatial footprints.
        C: (K, T) current temporal traces.
        sn: (H*W,) per-pixel noise std.
        ar_order: AR model order for deconvolution.
        n_iter: Number of BCD iterations.
        n_jobs: Number of parallel workers (-1 = all CPUs, 1 = serial Gauss-Seidel).
        device: 'cpu' or 'cuda'. GPU accelerates the (H*W × T) @ (H*W × K) projection.

    Returns:
        C_new: (K, T) updated calcium traces.
        S: (K, T) inferred spike trains.
    """
    K, T = C.shape
    xp = get_xp(device)

    if xp is not np:
        # GPU: convert A to dense for fast matmul (K is small; A is H*W × K)
        A_dense_xp = xp.asarray(A.toarray())
        Y_xp = xp.asarray(Y_flat)
        YA = to_numpy(Y_xp.T @ A_dense_xp)             # (T, K)
        AA = to_numpy(A_dense_xp.T @ A_dense_xp)       # (K, K)
        del Y_xp, A_dense_xp
    else:
        YA = Y_flat.T @ A
        AA = (A.T @ A).toarray()

    nA = np.maximum(np.diag(AA), 1e-10)

    C = C.copy()
    S = np.zeros_like(C)
    YrA = YA - (AA @ C).T

    if n_jobs == 1:
        # Gauss-Seidel: each update immediately uses the latest C[k]
        for _ in range(n_iter):
            for k in range(K):
                trace_k = (YrA[:, k] / nA[k] + C[k]).astype(np.float32)
                c_k, s_k = _deconvolve_one(trace_k, ar_order)
                delta = c_k - C[k]
                YrA -= np.outer(delta, AA[:, k])
                C[k] = c_k
                S[k] = s_k
    else:
        from joblib import Parallel, delayed

        # Jacobi: all K components deconvolved in parallel, then YrA updated
        for _ in range(n_iter):
            traces = [(YrA[:, k] / nA[k] + C[k]).astype(np.float32) for k in range(K)]
            results = Parallel(n_jobs=n_jobs)(
                delayed(_deconvolve_one)(trace_k, ar_order) for trace_k in traces
            )
            for k, (c_k, s_k) in enumerate(results):
                delta = c_k - C[k]
                YrA -= np.outer(delta, AA[:, k])
                C[k] = c_k
                S[k] = s_k

    return C, S
