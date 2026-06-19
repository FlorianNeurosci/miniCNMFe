"""Temporal trace update and AR deconvolution.

For each component k, the fluorescence trace C[k] is modeled as:
    C[k, t] = sum_τ g^τ * S[k, t-τ] + baseline
where S is the (non-negative) spike train and g is the AR decay constant.

Deconvolution uses OASIS (Online Active Set method to Infer Spikes):
    - Fast, exact solution for AR(1) and AR(2) models.
    - Implemented via the `oasis-deconv` PyPI package if available,
      with a pure-Python AR(1) fallback (PAVA algorithm).

Reference (algorithmic only): CaImAn temporal.py:update_temporal_components (line 64)
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from minicnmfe._utils import get_xp, to_numpy


# ---------------------------------------------------------------------------
# Physical decay time <-> AR(1) coefficient
# ---------------------------------------------------------------------------

def g_from_decay_time(decay_time_ms: float, frame_rate_hz: float) -> float:
    """AR(1) decay coefficient for an indicator with decay τ at a frame rate.

    ``g = exp(-1 / (fps · τ_s)) = exp(-1 / (fps · τ_ms / 1000))`` — the per-frame
    fraction of fluorescence retained by a single exponential of time constant
    ``decay_time_ms``. This is the same expression used to build the Bayesian
    ``g`` prior in ``pipeline.fit_extract``; exposed here so the synthetic
    simulator can generate traces with a physically meaningful, settable decay
    (e.g. GCaMP8m τ≈180 ms) instead of an arbitrary AR coefficient.

    Approximate single-AP somatic decay τ (ms): GCaMP6f ~140, jGCaMP7f ~160,
    jGCaMP8f ~70, jGCaMP8m ~180, jGCaMP8s ~350, GCaMP6s/7s ~1000.
    """
    return float(np.exp(-1.0 / (frame_rate_hz * decay_time_ms / 1000.0)))


def decay_time_from_g(g: float, frame_rate_hz: float) -> float:
    """Inverse of :func:`g_from_decay_time`: decay τ (ms) from an AR(1) ``g``.

    ``τ_ms = -1000 / (fps · ln g)``. ``g`` is clipped to ``(0, 1)`` for safety,
    so recovered coefficients of 0 or ≥1 do not blow up. Use this to report a
    pipeline's estimated ``g`` (ours or CaImAn) in interpretable physical units.
    """
    g = float(np.clip(g, 1e-6, 0.999999))
    return float(-1000.0 / (frame_rate_hz * np.log(g)))


# ---------------------------------------------------------------------------
# AR parameter estimation
# ---------------------------------------------------------------------------

def _detrend_poly(x: np.ndarray, order: int) -> np.ndarray:
    """Subtract a least-squares polynomial trend of given order from ``x``.

    ``order == 0`` reduces to mean subtraction (degree-0 polynomial = constant).
    Higher orders absorb linear, quadratic, etc. drift — useful before any
    operation that interprets autocorrelation as calcium decay (a long slow
    bleach trend has lag-1 autocorrelation near 1 and will dominate the
    Yule-Walker solution if not removed).
    """
    if order <= 0:
        return x - x.mean()
    t = np.arange(len(x), dtype=np.float64)
    coeffs = np.polyfit(t, x.astype(np.float64), int(order))
    return (x.astype(np.float64) - np.polyval(coeffs, t)).astype(x.dtype, copy=False)


def estimate_ar_params(
    trace: np.ndarray,
    p: int = 1,
    noise_range: tuple[float, float] = (0.25, 0.5),
    fudge_factor: float = 0.96,
    lags: int = 5,
    detrend_order: int = 0,
    g_prior: float | None = None,
    g_prior_weight: float = 0.5,
) -> tuple[np.ndarray, float]:
    """Estimate AR(p) decay constants and noise std from a fluorescence trace.

    Algorithm:
    1. Estimate noise via power in high-frequency bins (rfft).
    2. Centre the trace (or detrend with a polynomial of order
       ``detrend_order`` if a slow bleach / scope-warmup trend would
       otherwise inflate the autocorrelation).
    3. Fit AR(p) by solving the Yule-Walker equations on the autocorrelation.
    4. Either shrink toward ``g_prior`` (Bayesian path) or multiply by
       ``fudge_factor`` (legacy shrinkage toward zero).

    Args:
        trace: (T,) fluorescence trace.
        p: AR model order (1 or 2).
        noise_range: Frequency band [f_low, f_high] * Nyquist for noise estimate.
        fudge_factor: Shrinkage applied to g (< 1 avoids over-estimated decay).
            Ignored when ``g_prior`` is provided.
        lags: Number of autocorrelation lags used.
        detrend_order: NON-STANDARD. Polynomial order subtracted from the trace
            before Yule-Walker. ``0`` (default) = mean only (standard
            CNMF-E). Set ``2`` to absorb linear and exponential-like drift
            (typical photobleaching) so the AR estimate reflects calcium
            dynamics and not the bleach trajectory. Order 3+ starts to
            over-fit the AR(1) envelope and biases g downward.
        g_prior: Bayesian-prior target for the dominant decay coefficient
            g[0]. Typically derived from indicator τ + frame rate via
            ``g_target = exp(-1 / (fps * τ_ms / 1000))``. ``None``
            (default) selects the legacy ``fudge_factor`` path.
        g_prior_weight: Shrinkage weight on the prior, ``w ∈ [0, 1]``.
            ``0`` = pure Yule-Walker, ``1`` = pin at ``g_prior``. Only used
            when ``g_prior`` is not None. For AR(p > 1) the prior shrinkage
            applies to ``g[0]`` only; higher-order coefficients use the
            legacy path.

    Returns:
        g: AR coefficients, shape (p,).
        sn: Noise standard deviation.
    """
    T = len(trace)
    # Noise via high-frequency PSD (operates on the raw trace — high-frequency
    # bins are insensitive to slow drift, so no detrend is needed here).
    Xf = np.fft.rfft(trace)
    freqs = np.fft.rfftfreq(T)
    noise_mask = (freqs >= noise_range[0]) & (freqs <= noise_range[1])
    if noise_mask.sum() == 0:
        noise_mask = freqs >= noise_range[0]
    psd = np.abs(Xf[noise_mask]) ** 2 / T * 2
    sn = float(np.sqrt(np.exp(np.log(psd + 1e-10).mean())))

    # Yule-Walker on the *detrended* trace.
    trace_centered = _detrend_poly(trace, detrend_order)
    ac = np.array([
        np.dot(trace_centered[: T - k], trace_centered[k:]) / (T - k)
        for k in range(lags + 1)
    ])
    # R @ g = r  (Toeplitz system)
    R = np.array([[ac[abs(i - j)] for j in range(p)] for i in range(p)])
    r = ac[1 : p + 1]
    try:
        g_yw = np.linalg.solve(R, r)
    except np.linalg.LinAlgError:
        # Fallback if Yule-Walker is singular: use prior if available,
        # otherwise the legacy fudge-derived seed.
        if g_prior is not None:
            g_yw = np.array([float(g_prior)] * p)
        else:
            g_yw = np.array([fudge_factor ** (1.0 / max(p, 1))] * p)

    if g_prior is not None:
        # Bayesian shrinkage on g[0] only; higher orders keep the legacy
        # fudge_factor path so AR(p>1) stays consistent with its existing
        # multi-coefficient shape.
        w = float(np.clip(g_prior_weight, 0.0, 1.0))
        g = g_yw.copy().astype(np.float64)
        g[0] = (1.0 - w) * float(g_yw[0]) + w * float(g_prior)
        if p > 1:
            g[1:] = g[1:] * fudge_factor
    else:
        g = g_yw * fudge_factor
    # Clip to (0, 1) for stability. The lower bound must be strictly positive:
    # oasis' constrained_oasisAR1 returns a NaN (at the last frame) when g==0
    # exactly, which then poisons b0/W/the whole background. 1e-6 matches the
    # floor used by decay_time_to_g above; g~1e-6 deconvolves to ~the raw trace
    # (no AR smoothing), correct for a noise-like seed.
    g = np.clip(g, 1e-6, 0.9999)
    return g.astype(np.float32), float(sn)


# ---------------------------------------------------------------------------
# Deconvolution (OASIS with pure-Python AR1 fallback)
# ---------------------------------------------------------------------------

def _oasis_pava_run(y_bl: np.ndarray, g: float, lam: float) -> np.ndarray:
    """One PAVA sweep with L1 penalty ``lam`` on the spike train.

    Minimises ``||y - c||² + lam · Σ s[t]`` subject to
    ``c[t] >= g·c[t-1] >= 0`` (with ``s[t] = c[t] − g·c[t-1]``).

    Inside a pool the AR shape gives one spike at the start of value ``v``
    (and zero spikes during the decay), so the per-pool LS objective is
    minimised at ``v = max(0, (num − lam/2) / den)`` — the only change from
    plain PAVA is the ``lam/2`` shrinkage.
    """
    T = len(y_bl)
    pool_start = list(range(T))
    pool_length = [1] * T
    pool_num = [float(y_bl[t]) for t in range(T)]
    pool_den = [1.0] * T
    pool_val = [max(0.0, float(y_bl[t]) - lam / 2) for t in range(T)]

    def merge(i: int) -> None:
        g_pow = g ** pool_length[i]
        pool_num[i] += g_pow * pool_num[i + 1]
        pool_den[i] += (g_pow ** 2) * pool_den[i + 1]
        pool_length[i] += pool_length[i + 1]
        pool_val[i] = max(0.0, (pool_num[i] - lam / 2) / pool_den[i])

    i = 0
    while i < len(pool_start) - 1:
        # Boundary constraint c[t] >= g·c[t-1]: pool i ends at value
        # val_i·g^(len_i-1), so pool i+1 must start at >= g·(val_i·g^(len_i-1))
        # = val_i·g^(len_i). Merge when that is violated. (Using bare `g` here
        # instead of `g**len_i` spuriously over-merges smooth exact-g decays and
        # collapses the trace — it reconstructs a clean AR(1) at only r~0.4.)
        # The small tolerance suppresses float-noise merges on exact-g decays.
        if pool_val[i] * (g ** pool_length[i]) > pool_val[i + 1] + 1e-9:
            merge(i)
            pool_start.pop(i + 1)
            pool_length.pop(i + 1)
            pool_num.pop(i + 1)
            pool_den.pop(i + 1)
            pool_val.pop(i + 1)
            if i > 0:
                i -= 1
        else:
            i += 1

    c = np.zeros(T, dtype=np.float64)
    for k in range(len(pool_start)):
        start, length, val = pool_start[k], pool_length[k], pool_val[k]
        c[start:start + length] = val * (g ** np.arange(length))
    return c


def _oasis_ar1_pava(y: np.ndarray, g: float, sn: float) -> tuple[np.ndarray, np.ndarray, float]:
    """L1-penalised noise-constrained OASIS AR(1) deconvolution.

    Solves
        min_c  ‖y − c‖² + lam · Σ s[t]
        s.t.   c[t] >= g·c[t-1],  c[t] >= 0,  s[t] = c[t] − g·c[t-1]

    where ``lam`` is chosen by bisection to satisfy the noise constraint
    ``‖y − c‖² ≈ T·sn²``. This is the constrained-foopsi form from
    Friedrich et al. 2017 §2.1, implemented in pure Python so the
    fallback doesn't silently degrade when the Cython `oasis-deconv`
    package isn't installed.

    Args:
        y:  (T,) raw trace.
        g:  AR(1) decay coefficient.
        sn: per-component noise std (from the footprint-weighted formula
            in ``pipeline._sn_from_footprint``). Setting ``sn <= 0`` runs
            the unconstrained PAVA (legacy behaviour).

    Returns:
        c:  denoised calcium trace.
        s:  spike train ``s[t] = c[t] − g·c[t-1]``.
        bl: scalar baseline (median of y).
    """
    T = len(y)
    bl = float(np.median(y))
    y_bl = (y - bl).astype(np.float64)

    if sn <= 0 or T < 2:
        c = _oasis_pava_run(y_bl, g, 0.0)
    else:
        target = float(sn) ** 2 * T
        # lam = 0: smallest residual (overfit). If it's already at/above
        # the noise budget, no shrinkage needed.
        c = _oasis_pava_run(y_bl, g, 0.0)
        resid0 = float(np.sum((y_bl - c) ** 2))
        if resid0 < target:
            # Expand lam_hi until residual exceeds target, then bisect.
            lam_hi = max(2.0 * float(sn), 1e-3)
            for _ in range(20):
                c_hi = _oasis_pava_run(y_bl, g, lam_hi)
                if float(np.sum((y_bl - c_hi) ** 2)) >= target:
                    break
                lam_hi *= 2.0
            lam_lo = 0.0
            # 10 bisection iterations get ~3 decimal places of lam, plenty
            # for the noise-budget match. Cost ≈ 10 PAVA sweeps per neuron.
            for _ in range(10):
                lam_mid = 0.5 * (lam_lo + lam_hi)
                c_mid = _oasis_pava_run(y_bl, g, lam_mid)
                if float(np.sum((y_bl - c_mid) ** 2)) < target:
                    lam_lo = lam_mid
                else:
                    lam_hi = lam_mid
            # Use lam_hi (residual >= target ⇒ within budget).
            c = _oasis_pava_run(y_bl, g, lam_hi)

    s = np.zeros(T, dtype=np.float64)
    s[0] = c[0]
    s[1:] = np.maximum(c[1:] - g * c[:-1], 0)

    return c.astype(np.float32), s.astype(np.float32), float(bl)


def _guard_deconv_output(c, s, bl, trace):
    """Sanitize a (c, s, bl) deconvolution result.

    oasis can return a non-finite trace for pathological inputs (e.g. g==0,
    which NaNs the last frame). A single NaN here propagates through b0/W into
    the whole background and silently kills the extraction, so fall back to the
    clipped raw trace rather than let a NaN escape.
    """
    c = np.asarray(c, dtype=np.float32)
    s = np.asarray(s, dtype=np.float32)
    if not np.isfinite(c).all() or not np.isfinite(bl):
        c = trace.clip(0).astype(np.float32)
        s = np.zeros_like(c)
        bl = 0.0
    return c, s, float(bl)


def deconvolve(
    trace: np.ndarray,
    g: np.ndarray,
    sn: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Deconvolve a fluorescence trace to infer the spike train.

    Uses the `oasis-deconv` package (constrained OASIS).
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
            return _guard_deconv_output(c, s, bl, trace)
        else:
            from oasis.functions import deconvolve as oasis_deconvolve
            c, s, bl, g_out, _ = oasis_deconvolve(
                trace,
                penalty=1,
                g=tuple(float(gi) for gi in g),
                sn=sn,
            )
            return _guard_deconv_output(c, s, bl, trace)
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
    """Deconvolve a single trace (module-level so it is picklable).

    UNUSED / dead code: superseded by ``_deconvolve_with`` (which takes a
    pre-computed ``g``/``sn`` instead of re-estimating per call, avoiding
    fudge-factor drift). ``update_temporal`` only ever calls ``_deconvolve_with``.
    Retained pending a separate code-cleanup PR.
    """
    try:
        g_k, sn_k = estimate_ar_params(trace_k, p=ar_order)
        c_k, s_k, _ = deconvolve(trace_k, g_k, sn_k)
    except Exception:
        c_k = trace_k.clip(0)
        s_k = np.zeros_like(c_k)
    return c_k, s_k


def _deconvolve_with(trace_k: np.ndarray, g_k: np.ndarray, sn_k: float) -> tuple[np.ndarray, np.ndarray]:
    """Deconvolve a single trace with pre-computed g/sn (module-level for pickling)."""
    try:
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
    g_cached: list[np.ndarray] | None = None,
    sn_cached: np.ndarray | None = None,
    deconvolve: bool = True,
    detrend_order: int = 0,
    g_prior: float | None = None,
    g_prior_weight: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
    """Refine temporal traces by block coordinate descent.

    For each component k, the residual data after subtracting all other
    components is projected onto A[:,k], then deconvolved with OASIS.

    When n_jobs != 1 a Jacobi (parallel) update is used within each iteration:
    all K traces are deconvolved simultaneously with the residuals from the
    previous iteration. This converges slightly slower than Gauss-Seidel but
    scales linearly with available cores.

    When device='cuda' the initial projections (Y_flat.T @ A and A.T @ A) run
    on GPU. OASIS deconvolution always runs on CPU (no GPU implementation).

    The AR coefficient `g` (calcium decay) is estimated ONCE per component
    before the BCD loop and reused across iterations. Re-estimating g from
    an already-deconvolved trace each iteration causes systematic drift
    (Yule-Walker recovers ~g_prev, then fudge_factor=0.96 shrinks it again),
    which distorts decay shape and reduces correlation with ground truth.
    Pass `g_cached`/`sn_cached` to reuse values estimated upstream (e.g.
    from C_raw at init); otherwise estimation runs once on input C.

    Args:
        Y_flat: (H*W, T) background-subtracted movie.
        A: (H*W, K) sparse spatial footprints.
        C: (K, T) current temporal traces.
        sn: (H*W,) per-pixel noise std (currently unused — kept for API compat).
        ar_order: AR model order for deconvolution.
        n_iter: Number of BCD iterations.
        n_jobs: Number of parallel workers (-1 = all CPUs, 1 = serial Gauss-Seidel).
        device: 'cpu' or 'cuda'. GPU accelerates the (H*W × T) @ (H*W × K) projection.
        g_cached: Optional list of length K with pre-estimated AR coefs per component.
        sn_cached: Optional (K,) array of pre-estimated per-component noise std.
        detrend_order: NON-STANDARD. Polynomial order subtracted from each
            component's projected trace immediately before OASIS. ``0`` (default
            here, but the pipeline passes ``CNMFeParams.temporal_detrend_order``)
            disables the detrend and matches standard CNMF-E. Positive orders
            strip slow drift so OASIS sees the calcium transients on a flat
            baseline; the drift naturally flows into the residual YrA so the
            noisy projection ``C + YrA`` is unchanged.

    Returns:
        C_new: (K, T) updated calcium traces.
        S: (K, T) inferred spike trains.
        g_per_k: list of length K — AR coefs used for each component.
        sn_per_k: (K,) array — noise std used for each component.
    """
    K, T = C.shape
    xp = get_xp(device)

    # If Y_flat is a streaming subtractor, use its project_onto so we never
    # materialise the full (H*W, T) residual. GPU path requires a dense
    # numpy input — fall back to CPU projection when given a subtractor.
    if hasattr(Y_flat, "project_onto"):
        YA = Y_flat.project_onto(A, n_jobs=n_jobs)
        AA = (A.T @ A).toarray() if sp.issparse(A) else np.asarray(A.T @ A)
    elif xp is not np:
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

    # Estimate g/sn ONCE per component before the BCD loop. This avoids
    # the drift that comes from re-estimating from progressively shaped traces.
    if g_cached is None or sn_cached is None:
        g_per_k: list[np.ndarray] = []
        sn_per_k = np.zeros(K, dtype=np.float32)
        for k in range(K):
            try:
                g_k, sn_k = estimate_ar_params(
                    C[k], p=ar_order,
                    g_prior=g_prior, g_prior_weight=g_prior_weight,
                )
            except Exception:
                fallback_g = (
                    float(g_prior) if g_prior is not None
                    else 0.9 ** (1.0 / max(ar_order, 1))
                )
                g_k = np.array([fallback_g] * ar_order, dtype=np.float32)
                sn_k = float(np.std(C[k])) if np.std(C[k]) > 0 else 1.0
            g_per_k.append(g_k)
            sn_per_k[k] = sn_k
    else:
        g_per_k = list(g_cached)
        sn_per_k = np.asarray(sn_cached, dtype=np.float32).copy()

    C = C.copy()
    S = np.zeros_like(C)
    YrA = YA - (AA @ C).T

    if n_jobs == 1:
        # Gauss-Seidel: each update immediately uses the latest C[k]
        for _ in range(n_iter):
            for k in range(K):
                trace_k = (YrA[:, k] / nA[k] + C[k]).astype(np.float32)
                if deconvolve:
                    trace_for_oasis = (
                        _detrend_poly(trace_k, detrend_order)
                        if detrend_order > 0 else trace_k
                    )
                    c_k, s_k = _deconvolve_with(trace_for_oasis, g_per_k[k], float(sn_per_k[k]))
                else:
                    c_k = np.maximum(trace_k, 0.0)
                    s_k = np.zeros_like(c_k)
                delta = c_k - C[k]
                YrA -= np.outer(delta, AA[:, k])
                C[k] = c_k
                S[k] = s_k
    else:
        from joblib import Parallel, delayed

        # Jacobi: all K components deconvolved in parallel, then YrA updated
        for _ in range(n_iter):
            traces = [(YrA[:, k] / nA[k] + C[k]).astype(np.float32) for k in range(K)]
            if deconvolve:
                from threadpoolctl import threadpool_limits

                # Threads avoid loky's per-call pickling. OASIS's C extension
                # (from oasis-deconv) releases the GIL during deconv;
                # users on the pure-Python fallback get no parallelism either
                # way -- the threads path doesn't make that worse.
                # threadpool_limits caps inner BLAS to 1 so n_jobs worker
                # threads x n_cores BLAS threads doesn't oversubscribe.
                if detrend_order > 0:
                    traces_for_oasis = [_detrend_poly(t, detrend_order) for t in traces]
                else:
                    traces_for_oasis = traces
                with threadpool_limits(limits=1, user_api="blas"):
                    results = Parallel(n_jobs=n_jobs, prefer="threads")(
                        delayed(_deconvolve_with)(traces_for_oasis[k], g_per_k[k], float(sn_per_k[k]))
                        for k in range(K)
                    )
            else:
                results = [
                    (np.maximum(traces[k], 0.0), np.zeros_like(traces[k]))
                    for k in range(K)
                ]
            for k, (c_k, s_k) in enumerate(results):
                delta = c_k - C[k]
                YrA -= np.outer(delta, AA[:, k])
                C[k] = c_k
                S[k] = s_k

    return C, S, g_per_k, sn_per_k
