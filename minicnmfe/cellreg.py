"""Cross-session cell registration — a port of CellReg.

Port of the Ziv lab's CellReg (Sheintuch et al., 2017, *Cell Reports*), following
the MATLAB source at github.com/zivlab/CellReg (master, commit 7066851), the
probabilistic pipeline of ``demo.m``:

1. footprints normalised to unit sum (``normalize_spatial_footprints``), no
   thresholding;
2. sub-pixel centroids from Gaussian fits to the x / y marginal projections
   (``compute_centroid_locations``);
3. alignment to a reference session, translations (+ rotations), by normalised
   cross-correlation of *centroid projections* with footprint projections as the
   fallback; a session whose footprint projections do not correlate is flagged
   and left out (``align_images``);
4. all neighbouring cell pairs within ``maximal_distance``: centroid distance and
   ``corr2`` over the whole footprint images (``compute_data_distribution``);
5. P_same models on binned data — lognormal(1 - corr) + beta for correlation,
   lognormal + linear*sigmoid for distance — and the lower-cost model
   (``compute_*_model``, ``choose_best_model``);
6. greedy initial registration on the chosen feature's P_same = 0.5 crossing
   (``initial_registration_*``), then iterative clustering on P_same with the
   'Maximal similarity' rule, switches and cluster merging (``cluster_cells``),
   and cell scores (``compute_scores``).

Units. CellReg works in microns and needs ``microns_per_pixel``. Microns enter
only as window sizes, initial values and search ranges, all built from two
constants: a typical cell size (12 um; 10 um inside ``align_images``) and the
maximal pair distance (14 um in the 1-photon demo). Here they are expressed in
pixels through ONE number, ``cell_size_px`` (the pixel equivalent of 12 um),
scaled with CellReg's own ratios (see ``_scales``). ``cell_size_px=None``
estimates it as the median FWHM diameter of the footprints. That estimate is
not CellReg's scale: CellReg uses 12 um / microns_per_pixel, which equals the
estimate only when the footprints' FWHM diameter is 12 um (on CellReg's sample
data it is about 18 um). Pass ``cell_size_px = 12 / microns_per_pixel`` to
reproduce CellReg exactly.

Upstream slips. By default (``fix_upstream_slips=False``) the source's
behaviour is reproduced, slips included. ``fix_upstream_slips=True`` corrects:
* ``choose_best_model`` measures the distance model's FP/FN at the *correlation*
  model's index; with the fix each model uses its own index.
* ``align_images`` adds the rotation Gaussian-fit offset in bins to an angle in
  degrees; with the fix it is multiplied by ``rotation_step``.
* ``align_images`` searches rotations about (row H/2, col W/2) but applies them
  to the transposed image, i.e. about (row W/2, col H/2) for a non-square field
  of view; with the fix one centre, (H/2, W/2) in 1-based units, is used for both.
* the translation after a rotation is estimated from a centroid projection
  rebuilt from the UNROTATED centroids; with the fix the centroids are rotated.
* NaN P_same values (0/0 where both model densities vanish) stay NaN; with the
  fix they are set to 0.
* a Gaussian fit that diverges (offset outside its own fit window, e.g. for a
  footprint cut at the image edge) places the centroid wherever the fit ended;
  with the fix it falls back to the argmax.

Not reproducible: lsqcurvefit (distance model) and mle (the spatial model's
fallback when EM fails) are replaced by scipy optimisers with the source's
start values and tolerances, so fitted values can differ in the last digits.

Guards where the source produces NaN or an error (results are unchanged
wherever the source runs): a sub-pixel peak fit that breaks down (NaN) leaves
that axis at the integer peak, fixed start values when at most one distance or
correlation is available, start values clipped into the bounds, intersection
fallbacks (0.65 / 5 um) when the search range is empty, an all-zero footprint
kept at zero instead of NaN, non-finite pixels set to 0, and a failed session
left out of the registration (see ``align_sessions``).

Floating-point summation order differs from MATLAB; nothing else is intended to.

Extension, not in CellReg: ``align_all_pairs`` runs the same per-pair estimate on
every pair of sessions and fits one transform per session to all pairs that pass
the quality test, instead of aligning every session to one reference.

The previous implementation is kept as ``minicnmfe.cellreg_legacy``. A
``register_sessions`` call with its arguments (``align``, ``dist_thr_px``, ...)
is handed to it with a warning, so stored parameter sets keep working.
"""

from __future__ import annotations

import functools
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
from scipy.optimize import least_squares
from scipy.special import betaln, polygamma, psi
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from minicnmfe.pipeline import CNMFe


# ---------------------------------------------------------------------------
# small MATLAB equivalents
# ---------------------------------------------------------------------------

def _mround(x):
    """MATLAB round: half away from zero."""
    x = np.asarray(x, dtype=float)
    out = np.sign(x) * np.floor(np.abs(x) + 0.5)
    return out.astype(int) if out.ndim else int(out)


def _sigmoid(x, a, c):
    return 1.0 / (1.0 + np.exp(-a * (np.asarray(x, dtype=float) - c)))


def _lognpdf(x, mu, sigma):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    pos = x > 0
    out[pos] = np.exp(-0.5 * ((np.log(x[pos]) - mu) / sigma) ** 2) / (
        x[pos] * sigma * np.sqrt(2 * np.pi))
    return out


def _betapdf(x, p, q):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    ok = (x > 0) & (x < 1)
    out[ok] = np.exp((p - 1) * np.log(x[ok]) + (q - 1) * np.log1p(-x[ok]) - betaln(p, q))
    return out


def _lognfit(x):
    """MATLAB lognfit: [mean(log x), std(log x) with n - 1]."""
    lx = np.log(np.asarray(x, dtype=float))
    return float(lx.mean()), (float(lx.std(ddof=1)) if lx.size > 1 else 1.0)


def _betafit(x):
    """MATLAB betafit (maximum likelihood on (0, 1))."""
    from scipy.stats import beta
    x = np.clip(np.asarray(x, dtype=float), 1e-10, 1 - 1e-10)
    p, q, _, _ = beta.fit(x, floc=0, fscale=1)
    return float(p), float(q)


def _hist_centers(data, centers):
    """MATLAB ``hist(data, centers)``: bin edges at the midpoints between centres,
    the outer bins open-ended."""
    edges = 0.5 * (centers[:-1] + centers[1:])
    idx = np.searchsorted(edges, np.asarray(data, dtype=float), side="right")
    return np.bincount(idx, minlength=len(centers)).astype(float)


def gaussfit(x, y, sigma0=None, mu0=None, n_iter=50):
    """Port of CellReg's ``gaussfit.m``: Gauss-Newton fit of a normalised Gaussian
    pdf to (x, y). Returns (sigma, mu). Start values as in the source: mu at the
    maximum of y and sigma = 1 / (sqrt(2 pi) max y); ``sigma0`` replaces sigma only
    when ``mu0`` is not given (``nargin == 3``), so with both given (every CellReg
    call) ``sigma0`` is not used. A breakdown (singular normal matrix) gives NaN, as
    the source's ``(F'*F)^(-1)`` does."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    s = float(np.sum(np.diff(x) * 0.5 * (y[:-1] + y[1:])))
    if s > 1.5 or s < 0.5:
        y = y / s
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        i = int(np.argmax(y)) if np.all(np.isfinite(y)) else 0
        mu = float(x[i])
        sigma = float(1.0 / (np.sqrt(2 * np.pi) * y[i]))
        if mu0 is None and sigma0 is not None:
            sigma = float(sigma0)
        if mu0 is not None:
            mu = float(mu0)
        if not np.all(np.isfinite(y)):
            return float("nan"), float("nan")
        return _gauss_newton(x, y, sigma, mu, n_iter)


def _gauss_newton(x, y, sigma, mu, n_iter):
    for _ in range(n_iter):
        e = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
        f0 = e / (np.sqrt(2 * np.pi) * sigma)
        dfds = -e / (np.sqrt(2 * np.pi) * sigma ** 2) + f0 * ((x - mu) ** 2 / sigma ** 3)
        dfdm = f0 * (x - mu) / sigma ** 2
        F = np.column_stack([dfds, dfdm])
        try:
            step = np.linalg.solve(F.T @ F, F.T @ (y - f0))
        except np.linalg.LinAlgError:
            return float("nan"), float("nan")
        if not np.all(np.isfinite(step)):
            return float("nan"), float("nan")
        sigma, mu = sigma + step[0], mu + step[1]
    return sigma, mu


def _local_sum(A, shape):
    """Sums of A over every template-sized window of the zero-padded image, as
    MATLAB's normxcorr2 ``local_sum``: one cumulative sum per axis, so an empty
    window gives exactly 0."""
    m, n = shape
    s = np.cumsum(np.pad(A, ((m, m), (n, n))), axis=0)
    c = s[m:-1, :] - s[:-m - 1, :]
    s = np.cumsum(c, axis=1)
    return s[:, n:-1] - s[:, :-n - 1]


def normxcorr2(template, image):
    """MATLAB ``normxcorr2(template, A)``: full normalised cross-correlation,
    output (Ht + Ha - 1, Wt + Wa - 1); entries where the local image variance is
    ~0 are 0. If A is the template moved by +d, the peak is at 0-based index
    (Ht - 1 + d_y, Wt - 1 + d_x)."""
    from scipy.signal import fftconvolve
    T = np.asarray(template, dtype=float)
    A = np.asarray(image, dtype=float)
    n = T.size
    Tz = T - T.mean()
    num = fftconvolve(A, Tz[::-1, ::-1], mode="full")
    lsum = _local_sum(A, T.shape)
    lsum2 = _local_sum(A * A, T.shape)
    den_A = np.maximum(lsum2 - lsum ** 2 / n, 0.0)
    denom = np.sqrt(den_A * float((Tz * Tz).sum()))
    tol = np.sqrt(np.spacing(np.max(np.abs(denom)))) if denom.size else 0.0
    out = np.zeros_like(num)
    ok = denom > tol
    out[ok] = num[ok] / denom[ok]
    return out


# ---------------------------------------------------------------------------
# sizes: CellReg's micron constants as multiples of one pixel cell size
# ---------------------------------------------------------------------------

def _scales(cell_size_px):
    """CellReg's micron constants in pixels. ``cell_size_px`` stands for the 12 um
    typical cell size; every other length keeps its ratio to 12 um."""
    u = float(cell_size_px) / 12.0        # pixels per CellReg micron
    return {
        "um": u,
        "maximal_distance": 14.0 * u,     # demo.m (1-photon)
        "centroid_cell": 12.0 * u,        # compute_centroid_locations
        "align_cell": 10.0 * u,           # align_images
        "translate_radius": 25.0 * u,     # translate_spatial_footprint
        "rotate_radius": 30.0 * u,        # rotate_spatial_footprint
        "dist_fit_max": 9.0 * u,          # compute_centroid_distances_model
        "dist_c0": 6.0 * u,
        "dist_a0": 1.0 / u,               # a_0 = 1 per micron
        "intersect_lo": 1.0 * u,
        "intersect_hi": 10.0 * u,
    }


# ---------------------------------------------------------------------------
# footprints (sparse, unit sum)
# ---------------------------------------------------------------------------

@dataclass
class _Session:
    """One session's footprints as sparse rows, plus geometry."""

    fp: sp.csr_matrix                 # (K, H*W)
    dims: tuple[int, int]
    centroids: np.ndarray = None      # (K, 2) (y, x), 0-based, sub-pixel

    @property
    def K(self) -> int:
        return self.fp.shape[0]

    def image(self, k) -> np.ndarray:
        img = np.zeros(self.dims[0] * self.dims[1])
        lo, hi = self.fp.indptr[k], self.fp.indptr[k + 1]
        img[self.fp.indices[lo:hi]] = self.fp.data[lo:hi]
        return img.reshape(self.dims)

    def rows(self):
        """Per footprint: (sorted flat indices, values, sum, sum of squares)."""
        F = self.fp
        out = []
        for k in range(self.K):
            lo, hi = F.indptr[k], F.indptr[k + 1]
            i, v = F.indices[lo:hi], F.data[lo:hi]
            o = np.argsort(i, kind="stable")
            i, v = i[o], v[o]
            out.append((i, v, float(v.sum()), float((v * v).sum())))
        return out


def _load_session(src, *, accepted_only=False) -> _Session:
    from minicnmfe.pipeline import CNMFe
    model = CNMFe.load(src) if isinstance(src, (str, Path)) else src
    if model.A is None or model.dims is None:
        raise ValueError("Session has no footprints (model.A / model.dims is None).")
    dims = (int(model.dims[0]), int(model.dims[1]))
    A = model.A.tocsc()
    cols = np.arange(A.shape[1])
    if accepted_only and getattr(model, "accepted_mask", None) is not None:
        cols = np.flatnonzero(np.asarray(model.accepted_mask, dtype=bool))
    F = sp.csr_matrix(A[:, cols].T, dtype=np.float64)
    F.data[~np.isfinite(F.data)] = 0.0
    F.eliminate_zeros()
    s = np.asarray(F.sum(axis=1)).ravel()
    s[s == 0] = 1.0
    F = sp.csr_matrix(sp.diags(1.0 / s) @ F)
    F.sort_indices()
    return _Session(fp=F, dims=dims)


def adjust_fov_size(sessions) -> list:
    """``adjust_FOV_size``: zero-pad every session at the bottom and right to the
    largest height and width, keeping pixel (0, 0) in place."""
    H = max(s.dims[0] for s in sessions)
    W = max(s.dims[1] for s in sessions)
    out = []
    for s in sessions:
        h, w = s.dims
        if (h, w) == (H, W):
            out.append(s)
            continue
        F = s.fp.tocoo()
        fp = sp.csr_matrix((F.data, (F.row, (F.col // w) * W + F.col % w)), shape=(s.K, H * W))
        fp.sort_indices()
        out.append(_Session(fp=fp, dims=(H, W), centroids=s.centroids))
    return out


def estimate_cell_size_px(sessions) -> float:
    """Median FWHM diameter (px) over all footprints: 2*sqrt(area >= max/2 / pi)."""
    d = []
    for s in sessions:
        F = s.fp
        for k in range(F.shape[0]):
            v = F.data[F.indptr[k]:F.indptr[k + 1]]
            if v.size:
                d.append(2.0 * np.sqrt((v >= 0.5 * v.max()).sum() / np.pi))
    return float(np.median(d)) if d else 7.0


# ---------------------------------------------------------------------------
# centroids, projections
# ---------------------------------------------------------------------------

def compute_centroids(sess: _Session, cell_px: float, fix_upstream_slips=False) -> np.ndarray:
    """``compute_centroid_locations``: argmax of each marginal projection, then a
    Gaussian fit within +-round(2*cell) of it. (y, x), 0-based; NaN for an empty
    footprint (0/0 in the source). A fit that diverges is kept as in the source,
    or with ``fix_upstream_slips`` replaced by the argmax."""
    H, W = sess.dims
    g = _mround(2 * cell_px)
    xs = np.arange(-g, g + 1)
    out = np.zeros((sess.K, 2))
    F = sess.fp
    for k in range(sess.K):
        lo, hi = F.indptr[k], F.indptr[k + 1]
        idx, val = F.indices[lo:hi], F.data[lo:hi]
        xproj = np.bincount(idx % W, weights=val, minlength=W)
        yproj = np.bincount(idx // W, weights=val, minlength=H)
        c = []
        for proj in (xproj, yproj):
            m = int(np.argmax(proj))
            pad = np.concatenate([np.zeros(g), proj, np.zeros(g)])
            loc = pad[m:m + 2 * g + 1]
            ssum = loc.sum()
            if not ssum > 0:
                c.append(np.nan)
                continue
            mu = gaussfit(xs, loc / ssum, 0.5 * cell_px, 0.0)[1]
            if fix_upstream_slips and not (np.isfinite(mu) and abs(mu) <= g):
                mu = 0.0
            c.append(m + mu)
        out[k] = (c[1], c[0])
    return out


def footprints_projection(sess: _Session) -> np.ndarray:
    """``compute_footprints_projections``: each footprint / its max, < 0.5 -> 0, summed."""
    F = sess.fp
    proj = np.zeros(sess.dims[0] * sess.dims[1])
    for k in range(sess.K):
        lo, hi = F.indptr[k], F.indptr[k + 1]
        v = F.data[lo:hi]
        if v.size == 0 or v.max() <= 0:
            continue
        w = v / v.max()
        w[w < 0.5] = 0.0
        np.add.at(proj, F.indices[lo:hi], w)
    return proj.reshape(sess.dims)


def centroids_projection(centroids, dims) -> np.ndarray:
    """``compute_centroids_projections``: a 3x3 kernel (1 centre, 1/2 edges,
    1/4 corners) at each rounded centroid, summed."""
    H, W = dims
    img = np.zeros(dims)
    ker = np.array([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]])
    for cy, cx in centroids:
        if not (np.isfinite(cy) and np.isfinite(cx)):
            continue                                   # NaN fails every test in the source
        r, c = _mround(cy + 1) - 1, _mround(cx + 1) - 1   # MATLAB rounds 1-based coordinates
        if 1 <= r <= H - 3 and 1 <= c <= W - 3:
            img[r - 1:r + 2, c - 1:c + 2] += ker
        elif 0 <= r <= H - 2 and 0 <= c <= W - 2:
            img[r, c] += 1.0
    return img


# ---------------------------------------------------------------------------
# geometry: one inverse map for images and footprints
# ---------------------------------------------------------------------------

def _inverse_coords(yy, xx, theta_deg, shift, center):
    """Source coordinates for output pixels. The output is the input rotated by
    ``theta_deg`` about ``center`` and moved by ``-shift``: a session whose
    content sits at +shift relative to the reference is brought back onto it."""
    t = np.deg2rad(theta_deg)
    cy, cx = center
    y = yy + shift[0] - cy
    x = xx + shift[1] - cx
    return np.cos(t) * y - np.sin(t) * x + cy, np.sin(t) * y + np.cos(t) * x + cx


def _forward_points(pts, theta_deg, shift, center):
    """Where source points land under the transform of ``_inverse_coords``."""
    t = np.deg2rad(theta_deg)
    cy, cx = center
    y, x = pts[:, 0] - cy, pts[:, 1] - cx
    return np.column_stack([np.cos(t) * y + np.sin(t) * x + cy - shift[0],
                            -np.sin(t) * y + np.cos(t) * x + cx - shift[1]])


def _bilinear(img, sy, sx):
    """``interpolate_pixel_value``: bilinear, zero outside the image."""
    H, W = img.shape
    y0, x0 = np.floor(sy).astype(int), np.floor(sx).astype(int)
    y1, x1 = np.ceil(sy).astype(int), np.ceil(sx).astype(int)
    a, b = sx - x0, sy - y0

    def at(r, c):
        ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        v = np.zeros(r.shape)
        v[ok] = img[r[ok], c[ok]]
        return v
    return ((1 - a) * ((1 - b) * at(y0, x0) + b * at(y1, x0))
            + a * ((1 - b) * at(y0, x1) + b * at(y1, x1)))


def transform_image(img, theta_deg, shift, center):
    H, W = img.shape
    yy, xx = np.mgrid[:H, :W].astype(float)
    sy, sx = _inverse_coords(yy, xx, theta_deg, shift, center)
    return _bilinear(img, sy, sx)


def transform_session(sess: _Session, theta_deg, shift, center, radius) -> _Session:
    """``translate_spatial_footprint`` / ``rotate_spatial_footprint``: every
    footprint warped by the same transform; output pixels whose SOURCE location
    lies more than ``radius`` from the footprint's source centroid are zeroed."""
    H, W = sess.dims
    R = int(np.ceil(radius)) + 2
    dest = _forward_points(sess.centroids, theta_deg, shift, center) if sess.K else np.zeros((0, 2))
    rows, cols, vals = [], [], []
    for k in range(sess.K):
        oy, ox = dest[k]
        if not (np.isfinite(oy) and np.isfinite(ox)):
            continue                                   # empty footprint
        r0, r1 = max(0, int(np.floor(oy)) - R), min(H, int(np.floor(oy)) + R + 1)
        c0, c1 = max(0, int(np.floor(ox)) - R), min(W, int(np.floor(ox)) + R + 1)
        if r0 >= r1 or c0 >= c1:
            continue
        yy, xx = np.mgrid[r0:r1, c0:c1].astype(float)
        sy, sx = _inverse_coords(yy, xx, theta_deg, shift, center)
        cy, cx = sess.centroids[k]
        keep = np.hypot(sy - cy, sx - cx) <= radius
        if not keep.any():
            continue
        v = np.zeros(yy.shape)
        v[keep] = _bilinear(sess.image(k), sy[keep], sx[keep])
        nz = v > 0
        rows.append(np.full(int(nz.sum()), k))
        cols.append(yy[nz].astype(int) * W + xx[nz].astype(int))
        vals.append(v[nz])
    if rows:
        fp = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                           shape=(sess.K, H * W))
        fp.sort_indices()
    else:
        fp = sp.csr_matrix((sess.K, H * W))
    return _Session(fp=fp, dims=sess.dims)


# ---------------------------------------------------------------------------
# alignment (align_images, 'Translations' / 'Translations and Rotations')
# ---------------------------------------------------------------------------

@dataclass
class Alignment:
    shifts: np.ndarray                 # (n, 2) (dy, dx): content offset vs the reference, after rotation
    rotations: np.ndarray              # (n,) correction angle (deg) applied to the session
    max_cross_correlation: np.ndarray  # (n,) footprint-projection NCC peak (central third)
    used_centroid_projection: np.ndarray  # (n,) bool
    bad_sessions: list                 # failed the quality test -> left out


def _central_third(C):
    """CellReg's crop of a full xcorr map to its central third (1-based
    ``round(sz/2 - sz/6) : round(sz/2 + sz/6)``). Returns the crop and the 0-based
    offset of its first row / column."""
    sz = C.shape
    r0 = _mround(sz[0] / 2 - sz[0] / 6) - 1
    r1 = _mround(sz[0] / 2 + sz[0] / 6)
    c0 = _mround(sz[1] / 2 - sz[1] / 6) - 1
    c1 = _mround(sz[1] / 2 + sz[1] / 6)
    return C[r0:r1, c0:c1], (r0, c0)


def _subpixel_peak(P, cell_px):
    """CellReg's sub-pixel refinement of the xcorr peak: Gaussian fits along three
    rows and three columns around it. Returns (y, x) within P, 0-based."""
    x_ind = int(np.argmax(P.max(axis=0)))
    y_ind = int(np.argmax(P[:, x_ind]))
    g = _mround(1.5 * cell_px)
    h, w = P.shape
    x_ind = min(max(x_ind, g), w - g - 1)
    y_ind = min(max(y_ind, g), h - g - 1)
    xs = np.arange(-g, g + 1)
    s0 = cell_px / 5.0
    tx = P[y_ind - 1:y_ind + 2, x_ind - g:x_ind + g + 1]
    ty = P[y_ind - g:y_ind + g + 1, x_ind - 1:x_ind + 2]

    def fit(v, norm):
        n = norm.sum()
        if not np.isfinite(n) or n == 0:
            return 0.0
        mu = gaussfit(xs, v / n, s0, 0.0)[1]
        return mu if np.isfinite(mu) else 0.0
    # as in the source, row 3 is normalised by row 2's sum
    mx = [fit(tx[0], tx[0]), fit(tx[1], tx[1]), fit(tx[2], tx[1])]
    my = [fit(ty[:, 0], ty[:, 0]), fit(ty[:, 1], ty[:, 1]), fit(ty[:, 2], ty[:, 2])]
    sub_x = float(np.mean([mx[0], mx[1], mx[1], mx[2]]))
    sub_y = float(np.mean([my[0], my[1], my[1], my[2]]))
    return (y_ind + (sub_y if abs(sub_y) <= 1 else 0.0),
            x_ind + (sub_x if abs(sub_x) <= 1 else 0.0))


@dataclass
class PairEstimate:
    """``align_images`` for one (reference, moving) pair, before any warping."""

    theta: float                       # correction angle (deg) for the moving session
    shift: tuple                       # (dy, dx) content offset of the moving session, after rotation
    max_cross_correlation: float       # footprint-projection NCC peak (central third)
    median_cross_correlation: float    # median of the same central third
    used_centroid_projection: bool
    passed: bool                       # CellReg's quality test


def rotation_centers(dims, fix_upstream_slips=False):
    """(search centre, apply centre), 0-based (row, col). The source searches about
    (H/2, W/2) and applies rotations to the transposed image, i.e. about (W/2, H/2)."""
    H, W = dims
    search = (H / 2.0 - 1.0, W / 2.0 - 1.0)
    return search, (search if fix_upstream_slips else (W / 2.0 - 1.0, H / 2.0 - 1.0))


def estimate_pair(fp_ref, cp_ref, fp_mov, cp_mov, cents_mov, dims, cell_px, *, rotations=True,
                  maximal_rotation=30.0, rotation_step=0.5, minimal_rotation=0.3,
                  sufficient_correlation_centroids=0.2, sufficient_correlation_footprints=0.3,
                  fix_upstream_slips=False):
    """The per-session body of ``align_images``: rotation scan, translation from the
    sub-pixel xcorr peak, and the quality test (footprint-projection NCC peak >
    median + ``sufficient_correlation_footprints``). ``fp_*`` are footprint
    projections, ``cp_*`` centroid projections, ``cents_mov`` the moving session's
    centroids. The rotation is applied about ``rotation_centers(dims, fix)[1]``."""
    S = _scales(cell_px)
    H, W = dims
    center, apply_center = rotation_centers(dims, fix_upstream_slips)
    theta = 0.0
    if rotations:
        angles = np.arange(-maximal_rotation, maximal_rotation + 1e-9, rotation_step)

        def scan(mov, ref):
            return np.array([normxcorr2(ref, transform_image(mov, a, (0, 0), center)).max()
                             for a in angles])
        v = scan(cp_mov, cp_ref)
        if v.max() < sufficient_correlation_centroids:
            v = scan(fp_mov, fp_ref)
        ib = int(np.argmax(v))
        rr = _mround(5.0 / rotation_step)
        loc = np.zeros(2 * rr + 1)
        lo, hi = ib - rr, ib + rr + 1
        loc[max(0, -lo):2 * rr + 1 - max(0, hi - len(v))] = v[max(0, lo):min(len(v), hi)]
        loc = loc - loc.min()
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = gaussfit(np.arange(-rr, rr + 1), loc / loc.sum(), 0.1 * rr, 0.0)[1]
        if fix_upstream_slips:
            theta = float(angles[ib] + (mu * rotation_step if np.isfinite(mu) and abs(mu) <= rr
                                        else 0.0))
        else:
            theta = float(angles[ib] + mu)              # source: offset in bins added to degrees
        if not abs(theta) > minimal_rotation:           # also NaN: the source does not rotate
            theta = 0.0
    if theta:
        fp_s = transform_image(fp_mov, theta, (0, 0), apply_center)
        cp_s = (centroids_projection(_forward_points(cents_mov, theta, (0, 0), apply_center), dims)
                if fix_upstream_slips else cp_mov)     # source: rebuilt from unrotated centroids
    else:
        fp_s, cp_s = fp_mov, cp_mov
    full = normxcorr2(fp_ref, fp_s)
    cent = normxcorr2(cp_ref, cp_s)
    used_cent = bool(cent.max() >= sufficient_correlation_centroids)
    if not used_cent:
        cent = full
    part_full, _ = _central_third(full)
    maxcc, med = float(part_full.max()), float(np.median(part_full))
    part, (r0, c0) = _central_third(cent)
    py, px = _subpixel_peak(part, S["align_cell"])
    return PairEstimate(theta, (float(py + r0 - (H - 1)), float(px + c0 - (W - 1))), maxcc, med,
                        used_cent, bool(maxcc > med + sufficient_correlation_footprints))


def align_sessions(sessions, *, reference, cell_px, rotations=True,
                   maximal_rotation=30.0, rotation_step=0.5, minimal_rotation=0.3,
                   sufficient_correlation_centroids=0.2,
                   sufficient_correlation_footprints=0.3, correct_bad=False,
                   fix_upstream_slips=False):
    """``align_images``. Returns (corrected sessions, Alignment). A session that
    fails the quality test is listed in ``bad_sessions`` and keeps its uncorrected
    footprints, unless ``correct_bad`` (then it is transformed like the others and
    only flagged). The source leaves a failed session's corrected footprints empty
    and has no step that removes it (compute_data_distribution would then fail);
    ``register_sessions`` leaves it out of the registration instead."""
    S = _scales(cell_px)
    n = len(sessions)
    H, W = sessions[reference].dims
    _, apply_center = rotation_centers((H, W), fix_upstream_slips)
    fproj = [footprints_projection(s) for s in sessions]
    cproj = [centroids_projection(s.centroids, s.dims) for s in sessions]
    shifts = np.zeros((n, 2))
    rots = np.zeros(n)
    maxcc = np.full(n, np.nan)
    used_cent = np.zeros(n, bool)
    bad = []
    out = list(sessions)
    for s in range(n):
        if s == reference:
            continue
        e = estimate_pair(fproj[reference], cproj[reference], fproj[s], cproj[s],
                          sessions[s].centroids, (H, W), cell_px, rotations=rotations,
                          maximal_rotation=maximal_rotation, rotation_step=rotation_step,
                          minimal_rotation=minimal_rotation,
                          sufficient_correlation_centroids=sufficient_correlation_centroids,
                          sufficient_correlation_footprints=sufficient_correlation_footprints,
                          fix_upstream_slips=fix_upstream_slips)
        shifts[s] = e.shift
        rots[s] = e.theta
        maxcc[s] = e.max_cross_correlation
        used_cent[s] = e.used_centroid_projection
        if not e.passed:
            bad.append(s)
        if e.passed or correct_bad:
            radius = S["rotate_radius"] if e.theta else S["translate_radius"]
            corr = transform_session(sessions[s], e.theta, e.shift, apply_center, radius)
            corr.centroids = compute_centroids(corr, S["centroid_cell"], fix_upstream_slips)
            out[s] = corr
    return out, Alignment(shifts, rots, maxcc, used_cent, bad)


# ---------------------------------------------------------------------------
# all-pairs alignment (an extension; not in CellReg)
# ---------------------------------------------------------------------------
#
# align_images compares every session with ONE reference. Over weeks the cells two
# sessions share thin out, and far sessions fail the quality test even when their
# shift is right. Here estimate_pair runs on every pair of sessions; pairs that pass
# the quality test are edges, and one transform per session is fitted to all edges by
# robust least squares, with an anchor session fixed at the identity. In the
# convention of _forward_points, F_{t,s}(p) = Q(t)(p - c) + c - s with
# Q(a) Q(b) = Q(a + b), so an edge (i, j) whose estimate maps j into i predicts
#     t_j = t_i + t_ij,   s_j = s_i + Q(t_i) s_ij.

@dataclass
class AllPairsAlignment:
    transforms: np.ndarray             # (n, 3) correction angle (deg), dy, dx into the anchor's frame; NaN = unplaced
    anchor: int
    unplaced: list                     # sessions with no passing pair into the anchor's component
    pairs: np.ndarray                  # (m, 2) i < j; the estimate maps session j into session i
    estimates: np.ndarray              # (m, 3) theta, dy, dx of each pair
    max_cross_correlation: np.ndarray  # (m,)
    median_cross_correlation: np.ndarray  # (m,)
    passed: np.ndarray                 # (m,) CellReg's quality test
    inlier: np.ndarray                 # (m,) used in the final fit (passed, in the anchor's component, not an outlier)
    residual_px: np.ndarray            # (m,) fit vs estimate (angle as displacement at lever_px); NaN if unplaced
    lever_px: float
    dims: tuple                        # (H, W) of every session
    cell_size_px: float

    def warp_image(self, s, img):
        """Any (H, W) image of session ``s`` (e.g. its mean or correlation image) in the
        anchor's frame; 0 outside the session's field of view. None if ``s`` is unplaced."""
        if s in self.unplaced:
            return None
        H, W = self.dims
        t = self.transforms[s]
        return transform_image(np.asarray(img, float), t[0], (t[1], t[2]), (H / 2.0 - 1.0, W / 2.0 - 1.0))


def _Q(t_deg):
    t = np.deg2rad(t_deg)
    return np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])


def _compose(Ti, e):
    """Transform of j predicted from the transform of i and the edge estimate (i -> j)."""
    return np.r_[Ti[0] + e[0], Ti[1:] + _Q(Ti[0]) @ np.asarray(e[1:])]


def _edge_residual(T, i, j, e, lever_px):
    d = T[j] - _compose(T[i], e)
    return np.r_[np.deg2rad(d[0]) * lever_px, d[1:]]


def pair_graph_components(n, pairs):
    """Connected components of the graph with ``pairs`` as edges, largest first
    (ties: the one with the earliest session first)."""
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i, j in pairs:
        parent[find(int(i))] = find(int(j))
    groups = {}
    for k in range(n):
        groups.setdefault(find(k), []).append(k)
    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


def fit_session_transforms(n, pairs, estimates, use, *, anchor=None, lever_px=100.0,
                           f_scale=1.0, outlier_px=3.0, max_rounds=5):
    """One transform per session from the pair estimates ``use``d as edges.

    Soft-L1 least squares (``f_scale`` px); edges whose residual exceeds
    ``outlier_px`` are dropped and the fit repeated. ``anchor=None`` = the first
    session of the largest connected component. Returns (T (n, 3) with NaN rows for
    sessions not connected to the anchor, inlier mask, residuals (px), anchor)."""
    pairs = np.asarray(pairs, int).reshape(-1, 2)
    estimates = np.asarray(estimates, float).reshape(-1, 3)
    inlier = np.asarray(use, bool).copy()
    if anchor is None:
        anchor = pair_graph_components(n, pairs[inlier])[0][0]
    T = np.full((n, 3), np.nan)
    res = np.full(len(pairs), np.nan)
    for _ in range(max_rounds):
        conn = set(next(c for c in pair_graph_components(n, pairs[inlier]) if anchor in c))
        # initial guess: breadth-first composition from the anchor
        T0 = np.full((n, 3), np.nan)
        T0[anchor] = 0.0
        frontier = [anchor]
        while frontier:
            nxt = []
            for k in np.flatnonzero(inlier):
                i, j = pairs[k]
                if not np.isnan(T0[i, 0]) and np.isnan(T0[j, 0]):
                    T0[j] = _compose(T0[i], estimates[k])
                    nxt.append(j)
                elif not np.isnan(T0[j, 0]) and np.isnan(T0[i, 0]):
                    ti = T0[j, 0] - estimates[k, 0]
                    T0[i] = np.r_[ti, T0[j, 1:] - _Q(ti) @ estimates[k, 1:]]
                    nxt.append(i)
            frontier = nxt
        free = sorted(conn - {anchor})
        edges = [k for k in np.flatnonzero(inlier) if pairs[k, 0] in conn and pairs[k, 1] in conn]

        def unpack(x):
            Tx = T0.copy()
            Tx[free] = x.reshape(-1, 3)
            return Tx

        def resid(x):
            Tx = unpack(x)
            return np.concatenate([_edge_residual(Tx, *pairs[k], estimates[k], lever_px)
                                   for k in edges])
        T = unpack(least_squares(resid, T0[free].ravel(), loss="soft_l1", f_scale=f_scale).x) \
            if free and edges else T0
        res = np.array([np.linalg.norm(_edge_residual(T, *pairs[k], estimates[k], lever_px))
                        if pairs[k, 0] in conn and pairs[k, 1] in conn else np.nan
                        for k in range(len(pairs))])
        drop = inlier & (res > outlier_px)
        if not drop.any():
            break
        inlier &= ~drop
    inlier &= np.isfinite(res)           # edges outside the anchor's component are not used
    return T, inlier, res, int(anchor)


def align_all_pairs(sessions, *, cell_size_px=None, accepted_only=False, max_gap=None,
                    anchor=None, rotations=True, maximal_rotation=30.0, outlier_px=3.0, n_jobs=1):
    """Align sessions to each other from their footprints, without matching cells.

    Args:
        sessions: CNMFe models or result dirs (as for ``register_sessions``), in time
            order; smaller fields of view are zero-padded at the bottom and right.
        cell_size_px: as in ``register_sessions``; None = median FWHM footprint diameter.
        accepted_only: use only ``accepted_mask`` components.
        max_gap: estimate only pairs at most this many sessions apart (None = all pairs).
        anchor: session fixed at the identity; None = first session of the largest
            group of sessions connected by passing pairs.
        rotations, maximal_rotation: CellReg's rotation search (degrees).
        outlier_px: pairs that disagree with the fit by more than this are dropped.
        n_jobs: parallel pair estimates (joblib).

    Returns an ``AllPairsAlignment``; nothing is warped (see ``warp_image``).
    """
    sessions = adjust_fov_size([s if isinstance(s, _Session) else
                                _load_session(s, accepted_only=accepted_only) for s in sessions])
    n = len(sessions)
    dims = sessions[0].dims
    cell_px = float(cell_size_px) if cell_size_px else estimate_cell_size_px(sessions)
    for s in sessions:
        if s.centroids is None:
            s.centroids = compute_centroids(s, _scales(cell_px)["centroid_cell"], True)
    fproj = [footprints_projection(s) for s in sessions]
    cproj = [centroids_projection(s.centroids, s.dims) for s in sessions]
    pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n)
                      if max_gap is None or j - i <= max_gap], int).reshape(-1, 2)
    # the joint fit composes transforms about ONE centre and needs unbiased pair
    # estimates, so the corrected geometry (fix_upstream_slips=True) is used here
    kw = dict(rotations=rotations, maximal_rotation=maximal_rotation, fix_upstream_slips=True)
    if n_jobs == 1:
        est = [estimate_pair(fproj[i], cproj[i], fproj[j], cproj[j], sessions[j].centroids,
                             dims, cell_px, **kw) for i, j in pairs]
    else:
        from joblib import Parallel, delayed
        est = Parallel(n_jobs=n_jobs)(
            delayed(estimate_pair)(fproj[i], cproj[i], fproj[j], cproj[j], sessions[j].centroids,
                                   dims, cell_px, **kw) for i, j in pairs)
    return fit_pair_estimates(n, pairs, est, dims, cell_px, anchor=anchor, outlier_px=outlier_px)


def apply_all_pairs(sessions, ap, cell_px, fix_upstream_slips=False):
    """Warp every placed session with its all-pairs transform (about the one centre
    the fit uses) and recompute its centroids; unplaced sessions keep their
    footprints and are listed as bad. Returns (sessions, Alignment)."""
    S = _scales(cell_px)
    center = rotation_centers(ap.dims, True)[0]
    out = list(sessions)
    n = len(sessions)
    for s in range(n):
        if s in ap.unplaced or s == ap.anchor:
            continue
        th, dy, dx = (float(v) for v in ap.transforms[s])
        radius = S["rotate_radius"] if th else S["translate_radius"]
        corr = transform_session(sessions[s], th, (dy, dx), center, radius)
        corr.centroids = compute_centroids(corr, S["centroid_cell"], fix_upstream_slips)
        out[s] = corr
    T = np.nan_to_num(ap.transforms, nan=0.0)
    best = np.full(n, np.nan)
    for (i, j), c, ok in zip(ap.pairs, ap.max_cross_correlation, ap.inlier):
        if ok:
            best[i] = np.nanmax([best[i], c])
            best[j] = np.nanmax([best[j], c])
    return out, Alignment(T[:, 1:].copy(), T[:, 0].copy(), best, np.zeros(n, bool), list(ap.unplaced))


def fit_pair_estimates(n, pairs, est, dims, cell_size_px, *, anchor=None, outlier_px=3.0):
    """``AllPairsAlignment`` from a list of ``PairEstimate`` (one per row of ``pairs``)."""
    pairs = np.asarray(pairs, int).reshape(-1, 2)
    estimates = np.array([(e.theta, *e.shift) for e in est], float).reshape(-1, 3)
    passed = np.array([e.passed for e in est], bool)
    lever = min(dims) / 2.0
    if passed.any():
        T, inlier, res, anchor = fit_session_transforms(n, pairs, estimates, passed, anchor=anchor,
                                                        lever_px=lever, outlier_px=outlier_px)
    else:
        anchor = 0 if anchor is None else anchor
        T, inlier, res = np.full((n, 3), np.nan), passed.copy(), np.full(len(pairs), np.nan)
        T[anchor] = 0.0
    return AllPairsAlignment(
        T, anchor, [k for k in range(n) if np.isnan(T[k, 0])], pairs, estimates,
        np.array([e.max_cross_correlation for e in est]),
        np.array([e.median_cross_correlation for e in est]), passed, inlier, res, lever,
        tuple(dims), float(cell_size_px))


# ---------------------------------------------------------------------------
# pair similarities (compute_data_distribution)
# ---------------------------------------------------------------------------

def _corr2(ra, rb, n_pix):
    """corr2 of two whole footprint images, from (indices, values, sum, sumsq):
    Pearson over all ``n_pix`` pixels, zeros included. None if one is empty."""
    ia, va, sa, qa = ra
    ib, vb, sb, qb = rb
    if va.size == 0 or vb.size == 0:
        return None
    _, xa, xb = np.intersect1d(ia, ib, assume_unique=True, return_indices=True)
    sab = float(np.dot(va[xa], vb[xb]))
    ma, mb = sa / n_pix, sb / n_pix
    num = sab - n_pix * ma * mb
    den = np.sqrt((qa - n_pix * ma * ma) * (qb - n_pix * mb * mb))
    return float(num / den) if den > 0 else 0.0


@dataclass
class _AllToAll:
    """``all_to_all_*``: [session a][cell k][session b] -> neighbour indexes in b,
    their spatial correlations and centroid distances (None if not computed)."""

    idx: list
    corr: list
    dist: list


def compute_data_distribution(sessions, maximal_distance, active):
    """Neighbour pairs within ``maximal_distance`` (strict) between every ordered
    pair of active sessions. Returns (all_to_all, neighbour corr, neighbour dist);
    as in the source, the neighbour arrays hold every ordered pair (each unordered
    pair twice) and skip empty footprints."""
    n = len(sessions)
    n_pix = sessions[0].dims[0] * sessions[0].dims[1]
    rows = [s.rows() for s in sessions]
    idx = [[[None] * n for _ in range(s.K)] for s in sessions]
    corr = [[[None] * n for _ in range(s.K)] for s in sessions]
    dist = [[[None] * n for _ in range(s.K)] for s in sessions]
    finite = [np.flatnonzero(np.all(np.isfinite(s.centroids), axis=1)) if s.K else np.zeros(0, int)
              for s in sessions]                       # a NaN centroid is never a neighbour
    trees = [cKDTree(s.centroids[f]) if f.size else None for s, f in zip(sessions, finite)]
    cache = {}
    nb_corr, nb_dist = [], []
    for a in active:
        for b in active:
            if a == b:
                continue
            for k in range(sessions[a].K):
                c = sessions[a].centroids[k]
                cand = np.asarray(sorted(finite[b][trees[b].query_ball_point(c, r=maximal_distance)]),
                                  dtype=int) if trees[b] is not None and np.all(np.isfinite(c)) \
                    else np.zeros(0, int)
                d = np.hypot(*(sessions[b].centroids[cand] - c).T) if cand.size else np.zeros(0)
                keep = d < maximal_distance
                cand, d = cand[keep], d[keep]
                cv = np.zeros(cand.size)
                for l, j in enumerate(cand):
                    key = (a, k, b, int(j)) if a < b else (b, int(j), a, k)
                    if key not in cache:
                        cache[key] = _corr2(rows[key[0]][key[1]], rows[key[2]][key[3]], n_pix)
                    r = cache[key]
                    if r is None:
                        continue
                    cv[l] = r
                    nb_corr.append(r)
                    nb_dist.append(d[l])
                idx[a][k][b], corr[a][k][b], dist[a][k][b] = cand, cv, d
    nb_corr = np.asarray(nb_corr, float)
    nb_dist = np.asarray(nb_dist, float)
    nb_corr[nb_corr == 0] = 1e-10
    nb_corr[nb_corr == 1] = 1 - 1e-10
    nb_dist[nb_dist == 0] = 1e-10
    return _AllToAll(idx, corr, dist), nb_corr, nb_dist


def number_of_bins(n_cells_first, n_sessions, maximal_distance):
    """``estimate_number_of_bins``."""
    nb = min(10 * _mround((40 + n_cells_first / 600 * n_sessions ** 2) / 10), 100)
    d = np.linspace(0, maximal_distance, 2 * nb + 1)[1::2]
    c = np.linspace(0, 1, 2 * nb + 1)[1::2]
    return nb, d, c


# ---------------------------------------------------------------------------
# P_same models
# ---------------------------------------------------------------------------

@dataclass
class Model:
    """One P_same model on bin centres."""

    feature: str                     # "spatial" | "centroid"
    centers: np.ndarray
    p_same: np.ndarray
    same: np.ndarray                 # normalised same-cell density on the centres
    different: np.ndarray
    distribution: np.ndarray
    mse: float
    intersection: float              # feature value where the weighted densities cross
    params: dict = field(default_factory=dict)
    false_positive: float = float("nan")
    false_negative: float = float("nan")

    def lookup(self, values) -> np.ndarray:
        """``compute_p_same``: P_same of the nearest bin centre."""
        v = np.asarray(values, dtype=float)
        if v.size == 0:
            return np.zeros(0)
        i = np.argmin(np.abs(self.centers[None, :] - v.reshape(-1, 1)), axis=1)
        return self.p_same[i]


def _norm_density(x, centers, nb):
    width = (centers[1] - centers[0]) + (centers[-1] - centers[0])
    s = x.sum()
    return x / s * (nb / width) if s > 0 else x


def _mse(dist, weighted, centers, nb):
    width = (centers[1] - centers[0]) + (centers[-1] - centers[0])
    return float(np.sum(np.abs((dist - weighted) * width / nb)) / 2)


def _estimate_beta_mixture_params(w, data, maximal=1.0):
    """``estimate_beta_mixture_params``: weighted moments, then 100 Newton steps on
    the beta likelihood equations."""
    data = np.where(data == maximal, 0.9999 * maximal, data)
    sw = w.sum()
    if not sw > 0:
        return float("nan"), float("nan")
    g1 = float((w * np.log(data / maximal)).sum() / sw)
    g2 = float((w * np.log((maximal - data) / maximal)).sum() / sw)
    m = float((w * data).sum() / sw)
    v = float((w * (data - m) ** 2).sum() / sw)
    xbar, ssq = m / maximal, v / maximal ** 2
    if not (sw > 0 and ssq > 0):          # MATLAB gives NaN here -> the caller's mle fallback
        return float("nan"), float("nan")
    x = np.array([xbar * ((xbar * (1 - xbar) / ssq) - 1),
                  (1 - xbar) * ((xbar * (1 - xbar) / ssq) - 1)], dtype=float)
    for _ in range(100):
        p, q = x
        grad = np.array([psi(p) - psi(p + q) - g1, psi(q) - psi(p + q) - g2])
        t = polygamma(1, p + q)
        hess = np.array([[polygamma(1, p) - t, -t], [-t, polygamma(1, q) - t]])
        try:
            x = x - np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            return float("nan"), float("nan")
        if not np.all(np.isfinite(x)):
            return float("nan"), float("nan")
    return float(x[0]), float(x[1])


def fit_spatial_correlation_model(nb_corr, centers, fix_upstream_slips=False) -> Model:
    """``compute_spatial_correlations_model``. NaN P_same (0/0) stays NaN as in the
    source, or becomes 0 with ``fix_upstream_slips``."""
    nb = len(centers)
    c = nb_corr[(nb_corr >= 0) & (nb_corr <= 1)]
    data = 1 - c
    hi, lo = c >= 0.7, c < 0.7
    mu, sigma = _lognfit(data[hi]) if hi.sum() > 1 else (np.log(0.1), 1.0)
    p, q = _betafit(c[lo]) if lo.sum() > 1 else (2.0, 5.0)
    PI = 0.5
    for _ in range(100):
        ls = _lognpdf(data, mu, sigma)
        bd = _betapdf(data, p, q)
        with np.errstate(invalid="ignore", divide="ignore"):
            asg = PI * ls / (PI * ls + (1 - PI) * bd)
        PI = float(asg.sum() / len(asg))
        mu = float((asg * np.log(data)).sum() / asg.sum())
        sigma = float(np.sqrt((asg * (np.log(data) - mu) ** 2).sum() / asg.sum()))
        p, q = _estimate_beta_mixture_params(1 - asg, data, 1.0)
        if np.isnan(p):
            break                                      # NaN stays NaN in the source's loop
    if np.isnan(p):
        # the source falls back to an mle fit of the whole mixture
        from scipy.optimize import minimize

        def nll(th):
            pdf = th[0] * _lognpdf(data, th[1], th[2]) + (1 - th[0]) * _betapdf(data, th[3], th[4])
            return -np.sum(np.log(pdf + 1e-300))
        m0, s0 = _lognfit(data[hi]) if hi.sum() > 1 else (np.log(0.1), 1.0)
        a0, b0 = _betafit(c[lo]) if lo.sum() > 1 else (2.0, 5.0)
        r = minimize(nll, [0.5, m0, s0, a0, b0], method="L-BFGS-B",
                     bounds=[(0, 1), (None, None), (1e-6, None), (1e-6, None), (1e-6, None)])
        PI, mu, sigma, p, q = (float(v) for v in r.x)
    same = _norm_density(_lognpdf(1 - centers, mu, sigma), centers, nb)
    diff = _norm_density(_betapdf(1 - centers, p, q), centers, nb)
    same = same * _sigmoid(centers, 20.0, centers.min() + 0.5)
    same[:_mround(nb / 10)] = 0.0
    weighted = PI * same + (1 - PI) * diff
    dist = _norm_density(_hist_centers(c, centers), centers, nb)
    mse = _mse(dist, weighted, centers, nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        ps = PI * same / (PI * same + (1 - PI) * diff)
    if fix_upstream_slips:
        ps = np.nan_to_num(ps, nan=0.0)
    small = np.flatnonzero(same < 0.001 * same.max())
    if small.size:
        ps[small] = ps[small] * _sigmoid(np.arange(1, small.size + 1), 0.05 * small.size,
                                         0.8 * small.size)
    rng = np.flatnonzero(same > 0.001 * same.max())
    rng = rng[rng != nb - 1]
    if rng.size:
        i = int(np.argmin(np.abs(PI * same[rng] - (1 - PI) * diff[rng])))
        inter = _mround(100 * centers[rng[0] + i]) / 100
    else:
        inter = 0.65
    return Model("spatial", centers, ps, same, diff, dist, mse, float(inter),
                 {"PI": PI, "mu": mu, "sigma": sigma, "p": p, "q": q})


def fit_centroid_distance_model(nb_dist, centers, cell_px, fix_upstream_slips=False) -> Model:
    """``compute_centroid_distances_model``, transcribed in CellReg's micron units:
    ``mpp = 12 / cell_px`` stands for microns_per_pixel (see ``_scales``). As in the
    source, the start value of the lognormal is fitted to PIXEL distances, the
    model is p * (lognormal + (1 - p) * b * x * sigmoid), and the intersection is
    rounded to 0.01 um before it is converted back to pixels. Distances, centres
    and the intersection of the returned Model are in pixels; ``params`` are in
    microns. lsqcurvefit (trust-region-reflective, TolFun = TolX = 1e-6,
    MaxFunEvals 2000 including the 6 finite-difference evaluations per Jacobian)
    is approximated by scipy's 'trf' with the same tolerances and 2000 // 7
    evaluations."""
    S = _scales(cell_px)
    mpp = 1.0 / S["um"]
    nb = len(centers)
    xc = mpp * centers
    width = mpp * (centers[1] - centers[0]) + mpp * (centers[-1] - centers[0])
    hist = _hist_centers(nb_dist, centers)
    dist = hist / hist.sum() * (nb / width)
    data_to_fit = nb_dist[mpp * nb_dist < 9]
    mu0, s0 = _lognfit(data_to_fit) if data_to_fit.size > 1 else (np.log(3.0), 0.5)
    p_0 = data_to_fit.size / max(1, nb_dist.size)
    half = _mround(nb / 2) - 1
    b_0 = (dist[-1] - dist[half]) / (mpp * (centers[-1] - centers[half]))
    b_0 = b_0 / (1 - p_0) if p_0 < 1 else b_0
    lb = np.array([0, -np.inf, 0, 0, 0, 0], float)
    ub = np.array([1, np.inf, np.inf, np.inf, np.inf, np.inf], float)
    x0 = np.clip(np.array([p_0, mu0, s0, 1.0, 6.0, b_0], float), lb, ub)

    def F(x, xd):
        return x[0] * (_lognpdf(xd, x[1], max(x[2], 1e-12))
                       + (1 - x[0]) * x[5] * xd / (1 + np.exp(-x[3] * (xd - x[4]))))
    with np.errstate(over="ignore"):
        x = least_squares(lambda th: F(th, xc) - dist, x0, bounds=(lb, ub), method="trf",
                          ftol=1e-6, xtol=1e-6, gtol=1e-6, max_nfev=2000 // 7).x
        same = _lognpdf(xc, x[1], max(x[2], 1e-12))
        diff = x[5] * xc / (1 + np.exp(-x[3] * (xc - x[4])))
    same = same / same.sum() * (nb / width) if same.sum() > 0 else same
    diff = diff / diff.sum() * (nb / width) if diff.sum() > 0 else diff
    weighted = x[0] * same + (1 - x[0]) * diff
    rng = np.flatnonzero((centers > 1 / mpp) & (centers < 10 / mpp))
    if rng.size:
        i = int(np.argmin(np.abs(x[0] * same[rng] - (1 - x[0]) * diff[rng])))
        inter = _mround(100 * mpp * centers[rng[0] + i]) / 100 / mpp
    else:
        inter = 5.0 / mpp
    mse = float(np.sum(np.abs((dist - weighted) * width / nb)) / 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        ps = x[0] * same / (x[0] * same + (1 - x[0]) * diff)
    if fix_upstream_slips:
        ps = np.nan_to_num(ps, nan=0.0)
    ps[0] = ps[1]   # source: "avoid p_same going to 0 at 0 distance because of the lognormal"
    return Model("centroid", centers, ps, same, diff, dist, mse, float(inter),
                 {"p": x[0], "mu": x[1], "sigma": x[2], "a": x[3], "c": x[4], "b": x[5],
                  "microns_per_pixel": mpp})


def _min_index(v):
    """MATLAB ``[~, i] = min(v)``: NaN skipped, first index on ties, 0 if all NaN."""
    v = np.asarray(v, float)
    return 0 if np.all(np.isnan(v)) else int(np.nanargmin(v))


def choose_best_model(centroid: Model, spatial: Model, fix_upstream_slips=False) -> str:
    """``choose_best_model``: cost = FP + FN + MSE at the P_same ~ 0.5 bin. As in
    the source the distance model is measured at the CORRELATION model's index;
    with ``fix_upstream_slips`` at its own."""
    ic = _min_index(np.abs(0.5 - spatial.p_same))
    spatial.false_positive = float(spatial.different[ic:].sum() / spatial.different.sum())
    spatial.false_negative = float(spatial.same[:ic + 1].sum() / spatial.same.sum())
    idd = _min_index(np.abs(0.5 - centroid.p_same)) if fix_upstream_slips else ic
    centroid.false_positive = float(centroid.different[:idd + 1].sum() / centroid.different.sum())
    centroid.false_negative = float(centroid.same[idd:].sum() / centroid.same.sum())
    cost_c = centroid.false_positive + centroid.false_negative + centroid.mse
    cost_s = spatial.false_positive + spatial.false_negative + spatial.mse
    if np.isnan(cost_s) or (not np.isnan(cost_c) and cost_c <= cost_s):
        return "centroid"                              # MATLAB min(): NaN skipped, first on ties
    return "spatial"


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

def initial_registration(sessions, active, feature, threshold, maximal_distance):
    """``initial_registration_centroid_distances`` / ``_spatial_correlations``.
    Sessions in order build a growing list of registered cells (each row keeps the
    centroid and footprint of the cell that created it). A new cell goes to its
    best candidate within ``maximal_distance`` if it passes ``threshold`` and that
    session's slot is free, or held by a worse match (which then moves to a new
    row); otherwise it starts a new row."""
    n = len(sessions)
    n_pix = sessions[0].dims[0] * sessions[0].dims[1]
    rows_fp = {s: sessions[s].rows() for s in active} if feature == "spatial" else {}
    first = active[0]
    M = [{first: k} for k in range(sessions[first].K)]
    score = [{first: 0.0} for _ in M]
    ref_cent = [sessions[first].centroids[k] for k in range(sessions[first].K)]
    ref_src = [(first, k) for k in range(sessions[first].K)]
    for s in active[1:]:
        for k in range(sessions[s].K):
            c = sessions[s].centroids[k]
            rc = np.asarray(ref_cent) if ref_cent else np.zeros((0, 2))
            d = np.hypot(rc[:, 0] - c[0], rc[:, 1] - c[1]) if len(rc) else np.zeros(0)
            cand = np.flatnonzero(d < maximal_distance)
            placed = False
            if cand.size:
                if feature == "centroid":
                    vals = d[cand]
                    j = int(np.argmin(vals))
                    best = float(vals[j])
                    ok = best <= threshold
                else:
                    vals = np.array([(_corr2(rows_fp[ref_src[i][0]][ref_src[i][1]],
                                             rows_fp[s][k], n_pix) or 0.0) for i in cand])
                    j = int(np.argmax(vals))
                    best = float(vals[j])
                    ok = best >= threshold
                if ok:
                    i = int(cand[j])
                    if s not in M[i]:
                        M[i][s] = k
                        score[i][s] = best
                        placed = True
                    elif (best < score[i][s]) if feature == "centroid" else (best > score[i][s]):
                        old = M[i][s]
                        M.append({s: old})
                        score.append({s: 0.0})
                        ref_cent.append(sessions[s].centroids[old])
                        ref_src.append((s, old))
                        M[i][s] = k
                        score[i][s] = best
                        placed = True
            if not placed:
                M.append({s: k})
                # centroid version: a rejected or out-competed cell's new row keeps its
                # distance (so a closer cell can still switch in); a cell with no
                # candidate, and every new row of the spatial version, stores 0
                score.append({s: best if (feature == "centroid" and cand.size) else 0.0})
                ref_cent.append(c)
                ref_src.append((s, k))
    cmap = np.full((len(M), n), -1, dtype=int)
    for r, row in enumerate(M):
        for s, k in row.items():
            cmap[r, s] = k
    return cmap


def _psame_table(a2a, a2a_ps):
    """{(session a, cell k, session b, cell j): P_same} for every neighbour pair."""
    t = {}
    for a, per_a in enumerate(a2a.idx):
        for k, per_k in enumerate(per_a):
            for b, ids in enumerate(per_k):
                if ids is None:
                    continue
                ps = a2a_ps[a][k][b]
                for j, p in zip(ids, ps):
                    t[(a, k, b, int(j))] = float(p)
    return t


def cluster_cells(cmap, table, sessions, maximal_distance, threshold=0.5,
                  max_iterations=10, num_changes_thresh=10):
    """``cluster_cells`` with decision_type 'Maximal similarity'. ``table`` maps
    (a, k, b, j) -> P_same for neighbour pairs; a pair outside the table (not
    neighbours) contributes nothing, as an empty entry does in the source."""
    n = cmap.shape[1]
    M = [{s: int(k) for s, k in enumerate(row) if k >= 0} for row in cmap]
    cdist = 1.7 * maximal_distance

    def cent(row):
        return np.mean([sessions[s].centroids[k] for s, k in row.items()], axis=0)

    changes = [-1]
    it = 0
    while (changes[it] > num_changes_thresh or changes[it] == -1) and it + 1 < max_iterations:
        it += 1
        changes.append(0)
        C = [cent(r) for r in M]
        for s in range(n):
            cluster_ind = [i for i, r in enumerate(M) if s in r]
            for ci in cluster_ind:
                if s not in M[ci]:
                    continue
                k = M[ci][s]
                kc = sessions[s].centroids[k]
                Ca = np.asarray(C)
                cand = np.flatnonzero(np.hypot(Ca[:, 0] - kc[0], Ca[:, 1] - kc[1]) < cdist)
                if cand.size == 0:
                    continue
                max_in = np.zeros(cand.size)
                for mi, cl in enumerate(cand):
                    for sm, km in M[cl].items():
                        if sm != s:
                            v = table.get((s, k, sm, km))
                            if v is not None and v > max_in[mi]:
                                max_in[mi] = v
                bi = int(np.argmax(max_in))
                best, maxsim = int(cand[bi]), float(max_in[bi])
                if not any(sm != s for sm in M[best]):
                    continue
                if maxsim < threshold and len(M[ci]) > 1:
                    del M[ci][s]
                    M.append({s: k})
                    C.append(np.array(kc, float))
                    changes[it] += 1
                elif maxsim >= threshold and best != ci:
                    if s in M[best]:
                        tk = M[best][s]
                        tsim = sum(table.get((s, tk, sm, km), 0.0)
                                   for sm, km in M[best].items() if sm != s)
                        if maxsim > tsim:
                            M[best][s] = k
                            M.append({s: tk})
                            C.append(np.array(sessions[s].centroids[tk], float))
                            del M[ci][s]
                            changes[it] += 1
                    else:
                        del M[ci][s]
                        M[best][s] = k
                        changes[it] += 1
        M = [r for r in M if r]
        # merge clusters that share no session and hold a pair above threshold
        C = np.asarray([cent(r) for r in M])
        T = [dict(r) for r in M]
        for i in range(len(T)):
            if not T[i]:
                continue
            dv = np.hypot(C[:, 0] - C[i, 0], C[:, 1] - C[i, 1])
            for j in np.flatnonzero(dv < cdist):
                if j == i or not T[j] or (T[i].keys() & T[j].keys()):
                    continue
                vals = [table.get((sa, ka, sb, kb), 0.0)
                        for sa, ka in T[i].items() for sb, kb in T[j].items()]
                best = max((v for v in vals if not np.isnan(v)), default=0.0)   # max() skips NaN
                if best > threshold:
                    T[i].update(T[j])
                    T[j] = {}
                    C[j] = (1e9, 1e9)
                    changes[it] += 1
        M = [r for r in T if r]
    out = np.full((len(M), n), -1, dtype=int)
    for r, row in enumerate(M):
        for s, k in row.items():
            out[r, s] = k
    return out


def compute_scores(cmap, a2a, a2a_ps):
    """``compute_scores``: cell, true-positive, true-negative and exclusivity scores
    per cluster, and the P_same of the registered pairs (one per unordered pair)."""
    G, n = cmap.shape
    cs, pos, neg, exc = (np.full(G, np.nan) for _ in range(4))
    pairs = []
    for g in range(G):
        good = good_p = good_n = good_e = 0.0
        ncmp = ncmp_p = ncmp_n = 0
        for m in range(n):
            km = cmap[g, m]
            if km < 0:
                continue
            for k in range(n):
                if k == m:
                    continue
                ncmp += 1
                ids = a2a.idx[m][km][k]
                ps = a2a_ps[m][km][k]
                ids = np.zeros(0, int) if ids is None else ids
                ps = np.zeros(0) if ps is None else np.asarray(ps, float)
                if cmap[g, k] < 0:
                    ncmp_n += 1
                    v = 1.0 - ps.sum() if (ids.size and ps.size) else 1.0
                    good += v
                    good_n += v
                else:
                    ncmp_p += 1
                    hit = np.flatnonzero(ids == cmap[g, k])
                    if ps.size and hit.size:
                        tp = float(ps[hit[0]])
                        rest = np.delete(ps, hit[0])
                        if m < k and not np.isnan(tp):
                            pairs.append(tp)
                    else:
                        tp, rest = 0.0, ps
                    good_p += tp
                    if rest.size == 0:
                        good += tp
                        good_e += 1
                    else:
                        good += tp - rest.sum()
                        good_e += 1 - rest.sum()
        if ncmp_p:
            pos[g] = good_p / ncmp_p
            exc[g] = good_e / ncmp_p
        if ncmp_n:
            neg[g] = good_n / ncmp_n
        if ncmp:
            cs[g] = good / ncmp
    return cs, pos, neg, exc, np.asarray(pairs, float)


# ---------------------------------------------------------------------------
# result + entry point
# ---------------------------------------------------------------------------

@dataclass
class CellRegResult:
    """Output of :func:`register_sessions`. ``cell_to_index_map[g, s]`` is the
    component index in session ``s`` of global cell ``g`` (-1 = absent). Sessions
    that failed the alignment test (``bad_sessions``) have no entries unless
    ``params["on_bad_alignment"] == "keep"``."""

    cell_to_index_map: np.ndarray
    n_sessions: int
    transforms: np.ndarray            # (n, 3) (dy, dx, theta): offset after rotation; correction angle
                                      # (applied about params["rotation_center"])
    aligned_centroids: list
    dims: tuple
    bad_sessions: list
    max_cross_correlation: np.ndarray
    model_feature: str
    p_same_threshold: float
    cell_size_px: float
    maximal_distance_px: float
    cell_scores: np.ndarray
    cell_scores_positive: np.ndarray
    cell_scores_negative: np.ndarray
    cell_scores_exclusive: np.ndarray
    p_same_registered_pairs: np.ndarray
    models: dict = field(default_factory=dict)   # {"centroid": Model, "spatial": Model}
    params: dict = field(default_factory=dict)

    @property
    def n_global(self) -> int:
        return int(self.cell_to_index_map.shape[0])

    def n_registered(self, min_sessions: int = 2) -> int:
        return int(((self.cell_to_index_map >= 0).sum(axis=1) >= min_sessions).sum())

    @property
    def uncertain_fraction(self) -> float:
        """Fraction of registered pairs with 0.05 < P_same < 0.95."""
        p = self.p_same_registered_pairs
        return float(np.mean((p > 0.05) & (p < 0.95))) if p.size else float("nan")

    def save(self, output_dir) -> None:
        d = Path(output_dir)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "cell_to_index_map.npy", self.cell_to_index_map)
        np.save(d / "transforms.npy", self.transforms)
        np.savez(d / "aligned_centroids.npz",
                 **{f"session_{s}": c for s, c in enumerate(self.aligned_centroids)})
        np.savez(d / "scores.npz", cell=self.cell_scores, positive=self.cell_scores_positive,
                 negative=self.cell_scores_negative, exclusive=self.cell_scores_exclusive,
                 p_same_registered_pairs=self.p_same_registered_pairs,
                 max_cross_correlation=self.max_cross_correlation)
        models = {k: {"feature": m.feature, "centers": m.centers.tolist(),
                      "p_same": m.p_same.tolist(), "intersection": m.intersection,
                      "mse": m.mse, "false_positive": m.false_positive,
                      "false_negative": m.false_negative,
                      "params": {kk: float(vv) for kk, vv in m.params.items()}}
                  for k, m in self.models.items()}
        info = {"n_sessions": self.n_sessions, "n_global": self.n_global,
                "dims": list(self.dims), "bad_sessions": [int(b) for b in self.bad_sessions],
                "model_feature": self.model_feature, "p_same_threshold": self.p_same_threshold,
                "cell_size_px": self.cell_size_px, "maximal_distance_px": self.maximal_distance_px,
                "models": models, "params": self.params}
        (d / "cellreg_info.json").write_text(json.dumps(info, indent=2))

    @classmethod
    def load(cls, output_dir) -> "CellRegResult":
        d = Path(output_dir)
        info = json.loads((d / "cellreg_info.json").read_text())
        with np.load(d / "aligned_centroids.npz") as z:
            cents = [z[f"session_{s}"] for s in range(info["n_sessions"])]
        with np.load(d / "scores.npz") as sc:
            sc = {k: sc[k] for k in sc.files}
        models = {k: Model(m["feature"], np.asarray(m["centers"]), np.asarray(m["p_same"]),
                           np.zeros(0), np.zeros(0), np.zeros(0), m["mse"], m["intersection"],
                           m["params"], m["false_positive"], m["false_negative"])
                  for k, m in info["models"].items()}
        return cls(np.load(d / "cell_to_index_map.npy"), info["n_sessions"],
                   np.load(d / "transforms.npy"), cents, tuple(info["dims"]),
                   info["bad_sessions"], sc["max_cross_correlation"], info["model_feature"],
                   info["p_same_threshold"], info["cell_size_px"], info["maximal_distance_px"],
                   sc["cell"], sc["positive"], sc["negative"], sc["exclusive"],
                   sc["p_same_registered_pairs"], models, info["params"])


# arguments that only the previous implementation (cellreg_legacy) takes
_LEGACY_ONLY_KWARGS = frozenset({
    "microns_per_pixel", "max_distance_um", "max_distance_px", "align", "dist_thr_um",
    "dist_thr_px", "corr_thr", "corr_weight", "probabilistic", "p_same_thr",
    "registration_approach", "psame_feature", "clustering", "max_shift_px",
    "angle_range_deg", "angle_step_deg"})


def _legacy_arguments_go_to_cellreg_legacy(fn):
    @functools.wraps(fn)
    def wrapper(sessions, **kwargs):
        old = sorted(_LEGACY_ONLY_KWARGS.intersection(kwargs))
        if old:
            warnings.warn(f"register_sessions got arguments of the previous implementation "
                          f"{old}; running minicnmfe.cellreg_legacy.register_sessions",
                          FutureWarning, stacklevel=2)
            from minicnmfe import cellreg_legacy
            return cellreg_legacy.register_sessions(sessions, **kwargs)
        return fn(sessions, **kwargs)
    return wrapper


@_legacy_arguments_go_to_cellreg_legacy
def register_sessions(
    sessions: "list[CNMFe | str | Path]",
    *,
    cell_size_px: float | None = None,
    maximal_distance_px: float | None = None,
    reference: int = 0,
    accepted_only: bool = False,
    alignment: str = "translations_and_rotations",
    maximal_rotation: float = 30.0,
    p_same_threshold: float = 0.5,
    model: str = "best",
    on_bad_alignment: str = "exclude",
    fix_upstream_slips: bool = False,
    all_pairs: "AllPairsAlignment | None" = None,
    max_gap: int | None = None,
    n_jobs: int = 1,
    return_internals: bool = False,
):
    """Register cells across sessions with CellReg's probabilistic pipeline.

    Args:
        sessions: CNMFe models or result dirs; smaller fields of view are
            zero-padded at the bottom and right (``adjust_FOV_size``).
        cell_size_px: pixel equivalent of CellReg's 12 um typical cell size; every
            CellReg length is scaled from it. None = median FWHM footprint diameter.
        maximal_distance_px: pair search radius; None = CellReg's 14 um equivalent
            (14/12 * cell_size_px).
        reference: index of the reference session for alignment.
        accepted_only: use only ``accepted_mask`` components (indices in the map
            are then positions within that subset).
        alignment: "translations" or "translations_and_rotations" (CellReg's
            align_images against ``reference``; demo default), or "all_pairs": the
            fields of view are first aligned with ``align_all_pairs`` (or the given
            ``all_pairs``), each placed session is warped with its fitted transform
            (one rotation centre, the rotation applied whenever it is non-zero), and
            unplaced sessions are left out (listed in ``bad_sessions``).
        maximal_rotation: rotation search range in degrees (demo: 30).
        p_same_threshold: registration threshold on P_same (demo: 0.5).
        model: "best" (choose_best_model), "spatial" or "centroid".
        on_bad_alignment: "exclude" (as CellReg: a session whose footprint
            projection does not correlate with the reference is left out) or
            "keep" (aligned and registered anyway; still listed in bad_sessions).
        fix_upstream_slips: False (default) reproduces the source, slips included;
            True corrects the slips listed in the module docstring.
        all_pairs: a precomputed ``AllPairsAlignment`` for the same sessions
            (alignment="all_pairs"), e.g. one fitted on all components and reused
            with ``accepted_only``.
        max_gap, n_jobs: passed to ``align_all_pairs`` when ``all_pairs`` is None.
        return_internals: also return the intermediate objects (for diagnostics).
    """
    if len(sessions) < 2:
        raise ValueError("register_sessions needs at least 2 sessions.")
    S0 = adjust_fov_size([_load_session(s, accepted_only=accepted_only) for s in sessions])
    dims0 = S0[0].dims
    cell_px = float(cell_size_px) if cell_size_px else estimate_cell_size_px(S0)
    sc = _scales(cell_px)
    maxd = float(maximal_distance_px) if maximal_distance_px else sc["maximal_distance"]
    for s in S0:
        s.centroids = compute_centroids(s, sc["centroid_cell"], fix_upstream_slips)
    raw_centroids = [s.centroids.copy() for s in S0]

    if alignment == "all_pairs":
        ap = all_pairs if all_pairs is not None else align_all_pairs(
            S0, cell_size_px=cell_px, max_gap=max_gap, maximal_rotation=maximal_rotation, n_jobs=n_jobs)
        if len(ap.transforms) != len(S0) or tuple(ap.dims) != tuple(dims0):
            raise ValueError("all_pairs does not belong to these sessions (count or FOV size differ)")
        aligned, al = apply_all_pairs(S0, ap, cell_px, fix_upstream_slips)
    else:
        aligned, al = align_sessions(S0, reference=reference, cell_px=cell_px,
                                     rotations=(alignment == "translations_and_rotations"),
                                     maximal_rotation=maximal_rotation,
                                     correct_bad=(on_bad_alignment == "keep"),
                                     fix_upstream_slips=fix_upstream_slips)
    if on_bad_alignment == "keep":
        active = list(range(len(S0)))
    else:
        active = [s for s in range(len(S0)) if s not in al.bad_sessions]
    if len(active) < 2:
        raise ValueError(f"only {len(active)} session(s) aligned to the reference")

    nbins, dcent, ccent = number_of_bins(S0[0].K, len(S0), maxd)
    a2a, nb_corr, nb_dist = compute_data_distribution(aligned, maxd, active)
    m_c = fit_centroid_distance_model(nb_dist, dcent, cell_px, fix_upstream_slips)
    m_s = fit_spatial_correlation_model(nb_corr, ccent, fix_upstream_slips)
    best = choose_best_model(m_c, m_s, fix_upstream_slips)
    feature = best if model == "best" else model
    chosen = m_c if feature == "centroid" else m_s

    n = len(S0)
    a2a_ps = [[[None] * n for _ in range(s.K)] for s in aligned]
    for a in active:
        for k in range(aligned[a].K):
            for b in active:
                if a != b and a2a.idx[a][k][b] is not None:
                    vals = a2a.dist[a][k][b] if feature == "centroid" else a2a.corr[a][k][b]
                    a2a_ps[a][k][b] = chosen.lookup(vals)
    table = _psame_table(a2a, a2a_ps)

    cmap0 = initial_registration(aligned, active, feature, chosen.intersection, maxd)
    cmap = cluster_cells(cmap0, table, aligned, maxd, threshold=p_same_threshold)
    cs, pos, neg, exc, pairs = compute_scores(cmap, a2a, a2a_ps)

    transforms = np.column_stack([al.shifts, al.rotations])
    params = {"cell_size_px": cell_px, "maximal_distance_px": maxd, "reference": reference,
              "accepted_only": accepted_only, "alignment": alignment,
              "maximal_rotation": maximal_rotation, "p_same_threshold": p_same_threshold,
              "model": model, "on_bad_alignment": on_bad_alignment, "number_of_bins": int(nbins),
              "fix_upstream_slips": bool(fix_upstream_slips),
              "rotation_center": list(rotation_centers(dims0, fix_upstream_slips or alignment == "all_pairs")[1]),
              "intersection_centroid_px": m_c.intersection,
              "intersection_spatial": m_s.intersection}
    res = CellRegResult(cmap, n, transforms, [s.centroids for s in aligned], dims0,
                        list(al.bad_sessions), al.max_cross_correlation, feature,
                        p_same_threshold, cell_px, maxd, cs, pos, neg, exc, pairs,
                        {"centroid": m_c, "spatial": m_s}, params)
    if return_internals:
        return res, {"sessions": aligned, "raw_centroids": raw_centroids, "alignment": al,
                     "chosen": chosen, "a2a": a2a, "a2a_ps": a2a_ps,
                     "table": table, "nb_corr": nb_corr, "nb_dist": nb_dist,
                     "initial_map": cmap0, "active": active}
    return res
