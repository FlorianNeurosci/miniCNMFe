# Seeds: the CORR and PNR images

Source: `minicnmfe/preprocess.py`. These two summary images are how CNMF-E finds
candidate neuron locations. In the main pipeline they are computed *inside* the
greedy initialization loop (see [Initialization](initialization.md)); the
standalone `correlation_pnr` function documented here is the same computation
exposed for diagnostics.

## Center-surround spatial filter

1-photon background is broad and low-frequency; neurons are compact blobs of
radius ≈ `sigma` pixels. To suppress the former and keep the latter, frames are
convolved with a **center-surround PSF** (`make_center_surround_psf`):

- a 2-D Gaussian of width `sigma`,
- restricted to a disk of radius `3·sigma`,
- with its **mean subtracted on the support**, so the kernel sums to ≈ 0.

A sum-zero kernel removes any spatially uniform offset while passing
neuron-scale structure. (Set `center_psf=False` for a plain Gaussian, or `sigma=None`
for no filtering — not recommended for 1p.)

## Per-pixel noise (PNR denominator)

`estimate_noise` estimates each pixel's noise standard deviation from the
**high-frequency power spectrum** along time. It takes the real FFT of every
pixel's trace, keeps the frequency band `noise_range · Nyquist` (default
`0.25–0.5`, i.e. the top half of the spectrum where calcium signal is absent),
and reduces the one-sided PSD across those bins. The default reduction is
`logmexp` — the exponential of the mean log-PSD (a geometric mean, robust to
outliers) — with `mean` / `median` also available. Returns `sn(H, W)`.

> Note: `estimate_noise` loads the whole movie into RAM for the FFT (it needs all
> `T` frames); the package's streaming, low-RAM paths live in `background.py` /
> `pipeline.py`, not here.

## PNR — peak-to-noise ratio

After center-subtracting the filtered movie in time,

```
PNR(h, w) = max_t filtered[t, h, w] / sn(h, w)
```

(`correlation_pnr`). A pixel scores high when its brightest transient is large
relative to its own noise floor. PNR is a **peak** statistic, so it must see the
full time axis — striding the max under-estimates it.

## CORR — local correlation

`local_correlations_fft` gives each pixel the mean **Pearson correlation** of its
trace with its 8 spatial neighbours over time. The implementation self-recenters
(subtracts each pixel's time-mean) and divides by std, then averages the
neighbour products via interior-slice multiplies (no FFT, despite the name). Edge
pixels are divided by their actual neighbour count (5 at corners, 8 in the bulk).
Before correlating, the filtered movie is thresholded at `3·sn` so only suprathreshold
activity contributes. Result is bounded in `[-1, 1]`.

A real neuron lights up its neighbours together → high CORR; isolated noise does
not.

## The seed map

Seeds are the **local maxima of `CORR × PNR`**, keeping only pixels with
`CORR ≥ min_corr` and `PNR ≥ min_pnr` (`detect_seeds` uses
`skimage.feature.peak_local_max`, sorted by score descending). High on *both*
axes means "structured **and** bright" — the signature of a soma. These thresholds
(`min_corr`, `min_pnr`) are the primary knobs controlling how many neurons are
detected.

`stride` subsamples time before computing the images: the per-pixel reductions
tolerate it, roughly dividing wall time, with little effect on the seed map
(default `1` = no subsampling).
