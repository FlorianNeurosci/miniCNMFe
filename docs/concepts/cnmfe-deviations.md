# How this package differs from published CNMF-E

Every known difference from the published algorithm, and why it exists. In `CNMFeParams`
the matching fields are tagged `[NON-STANDARD]`.

Compared against:

- **Paper:** Zhou et al. 2018, *Efficient and accurate extraction of in vivo calcium signals
  from microendoscopic video data*, eLife 7:e28728.
- **MATLAB:** the `CNMF_E` code, read from the copy bundled with CaliAli, which has local edits.
- **CaImAn:** `caiman 1.13.1`, stage functions called through `validation/caiman_swap/`.

The paper does not specify every detail. Where it is silent, MATLAB is listed instead.

## Background model

The ring background follows the paper: `W` is fit on `X = Y - AC - b0`, applied as
`B = W X` (Eq. 8, 11), and re-estimated inside the loop, with `b0 = mean(Y - AC)` (Eq. 10).
`Y_bg` is held fixed while `A` and `C` update.

| Difference | Paper / MATLAB | Ours | Effect |
|---|---|---|---|
| Outlier clipping in the `W` fit (Eq. 12) | clips `X` at `B + ζσ`, ζ=10 | none (CaImAn neither) | Measured none: 0.001% of entries clipped, results unchanged |
| Ring radius | paper `2l`; MATLAB `1.5·gSiz`, `gSiz = 4·gSig` | `1.5·(4σ+1)`, truncated to an integer | Within 1 px of MATLAB and CaImAn; below the paper's rule of thumb |
| Time subsampling of the `W` fit | MATLAB strides | strides (`bg_tsub`); CaImAn block-averages | Speed trade-off |
| Global term | none | `global_bg_rank`, off in `CNMFeParams`, **on (1) in the tuner defaults** | Absorbs slow drift on long recordings |

`ring_constrain_sum` and `ring_fit_once` are opt-in. `ring_apply_to_raw` is deprecated and
flagged for removal.

## Everything else

These are deliberate, and on by default unless noted. Their effects have not been measured
against the paper's choices.

| Stage | Paper / MATLAB | Ours | Why |
|---|---|---|---|
| Seeding | greedy `argmax L·P` above `Lmin`, `Pmin` | `seed_method='mixture'`: smoothed `L·P` maxima split by a two-component fit | Removes the thresholds; `min_corr`, `min_pnr` have no effect under it |
| Neuron / background pixels around a seed | r ≥ 0.7 / r ≤ 0.3 | r > 0.8 / r < 0.4 | Tuned against over-detection |
| Missed cells from the residual | optional step 4, smaller thresholds; CaImAn runs it automatically | not implemented | |
| Spatial solver | HALS, no sparsity term in (P-S) | positive LASSO plus ridge (`spatial_ridge`) | Convergence on correlated traces |
| Footprint cleanup | morphological opening, largest component | `nrg` thresholding at 0.95, closing, circular constraint, largest component | Tighter footprints on dense FOVs; tuned before the ring fix |
| Deconvolution | OASIS, AR(2) usual; MATLAB AR(1), re-optimises AR, `smin=-5` | OASIS AR(1), `penalty=1`, no `smin` | |
| AR coefficient | MATLAB re-optimises per call | one pooled `g`, estimated once and cached (`global_ar`) | Re-estimating drifted `g` towards 0 |
| AR prior | none | shrinks `g` towards `decay_time_ms` when set; **on in the tuner defaults** | Yule-Walker is biased upward by drift |
| First temporal pass | deconvolved | `skip_first_deconv=True` | Speed |
| Merging rule | spatial and temporal correlation; MATLAB also distance + correlation | temporal correlation AND (overlap OR centre distance) | Catches duplicates with disjoint supports |
| Merged component | rank-1 NMF | summed footprint, mean trace | Avoids re-deconvolving inside the merge |

## Fixed

Three departures from the paper were corrected in November 2026. All were present from the
initial commit, and the first two were errors rather than choices.

| Was | Now |
|---|---|
| Ring applied to raw data, `W(Y-b0)` | `W(Y-AC-b0)` (Eq. 8, 11) |
| `W` solved once from the initial traces | re-estimated in every iteration |
| Ring radius `1.5(2σ+1)` | `1.5(4σ+1)` |

The old formula gives `Y_bg ≈ (I-W)AC`, which was then fitted by `AC`. That eroded footprints,
and subtracted each neuron from cells about a ring radius away. On the simulator, footprints
matched `relu((I-W)a_true)` at r 0.96 against 0.82 for `a_true`, and predicted cross-talk
matched measured cross-talk across 402 pairs at r 0.965, slope 1.02, with nothing fitted.

Extractions made before this fix are not comparable with new ones. To reproduce them, set
`ring_apply_to_raw=True`, `ring_fit_once=True` and
`ring_size_factor=1.5*(2*sigma+1)/(4*sigma+1)`; the output is then bit-identical.

On real sessions (5000 frames, default config) the negative correlation between cells near
ring distance went from -0.138 / -0.133 to +0.200 / +0.108 at 9-12 / 12-15 px on m0010798, where
the curve now matches CaImAn's, and from -0.048 / -0.320 to +0.274 / +0.098 on m0010800. Two som
sessions were inconclusive: m0000163 has far-pair r near +0.3 in both versions, and m0000166
keeps a dip at 12-15 px on 10 pairs. That dip depends on the radius (-0.14 ± 0.05 at 16.8 px,
+0.03 ± 0.05 at 22.4 px), so it is a ring effect and `ring_size_factor` near 2, the paper's `2l`,
may suit such sessions better. This is one session.

Cross-talk on the simulator is reduced, not removed. Neither more iterations nor clipping
removes the rest, and its cause is not known. Extraction takes about twice as long.
