---
tags: [minicnmfe, math, algorithm]
---

# CNMFe — Mathematical Description

> See also: [algorithm ELI5](./algorithm-eli5.md) for the intuitive version, [architecture](./architecture.md) for code locations.

## 1. Problem Statement

Given a movie $\mathbf{Y} \in \mathbb{R}^{T \times H \times W}$ from a 1-photon (wide-field) calcium imaging experiment, find:

$$\mathbf{Y}_{\text{flat}} \approx \mathbf{A}\mathbf{C} + \mathbf{b}_0\mathbf{1}^\top + \mathbf{W}(\mathbf{Y}_{\text{flat}} - \mathbf{b}_0\mathbf{1}^\top) + \boldsymbol{\epsilon}$$

where:
- $\mathbf{Y}_{\text{flat}} \in \mathbb{R}^{HW \times T}$ — movie reshaped (pixels × frames)
- $\mathbf{A} \in \mathbb{R}^{HW \times K}_{\geq 0}$ — sparse spatial footprints
- $\mathbf{C} \in \mathbb{R}^{K \times T}_{\geq 0}$ — calcium traces
- $\mathbf{b}_0 \in \mathbb{R}^{HW}$ — per-pixel baseline
- $\mathbf{W} \in \mathbb{R}^{HW \times HW}$ — ring background weight matrix (sparse)
- $\boldsymbol{\epsilon}$ — i.i.d. Gaussian noise

---

## 2. Motion Correction

### 2.1 Shift Estimation

For frame $\mathbf{f}_t$ and template $\mathbf{T}$, the cross-power spectrum is:

$$\mathbf{P} = \frac{\mathcal{F}(\mathbf{T}) \cdot \overline{\mathcal{F}(\mathbf{f}_t)}}{|\mathcal{F}(\mathbf{T}) \cdot \overline{\mathcal{F}(\mathbf{f}_t)}|}$$

The shift $(\delta_y, \delta_x)$ is the location of $\max |\mathcal{F}^{-1}(\mathbf{P})|$, refined to subpixel accuracy via an upsampled DFT (matrix-multiply DFT, `skimage.registration.phase_cross_correlation`).

### 2.2 Shift Application

Shift applied as a Fourier-domain phase ramp (no spatial interpolation):

$$\hat{\mathbf{f}}_t = \mathcal{F}^{-1}\!\left[\mathcal{F}(\mathbf{f}_t) \cdot \exp\!\left(2\pi i \left(-\delta_y \frac{N_r}{H} - \delta_x \frac{N_c}{W}\right)\right)\right]$$

where $N_r$, $N_c$ are the frequency indices arranged by `ifftshift`.

> **Note:** The shift vector uses the convention $(\delta_y, \delta_x)$ — row first, column second — consistent with numpy array indexing.

---

## 3. Preprocessing

### 3.1 Noise Estimation

For each pixel $(h, w)$, estimate noise std from the high-frequency PSD:

$$\hat{\sigma}_{hw} = \sqrt{\exp\!\left(\frac{1}{|\Omega|}\sum_{\omega \in \Omega} \log P_{hw}(\omega)\right)}$$

where $\Omega = \{\omega : f_\text{low} \le \omega \le f_\text{high}\}$ (default $[0.25, 0.5]$ × Nyquist) and $P_{hw}(\omega) = \frac{2}{T}|\hat{Y}_{hw}(\omega)|^2$ is the one-sided PSD.

### 3.2 Center-Surround PSF

The spatial filter suppresses diffuse background (DC + low spatial frequencies) while preserving signals at the scale of a neuron:

$$\mathbf{k}_\text{csf} = \mathbf{g}_\sigma - \bar{\mathbf{g}}_\sigma$$

where $\mathbf{g}_\sigma$ is a Gaussian kernel with std $\sigma$ and $\bar{\mathbf{g}}_\sigma$ is its mean over the nonzero support. This is approximately a difference-of-Gaussians (DoG) filter.

### 3.3 CORR Image

Local correlation between each pixel and its 8 neighbours, computed via FFT shift trick:

$$\text{CN}(h,w) = \frac{1}{|\mathcal{N}|}\sum_{(h',w') \in \mathcal{N}(h,w)} \text{corr}\!\left(\mathbf{y}_{hw}, \mathbf{y}_{h'w'}\right)$$

where $\mathcal{N}$ are the 8 immediate neighbours and correlation is Pearson over time.

### 3.4 PNR Image

$$\text{PNR}(h,w) = \frac{\max_t \tilde{y}_{hw}(t)}{\hat{\sigma}_{hw}}$$

where $\tilde{y}_{hw}$ is the PSF-filtered trace at pixel $(h,w)$.

---

## 4. Greedy Initialization

### 4.1 Seed Detection

Score image: $\mathbf{S} = \text{CN} \cdot \text{PNR}$, masked by $\text{CN} \geq \rho_\text{min}$ and $\text{PNR} \geq \text{PNR}_\text{min}$.

Seeds are local maxima of $\mathbf{S}$ (via `skimage.feature.peak_local_max`), sorted by score descending.

### 4.2 Component Extraction

For seed pixel $(r,c)$, extract a patch of radius $r_\text{patch} = \max(3\sigma, 5)$:

1. **Normalise** each pixel trace to unit std: $\hat{y}_p = (y_p - \bar{y}_p)/\text{std}(y_p)$

2. **Neuron pixels**: $\mathcal{I}_n = \{p : \text{corr}(\hat{y}_\text{seed}, \hat{y}_p) > \tau_n\}$ — default $\tau_n = 0.8$ (`init_min_corr_neuron`)

3. **Background pixels**: $\mathcal{I}_b = \{p : \text{corr}(\hat{y}_\text{seed}, \hat{y}_p) < \tau_b\}$ — default $\tau_b = 0.4$ (`init_max_corr_bg`)

4. **Trace**: $c_i = \text{mean}_{p \in \mathcal{I}_n} \hat{y}_p$

5. **Background**: $y_\text{bg} = \text{median}_{p \in \mathcal{I}_b} y_p$ (raw traces)

6. **OLS footprint**: solve $[c_i \,|\, y_\text{bg} \,|\, \mathbf{1}] \cdot \boldsymbol{\beta} \approx \mathbf{Y}_\text{patch}$ for $\boldsymbol{\beta}$; set $a_i = \boldsymbol{\beta}_{0,:}\text{.clip}(0)$

7. **Shape constraints**: circular constraint (zero pixels $> \alpha \cdot R$ from centroid where $R = \sqrt{\text{area}/\pi}$, default $\alpha = 2.5$ via `circular_max_dist_factor`) and connectivity constraint (keep largest connected component).

### 4.3 AR(1) Deconvolution of Seed Trace

After extraction, deconvolve $c_i$ using OASIS (see §6) to obtain the clean trace $\tilde{c}_i$.

### 4.4 Subtraction and Update

$$\mathbf{Y}_\text{flat}[:, r_0:r_1, c_0:c_1] \mathrel{-}= a_i[\text{np.newaxis}] \cdot \tilde{c}_i[:,\text{np.newaxis},\text{np.newaxis}]$$

Then recompute CORR/PNR on the updated patch and suppress a disk of radius

$$r_\text{supp} = \max\!\big(\lfloor f \cdot \sigma \rfloor,\;\lfloor 2\sigma + 1 \rfloor\big)$$

around every already-found centre, where $f$ is `seed_suppress_factor` (default 2.0). The disk must cover the neuron's actual support (FWHM $\approx 2\sigma$) — otherwise the residual halo from incomplete subtraction reseeds at a pixel a few px from the original centre and the same neuron is detected multiple times.

---

## 5. Ring Background

### 5.1 Ring Geometry

For pixel $i$, define its ring neighbourhood:

$$\mathcal{R}_i = \{j : r \le d(i,j) \le r+1\}, \quad r = r_\text{factor} \cdot (2\sigma+1)$$

### 5.2 Weight Estimation

Compute the neural-subtracted residual:

$$\mathbf{X} = \mathbf{Y}_\text{flat} - \mathbf{A}\mathbf{C} - \mathbf{b}_0\mathbf{1}^\top$$

For each pixel $i$, solve the ridge regression:

$$\mathbf{w}_i = \arg\min_{\mathbf{w}} \|\mathbf{X}[i,:] - \mathbf{w}^\top\mathbf{X}[\mathcal{R}_i,:]\|^2 + \lambda\|\mathbf{w}\|^2$$

Closed form: $\mathbf{w}_i = (\mathbf{X}[\mathcal{R}_i,:]\mathbf{X}[\mathcal{R}_i,:]^\top + \lambda\mathbf{I})^{-1}\mathbf{X}[\mathcal{R}_i,:]\mathbf{X}[i,:]^\top$

Baseline: $b_{0,i} = \text{mean}_t(Y_{\text{flat},i} - \mathbf{w}_i^\top \mathbf{Y}_{\text{flat},\mathcal{R}_i})$

### 5.3 Background Subtraction

$$\mathbf{Y}_\text{res} = \mathbf{Y}_\text{flat} - \mathbf{b}_0\mathbf{1}^\top - \mathbf{W}(\mathbf{Y}_\text{flat} - \mathbf{b}_0\mathbf{1}^\top)$$

---

## 6. Temporal Update

### 6.1 Block Coordinate Descent

Objective: find $\mathbf{C}$ minimising $\|\mathbf{Y}_\text{res} - \mathbf{A}\mathbf{C}\|_F^2$ subject to the AR(1) constraint on each row.

Pre-compute: $\mathbf{Y}\mathbf{A} = \mathbf{Y}_\text{res}^\top\mathbf{A}$ and $\mathbf{A}^\top\mathbf{A}$.

**Gauss-Seidel** (serial, `n_jobs=1`): for each component $k$:

$$\text{trace}_k = \frac{(\mathbf{Y}\mathbf{A})_{:,k} - (\mathbf{A}^\top\mathbf{A}\,\mathbf{C})_{k,:}^\top + [\mathbf{A}^\top\mathbf{A}]_{kk}\,\mathbf{C}_{k,:}^\top}{[\mathbf{A}^\top\mathbf{A}]_{kk}}$$

Then deconvolve $\text{trace}_k$ with OASIS → $(c_k, s_k)$. Update residuals immediately.

**Jacobi** (parallel, `n_jobs!=1`): compute all $\text{trace}_k$ simultaneously from the current $\mathbf{C}$, deconvolve in parallel, then update all $\mathbf{C}$ and residuals.

### 6.2 AR(1) Parameter Estimation

The AR coefficient $\mathbf{g}$ is estimated **once** per pipeline run (not per BCD iteration) and cached for every subsequent `update_temporal` call. Re-estimating from already-deconvolved traces re-applies the fudge factor on top of the OASIS-imposed shape, drifting $\mathbf{g}$ toward 0 across iterations and distorting the calcium decay shape.

The pipeline pools all components' raw traces into a single concatenated vector before estimation:

$$\mathbf{c}_\text{pool} = \big[\mathbf{c}_{\text{raw},1};\;\mathbf{c}_{\text{raw},2};\;\ldots;\;\mathbf{c}_{\text{raw},K}\big] \in \mathbb{R}^{KT}$$

Per-component estimation on $T \approx 300$ traces has a $\sim 0.1$ spread in the recovered $g$ even on clean ground truth; pooling gives an effective sample length of $KT$ and a much more stable estimate (assumes all neurons share the same calcium indicator dynamics — the common case for one recording).

1. **Noise** (per component, used by OASIS): $\hat{\sigma}_k = \sqrt{\exp(\text{mean}[\log P_k(\omega), \omega \in \Omega])}$ on each $\mathbf{c}_{\text{raw},k}$.

2. **Yule-Walker** (on $\mathbf{c}_\text{pool}$): build autocorrelation $r(k) = \frac{1}{T_\text{pool}-k}\sum_t (y_t - \bar{y})(y_{t+k} - \bar{y})$.

   Solve the Toeplitz system $\mathbf{R}\,\mathbf{g} = \mathbf{r}$ where $[\mathbf{R}]_{ij} = r(|i-j|)$ and $[\mathbf{r}]_k = r(k+1)$.

3. **Shrinkage** — two paths, selected automatically:

   **(a) Bayesian prior** (when both `CNMFeParams.decay_time_ms` and `frame_rate_hz` are set). Derive the indicator's physical target

   $$g_\text{target} = \exp\!\left(-\frac{1}{f_\text{Hz}\cdot \tau_\text{ms}/1000}\right)$$

   and shrink the dominant coefficient toward it:

   $$g_0 \leftarrow (1 - w)\,g_{0,\text{yw}} + w\,g_\text{target},\qquad w = g_\text{prior\_weight} \in [0,1]$$

   Higher-order coefficients (for AR$(p>1)$) keep the legacy multiplier $\mathbf{g}_{1:} \leftarrow 0.96\,\mathbf{g}_{1:}$ since the prior is a single-scalar target.

   **(b) Legacy multiplicative** (when either `decay_time_ms` or `frame_rate_hz` is `None`): $\mathbf{g} \leftarrow \texttt{fudge\_factor}\cdot\mathbf{g}$ with `fudge_factor=0.96` default. Unitless prior toward 0.

   Both paths end with $\mathbf{g} \leftarrow \mathrm{clip}(\mathbf{g}, 0, 0.9999)$ for numerical stability.

4. **Cache**: store $\mathbf{g}_k = \mathbf{g}$ (broadcast to all components) and $\hat{\sigma}_k$. After merging, the cache for component $j$ inherits from the strongest member: $\mathbf{g}_j \leftarrow \mathbf{g}_{\text{members}_j[0]}$ (no re-estimation, no drift).

**Why the prior path matters.** On miniscope data the ring background often
under-subtracts slow drift, so the autocorrelation $r(1)$ is contaminated by
the drift and Yule-Walker reports $g_\text{yw} \to 1$ for any indicator.
$\texttt{fudge\_factor}$ then clamps every estimate at the ceiling regardless
of the indicator's true $\tau$. The Bayesian prior replaces this unitless
shrinkage with a physical-units-grounded target — calibrate $\tau_\text{ms}$
from the indicator (GCaMP6f ~140, jGCaMP8m ~180, jGCaMP8s ~350, etc.) and the
prior anchors $\mathbf{g}$ where biology says it should be.

### 6.3 OASIS Deconvolution (AR(1))

Solve the constrained problem:

$$\min_{\mathbf{c}, \mathbf{s}} \|\mathbf{y} - \mathbf{c}\|^2 \quad\text{s.t.}\quad c_t \geq g\,c_{t-1},\;\; c_t \geq 0,\;\; s_t = c_t - g\,c_{t-1} \geq 0$$

via the Pool-Adjacent-Violators Algorithm (PAVA). Each pool $k$ represents a contiguous segment where $c_t = v_k \cdot g^{t - \tau_k}$:

$$v_k = \max\!\left(0,\; \frac{\sum_{t \in \mathcal{P}_k} g^{t-\tau_k}(y_t - b)}{{\sum_{t \in \mathcal{P}_k} g^{2(t-\tau_k)}}}\right)$$

Pools are merged when the AR(1) constraint $c_t \geq g\,c_{t-1}$ is violated between adjacent pools.

---

## 7. Spatial Update

### 7.1 Support Computation

For each pixel $p$, find active components:

$$\mathcal{A}_p = \{k : \text{dilation}(\mathbf{A}_{:,k})[p] > 0\}$$

where dilation uses a disk of radius `dilation_radius` pixels.

### 7.2 Per-Pixel LASSO

For each pixel $p$:

$$\lambda_p = \frac{\hat{\sigma}_p}{2} \sqrt{\max_k \lambda_\text{max}\!\left(\mathbf{C}[\mathcal{A}_p,:]\mathbf{C}[\mathcal{A}_p,:]^\top\right) / T}$$

$$\mathbf{a}[\mathcal{A}_p] = \arg\min_{\mathbf{a} \geq 0} \frac{1}{2T}\|Y_p - \mathbf{C}[\mathcal{A}_p,:]^\top\mathbf{a}\|^2 + \lambda_p\|\mathbf{a}\|_1 + \frac{\beta}{2}\|\mathbf{a}\|^2$$

Solved via positive elastic-net coordinate descent using sklearn's `enet_coordinate_descent_gram` (a positive elastic-net coordinate-descent Gram solver — **not** `LassoLars`). On top of the L1 penalty it carries an L2 ridge term with weight $\beta = \texttt{spatial\_ridge} \cdot \max(\text{diag}(\text{Gram}))$ (default `spatial_ridge=1e-2`), which conditions the near-singular Gram matrix so the coordinate descent converges quickly when active traces are correlated.

### 7.3 Footprint Thresholding

After regression, for each component $k$:
1. Apply $3\times3$ median filter to $\mathbf{a}_k$ reshaped to $(H, W)$
2. Threshold the footprint. The **package default** is energy-based thresholding (`spatial_thr_method="nrg"`, `spatial_nrg_thr=0.95`): keep the brightest pixels whose summed $a^2$ reaches 95% of the footprint's total energy, zeroing the rest. The legacy peak-relative rule — zero pixels $< 0.1 \cdot \max(\mathbf{a}_k)$ — is the `spatial_thr_method="max"` option.
3. Keep only the largest connected component

---

## 8. Component Merging

Build overlap matrix $\mathbf{O} = \mathbf{A}^\top\mathbf{A}$ (sparse).

**Jaccard overlap**: $J(i,j) = \dfrac{O_{ij}}{\|\mathbf{a}_i\|_2^2 + \|\mathbf{a}_j\|_2^2 - O_{ij}}$

**Centre-of-mass distance**: $d(i,j) = \|\mathbf{p}_i - \mathbf{p}_j\|_2$ where $\mathbf{p}_k = \dfrac{\sum_q q\,a_{k,q}}{\sum_q a_{k,q}}$ is the weighted centroid of component $k$ in pixel coordinates $q = (\text{row}, \text{col})$.

**Temporal correlation**: $\rho(i,j) = \text{corr}(\mathbf{c}_i, \mathbf{c}_j)$.

**Merge edge** $(i,j)$ if traces are correlated *and* footprints either overlap *or* sit close:

$$|\rho(i,j)| > \theta_\text{corr}\quad\text{AND}\quad\big(J(i,j) > \theta_\text{overlap}\;\text{OR}\;d(i,j) < f_\text{ctr}\,\sigma\big)$$

Defaults: $\theta_\text{corr}=0.85$, $\theta_\text{overlap}=0.5$, $f_\text{ctr}=2.0$ (`merge_centre_dist_factor`).

The centre-distance fallback catches duplicate detections of the same neuron: after `threshold_footprint` keeps only the largest connected component around each detection's peak, two duplicates of one neuron may end up with **disjoint supports** ($J \approx 0$) despite tracking nearly the same trace ($\rho \approx 1$). Centre proximity is the robust fallback for that case.

For each merge group $\mathcal{G}$:

$$\mathbf{a}_\text{new} = \texttt{threshold\_footprint}\!\Big(\sum_{k \in \mathcal{G}} \mathbf{a}_k\Big),\qquad \mathbf{c}_\text{new} = \max\!\big(0,\;\text{mean}_{k \in \mathcal{G}} \mathbf{c}_k\big)$$

Re-deconvolution is **deferred** to the next `update_temporal` pass, which uses the persistent per-component AR cache (re-deconvolving here would require re-estimating $g$ from a corrupted intermediate trace, re-introducing the fudge-factor drift §6.2 fixes).

`merge_components` returns `members_per_group` — a list of length $K_\text{new}$ where `members_per_group[j]` lists the original indices that fused into output $j$. The pipeline uses this to update the AR cache: $\mathbf{g}_j \leftarrow \mathbf{g}_{\text{members}_j[0]}$, $\hat{\sigma}_j \leftarrow \hat{\sigma}_{\text{members}_j[0]}$.

### 8.1 When merging runs

The pipeline runs `merge_components` **twice** per outer iteration in iteration 0:

- **Pre-spatial merge** (iteration 0 only): catches duplicates from greedy init while their footprints still overlap, before `update_spatial` runs `threshold_footprint` and separates duplicate cores.
- **Post-temporal merge** (every iteration): standard merge after the spatial+temporal cycle.
