---
tags: [minicnmfe, explainer, eli5]
---

# CNMFe — Explained Like You're Five

> For the equations behind each step, see [algorithm math](./algorithm-math.md). For where to find the code, see [architecture](./architecture.md).

---

## What are we trying to do?

Imagine you filmed hundreds of glowing fireflies in a dark field, but through frosted glass. Every frame of video shows blobs of light — some are individual fireflies flashing, but there's also a diffuse background glow from all the fireflies you can't quite see.

Your job: figure out *which firefly is where* and *when exactly each one flashed*.

In real life: you recorded neurons in a mouse brain. Each neuron is a tiny blob of light that gets brighter when it fires. The "frosted glass" is the tissue scattering — a messy background that varies slowly everywhere.

**CNMFe** is the algorithm that separates the individual neuron signals from this noisy, correlated background.

---

## Step 1 — Stabilise the video (Motion Correction)

The camera or the brain wiggles. So frame 1 and frame 2 might be shifted by a pixel or two.

> **Analogy:** Like stacking holiday photos — you line them all up before you start looking for differences.

We take each frame and find how far it drifted from a reference (the average of the first few hundred frames), then shift it back. We use a math trick called *phase correlation* (basically: compare patterns in frequency space) which gives subpixel accuracy — we can correct shifts of 0.3 pixels, not just whole pixels.

---

## Step 2 — Estimate the background noise level

Every pixel has some random flicker even when nothing is happening. We measure how *noisy* each pixel is by looking at the high-frequency wiggles in its brightness over time (neurons change slowly; noise is fast).

> **Analogy:** Tuning out the static on a radio to hear how strong the signal is.

---

## Step 3 — Find where the neurons are (CORR and PNR images)

We make two "summary pictures" to show us where neurons probably are:

- **CORR (Correlation image):** At each pixel, how similar is its signal to its neighbours? Neurons are compact — a neuron pixel and its immediate neighbours all flash together. Random noise doesn't.

- **PNR (Peak-to-Noise Ratio):** At each pixel, how tall is the biggest flash compared to the background noise? A real neuron flash is much bigger than random noise.

> **Analogy:** CORR is like asking "does this pixel move in sync with the ones around it?" PNR is asking "does this pixel ever get *much* brighter than its usual noise level?"

Multiplying CORR × PNR gives a score map. Bright spots in this map are likely neurons.

We also run the video through a *center-surround filter* first — this sharpens neuron-shaped signals and suppresses the diffuse background (like sharpening a blurry photo before you look for faces).

---

## Step 4 — Find initial neurons, one by one (Greedy Initialization)

We pick neurons one at a time, starting with the brightest (highest CORR×PNR score):

1. Pick the top candidate pixel (the "seed").
2. Look at a small patch around it. Find pixels whose brightness goes up and down together with the seed — those are probably part of the same neuron.
3. Fit a small football-shaped blob (the "spatial footprint") to those pixels.
4. Record the average brightness over time (the "trace").
5. **Subtract** this neuron's signal from the video.
6. Recompute the CORR/PNR map on the cleaned-up video.
7. Repeat for the next brightest spot.

> **Analogy:** Like peeling away the loudest instrument in a song one by one — after removing the drums, you can hear the bass more clearly; after removing the bass, the guitar emerges.

We stop when no pixels are bright enough to be a new neuron.

---

## Step 5 — Model the background (Ring Background)

Even after finding all neurons, there's still a spatially correlated background: the diffuse glow that drifts slowly across the whole field. CNMFe uses a clever model for this:

For each pixel, predict its background from a **ring of pixels** around it (at distance ~1.5 × neuron size). Nearby pixels share the same diffuse glow, so a weighted sum of the ring neighbours is a good predictor.

> **Analogy:** If all the streetlights on your block flicker together, you can predict how bright your window is from how bright your neighbours' windows are.

We fit the ring weights using simple linear regression — once. The resulting "background model" can then be subtracted from every frame.

---

## Step 6 — Refine everything (Iterative Updates)

Our initial guesses for neuron shapes and traces aren't perfect. So we take turns improving them:

### Update shapes (Spatial Update)

Given the current traces, re-fit each neuron's shape. For each pixel: figure out which neurons overlap it, then solve "how much does each neuron contribute to this pixel's brightness?" This is a small LASSO regression (fancy least-squares that also pushes small weights to zero, keeping footprints sparse and clean).

### Update traces (Temporal Update)

Given the updated shapes, re-fit each neuron's trace. For each neuron: take everything left in the video after subtracting all the *other* neurons, and project it onto this neuron's footprint. Then denoise the trace using the AR(1) model (described below).

### Merge duplicates (Merging)

Sometimes two nearby seeds got picked for the same neuron. We check every pair of components: if their traces look nearly identical, *and* either their footprints overlap *or* their centres of mass sit close together, they're probably the same neuron — merge them.

> **Why both rules?** After cleaning up footprints (we threshold-and-keep-the-largest-blob), two duplicates of the same neuron can end up with no spatial overlap at all — each kept just the bright core around its own peak. The centre-distance rule catches these: even when the cleaned blobs don't touch, their centroids are still right next to each other.

We do this whole update cycle 2–3 times until things stabilise.

---

## Step 6b — What's the AR(1) model?

Calcium imaging measures a chemical (GCaMP) that glows when flooded with calcium. Neurons fire, calcium rushes in, glow rises sharply, then decays exponentially.

The AR(1) model says:

> **brightness now** = (decay × brightness yesterday) + (this spike)

The "decay" is a number slightly less than 1 (like 0.9). With this model, given a noisy brightness trace, we can work backwards to find the exact times the neuron spiked — this is *deconvolution*.

> **Analogy:** If you know a bathtub drains at 10% per second, and you see the water level, you can calculate exactly when and how much water was poured in.

We use an algorithm called **OASIS** (Online Active Set to Infer Spikes) — it's fast and gives an exact solution.

### How do we know the "drain rate" of the bathtub?

The decay number `g` depends on the indicator (which GCaMP variant you're using) and how fast you record. You can let the pipeline guess it from the data (fine on clean recordings), but on noisy 1-photon recordings the slow background drift makes the guess wrong — it usually says "the bathtub drains very slowly" even when it doesn't.

The fix: tell the pipeline what you actually used. Set `decay_time_ms` (e.g. 140 for GCaMP6f, 180 for jGCaMP8m) and `frame_rate_hz`. The pipeline turns those into the right `g` and uses it as a Bayesian *prior* — the data gets to nudge `g` away from the target if it really disagrees, but mostly it just anchors `g` where biology says it should be. See the [API reference](../api/index.md) for the indicator table.

---

## What comes out?

After all this, you have:

- **A** — for each neuron: a map showing which pixels it covers (its "footprint")
- **C** — for each neuron: its smooth calcium trace over time (denoised by OASIS into a clean AR(1) shape)
- **S** — for each neuron: exactly when it fired (the spike train)
- **YrA** — the *noisy* version of each calcium trace: what's left in the data at that footprint after subtracting all other neurons. `C + YrA` is the noisy-but-shape-faithful trace, useful when you want to compare against an external reference.

> **Analogy:** Like unmixing a cocktail party recording into individual speakers (who is talking), when they spoke, and how loud each word was.

### Wait, why two flavours of `C`?

`C` is what OASIS *thinks* the calcium trace should look like, given the AR(1) decay model. It's clean, smooth, and ideal for spike analyses.

`C + YrA` is what's actually *in the data* at that footprint. It's noisier, but its shape matches the underlying biology more faithfully — handy if you're correlating with another signal, or if your analysis cares about exact spike timing rather than smooth amplitude.

---

## Why is this hard for 1-photon imaging?

In 2-photon imaging, the laser only excites one thin focal plane, so background is small.

In 1-photon imaging, *everything* in the tissue fluoresces — nearby cells, axons, blood vessels, out-of-focus neurons. The background is huge and spatially correlated. Standard NMF methods confuse background with neurons.

CNMFe's ring-model background is specifically designed to separate this correlated background from the compact, structured neuron signals.
