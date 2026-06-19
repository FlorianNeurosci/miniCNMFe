# Merging

Source: `minicnmfe/merging.py`. Entry: `merge_components(...)`. Called inside the
BCD loop (plus an extra pre-merge on iteration 0) to fuse multiple detections of
the same neuron.

## The merge rule

Two components `i`, `j` are merged when their traces are correlated **and** they
are spatially close:

```
|Pearson(C[i], C[j])| > thr_corr
  AND
( Jaccard(i, j) > thr_overlap  OR  centre_dist(i, j) < centre_dist_factor · sigma )
```

- **Temporal** — Pearson correlation of the (mean-subtracted, unit-norm) traces,
  thresholded at `thr_corr` (default 0.85).
- **Spatial** — either footprint overlap (Jaccard `= O[i,j] / (‖a_i‖² + ‖a_j‖² −
  O[i,j])`, where `O = AᵀA`) above `thr_overlap` (default 0.5), **or** the
  centre-of-mass distance below `centre_dist_factor · sigma`.

The **centre-distance fallback** is the important part: `threshold_footprint`
keeps only the largest connected component, so two detections of the same neuron
can end up with *disjoint* supports (Jaccard ≈ 0) even though they track the same
cell. Centre proximity catches those. (Set `centre_dist_factor = 0` to disable and
get standard overlap-only merging.) Requires `sigma` and `dims`.

## How a group is fused

The merge graph (temporally correlated AND spatially linked) is treated as
undirected, and its connected components define the groups
(`scipy.sparse.csgraph.connected_components`). For each group of more than one:

- **Footprint** = sum of the members' footprints, re-cleaned with
  `threshold_footprint` to keep the fused blob compact.
- **Trace** = mean of the members' traces, clipped to `≥ 0`.

The merged trace is **not** re-deconvolved here — that is deferred to the caller's
next `update_temporal`, which uses the cached AR coefficient `g`. Re-estimating
`g` inside the merge would re-introduce `fudge_factor` drift (see
[Temporal update](temporal-update.md)).

## Return value

`merge_components` returns a **4-tuple** `(A_merged, C_merged, n_merged,
members_per_group)`. `members_per_group[j]` is the array of original component
indices that fused into output `j` (singletons have length 1). The pipeline uses
it to keep the per-component AR cache (`g_per_k`, `sn_per_k`) and `C_raw` aligned
with the new column order — each merged group inherits from its first member
(`_cache_after_merge`).
