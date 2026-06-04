# --- Comparison Notebook: Custom vs. CaImAn Rigid Motion Correction ---

import numpy as np
import matplotlib.pyplot as plt
import zarr
import os
from pathlib import Path

# Import your implementation
from minicnmfe.motion_correction import motion_correction_rigid
from minicnmfe.io import avi_to_zarr


# 1. Load Real Video Data
# (Assuming data is stored in a Zarr format as seen in the tutorial)
# zarr_path = "D:/code/claude_cnmfe/real_vids/movie.zarr"
zarr_path = 'D:/Code/claude_cnmfe/real_vids/0_ds.zarr'
zarr = avi_to_zarr('D:/Code/claude_cnmfe/real_vids/0.avi', dest =zarr_path)
dataset = zarr
# Use a subset of frames for faster testing (e.g., first 1000 frames)
movie_data = np.asarray(dataset[:1000], dtype=np.float32)
T, H, W = movie_data.shape
print(f"Loaded movie with {T} frames, shape: {H}x{W}")

# 2. Set Shared Parameters
# We use gSig_filt=7 as it was noted to perform well with CaImAn
params = {
    "max_shift": (15, 15),
    "gSig_filt": 7,
    "upsample_factor": 10  # Match CaImAn's default subpixel precision
}

# 3. Run Custom Motion Correction
print("Running Custom Motion Correction...")
# Your function returns (corrected_movie, shifts)
corrected_my, shifts_my = motion_correction_rigid(
    movie_data,
    max_shift=params["max_shift"],
    gSig_filt=params["gSig_filt"],
    upsample_factor=params["upsample_factor"],
)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot dy (Vertical)
axes[0].plot(shifts_my[:, 0], label='My Implementation (dy)', alpha=0.8)
# axes[0].plot(shifts_caiman[:, 0], '--', label='CaImAn (dy)', alpha=0.8)
axes[0].set_ylabel('Shift (pixels)')
axes[0].set_title('Vertical Shifts (dy) Comparison')
axes[0].legend()
plt.show()
# Import CaImAn wrapper
from caiman.motion_correction import MotionCorrect
# 4. Run CaImAn Motion Correction
print("Running CaImAn Motion Correction...")

from caiman.source_extraction.cnmf import params as params
import caiman as cm

frate = 20
decay_time = 0.7
movie_path = 'D:/Code/claude_cnmfe/real_vids/0.avi'
# motion correction parameters
motion_correct = True    # flag for performing motion correction
pw_rigid = False         # flag for performing piecewise-rigid motion correction (otherwise just rigid)
gSig_filt = (7, 7)       # sigma for high pass spatial filter applied before motion correction, used in 1p data
max_shifts = (10, 10)      # maximum allowed rigid shift
strides = (48, 48)       # start a new patch for pw-rigid motion correction every x pixels
overlaps = (24, 24)      # overlap between patches (size of patch = strides + overlaps)
max_deviation_rigid = 3  # maximum deviation allowed for patch with respect to rigid shifts
border_nan = 'copy'      # replicate values along the boundaries

mc_dict = {
    'fnames': movie_path,
    'fr': frate,
    'decay_time': decay_time,
    'pw_rigid': pw_rigid,
    'max_shifts': max_shifts,
    'gSig_filt': gSig_filt,
    'strides': strides,
    'overlaps': overlaps,
    'max_deviation_rigid': max_deviation_rigid,
    'border_nan': border_nan
}

parameters = params.CNMFParams(params_dict=mc_dict)

num_processors_to_use = 2
_, cluster, n_processes = cm.cluster.setup_cluster(backend='multiprocessing',
                                                 n_processes=num_processors_to_use,
                                                 ignore_preexisting=False)

print(f"Successfully set up cluster with {n_processes} processes")
mot_correct = MotionCorrect(movie_path, dview=cluster, **parameters.get_group('motion'))
mot_correct.motion_correct(save_movie=True)


# Extract shifts (CaImAn stores them as a list of tuples: [(dy, dx), ...])
shifts_caiman = np.array(mot_correct.shifts_rig)

# 5. Compare Shifts
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot dy (Vertical)
axes[0].plot(shifts_my[:, 0], label='My Implementation (dy)', alpha=0.8)
axes[0].plot(shifts_caiman[:, 0], '--', label='CaImAn (dy)', alpha=0.8)
axes[0].set_ylabel('Shift (pixels)')
axes[0].set_title('Vertical Shifts (dy) Comparison')
axes[0].legend()

# Plot dx (Horizontal)
axes[1].plot(shifts_my[:, 1], label='My Implementation (dx)', color='orange', alpha=0.8)
axes[1].plot(shifts_caiman[:, 1], '--', label='CaImAn (dx)', color='red', alpha=0.8)
axes[1].set_ylabel('Shift (pixels)')
axes[1].set_xlabel('Frame #')
axes[1].set_title('Horizontal Shifts (dx) Comparison')
axes[1].legend()

plt.tight_layout()
plt.show()

# 6. Statistical Comparison
diff_dy = shifts_my[:, 0] - shifts_caiman[:, 0]
diff_dx = shifts_my[:, 1] - shifts_caiman[:, 1]

print(f"Mean Difference dy: {np.mean(diff_dy):.4f} px")
print(f"Mean Difference dx: {np.mean(diff_dx):.4f} px")
print(f"RMS Difference dy: {np.sqrt(np.mean(diff_dy**2)):.4f} px")
print(f"RMS Difference dx: {np.sqrt(np.mean(diff_dx**2)):.4f} px")

# 7. Visual Inspection (Max Projections)
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.imshow(np.max(movie_data, axis=0), cmap='gray')
plt.title("Raw Max Projection")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(np.max(corrected_my, axis=0), cmap='gray')
plt.title("Corrected Max Projection (My)")
plt.axis('off')
plt.show()

np.save('D:/code/claude_cnmfe/tmp/outcome_mc_caiman.npy', shifts_caiman)
# from minicnmfe.io import save_zarr
#
# save_zarr(corrected_my, 'D:/Code/claude_cnmfe/real_vids/mc_0.zarr', dtype = 'int32')
#
# np.max(corrected_my.astype(int))


