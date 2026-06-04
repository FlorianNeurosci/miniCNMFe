from minicnmfe.io import open_zarr
from minicnmfe.pipeline import CNMFeParams
from minicnmfe.pipeline import CNMFe
import numpy as np

# Import your implementation
from minicnmfe.motion_correction import motion_correction_rigid
from minicnmfe.io import avi_to_zarr

vid_paths = [f'O:/archive/projects/2023_intercontext/PICAST/data/0_raw/20260506_m0009937_wt_1106/miniscope_video/{i}.avi' for i in np.arange(5)]
vid_paths = [f'D:/Code/claude_cnmfe/real_vids/161.avi']
zarray = avi_to_zarr(vid_paths, dest = 'D:/Code/claude_cnmfe/real_vids/20260511_161.zarr')
dataset = zarray
# # Use a subset of frames for faster testing (e.g., first 1000 frames)
# movie_data = np.asarray(dataset[:], dtype=np.float32)
# T, H, W = movie_data.shape
# print(f"Loaded movie with {T} frames, shape: {H}x{W}")

# 2. Set Shared Parameters
# We use gSig_filt=7 as it was noted to perform well with CaImAn
params = {
    "max_shift": (30, 30),
    "gSig_filt": 5,
    "upsample_factor": 10  # Match CaImAn's default subpixel precision

}

import time
# 3. Run Custom Motion Correction
print("Running Custom Motion Correction...")
# Your function returns (corrected_movie, shifts)
start = time.time()
corrected_my, shifts_my = motion_correction_rigid(
    dataset,
    max_shift=params["max_shift"],
    gSig_filt=params["gSig_filt"],
    upsample_factor=params["upsample_factor"],
    zarr_path = 'D:/Code/claude_cnmfe/real_vids/161.zarr',
    niter_rig = 2,

)
end = time.time()
print(f'duration was {end-start}')

import matplotlib.pyplot as plt

plt.figure()
plt.plot(shifts_my)
plt.show()

mc_corr_vid = open_zarr('D:/Code/claude_cnmfe/real_vids/mc_0.zarr')

params = CNMFeParams(
    sigma = 5.0
)

cnmfe_model = CNMFe(params = params)

out = cnmfe_model.fit(mc_corr_vid, do_motion_correction=False)

out.A

out.C

out.YrA

import numpy as np
import cv2
from tqdm import tqdm

movie = mc_corr_vid

output_path = "dff_video_real.mp4"
fps = 100
chunk_size = 100

# ------------------------------------------------------------------
# BETTER BASELINE
# ------------------------------------------------------------------
# use lower percentile but from MANY frames
sample_idx = np.linspace(0, movie.shape[0]-1, 300).astype(int)

sample = movie[sample_idx].astype(np.float32)

# real ΔF/F baseline
F0 = np.percentile(sample, 10, axis=0)

# avoid exploding pixels
F0[F0 < np.median(F0) * 0.2] = np.median(F0)

# ------------------------------------------------------------------
# VIDEO WRITER
# ------------------------------------------------------------------
T, H, W = movie.shape

writer = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (W, H),
    isColor=False
)

# ------------------------------------------------------------------
# PROCESS
# ------------------------------------------------------------------
for start in tqdm(range(0, T, chunk_size)):

    stop = min(start + chunk_size, T)

    chunk = movie[start:stop].astype(np.float32)

    # REAL dF/F
    chunk = (chunk - F0[None]) / F0[None]

    # ------------------------------------------------------------------
    # KEY DIFFERENCE:
    # normalize EACH FRAME independently for visualization
    # ------------------------------------------------------------------

    for frame in chunk:

        # robust contrast
        vmin = np.percentile(frame, 1)
        vmax = np.percentile(frame, 99.5)

        frame = np.clip(frame, vmin, vmax)

        frame = (frame - vmin) / (vmax - vmin + 1e-8)

        frame8 = (frame * 255).astype(np.uint8)

        writer.write(frame8)

writer.release()

print("Done.")

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import issparse
from skimage.measure import find_contours

A = out.A  # shape: (pixels, neurons)

# movie dimensions
d1, d2 = 600, 600   # adjust to your FOV

fig, ax = plt.subplots(figsize=(8, 8))

# background
ax.imshow(np.mean(mc_corr_vid,axis = 0), cmap="gray")

for i in range(A.shape[1]):

    # get spatial footprint
    spatial = A[:, i].toarray().reshape(d1, d2, order="C")

    # normalize
    spatial = spatial / spatial.max()

    # contour at 20% peak
    contours = find_contours(spatial, level=0.2)

    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1)

ax.set_xlim(0, d2)
ax.set_ylim(d1, 0)
ax.set_aspect("equal")
plt.show()

import matplotlib.pyplot as plt
import numpy as np

raw = out.YrA + out.C
denoised = out.C

n = raw.shape[0]

fig, axes = plt.subplots(
    nrows=n,
    ncols=1,
    figsize=(12, 1.5 * n),
    sharex=True
)

if n == 1:
    axes = [axes]

for i, ax in enumerate(axes):

    ax.plot(raw[i], alpha=0.6, label="YrA + C")
    ax.plot(denoised[i], linewidth=1.5, label="C")

    ax.set_ylabel(f"{i}", rotation=0, labelpad=15)

    if i == 0:
        ax.legend()

plt.tight_layout()
plt.show()