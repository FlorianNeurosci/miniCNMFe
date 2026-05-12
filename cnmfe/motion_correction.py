import cv2
import numpy as np
from tqdm import tqdm


# =============================================================================
# CAIMAN-COMPATIBLE HIGH PASS FILTER
# =============================================================================

def high_pass_filter_space(img_orig, gSig_filt):

    if np.isscalar(gSig_filt):
        gSig_filt = [gSig_filt, gSig_filt]

    ksize = tuple([(3 * i) // 2 * 2 + 1 for i in gSig_filt])

    ker = cv2.getGaussianKernel(ksize[0], gSig_filt[0])
    ker2D = ker.dot(ker.T)

    nz = np.nonzero(ker2D >= ker2D[:, 0].max())
    zz = np.nonzero(ker2D < ker2D[:, 0].max())

    ker2D[nz] -= ker2D[nz].mean()
    ker2D[zz] = 0

    return cv2.filter2D(
        img_orig.astype(np.float32),
        -1,
        ker2D,
        borderType=cv2.BORDER_REFLECT
    )


# =============================================================================
# CAIMAN BIN MEDIAN
# =============================================================================

def caiman_bin_median(mat, window=10):

    T, d1, d2 = mat.shape

    if T < window:
        window = T

    num_windows = int(T // window)

    if num_windows == 0:
        return np.median(mat, axis=0)

    num_frames = num_windows * window

    return np.nanmedian(
        np.nanmean(
            np.reshape(
                mat[:num_frames],
                (window, num_windows, d1, d2)
            ),
            axis=0
        ),
        axis=0
    )


# =============================================================================
# OPENCV FFT
# =============================================================================

def cv2_fft2(img):

    dft = cv2.dft(
        img.astype(np.float32),
        flags=cv2.DFT_COMPLEX_OUTPUT + cv2.DFT_SCALE
    )

    return dft[:, :, 0] + 1j * dft[:, :, 1]


def cv2_ifft2(freq):

    freq_cv = np.dstack([
        np.real(freq),
        np.imag(freq)
    ]).astype(np.float32)

    out = cv2.dft(
        freq_cv,
        flags=cv2.DFT_INVERSE + cv2.DFT_SCALE
    )

    return out[:, :, 0] + 1j * out[:, :, 1]


# =============================================================================
# UPSAMPLED DFT
# =============================================================================
def _upsampled_dft(data,
                   upsampled_region_size,
                   upsample_factor=1,
                   axis_offsets=None):

    """
    Upsampled DFT used for subpixel registration.

    Directly adapted from skimage/CaImAn implementation.
    """

    if axis_offsets is None:
        axis_offsets = [0, 0]

    im2pi = 1j * 2 * np.pi

    nr, nc = data.shape

    # -----------------------------------------------------------------
    # kernel for columns
    # shape:
    #   (nc, upsampled_region_size)
    # -----------------------------------------------------------------

    kernc = np.exp(
        (-im2pi / (nc * upsample_factor))
        * (
            (np.fft.ifftshift(np.arange(nc)) - np.floor(nc / 2))[:, None]
        )
        * (
            np.arange(upsampled_region_size)[None, :]
            - axis_offsets[1]
        )
    )

    # -----------------------------------------------------------------
    # kernel for rows
    # shape:
    #   (upsampled_region_size, nr)
    # -----------------------------------------------------------------

    kernr = np.exp(
        (-im2pi / (nr * upsample_factor))
        * (
            (np.arange(upsampled_region_size)[:, None])
            - axis_offsets[0]
        )
        * (
            np.fft.ifftshift(np.arange(nr))[None, :]
            - np.floor(nr / 2)
        )
    )

    return kernr @ data @ kernc

# =============================================================================
# CAIMAN-COMPATIBLE SHIFT ESTIMATION
# =============================================================================

def register_translation_caiman(
        src_image,
        target_image,
        upsample_factor=10,
        max_shifts=(20, 20),
):

    src_freq = cv2_fft2(src_image)
    tgt_freq = cv2_fft2(target_image)

    image_product = src_freq * tgt_freq.conj()

    eps = np.finfo(np.float32).eps

    image_product /= np.maximum(
        np.abs(image_product),
        100 * eps
    )

    cross_corr = cv2_ifft2(image_product)
    cross_corr = np.abs(cross_corr)

    # -----------------------------------------------------------------
    # constrain shifts EXACTLY like CaImAn
    # -----------------------------------------------------------------

    constrained = cross_corr.copy()

    constrained[
        max_shifts[0]:-max_shifts[0],
        :
    ] = 0

    constrained[
        :,
        max_shifts[1]:-max_shifts[1]
    ] = 0

    maxima = np.unravel_index(
        np.argmax(constrained),
        constrained.shape
    )

    midpoints = np.array([
        np.fix(axis_size / 2)
        for axis_size in src_image.shape
    ])

    shifts = np.array(maxima, dtype=np.float64)

    shifts[shifts > midpoints] -= np.array(
        src_image.shape
    )[shifts > midpoints]

    # -----------------------------------------------------------------
    # subpixel refinement
    # -----------------------------------------------------------------

    if upsample_factor > 1:

        upsampled_region_size = int(
            np.ceil(upsample_factor * 1.5)
        )

        dftshift = np.fix(
            upsampled_region_size / 2.0
        )

        sample_region_offset = (
            dftshift - shifts * upsample_factor
        )

        cross_corr_up = _upsampled_dft(
            image_product.conj(),
            upsampled_region_size,
            upsample_factor,
            sample_region_offset
        ).conj()

        cross_corr_up = np.abs(cross_corr_up)

        maxima = np.unravel_index(
            np.argmax(cross_corr_up),
            cross_corr_up.shape
        )

        maxima = np.array(
            maxima,
            dtype=np.float64
        )

        maxima -= dftshift

        shifts += maxima / upsample_factor

    return (
        float(shifts[0]),
        float(shifts[1])
    )


# =============================================================================
# APPLY SHIFT EXACTLY LIKE CAIMAN
# =============================================================================
def apply_shift_caiman(img, shift):

    row_shift, col_shift = shift

    h, w = img.shape

    M = np.array([
        [1, 0, col_shift],
        [0, 1, row_shift]
    ], dtype=np.float32)

    shifted = cv2.warpAffine(
        img.astype(np.float32),
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    shifted = np.nan_to_num(shifted)

    shifted = np.clip(
        shifted,
        0,
        None
    )

    return shifted


# def apply_shift_caiman(img, shift):
# # we thought this function introduces some artefacts, but that is likely wrong. so consider switching back but the
# # other function is giving the same correspondence to caimans mc.
#
#     row_shift, col_shift = shift
#
#     h, w = img.shape
#
#     M = np.array([
#         [1, 0, col_shift],
#         [0, 1, row_shift]
#     ], dtype=np.float32)
#
#     min_ = np.nanmin(img)
#     max_ = np.nanmax(img)
#
#     shifted = cv2.warpAffine(
#         img.astype(np.float32),
#         M,
#         (w, h),
#         flags=cv2.INTER_CUBIC,
#         borderMode=cv2.BORDER_REFLECT
#     )
#
#     shifted = np.clip(
#         shifted,
#         min_,
#         max_
#     )
#
#     return shifted


# =============================================================================
# MAIN PUBLIC FUNCTION
# =============================================================================


def motion_correction_rigid(
        movie,
        max_shift=(20, 20),
        gSig_filt=7,
        upsample_factor=10,
        niter_rig=1,
        bin_window=100,
        template=None,
        verbose=True,
        out=None,
        chunk_size=500,
):
    """
    Works with:
        - numpy arrays
        - memmaps
        - zarr arrays

    Parameters
    ----------
    movie : array-like
        Shape (T, H, W)

    out : None | str | zarr.Array
        If provided:
            - creates/writes corrected movie lazily
            - avoids holding full movie in RAM

    chunk_size : int
        Number of frames processed at once.
    """
    import zarr
    import numpy as np
    from tqdm import tqdm
    T, H, W = movie.shape

    # -----------------------------------------------------------------
    # output allocation
    # -----------------------------------------------------------------

    if out is None:

        corrected = np.asarray(movie, dtype=np.float32).copy()

    else:

        if isinstance(out, str):

            corrected = zarr.open(
                out,
                mode="w",
                shape=(T, H, W),
                chunks=(min(chunk_size, T), H, W),
                dtype="float32",
            )

        else:
            corrected = out

        # initialize from input
        for start in range(0, T, chunk_size):

            stop = min(start + chunk_size, T)

            corrected[start:stop] = np.asarray(
                movie[start:stop],
                dtype=np.float32
            )

    shifts_total = np.zeros((T, 2), dtype=np.float32)

    # -----------------------------------------------------------------
    # initial template
    # -----------------------------------------------------------------

    if template is None:

        filtered = np.zeros((T, H, W), dtype=np.float32)

        iterator = range(0, T, chunk_size)

        if verbose:
            iterator = tqdm(iterator, desc="initial filtering")

        for start in iterator:

            stop = min(start + chunk_size, T)

            chunk = np.asarray(
                movie[start:stop],
                dtype=np.float32
            )

            for i in range(chunk.shape[0]):

                filtered[start + i] = high_pass_filter_space(
                    chunk[i],
                    gSig_filt
                )

        template = caiman_bin_median(
            filtered,
            window=bin_window
        )

    # -----------------------------------------------------------------
    # rigid iterations
    # -----------------------------------------------------------------

    for iteration in range(niter_rig):

        if verbose:
            print(
                f"\nRigid iteration "
                f"{iteration+1}/{niter_rig}"
            )

        filtered_template = high_pass_filter_space(
            template,
            gSig_filt
        )

        shifts_iter = np.zeros((T, 2), dtype=np.float32)

        iterator = range(T)

        if verbose:
            iterator = tqdm(iterator)

        # -------------------------------------------------------------
        # framewise registration
        # -------------------------------------------------------------

        for t in iterator:

            frame = np.asarray(
                corrected[t],
                dtype=np.float32
            )

            filtered_frame = high_pass_filter_space(
                frame,
                gSig_filt
            )

            shift = register_translation_caiman(
                filtered_template,
                filtered_frame,
                upsample_factor=upsample_factor,
                max_shifts=max_shift
            )

            corrected_frame = apply_shift_caiman(
                frame,
                shift
            )

            corrected[t] = corrected_frame
            shifts_iter[t] = shift

        shifts_total += shifts_iter

        # -------------------------------------------------------------
        # update template
        # -------------------------------------------------------------

        filtered_corrected = np.zeros(
            (T, H, W),
            dtype=np.float32
        )

        iterator = range(0, T, chunk_size)

        if verbose:
            iterator = tqdm(iterator, desc="template update")

        for start in iterator:

            stop = min(start + chunk_size, T)

            chunk = np.asarray(
                corrected[start:stop],
                dtype=np.float32
            )

            for i in range(chunk.shape[0]):

                filtered_corrected[start + i] = (
                    high_pass_filter_space(
                        chunk[i],
                        gSig_filt
                    )
                )

        template = caiman_bin_median(
            filtered_corrected,
            window=bin_window
        )

    return corrected, shifts_total