"""cnmfe — clean CNMFe for 1-photon calcium imaging."""

from cnmfe.evaluate import auto_evaluate_components
from cnmfe.io import (
    avi_to_zarr,
    open_zarr,
    save_zarr,
    stage_zarr_to_local,
    transpose_zarr_to_pixel_major,
)
from cnmfe.pipeline import CNMFe, CNMFeParams

__all__ = [
    "CNMFe",
    "CNMFeParams",
    "auto_evaluate_components",
    "avi_to_zarr",
    "open_zarr",
    "save_zarr",
    "stage_zarr_to_local",
    "transpose_zarr_to_pixel_major",
]
