"""minicnmfe — clean CNMFe for 1-photon calcium imaging."""

from minicnmfe.cellreg import CellRegResult, register_sessions
from minicnmfe.concat_avis_to_zarr import concat_avis_to_zarr
from minicnmfe.evaluate import auto_evaluate_components
from minicnmfe.io import (
    avi_to_zarr,
    open_zarr,
    save_zarr,
    stage_zarr_to_local,
    transpose_zarr_to_pixel_major,
)
from minicnmfe.pipeline import CNMFe, CNMFeParams

__all__ = [
    "CNMFe",
    "CNMFeParams",
    "CellRegResult",
    "auto_evaluate_components",
    "avi_to_zarr",
    "concat_avis_to_zarr",
    "open_zarr",
    "register_sessions",
    "save_zarr",
    "stage_zarr_to_local",
    "transpose_zarr_to_pixel_major",
]
