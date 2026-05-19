"""cnmfe — clean CNMFe for 1-photon calcium imaging."""

from cnmfe.evaluate import auto_evaluate_components
from cnmfe.io import avi_to_zarr, open_zarr, save_zarr
from cnmfe.pipeline import CNMFe, CNMFeParams

__all__ = [
    "CNMFe",
    "CNMFeParams",
    "auto_evaluate_components",
    "avi_to_zarr",
    "open_zarr",
    "save_zarr",
]
