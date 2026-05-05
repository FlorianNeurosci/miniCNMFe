"""cnmfe — clean CNMFe for 1-photon calcium imaging."""

from cnmfe.io import avi_to_zarr, open_zarr, save_zarr
from cnmfe.pipeline import CNMFe, CNMFeParams

__all__ = ["CNMFe", "CNMFeParams", "avi_to_zarr", "open_zarr", "save_zarr"]
