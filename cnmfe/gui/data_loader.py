"""Load a CNMFe results directory + raw AVI folder into a coherent
``SessionData`` object for the curation GUI.

Coordinate space (per the approved plan): the GUI **displays raw native AVI
frames** and **maps the footprints into native coords** via the existing
``CNMFe.place_in_full_fov`` and ``CNMFe.upsample_to_native`` helpers. Traces
stay at *extraction* rate and frame count; a ``TimeMap`` maps the user's
native frame index back to an extraction frame index so the trace cursor
tracks the movie cursor.

This module handles all four cases of ``{cutout?, downsample?}``:
* neither — A and traces already in native coords
* cutout only — call ``place_in_full_fov``
* downsample only — call ``upsample_to_native`` with a ``ds_meta.json``
* both — ``place_in_full_fov`` first, then ``upsample_to_native``

It deliberately does NOT load the AVI movie here — that's owned by
``AviReader``. We only need ``orig_dims`` for the contour overlay.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.sparse as sp

from cnmfe.pipeline import CNMFe

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# TimeMap
# ----------------------------------------------------------------------

@dataclass
class TimeMap:
    """Map between native AVI frames and extraction-space frames.

    Native time t_native passes through the cutout temporal crop ``[t0, t1)``
    (if any) and is then binned by ``tsub`` (if any):

        t_extraction = (t_native - t0) // tsub      when t0 <= t_native < t1
                     = None                         otherwise

        t_native     = t0 + t_extraction * tsub
    """

    t0_cutout: int
    t1_cutout: int
    tsub: int
    orig_T: int
    extraction_T: int

    def native_to_extraction(self, t_native: int) -> int | None:
        if t_native < self.t0_cutout or t_native >= self.t1_cutout:
            return None
        t_e = (t_native - self.t0_cutout) // self.tsub
        if t_e >= self.extraction_T:
            return None
        return int(t_e)

    def extraction_to_native(self, t_e: int) -> int:
        return int(self.t0_cutout + int(t_e) * self.tsub)


# ----------------------------------------------------------------------
# SessionData
# ----------------------------------------------------------------------

@dataclass
class SessionData:
    """Everything the GUI needs once a session is opened.

    Attributes
    ----------
    model_native : CNMFe
        The original model with A mapped to NATIVE FOV coords (sparse).
        Don't use its ``C/S/YrA`` directly for the trace panel — those are
        kept separately in the extraction-rate ``C`` / ``S`` / ``YrA``
        fields here.
    A_native_csc : sp.csc_matrix
        ``(H_native * W_native, K)`` native-FOV footprints.
    H, W : int
        Native FOV dimensions (match the AVI frame).
    K : int
        Component count.
    centroids : np.ndarray
        ``(K, 2)`` ``[y, x]`` in native pixel coords.
    C, YrA, S : np.ndarray
        ``(K, T_e)`` at the EXTRACTION frame rate.
    eval_info : dict | None
    auto_accepted : np.ndarray
        ``(K,)`` bool — the original ``accepted_mask.npy``.
    time_map : TimeMap
    cutout : dict | None
    ds_meta : dict | None
    """

    results_dir: Path
    avi_folder: Path
    model_native: CNMFe
    A_native_csc: sp.csc_matrix
    H: int
    W: int
    K: int
    centroids: np.ndarray
    C: np.ndarray
    YrA: np.ndarray | None
    S: np.ndarray
    eval_info: dict | None
    auto_accepted: np.ndarray
    time_map: TimeMap
    cutout: dict | None
    ds_meta: dict | None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        results_dir: "str | Path",
        avi_folder: "str | Path",
        *,
        ds_meta_path: "str | Path | None" = None,
        expected_avi_dims: tuple[int, int] | None = None,
        expected_avi_n_frames: int | None = None,
    ) -> "SessionData":
        """Read a results dir + (optional) ``ds_meta.json`` into a SessionData.

        ``expected_avi_dims`` / ``expected_avi_n_frames`` are consumed when the
        caller has already opened an ``AviReader`` — we validate against those
        to catch wrong-folder mistakes. Pass None to skip.
        """
        results_dir = Path(results_dir)
        avi_folder = Path(avi_folder)
        if not results_dir.is_dir():
            raise FileNotFoundError(f"results dir not found: {results_dir}")

        # ----- 1. Load the raw model + manifest -----
        model = CNMFe.load(results_dir)
        if model.A is None or model.C is None:
            raise RuntimeError(
                f"{results_dir} does not contain a fit model (A/C missing)."
            )
        K = int(model.A.shape[1])
        T_e = int(model.C.shape[1])
        cutout = model.cutout  # set by CNMFe.load from manifest.json

        # ----- 2. Locate ds_meta.json if not given -----
        ds_meta = None
        if ds_meta_path is not None:
            ds_meta_path = Path(ds_meta_path)
            ds_meta = json.loads(ds_meta_path.read_text())
        else:
            for cand in (
                results_dir / "ds_meta.json",
                results_dir.parent / "ds_meta.json",
            ):
                if cand.exists():
                    ds_meta = json.loads(cand.read_text())
                    logger.info("Loaded ds_meta from %s", cand)
                    break

        # ----- 3. Build a NATIVE-FOV model (A only; traces kept at T_e) -----
        # Order matters when both transforms are present:
        # the model lives in extraction-space (cropped + downsampled), and
        # the cutout metadata records the FULL native FOV. So we must:
        #   (a) undo the downsample first -- this lifts A into the
        #       "cropped native" frame (matches ds_meta.orig_dims), and
        #   (b) then place that cropped-native A back into the full native
        #       FOV using the cutout bbox.
        # When only one is present, the other step is a no-op.
        m = model
        if ds_meta is not None:
            m = m.upsample_to_native(
                orig_dims=tuple(ds_meta["orig_dims"]),
                orig_T=int(ds_meta["orig_T"]),
                ds_meta=None,
                spatial_order=1,
            )
            # upsample_to_native doesn't carry self.cutout forward; restore
            # it so place_in_full_fov can find it on the next call.
            m.cutout = cutout
        if cutout is not None:
            m = m.place_in_full_fov(place_time=False)
            # m.dims is now full native; A is sparse at (H_native * W_native, K).

        A_native = m.A.tocsc()
        H, W = m.dims  # may equal model.dims if no transforms applied

        # ----- 4. Sanity check vs AVI folder -----
        if expected_avi_dims is not None and expected_avi_dims != (H, W):
            raise RuntimeError(
                f"AVI dims {expected_avi_dims} don't match native model dims "
                f"{(H, W)}. Wrong --avi-folder, or missing --ds-meta?"
            )

        # ----- 5. TimeMap -----
        if cutout is not None:
            t0, t1 = int(cutout["t_range"][0]), int(cutout["t_range"][1])
            orig_T_from_cutout = int(cutout["orig_T"])
        else:
            t0, t1 = 0, T_e * (int(ds_meta["tsub"]) if ds_meta else 1)
            orig_T_from_cutout = (
                int(ds_meta["orig_T"]) if ds_meta else T_e
            )
        tsub = int(ds_meta["tsub"]) if ds_meta else 1
        orig_T = (
            int(ds_meta["orig_T"]) if ds_meta else orig_T_from_cutout
        )
        # If both cutout and ds_meta are present, they should agree on orig_T.
        time_map = TimeMap(
            t0_cutout=t0,
            t1_cutout=t1,
            tsub=tsub,
            orig_T=orig_T,
            extraction_T=T_e,
        )

        if (
            expected_avi_n_frames is not None
            and expected_avi_n_frames != orig_T
        ):
            logger.warning(
                "AVI has %d frames but expected orig_T=%d (cutout/ds_meta). "
                "Possibly wrong AVI folder.",
                expected_avi_n_frames,
                orig_T,
            )

        # ----- 6. Centroids in native coords -----
        from cnmfe.gui.contours import precompute_centroids

        centroids = precompute_centroids(A_native, H, W)

        # ----- 7. auto_accepted mask -----
        if model.accepted_mask is None:
            auto_accepted = np.ones(K, dtype=bool)
        else:
            auto_accepted = np.asarray(model.accepted_mask, dtype=bool)

        return cls(
            results_dir=results_dir,
            avi_folder=avi_folder,
            model_native=m,
            A_native_csc=A_native,
            H=int(H),
            W=int(W),
            K=K,
            centroids=centroids,
            C=np.asarray(model.C, dtype=np.float32),
            YrA=(
                None if model.YrA is None
                else np.asarray(model.YrA, dtype=np.float32)
            ),
            S=np.asarray(model.S, dtype=np.float32),
            eval_info=model.eval_info,
            auto_accepted=auto_accepted,
            time_map=time_map,
            cutout=cutout,
            ds_meta=ds_meta,
        )

    # ------------------------------------------------------------------
    # Lightweight accessors for widgets
    # ------------------------------------------------------------------

    def footprint_2d(self, k: int) -> np.ndarray:
        """Return ``(H, W)`` float32 dense footprint in native coords."""
        from cnmfe.gui.contours import footprint_image

        return footprint_image(self.A_native_csc, k, self.H, self.W)

    def peak_native_frame(self, k: int) -> int:
        """Native frame index of ``argmax(C[k])`` — handy for jumping to a
        component's strongest event."""
        if self.C.shape[1] == 0:
            return self.time_map.t0_cutout
        t_e = int(np.argmax(self.C[k]))
        return self.time_map.extraction_to_native(t_e)

    def summary_columns(self) -> dict[str, np.ndarray]:
        """Per-component table columns: snr_amp, pixel_count, pass flags."""
        K = self.K
        if self.eval_info is None:
            return {
                "snr_amp": np.full(K, np.nan, dtype=np.float32),
                "pixel_count": np.zeros(K, dtype=np.int64),
                "pixel_pass": np.ones(K, dtype=bool),
                "snr_pass": np.ones(K, dtype=bool),
            }
        return {
            "snr_amp": np.asarray(self.eval_info["snr_amp"], dtype=np.float32),
            "pixel_count": np.asarray(
                self.eval_info["pixel_count"], dtype=np.int64
            ),
            "pixel_pass": np.asarray(self.eval_info["pixel_pass"], dtype=bool),
            "snr_pass": np.asarray(self.eval_info["snr_pass"], dtype=bool),
        }
