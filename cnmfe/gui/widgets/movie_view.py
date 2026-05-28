"""Movie pane: matplotlib QtAgg canvas showing one native AVI frame with the
selected component's contour and centroid overlaid.

Design points
-------------
* One ``imshow`` artist is created on first frame and reused via ``set_data``
  on every frame change -- cheap O(H·W) blit, no canvas teardown.
* Contour collections are removed-and-re-added when the selected component
  changes (a small Matplotlib API quirk: ``QuadContourSet`` has no in-place
  update). The frame imshow is NOT touched on a contour change.
* Contrast (vmin/vmax) is auto-set on first frame; the user can pin it with
  ``set_contrast``.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MovieView(QWidget):
    """Display ``(H, W) uint8`` frames with a selectable contour overlay."""

    # Emitted when the user clicks on the image (we forward image coords as
    # (y, x) so the panel can label or seed an ROI later).
    pixelClicked = pyqtSignal(int, int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._fig = Figure(figsize=(5, 4), constrained_layout=True)
        self._ax = self._fig.add_subplot(111)
        self._ax.set_axis_off()
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

        # Artists -- created on first call to set_frame / set_contour.
        self._im = None
        self._contour_artists: list = []
        self._centroid_artist = None

        self._vmin = None
        self._vmax = None
        self._dims: tuple[int, int] | None = None

        self._canvas.mpl_connect("button_press_event", self._on_click)

    # ---- frame -------------------------------------------------------

    def set_frame(self, frame: np.ndarray) -> None:
        """Update the displayed frame in-place."""
        if frame.ndim != 2:
            raise ValueError(f"frame must be 2D, got {frame.shape}")
        if self._im is None:
            # First frame: pick contrast from percentiles and stamp dims.
            self._dims = frame.shape
            if self._vmin is None or self._vmax is None:
                self._vmin = float(np.percentile(frame, 1))
                self._vmax = float(np.percentile(frame, 99))
                if self._vmax <= self._vmin:
                    self._vmax = self._vmin + 1.0
            self._im = self._ax.imshow(
                frame,
                cmap="gray",
                vmin=self._vmin,
                vmax=self._vmax,
                interpolation="nearest",
            )
            self._ax.set_xlim(-0.5, frame.shape[1] - 0.5)
            self._ax.set_ylim(frame.shape[0] - 0.5, -0.5)
        else:
            if frame.shape != self._dims:
                raise ValueError(
                    f"frame shape {frame.shape} != initial {self._dims}"
                )
            self._im.set_data(frame)
        self._canvas.draw_idle()

    def set_contrast(self, vmin: float, vmax: float) -> None:
        self._vmin, self._vmax = float(vmin), float(vmax)
        if self._im is not None:
            self._im.set_clim(self._vmin, self._vmax)
            self._canvas.draw_idle()

    # ---- contour + centroid -----------------------------------------

    def set_contour(
        self,
        contours: list[np.ndarray] | None,
        centroid: tuple[float, float] | None = None,
        color: str = "lime",
    ) -> None:
        """Replace the contour overlay with the given list of polylines.

        Each polyline is an ``(N, 2)`` array in ``(row, col)`` order
        (the ``skimage.measure.find_contours`` convention).
        """
        # Remove previous contour collection.
        for a in self._contour_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._contour_artists = []
        if contours:
            for c in contours:
                line, = self._ax.plot(
                    c[:, 1], c[:, 0], color=color, lw=1.2, alpha=0.9
                )
                self._contour_artists.append(line)

        # Centroid marker.
        if self._centroid_artist is not None:
            try:
                self._centroid_artist.remove()
            except Exception:
                pass
            self._centroid_artist = None
        if centroid is not None:
            cy, cx = centroid
            self._centroid_artist = self._ax.scatter(
                [cx], [cy], s=24, marker="+", c=color, linewidths=1.2
            )

        self._canvas.draw_idle()

    def clear_contour(self) -> None:
        self.set_contour(None, None)

    # ---- internal ---------------------------------------------------

    def _on_click(self, event) -> None:
        if event.inaxes is not self._ax or event.xdata is None:
            return
        y, x = int(round(event.ydata)), int(round(event.xdata))
        if self._dims is None:
            return
        H, W = self._dims
        if 0 <= y < H and 0 <= x < W:
            self.pixelClicked.emit(y, x)
