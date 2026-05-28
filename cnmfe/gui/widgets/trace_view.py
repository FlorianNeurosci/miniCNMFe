"""Trace pane: C + YrA (faint), C (bold), S vlines on a twin axis, plus a
vertical cursor at the current extraction frame. Clicking the canvas emits
``frameClicked(t_native: int)`` so the transport bar can follow.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class TraceView(QWidget):
    """Single-component trace canvas."""

    frameClicked = pyqtSignal(int)  # native frame index

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._fig = Figure(figsize=(6, 2.5), constrained_layout=True)
        self._ax = self._fig.add_subplot(111)
        self._ax.set_ylabel("C / C+YrA")
        self._ax.set_xlabel("extraction frame")
        self._ax2 = self._ax.twinx()
        self._ax2.set_ylabel("S", color="tomato")
        self._ax2.tick_params(axis="y", colors="tomato")
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

        self._line_cyra = None
        self._line_c = None
        self._spikes = None
        self._cursor = self._ax.axvline(0, color="red", lw=0.8, alpha=0.7)
        self._title = self._ax.set_title("")

        self._time_map = None  # set by main window
        self._canvas.mpl_connect("button_press_event", self._on_click)

    # ---- model wiring -----------------------------------------------

    def set_time_map(self, time_map) -> None:
        self._time_map = time_map

    # ---- per-component update ---------------------------------------

    def set_component(
        self,
        k: int,
        C: np.ndarray,
        S: np.ndarray,
        YrA: np.ndarray | None,
    ) -> None:
        """Plot traces for component ``k``."""
        T = C.shape[1]
        xs = np.arange(T, dtype=np.float32)
        c = np.asarray(C[k], dtype=np.float32)
        s = np.asarray(S[k], dtype=np.float32)
        cyra = c + (np.asarray(YrA[k], dtype=np.float32) if YrA is not None else 0.0)

        if self._line_cyra is None:
            (self._line_cyra,) = self._ax.plot(
                xs, cyra, lw=0.7, color="0.5", alpha=0.7, label="C+YrA"
            )
            (self._line_c,) = self._ax.plot(
                xs, c, lw=1.1, color="k", label="C"
            )
            self._ax.legend(loc="upper right", fontsize=8)
        else:
            self._line_cyra.set_data(xs, cyra)
            self._line_c.set_data(xs, c)

        # Spikes: remove + re-add the vlines collection on every k change.
        if self._spikes is not None:
            try:
                self._spikes.remove()
            except Exception:
                pass
            self._spikes = None
        nz = np.nonzero(s)[0]
        if nz.size:
            self._spikes = self._ax2.vlines(
                nz, 0, s[nz], color="tomato", alpha=0.7, linewidth=1.0
            )

        # Axes limits.
        self._ax.set_xlim(0, max(1, T - 1))
        lo, hi = float(min(c.min(), cyra.min())), float(max(c.max(), cyra.max()))
        if hi <= lo:
            hi = lo + 1.0
        pad = 0.05 * (hi - lo)
        self._ax.set_ylim(lo - pad, hi + pad)
        if nz.size:
            self._ax2.set_ylim(0, float(s[nz].max()) * 1.05)
        else:
            self._ax2.set_ylim(0, 1.0)

        self._title.set_text(f"component k={k}")
        self._canvas.draw_idle()

    # ---- cursor at a native frame -----------------------------------

    def set_cursor_native(self, t_native: int) -> None:
        """Move the vertical cursor to the extraction frame corresponding to
        ``t_native``. If ``t_native`` is outside the cutout window the cursor
        is dimmed and clamped to the boundary.
        """
        if self._time_map is None:
            self._cursor.set_xdata([t_native])
            self._canvas.draw_idle()
            return
        t_e = self._time_map.native_to_extraction(t_native)
        if t_e is None:
            self._cursor.set_alpha(0.25)
            # clamp to nearest valid t_e
            t_clamped = min(
                self._time_map.extraction_T - 1,
                max(
                    0,
                    (t_native - self._time_map.t0_cutout)
                    // max(1, self._time_map.tsub),
                ),
            )
            self._cursor.set_xdata([t_clamped])
        else:
            self._cursor.set_alpha(0.7)
            self._cursor.set_xdata([t_e])
        self._canvas.draw_idle()

    # ---- click forwarding -------------------------------------------

    def _on_click(self, event) -> None:
        if event.inaxes not in (self._ax, self._ax2) or event.xdata is None:
            return
        t_e = int(round(event.xdata))
        if self._time_map is not None:
            t_native = self._time_map.extraction_to_native(t_e)
        else:
            t_native = t_e
        self.frameClicked.emit(int(t_native))
