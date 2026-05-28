"""Main window for the CNMFe curation GUI.

Layout
------
``QMainWindow`` -> central ``QSplitter`` (horizontal):
    LEFT  : ``ComponentList`` (table + filter toolbar)
    RIGHT : ``QSplitter`` (vertical):
              TOP    : ``MovieView`` + ``TransportBar``
              BOTTOM : ``TraceView`` + ``CurationPanel``

Wiring (signal -> slot)
-----------------------
* ComponentList.componentSelected(k)   -> show k everywhere, jump to peak frame
* TransportBar.frameChanged(t_native)  -> MovieView, TraceView cursor
* TraceView.frameClicked(t_native)     -> TransportBar
* CurationPanel.*                      -> CurationStore mutators
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSlider,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from cnmfe.gui.avi_reader import AviReader
from cnmfe.gui.contours import component_contour, footprint_image
from cnmfe.gui.curation_store import CurationStore, KMismatchError
from cnmfe.gui.data_loader import SessionData
from cnmfe.gui.widgets.component_list import ComponentList
from cnmfe.gui.widgets.curation_panel import CurationPanel
from cnmfe.gui.widgets.movie_view import MovieView
from cnmfe.gui.widgets.trace_view import TraceView
from cnmfe.gui.widgets.transport import TransportBar

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Loading flow (factored so __main__.py and a test fixture can share it)
# ----------------------------------------------------------------------

def open_session(
    results_dir: "str | Path",
    avi_folder: "str | Path",
    *,
    ds_meta_path: "str | Path | None" = None,
    pattern: str = "*.avi",
) -> tuple[SessionData, AviReader, CurationStore]:
    """Load all the pieces needed by ``MainWindow``."""
    results_dir = Path(results_dir)
    avi_folder = Path(avi_folder)

    reader = AviReader(avi_folder, pattern=pattern)
    session = SessionData.load(
        results_dir,
        avi_folder,
        ds_meta_path=ds_meta_path,
        expected_avi_dims=reader.dims,
        expected_avi_n_frames=reader.n_frames,
    )
    store = CurationStore.load_or_seed(
        results_dir, auto_accepted_mask=session.auto_accepted
    )
    return session, reader, store


# ----------------------------------------------------------------------
# MainWindow
# ----------------------------------------------------------------------

class MainWindow(QMainWindow):
    """The whole curation GUI."""

    def __init__(
        self,
        session: SessionData,
        reader: AviReader,
        store: CurationStore,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.session = session
        self.reader = reader
        self.store = store
        self._current_k: int | None = None
        self._dirty = False

        self.setWindowTitle(f"cnmfe curation — {session.results_dir}")
        self.resize(1400, 900)

        # ---- Widgets ----
        self.component_list = ComponentList(session, store)
        self.movie_view = MovieView()
        self.transport = TransportBar(reader.n_frames)
        self.trace_view = TraceView()
        self.trace_view.set_time_map(session.time_map)
        self.curation_panel = CurationPanel()
        self.view_controls = self._build_view_controls()

        # ---- Layout ----
        right_top = QWidget()
        right_top_layout = QVBoxLayout(right_top)
        right_top_layout.setContentsMargins(0, 0, 0, 0)
        right_top_layout.addWidget(self.movie_view, 1)
        right_top_layout.addWidget(self.view_controls, 0)
        right_top_layout.addWidget(self.transport, 0)

        right_bot = QWidget()
        right_bot_layout = QVBoxLayout(right_bot)
        right_bot_layout.setContentsMargins(0, 0, 0, 0)
        right_bot_layout.addWidget(self.trace_view, 1)
        right_bot_layout.addWidget(self.curation_panel, 0)

        right_split = QSplitter(Qt.Orientation.Vertical)
        right_split.addWidget(right_top)
        right_split.addWidget(right_bot)
        right_split.setStretchFactor(0, 3)
        right_split.setStretchFactor(1, 2)

        main_split = QSplitter(Qt.Orientation.Horizontal)
        main_split.addWidget(self.component_list)
        main_split.addWidget(right_split)
        main_split.setStretchFactor(0, 1)
        main_split.setStretchFactor(1, 3)
        self.setCentralWidget(main_split)

        # ---- Status bar ----
        self.status = QStatusBar(self)
        self.setStatusBar(self.status)
        self._save_status = QLabel("saved")
        self.status.addPermanentWidget(self._save_status)

        # ---- Menu ----
        self._build_menu()

        # ---- Wiring ----
        self.component_list.componentSelected.connect(self._on_component_selected)
        self.component_list.mergeRequested.connect(self._on_merge_from_list)
        self.transport.frameChanged.connect(self._on_frame_changed)
        self.trace_view.frameClicked.connect(self.transport.set_frame)
        self.curation_panel.acceptToggled.connect(self._on_accept)
        self.curation_panel.noteEdited.connect(self._on_note)
        self.curation_panel.tagsEdited.connect(self._on_tags)
        self.curation_panel.markReviewed.connect(self._on_reviewed)
        self.curation_panel.nextUnreviewed.connect(self._jump_next_unreviewed)
        self.curation_panel.mergeRequested.connect(
            self.component_list.emit_merge_for_selection
        )
        self.curation_panel.splitRequested.connect(self._on_split)

        # ---- Shortcuts ----
        self._install_shortcuts()

        # ---- Initial state ----
        self.transport.set_frame(0, emit=False)
        first_frame = self.reader.get(0)
        self.movie_view.set_frame(first_frame)
        if session.K > 0:
            self.component_list.jump_to(0)

    # ------------------------------------------------------------------
    # View controls (contrast + contour level)
    # ------------------------------------------------------------------

    def _build_view_controls(self) -> QWidget:
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(4, 0, 4, 0)

        h.addWidget(QLabel("contrast lo:"))
        self.vmin_spin = QDoubleSpinBox()
        self.vmin_spin.setRange(0.0, 255.0)
        self.vmin_spin.setValue(0.0)
        self.vmin_spin.setDecimals(0)
        h.addWidget(self.vmin_spin)

        h.addWidget(QLabel("hi:"))
        self.vmax_spin = QDoubleSpinBox()
        self.vmax_spin.setRange(1.0, 255.0)
        self.vmax_spin.setValue(255.0)
        self.vmax_spin.setDecimals(0)
        h.addWidget(self.vmax_spin)

        self.auto_btn = QLabel("  ")  # spacer
        h.addWidget(self.auto_btn)

        h.addStretch(1)

        h.addWidget(QLabel("contour level (× peak):"))
        self.level_spin = QDoubleSpinBox()
        self.level_spin.setRange(0.05, 0.95)
        self.level_spin.setSingleStep(0.05)
        self.level_spin.setValue(0.30)
        self.level_spin.setDecimals(2)
        h.addWidget(self.level_spin)

        self.vmin_spin.valueChanged.connect(self._apply_contrast)
        self.vmax_spin.valueChanged.connect(self._apply_contrast)
        self.level_spin.valueChanged.connect(self._refresh_contour)
        return box

    def _apply_contrast(self, _=None) -> None:
        lo, hi = float(self.vmin_spin.value()), float(self.vmax_spin.value())
        if hi <= lo:
            hi = lo + 1.0
        self.movie_view.set_contrast(lo, hi)

    def _refresh_contour(self, _=None) -> None:
        if self._current_k is None:
            return
        fp = footprint_image(
            self.session.A_native_csc, self._current_k, self.session.H, self.session.W
        )
        contours = component_contour(fp, level_frac=float(self.level_spin.value()))
        centroid = tuple(float(v) for v in self.session.centroids[self._current_k])
        self.movie_view.set_contour(contours, centroid=centroid)

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")

        save_act = QAction("Save now", self)
        save_act.setShortcut(QKeySequence("Ctrl+S"))
        save_act.triggered.connect(self._save_now)
        file_menu.addAction(save_act)

        reviewer_act = QAction("Set reviewer…", self)
        reviewer_act.triggered.connect(self._set_reviewer)
        file_menu.addAction(reviewer_act)

        file_menu.addSeparator()

        quit_act = QAction("Quit", self)
        quit_act.setShortcut(QKeySequence("Ctrl+Q"))
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

    # ------------------------------------------------------------------
    # Shortcuts (work globally regardless of focused widget)
    # ------------------------------------------------------------------

    def _install_shortcuts(self) -> None:
        QShortcut(QKeySequence("Space"), self).activated.connect(
            self.transport.toggle_play
        )
        QShortcut(QKeySequence("Right"), self).activated.connect(
            lambda: self.transport.set_frame(self.transport.current_frame() + 1)
        )
        QShortcut(QKeySequence("Left"), self).activated.connect(
            lambda: self.transport.set_frame(self.transport.current_frame() - 1)
        )
        QShortcut(QKeySequence("Shift+Right"), self).activated.connect(
            lambda: self.transport.set_frame(self.transport.current_frame() + 10)
        )
        QShortcut(QKeySequence("Shift+Left"), self).activated.connect(
            lambda: self.transport.set_frame(self.transport.current_frame() - 10)
        )
        QShortcut(QKeySequence("J"), self).activated.connect(self._prev_component)
        QShortcut(QKeySequence("K"), self).activated.connect(self._next_component)
        QShortcut(QKeySequence("A"), self).activated.connect(self._toggle_accept)
        QShortcut(QKeySequence("R"), self).activated.connect(self._toggle_reviewed)
        QShortcut(QKeySequence("N"), self).activated.connect(self._jump_next_unreviewed)

    def _prev_component(self) -> None:
        if self._current_k is None or self.session.K == 0:
            return
        nxt = (self._current_k - 1) % self.session.K
        self.component_list.jump_to(nxt)

    def _next_component(self) -> None:
        if self._current_k is None or self.session.K == 0:
            return
        nxt = (self._current_k + 1) % self.session.K
        self.component_list.jump_to(nxt)

    def _toggle_accept(self) -> None:
        if self._current_k is None:
            return
        comp = self.store.component(self._current_k)
        self._on_accept(self._current_k, not comp.accepted)
        # Refresh the panel so the toggle button reflects reality.
        self.curation_panel.load_component(self._current_k, self.store.component(self._current_k))

    def _toggle_reviewed(self) -> None:
        if self._current_k is None:
            return
        comp = self.store.component(self._current_k)
        new = comp.review_state != "reviewed"
        self._on_reviewed(self._current_k, new)
        self.curation_panel.load_component(self._current_k, self.store.component(self._current_k))

    def _jump_next_unreviewed(self) -> None:
        K = self.session.K
        start = (self._current_k + 1) if self._current_k is not None else 0
        for offset in range(K):
            k = (start + offset) % K
            if self.store.component(k).review_state == "unreviewed":
                self.component_list.jump_to(k)
                return
        self.status.showMessage("No more unreviewed components.", 3000)

    # ------------------------------------------------------------------
    # Selection / frame
    # ------------------------------------------------------------------

    def _on_component_selected(self, k: int) -> None:
        k = int(k)
        self._current_k = k
        # Movie: contour + centroid (uses current contour-level setting)
        fp = footprint_image(self.session.A_native_csc, k, self.session.H, self.session.W)
        level = float(self.level_spin.value()) if hasattr(self, "level_spin") else 0.3
        contours = component_contour(fp, level_frac=level)
        centroid = tuple(float(v) for v in self.session.centroids[k])
        self.movie_view.set_contour(contours, centroid=centroid)

        # Jump to the peak frame of this component (best chance of seeing a transient).
        peak_t_native = self.session.peak_native_frame(k)
        peak_t_native = max(0, min(self.reader.n_frames - 1, peak_t_native))
        self.transport.set_frame(peak_t_native)

        # Trace + panel.
        self.trace_view.set_component(k, self.session.C, self.session.S, self.session.YrA)
        self.trace_view.set_cursor_native(peak_t_native)
        self.curation_panel.load_component(k, self.store.component(k))

    def _on_frame_changed(self, t_native: int) -> None:
        t_native = int(t_native)
        try:
            frame = self.reader.get(t_native)
        except IndexError:
            return
        self.movie_view.set_frame(frame)
        self.trace_view.set_cursor_native(t_native)
        t_e = self.session.time_map.native_to_extraction(t_native)
        self.transport.set_extraction_label(t_native, t_e)

    # ------------------------------------------------------------------
    # Curation mutations
    # ------------------------------------------------------------------

    def _flash_save(self) -> None:
        self._save_status.setText("saved")

    def _on_accept(self, k: int, value: bool) -> None:
        self.store.set_accepted(k, value)
        self.store.save()
        self.component_list.row_changed(k)
        self._flash_save()

    def _on_note(self, k: int, note: str) -> None:
        self.store.set_note(k, note)
        self.store.save()
        self.component_list.row_changed(k)
        self._flash_save()

    def _on_tags(self, k: int, tags: list[str]) -> None:
        self.store.set_tags(k, tags)
        self.store.save()
        self.component_list.row_changed(k)
        self._flash_save()

    def _on_reviewed(self, k: int, reviewed: bool) -> None:
        self.store.mark_reviewed(k, reviewed)
        self.store.save()
        self.component_list.row_changed(k)
        self._flash_save()

    def _on_merge_from_list(self, ks: list[int]) -> None:
        try:
            self.store.add_merge_group(ks)
        except Exception as e:
            QMessageBox.warning(self, "Merge failed", str(e))
            return
        self.store.save()
        self.status.showMessage(f"Queued merge of components {ks}.", 3000)
        self._flash_save()

    def _on_split(self, k: int, note: str) -> None:
        self.store.add_split_request(k, note)
        self.store.save()
        self.status.showMessage(f"Queued split request for k={k}.", 3000)
        self._flash_save()

    def _save_now(self) -> None:
        self.store.save()
        self._flash_save()

    def _set_reviewer(self) -> None:
        name, ok = QInputDialog.getText(
            self, "Set reviewer", "Reviewer name:", text=self.store.state.reviewer or ""
        )
        if ok:
            self.store.set_reviewer(name or None)
            self.store.save()
            self._flash_save()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        try:
            self.store.save()
        except Exception as e:
            QMessageBox.warning(self, "Save failed", str(e))
        try:
            self.reader.close()
        except Exception:
            pass
        super().closeEvent(event)
