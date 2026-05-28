"""Sortable, filterable table of components.

Columns: ``k | ✓ | snr_amp | pixel_count | tag | note | reviewed``.

Selecting a row emits ``componentSelected(k: int)`` so the rest of the GUI
follows. Ctrl/Shift-click for multi-select gives the parent window a list
to feed into ``merge``.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QSortFilterProxyModel,
    Qt,
    pyqtSignal,
)
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)


COLUMNS = ["k", "✓", "snr_amp", "pixels", "tags", "note", "rev"]


def _row_color(accepted: bool, auto_accepted: bool) -> QColor:
    if accepted and auto_accepted:
        return QColor(225, 245, 225)   # light green
    if not accepted and auto_accepted:
        return QColor(250, 220, 220)   # light red (manual reject)
    if accepted and not auto_accepted:
        return QColor(220, 230, 255)   # light blue (manual rescue)
    return QColor(235, 235, 235)       # light grey (both False)


class ComponentTableModel(QAbstractTableModel):
    """Wraps a ``SessionData`` + ``CurationStore`` for the QTableView."""

    def __init__(self, session, store, parent=None):
        super().__init__(parent)
        self.session = session
        self.store = store
        cols = self.session.summary_columns()
        self._snr = cols["snr_amp"]
        self._npx = cols["pixel_count"]

    # ---- Qt model API ------------------------------------------------

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else self.session.K

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if (
            role != Qt.ItemDataRole.DisplayRole
            or orientation != Qt.Orientation.Horizontal
        ):
            return None
        return COLUMNS[section]

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        k = index.row()
        c = index.column()
        comp = self.store.component(k)

        if role == Qt.ItemDataRole.DisplayRole:
            if c == 0:
                return str(k)
            if c == 1:
                return "✓" if comp.accepted else "✗"
            if c == 2:
                v = float(self._snr[k])
                return f"{v:.1f}" if np.isfinite(v) else "—"
            if c == 3:
                return str(int(self._npx[k]))
            if c == 4:
                return ", ".join(comp.tags)
            if c == 5:
                return comp.note
            if c == 6:
                return "·" if comp.review_state == "unreviewed" else "R"
            return ""

        if role == Qt.ItemDataRole.TextAlignmentRole:
            if c in (0, 1, 2, 3, 6):
                return int(Qt.AlignmentFlag.AlignCenter)

        if role == Qt.ItemDataRole.BackgroundRole:
            return QBrush(_row_color(comp.accepted, comp.auto_accepted))

        # Sort by numeric value for snr_amp / pixels.
        if role == Qt.ItemDataRole.UserRole:
            if c == 0:
                return k
            if c == 1:
                return int(comp.accepted)
            if c == 2:
                v = float(self._snr[k])
                return v if np.isfinite(v) else -1.0
            if c == 3:
                return int(self._npx[k])
            if c == 6:
                return int(comp.review_state == "reviewed")
            return self.data(index, Qt.ItemDataRole.DisplayRole)

        return None

    # ---- mutation broadcasting --------------------------------------

    def row_changed(self, k: int) -> None:
        top = self.index(k, 0)
        bot = self.index(k, self.columnCount() - 1)
        self.dataChanged.emit(top, bot)


class ComponentList(QWidget):
    """Filterable table view + simple filter buttons."""

    componentSelected = pyqtSignal(int)            # single-click on a row
    mergeRequested = pyqtSignal(list)              # ctrl-click multi-select

    def __init__(self, session, store, parent: QWidget | None = None):
        super().__init__(parent)
        self.session = session
        self.store = store
        self.model = ComponentTableModel(session, store)

        # Proxy enables sorting by the UserRole numeric value.
        self.proxy = QSortFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.setSortRole(Qt.ItemDataRole.UserRole)
        self.proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)

        self.view = QTableView(self)
        self.view.setModel(self.proxy)
        self.view.setSortingEnabled(True)
        self.view.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        self.view.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.view.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.view.verticalHeader().setVisible(False)
        h = self.view.horizontalHeader()
        h.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        h.setStretchLastSection(True)

        # Top filter bar.
        self.filter_label = QLabel(f"K = {session.K}")
        self.btn_all = QPushButton("All")
        self.btn_unreviewed = QPushButton("Unreviewed")
        self.btn_rejected = QPushButton("Rejected")
        self.btn_touched = QPushButton("Manually touched")
        for b in (self.btn_all, self.btn_unreviewed, self.btn_rejected, self.btn_touched):
            b.setCheckable(True)
        self.btn_all.setChecked(True)

        topbar = QHBoxLayout()
        topbar.setContentsMargins(2, 2, 2, 2)
        topbar.addWidget(self.filter_label)
        for b in (self.btn_all, self.btn_unreviewed, self.btn_rejected, self.btn_touched):
            topbar.addWidget(b)
        topbar.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addLayout(topbar)
        layout.addWidget(self.view)

        # Wiring
        sel = self.view.selectionModel()
        sel.currentRowChanged.connect(self._on_current_row)
        for b in (self.btn_all, self.btn_unreviewed, self.btn_rejected, self.btn_touched):
            b.clicked.connect(self._on_filter_clicked)

        # The filter is implemented by overriding filterAcceptsRow via a
        # dynamic callable; QSortFilterProxyModel is too restrictive otherwise.
        self._filter_mode = "all"
        self.proxy.filterAcceptsRow = self._filter_accepts_row  # type: ignore

    # ---- selection ---------------------------------------------------

    def _on_current_row(self, current, previous) -> None:
        if not current.isValid():
            return
        src = self.proxy.mapToSource(current)
        k = src.row()
        self.componentSelected.emit(int(k))

    def selected_ks(self) -> list[int]:
        rows = self.view.selectionModel().selectedRows()
        return [
            self.proxy.mapToSource(r).row() for r in rows
        ]

    def emit_merge_for_selection(self) -> None:
        ks = sorted(set(self.selected_ks()))
        if len(ks) >= 2:
            self.mergeRequested.emit(ks)

    # ---- filters -----------------------------------------------------

    def _on_filter_clicked(self) -> None:
        sender = self.sender()
        # Uncheck the others.
        for b in (
            self.btn_all,
            self.btn_unreviewed,
            self.btn_rejected,
            self.btn_touched,
        ):
            if b is not sender:
                b.setChecked(False)
        if not sender.isChecked():
            # User unchecked the active one -> fall back to All.
            self.btn_all.setChecked(True)
            self._filter_mode = "all"
        else:
            text = sender.text().lower()
            self._filter_mode = (
                "unreviewed" if "unreviewed" in text
                else "rejected" if "rejected" in text
                else "touched" if "touched" in text
                else "all"
            )
        self.proxy.invalidateFilter()

    def _filter_accepts_row(self, source_row: int, source_parent) -> bool:
        comp = self.store.component(source_row)
        mode = self._filter_mode
        if mode == "all":
            return True
        if mode == "unreviewed":
            return comp.review_state == "unreviewed"
        if mode == "rejected":
            return not comp.accepted
        if mode == "touched":
            return comp.accepted != comp.auto_accepted or bool(comp.note) or bool(comp.tags) or comp.review_state == "reviewed"
        return True

    # ---- external mutators -----------------------------------------

    def row_changed(self, k: int) -> None:
        self.model.row_changed(k)

    def jump_to(self, k: int) -> None:
        """Select component k (in source coords), even under a filter."""
        src = self.model.index(int(k), 0)
        proxy = self.proxy.mapFromSource(src)
        if not proxy.isValid():
            # Filtered out -- temporarily fall back to All and retry.
            self.btn_all.setChecked(True)
            self._filter_mode = "all"
            self.proxy.invalidateFilter()
            proxy = self.proxy.mapFromSource(src)
        self.view.setCurrentIndex(proxy)
        self.view.scrollTo(proxy)
