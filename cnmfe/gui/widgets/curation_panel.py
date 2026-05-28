"""Right-hand-side curation panel.

A compact column of widgets that lives below the trace view:
* Accept / Reject toggle (the dominant action)
* Note field (debounced save)
* Tags entry (free-form, comma-separated)
* Buttons: Merge selected, Flag for split, Mark reviewed, Next unreviewed
"""

from __future__ import annotations

from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class CurationPanel(QWidget):
    """Drives mutations on the parent's ``CurationStore``."""

    acceptToggled = pyqtSignal(int, bool)
    noteEdited = pyqtSignal(int, str)
    tagsEdited = pyqtSignal(int, list)
    markReviewed = pyqtSignal(int, bool)
    nextUnreviewed = pyqtSignal()
    mergeRequested = pyqtSignal()
    splitRequested = pyqtSignal(int, str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._k: int | None = None

        # Widgets
        self.header = QLabel("(no component selected)")
        self.header.setStyleSheet("font-weight: bold;")
        self.accept_btn = QPushButton("Accepted ✓")
        self.accept_btn.setCheckable(True)
        self.accept_btn.setMinimumHeight(36)
        self.accept_btn.setStyleSheet("QPushButton { font-size: 14px; }")
        self.note_field = QLineEdit()
        self.note_field.setPlaceholderText("Note…")
        self.tags_field = QLineEdit()
        self.tags_field.setPlaceholderText("Tags (comma-separated)")

        self.review_btn = QPushButton("Mark reviewed")
        self.next_btn = QPushButton("Next unreviewed")
        self.merge_btn = QPushButton("Merge selected")
        self.split_btn = QPushButton("Flag for split")

        # Layout
        grid = QGridLayout()
        grid.setContentsMargins(4, 4, 4, 4)
        grid.addWidget(self.header, 0, 0, 1, 2)
        grid.addWidget(self.accept_btn, 1, 0, 1, 2)
        grid.addWidget(QLabel("Note:"), 2, 0)
        grid.addWidget(self.note_field, 2, 1)
        grid.addWidget(QLabel("Tags:"), 3, 0)
        grid.addWidget(self.tags_field, 3, 1)
        row_btns1 = QHBoxLayout()
        row_btns1.addWidget(self.review_btn)
        row_btns1.addWidget(self.next_btn)
        row_btns2 = QHBoxLayout()
        row_btns2.addWidget(self.merge_btn)
        row_btns2.addWidget(self.split_btn)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(grid)
        layout.addLayout(row_btns1)
        layout.addLayout(row_btns2)
        layout.addStretch(1)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        # Wiring
        self.accept_btn.clicked.connect(self._on_accept_clicked)
        self.review_btn.clicked.connect(self._on_review_clicked)
        self.next_btn.clicked.connect(self.nextUnreviewed.emit)
        self.merge_btn.clicked.connect(self.mergeRequested.emit)
        self.split_btn.clicked.connect(self._on_split_clicked)

        # Debounced save on note/tags typing.
        self._note_timer = QTimer(self)
        self._note_timer.setSingleShot(True)
        self._note_timer.setInterval(250)
        self._note_timer.timeout.connect(self._emit_note)
        self.note_field.textEdited.connect(lambda _: self._note_timer.start())

        self._tags_timer = QTimer(self)
        self._tags_timer.setSingleShot(True)
        self._tags_timer.setInterval(250)
        self._tags_timer.timeout.connect(self._emit_tags)
        self.tags_field.textEdited.connect(lambda _: self._tags_timer.start())

    # ---- public API --------------------------------------------------

    def load_component(self, k: int, comp) -> None:
        """Display the curation state for component ``k``."""
        self._k = int(k)
        self.header.setText(
            f"k = {k}   ·   auto: {'accept' if comp.auto_accepted else 'reject'}"
            f"   ·   review: {comp.review_state}"
        )
        self.accept_btn.blockSignals(True)
        self.accept_btn.setChecked(bool(comp.accepted))
        self._refresh_accept_button(comp.accepted)
        self.accept_btn.blockSignals(False)

        self.note_field.blockSignals(True)
        self.note_field.setText(comp.note)
        self.note_field.blockSignals(False)

        self.tags_field.blockSignals(True)
        self.tags_field.setText(", ".join(comp.tags))
        self.tags_field.blockSignals(False)

        self.review_btn.setText(
            "Unmark reviewed" if comp.review_state == "reviewed" else "Mark reviewed"
        )

    def _refresh_accept_button(self, accepted: bool) -> None:
        if accepted:
            self.accept_btn.setText("Accepted ✓")
            self.accept_btn.setStyleSheet(
                "QPushButton { background: #d0e8d0; font-size: 14px; }"
            )
        else:
            self.accept_btn.setText("Rejected ✗")
            self.accept_btn.setStyleSheet(
                "QPushButton { background: #f0c8c8; font-size: 14px; }"
            )

    # ---- slots -------------------------------------------------------

    def _on_accept_clicked(self) -> None:
        if self._k is None:
            return
        new = self.accept_btn.isChecked()
        self._refresh_accept_button(new)
        self.acceptToggled.emit(self._k, bool(new))

    def _on_review_clicked(self) -> None:
        if self._k is None:
            return
        is_reviewed = self.review_btn.text().startswith("Unmark")
        self.markReviewed.emit(self._k, not is_reviewed)
        self.review_btn.setText(
            "Mark reviewed" if is_reviewed else "Unmark reviewed"
        )

    def _on_split_clicked(self) -> None:
        if self._k is None:
            return
        note, ok = QInputDialog.getText(
            self, "Flag for split", f"Note for splitting component k={self._k}:"
        )
        if ok:
            self.splitRequested.emit(self._k, note)

    def _emit_note(self) -> None:
        if self._k is None:
            return
        self.noteEdited.emit(self._k, self.note_field.text())

    def _emit_tags(self) -> None:
        if self._k is None:
            return
        tags = [t.strip() for t in self.tags_field.text().split(",") if t.strip()]
        self.tagsEdited.emit(self._k, tags)

    def keyPressEvent(self, event):
        # Toggle accept with 'A' even when this widget has focus.
        if event.key() in (Qt.Key.Key_A,) and self._k is not None:
            self.accept_btn.click()
            return
        super().keyPressEvent(event)
