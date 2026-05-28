"""Movie transport bar: slider + prev/next + play/pause + frame label."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QWidget,
)


class TransportBar(QWidget):
    """Emits ``frameChanged(t_native: int)`` whenever the user moves time."""

    frameChanged = pyqtSignal(int)

    def __init__(self, n_frames: int, parent: QWidget | None = None):
        super().__init__(parent)
        self.n_frames = int(n_frames)
        self._t = 0
        self._fps_target = 20  # play-back fps

        self.prev_btn = QPushButton("◀")
        self.play_btn = QPushButton("▶")
        self.next_btn = QPushButton("▶|")
        for b in (self.prev_btn, self.play_btn, self.next_btn):
            b.setFixedWidth(36)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, max(0, self.n_frames - 1))
        self.slider.setSingleStep(1)
        self.slider.setPageStep(max(1, self.n_frames // 50))

        self.spin = QSpinBox()
        self.spin.setRange(0, max(0, self.n_frames - 1))
        self.spin.setFixedWidth(80)

        self.label = QLabel(self._format_label(0))
        self.label.setFixedWidth(180)

        self._timer = QTimer(self)
        self._timer.setInterval(int(1000 / self._fps_target))
        self._timer.timeout.connect(self._on_tick)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        for w in (
            self.prev_btn,
            self.play_btn,
            self.next_btn,
            self.slider,
            self.spin,
            self.label,
        ):
            layout.addWidget(w)
        layout.setStretchFactor(self.slider, 1)

        # Wiring
        self.prev_btn.clicked.connect(lambda: self.set_frame(self._t - 1))
        self.next_btn.clicked.connect(lambda: self.set_frame(self._t + 1))
        self.play_btn.clicked.connect(self.toggle_play)
        self.slider.valueChanged.connect(self._on_slider)
        self.spin.valueChanged.connect(self._on_spin)

    # ------------------------------------------------------------------

    def _format_label(self, t: int, t_e: int | None = None) -> str:
        if t_e is None:
            return f"frame {t} / {self.n_frames - 1}"
        return f"frame {t} (t_e={t_e}) / {self.n_frames - 1}"

    def set_frame(self, t: int, *, emit: bool = True) -> None:
        t = max(0, min(self.n_frames - 1, int(t)))
        if t == self._t:
            return
        self._t = t
        # Block signals so we don't re-emit through slider/spin.
        for w in (self.slider, self.spin):
            w.blockSignals(True)
        self.slider.setValue(t)
        self.spin.setValue(t)
        for w in (self.slider, self.spin):
            w.blockSignals(False)
        self.label.setText(self._format_label(t))
        if emit:
            self.frameChanged.emit(t)

    def current_frame(self) -> int:
        return self._t

    def set_extraction_label(self, t_native: int, t_e: int | None) -> None:
        self.label.setText(self._format_label(t_native, t_e))

    # ---- play/pause --------------------------------------------------

    def toggle_play(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self.play_btn.setText("▶")
        else:
            self._timer.start()
            self.play_btn.setText("⏸")

    def _on_tick(self) -> None:
        nxt = self._t + 1
        if nxt >= self.n_frames:
            nxt = 0
        self.set_frame(nxt)

    # ---- slot wiring -------------------------------------------------

    def _on_slider(self, v: int) -> None:
        self.set_frame(int(v))

    def _on_spin(self, v: int) -> None:
        self.set_frame(int(v))
