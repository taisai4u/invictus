from PyQt6.QtCore import QElapsedTimer, QTimer, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)

_SLIDER_MAX = 10000


class PlaybackControls(QWidget):
    timestamp_changed = pyqtSignal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._t_min: float = 0.0
        self._t_max: float = 1.0

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self._play_button = QPushButton("Play")
        self._play_button.setEnabled(False)
        self._play_button.clicked.connect(self._toggle_play)
        layout.addWidget(self._play_button)

        self._time_label = QLabel("0:00 / 0:00")
        layout.addWidget(self._time_label)

        self._scrubber = QSlider(Qt.Orientation.Horizontal)
        self._scrubber.setRange(0, _SLIDER_MAX)
        self._scrubber.setValue(0)
        self._scrubber.setEnabled(False)
        self._scrubber.valueChanged.connect(self._on_scrub)
        layout.addWidget(self._scrubber)

        self.setLayout(layout)

        self._timer = QTimer()
        self._timer.setInterval(16)  # ~60 fps
        self._timer.timeout.connect(self._tick)

        self._clock = QElapsedTimer()
        self._t_at_play: float = 0.0

    def set_range(self, t_min: float, t_max: float) -> None:
        self._stop()
        self._t_min = t_min
        self._t_max = t_max
        self._scrubber.setValue(0)
        self._scrubber.setEnabled(True)
        self._play_button.setEnabled(True)
        self._on_scrub(0)

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        return f"{m}:{s:02d}"

    def _on_scrub(self, value: int) -> None:
        t = self._t_min + (value / _SLIDER_MAX) * (self._t_max - self._t_min)
        current = t - self._t_min
        total = self._t_max - self._t_min
        self._time_label.setText(f"{self._fmt_time(current)} / {self._fmt_time(total)}")
        self.timestamp_changed.emit(t)

    def _toggle_play(self) -> None:
        if self._timer.isActive():
            self._stop()
        else:
            if self._scrubber.value() >= _SLIDER_MAX:
                self._scrubber.setValue(0)
            self._t_at_play = self._t_min + (self._scrubber.value() / _SLIDER_MAX) * (self._t_max - self._t_min)
            self._clock.start()
            self._play_button.setText("Pause")
            self._timer.start()

    def _tick(self) -> None:
        elapsed_s = self._clock.elapsed() / 1000.0
        t = self._t_at_play + elapsed_s
        frac = (t - self._t_min) / (self._t_max - self._t_min)
        if frac >= 1.0:
            self._scrubber.setValue(_SLIDER_MAX)
            self._stop()
        else:
            self._scrubber.setValue(round(frac * _SLIDER_MAX))

    def _stop(self) -> None:
        self._timer.stop()
        self._play_button.setText("Play")
