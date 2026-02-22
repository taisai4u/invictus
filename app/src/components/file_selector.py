from pathlib import Path

from PyQt5.QtCore import QSettings, pyqtSignal
from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)

_SETTINGS_KEY = "file_selector/last_dir"


class FileSelector(QWidget):
    file_selected = pyqtSignal(str)

    def __init__(self, file_filter="All Files (*)", parent=None):
        super().__init__(parent)
        self._file_filter = file_filter
        self._settings = QSettings("Invictus", "MissionAnalyzer")

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self._path_edit = QLabel("")

        browse_button = QPushButton("Load CSV")
        browse_button.clicked.connect(self._browse)

        layout.addWidget(self._path_edit, stretch=1)
        layout.addWidget(browse_button)

        self.setLayout(layout)

    def _browse(self):
        last_dir = self._settings.value(_SETTINGS_KEY, "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Select File", last_dir, self._file_filter
        )
        if path:
            self._path_edit.setText(Path(path).name)
            self._settings.setValue(_SETTINGS_KEY, str(Path(path).parent))
            self.file_selected.emit(path)

    def set_summary(self, duration: float, samples: int) -> None:
        name = self._path_edit.text()
        self._path_edit.setText(f"{name}  —  {duration:.2f}s  —  {samples:,} samples")

    def path(self) -> str:
        return self._path_edit.text()
