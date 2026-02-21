from pathlib import Path

from PyQt6.QtCore import QSettings, pyqtSignal
from PyQt6.QtWidgets import (
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

    def __init__(self, label="File:", file_filter="All Files (*)", parent=None):
        super().__init__(parent)
        self._file_filter = file_filter
        self._settings = QSettings("Invictus", "MissionAnalyzer")

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("No file selected")
        self._path_edit.setReadOnly(True)

        browse_button = QPushButton("Browse…")
        browse_button.clicked.connect(self._browse)

        layout.addWidget(QLabel(label))
        layout.addWidget(self._path_edit, stretch=1)
        layout.addWidget(browse_button)

        self.setLayout(layout)

    def _browse(self):
        last_dir = self._settings.value(_SETTINGS_KEY, "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Select File", last_dir, self._file_filter
        )
        if path:
            self._path_edit.setText(path)
            self._settings.setValue(_SETTINGS_KEY, str(Path(path).parent))
            self.file_selected.emit(path)

    def path(self) -> str:
        return self._path_edit.text()
