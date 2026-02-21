#!/usr/bin/env python

import polars as pl
from PyQt6.QtWidgets import QApplication, QDialog, QVBoxLayout

from components.charts import Charts
from components.file_selector import FileSelector
from components.mission_stats import MissionStats
from components.playback_controls import PlaybackControls
from data_loader import DataLoader, CSVDataLoader


class MainWindow(QDialog):
    def __init__(self, data_loader: DataLoader, parent=None):
        super(MainWindow, self).__init__(parent)
        self._data_loader = data_loader
        self._data: pl.DataFrame | None = None

        mainLayout = QVBoxLayout()

        self.file_selector = FileSelector(
            file_filter="CSV Files (*.csv);;All Files (*)"
        )
        self.file_selector.file_selected.connect(self._on_file_selected)
        mainLayout.addWidget(self.file_selector)

        self._stats = MissionStats()
        mainLayout.addWidget(self._stats)

        self.charts = Charts()
        mainLayout.addWidget(self.charts)

        self._playback = PlaybackControls()
        self._playback.timestamp_changed.connect(self.charts.set_timestamp)
        mainLayout.addWidget(self._playback)

        self.setLayout(mainLayout)
        self.setWindowTitle("Invictus Mission Analyzer")
        self.resize(1000, 800)

    def _on_file_selected(self, filepath: str) -> None:
        self._data = self._data_loader.load_data(filepath)
        self.charts.load(self._data)

        ts = self._data["timestamp"]
        t_min = float(ts.min())  # type: ignore[arg-type]
        t_max = float(ts.max())  # type: ignore[arg-type]
        self._playback.set_range(t_min, t_max)
        self.file_selector.set_summary(t_max - t_min, len(self._data))
        self._stats.load(self._data)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    main_window = MainWindow(data_loader=CSVDataLoader())
    main_window.show()
    sys.exit(app.exec())
