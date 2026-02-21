from typing import Sequence

import polars as pl
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QVBoxLayout, QWidget

COLORS = [
    (255, 100, 100),  # red
    (100, 200, 255),  # blue
    (100, 255, 150),  # green
    (255, 200, 80),   # yellow
    (200, 100, 255),  # purple
    (255, 160, 80),   # orange
]


class LineChart(QWidget):
    def __init__(
        self,
        time_col: str,
        value_cols: Sequence[str],
        title: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._time_col = time_col
        self._value_cols = list(value_cols)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        pg.setConfigOptions(antialias=False)

        self._plot_widget = pg.PlotWidget(title=title)
        self._plot_widget.setLabel("bottom", time_col)
        self._plot_widget.addLegend()
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._plot_widget.setMouseEnabled(x=True, y=True)
        self._plot_widget.setClipToView(True)
        self._plot_widget.setDownsampling(auto=True, mode="peak")

        layout.addWidget(self._plot_widget)
        self.setLayout(layout)

        self._cursor = pg.InfiniteLine(
            angle=90, pen=pg.mkPen("w", width=1, style=Qt.PenStyle.DashLine)
        )
        self._cursor.setVisible(False)
        self._plot_widget.addItem(self._cursor)

        self._curves: dict[str, pg.PlotDataItem] = {}
        for i, col in enumerate(self._value_cols):
            color = COLORS[i % len(COLORS)]
            curve = self._plot_widget.plot(
                name=col,
                pen=pg.mkPen(color=color, width=1),
            )
            curve.setSkipFiniteCheck(True)
            self._curves[col] = curve

    def load(self, data: pl.DataFrame) -> None:
        if self._time_col not in data.columns:
            raise ValueError(f"Time column '{self._time_col}' not found in data")

        x = data[self._time_col].to_numpy()
        for col, curve in self._curves.items():
            if col not in data.columns:
                raise ValueError(f"Value column '{col}' not found in data")
            curve.setData(x=x, y=data[col].to_numpy())

    def set_timestamp(self, t: float) -> None:
        self._cursor.setValue(t)
        self._cursor.setVisible(True)

    def clear(self) -> None:
        self._cursor.setVisible(False)
        for curve in self._curves.values():
            curve.setData(x=[], y=[])
