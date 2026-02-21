from typing import Sequence

import polars as pl
import pyqtgraph as pg
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

        self._plot_widget = pg.PlotWidget(title=title)
        self._plot_widget.setLabel("bottom", time_col)
        self._plot_widget.addLegend()
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._plot_widget.setMouseEnabled(x=True, y=True)

        layout.addWidget(self._plot_widget)
        self.setLayout(layout)

        self._curves: dict[str, pg.PlotDataItem] = {}
        for i, col in enumerate(self._value_cols):
            color = COLORS[i % len(COLORS)]
            self._curves[col] = self._plot_widget.plot(
                name=col,
                pen=pg.mkPen(color=color, width=1.5),
            )

    def load(self, data: pl.DataFrame) -> None:
        if self._time_col not in data.columns:
            raise ValueError(f"Time column '{self._time_col}' not found in data")

        x = data[self._time_col].to_numpy()
        for col, curve in self._curves.items():
            if col not in data.columns:
                raise ValueError(f"Value column '{col}' not found in data")
            curve.setData(x=x, y=data[col].to_numpy())

    def clear(self) -> None:
        for curve in self._curves.values():
            curve.setData(x=[], y=[])
