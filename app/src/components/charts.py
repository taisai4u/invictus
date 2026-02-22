import polars as pl
from PyQt5.QtWidgets import QScrollArea, QVBoxLayout, QWidget

from components.line_chart import LineChart


class Charts(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._accel_chart = LineChart(
            time_col="timestamp",
            value_cols=["accel_x", "accel_y", "accel_z"],
            title="Acceleration",
        )
        self._gyro_chart = LineChart(
            time_col="timestamp",
            value_cols=["gyro_x", "gyro_y", "gyro_z"],
            title="Gyroscope",
        )
        self._mag_chart = LineChart(
            time_col="timestamp",
            value_cols=["mag_x", "mag_y", "mag_z"],
            title="Magnetometer",
        )
        self._pressure_chart = LineChart(
            time_col="timestamp",
            value_cols=["pressure"],
            title="Pressure",
        )
        self._temperature_chart = LineChart(
            time_col="timestamp",
            value_cols=["temperature"],
            title="Temperature",
        )

        inner = QWidget()
        inner_layout = QVBoxLayout()
        inner_layout.setSpacing(8)
        for chart in self._all_charts():
            chart.setMinimumHeight(220)
            inner_layout.addWidget(chart)
        inner.setLayout(inner_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)
        self.setLayout(layout)

    def _all_charts(self) -> list[LineChart]:
        return [
            self._accel_chart,
            self._gyro_chart,
            self._mag_chart,
            self._pressure_chart,
            self._temperature_chart,
        ]

    def load(self, data: pl.DataFrame) -> None:
        for chart in self._all_charts():
            chart.load(data)

    def set_timestamp(self, t: float) -> None:
        for chart in self._all_charts():
            chart.set_timestamp(t)

    def clear(self) -> None:
        for chart in self._all_charts():
            chart.clear()
