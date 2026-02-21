import math

import numpy as np
import polars as pl
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

G = 9.80665
PA_PER_M = 12.0  # rough sea-level approximation: ~12 Pa per metre
M_TO_FT = 3.28084
SEA_LEVEL_PA = 101325.0


def _pressure_to_altitude(pressure_pa: float) -> float:
    return (SEA_LEVEL_PA - pressure_pa) / PA_PER_M


class MissionStats(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._duration_label = QLabel()
        self._sample_rate_label = QLabel()
        self._max_alt_label = QLabel()
        self._max_accel_label = QLabel()

        for label in [
            self._duration_label,
            self._sample_rate_label,
            self._max_alt_label,
            self._max_accel_label,
        ]:
            layout.addWidget(label)

        self.setLayout(layout)
        self._clear()

    def load(self, data: pl.DataFrame) -> None:
        ts = data["timestamp"]
        t_min = float(ts.min())  # type: ignore[arg-type]
        t_max = float(ts.max())  # type: ignore[arg-type]
        duration = t_max - t_min
        n = len(data)

        sample_rate = (n - 1) / duration if duration > 0 else 0.0

        min_pressure = float(data["pressure"].min())  # type: ignore[arg-type]
        max_alt_m = _pressure_to_altitude(min_pressure)
        max_alt_ft = max_alt_m * M_TO_FT

        ax = data["accel_x"].to_numpy()
        ay = data["accel_y"].to_numpy()
        az = data["accel_z"].to_numpy()
        accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
        max_accel = float(np.nanmax(accel_mag))
        max_accel_g = max_accel / G

        self._duration_label.setText(f"Duration {duration:.2f} s")
        self._sample_rate_label.setText(f"Sample Rate {sample_rate:.0f} Hz")
        self._max_alt_label.setText(
            f"Max Altitude {max_alt_m:.1f} m ({max_alt_ft:.0f} ft)"
        )
        self._max_accel_label.setText(
            f"Max Acceleration {max_accel:.1f} m/s² ({max_accel_g:.1f} G)"
        )

    def _clear(self) -> None:
        self._duration_label.setText("Duration — s")
        self._sample_rate_label.setText("Sample Rate — Hz")
        self._max_alt_label.setText("Max Altitude — m (— ft)")
        self._max_accel_label.setText("Max Acceleration — m/s² (— G)")
