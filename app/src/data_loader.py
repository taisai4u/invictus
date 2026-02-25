from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass
class FlightDataRow:
    timestamp: float
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    mag_x: float
    mag_y: float
    mag_z: float
    pressure: float
    temperature: float
    pos_x: float
    pos_y: float
    pos_z: float
    vel_x: float
    vel_y: float
    vel_z: float
    quat_w: float
    quat_x: float
    quat_y: float
    quat_z: float


SCHEMA = {
    "timestamp": pl.Float64,
    "accel_x": pl.Float64,
    "accel_y": pl.Float64,
    "accel_z": pl.Float64,
    "gyro_x": pl.Float64,
    "gyro_y": pl.Float64,
    "gyro_z": pl.Float64,
    "mag_x": pl.Float64,
    "mag_y": pl.Float64,
    "mag_z": pl.Float64,
    "pressure": pl.Float64,
    "temperature": pl.Float64,
    "pos_x": pl.Float64,
    "pos_y": pl.Float64,
    "pos_z": pl.Float64,
    "vel_x": pl.Float64,
    "vel_y": pl.Float64,
    "vel_z": pl.Float64,
    "quat_w": pl.Float64,
    "quat_x": pl.Float64,
    "quat_y": pl.Float64,
    "quat_z": pl.Float64,
}


class DataLoader(ABC):
    @abstractmethod
    def load_data(self, filepath: str | Path) -> pl.DataFrame: ...


class CSVDataLoader(DataLoader):
    def load_data(self, filepath: str | Path) -> pl.DataFrame:
        return pl.read_csv(
            filepath,
            has_header=True,
            schema_overrides=SCHEMA,
            ignore_errors=True,
        )
