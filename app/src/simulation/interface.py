from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

State = NDArray[np.floating[Any]]
SensorRow = NDArray[np.floating[Any]]

# Sensor-row column indices
TIMESTAMP = 0
ACCEL_X, ACCEL_Y, ACCEL_Z = 1, 2, 3
GYRO_X, GYRO_Y, GYRO_Z = 4, 5, 6
MAG_X, MAG_Y, MAG_Z = 7, 8, 9
PRESSURE = 10
TEMPERATURE = 11
GT_POS_X, GT_POS_Y, GT_POS_Z = 12, 13, 14
GT_VEL_X, GT_VEL_Y, GT_VEL_Z = 15, 16, 17
GT_Q_W, GT_Q_X, GT_Q_Y, GT_Q_Z = 18, 19, 20, 21

SENSOR_DIM = 22

# State-vector layout:
# [x, y, z, vx, vy, vz, qw, qx, qy, qz, bg_x, bg_y, bg_z, ba_x, ba_y, ba_z]
POS_X, POS_Y, POS_Z = 0, 1, 2
VEL_X, VEL_Y, VEL_Z = 3, 4, 5
Q_W, Q_X, Q_Y, Q_Z = 6, 7, 8, 9
BG_X, BG_Y, BG_Z = 10, 11, 12
BA_X, BA_Y, BA_Z = 13, 14, 15

STATE_DIM = 16


def make_initial_state() -> State:
    state = np.zeros(STATE_DIM, dtype=np.float64)
    state[Q_W] = 1.0  # identity quaternion
    return state


class StateEstimator(ABC):
    @abstractmethod
    def step(self, dt: float, sensor_row: SensorRow) -> State:
        """Advance one step and return the current state estimate."""
        ...

    @property
    @abstractmethod
    def state(self) -> State:
        """Current state estimate of shape (STATE_DIM,)."""
        ...
