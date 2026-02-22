from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

State = NDArray[np.floating[Any]]
SensorRow = NDArray[np.floating[Any]]

TIMESTAMP = 0
ACCEL_X, ACCEL_Y, ACCEL_Z = 1, 2, 3
GYRO_X, GYRO_Y, GYRO_Z = 4, 5, 6
MAG_X, MAG_Y, MAG_Z = 7, 8, 9
PRESSURE = 10
TEMPERATURE = 11

POS_X, POS_Y, POS_Z = 0, 1, 2
VEL_X, VEL_Y, VEL_Z = 3, 4, 5
Q_W, Q_X, Q_Y, Q_Z = 6, 7, 8, 9

STATE_DIM = 10
SENSOR_DIM = 12


def make_initial_state() -> State:
    state = np.zeros(STATE_DIM, dtype=np.float64)
    state[Q_W] = 1.0  # identity quaternion
    return state


class StateEstimator(ABC):
    @abstractmethod
    def step(self, dt: float, state: State, sensor_row: SensorRow) -> State:
        """Advance one simulation step.

        Args:
            dt: Time delta in seconds since last step.
            state: Mutable state vector of shape (STATE_DIM,).
                   [x, y, z, vx, vy, vz, qw, qx, qy, qz]
            sensor_row: Sensor readings of shape (SENSOR_DIM,) — a row view
                        from the pre-converted NumPy array.

        Returns:
            The updated state vector (may be the same array mutated in-place).
        """
        ...
