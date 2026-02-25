import numpy as np
import quaternion

from simulation.interface import (
    ACCEL_X,
    ACCEL_Z,
    GYRO_X,
    GYRO_Z,
    POS_X,
    POS_Z,
    Q_W,
    Q_Z,
    VEL_X,
    VEL_Z,
    SensorRow,
    State,
    StateEstimator,
    make_initial_state,
)

_GRAVITY_WORLD = np.array([0.0, 0.0, -9.80665])


class EulerStateEstimator(StateEstimator):
    def __init__(self) -> None:
        self._state = make_initial_state()

    def step(self, dt: float, sensor_data: SensorRow) -> State:
        s = self._state
        pos = s[POS_X : POS_Z + 1]
        vel = s[VEL_X : VEL_Z + 1]
        q = s[Q_W : Q_Z + 1]

        accel = sensor_data[ACCEL_X : ACCEL_Z + 1]
        gyro = sensor_data[GYRO_X : GYRO_Z + 1]

        pos += vel * dt
        q_cur = quaternion.quaternion(q[0], q[1], q[2], q[3])

        # Accelerometer measures specific force (includes gravity reaction).
        # True inertial acceleration = R(q) * f_body + g_world
        accel_world = quaternion.rotate_vectors(q_cur, accel) + _GRAVITY_WORLD
        vel += accel_world * dt

        omega_q = quaternion.quaternion(0, gyro[0], gyro[1], gyro[2])
        q_new = (q_cur + 0.5 * dt * (q_cur * omega_q)).normalized()
        s[Q_W : Q_Z + 1] = quaternion.as_float_array(q_new)

        return s

    @property
    def state(self) -> State:
        return self._state
