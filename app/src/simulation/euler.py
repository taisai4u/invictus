from simulation.interface import (
    ACCEL_X,
    ACCEL_Z,
    GYRO_X,
    GYRO_Z,
    MAG_X,
    MAG_Z,
    POS_X,
    POS_Z,
    PRESSURE,
    Q_W,
    Q_Z,
    TEMPERATURE,
    VEL_X,
    VEL_Z,
    SensorRow,
    State,
    StateEstimator,
)
import numpy as np
import quaternion

_GRAVITY_WORLD = np.array([0.0, 0.0, -9.80665])


class EulerStateEstimator(StateEstimator):
    def step(self, dt: float, state: State, sensor_data: SensorRow) -> State:
        pos = state[POS_X : POS_Z + 1]
        vel = state[VEL_X : VEL_Z + 1]
        q = state[Q_W : Q_Z + 1]

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
        state[Q_W:Q_Z + 1] = quaternion.as_float_array(q_new)

        return state
