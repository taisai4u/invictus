from simulation.interface import (
    GT_POS_X,
    GT_POS_Z,
    GT_Q_W,
    GT_Q_Z,
    GT_VEL_X,
    GT_VEL_Z,
    POS_X,
    Q_W,
    SensorRow,
    State,
    StateEstimator,
    VEL_X,
    make_initial_state,
)


class GroundTruthEstimator(StateEstimator):
    """State estimator that reads ground truth values directly from the sensor row."""

    def __init__(self) -> None:
        self._state = make_initial_state()

    def step(self, dt: float, sensor_row: SensorRow) -> State:
        self._state[POS_X : POS_X + 3] = sensor_row[GT_POS_X : GT_POS_Z + 1]
        self._state[VEL_X : VEL_X + 3] = sensor_row[GT_VEL_X : GT_VEL_Z + 1]
        self._state[Q_W : Q_W + 4] = sensor_row[GT_Q_W : GT_Q_Z + 1]
        return self._state

    @property
    def state(self) -> State:
        return self._state
