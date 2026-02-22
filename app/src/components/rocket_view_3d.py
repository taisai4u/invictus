from __future__ import annotations

import numpy as np
import polars as pl
import pyvista as pv
import quaternion as quat_mod
from pyvistaqt import QtInteractor
from PyQt5.QtWidgets import QFrame, QPushButton, QVBoxLayout

from simulation.euler import EulerStateEstimator
from simulation.interface import (
    POS_X,
    POS_Z,
    Q_W,
    Q_Z,
    STATE_DIM,
    TIMESTAMP,
    make_initial_state,
)

_AXIS_LEN = 1.0  # base arrow length in world units (rescaled each frame)
_AXIS_SCREEN_FRAC = 0.05  # arrows occupy ~5% of viewport height
_ROCKET_HEIGHT = 0.3
_ROCKET_RADIUS = 0.1
_TRAIL_COLOR = "#ffcc33"
_BG_COLOR = "#1a1a2e"


class RocketView3D(QFrame):
    """3D visualisation of rocket position, orientation axes, and trajectory."""

    def __init__(self, parent: QFrame | None = None) -> None:
        super().__init__(parent)

        vlayout = QVBoxLayout()
        vlayout.setContentsMargins(0, 0, 0, 0)

        self._plotter = QtInteractor(self)  # type: ignore[arg-type]
        vlayout.addWidget(self._plotter.interactor)  # type: ignore[arg-type]

        reset_btn = QPushButton("Reset Camera")
        reset_btn.clicked.connect(self._reset_camera)
        vlayout.addWidget(reset_btn)

        self.setLayout(vlayout)

        self._timestamps: np.ndarray | None = None
        self._states: np.ndarray | None = None
        self._positions: np.ndarray | None = None
        self._trail_mesh: pv.PolyData | None = None
        self._trail_actor = None

        self._rocket_actor, self._axis_actors = self._build_scene()

    def _build_scene(self):
        p = self._plotter

        rocket_actor = p.add_mesh(
            pv.Cone(
                center=(0, 0, 0),
                direction=(0, 0, 1),
                height=_ROCKET_HEIGHT,
                radius=_ROCKET_RADIUS,
                resolution=24,
            ),
            color="lightgrey",
        )

        axis_actors = []
        for direction, color in (
            ((1, 0, 0), "red"),
            ((0, 1, 0), "green"),
            ((0, 0, 1), "blue"),
        ):
            d = tuple(c * _AXIS_LEN for c in direction)
            actor = p.add_mesh(
                pv.Arrow(
                    start=(0, 0, 0),
                    direction=d,
                    # shaft_radius=0.03,
                    # tip_radius=0.07,
                    # tip_length=0.1,
                ),
                color=color,
            )
            axis_actors.append(actor)

        p.add_axes()
        p.set_background(_BG_COLOR)
        p.enable_parallel_projection()
        p.view_isometric()

        return rocket_actor, axis_actors

    # ------------------------------------------------------------------
    # Data loading — runs the Euler estimator over the full dataset once
    # ------------------------------------------------------------------

    def load(self, data: pl.DataFrame) -> None:
        if self._trail_actor is not None:
            self._plotter.remove_actor(self._trail_actor)
            self._trail_actor = None

        sensor_data = data.to_numpy()
        estimator = EulerStateEstimator()
        state = make_initial_state()

        n = len(sensor_data)
        states = np.empty((n, STATE_DIM), dtype=np.float64)
        states[0] = state.copy()
        for i in range(1, n):
            dt = sensor_data[i, TIMESTAMP] - sensor_data[i - 1, TIMESTAMP]
            state = estimator.step(dt, state, sensor_data[i])
            states[i] = state.copy()

        self._timestamps = sensor_data[:, TIMESTAMP]
        self._states = states
        self._positions = states[:, POS_X : POS_Z + 1].copy()

        self._trail_mesh = pv.PolyData(self._positions.copy())
        self._trail_mesh.lines = np.hstack([[n], np.arange(n)])
        self._trail_actor = self._plotter.add_mesh(
            self._trail_mesh, color=_TRAIL_COLOR, line_width=2
        )

        # Fit camera to the full trajectory so the entire flight zone is visible
        all_pos = self._positions
        bounds = (
            all_pos[:, 0].min(),
            all_pos[:, 0].max(),
            all_pos[:, 1].min(),
            all_pos[:, 1].max(),
            all_pos[:, 2].min(),
            all_pos[:, 2].max(),
        )
        self._plotter.show_grid(
            bounds=bounds,
            grid="back",
            color="#444466",
            font_size=8,
        )
        self.set_timestamp(self._timestamps[0])
        self._plotter.reset_camera()
        self._plotter.view_isometric()

    # ------------------------------------------------------------------
    # Playback sync — called at up to 60 Hz by PlaybackControls
    # ------------------------------------------------------------------

    def set_timestamp(self, t: float) -> None:
        if self._timestamps is None or self._states is None:
            return

        idx = int(np.searchsorted(self._timestamps, t, side="right")) - 1
        idx = int(np.clip(idx, 0, len(self._timestamps) - 1))

        state = self._states[idx]
        pos = state[POS_X : POS_Z + 1]
        q = state[Q_W : Q_Z + 1]

        rot = quat_mod.as_rotation_matrix(quat_mod.quaternion(q[0], q[1], q[2], q[3]))

        xform = np.eye(4)
        xform[:3, :3] = rot
        xform[:3, 3] = pos

        self._rocket_actor.user_matrix = xform

        # Scale axes so they stay a constant screen size
        cam = self._plotter.renderer.GetActiveCamera()
        ps = cam.GetParallelScale()
        s = (_AXIS_SCREEN_FRAC * 2.0 * ps) / _AXIS_LEN if ps > 0 else 1.0

        xform_axes = np.eye(4)
        xform_axes[:3, :3] = rot * s
        xform_axes[:3, 3] = pos

        for actor in self._axis_actors:
            actor.user_matrix = xform_axes

        if self._trail_mesh is not None and self._positions is not None:
            pts = self._positions.copy()
            if idx < len(pts) - 1:
                pts[idx + 1 :] = pts[idx]
            self._trail_mesh.points = pts

        self._plotter.renderer.ResetCameraClippingRange()
        self._plotter.render()

    # ------------------------------------------------------------------

    def _reset_camera(self) -> None:
        self._plotter.reset_camera()
        self._plotter.view_isometric()
        self._plotter.render()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._plotter.close()
        super().closeEvent(event)
