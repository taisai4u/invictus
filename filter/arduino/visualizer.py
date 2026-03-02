from __future__ import annotations

import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from qtpy.QtWidgets import QApplication
from scipy.spatial.transform import Rotation

_AXIS_SCREEN_FRACTION = 0.1
_TRAIL_COLOR = "#ffcc33"
_ELLIPSOID_COLOR = "cyan"
_ELLIPSOID_OPACITY = 0.15
_BG_COLOR = "#1a1a2e"
_MAX_TRAIL_POINTS = 10000
_MIN_PARALLEL_SCALE = 0.5  # minimum half-height of viewport in meters (1m x 1m view)


class LiveVisualizer:
    def __init__(self) -> None:
        self._plotter = BackgroundPlotter(title="Live Filter Estimate")
        self._plotter.set_background(_BG_COLOR)
        self._plotter.add_axes()
        self._plotter.show_grid(
            xtitle="X (m)", ytitle="Y (m)", ztitle="Z (m)",
            font_size=8, color="gray",
        )
        self._plotter.enable_parallel_projection()
        self._plotter.view_isometric()

        self._axis_actors = self._build_axes()
        self._trail_points: list[np.ndarray] = []
        self._trail_mesh = pv.PolyData()
        self._trail_actor = None

        self._ellipsoid_source = pv.Sphere(
            radius=1.0, theta_resolution=16, phi_resolution=16
        )
        self._ellipsoid_actor = self._plotter.add_mesh(
            self._ellipsoid_source,
            color=_ELLIPSOID_COLOR,
            opacity=_ELLIPSOID_OPACITY,
            style="wireframe",
        )

    def _build_axes(self) -> list:
        actors = []
        for direction, color in (
            ((1, 0, 0), "red"),
            ((0, 1, 0), "green"),
            ((0, 0, 1), "blue"),
        ):
            actor = self._plotter.add_mesh(
                pv.Arrow(start=(0, 0, 0), direction=direction),
                color=color,
            )
            actors.append(actor)
        return actors

    def update(self, x_nom: np.ndarray, P: np.ndarray) -> None:
        pos = x_nom[0:3]
        rot = Rotation.from_quat(x_nom[6:10], scalar_first=True).as_matrix()

        self._update_trail(pos)
        self._update_ellipsoid(pos, P[0:3, 0:3])

        for actor in self._axis_actors:
            actor.SetVisibility(False)
        self._ellipsoid_actor.SetVisibility(False)
        self._plotter.reset_camera()
        self._plotter.camera.parallel_scale = max(
            self._plotter.camera.parallel_scale, _MIN_PARALLEL_SCALE
        )
        self._ellipsoid_actor.SetVisibility(True)
        for actor in self._axis_actors:
            actor.SetVisibility(True)

        scale = self._plotter.camera.parallel_scale * _AXIS_SCREEN_FRACTION
        xform = np.eye(4)
        xform[:3, :3] = rot * scale
        xform[:3, 3] = pos

        for actor in self._axis_actors:
            actor.user_matrix = xform

        self._plotter.render()

    def _update_trail(self, pos: np.ndarray) -> None:
        self._trail_points.append(pos.copy())
        if len(self._trail_points) > _MAX_TRAIL_POINTS:
            self._trail_points = self._trail_points[-_MAX_TRAIL_POINTS:]

        n = len(self._trail_points)
        if n < 2:
            return

        points = np.array(self._trail_points)
        self._trail_mesh.points = points
        self._trail_mesh.lines = np.hstack([[n], np.arange(n)])

        if self._trail_actor is None:
            self._trail_actor = self._plotter.add_mesh(
                self._trail_mesh, color=_TRAIL_COLOR, line_width=2
            )

    def _update_ellipsoid(self, pos: np.ndarray, P_pos: np.ndarray) -> None:
        eigvals, eigvecs = np.linalg.eigh(P_pos)
        radii = 2 * np.sqrt(np.maximum(eigvals, 0))

        xform = np.eye(4)
        xform[:3, :3] = eigvecs @ np.diag(radii)
        xform[:3, 3] = pos
        self._ellipsoid_actor.user_matrix = xform

    def process_events(self) -> None:
        app = QApplication.instance()
        if app:
            app.processEvents()

    def close(self) -> None:
        self._plotter.close()
