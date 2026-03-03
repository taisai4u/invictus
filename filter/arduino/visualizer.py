from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from qtpy.QtWidgets import QApplication
from scipy.spatial.transform import Rotation
from scipy.stats import chi2

_AXIS_SCREEN_FRACTION = 0.1
_TRAIL_COLOR = "#ffcc33"
_ELLIPSOID_COLOR = "cyan"
_ELLIPSOID_OPACITY = 0.15
_MAG_COLOR = "magenta"
_ACCEL_COLOR = "yellow"
_BG_COLOR = "#1a1a2e"
_MAX_TRAIL_POINTS = 10000
_MIN_PARALLEL_SCALE = 0.5  # minimum half-height of viewport in meters (1m x 1m view)


class NISTracker:
    def __init__(self, nz: int, alpha: float = 0.05):
        self.nz = nz
        self.lower_bound: float = chi2.ppf(alpha / 2, nz)
        self.upper_bound: float = chi2.ppf(1 - alpha / 2, nz)
        self.total_count: int = 0
        self.inconsistent_count: int = 0
        self.nis_sum: float = 0.0

    def record(self, nis: float):
        self.total_count += 1
        self.nis_sum += nis
        if not (self.lower_bound <= nis <= self.upper_bound):
            self.inconsistent_count += 1

    @property
    def mean_nis(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.nis_sum / self.total_count

    @property
    def inconsistency_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.inconsistent_count / self.total_count


class LiveVisualizer:
    def __init__(self) -> None:
        self._plotter = BackgroundPlotter(title="Live Filter Estimate")
        self._plotter.set_background(_BG_COLOR)
        self._plotter.add_axes()
        self._plotter.show_grid(
            xtitle="X (m)",
            ytitle="Y (m)",
            ztitle="Z (m)",
            font_size=8,
            color="gray",
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

        self._mag_actor = self._plotter.add_mesh(
            pv.Arrow(start=(0, 0, 0), direction=(1, 0, 0)),
            color=_MAG_COLOR,
        )
        self._accel_actor = self._plotter.add_mesh(
            pv.Arrow(start=(0, 0, 0), direction=(1, 0, 0)),
            color=_ACCEL_COLOR,
        )
        self._mag_direction = np.array([1.0, 0.0, 0.0])
        self._accel_direction = np.array([0.0, 0.0, -1.0])

        self._nis_text_actor = None
        self._bias_text_actor = None

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

    def update_nis_overlay(self, nis_trackers: dict[str, NISTracker]) -> None:
        lines = []
        for name, tracker in nis_trackers.items():
            if tracker.total_count == 0:
                continue
            lines.append(
                f"{name}: NIS={tracker.mean_nis:.2f} "
                f"[{tracker.lower_bound:.2f}, {tracker.upper_bound:.2f}]  "
                f"incon={tracker.inconsistency_rate:.1%} ({tracker.total_count})"
            )
        text = "\n".join(lines)
        if self._nis_text_actor is None:
            self._nis_text_actor = self._plotter.add_text(
                text,
                position="upper_left",
                font_size=8,
                color="white",
                name="nis_overlay",
            )
        else:
            self._nis_text_actor.SetText(2, text)

    def update_bias_overlay(self, accel_bias: np.ndarray, gyro_bias: np.ndarray) -> None:
        text = (
            f"Accel bias: [{accel_bias[0]:+.4f}, {accel_bias[1]:+.4f}, {accel_bias[2]:+.4f}] m/s²\n"
            f"Gyro bias:  [{gyro_bias[0]:+.5f}, {gyro_bias[1]:+.5f}, {gyro_bias[2]:+.5f}] rad/s"
        )
        if self._bias_text_actor is None:
            self._bias_text_actor = self._plotter.add_text(
                text,
                position="upper_right",
                font_size=8,
                color="white",
                name="bias_overlay",
            )
        else:
            self._bias_text_actor.SetText(3, text)

    def update_measurements(self, accel: np.ndarray, mag: np.ndarray) -> None:
        accel_norm = np.linalg.norm(accel)
        mag_norm = np.linalg.norm(mag)
        if accel_norm > 0:
            self._accel_direction = accel / accel_norm
        if mag_norm > 0:
            self._mag_direction = mag / mag_norm

    def update(self, x_nom: np.ndarray, P: np.ndarray) -> None:
        pos = x_nom[0:3]
        rot = Rotation.from_quat(x_nom[6:10], scalar_first=True).as_matrix()

        self._update_trail(pos)
        self._update_ellipsoid(pos, P[0:3, 0:3])

        hidden_actors = [*self._axis_actors, self._ellipsoid_actor, self._mag_actor, self._accel_actor]
        for actor in hidden_actors:
            actor.SetVisibility(False)
        self._plotter.reset_camera()
        self._plotter.camera.parallel_scale = max(
            self._plotter.camera.parallel_scale, _MIN_PARALLEL_SCALE
        )
        for actor in hidden_actors:
            actor.SetVisibility(True)

        scale = self._plotter.camera.parallel_scale * _AXIS_SCREEN_FRACTION
        xform = np.eye(4)
        xform[:3, :3] = rot * scale
        xform[:3, 3] = pos

        for actor in self._axis_actors:
            actor.user_matrix = xform

        mag_xform = np.eye(4)
        mag_xform[:3, :3] = self._rotation_from_direction(self._mag_direction) * scale
        mag_xform[:3, 3] = pos
        self._mag_actor.user_matrix = mag_xform

        accel_xform = np.eye(4)
        accel_xform[:3, :3] = self._rotation_from_direction(self._accel_direction) * scale
        accel_xform[:3, 3] = pos
        self._accel_actor.user_matrix = accel_xform

        self._plotter.render()

    @staticmethod
    def _rotation_from_direction(direction: np.ndarray) -> np.ndarray:
        """Return a 3x3 rotation matrix that maps [1,0,0] to the given direction."""
        d = direction / np.linalg.norm(direction)
        source = np.array([1.0, 0.0, 0.0])
        cross = np.cross(source, d)
        sin_angle = np.linalg.norm(cross)
        cos_angle = np.dot(source, d)
        if sin_angle < 1e-8:
            return np.eye(3) if cos_angle > 0 else np.diag([-1.0, -1.0, 1.0])
        K = np.array([
            [0, -cross[2], cross[1]],
            [cross[2], 0, -cross[0]],
            [-cross[1], cross[0], 0],
        ]) / sin_angle
        return np.eye(3) + sin_angle * K + (1 - cos_angle) * K @ K

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
