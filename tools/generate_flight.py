#!/usr/bin/env python
"""
Simulate a model-rocket flight and generate synthetic IMU sensor data.

Physics model:
  - 6DOF rigid body: position, velocity, orientation (quaternion), angular velocity
  - Constant-thrust motor with finite burn time
  - Aerodynamic drag proportional to v²
  - Weathervaning torque (restoring torque that aligns body axis with velocity)
  - Parachute deployment at apogee (high-drag descent)
  - Earth-frame magnetic field vector (constant)
  - Barometric pressure model (ISA atmosphere)

Sensor models (body-frame, with configurable Gaussian noise):
  - Accelerometer: specific force  f = a_true − g  (in body frame)
  - Gyroscope: angular velocity (in body frame)
  - Magnetometer: Earth B-field rotated into body frame
  - Barometer: pressure from altitude (ISA model)
  - Thermometer: constant + noise

Output: CSV with the same schema as the app's data loader.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Quaternion helpers (w, x, y, z convention — no external deps)
# ---------------------------------------------------------------------------

def quat_mult(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Return 3×3 rotation matrix R such that v_world = R @ v_body."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q (body→world)."""
    return quat_to_rotmat(q) @ v


# ---------------------------------------------------------------------------
# Atmosphere / environment
# ---------------------------------------------------------------------------

G = 9.80665
SEA_LEVEL_PRESSURE = 101325.0
SEA_LEVEL_TEMP = 288.15       # K
LAPSE_RATE = 0.0065           # K/m

# Earth magnetic field (approximate mid-latitude, NED → converted to ENU/XYZ)
MAG_FIELD_WORLD = np.array([20.0, -3.5, 45.0])  # microtesla, in world frame (X=N, Y=E, Z=Up-ish)


def pressure_from_altitude(alt_m: float) -> float:
    """ISA barometric formula."""
    if alt_m < 0:
        alt_m = 0.0
    return SEA_LEVEL_PRESSURE * (1 - LAPSE_RATE * alt_m / SEA_LEVEL_TEMP) ** (G / (LAPSE_RATE * 287.05))


# ---------------------------------------------------------------------------
# Rocket parameters
# ---------------------------------------------------------------------------

@dataclass
class RocketParams:
    mass_dry: float = 0.8             # kg (without motor propellant)
    mass_propellant: float = 0.05     # kg
    thrust: float = 20.0              # N
    burn_time: float = 2.0            # s
    cd_body: float = 0.5              # drag coefficient (powered / coast)
    cd_chute: float = 1.5             # drag coefficient (under parachute)
    ref_area: float = 0.001          # m² (body cross-section)
    chute_area: float = 0.07          # m² (parachute)
    length: float = 0.4               # m (for torque arm)
    moment_of_inertia: float = 0.01   # kg·m² (about transverse axis)
    weathervane_coeff: float = 0.3    # N·m per rad of angle-of-attack per (m/s)²
    initial_tilt_deg: float = 2.0     # small initial tilt from vertical


# ---------------------------------------------------------------------------
# Noise parameters
# ---------------------------------------------------------------------------

@dataclass
class SensorNoise:
    accel_sigma: float = 0.05         # m/s²
    gyro_sigma: float = 0.003         # rad/s
    mag_sigma: float = 0.5            # microtesla
    pressure_sigma: float = 5.0       # Pa
    temp_sigma: float = 0.1           # °C
    gyro_bias: np.ndarray = field(default_factory=lambda: np.array([0.001, -0.0005, 0.0008]))


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(
    params: RocketParams,
    noise: SensorNoise,
    dt: float = 0.001,
    duration: float = 30.0,
    seed: int = 42,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    n_steps = int(duration / dt)

    # State vectors
    pos = np.array([0.0, 0.0, 0.0])
    vel = np.array([0.0, 0.0, 0.0])

    tilt_rad = math.radians(params.initial_tilt_deg)
    q = quat_normalize(np.array([
        math.cos(tilt_rad / 2),
        0.0,
        math.sin(tilt_rad / 2),
        0.0,
    ]))
    omega = np.array([0.0, 0.0, 0.0])  # angular velocity in body frame

    mass_rate = params.mass_propellant / params.burn_time if params.burn_time > 0 else 0.0

    rows: list[dict[str, float]] = []
    chute_deployed = False
    landed = False
    apogee_reached = False

    for i in range(n_steps + 1):
        t = i * dt
        R = quat_to_rotmat(q)
        R_inv = R.T  # world→body

        # --- Flight phase ---
        burning = t < params.burn_time
        mass = params.mass_dry + params.mass_propellant - mass_rate * min(t, params.burn_time)
        speed = np.linalg.norm(vel)

        if not apogee_reached and not burning and vel[2] < 0 and pos[2] > 1.0:
            apogee_reached = True
            chute_deployed = True

        if pos[2] <= 0.0 and t > params.burn_time and vel[2] <= 0:
            landed = True
            pos[2] = 0.0
            vel[:] = 0.0
            omega[:] = 0.0

        # --- Forces (world frame) ---
        gravity_world = np.array([0.0, 0.0, -G * mass])

        # Thrust along body +Z
        if burning:
            thrust_body = np.array([0.0, 0.0, params.thrust])
            thrust_world = R @ thrust_body
        else:
            thrust_world = np.zeros(3)

        # Aerodynamic drag (opposes velocity)
        if speed > 0.01 and not landed:
            rho = 1.225
            if chute_deployed:
                drag_area = params.cd_chute * params.chute_area
            else:
                drag_area = params.cd_body * params.ref_area
            drag_mag = 0.5 * rho * speed**2 * drag_area
            drag_world = -drag_mag * (vel / speed)
        else:
            drag_world = np.zeros(3)

        force_world = gravity_world + thrust_world + drag_world

        # --- Torques (body frame) ---
        torque_body = np.zeros(3)
        if speed > 0.5 and not landed:
            vel_body = R_inv @ vel
            vel_body_norm = vel_body / np.linalg.norm(vel_body)
            body_axis = np.array([0.0, 0.0, 1.0])

            cross = np.cross(body_axis, vel_body_norm)
            sin_aoa = np.linalg.norm(cross)
            if sin_aoa > 1e-6:
                torque_axis = cross / sin_aoa
                torque_mag = params.weathervane_coeff * sin_aoa * speed**2
                torque_body = torque_mag * torque_axis

        # Damping torque
        torque_body -= 0.005 * omega

        # --- True acceleration (world frame) ---
        if landed:
            accel_true_world = np.zeros(3)
        else:
            accel_true_world = force_world / mass

        # --- Generate sensor readings ---

        # Accelerometer: specific force in body frame = a_true − g, measured in body
        g_world = np.array([0.0, 0.0, -G])
        specific_force_world = accel_true_world - g_world
        accel_body = R_inv @ specific_force_world
        accel_meas = accel_body + rng.normal(0, noise.accel_sigma, 3)

        # Gyroscope: angular velocity in body frame
        gyro_meas = omega + noise.gyro_bias + rng.normal(0, noise.gyro_sigma, 3)

        # Magnetometer: Earth field in body frame
        mag_body = R_inv @ MAG_FIELD_WORLD
        mag_meas = mag_body + rng.normal(0, noise.mag_sigma, 3)

        # Barometer
        pressure_true = pressure_from_altitude(pos[2])
        pressure_meas = pressure_true + rng.normal(0, noise.pressure_sigma)

        # Temperature
        temp_true = 15.0 - LAPSE_RATE * pos[2]
        temp_meas = temp_true + rng.normal(0, noise.temp_sigma)

        rows.append({
            "timestamp": round(t, 6),
            "accel_x": accel_meas[0],
            "accel_y": accel_meas[1],
            "accel_z": accel_meas[2],
            "gyro_x": gyro_meas[0],
            "gyro_y": gyro_meas[1],
            "gyro_z": gyro_meas[2],
            "mag_x": mag_meas[0],
            "mag_y": mag_meas[1],
            "mag_z": mag_meas[2],
            "pressure": pressure_meas,
            "temperature": temp_meas,
        })

        # --- Integrate dynamics ---
        if not landed:
            vel += accel_true_world * dt
            pos += vel * dt

            # Quaternion kinematics: dq/dt = 0.5 * q ⊗ ω
            omega_q = np.array([0.0, omega[0], omega[1], omega[2]])
            q_dot = 0.5 * quat_mult(q, omega_q)
            q = quat_normalize(q + q_dot * dt)

            # Angular acceleration
            alpha = torque_body / params.moment_of_inertia
            omega += alpha * dt

    return rows


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

COLUMNS = [
    "timestamp", "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z",
    "mag_x", "mag_y", "mag_z",
    "pressure", "temperature",
]


def write_csv(rows: list[dict[str, float]], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic rocket flight data")
    parser.add_argument("-o", "--output", type=Path, default=Path("simulated_flight.csv"))
    parser.add_argument("--dt", type=float, default=0.001, help="Timestep in seconds")
    parser.add_argument("--duration", type=float, default=30.0, help="Total sim duration in seconds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thrust", type=float, default=20.0, help="Motor thrust in N")
    parser.add_argument("--burn-time", type=float, default=2.0, help="Burn time in seconds")
    parser.add_argument("--tilt", type=float, default=2.0, help="Initial tilt from vertical in degrees")
    parser.add_argument("--mass", type=float, default=0.8, help="Dry mass in kg")
    args = parser.parse_args()

    params = RocketParams(
        thrust=args.thrust,
        burn_time=args.burn_time,
        initial_tilt_deg=args.tilt,
        mass_dry=args.mass,
    )
    noise = SensorNoise()

    print(f"Simulating: {args.thrust:.0f}N for {args.burn_time:.1f}s, "
          f"mass={args.mass:.2f}kg, tilt={args.tilt:.1f}°, dt={args.dt}s")

    rows = simulate(params, noise, dt=args.dt, duration=args.duration, seed=args.seed)
    write_csv(rows, args.output)

    # Print flight summary from barometric pressure
    min_pressure = min(r["pressure"] for r in rows)
    est_apogee = (1 - (min_pressure / SEA_LEVEL_PRESSURE) ** (1 / 5.2561)) * SEA_LEVEL_TEMP / LAPSE_RATE
    max_accel_z = max(r["accel_z"] for r in rows)
    print(f"  Est. apogee: {est_apogee:.1f} m (from barometer)")
    print(f"  Peak accel_z: {max_accel_z:.1f} m/s²")
    print(f"  Duration: {rows[-1]['timestamp']:.1f}s, {len(rows)} samples")


if __name__ == "__main__":
    main()
