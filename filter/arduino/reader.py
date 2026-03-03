import signal
import struct
import sys
import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import chi2
import serial

from .visualizer import LiveVisualizer, NISTracker
from ..main import FlightFilter, get_orientation_and_covariance, skew_symmetric

np.set_printoptions(precision=3)

SYNC = 0xAA
PKT_IMU = 0x01
PKT_BARO = 0x02

# BNO055 raw int16 -> SI conversion factors
ACCEL_SCALE = 1.0 / 100.0  # LSB -> m/s²
MAG_SCALE = 1.0 / 16.0  # LSB -> μT
GYRO_SCALE = 1.0 / 16.0 * np.pi / 180.0  # LSB -> deg/s -> rad/s

# --- Physical constants ---
G = np.array([0, 0, -9.81])  # gravity vector [m/s^2]

# Magnetic field reference (world frame, normalized)
# Look up for your location at: https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml
MAG_INCLINATION_DEG = 64.78194  # angle below horizontal (positive = into ground)
MAG_DECLINATION_DEG = -10.45  # angle east of true north (irrelevant without GPS)
_inc = np.radians(MAG_INCLINATION_DEG)
_dec = np.radians(MAG_DECLINATION_DEG)
NORTH = np.array(
    [np.cos(_inc) * np.cos(_dec), np.cos(_inc) * np.sin(_dec), -np.sin(_inc)]
)

# --- IMU noise (continuous-time specifications) ---
# noise_rms = noise_density * sqrt(sample_rate * 1.57)
# The 1.57 factor is a common approximation for the "noise equivalent bandwidth" of a first-order low-pass filter.)
# noise density from datasheet: 150 μg/√Hz (too optimistic for real-world)
# SIGMA_ACCEL_NOISE = (
#     150 * 9.81 * 1e-6 * np.sqrt(100 * 1.57)
# )  # accelerometer white noise [m/s^2]
SIGMA_ACCEL_NOISE = 0.025  # experimentally determined
# SIGMA_GYRO_NOISE = (
#     0.014 * np.pi / 180.0 * np.sqrt(100 * 1.57)
# )  # 0.014 deg / s / sqrt(Hz)
SIGMA_GYRO_NOISE = 0.0026  # experimentally determined
SIGMA_ACCEL_WALK = 0.0001  # accelerometer bias random walk [m/s^2/√s] (estimate)
SIGMA_GYRO_WALK = 0.0001  # gyroscope bias random walk [rad/s/√s] (estimate)

# --- Sensor noise for observation models ---
SIGMA_GPS = np.array([3, 3, 50])  # GPS position noise [m]
SIGMA_BAROMETER = 3.812  # barometer noise [Pa]
# SIGMA_MAGNETOMETER = 1.0  # magnetometer noise [uT]
SIGMA_MAGNETOMETER = 3.911  # experimentally determined

MAG_UPDATE_INTERVAL_US = 0.1 * 1e6  # 0.1 seconds
ACCEL_UPDATE_INTERVAL_US = 0.1 * 1e6  # 0.1 seconds

SEA_LEVEL_PRESSURE = 101325.0
CALIBRATION_DURATION_US = 5 * 1e6  # 5 seconds


@dataclass
class ImuReading:
    timestamp_us: int
    accel: np.ndarray  # m/s², shape (3,)
    mag: np.ndarray  # μT, shape (3,)
    gyro: np.ndarray  # rad/s, shape (3,)


@dataclass
class BaroReading:
    timestamp_us: int
    pressure_pa: float


PAYLOAD_SIZES = {
    PKT_IMU: 18,
    PKT_BARO: 4,
}


def decode_imu(timestamp_us: int, payload: bytes) -> ImuReading:
    raw = struct.unpack("<9h", payload)
    return ImuReading(
        timestamp_us=timestamp_us,
        accel=np.array(raw[0:3]) * ACCEL_SCALE,
        mag=np.array(raw[3:6]) * MAG_SCALE,
        gyro=np.array(raw[6:9]) * GYRO_SCALE,
    )


def decode_baro(timestamp_us: int, payload: bytes) -> BaroReading:
    (pascals,) = struct.unpack("<i", payload)
    return BaroReading(
        timestamp_us=timestamp_us,
        pressure_pa=pascals,
    )


DECODERS = {
    PKT_IMU: decode_imu,
    PKT_BARO: decode_baro,
}


def read_packets(port: serial.Serial):
    """Yields decoded sensor readings from the serial stream."""
    while True:
        # Scan for sync byte
        if port.read(1)[0] != SYNC:
            continue

        header = port.read(6)
        if len(header) < 6:
            continue
        pkt_type, length, timestamp_us = struct.unpack("<BBI", header)

        expected_len = PAYLOAD_SIZES.get(pkt_type)
        if expected_len is None or length != expected_len:
            continue

        payload = port.read(length)
        if len(payload) < length:
            continue

        crc_byte = port.read(1)
        if len(crc_byte) < 1:
            continue

        # Verify CRC (XOR of all bytes after sync)
        crc = 0
        for b in header + payload:
            crc ^= b
        if crc != crc_byte[0]:
            continue

        yield DECODERS[pkt_type](timestamp_us, payload)


def pressure_to_altitude(pressure_pa: float) -> float:
    return 44330 * (1 - (pressure_pa / SEA_LEVEL_PRESSURE) ** 0.1903)


def is_imu_static(
    a_m: np.ndarray,
    w_m: np.ndarray,
    g: np.ndarray,
    static_gyro_ln_magnitude_mean: float,
    static_gyro_ln_magnitude_var: float,
) -> bool:
    criterion_a1 = np.abs(np.linalg.norm(a_m) - np.linalg.norm(g)) < 0.2
    x = np.log(np.linalg.norm(w_m) + 0.00001)
    D2 = (x - static_gyro_ln_magnitude_mean) ** 2 / static_gyro_ln_magnitude_var
    criterion_a2 = D2 < chi2.ppf(0.997, 1)
    return criterion_a1 and criterion_a2


def is_mag_interference_absent(
    a_m: np.ndarray,
    w_m: np.ndarray,
    m_m: np.ndarray,
    static_accel_mag_angle_mean: float,
    static_accel_mag_angle_var: float,
    is_imu_static: bool,
    last_m_m: np.ndarray,
    dt: float,
    x_nom: np.ndarray,
) -> bool:
    if is_imu_static:
        cos_angle = np.clip(
            np.dot(m_m, a_m) / (np.linalg.norm(m_m) * np.linalg.norm(a_m)), -1, 1
        )
        x = np.abs(np.arccos(cos_angle))
        D2 = (x - static_accel_mag_angle_mean) ** 2 / static_accel_mag_angle_var
        criterion_m1 = D2 < chi2.ppf(0.997, 1)
        return criterion_m1
    else:
        yaw_angvel_mag = np.abs(
            (1 / dt)
            * np.arccos(
                np.clip(
                    np.dot(m_m, last_m_m)
                    / (np.linalg.norm(m_m) * np.linalg.norm(last_m_m)),
                    -1,
                    1,
                )
            )
        )
        phi, theta, psi = Rotation.from_quat(x_nom[6:10], scalar_first=True).as_euler(
            "xyz", degrees=False
        )
        yaw_angvel_gyro = np.abs(
            -(np.sin(phi) / np.cos(theta)) * w_m[1]
            + (np.cos(phi) / np.cos(theta)) * w_m[2]
        )
        criterion_m2 = (
            np.abs(yaw_angvel_mag - yaw_angvel_gyro)
            <= 10.0 * np.pi / 180.0  # 10 degrees/s
        )
        return criterion_m2


def is_filter_consistent(nis: float, nz: int, alpha=0.05):
    lower_bound = chi2.ppf(alpha / 2, nz)
    upper_bound = chi2.ppf(1 - alpha / 2, nz)
    return lower_bound <= nis <= upper_bound


def normalization_jacobian(a: np.ndarray) -> np.ndarray:
    a_norm = np.linalg.norm(a)
    return (np.eye(3) - np.outer(a, a) / a_norm**2) / a_norm


class AdaptiveR:
    def __init__(self, R: np.ndarray, min_samples: int = 50):
        self.R = R.copy()
        self.min_samples = min_samples
        self._yy_sum = np.zeros_like(R)
        self._hpht_sum = np.zeros_like(R)
        self._n = 0

    def record(self, innovation: np.ndarray, hpht: np.ndarray):
        self._yy_sum += np.outer(innovation, innovation)
        self._hpht_sum += hpht
        self._n += 1

    def adapt(self):
        if self._n < self.min_samples:
            return
        R_new = self._yy_sum / self._n - self._hpht_sum / self._n
        eigvals, eigvecs = np.linalg.eigh(R_new)
        eigvals = np.maximum(eigvals, 1e-15)
        self.R = eigvecs @ np.diag(eigvals) @ eigvecs.T
        self._yy_sum = np.zeros_like(self.R)
        self._hpht_sum = np.zeros_like(self.R)
        self._n = 0


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <serial_port> [baud]")
        print(f"  e.g. {sys.argv[0]} /dev/tty.usbmodem14101 115200")
        sys.exit(1)

    port_name = sys.argv[1]
    baud = int(sys.argv[2]) if len(sys.argv) > 2 else 115200

    port = serial.Serial(port_name, baud)
    time.sleep(2)  # wait for Arduino to reset after serial open
    port.reset_input_buffer()  # discard stale pre-reset data
    print(f"Connected to {port_name} @ {baud} baud. Waiting for packets...")

    first_imu_reading: ImuReading | None = None
    first_baro_reading: BaroReading | None = None
    kf: FlightFilter | None = None

    calibration_start_timestamp_us = 0
    last_imu_timestamp_us = 0
    last_mag_update_timestamp_us = 0
    last_accel_update_timestamp_us = 0
    last_viz_update_timestamp_us = 0

    static_gyro_magnitudes = []
    static_gyro_ln_magnitude_mean = 0
    static_gyro_ln_magnitude_var = 0
    static_accel_mag_angles = []
    static_accel_mag_angle_mean = 0
    static_accel_mag_angle_var = 0

    def h_barometer(x_nom):
        ratio = max(1 - x_nom[2] / 44330, 1e-12)
        return SEA_LEVEL_PRESSURE * ratio ** (1 / 0.1903)

    # R_barometer = np.array([[SIGMA_BAROMETER**2]])
    # R_barometer = np.array([[6.6364]])

    R_ADAPT_INTERVAL_US = 30 * 1e6  # 30 seconds
    adaptive_Rs = {
        "ZUPT": AdaptiveR(
            np.array(
                [
                    [1.000e-15, 5.975e-32, -1.060e-32],
                    [1.926e-32, 1.000e-15, -6.153e-32],
                    [0.000e00, -9.861e-32, 1.000e-15],
                ]
            )
        ),
        "Mag": AdaptiveR(
            np.array(
                [
                    [2.713e-03, -2.259e-04, -2.290e-04],
                    [-2.259e-04, 1.662e-04, 2.027e-05],
                    [-2.290e-04, 2.027e-05, 1.946e-05],
                ]
            )
        ),
        "Accel": AdaptiveR(
            np.array(
                [
                    [1.554e-06, -6.634e-07, 4.738e-08],
                    [-6.634e-07, 2.731e-06, -7.760e-07],
                    [4.738e-08, -7.760e-07, 2.348e-07],
                ]
            )
        ),
        "Baro": AdaptiveR(np.array([[4.424]])),
    }
    last_R_adapt_timestamp_us = 0

    def h_magnetometer(x_nom):
        world_orientation = Rotation.from_quat(x_nom[6:10], scalar_first=True)
        return world_orientation.inv().apply(NORTH)

    def h_accelerometer(x_nom):
        world_orientation = Rotation.from_quat(x_nom[6:10], scalar_first=True)
        g_body = world_orientation.inv().apply(-G)
        a_b = x_nom[10:13]
        predicted = g_body + a_b
        return predicted / np.linalg.norm(predicted)

    def h_velocity(x_nom):
        return x_nom[3:6]

    # calibration
    print("Calibrating...")
    for reading in read_packets(port):
        match reading:
            case ImuReading(timestamp_us=t, accel=a, gyro=w, mag=m):
                if calibration_start_timestamp_us == 0:
                    calibration_start_timestamp_us = t
                static_gyro_magnitudes.append(np.linalg.norm(w))
                theta = np.abs(
                    np.arccos(np.dot(m, a) / (np.linalg.norm(m) * np.linalg.norm(a)))
                )
                static_accel_mag_angles.append(theta)
                if t - calibration_start_timestamp_us > CALIBRATION_DURATION_US:
                    static_gyro_ln_magnitude = np.log(
                        np.array(static_gyro_magnitudes) + 0.00001
                    )
                    static_gyro_ln_magnitude_mean = np.mean(
                        static_gyro_ln_magnitude
                    ).item()
                    static_gyro_ln_magnitude_var = np.var(
                        static_gyro_ln_magnitude
                    ).item()
                    static_accel_mag_angle = np.array(static_accel_mag_angles)
                    static_accel_mag_angle_mean = np.mean(static_accel_mag_angle).item()
                    static_accel_mag_angle_var = np.var(static_accel_mag_angle).item()
                    break
            case BaroReading(timestamp_us=t, pressure_pa=p):
                pass
    print("Calibration complete.")
    print(f"Static gyro ln magnitude mean: {static_gyro_ln_magnitude_mean}")
    print(f"Static gyro ln magnitude var: {static_gyro_ln_magnitude_var}")

    viz = LiveVisualizer()
    viz.process_events()

    nis_trackers = {
        "Mag": NISTracker(nz=3),
        "Accel": NISTracker(nz=3),
        "Baro": NISTracker(nz=1),
        "ZUPT": NISTracker(nz=3),
    }
    innovations: dict[str, list[np.ndarray]] = {
        "Mag": [],
        "Accel": [],
        "Baro": [],
        "ZUPT": [],
    }

    R_VARIABLE_NAMES = {
        "Mag": "R_magnetometer_normalized",
        "Accel": "R_accel_normalized",
        "Baro": "R_barometer",
        "ZUPT": "R_zupt",
    }

    def print_innovation_covariances():
        print("\n--- Innovation Covariances (use as R) ---")
        for name, innov_list in innovations.items():
            if len(innov_list) < 2:
                print(f"{name}: not enough samples ({len(innov_list)})")
                continue
            innov_array = np.array(innov_list)
            cov = np.cov(innov_array.T)
            var_name = R_VARIABLE_NAMES[name]
            print(f"# {name} (n={len(innov_list)}, mean={innov_array.mean(axis=0)})")
            print(
                f"{var_name} = np.array({np.array2string(np.atleast_2d(cov), separator=', ')})"
            )
        print("\n--- Current Adaptive R values ---")
        for name, ar in adaptive_Rs.items():
            var_name = R_VARIABLE_NAMES[name]
            print(
                f"{var_name} = np.array({np.array2string(np.atleast_2d(ar.R), separator=', ')})"
            )

    def handle_exit(sig, frame):
        print_innovation_covariances()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)

    last_m_m: np.ndarray | None = None

    for reading in read_packets(port):
        match reading:
            case ImuReading(timestamp_us=t, accel=a, gyro=w, mag=m):
                if not first_imu_reading:
                    first_imu_reading = reading
                    print(f"First IMU reading at t={t}us")
                elif kf:
                    dt = (t - last_imu_timestamp_us) / 1e6
                    u = np.concatenate((a, w))
                    kf.predict(u, dt)
                    imu_static = is_imu_static(
                        a,
                        w,
                        G,
                        static_gyro_ln_magnitude_mean,
                        static_gyro_ln_magnitude_var,
                    )
                    mag_norm = np.linalg.norm(m)
                    last_m_m_norm = (
                        np.linalg.norm(last_m_m) if last_m_m is not None else 0
                    )
                    if imu_static:
                        # ZUPT (zero-velocity update)
                        z = np.zeros(3)
                        H_x_zupt = np.zeros((3, 16))
                        H_x_zupt[0:3, 3:6] = np.eye(3)
                        # R_zupt = np.array(
                        #     [
                        #         [1.131e-05, -5.442e-06, -1.518e-06],
                        #         [-5.442e-06, 8.847e-06, -9.585e-07],
                        #         [-1.518e-06, -9.585e-07, 3.959e-06],
                        #     ]
                        # )
                        H_zupt = H_x_zupt @ kf.get_X_dx()
                        hpht_zupt = H_zupt @ kf.f.P @ H_zupt.T
                        ll, accepted, nis, innovation = kf.update(
                            h_velocity,
                            z,
                            adaptive_Rs["ZUPT"].R,
                            H_x_zupt,
                            gating_threshold=1,
                        )
                        if accepted:
                            adaptive_Rs["ZUPT"].record(innovation, hpht_zupt)
                            nis_trackers["ZUPT"].record(nis)
                            innovations["ZUPT"].append(innovation)
                    if (
                        t - last_mag_update_timestamp_us > MAG_UPDATE_INTERVAL_US
                        and mag_norm > 0
                        and last_m_m is not None
                        and last_m_m_norm > 0
                    ):
                        z = m / mag_norm

                        H_x_magnetometer = np.zeros((3, 16))
                        H_x_magnetometer[0:3, 6:10] = kf.get_inverse_rotation_H_x(NORTH)
                        # N = normalization_jacobian(m)
                        # R_magnetometer_normalized = (
                        #     N @ (np.eye(3) * SIGMA_MAGNETOMETER**2) @ N.T
                        # )
                        # R_magnetometer_normalized = np.eye(3) * 0.00225

                        mag_interference_absent = is_mag_interference_absent(
                            a,
                            w,
                            m,
                            static_accel_mag_angle_mean,
                            static_accel_mag_angle_var,
                            imu_static,
                            last_m_m,
                            dt,
                            kf.x_nom,
                        )
                        if mag_interference_absent:
                            H_mag = H_x_magnetometer @ kf.get_X_dx()
                            hpht_mag = H_mag @ kf.f.P @ H_mag.T
                            ll, accepted, nis, innovation = kf.update(
                                h_magnetometer,
                                z,
                                adaptive_Rs["Mag"].R,
                                H_x_magnetometer,
                                gating_threshold=1,
                            )
                            if not accepted:
                                print(
                                    f"Magnetometer measurement rejected at t={t}us",
                                )
                                print(f"measurement: {m}")
                            else:
                                adaptive_Rs["Mag"].record(innovation, hpht_mag)
                                last_mag_update_timestamp_us = t
                                nis_trackers["Mag"].record(nis)
                                innovations["Mag"].append(innovation)
                        else:
                            pass
                    if t - last_accel_update_timestamp_us > ACCEL_UPDATE_INTERVAL_US:
                        if imu_static:
                            z = a / np.linalg.norm(a)

                            world_orientation = Rotation.from_quat(
                                kf.x_nom[6:10], scalar_first=True
                            )
                            raw_pred = (
                                world_orientation.inv().apply(-G) + kf.x_nom[10:13]
                            )
                            N_pred = normalization_jacobian(raw_pred)
                            H_x_accel = np.zeros((3, 16))
                            H_x_accel[0:3, 6:10] = N_pred @ kf.get_inverse_rotation_H_x(
                                -G
                            )
                            H_x_accel[0:3, 10:13] = N_pred @ np.eye(3)
                            # N = normalization_jacobian(a)
                            # R_accel_normalized = (
                            #     N @ (np.eye(3) * SIGMA_ACCEL_NOISE**2) @ N.T
                            # )
                            # R_accel_normalized = np.eye(3) * 0.0000083523
                            H_accel = H_x_accel @ kf.get_X_dx()
                            hpht_accel = H_accel @ kf.f.P @ H_accel.T
                            ll, accepted, nis, innovation = kf.update(
                                h_accelerometer,
                                z,
                                adaptive_Rs["Accel"].R,
                                H_x_accel,
                                gating_threshold=1,
                            )

                            if not accepted:
                                print(f"Accelerometer measurement rejected at t={t}us")
                            else:
                                adaptive_Rs["Accel"].record(innovation, hpht_accel)
                                last_accel_update_timestamp_us = t
                                nis_trackers["Accel"].record(nis)
                                innovations["Accel"].append(innovation)

                    # adapt R every 30 seconds
                    if t - last_R_adapt_timestamp_us > R_ADAPT_INTERVAL_US:
                        for name, ar in adaptive_Rs.items():
                            ar.adapt()
                        last_R_adapt_timestamp_us = t
                        print("Adapted R values:")
                        for name, ar in adaptive_Rs.items():
                            print(f"  {name}: diag={np.diag(ar.R)}")

                    if t - last_viz_update_timestamp_us > 1 / 30 * 1e6:
                        viz.update_measurements(a, m)
                        viz.update(kf.x_nom, kf.f.P)
                        viz.update_nis_overlay(nis_trackers)
                        viz.update_bias_overlay(kf.x_nom[10:13], kf.x_nom[13:16])
                        last_viz_update_timestamp_us = t

                viz.process_events()
                last_imu_timestamp_us = t
                last_m_m = m
            case BaroReading(timestamp_us=t, pressure_pa=p):
                if not first_baro_reading:
                    first_baro_reading = reading
                    print(f"First baro reading at t={t}us, p={p:.2f} Pa")
                elif kf:
                    H_x_barometer = np.zeros((1, 16))
                    ratio = max(1 - kf.x_nom[2] / 44330, 1e-12)
                    H_x_barometer[0, 2] = (
                        -1000 * SEA_LEVEL_PRESSURE * ratio ** (8097 / 1903) / 8435999
                    )
                    H_baro = H_x_barometer @ kf.get_X_dx()
                    hpht_baro = H_baro @ kf.f.P @ H_baro.T
                    ll, accepted, nis, innovation = kf.update(
                        h_barometer,
                        np.array([p]),
                        adaptive_Rs["Baro"].R,
                        H_x_barometer,
                        gating_threshold=1,
                    )
                    if not accepted:
                        print(
                            f"Barometer measurement rejected at t={t}us, p={p:.2f} Pa",
                        )
                        print(
                            f"filter altitude: {kf.x_nom[2]} m, covariance: {kf.f.P[2, 2]}"
                        )
                        print(
                            f"""measured pressure to altitude: {pressure_to_altitude(p)} m, covariance: {(
                (8436.2 / p) * ((p / SEA_LEVEL_PRESSURE) ** 0.1903) * SIGMA_BAROMETER
            ) ** 2}"""
                        )
                    else:
                        adaptive_Rs["Baro"].record(innovation, hpht_baro)
                        nis_trackers["Baro"].record(nis)
                        innovations["Baro"].append(innovation)

        if not kf and first_imu_reading and first_baro_reading:
            print(
                f"Initializing filter with accel={first_imu_reading.accel}, mag={first_imu_reading.mag}",
            )
            q, R_cov = get_orientation_and_covariance(
                first_imu_reading.accel,
                first_imu_reading.mag,
                np.ones(3) * SIGMA_ACCEL_NOISE,
                np.ones(3) * SIGMA_MAGNETOMETER,
                G,
                NORTH,
            )
            x_nom = np.concatenate(
                (
                    np.array(
                        [
                            0,
                            0,
                            pressure_to_altitude(first_baro_reading.pressure_pa),
                        ]
                    ),
                    np.zeros(3),
                    q,
                    np.zeros(3),
                    np.zeros(3),
                )
            )
            p = first_baro_reading.pressure_pa
            P = np.zeros((15, 15))
            P[0:2, 0:2] = (
                np.eye(2) * 0.01**2
            )  # horizontal position: self-defined origin
            P[2, 2] = (
                (8436.2 / p) * ((p / SEA_LEVEL_PRESSURE) ** 0.1903) * SIGMA_BAROMETER
            ) ** 2  # altitude from barometer
            P[3:6, 3:6] = np.eye(3) * 0.01**2  # velocity: starts stationary
            P[6:9, 6:9] = R_cov
            SIGMA_ACCEL_BIAS_INIT = (
                80 * 9.81 * 1e-3
            )  # BNO055 accel offset: typical 80mg, max 150mg
            SIGMA_GYRO_BIAS_INIT = np.radians(
                1.0
            )  # BNO055 gyro zero-rate offset: typical ±1°/s, max +3°/s
            P[9:12, 9:12] = np.eye(3) * SIGMA_ACCEL_BIAS_INIT**2
            P[12:15, 12:15] = np.eye(3) * SIGMA_GYRO_BIAS_INIT**2
            print(f"Initial state: {x_nom}")
            print(f"Initial covariance: {P}")
            kf = FlightFilter(
                x_nom=x_nom,
                P=P,
                sigma_a_noise=SIGMA_ACCEL_NOISE,
                sigma_w_noise=SIGMA_GYRO_NOISE,
                sigma_a_walk=SIGMA_ACCEL_WALK,
                sigma_w_walk=SIGMA_GYRO_WALK,
                g=G,
            )
            print("Filter initialized. Opening visualization...")


if __name__ == "__main__":
    main()
