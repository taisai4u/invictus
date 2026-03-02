import struct
import sys
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import chi2
import serial

from .visualizer import LiveVisualizer
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
# noise density from datasheet: 150 μg/√Hz
SIGMA_ACCEL_NOISE = (
    150 * np.sqrt(1000 * 1.57) * 9.81 * 1e-6
)  # accelerometer white noise [m/s^2]
SIGMA_GYRO_NOISE = 0.1  # gyroscope white noise [rad/s]
SIGMA_ACCEL_WALK = 0.001  # accelerometer bias random walk [m/s^2/√s]
SIGMA_GYRO_WALK = 0.0001  # gyroscope bias random walk [rad/s/√s]

# --- Sensor noise for observation models ---
SIGMA_GPS = np.array([3, 3, 50])  # GPS position noise [m]
SIGMA_BAROMETER = 1.3  # barometer noise [Pa]
SIGMA_MAGNETOMETER = 1.0  # magnetometer noise [uT]

MAG_UPDATE_INTERVAL_US = 0.1 * 1e6  # 0.1 seconds

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
    x = np.log(np.linalg.norm(w_m))
    D2 = (x - static_gyro_ln_magnitude_mean) ** 2 / static_gyro_ln_magnitude_var
    criterion_a2 = D2 < chi2.ppf(0.997, 1)
    return criterion_a1 and criterion_a2


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <serial_port> [baud]")
        print(f"  e.g. {sys.argv[0]} /dev/tty.usbmodem14101 115200")
        sys.exit(1)

    port_name = sys.argv[1]
    baud = int(sys.argv[2]) if len(sys.argv) > 2 else 115200

    port = serial.Serial(port_name, baud)
    print(f"Connected to {port_name} @ {baud} baud. Waiting for packets...")

    first_imu_reading: ImuReading | None = None
    first_baro_reading: BaroReading | None = None
    kf: FlightFilter | None = None

    calibration_start_timestamp_us = 0
    last_imu_timestamp_us = 0
    last_mag_update_timestamp_us = 0
    last_viz_update_timestamp_us = 0

    static_gyro_magnitudes = []
    static_gyro_ln_magnitude_mean = 0
    static_gyro_ln_magnitude_var = 0

    def h_barometer(x_nom):
        ratio = max(1 - x_nom[2] / 44330, 1e-12)
        return SEA_LEVEL_PRESSURE * ratio ** (1 / 0.1903)

    R_barometer = np.array([[SIGMA_BAROMETER**2]])

    def h_magnetometer(x_nom):
        world_orientation = Rotation.from_quat(x_nom[6:10], scalar_first=True)
        return world_orientation.inv().apply(NORTH)

    # calibration
    print("Calibrating...")
    for reading in read_packets(port):
        match reading:
            case ImuReading(timestamp_us=t, accel=a, gyro=w, mag=m):
                if calibration_start_timestamp_us == 0:
                    calibration_start_timestamp_us = t
                static_gyro_magnitudes.append(np.linalg.norm(w))
                if t - calibration_start_timestamp_us > CALIBRATION_DURATION_US:
                    print("calibration duration reached")
                    static_gyro_ln_magnitude = np.log(
                        np.array(static_gyro_magnitudes) + 0.001
                    )
                    static_gyro_ln_magnitude_mean = np.mean(
                        static_gyro_ln_magnitude
                    ).item()
                    static_gyro_ln_magnitude_var = np.var(
                        static_gyro_ln_magnitude
                    ).item()
                    break
            case BaroReading(timestamp_us=t, pressure_pa=p):
                pass
    print("Calibration complete.")
    print(f"Static gyro ln magnitude mean: {static_gyro_ln_magnitude_mean}")
    print(f"Static gyro ln magnitude var: {static_gyro_ln_magnitude_var}")

    viz = LiveVisualizer()
    viz.process_events()

    for reading in read_packets(port):
        match reading:
            case ImuReading(timestamp_us=t, accel=a, gyro=w, mag=m):
                if not first_imu_reading:
                    first_imu_reading = reading
                    print(f"First IMU reading at t={t}us")
                elif kf:
                    dt = (t - last_imu_timestamp_us) / 1e6
                    u = np.concatenate((a, w))
                    # print(f"Predicting with u={u}")
                    kf.predict(u, dt)
                    if t - last_mag_update_timestamp_us > MAG_UPDATE_INTERVAL_US:
                        z = m / np.linalg.norm(m)
                        q0, q1, q2, q3 = kf.x_nom[6:10]
                        p = np.array([q1, q2, q3])

                        H_x_magnetometer = np.zeros((3, 16))
                        rot = Rotation.from_quat(kf.x_nom[6:10], scalar_first=True)
                        H_x_magnetometer[0:3, 6:10] = kf.get_inverse_rotation_H_x(NORTH)
                        R_magnetometer = (
                            np.eye(3)
                            * SIGMA_MAGNETOMETER**2
                            * 100
                            / np.linalg.norm(m) ** 2
                        )

                        if not is_imu_static(
                            a,
                            w,
                            G,
                            static_gyro_ln_magnitude_mean,
                            static_gyro_ln_magnitude_var,
                        ):
                            print(f"IMU is not static at t={t}us")
                        ll, accepted = kf.update(
                            h_magnetometer,
                            z,
                            R_magnetometer,
                            H_x_magnetometer,
                            gating_threshold=0.997,
                        )
                        if not accepted:
                            print(
                                f"Magnetometer measurement rejected at t={t}us",
                            )
                            print(f"measurement: {m}")
                        last_mag_update_timestamp_us = t

                    if t - last_viz_update_timestamp_us > 1 / 30 * 1e6:
                        viz.update(kf.x_nom, kf.f.P)
                        last_viz_update_timestamp_us = t

                viz.process_events()
                last_imu_timestamp_us = t
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
            #         ll, accepted = kf.update(
            #             h_barometer,
            #             np.array([p]),
            #             R_barometer,
            #             H_x_barometer,
            #             # gating_threshold=1,
            #         )
            #         if not accepted:
            #             print(
            #                 f"Barometer measurement rejected at t={t}us, p={p:.2f} Pa",
            #             )
            #             print(
            #                 f"filter altitude: {kf.x_nom[2]} m, covariance: {kf.f.P[2, 2]}"
            #             )
            #             print(
            #                 f"""measured pressure to altitude: {pressure_to_altitude(p)} m, covariance: {(
            #     (8436.2 / p) * ((p / SEA_LEVEL_PRESSURE) ** 0.1903) * SIGMA_BAROMETER
            # ) ** 2}"""
            #             )

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
            P = np.eye(15) * 500
            P[0:3, 0:3] = np.eye(3) * SIGMA_GPS**2
            P[2, 2] = (
                (8436.2 / p) * ((p / SEA_LEVEL_PRESSURE) ** 0.1903) * SIGMA_BAROMETER
            ) ** 2
            P[6:9, 6:9] = R_cov
            SIGMA_ACCEL_BIAS_INIT = 1.0  # BNO055 accel offset: typical 80mg, max 150mg
            SIGMA_GYRO_BIAS_INIT = np.radians(
                3.0
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
