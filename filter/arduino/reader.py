import struct
import sys
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation
import serial

from filter.main import FlightFilter, get_orientation_and_covariance, skew_symmetric

SYNC = 0xAA
PKT_IMU = 0x01
PKT_BARO = 0x02

# BNO055 raw int16 -> SI conversion factors
ACCEL_SCALE = 1.0 / 100.0  # LSB -> m/s²
MAG_SCALE = 1.0 / 16.0  # LSB -> μT
GYRO_SCALE = 1.0 / 900.0  # LSB -> rad/s

# --- Physical constants ---
G = np.array([0, 0, -9.81])  # gravity vector [m/s^2]
NORTH = np.array([1, 0, 0])  # magnetic north reference (world frame)

# --- IMU noise (continuous-time specifications) ---
SIGMA_ACCEL_NOISE = 0.5  # accelerometer white noise [m/s^2]
SIGMA_GYRO_NOISE = 0.01  # gyroscope white noise [rad/s]
SIGMA_ACCEL_WALK = 0.001  # accelerometer bias random walk [m/s^2/√s]
SIGMA_GYRO_WALK = 0.0001  # gyroscope bias random walk [rad/s/√s]

# --- Sensor noise for observation models ---
SIGMA_GPS = np.array([3, 3, 50])  # GPS position noise [m]
SIGMA_BAROMETER = 0.5  # barometer noise [Pa]
SIGMA_MAGNETOMETER = 0.05  # magnetometer noise [normalized]

MAG_UPDATE_INTERVAL_US = 0.1 * 1e6  # 0.1 seconds

SEA_LEVEL_PRESSURE = 101325.0


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
    (centipascals,) = struct.unpack("<i", payload)
    return BaroReading(
        timestamp_us=timestamp_us,
        pressure_pa=centipascals / 100.0,
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


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <serial_port> [baud]")
        print(f"  e.g. {sys.argv[0]} /dev/tty.usbmodem14101 115200")
        sys.exit(1)

    port_name = sys.argv[1]
    baud = int(sys.argv[2]) if len(sys.argv) > 2 else 115200

    port = serial.Serial(port_name, baud)

    first_imu_reading: ImuReading | None = None
    first_baro_reading: BaroReading | None = None
    kf: FlightFilter | None = None

    last_timestamp_us = 0
    last_mag_update_timestamp_us = 0

    def h_barometer(x_nom):
        return SEA_LEVEL_PRESSURE * (1 - x_nom[2] / 44330) ** (1 / 0.1903)

    R_barometer = np.array([[SIGMA_BAROMETER**2]])

    def h_magnetometer(x_nom):
        world_orientation = Rotation.from_quat(x_nom[6:10], scalar_first=True)
        return world_orientation.inv().apply(NORTH)

    R_magnetometer = np.eye(3) * SIGMA_MAGNETOMETER**2

    for reading in read_packets(port):
        match reading:
            case ImuReading(timestamp_us=t, accel=a, gyro=g, mag=m):
                # print(
                #     f"IMU  t={t:>12}us "
                #     f"a=[{a[0]:+8.3f} {a[1]:+8.3f} {a[2]:+8.3f}] m/s²  "
                #     f"g=[{g[0]:+8.4f} {g[1]:+8.4f} {g[2]:+8.4f}] rad/s  "
                #     f"m=[{m[0]:+8.2f} {m[1]:+8.2f} {m[2]:+8.2f}] μT"
                # )
                if not first_imu_reading:
                    first_imu_reading = reading
                if kf is None:
                    continue
                dt = (t - last_timestamp_us) / 1e6
                kf.predict(np.concatenate((a, g)), dt)
                if t - last_mag_update_timestamp_us > MAG_UPDATE_INTERVAL_US:
                    z = m / np.linalg.norm(m)
                    q0, q1, q2, q3 = kf.x_nom[6:10]
                    p = np.array([q1, q2, q3])

                    H_x_magnetometer = np.zeros((3, 16))
                    H_x_magnetometer[0:3, 6:7] = 2 * (
                        q0 * NORTH.reshape(3, 1) + np.cross(NORTH, p).reshape(3, 1)
                    )
                    H_x_magnetometer[0:3, 7:10] = 2 * (
                        np.dot(p, NORTH) * np.eye(3)
                        + np.outer(p, NORTH)
                        - np.outer(NORTH, p)
                        + q0 * skew_symmetric(NORTH)
                    )

                    ll, accepted = kf.update(
                        h_magnetometer, z, R_magnetometer, H_x_magnetometer
                    )
                    last_mag_update_timestamp_us = t

                last_timestamp_us = t
            case BaroReading(timestamp_us=t, pressure_pa=p):
                # print(f"BARO t={t:>12}us  p={p:>10.2f} Pa")
                if not first_baro_reading:
                    first_baro_reading = reading
                if kf is None:
                    continue

                H_x_barometer = np.zeros((1, 16))
                H_x_barometer[0, 2] = (
                    -1000
                    * SEA_LEVEL_PRESSURE
                    * (1 - kf.x_nom[2] / 44330) ** (8097 / 1903)
                    / 8435999
                )
                ll, accepted = kf.update(
                    h_barometer, np.array([p]), R_barometer, H_x_barometer
                )
                last_timestamp_us = t
        if first_imu_reading and first_baro_reading and kf is None:
            # initialize the filter
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
            P = np.eye(15) * 500
            P[0:3, 0:3] = np.eye(3) * SIGMA_GPS**2
            P[2, 2] = SIGMA_BAROMETER**2
            P[6:9, 6:9] = R_cov
            P[9:12, 9:12] = np.eye(3) * SIGMA_ACCEL_WALK**2 * 2
            P[12:15, 12:15] = np.eye(3) * SIGMA_GYRO_WALK**2 * 2
            kf = FlightFilter(
                x_nom=x_nom,
                P=P,
                sigma_a_noise=SIGMA_ACCEL_NOISE,
                sigma_w_noise=SIGMA_GYRO_NOISE,
                sigma_a_walk=SIGMA_ACCEL_WALK,
                sigma_w_walk=SIGMA_GYRO_WALK,
                g=G,
            )
            continue


if __name__ == "__main__":
    main()
