import sys
import time

import numpy as np
import serial

from .reader import read_packets, ImuReading, BaroReading

np.set_printoptions(precision=6)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <serial_port> [baud]")
        sys.exit(1)

    port_name = sys.argv[1]
    baud = int(sys.argv[2]) if len(sys.argv) > 2 else 115200

    port = serial.Serial(port_name, baud)
    time.sleep(2)
    port.reset_input_buffer()
    print(f"Connected to {port_name} @ {baud} baud. Recording... (Ctrl+C to stop)")

    accels = []
    mags = []
    gyros = []
    pressures = []

    try:
        for reading in read_packets(port):
            match reading:
                case ImuReading(accel=a, mag=m, gyro=w):
                    accels.append(a)
                    mags.append(m)
                    gyros.append(w)
                case BaroReading(pressure_pa=p):
                    pressures.append(p)
    except KeyboardInterrupt:
        pass

    port.close()

    n_imu = len(accels)
    n_baro = len(pressures)
    print(f"\nRecorded {n_imu} IMU and {n_baro} barometer readings.\n")

    if n_imu == 0 and n_baro == 0:
        print("No data recorded.")
        return

    if n_imu > 0:
        accels = np.array(accels)
        mags = np.array(mags)
        gyros = np.array(gyros)

        print("Accelerometer [m/s²] (x, y, z):")
        print(f"  mean: {accels.mean(axis=0)}")
        print(f"  std:  {accels.std(axis=0)}")

        print("Magnetometer [μT] (x, y, z):")
        print(f"  mean: {mags.mean(axis=0)}")
        print(f"  std:  {mags.std(axis=0)}")

        print("Gyroscope [rad/s] (x, y, z):")
        print(f"  mean: {gyros.mean(axis=0)}")
        print(f"  std:  {gyros.std(axis=0)}")

    if n_baro > 0:
        pressures = np.array(pressures)

        print("Barometer [Pa]:")
        print(f"  mean: {pressures.mean()}")
        print(f"  std:  {pressures.std()}")


if __name__ == "__main__":
    main()
