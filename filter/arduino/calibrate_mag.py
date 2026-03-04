import sys
import time

import numpy as np
import serial

from .reader import read_packets, ImuReading


def collect_mag_samples(port_name: str, baud: int, n_samples: int) -> np.ndarray:
    port = serial.Serial(port_name, baud)
    time.sleep(2)
    port.reset_input_buffer()
    print(f"Connected to {port_name} @ {baud} baud.")
    print(f"Slowly rotate the sensor through all orientations.")
    print(f"Collecting {n_samples} magnetometer samples...\n")

    samples = []
    for reading in read_packets(port):
        if not isinstance(reading, ImuReading):
            continue
        samples.append(reading.mag.copy())
        if len(samples) % 100 == 0:
            print(f"  {len(samples)}/{n_samples} samples collected")
        if len(samples) >= n_samples:
            break

    port.close()
    return np.array(samples)


def fit_sphere(samples: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit a sphere to 3D points using least-squares.

    Model: (x - cx)^2 + (y - cy)^2 + (z - cz)^2 = r^2
    Rewrite as: 2*cx*x + 2*cy*y + 2*cz*z + (r^2 - cx^2 - cy^2 - cz^2) = x^2 + y^2 + z^2

    This is linear in [2*cx, 2*cy, 2*cz, r^2 - |c|^2], so solve with least squares.
    """
    x, y, z = samples[:, 0], samples[:, 1], samples[:, 2]
    A = np.column_stack([x, y, z, np.ones(len(x))])
    b = x**2 + y**2 + z**2
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    center = result[:3] / 2.0
    radius = np.sqrt(result[3] + np.dot(center, center))
    return center, radius


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python -m filter.arduino.calibrate_mag <serial_port> [baud] [n_samples]")
        sys.exit(1)

    port_name = sys.argv[1]
    baud = int(sys.argv[2]) if len(sys.argv) > 2 else 115200
    n_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 1000

    samples = collect_mag_samples(port_name, baud, n_samples)

    hard_iron_offset, radius = fit_sphere(samples)

    corrected = samples - hard_iron_offset
    corrected_norms = np.linalg.norm(corrected, axis=1)

    print(f"\n--- Results ---")
    print(f"Hard-iron offset: {hard_iron_offset}")
    print(f"Fitted sphere radius: {radius:.4f} uT")
    print(f"Corrected field magnitude: {corrected_norms.mean():.4f} +/- {corrected_norms.std():.4f} uT")
    print(f"\nRaw field magnitude: {np.linalg.norm(samples, axis=1).mean():.4f} +/- {np.linalg.norm(samples, axis=1).std():.4f} uT")
    print(f"\nAdd to reader.py:")
    print(f"HARD_IRON_OFFSET = np.array([{hard_iron_offset[0]:.6f}, {hard_iron_offset[1]:.6f}, {hard_iron_offset[2]:.6f}])")


if __name__ == "__main__":
    main()
