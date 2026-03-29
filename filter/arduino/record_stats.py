import sys
import time

import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider
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

        norms = np.linalg.norm(mags, axis=1, keepdims=True)
        mags_normalized = mags / norms
        print("Magnetometer (normalized) (x, y, z):")
        print(f"  mean: {mags_normalized.mean(axis=0)}")
        print(f"  cov:\n{np.cov(mags_normalized, rowvar=False)}")

        print("Gyroscope [rad/s] (x, y, z):")
        print(f"  mean: {gyros.mean(axis=0)}")
        print(f"  std:  {gyros.std(axis=0)}")

        if n_imu > 1:
            plot_allan_variance(accels, gyros, dt=1 / 100)

    if n_baro > 0:
        pressures = np.array(pressures)

        print("Barometer [Pa]:")
        print(f"  mean: {pressures.mean()}")
        print(f"  std:  {pressures.std()}")


def allan_variance(data: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(data)
    max_cluster = n // 2
    cluster_sizes = np.unique(
        np.logspace(0, np.log10(max_cluster), num=100).astype(int)
    )
    taus = cluster_sizes * dt
    avars = np.empty((len(cluster_sizes), data.shape[1]))
    for i, m in enumerate(cluster_sizes):
        clusters = data[: n - n % m].reshape(-1, m, data.shape[1]).mean(axis=1)
        avars[i] = np.mean(np.diff(clusters, axis=0) ** 2, axis=0) / 2
    return taus, avars


NOISE_TYPES = {
    "white_noise": ("N (white noise)", -0.5, ":"),
    "bias_instability": ("B (bias instability)", 0.0, "-."),
    "random_walk": ("K (random walk)", 0.5, "--"),
}


def fit_slope_line(
    taus: np.ndarray, adevs_1d: np.ndarray, slope: float, tau_range: tuple[float, float]
) -> float:
    """Fit a line with fixed slope on log-log scale within tau_range. Returns coefficient c where adev = c * tau^slope."""
    mask = (taus >= tau_range[0]) & (taus <= tau_range[1])
    log_c = np.mean(np.log(adevs_1d[mask]) - slope * np.log(taus[mask]))
    return np.exp(log_c)


def fit_noise_lines(
    taus: np.ndarray, adevs: np.ndarray, tau_ranges: dict[str, tuple[float, float]]
) -> dict[str, np.ndarray]:
    """Fit white noise (-1/2), bias instability (0), and random walk (+1/2) lines per axis within given τ ranges."""
    coeffs: dict[str, np.ndarray] = {}
    for name, (_, slope, _) in NOISE_TYPES.items():
        coeffs[name] = np.array([
            fit_slope_line(taus, adevs[:, ax], slope, tau_ranges[name])
            for ax in range(adevs.shape[1])
        ])
    return coeffs


def print_noise_params(sensor: str, coeffs: dict[str, np.ndarray]):
    labels = ["x", "y", "z"]
    n_vals = coeffs["white_noise"]            # ADEV at τ=1s (1^-0.5 = 1)
    b_vals = coeffs["bias_instability"]
    k_vals = coeffs["random_walk"] * 3.0 ** 0.5  # ADEV at τ=3s
    print(f"  {sensor}:")
    print(f"    N (white noise, τ=1s):       {', '.join(f'{labels[i]}={n_vals[i]:.6f}' for i in range(3))}")
    print(f"    B (bias instability):         {', '.join(f'{labels[i]}={b_vals[i]:.6f}' for i in range(3))}")
    print(f"    K (random walk, τ=3s):        {', '.join(f'{labels[i]}={k_vals[i]:.6f}' for i in range(3))}")


def plot_allan_variance(accels: np.ndarray, gyros: np.ndarray, dt: float):
    taus_a, avars_a = allan_variance(accels, dt)
    taus_g, avars_g = allan_variance(gyros, dt)
    adevs_a = np.sqrt(avars_a)
    adevs_g = np.sqrt(avars_g)

    tau_min_a, tau_max_a = taus_a[0], taus_a[-1]
    tau_min_g, tau_max_g = taus_g[0], taus_g[-1]
    log_min_a, log_max_a = np.log10(tau_min_a), np.log10(tau_max_a)
    log_min_g, log_max_g = np.log10(tau_min_g), np.log10(tau_max_g)

    default_ranges_a = {
        "white_noise": (tau_min_a, tau_max_a),
        "bias_instability": (tau_min_a, tau_max_a),
        "random_walk": (tau_min_a, tau_max_a),
    }
    default_ranges_g = {
        "white_noise": (tau_min_g, tau_max_g),
        "bias_instability": (tau_min_g, tau_max_g),
        "random_walk": (tau_min_g, tau_max_g),
    }

    coeffs_a = fit_noise_lines(taus_a, adevs_a, default_ranges_a)
    coeffs_g = fit_noise_lines(taus_g, adevs_g, default_ranges_g)

    print("\nAllan Deviation noise parameters:")
    print_noise_params("Accelerometer [m/s²]", coeffs_a)
    print_noise_params("Gyroscope [rad/s]", coeffs_g)

    labels = ["x", "y", "z"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    n_sliders = len(NOISE_TYPES)
    fig = plt.figure(figsize=(14, 5 + n_sliders * 0.7))
    gs = fig.add_gridspec(1 + n_sliders, 2, height_ratios=[5] + [0.5] * n_sliders, hspace=0.4)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_g = fig.add_subplot(gs[0, 1])

    # Plot raw data
    for axis in range(3):
        ax_a.loglog(taus_a, adevs_a[:, axis], color=colors[axis], alpha=0.5, label=labels[axis])
        ax_g.loglog(taus_g, adevs_g[:, axis], color=colors[axis], alpha=0.5, label=labels[axis])

    # Plot fit lines (store references for updating)
    fit_lines_a: dict[str, list[plt.Line2D]] = {}
    fit_lines_g: dict[str, list[plt.Line2D]] = {}
    for key, (_, slope, ls) in NOISE_TYPES.items():
        fit_lines_a[key] = []
        fit_lines_g[key] = []
        for axis in range(3):
            line_a, = ax_a.loglog(taus_a, coeffs_a[key][axis] * taus_a ** slope, color=colors[axis], linestyle=ls, alpha=0.7)
            line_g, = ax_g.loglog(taus_g, coeffs_g[key][axis] * taus_g ** slope, color=colors[axis], linestyle=ls, alpha=0.7)
            fit_lines_a[key].append(line_a)
            fit_lines_g[key].append(line_g)

    for ax, title in [(ax_a, "Accelerometer"), (ax_g, "Gyroscope")]:
        ax.set_xlabel("τ (s)")
        ax.set_ylabel("Allan Deviation")
        ax.set_title(f"{title} Allan Deviation")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

    # Create range sliders for each noise type
    sliders: list[tuple[str, RangeSlider, RangeSlider]] = []
    for i, (key, (desc, _, _)) in enumerate(NOISE_TYPES.items()):
        ax_sl_a = fig.add_subplot(gs[1 + i, 0])
        ax_sl_g = fig.add_subplot(gs[1 + i, 1])
        slider_a = RangeSlider(ax_sl_a, f"{desc}", log_min_a, log_max_a, valinit=(log_min_a, log_max_a))
        slider_g = RangeSlider(ax_sl_g, f"{desc}", log_min_g, log_max_g, valinit=(log_min_g, log_max_g))
        sliders.append((key, slider_a, slider_g))

    def on_slider_changed(_):
        for key, sl_a, sl_g in sliders:
            _, slope, _ = NOISE_TYPES[key]
            tau_range_a = (10 ** sl_a.val[0], 10 ** sl_a.val[1])
            tau_range_g = (10 ** sl_g.val[0], 10 ** sl_g.val[1])
            for axis in range(3):
                c_a = fit_slope_line(taus_a, adevs_a[:, axis], slope, tau_range_a)
                c_g = fit_slope_line(taus_g, adevs_g[:, axis], slope, tau_range_g)
                fit_lines_a[key][axis].set_ydata(c_a * taus_a ** slope)
                fit_lines_g[key][axis].set_ydata(c_g * taus_g ** slope)
        fig.canvas.draw_idle()

    for _, sl_a, sl_g in sliders:
        sl_a.on_changed(on_slider_changed)
        sl_g.on_changed(on_slider_changed)

    plt.show()


if __name__ == "__main__":
    main()
