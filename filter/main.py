# %%
from math import atan2
from typing import cast
from filterpy.kalman import ExtendedKalmanFilter
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.linalg import block_diag
from scipy.stats import chi2


def skew_symmetric(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class FlightFilter:
    def __init__(
        self, x_nom, P, sigma_a_noise, sigma_w_noise, sigma_a_walk, sigma_w_walk, g
    ):
        self.f = ExtendedKalmanFilter(dim_x=3 * 5, dim_z=3, dim_u=3 * 2)
        self.f.x = np.zeros(3 * 5)
        self.f.P = P
        self.sigma_a_noise = sigma_a_noise
        self.sigma_w_noise = sigma_w_noise
        self.sigma_a_walk = sigma_a_walk
        self.sigma_w_walk = sigma_w_walk
        self.x_nom = x_nom
        self.g = g

    def get_rotation_matrix(self):
        q = self.x_nom[6:10]
        return Rotation.from_quat(q, scalar_first=True).as_matrix()

    def get_F_x(self, u, dt):
        a_m = u[0:3]
        w_m = u[3:6]
        a_b = self.x_nom[10:13]
        w_b = self.x_nom[13:16]
        R = self.get_rotation_matrix()
        F_x = np.eye(3 * 5)
        F_x[0:3, 3:6] = np.eye(3) * dt

        F_x[3:6, 6:9] = -R @ skew_symmetric(a_m - a_b) * dt
        F_x[3:6, 9:12] = -R * dt

        F_x[6:9, 6:9] = Rotation.from_rotvec((w_m - w_b) * dt).as_matrix().T
        F_x[6:9, 12:15] = -np.eye(3) * dt
        return F_x

    def predict(self, u, dt):
        V_i = np.eye(3) * (self.sigma_a_noise**2) * dt**2
        Theta_i = np.eye(3) * (self.sigma_w_noise**2) * dt**2
        A_i = np.eye(3) * (self.sigma_a_walk**2) * dt
        Omega_i = np.eye(3) * (self.sigma_w_walk**2) * dt
        F_i = np.vstack(
            (
                np.zeros((3, 3 * 4)),
                np.eye(3 * 4),
            )
        )
        Q_i = cast(np.ndarray, block_diag(V_i, Theta_i, A_i, Omega_i))
        self.f.Q = F_i @ Q_i @ F_i.T
        F_x = self.get_F_x(u, dt)

        # update error state: x, P
        # no update to x, since it's the error state with mean of 0 always
        self.f.P = F_x @ self.f.P @ F_x.T + self.f.Q

        # update nominal state: x_nom
        # euler's approximation
        R = self.get_rotation_matrix()
        a_m = u[0:3]
        w_m = u[3:6]
        a_b = self.x_nom[10:13]
        w_b = self.x_nom[13:16]
        g = self.g
        v = self.x_nom[3:6]
        self.x_nom[0:3] += v * dt + 0.5 * (R @ (a_m - a_b) + g) * dt**2
        self.x_nom[3:6] += (R @ (a_m - a_b) + g) * dt
        self.x_nom[6:10] = (
            Rotation.from_quat(self.x_nom[6:10], scalar_first=True)
            * Rotation.from_rotvec((w_m - w_b) * dt)
        ).as_quat(scalar_first=True)

    def get_X_dx(self):
        qw, qx, qy, qz = self.x_nom[6:10]
        Q_dtheta = 0.5 * np.array(
            [[-qx, -qy, -qz], [qw, -qz, qy], [qz, qw, -qx], [-qy, qx, qw]]
        )
        X_dx = np.zeros((16, 15))
        X_dx[0:6, 0:6] = np.eye(6)
        X_dx[6:10, 6:9] = Q_dtheta
        X_dx[10:16, 9:15] = np.eye(6)
        return X_dx

    def update(self, h, z, R, H_x, gating_threshold=0.997):
        H = H_x @ self.get_X_dx()
        y = z - h(self.x_nom)
        S = H @ self.f.P @ H.T + R

        # dof of the measurement
        k = len(y)

        K = self.f.P @ H.T @ np.linalg.inv(S)
        log_likelihood = -0.5 * (
            k * np.log(2 * np.pi) + np.log(np.linalg.det(S)) + y @ np.linalg.solve(S, y)
        )

        # mahalanobis outlier rejection
        # D^2 = y^T * S^-1 * y
        # We use np.linalg.solve(S, y) instead of inv(S) @ y for better numerical stability
        D2 = y.T @ np.linalg.solve(S, y)
        chi2_threshold = chi2.ppf(gating_threshold, k)
        if D2 > chi2_threshold:
            # reject measurement
            print(
                f"Rejected measurement with D2 = {D2}, DOF={k}, Threshold = {chi2_threshold}."
            )
            return log_likelihood, False

        # update error state: x, P
        self.f.x = self.f.x + K @ y
        A = np.eye(15) - K @ H
        self.f.P = A @ self.f.P @ A.T + K @ R @ K.T

        # inject error state into nominal state
        self.x_nom[0:6] += self.f.x[0:6]
        self.x_nom[6:10] = (
            Rotation.from_quat(self.x_nom[6:10], scalar_first=True)
            * Rotation.from_rotvec(self.f.x[6:9])
        ).as_quat(scalar_first=True)
        self.x_nom[10:16] += self.f.x[9:15]

        # reset error state to 0, adjust x, P to account for injection
        G = np.eye(15)
        G[6:9, 6:9] -= skew_symmetric(0.5 * self.f.x[6:9])
        self.f.P = G @ self.f.P @ G.T
        self.f.x = np.zeros(15)

        return log_likelihood, True


# %%
import numpy as np
from numpy.random import randn
from scipy.spatial.transform import Rotation


class RocketSim:
    def __init__(
        self,
        pos_0,
        vel_0,
        omega_0,
        quat_0: Rotation,
        a_b_0,
        w_b_0,
        m,
        g,
        north,
        I,
        f_thrust_t,
        f_torque_t,
        sigma_a_noise,
        sigma_w_noise,
        sigma_a_walk,
        sigma_w_walk,
        Cd_axial,
        Cd_lateral,
        A_ref,
        A_lateral,
        rho,
        Cd_roll,
        Cd_pitch,
        sigma_force,
        sigma_torque,
        sigma_gps,
        sigma_altimeter,
        sigma_magnetometer,
    ):
        self.I = I
        self.I_inv = np.linalg.inv(I)
        self.m = m
        self.g = g
        self.north = north
        self.pos = pos_0.copy()
        self.vel = vel_0.copy()
        self.quat = quat_0
        self.omega = omega_0.copy()
        self.a_b = a_b_0.copy()
        self.w_b = w_b_0.copy()
        self.f_thrust_t = f_thrust_t
        self.f_torque_t = f_torque_t
        # IMU noise
        self.sigma_a_noise = sigma_a_noise
        self.sigma_w_noise = sigma_w_noise
        self.sigma_a_walk = sigma_a_walk
        self.sigma_w_walk = sigma_w_walk
        # Aero drag
        self.Cd_axial = Cd_axial
        self.Cd_lateral = Cd_lateral
        self.A_ref = A_ref
        self.A_lateral = A_lateral
        self.rho = rho
        # Aero rotational damping
        self.Cd_roll = Cd_roll
        self.Cd_pitch = Cd_pitch
        # Process noise
        self.sigma_force = sigma_force
        self.sigma_torque = sigma_torque
        self.sigma_gps = sigma_gps
        self.sigma_altimeter = sigma_altimeter
        self.sigma_magnetometer = sigma_magnetometer

    def _compute_aero_force(self):
        """Compute aerodynamic drag force in body frame.

        Drag is split into axial (along body z) and lateral components
        because a rocket has very different drag profiles nose-on vs sideways.
        """
        vel_body = self.quat.inv().apply(self.vel)
        speed = np.linalg.norm(vel_body)
        if speed < 1e-6:
            return np.zeros(3)

        # Decompose into axial (along body z) and lateral
        v_axial = vel_body[2]
        v_lateral = vel_body[:2]
        speed_lateral = np.linalg.norm(v_lateral)

        # Axial drag: opposes axial velocity
        F_axial = (
            -np.sign(v_axial) * 0.5 * self.rho * v_axial**2 * self.Cd_axial * self.A_ref
        )

        # Lateral drag: opposes lateral velocity
        if speed_lateral > 1e-6:
            drag_lat_mag = (
                0.5 * self.rho * speed_lateral**2 * self.Cd_lateral * self.A_lateral
            )
            F_lateral = -v_lateral / speed_lateral * drag_lat_mag
        else:
            F_lateral = np.zeros(2)

        return np.array([F_lateral[0], F_lateral[1], F_axial])

    def _compute_aero_damping_torque(self):
        """Compute aerodynamic rotational damping torque in body frame.

        Opposes angular rates — models the fact that a spinning/tumbling rocket
        in air experiences restoring torques from aerodynamic pressure.
        """
        return -np.array(
            [
                self.Cd_pitch * self.omega[0],  # pitch damping
                self.Cd_pitch * self.omega[1],  # yaw damping
                self.Cd_roll * self.omega[2],  # roll damping
            ]
        )

    def step(self, t, dt, grounded=False):
        # --- Translational dynamics (world frame) ---
        thrust_world = self.quat.apply(self.f_thrust_t(t))
        drag_body = self._compute_aero_force()
        drag_world = self.quat.apply(drag_body)
        force_noise_world = randn(3) * self.sigma_force

        acc = (thrust_world + drag_world + force_noise_world) / self.m + self.g
        if grounded:
            acc = np.zeros(3)
        self.pos += self.vel * dt + 0.5 * acc * dt**2
        self.vel += acc * dt

        # --- Rotational dynamics (body frame) ---
        torques = self.f_torque_t(t)
        aero_damping = self._compute_aero_damping_torque()
        torque_noise = randn(3) * self.sigma_torque

        self.quat = self.quat * Rotation.from_rotvec(self.omega * dt)
        omega_dot = self.I_inv @ (
            torques
            + aero_damping
            + torque_noise
            - np.cross(self.omega, self.I @ self.omega)
        )
        self.omega += omega_dot * dt

        # --- Bias random walk ---
        self.a_b += randn(3) * self.sigma_a_walk * np.sqrt(dt)
        self.w_b += randn(3) * self.sigma_w_walk * np.sqrt(dt)

    def get_imu_reading(self, t):
        """Generate noisy IMU measurement from true state (eqs 231-232)."""
        thrust_body = self.f_thrust_t(t) / self.m
        drag_body = self._compute_aero_force() / self.m
        gravity_body = self.quat.inv().apply(self.g)
        gravity_body_experienced = gravity_body if t > START_TIME else np.zeros(3)
        a_true = thrust_body + drag_body + gravity_body_experienced - gravity_body

        w_true = self.omega

        a_m = a_true + self.a_b + randn(3) * self.sigma_a_noise
        w_m = w_true + self.w_b + randn(3) * self.sigma_w_noise
        return a_m, w_m

    def get_gps_reading(self):
        """GPS: world-frame position + noise."""
        return self.pos + randn(3) * self.sigma_gps

    def get_altimeter_reading(self):
        """Barometric altimeter: z-position + noise."""
        return self.pos[2] + randn() * self.sigma_altimeter

    def get_magnetometer_reading(self):
        """Magnetometer: world north expressed in body frame + noise."""
        mag_body = self.quat.inv().apply(self.north)
        return mag_body + randn(3) * self.sigma_magnetometer


# %%
# --- Physical constants ---
M = 15.0  # mass [kg]
G = np.array([0, 0, -9.81])  # gravity vector [m/s^2]
NORTH = np.array([1, 0, 0])  # magnetic north reference (world frame)

# --- Inertia tensor (body frame: z = rocket long axis) ---
I_PITCH = 2.0  # about body x [kg·m^2]
I_YAW = 2.0  # about body y [kg·m^2]
I_ROLL = 0.08  # about body z (long axis) [kg·m^2]
I_BODY = np.diag([I_PITCH, I_YAW, I_ROLL])

# --- Thrust ---
START_TIME = 7.5  # [s]
BURN_TIME = 8.5  # [s]
THRUST_MAG = 600.0  # [N]

# --- Launch geometry ---
LAUNCH_ANGLE = 15.0  # degrees off vertical toward +x

# --- Simulation timing ---
DT = 0.001  # integration timestep [s] (1 kHz)
T_MAX = 40.0  # max sim time [s]

# --- IMU noise (continuous-time specifications) ---
SIGMA_ACCEL_NOISE = 0.5  # accelerometer white noise [m/s^2]
SIGMA_GYRO_NOISE = 0.01  # gyroscope white noise [rad/s]
SIGMA_ACCEL_WALK = 0.001  # accelerometer bias random walk [m/s^2/√s]
SIGMA_GYRO_WALK = 0.0001  # gyroscope bias random walk [rad/s/√s]

# --- Aerodynamic drag ---
CD_AXIAL = 0.3  # axial drag coefficient (nose-on)
CD_LATERAL = 1.2  # lateral drag coefficient (broadside)
A_REF = 0.01  # axial reference area [m^2]
A_LATERAL = 0.15  # lateral reference area [m^2]
RHO = 1.225  # air density [kg/m^3]

# --- Aerodynamic rotational damping ---
CD_ROLL_DAMP = 0.001  # roll damping [N·m·s/rad]
CD_PITCH_DAMP = 0.05  # pitch/yaw damping [N·m·s/rad]

# --- Process noise (unmodeled disturbances in true dynamics) ---
SIGMA_FORCE_NOISE = 0.5  # translational disturbance [N]
SIGMA_TORQUE_NOISE = 0.01  # rotational disturbance [N·m]

# --- Sensor noise for observation models ---
SIGMA_GPS = np.array([3, 3, 100])  # GPS position noise [m]
SIGMA_ALTIMETER = 0.5  # altimeter noise [m]
SIGMA_MAGNETOMETER = 0.05  # magnetometer noise [normalized]

GPS_INTERVAL = 300
ALTIMETER_INTERVAL = 100
MAGNETOMETER_INTERVAL = 234

# %%
# ============================================================
# Scenario: Angled launch with canard-induced spin
# ============================================================


def run_simulation():
    # Initial conditions
    quat_0 = Rotation.from_euler("y", LAUNCH_ANGLE, degrees=True)
    pos_0 = np.array([0.0, 0.0, 0.5])
    vel_0 = np.zeros(3)
    omega_0 = np.zeros(3)
    a_b_0 = np.zeros(3)
    w_b_0 = np.zeros(3)
    print("quat_0: ", quat_0.as_quat(scalar_first=True))
    print("pos_0: ", pos_0)

    # Thrust (body z-axis)
    def f_thrust_t(t):
        if t < START_TIME:
            return np.zeros(3)
        if t < BURN_TIME:
            return np.array([0, 0, THRUST_MAG])
        return np.zeros(3)

    # Torque from canards
    def f_torque_t(t):
        if t < START_TIME:
            return np.zeros(3)
        if t < BURN_TIME:
            roll = 0.06 + 0.02 * np.sin(4.0 * t)
            pitch = 0.3 * np.sin(2.0 * t)
            yaw = -0.2 * np.cos(1.5 * t)
        elif t < BURN_TIME + 5.0:
            decay = np.exp(-(t - BURN_TIME) * 0.8)
            roll = 0.08 * decay
            pitch = 0.05 * decay * np.sin(3 * t)
            yaw = 0.03 * decay * np.cos(2 * t)
        else:
            roll = 0.0
            pitch = 0.0
            yaw = 0.0
        return np.array([pitch, yaw, roll])

    # Create sim
    sim = RocketSim(
        pos_0=pos_0,
        vel_0=vel_0,
        omega_0=omega_0,
        quat_0=quat_0,
        a_b_0=a_b_0,
        w_b_0=w_b_0,
        m=M,
        g=G,
        north=NORTH,
        I=I_BODY,
        f_thrust_t=f_thrust_t,
        f_torque_t=f_torque_t,
        sigma_a_noise=SIGMA_ACCEL_NOISE,
        sigma_w_noise=SIGMA_GYRO_NOISE,
        sigma_a_walk=SIGMA_ACCEL_WALK,
        sigma_w_walk=SIGMA_GYRO_WALK,
        A_lateral=A_LATERAL,
        A_ref=A_REF,
        Cd_axial=CD_AXIAL,
        Cd_lateral=CD_LATERAL,
        Cd_roll=CD_ROLL_DAMP,
        Cd_pitch=CD_PITCH_DAMP,
        rho=RHO,
        sigma_force=SIGMA_FORCE_NOISE,
        sigma_torque=SIGMA_TORQUE_NOISE,
        sigma_gps=SIGMA_GPS,
        sigma_altimeter=SIGMA_ALTIMETER,
        sigma_magnetometer=SIGMA_MAGNETOMETER,
    )

    def h_gps(x_nom):
        return x_nom[0:3]

    R_gps = np.diag(SIGMA_GPS) ** 2
    H_x_gps = np.zeros((3, 16))
    H_x_gps[0:3, 0:3] = np.eye(3)

    def h_altimeter(x_nom):
        return np.array([x_nom[2]])

    R_altimeter = np.array([[SIGMA_ALTIMETER**2]])
    H_x_altimeter = np.zeros((1, 16))
    H_x_altimeter[0, 2] = 1.0

    def h_magnetometer(x_nom):
        world_orientation = Rotation.from_quat(x_nom[6:10], scalar_first=True)
        return world_orientation.inv().apply(NORTH)

    R_magnetometer = np.eye(3) * SIGMA_MAGNETOMETER**2

    # Run
    n_steps = int(T_MAX / DT)
    downsample = 10
    n_stored = n_steps // downsample

    times = np.zeros(n_stored)
    positions = np.zeros((n_stored, 3))
    velocities = np.zeros((n_stored, 3))
    omegas = np.zeros((n_stored, 3))
    eulers = np.zeros((n_stored, 3))
    quats = np.zeros((n_stored, 4))
    body_z_world = np.zeros((n_stored, 3))

    kf_positions = np.zeros((n_stored, 3))
    kf_velocities = np.zeros((n_stored, 3))
    kf_eulers = np.zeros((n_stored, 3))
    kf_P_pos = np.zeros((n_stored, 3))
    kf_P_vel = np.zeros((n_stored, 3))
    kf_P_att = np.zeros((n_stored, 3))
    kf_P_pos_full = np.zeros((n_stored, 3, 3))
    kf_P_full = np.zeros((n_stored, 15, 15))
    kf_x_nom_full = np.zeros((n_stored, 16))
    true_a_b = np.zeros((n_stored, 3))
    true_w_b = np.zeros((n_stored, 3))

    gps_log_likelihoods = []
    gps_ll_times = []
    alt_log_likelihoods = []
    alt_ll_times = []
    mag_log_likelihoods = []
    mag_ll_times = []

    first_states_logged = False

    # initialize the filter
    x_nom = np.zeros(16)
    P = np.eye(15) * 500
    P[9:12, 9:12] = np.eye(3) * SIGMA_ACCEL_WALK**2 * 10
    P[12:15, 12:15] = np.eye(3) * SIGMA_GYRO_WALK**2 * 10
    gps = sim.get_gps_reading()
    x_nom[0:3] = gps
    P[0:3, 0:3] = R_gps
    a_m, w_m = sim.get_imu_reading(0)
    m_m = sim.get_magnetometer_reading()
    q, R_cov = get_orientation_and_covariance(
        a_m,
        m_m,
        np.ones(3) * SIGMA_ACCEL_NOISE,
        np.ones(3) * SIGMA_MAGNETOMETER,
    )
    x_nom[6:10] = q
    P[6:9, 6:9] = R_cov
    kf = FlightFilter(
        x_nom=x_nom,
        P=P,
        sigma_a_noise=SIGMA_ACCEL_NOISE,
        sigma_w_noise=SIGMA_GYRO_NOISE,
        sigma_a_walk=SIGMA_ACCEL_WALK,
        sigma_w_walk=SIGMA_GYRO_WALK,
        g=G,
    )

    store_idx = 0
    for i in range(n_steps):
        t = i * DT
        if not first_states_logged:
            np.set_printoptions(precision=3)
            print("time: ", t)
            print("nominal state: ", kf.x_nom)
            print("covariance in position: ", kf.f.P[0:3, 0:3])
            print("covariance in velocity: ", kf.f.P[3:6, 3:6])
            print("covariance in orientation: ", kf.f.P[6:9, 6:9])
            print("covariance in bias in acceleration: ", kf.f.P[10:13, 10:13])
            print("covariance in bias in gyro: ", kf.f.P[13:16, 13:16])
            first_states_logged = True
        a_m, w_m = sim.get_imu_reading(t)
        kf.predict(np.concatenate((a_m, w_m)), DT)

        if i % GPS_INTERVAL == 0:
            z = sim.get_gps_reading()
            ll, accepted = kf.update(h_gps, z, R_gps, H_x_gps)
            gps_log_likelihoods.append(ll)
            gps_ll_times.append(t)
            if not accepted:
                print("GPS measurement rejected at time: ", t)
                print("measurement: ", z)

        if i % ALTIMETER_INTERVAL == 0:
            z = np.array([sim.get_altimeter_reading()])
            ll, accepted = kf.update(h_altimeter, z, R_altimeter, H_x_altimeter)
            alt_log_likelihoods.append(ll)
            alt_ll_times.append(t)
            if not accepted:
                print("Altitude measurement rejected at time: ", t)
                print("measurement: ", z)

        if i % MAGNETOMETER_INTERVAL == 0:
            z = sim.get_magnetometer_reading()
            z = z / np.linalg.norm(z)
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
            mag_log_likelihoods.append(ll)
            mag_ll_times.append(t)
            if not accepted:
                print("Magnetometer measurement rejected at time: ", t)
                print("measurement: ", z)

        if i % downsample == 0:
            times[store_idx] = t
            positions[store_idx] = sim.pos.copy()
            velocities[store_idx] = sim.vel.copy()
            omegas[store_idx] = sim.omega.copy()
            eulers[store_idx] = sim.quat.as_euler("xyz", degrees=True)
            quats[store_idx] = sim.quat.as_quat(scalar_first=True)
            body_z_world[store_idx] = sim.quat.apply(np.array([0, 0, 1]))

            kf_positions[store_idx] = kf.x_nom[0:3]
            kf_velocities[store_idx] = kf.x_nom[3:6]
            kf_eulers[store_idx] = Rotation.from_quat(
                kf.x_nom[6:10], scalar_first=True
            ).as_euler("xyz", degrees=True)
            P_diag = np.diag(kf.f.P)
            kf_P_pos[store_idx] = P_diag[0:3]
            kf_P_vel[store_idx] = P_diag[3:6]
            kf_P_att[store_idx] = P_diag[6:9]
            kf_P_pos_full[store_idx] = kf.f.P[0:3, 0:3]
            kf_P_full[store_idx] = kf.f.P.copy()
            kf_x_nom_full[store_idx] = kf.x_nom.copy()
            true_a_b[store_idx] = sim.a_b.copy()
            true_w_b[store_idx] = sim.w_b.copy()
            store_idx += 1

        if sim.pos[2] < 0 and t > BURN_TIME:
            times = times[:store_idx]
            positions = positions[:store_idx]
            velocities = velocities[:store_idx]
            omegas = omegas[:store_idx]
            eulers = eulers[:store_idx]
            quats = quats[:store_idx]
            body_z_world = body_z_world[:store_idx]
            kf_positions = kf_positions[:store_idx]
            kf_velocities = kf_velocities[:store_idx]
            kf_eulers = kf_eulers[:store_idx]
            kf_P_pos = kf_P_pos[:store_idx]
            kf_P_vel = kf_P_vel[:store_idx]
            kf_P_att = kf_P_att[:store_idx]
            kf_P_pos_full = kf_P_pos_full[:store_idx]
            kf_P_full = kf_P_full[:store_idx]
            kf_x_nom_full = kf_x_nom_full[:store_idx]
            true_a_b = true_a_b[:store_idx]
            true_w_b = true_w_b[:store_idx]
            break

        sim.step(t, DT, grounded=t < START_TIME)

    return {
        "times": times,
        "positions": positions,
        "velocities": velocities,
        "omegas": omegas,
        "eulers": eulers,
        "quats": quats,
        "body_z_world": body_z_world,
        "burn_time": BURN_TIME,
        "start_time": START_TIME,
        "kf_positions": kf_positions,
        "kf_velocities": kf_velocities,
        "kf_eulers": kf_eulers,
        "kf_P_pos": kf_P_pos,
        "kf_P_vel": kf_P_vel,
        "kf_P_att": kf_P_att,
        "kf_P_pos_full": kf_P_pos_full,
        "gps_log_likelihoods": np.array(gps_log_likelihoods),
        "gps_ll_times": np.array(gps_ll_times),
        "alt_log_likelihoods": np.array(alt_log_likelihoods),
        "alt_ll_times": np.array(alt_ll_times),
        "mag_log_likelihoods": np.array(mag_log_likelihoods),
        "mag_ll_times": np.array(mag_ll_times),
        "kf_P_full": kf_P_full,
        "kf_x_nom_full": kf_x_nom_full,
        "true_a_b": true_a_b,
        "true_w_b": true_w_b,
    }


import numpy as np
from scipy.spatial.transform import Rotation


def get_orientation_and_covariance(a_m, m_m, sigma_a, sigma_m):
    a_m = np.asarray(a_m, dtype=float)
    m_m = np.asarray(m_m, dtype=float)

    # Enforce sigmas as 3-element arrays
    sigma_a = np.asarray(sigma_a, dtype=float).flatten()
    sigma_m = np.asarray(sigma_m, dtype=float).flatten()

    # Create the measured orthogonal basis (Body Frame)
    v1_b = a_m / np.linalg.norm(a_m)

    v2_b = np.cross(v1_b, m_m)
    v2_b = v2_b / np.linalg.norm(v2_b)

    v3_b = np.cross(v1_b, v2_b)

    # Body triad matrix: [v1_b | v2_b | v3_b]
    M_B = np.column_stack((v1_b, v2_b, v3_b))

    # reference vectors
    w_a = -G
    w_m = NORTH

    # reference orthogonal basis
    v1_w = w_a
    v2_w = np.cross(v1_w, w_m)
    v2_w = v2_w / np.linalg.norm(v2_w)
    v3_w = np.cross(v1_w, v2_w)

    # World triad matrix: [v1_w | v2_w | v3_w]
    M_W = np.column_stack((v1_w, v2_w, v3_w))

    # Compute Rotation Matrix: R_mat maps body vectors to world vectors
    # M_W = R_mat * M_B  =>  R_mat = M_W * M_B^T
    R_mat = M_W @ M_B.T

    # Convert to scalar-first quaternion
    q = Rotation.from_matrix(R_mat).as_quat(scalar_first=True)

    # ==========================================
    # Numerical Jacobian for Covariance (R_cov)
    # ==========================================
    sensors = np.concatenate((a_m, m_m))

    def compute_euler(s):
        ax, ay, az = s[0], s[1], s[2]
        mx, my, mz = s[3], s[4], s[5]

        # Euler proxy to map local sensor sensitivities to 3-DOF rotational errors
        phi = np.arctan2(ay, az)
        theta = np.arctan2(-ax, np.sqrt(ay**2 + az**2))

        mx_prime = (
            mx * np.cos(theta)
            + my * np.sin(phi) * np.sin(theta)
            + mz * np.cos(phi) * np.sin(theta)
        )
        my_prime = my * np.cos(phi) - mz * np.sin(phi)
        psi = np.arctan2(-my_prime, mx_prime)

        return np.array([phi, theta, psi])

    J = np.zeros((3, 6))
    epsilon = 1e-5

    for i in range(6):
        s_plus = sensors.copy()
        s_minus = sensors.copy()

        s_plus[i] += epsilon
        s_minus[i] -= epsilon

        euler_plus = compute_euler(s_plus)
        euler_minus = compute_euler(s_minus)

        diff = euler_plus - euler_minus
        # Handle phase wrapping across boundaries
        diff = (diff + np.pi) % (2 * np.pi) - np.pi

        J[:, i] = diff / (2.0 * epsilon)

    # Build the 6x6 diagonal sensor noise covariance matrix
    Sigma_x = np.diag(np.concatenate((sigma_a**2, sigma_m**2)))

    # Calculate the 3x3 covariance matrix
    R_cov = J @ Sigma_x @ J.T

    return q, R_cov


# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# Interactive Plotly Dashboard
# ============================================================


def plot_results(data):
    times = data["times"]
    pos = data["positions"]
    vel = data["velocities"]
    omega = data["omegas"]
    euler = data["eulers"]
    quats = data["quats"]
    bz = data["body_z_world"]
    burn_time = data["burn_time"]
    start_time = data["start_time"]
    kf_pos = data["kf_positions"]
    kf_vel = data["kf_velocities"]
    sigma2_pos = 2 * np.sqrt(np.maximum(data["kf_P_pos"], 0))
    sigma2_vel = 2 * np.sqrt(np.maximum(data["kf_P_vel"], 0))
    sigma2_att = 2 * np.sqrt(np.maximum(data["kf_P_att"], 0))
    kf_P_pos_full = data["kf_P_pos_full"]

    apogee_idx = np.argmax(pos[:, 2])
    apogee_time = times[apogee_idx]
    apogee_alt = pos[apogee_idx, 2]

    ci_fill_color = [
        "rgba(255,107,107,0.15)",
        "rgba(81,207,102,0.15)",
        "rgba(51,154,240,0.15)",
    ]
    ci_line_color = [
        "rgba(255,107,107,0.4)",
        "rgba(81,207,102,0.4)",
        "rgba(51,154,240,0.4)",
    ]

    # ======== FIGURE 1: 3D Trajectory ========
    fig_3d = go.Figure()

    fig_3d.add_trace(
        go.Scatter3d(
            x=pos[:, 0],
            y=pos[:, 1],
            z=pos[:, 2],
            mode="lines",
            line=dict(
                color=times,
                colorscale="Inferno",
                width=4,
                colorbar=dict(title="Time (s)", x=1.0),
            ),
            name="Ground Truth",
            hovertemplate="x: %{x:.1f}m<br>y: %{y:.1f}m<br>z: %{z:.1f}m<extra></extra>",
        )
    )

    fig_3d.add_trace(
        go.Scatter3d(
            x=kf_pos[:, 0],
            y=kf_pos[:, 1],
            z=kf_pos[:, 2],
            mode="lines",
            line=dict(color="rgba(0, 255, 180, 0.8)", width=3, dash="dash"),
            name="Filter Estimate",
            hovertemplate="x: %{x:.1f}m<br>y: %{y:.1f}m<br>z: %{z:.1f}m<extra></extra>",
        )
    )

    ellipse_spacing = max(1, len(times) // 25)
    n_ring = 48
    theta = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    for idx in range(0, len(times), ellipse_spacing):
        P3 = kf_P_pos_full[idx]
        eigvals, eigvecs = np.linalg.eigh(P3)
        radii = 2 * np.sqrt(np.maximum(eigvals, 0))
        c = kf_pos[idx]
        for plane, plane_color in [
            ((0, 1), "rgba(255, 200, 50, 0.25)"),
            ((0, 2), "rgba(50, 200, 255, 0.25)"),
            ((1, 2), "rgba(255, 100, 200, 0.25)"),
        ]:
            circle = np.zeros((n_ring, 3))
            circle[:, plane[0]] = np.cos(theta)
            circle[:, plane[1]] = np.sin(theta)
            pts = circle * radii @ eigvecs.T + c
            fig_3d.add_trace(
                go.Scatter3d(
                    x=np.append(pts[:, 0], pts[0, 0]),
                    y=np.append(pts[:, 1], pts[0, 1]),
                    z=np.append(pts[:, 2], pts[0, 2]),
                    mode="lines",
                    line=dict(color=plane_color, width=2),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    arrow_spacing = max(1, len(times) // 40)
    arrow_scale = apogee_alt * 0.04
    for i in range(0, len(times), arrow_spacing):
        fig_3d.add_trace(
            go.Scatter3d(
                x=[pos[i, 0], pos[i, 0] + bz[i, 0] * arrow_scale],
                y=[pos[i, 1], pos[i, 1] + bz[i, 1] * arrow_scale],
                z=[pos[i, 2], pos[i, 2] + bz[i, 2] * arrow_scale],
                mode="lines",
                line=dict(color="rgba(0, 200, 255, 0.5)", width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig_3d.add_trace(
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode="markers",
            marker=dict(size=6, color="lime", symbol="diamond"),
            name="Launch",
        )
    )

    fig_3d.add_trace(
        go.Scatter3d(
            x=[pos[apogee_idx, 0]],
            y=[pos[apogee_idx, 1]],
            z=[pos[apogee_idx, 2]],
            mode="markers+text",
            marker=dict(size=6, color="red", symbol="x"),
            text=[f"Apogee: {apogee_alt:.0f}m @ {apogee_time:.1f}s"],
            textposition="top center",
            textfont=dict(size=10, color="red"),
            name="Apogee",
        )
    )

    gt_mid = (pos.max(axis=0) + pos.min(axis=0)) / 2
    gt_range = pos.max(axis=0) - pos.min(axis=0)
    half_extent = gt_range.max() / 2 * 1.2
    axis_ranges = [[gt_mid[i] - half_extent, gt_mid[i] + half_extent] for i in range(3)]

    fig_3d.update_layout(
        title=dict(
            text="Rocket Trajectory — Ground Truth vs Filter (2σ ellipses)",
            font=dict(size=18),
        ),
        scene=dict(
            xaxis_title="Downrange X (m)",
            yaxis_title="Crossrange Y (m)",
            zaxis_title="Altitude (m)",
            aspectmode="data",
            bgcolor="rgb(15, 15, 25)",
            xaxis=dict(
                gridcolor="rgba(255,255,255,0.1)",
                zerolinecolor="rgba(255,255,255,0.2)",
                range=axis_ranges[0],
            ),
            yaxis=dict(
                gridcolor="rgba(255,255,255,0.1)",
                zerolinecolor="rgba(255,255,255,0.2)",
                range=axis_ranges[1],
            ),
            zaxis=dict(
                gridcolor="rgba(255,255,255,0.1)",
                zerolinecolor="rgba(255,255,255,0.2)",
                range=axis_ranges[2],
            ),
        ),
        paper_bgcolor="rgb(20, 20, 35)",
        font=dict(color="white"),
        width=900,
        height=700,
        margin=dict(l=0, r=0, t=50, b=0),
    )

    # ======== FIGURE 2: State time series ========
    fig_ts = make_subplots(
        rows=4,
        cols=2,
        subplot_titles=(
            "Position (World)",
            "Velocity (World)",
            "Angular Rates (Body)",
            "Euler Angles",
            "Quaternion",
            "Altitude vs Downrange",
            "Speed",
            "Total Angular Rate",
        ),
        vertical_spacing=0.06,
        horizontal_spacing=0.08,
    )

    axis_colors = ["#ff6b6b", "#51cf66", "#339af0"]
    axis_names = ["x", "y", "z"]

    for i in range(3):
        fig_ts.add_trace(
            go.Scatter(
                x=times,
                y=pos[:, i],
                name=f"p_{axis_names[i]}",
                line=dict(color=axis_colors[i], width=1.5),
                legendgroup="pos",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig_ts.add_trace(
            go.Scatter(
                x=times,
                y=kf_pos[:, i] + sigma2_pos[:, i],
                mode="lines",
                line=dict(width=0),
                legendgroup="pos",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        fig_ts.add_trace(
            go.Scatter(
                x=times,
                y=kf_pos[:, i] - sigma2_pos[:, i],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=ci_fill_color[i],
                legendgroup="pos",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        fig_ts.add_trace(
            go.Scatter(
                x=times,
                y=kf_pos[:, i],
                name=f"kf_p_{axis_names[i]}",
                line=dict(color=axis_colors[i], width=1, dash="dash"),
                legendgroup="pos",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    for i in range(3):
        fig_ts.add_trace(
            go.Scatter(
                x=times,
                y=vel[:, i],
                name=f"v_{axis_names[i]}",
                line=dict(color=axis_colors[i], width=1.5, dash="dot"),
                legendgroup="vel",
                showlegend=True,
            ),
            row=1,
            col=2,
        )
        fig_ts.add_trace(
            go.Scatter(
                x=times,
                y=kf_vel[:, i] + sigma2_vel[:, i],
                mode="lines",
                line=dict(width=0),
                legendgroup="vel",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=2,
        )
        fig_ts.add_trace(
            go.Scatter(
                x=times,
                y=kf_vel[:, i] - sigma2_vel[:, i],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=ci_fill_color[i],
                legendgroup="vel",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=2,
        )
        fig_ts.add_trace(
            go.Scatter(
                x=times,
                y=kf_vel[:, i],
                name=f"kf_v_{axis_names[i]}",
                line=dict(color=axis_colors[i], width=1, dash="dash"),
                legendgroup="vel",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    for i in range(3):
        fig_ts.add_trace(
            go.Scatter(
                x=times,
                y=np.degrees(omega[:, i]),
                name=f"ω_{axis_names[i]}",
                line=dict(color=axis_colors[i], width=1.5),
                legendgroup="omega",
                showlegend=True,
            ),
            row=2,
            col=1,
        )

    euler_names = ["Roll", "Pitch", "Yaw"]
    kf_euler = data["kf_eulers"]
    sigma2_att_deg = np.degrees(sigma2_att)
    for i in range(3):
        fig_ts.add_trace(
            go.Scatter(
                x=times,
                y=euler[:, i],
                name=euler_names[i],
                line=dict(color=axis_colors[i], width=1.5, dash="dashdot"),
                legendgroup="euler",
                showlegend=True,
            ),
            row=2,
            col=2,
        )
        fig_ts.add_trace(
            go.Scatter(
                x=times,
                y=kf_euler[:, i] + sigma2_att_deg[:, i],
                mode="lines",
                line=dict(width=0),
                legendgroup="euler",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2,
            col=2,
        )
        fig_ts.add_trace(
            go.Scatter(
                x=times,
                y=kf_euler[:, i] - sigma2_att_deg[:, i],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=ci_fill_color[i],
                legendgroup="euler",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2,
            col=2,
        )
        fig_ts.add_trace(
            go.Scatter(
                x=times,
                y=kf_euler[:, i],
                name=f"kf_{euler_names[i]}",
                line=dict(color=axis_colors[i], width=1, dash="dash"),
                legendgroup="euler",
                showlegend=False,
            ),
            row=2,
            col=2,
        )

    quat_colors = ["#ffd43b", "#ff6b6b", "#51cf66", "#339af0"]
    quat_names = ["w", "x", "y", "z"]
    for i in range(4):
        fig_ts.add_trace(
            go.Scatter(
                x=times,
                y=quats[:, i],
                name=f"q_{quat_names[i]}",
                line=dict(color=quat_colors[i], width=1.5),
                legendgroup="quat",
                showlegend=True,
            ),
            row=3,
            col=1,
        )

    downrange = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
    fig_ts.add_trace(
        go.Scatter(
            x=downrange,
            y=pos[:, 2],
            name="Profile",
            line=dict(color="#ffd43b", width=2),
            showlegend=False,
        ),
        row=3,
        col=2,
    )
    fig_ts.add_trace(
        go.Scatter(
            x=[downrange[apogee_idx]],
            y=[apogee_alt],
            mode="markers",
            marker=dict(size=8, color="red", symbol="x"),
            name="Apogee",
            showlegend=False,
        ),
        row=3,
        col=2,
    )

    speed = np.linalg.norm(vel, axis=1)
    kf_speed = np.linalg.norm(kf_vel, axis=1)
    sigma2_speed = np.sqrt(
        np.sum(
            sigma2_vel**2 * (kf_vel / np.maximum(kf_speed, 1e-6)[:, None]) ** 2, axis=1
        )
    )
    fig_ts.add_trace(
        go.Scatter(
            x=times,
            y=speed,
            name="|v|",
            line=dict(color="#e599f7", width=2),
            showlegend=False,
        ),
        row=4,
        col=1,
    )
    fig_ts.add_trace(
        go.Scatter(
            x=times,
            y=kf_speed + sigma2_speed,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=4,
        col=1,
    )
    fig_ts.add_trace(
        go.Scatter(
            x=times,
            y=np.maximum(kf_speed - sigma2_speed, 0),
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(229,153,247,0.15)",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=4,
        col=1,
    )
    fig_ts.add_trace(
        go.Scatter(
            x=times,
            y=kf_speed,
            name="kf |v|",
            line=dict(color="#e599f7", width=1, dash="dash"),
            showlegend=False,
        ),
        row=4,
        col=1,
    )

    omega_mag = np.linalg.norm(omega, axis=1)
    fig_ts.add_trace(
        go.Scatter(
            x=times,
            y=np.degrees(omega_mag),
            name="|ω|",
            line=dict(color="#ffa94d", width=2),
            showlegend=False,
        ),
        row=4,
        col=2,
    )

    for row in range(1, 5):
        for col in range(1, 3):
            fig_ts.add_vline(
                x=burn_time,
                row=row,
                col=col,
                line=dict(color="rgba(255,100,100,0.4)", width=1, dash="dash"),
            )
            fig_ts.add_vline(
                x=start_time,
                row=row,
                col=col,
                line=dict(color="rgba(100,255,100,0.4)", width=1, dash="dash"),
            )

    fig_ts.update_yaxes(title_text="m", row=1, col=1)
    fig_ts.update_yaxes(title_text="m/s", row=1, col=2)
    fig_ts.update_yaxes(title_text="°/s", row=2, col=1)
    fig_ts.update_yaxes(title_text="°", row=2, col=2)
    fig_ts.update_yaxes(title_text="", row=3, col=1)
    fig_ts.update_yaxes(title_text="Alt (m)", row=3, col=2)
    fig_ts.update_yaxes(title_text="m/s", row=4, col=1)
    fig_ts.update_yaxes(title_text="°/s", row=4, col=2)
    fig_ts.update_xaxes(title_text="Time (s)", row=4, col=1)
    fig_ts.update_xaxes(title_text="Time (s)", row=4, col=2)
    fig_ts.update_xaxes(title_text="Downrange (m)", row=3, col=2)

    fig_ts.update_layout(
        title=dict(
            text="Flight Telemetry — Ground Truth vs Filter (2σ CI)",
            font=dict(size=18),
        ),
        paper_bgcolor="rgb(20, 20, 35)",
        plot_bgcolor="rgb(25, 25, 40)",
        font=dict(color="white", size=10),
        width=1200,
        height=1000,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.05,
            xanchor="center",
            x=0.5,
            font=dict(size=9),
        ),
    )

    for i in range(1, 5):
        for j in range(1, 3):
            fig_ts.update_xaxes(
                gridcolor="rgba(255,255,255,0.07)",
                zerolinecolor="rgba(255,255,255,0.15)",
                row=i,
                col=j,
            )
            fig_ts.update_yaxes(
                gridcolor="rgba(255,255,255,0.07)",
                zerolinecolor="rgba(255,255,255,0.15)",
                row=i,
                col=j,
            )

    # ======== FIGURE 3: Log-Likelihood ========
    fig_ll = go.Figure()
    fig_ll.add_trace(
        go.Scatter(
            x=data["gps_ll_times"],
            y=data["gps_log_likelihoods"],
            mode="lines+markers",
            line=dict(color="#339af0", width=1.5),
            marker=dict(size=4, color="#339af0"),
            name="GPS",
        )
    )
    fig_ll.add_trace(
        go.Scatter(
            x=data["alt_ll_times"],
            y=data["alt_log_likelihoods"],
            mode="lines+markers",
            line=dict(color="#51cf66", width=1.5),
            marker=dict(size=4, color="#51cf66"),
            name="Altimeter",
        )
    )
    fig_ll.add_trace(
        go.Scatter(
            x=data["mag_ll_times"],
            y=data["mag_log_likelihoods"],
            mode="lines+markers",
            line=dict(color="#ffd43b", width=1.5),
            marker=dict(size=4, color="#ffd43b"),
            name="Magnetometer",
        )
    )
    fig_ll.add_vline(
        x=burn_time,
        line=dict(color="rgba(255,100,100,0.4)", width=1, dash="dash"),
    )
    fig_ll.add_vline(
        x=start_time,
        line=dict(color="rgba(100,255,100,0.4)", width=1, dash="dash"),
    )
    fig_ll.update_layout(
        title=dict(text="Measurement Log-Likelihood", font=dict(size=18)),
        xaxis_title="Time (s)",
        yaxis_title="Log-Likelihood",
        paper_bgcolor="rgb(20, 20, 35)",
        plot_bgcolor="rgb(25, 25, 40)",
        font=dict(color="white", size=10),
        width=900,
        height=400,
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.07)",
            zerolinecolor="rgba(255,255,255,0.15)",
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.07)",
            zerolinecolor="rgba(255,255,255,0.15)",
        ),
    )

    # ======== FIGURE 4: NEES ========
    n = len(times)
    nees = np.zeros(n)
    pos_true = data["positions"]
    vel_true = data["velocities"]
    quats_true = data["quats"]
    a_b_true = data["true_a_b"]
    w_b_true = data["true_w_b"]
    x_nom_full = data["kf_x_nom_full"]
    P_full = data["kf_P_full"]

    for i in range(n):
        dx = np.zeros(15)
        dx[0:3] = pos_true[i] - x_nom_full[i, 0:3]
        dx[3:6] = vel_true[i] - x_nom_full[i, 3:6]
        q_true = Rotation.from_quat(quats_true[i], scalar_first=True)
        q_est = Rotation.from_quat(x_nom_full[i, 6:10], scalar_first=True)
        dq = q_est.inv() * q_true
        dx[6:9] = dq.as_rotvec()
        dx[9:12] = a_b_true[i] - x_nom_full[i, 10:13]
        dx[12:15] = w_b_true[i] - x_nom_full[i, 13:16]
        nees[i] = dx @ np.linalg.solve(P_full[i], dx)

    fig_nees = go.Figure()
    fig_nees.add_trace(
        go.Scatter(
            x=times,
            y=nees,
            mode="lines",
            line=dict(color="#51cf66", width=1.5),
            name="NEES",
        )
    )
    fig_nees.add_hline(
        y=15,
        line=dict(color="rgba(255,255,255,0.4)", width=1, dash="dash"),
        annotation_text="E[NEES] = dim(x) = 15",
        annotation_font_color="white",
    )
    fig_nees.add_vline(
        x=burn_time,
        line=dict(color="rgba(255,100,100,0.4)", width=1, dash="dash"),
    )
    fig_nees.add_vline(
        x=start_time,
        line=dict(color="rgba(100,255,100,0.4)", width=1, dash="dash"),
    )
    fig_nees.update_layout(
        title=dict(
            text="NEES (Normalized Estimation Error Squared)", font=dict(size=18)
        ),
        xaxis_title="Time (s)",
        yaxis_title="NEES",
        paper_bgcolor="rgb(20, 20, 35)",
        plot_bgcolor="rgb(25, 25, 40)",
        font=dict(color="white", size=10),
        width=900,
        height=400,
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.07)",
            zerolinecolor="rgba(255,255,255,0.15)",
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.07)",
            zerolinecolor="rgba(255,255,255,0.15)",
        ),
    )

    return fig_3d, fig_ts, fig_ll, fig_nees, nees


print("Running rocket simulation...")
data = run_simulation()

t = data["times"]
pos = data["positions"]
apogee_idx = np.argmax(pos[:, 2])
print(f"Apogee: {pos[apogee_idx, 2]:.1f} m at t={t[apogee_idx]:.1f} s")
print(f"Max speed: {np.max(np.linalg.norm(data['velocities'], axis=1)):.1f} m/s")
print(f"Max roll rate: {np.max(np.abs(np.degrees(data['omegas'][:, 2]))):.1f} °/s")
print(f"Flight time: {t[-1]:.1f} s")
print(f"Landing distance: {np.sqrt(pos[-1, 0]**2 + pos[-1, 1]**2):.1f} m")
print(f"Final quaternion: {data['quats'][-1]}")

fig_3d, fig_ts, fig_ll, fig_nees, nees = plot_results(data)
fig_3d.show()
fig_ts.show()
fig_ll.show()
fig_nees.show()

print(f"Average NEES: {np.mean(nees):.3f}")

# %%
