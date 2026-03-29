Fs = 100;
samplesPerRead = 10;
runTime = 12000;
dt = 1/Fs;

accel_offset=[-33 -8 -18].*0.01; % 1 LSB = 0.01 m/s^2
gyro_offset=[0 -1 0].*deg2rad(1/16); % 1 LSB = 1/16 deg/s
mag_offset=[-614 59 -355].*1/16; % 1 LSB = 1/16 uT

if ~(exist('a', 'var') && isvalid(a))
    a = arduino;
end
if ~(exist('imu', 'var') && isa(imu, 'bno055'))
    imu = bno055(a, 'SampleRate', Fs, 'OutputFormat', 'matrix', ...
        'SamplesPerRead', samplesPerRead, 'I2CAddress', 0x28, 'OperatingMode', 'amg');
end

accel_noise_density=180e-6*9.81; % 150 ug / sqrt(Hz)
gyro_noise_density=deg2rad(0.014); % 0.014 deg/s / sqrt(Hz)
mag_noise=15.4; % 1.4 uT

% IMU noise parameters — tune for your sensor
sig_an = accel_noise_density*sqrt(Fs);      % accelerometer white noise (m/s^2/sqrt(Hz))
sig_wn = gyro_noise_density*sqrt(Fs);      % gyroscope white noise (rad/s/sqrt(Hz))
sig_aw = sqrt(1e-4);  % accelerometer bias random walk
sig_ww = sqrt(1e-8);  % gyroscope bias random walk
g = [0; 0; -9.81];     % gravity in world frame (z-up)

% Initial state: [pos(3); vel(3); quat(4, w x y z); accel_bias(3); gyro_bias(3)]
x_nom = zeros(16, 1);
x_nom(7) = 1;  % identity quaternion
sigma_accel_bias_init = 80e-3 * 9.81; % BNO055 typical accel offset: ~80mg
sigma_gyro_bias_init  = deg2rad(1.0);  % BNO055 typical gyro zero-rate offset: ~1 deg/s
P = eye(15) * 100.0;
P(10:12,10:12) = eye(3) .* sigma_accel_bias_init.^2;
P(13:15,13:15) = eye(3) .* sigma_gyro_bias_init.^2;

kf = ESKF(x_nom, P, sig_an, sig_wn, sig_aw, sig_ww, g);

% Calibration — hold sensor still
% Collects magnetometer reference and accel-mag angle statistics for
% interference detection (mirrors reader.py calibration phase)
fprintf('Calibrating — hold sensor still...\n');
accel_mag_angles = [];
calibration_batches = 30;
for ci = 1:calibration_batches
    [accel_c, ~, mag_c] = imu();
    accel_c=accel_c-accel_offset;
    mag_c=mag_c-mag_offset;
    for si = 1:size(accel_c, 1)
        a_i = accel_c(si,:).';
        m_i = mag_c(si,:).';
        cos_angle = dot(m_i, a_i) / (norm(m_i) * norm(a_i));
        accel_mag_angles(end+1) = acos(max(-1, min(1, cos_angle))); %#ok<SAGROW>
    end
end
m_ref = mean(mag_c, 1).'; % This only works if you hold it flat and aligned with global ENU
% m_ref = [-3.6571 21.3273 -45.0158].'; % ENU
static_accel_mag_angle_mean = mean(accel_mag_angles);
static_accel_mag_angle_var = var(accel_mag_angles);
fprintf('Calibration done. Accel-mag angle: %.2f ± %.4f rad\n', ...
    static_accel_mag_angle_mean, sqrt(static_accel_mag_angle_var));

dt_batch = samplesPerRead / Fs;
last_m_m = [];

% Set up 3D orientation visualization
fig = figure('Name', 'ESKF Orientation');
ax = axes(fig);
hold(ax, 'on');
axis(ax, 'equal');
xlim(ax, [-1.5 1.5]); ylim(ax, [-1.5 1.5]); zlim(ax, [-1.5 1.5]);
axis(ax, 'manual');
xlabel(ax, 'X'); ylabel(ax, 'Y'); zlabel(ax, 'Z');
title(ax, 'Body Frame Orientation');
view(ax, 3);
grid(ax, 'on');
rotate3d(ax, 'on');

% Static world frame axes (gray)
quiver3(ax, 0,0,0, 1,0,0, 0, 'Color', [0.7 0.7 0.7], 'LineWidth', 1);
quiver3(ax, 0,0,0, 0,1,0, 0, 'Color', [0.7 0.7 0.7], 'LineWidth', 1);
quiver3(ax, 0,0,0, 0,0,1, 0, 'Color', [0.7 0.7 0.7], 'LineWidth', 1);

% Static magnetic reference vector (faint)
m_ref_unit = m_ref / norm(m_ref);
quiver3(ax, 0,0,0, m_ref_unit(1),m_ref_unit(2),m_ref_unit(3), 0, ...
    'Color', [0.5 0 0.8] .* 0.25, 'LineWidth', 1.5, 'LineStyle', '--');

% Body frame axes — updated each iteration
hx = quiver3(ax, 0,0,0, 1,0,0, 0, 'r', 'LineWidth', 2.5);
hy = quiver3(ax, 0,0,0, 0,1,0, 0, 'g', 'LineWidth', 2.5);
hz = quiver3(ax, 0,0,0, 0,0,1, 0, 'b', 'LineWidth', 2.5);
hm = quiver3(ax, 0,0,0, 0,1,0, 0, 'Color', [0.5 0 0.8], 'LineWidth', 2.5);
ha = quiver3(ax, 0,0,0, 0,0,1, 0, 'Color', [1 0.5 0], 'LineWidth', 2.5);
legend(ax, '', '', '', 'Mag ref', 'X', 'Y', 'Z', 'Mag', 'Accel', 'Location', 'northeast');

mag_color   = [0.5 0 0.8];
accel_color = [1 0.5 0];
fade = 0.15; % opacity when not used
mag_color_faded   = mag_color   .* fade;
accel_color_faded = accel_color .* fade;

fprintf('Running for %d seconds...\n', runTime);
tic
while toc <= runTime
    [accel, gyro, mag] = imu();
    accel=accel-accel_offset;
    gyro=gyro-gyro_offset;
    mag=mag-mag_offset;

    % Predict with each sample
    for i = 1:size(accel, 1)
        kf = kf.predict([accel(i,:).'; gyro(i,:).'], dt);
    end

    accel_sample = mean(accel, 1).';
    gyro_sample = mean(gyro, 1).';

    % Accelerometer update — gravity-based tilt correction, only when static
    accel_used = false;
    imu_static = abs(norm(accel_sample) - 9.81) < 0.2 && norm(gyro_sample) < deg2rad(5);
    if imu_static
        H_x_accel = zeros(3, 16);
        H_x_accel(1:3, 7:10) = kf.get_inverse_rotation_H_x(-g);
        H_x_accel(1:3, 11:13) = eye(3);
        R_accel = eye(3) .* sig_an.^2;
        pred_accel = kf.get_rotation_matrix().' * -g + kf.x_nom(11:13);
        
        % kf = kf.update(pred_accel, accel_sample, R_accel, H_x_accel);
        % accel_used = true;
        % Mahalanobis gating
        y = accel_sample - pred_accel;
        H = H_x_accel * kf.get_X_dx();
        S = H * kf.P * H.' + R_accel;
        D2 = y.' * (S \ y);
        if D2 < chi2inv(0.997, 3)
            kf = kf.update(pred_accel, accel_sample, R_accel, H_x_accel);
            accel_used = true;
        end
    end

    % Magnetometer update
    mag_used = false;
    m_meas = mean(mag, 1).';
    m_norm = norm(m_meas);
    if m_norm > 0 && ~isempty(last_m_m)
        % Check for magnetic interference
        if imu_static
            % Static: angle between accel and mag must match calibrated baseline
            cos_angle = dot(m_meas, accel_sample) / (m_norm * norm(accel_sample));
            angle = acos(max(-1, min(1, cos_angle)));
            D2_angle = (angle - static_accel_mag_angle_mean)^2 / static_accel_mag_angle_var;
            mag_interference_absent = D2_angle < chi2inv(0.997, 1);
        else
            % Moving: yaw rate from mag change must match yaw rate from gyro
            cos_mm = dot(m_meas, last_m_m) / (m_norm * norm(last_m_m));
            yaw_angvel_mag = abs(acos(max(-1, min(1, cos_mm))) / dt_batch);
            eul = quat2eul(compact(kf.get_quat()), 'XYZ');
            phi = eul(1); theta = eul(2);
            yaw_angvel_gyro = abs(-(sin(phi)/cos(theta)) * gyro_sample(2) ...
                                  + (cos(phi)/cos(theta)) * gyro_sample(3));
            mag_interference_absent = abs(yaw_angvel_mag - yaw_angvel_gyro) <= deg2rad(10);
        end

        if mag_interference_absent
            z_mag = m_meas;
            R_mag = eye(3) .* mag_noise^2;
            H_x_mag = zeros(3, 16);
            H_x_mag(1:3, 7:10) = kf.get_inverse_rotation_H_x(m_ref);
            pred_mag = kf.get_rotation_matrix().' * m_ref;

            % kf = kf.update(pred_mag, z_mag, R_mag, H_x_mag);
            % mag_used = true;
            % Mahalanobis gating
            y_mag = z_mag - pred_mag;
            H_mag = H_x_mag * kf.get_X_dx();
            S_mag = H_mag * kf.P * H_mag.' + R_mag;
            D2_mag = y_mag.' * (S_mag \ y_mag);
            if D2_mag < chi2inv(0.997, 3)
                kf = kf.update(pred_mag, z_mag, R_mag, H_x_mag);
                mag_used = true;
            end
        end
    end
    last_m_m = m_meas;

    % Update visualization — columns of R are body axes expressed in world frame
    R = kf.get_rotation_matrix();
    hx.UData = R(1,1); hx.VData = R(2,1); hx.WData = R(3,1);
    hy.UData = R(1,2); hy.VData = R(2,2); hy.WData = R(3,2);
    hz.UData = R(1,3); hz.VData = R(2,3); hz.WData = R(3,3);

    % Magnetometer and acceleration readings normalized to unit length for display
    m_unit = m_meas ./ m_norm;
    hm.UData = m_unit(1); hm.VData = m_unit(2); hm.WData = m_unit(3);
    hm.Color = mag_color .* mag_used + mag_color_faded .* ~mag_used;
    a_unit = accel_sample ./ norm(accel_sample);
    ha.UData = a_unit(1); ha.VData = a_unit(2); ha.WData = a_unit(3);
    ha.Color = accel_color .* accel_used + accel_color_faded .* ~accel_used;
    drawnow;
end
