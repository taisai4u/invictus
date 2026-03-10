imuFs = 160;
gpsFs = 1;

% Define where on the Earth this simulated scenario takes place using the
% latitude, longitude and altitude.
refloc = [42.2825 -72.3430 53.0352];


% Validate that the |gpsFs| divides |imuFs|. This allows the sensor sample 
% rates to be simulated using a nested for loop without complex sample rate
% matching.

imuSamplesPerGPS = (imuFs/gpsFs);
assert(imuSamplesPerGPS == fix(imuSamplesPerGPS), ...
    'GPS sampling rate must be an integer factor of IMU sampling rate.');

% uav trajectory
load LoggedQuadcopter.mat trajData;
trajOrient = trajData.Orientation;
trajVel = trajData.Velocity;
trajPos = trajData.Position;
trajAcc = trajData.Acceleration;
trajAngVel = trajData.AngularVelocity;

rng(1);

% simulate GPS
gps = gpsSensor('UpdateRate', gpsFs);
gps.ReferenceLocation = refloc;
gps.DecayFactor = 0.3;
gps.HorizontalPositionAccuracy = 1.6;
gps.VerticalPositionAccuracy = 1.6;
gps.VelocityAccuracy = 0.1;

% simulate IMU
imu = imuSensor('accel-gyro-mag', 'SampleRate', imuFs);
imu.MagneticField = [19.5281 -5.0741 48.0067];

% Accelerometer
imu.Accelerometer.MeasurementRange =  19.6133;
imu.Accelerometer.Resolution = 0.0023928;
imu.Accelerometer.ConstantBias = 0.19;
imu.Accelerometer.NoiseDensity = 0.0012356;

% Gyroscope
imu.Gyroscope.MeasurementRange = deg2rad(250);
imu.Gyroscope.Resolution = deg2rad(0.0625);
imu.Gyroscope.ConstantBias = deg2rad(3.125);
imu.Gyroscope.AxesMisalignment = 1.5;
imu.Gyroscope.NoiseDensity = deg2rad(0.025);

% Magnetometer
imu.Magnetometer.MeasurementRange = 1000;
imu.Magnetometer.Resolution = 0.1;
imu.Magnetometer.ConstantBias = 100;
imu.Magnetometer.NoiseDensity = 0.3/ sqrt(50);

% intialize filter state
% using ground truth for now
initstate = zeros(16, 1);
initstate(1:3) = mean(trajPos(1:100,:), 1)';
initstate(4:6) = mean(trajVel(1:100, :), 1)';
initstate(7:10) = compact(meanrot(trajOrient(1:100)))';
initstate(11:13) = imu.Accelerometer.ConstantBias;
initstate(14:16) = imu.Gyroscope.ConstantBias;

% initialize variances
Rmag = 0.0862; % magnetometer
Rvel = 0.0051; % gps velocity
Rpos = 5.169; % gps position

P = 1e-9*eye(15);
sig_an = imu.Accelerometer.NoiseDensity * sqrt(imuFs);
sig_wn = imu.Gyroscope.NoiseDensity * sqrt(imuFs);
sig_aw = sqrt(0.010716);
sig_ww = sqrt(1.3436e-14);
g = [0; 0; -9.81];
kf = ESKF(initstate, P, sig_an, sig_wn, sig_aw, sig_ww, g);

% process noise
% fusionfilt.AccelerometerBiasNoise = 0.010716; 
% fusionfilt.AccelerometerNoise = 9.7785; 
% fusionfilt.GyroscopeBiasNoise = 1.3436e-14; 
% fusionfilt.GyroscopeNoise =  0.00016528; 
% fusionfilt.MagnetometerBiasNoise = 2.189e-11;
% fusionfilt.GeomagneticVectorNoise = 7.67e-13;

% initialize scopes
useErrScope = true; % streaming error plot
usePoseView = true; % 3D pose viewer

if useErrScope
    errscope = HelperScrollingPlotter(...
        'NumInputs', 4, ...
        'TimeSpan', 10, ...
        'SampleRate', imuFs, ...
        'YLabel', {'degrees', ...
        'meters', ...
        'meters', ...
        'meters'}, ...
        'Title', {'Quaternion Distance', ...
        'Position X Error', ...
        'Position Y Error', ...
        'Position Z Error'}, ...
        'YLimits', ...
        [ -3, 3
        -2, 2
        -2 2
        -2 2]);
end

if usePoseView
    posescope = HelperPoseViewer(...
        'XPositionLimits', [-15 15], ...
        'YPositionLimits', [-15, 15], ...
        'ZPositionLimits', [-10 10]);
end

% simulation loop
secondsToSimulate = 50; % out of 142 secs
numsamples = secondsToSimulate*imuFs;

loopBound = floor(numsamples);
loopBound = floor(loopBound/imuFs)*imuFs; % ensure enough IMU samples

% log data
pqorient = quaternion.zeros(loopBound, 1);
pqpos = zeros(loopBound, 3);
pqvars = zeros(loopBound, 15, 15);

fcnt = 1;

while (fcnt <= loopBound) % update loop, at GPS frequency
    % predict loop, at IMU frequency
    for ff=1:imuSamplesPerGPS
           % simulate the IMU data at the current pose
           [accel, gyro, mag] = imu(trajAcc(fcnt,:), trajAngVel(fcnt,:), trajOrient(fcnt,:));

           kf = kf.predict([accel(:); gyro(:)], 1/imuFs);

           [fusedPos, fusedOrient] = kf.pose();

           % save
           pqorient(fcnt) = fusedOrient;
           pqpos(fcnt,:) = fusedPos';
           pqvars(fcnt,:,:) = kf.P;

           % compute errors and plot
           if useErrScope
               orientErr = rad2deg(dist(fusedOrient, trajOrient(fcnt)));
               posErr = fusedPos' - trajPos(fcnt,:);
               errscope(orientErr, posErr(1), posErr(2), posErr(3));
           end

           % update the pose viewer
           if usePoseView
               posescope(pqpos(fcnt,:), pqorient(fcnt), ...
                   trajPos(fcnt,:), trajOrient(fcnt));

           end
           fcnt = fcnt + 1;
    end

    % simulate GPS
    [lla, gpsvel] = gps(trajPos(fcnt,:), trajVel(fcnt,:));

    % GPS position + velocity update
    gpsPos = lla2ned(lla, refloc, 'ellipsoid');
    H_x_gps = zeros(6, 16);
    H_x_gps(1:3, 1:3) = eye(3);
    H_x_gps(4:6, 4:6) = eye(3);
    R_gps = blkdiag(eye(3) .* Rpos, eye(3) .* Rvel);
    kf = kf.update(kf.x_nom(1:6), [gpsPos(:); gpsvel(:)], R_gps, H_x_gps);

    % Magnetometer update
    m_ref = imu.MagneticField(:);
    H_x_mag = zeros(3, 16);
    H_x_mag(1:3, 7:10) = kf.get_inverse_rotation_H_x(m_ref);
    kf = kf.update(kf.get_rotation_matrix()' * m_ref, mag(:) - imu.Magnetometer.ConstantBias(:), eye(3) .* Rmag, H_x_mag);
end

% RMS error computation
dpos = pqpos(1:loopBound,:) - trajPos(1:loopBound,:);

% For orientation, quaternion distance is a much better alternative to
% subtracting Euler angles, which have discontinuities. The quaternion
% distance can be computed with the |dist| function, which gives the
% angular difference in orientation in radians. Convert to degrees
% for display in the command window. 

dquat = rad2deg(dist(pqorient(1:loopBound), trajOrient(1:loopBound)));

fprintf('\n\nEnd-to-End Simulation Position RMS Error\n');
posrms = sqrt(mean(dpos.^2));
fprintf('\tX: %.2f , Y: %.2f, Z: %.2f   (meters)\n\n',posrms(1), ...
    posrms(2), posrms(3));

fprintf('End-to-End Quaternion Distance RMS Error (degrees) \n');
fprintf('\t%.2f (degrees)\n\n', sqrt(mean(dquat.^2)));