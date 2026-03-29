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
% position is in NED, IMU frame is ENU
load LoggedQuadcopter.mat trajData;
trajOrient = trajData.Orientation; % body to world rotation
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

% Barometer
function pressure = h_baro(h)
    ratio=max(1-(-h)./44330,1e-12);
    pressure=101325.*ratio.^(1./0.1903);
end
sig_baro=1.3; % Pa
function pressure_measured = baroSensor(h)
    pressure_measured=h_baro(h)+randn(1).*1.3;
end

% intialize filter state
initstate = zeros(16, 1);
[lla,vel] = gps(trajPos(1:1,:), trajVel(1:1,:));
initstate(1:3) = lla2ned(lla, refloc, 'ellipsoid');
initstate(4:6) = vel;
initstate(7:10) = compact(meanrot(trajOrient(1:100)));
% do NOT set to 0 and have the filter learn the initial true value!
% having ANY significant uncertainty in the bias makes the filter diverge
initstate(11:13) = imu.Accelerometer.ConstantBias;
initstate(14:16) = imu.Gyroscope.ConstantBias;

% initialize variances
Rmag = 0.0862; % magnetometer
Rvel = 0.0051; % gps velocity
Rpos = 5.169; % gps position
Rbaro = sig_baro.^2;

P = 500*eye(15);
P(1:3,1:3) = eye(3).*Rpos^2;
P(4:6,4:6) = eye(3).*Rvel^2;
P(7:9,7:9) = eye(3).*0.00001523598776.^2;
P(10:12,10:12) = eye(3).*1e-9;
P(13:15,13:15) = eye(3).*1e-9;
sig_an = imu.Accelerometer.NoiseDensity * sqrt(imuFs);
sig_wn = imu.Gyroscope.NoiseDensity * sqrt(imuFs);
sig_aw = sqrt(0.010716);
sig_ww = sqrt(1.3436e-14);
g = [0; 0; 9.81];
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

function [kf, rms_pos, rms_orient] = evalFilter(kf, dt, gpsInterval, imu, gps, refloc, Rpos, Rmag, Rvel, Rbaro, trajPos, trajVel, trajAcc, trajOrient, trajAngVel, errscope, posescope)
    % simulation loop
    secondsToSimulate = 100; % out of 142 secs
    numsamples = secondsToSimulate/dt;
    
    loopBound = floor(numsamples);
    loopBound = floor(loopBound*dt)/dt; % ensure enough IMU samples
    % loopBound = 2;
    
    % log data
    pqorient = quaternion.zeros(loopBound, 1);
    pqpos = zeros(loopBound, 3);
    pqvars = zeros(loopBound, 15, 15);
    
    fcnt = 1;
    
    while (fcnt <= loopBound) % update loop, at GPS frequency
        % predict loop, at IMU frequency
        for ff=1:gpsInterval
               % simulate the IMU data at the current pose
               [accel, gyro, mag] = imu(trajAcc(fcnt,:), trajAngVel(fcnt,:), trajOrient(fcnt,:));
               % if 101<fcnt<201
               %     trajAcc(fcnt,:)-(rotatepoint(trajOrient(fcnt),-accel)+kf.g.')
               %     this difference should be quite close to zero, proving
               %     that this is the right formula to go from measured to
               %     true acceleration
               % end
                
               % negating the accelerometer measurement because 
               % matlab's imuSensor returns the negative of the specific
               % force
               kf = kf.predict([-accel(:); gyro(:)], dt);
    
               [fusedPos, fusedOrient] = kf.pose();
    
               % save
               pqorient(fcnt) = fusedOrient;
               pqpos(fcnt,:) = fusedPos';
               pqvars(fcnt,:,:) = kf.P;
    
               % compute errors and plot
               if ~isempty(errscope)
                   orientErr = rad2deg(dist(fusedOrient, trajOrient(fcnt)));
                   posErr = fusedPos' - trajPos(fcnt,:);
                   errscope(orientErr, posErr(1), posErr(2), posErr(3));
               end
    
               % update the pose viewer
               if ~isempty(posescope)
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

        % barometer update
        alt=trajPos(fcnt,3);
        baro = baroSensor(alt);
        H_x_baro = zeros(1,16);
        ratio=max(1-alt./44330,1e-12);
        H_x_baro(1,3)=-(-1000.*101325.*ratio.^(8087./1903) ./ 8435999);
        kf=kf.update(h_baro(kf.x_nom(3)),baro,Rbaro,H_x_baro);
    end
    
    % RMS error computation
    dpos = pqpos(1:loopBound,:) - trajPos(1:loopBound,:);
    rms_pos = sqrt(mean(dpos.^2));
    % For orientation, quaternion distance is a much better alternative to
    % subtracting Euler angles, which have discontinuities. The quaternion
    % distance can be computed with the |dist| function, which gives the
    % angular difference in orientation in radians. Convert to degrees
    % for display in the command window. 
    dquat = rad2deg(dist(pqorient(1:loopBound), trajOrient(1:loopBound)));
    rms_orient = sqrt(mean(dquat.^2));
end

[kf, rms_pos, rms_orient] = evalFilter(kf, 1./imuFs, imuSamplesPerGPS, imu, gps, refloc, Rpos, Rmag, Rvel, Rbaro, trajPos, trajVel, trajAcc, trajOrient, trajAngVel, errscope, posescope);
fprintf('\n\nEnd-to-End Simulation Position RMS Error\n');
fprintf('\tX: %.2f , Y: %.2f, Z: %.2f   (meters)\n\n',rms_pos(1), rms_pos(2), rms_pos(3));
fprintf('End-to-End Quaternion Distance RMS Error (degrees) \n');
fprintf('\t%.2f (degrees)\n\n', rms_orient);
% disp(kf.x_nom);
% disp(kf.P)