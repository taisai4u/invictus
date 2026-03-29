% Specify the training trajectory
trajectoryTrain = waypointTrajectory( ...
    [96.4 159.2 0; 2047 197 0;2245 -248 0; 2407 -927 0], ...
    [0; 71; 87; 110], ...
    'GroundSpeed', [30; 30; 30; 30], ...
    'SampleRate', 2);

dtTrain = 1/trajectoryTrain.SampleRate;
timeTrain = (0:dtTrain:trajectoryTrain.TimeOfArrival(end));
[posTrain, ~, velTrain] = lookupPose(trajectoryTrain, timeTrain);

% Specify the test trajectory
trajectoryTest = waypointTrajectory( ...
    [-2.3 72 0; -137 -204 0; -572 -937 0; -804 -1053 0; -887 -1349 0; ...
     -674 -1608 0; 368 -1604 0; 730 -1599 0; 1633 -1581 0; 1742 -1586 0], ...
    [0; 8; 34; 42; 53; 64; 97; 107; 133; 136], ...
    'GroundSpeed', [35; 35; 34; 30; 30; 30; 35; 35; 35; 35], ...
    'SampleRate', 2);

dtTest = 1/trajectoryTest.SampleRate;
timeTest = (0:dtTest:trajectoryTest.TimeOfArrival(end));
[posTest, ~, velTest] = lookupPose(trajectoryTest, timeTest);

% Plot the trajectories
figure
plot(posTrain(:,1), posTrain(:,2), '.', ...
    posTest(:,1), posTest(:,2), '.');
axis equal;
grid on;
legend('Training', 'Test');
xlabel('X position (m)');
ylabel('Y position (m)');
title('True Position');

KF = trackingKF('MotionModel','2D Constant Velocity');

trueStateTrain = [posTrain(:,1), velTrain(:,1), posTrain(:,2), velTrain(:,2)]';
trueStateTest = [posTest(:,1), velTest(:,1), posTest(:,2), velTest(:,2)]';

% simulate measurements
s = rng;
rng(2021);m
posSelector = KF.MeasurementModel; % position from state
rmsSensorNoise = 5; % RMS deviation of sensor data noise [m]

truePosTrain = posSelector * trueStateTrain;
measPosTrain = truePosTrain + rmsSensorNoise * randn(size(truePosTrain))/sqrt(2); 
% rmsSensorNoise is the deviation in position from true (not per axis)
% divided by sqrt(2) because rmsSensorNoise^2 = RMS_axis^2 + RMS_axis^2
% 2*RMS_axis^2 = rmsSensorNoise^2
% RMS_axis = rmsSensorNoise/sqrt(2)

truePosTest = posSelector * trueStateTest;
measPosTest = truePosTest + rmsSensorNoise * randn(size(truePosTest))/sqrt(2);

% initialize state using first two measurements
initStateTrain([1 3]) = measPosTrain(:,1);
initStateTrain([2 4]) = (measPosTrain(:,2) - measPosTrain(:,1))./dtTrain;

initStateTest([1 3]) = measPosTest(:,1);
initStateTest([2 4]) = (measPosTest(:,2) - measPosTest(:,1))./dtTest;

initStateCov = diag([1 2 1 2] * rmsSensorNoise.^2); % velocity is difference between two measurements, so sig^2 = sig_1^2 + sig_2^2 (sum of two gaussians)

% Process noise can be estimated via the expected deviation from constant velocity using a mean squared step change in velocity at each time step. 
% Using the scalar form for process noise ensures that the components in the x- or y- directions are treated equally.
accel = diff(velTrain)./dtTrain;
Qinit = var(vecnorm(accel, 2, 1)); % variance in acceleration magnitude

% measurement noise
Rinit = rmsSensorNoise.^2;

KF.ProcessNoise = Qinit;
KF.MeasurementNoise = Rinit;

% compare error of filter vs raw measurements
% errorTunedTrain = evaluateFilter(KF, initStateTrain, initStateCov, posSelector, dtTrain, measPosTrain, truePosTrain);
% disp(errorTunedTrain);
% disp(rms(vecnorm(measPosTrain - truePosTrain, 2, 1)));

% errorTunedTest = evaluateFilter(KF, initStateTest, initStateCov, posSelector, dtTest, measPosTest, truePosTest);
% disp(errorTunedTest);
% disp(rms(vecnorm(measPosTest - truePosTest, 2, 1)));

% find optimal process noise by testing all values in a range
nSweep = 100;
qSweep = linspace(1, 50, nSweep);
errorTunedTrain = zeros(1, nSweep);
errorTunedTest = zeros(1, nSweep);
for i = 1:nSweep
    KF.ProcessNoise = qSweep(i);
    errorTunedTrain(i) = evaluateFilter(KF, initStateTrain, initStateCov, posSelector, dtTrain, measPosTrain, truePosTrain);
    errorTunedTest(i) = evaluateFilter(KF, initStateTest, initStateCov, posSelector, dtTest, measPosTest, truePosTest);
end
plot(qSweep, errorTunedTrain, '-', qSweep, errorTunedTest, '-');
legend('Training', 'Test');
xlabel('Process Noise (Q)');
ylabel('RMS Position Error [m]');

[minErrorTunedTrain, iMinTrain] = min(errorTunedTrain);
[minErrorTunedTest, iMinTest] = min(errorTunedTest);
disp("Minimum error for training set: " + minErrorTunedTrain + ", Q: " + qSweep(iMinTrain));
disp("Minimum error for test set: " + minErrorTunedTest + ", Q: " + qSweep(iMinTest));

% optimize over 2 dimensions
n = length(timeTrain);
measErr = measPosTrain - posSelector * trueStateTrain;
sumR = norm(measErr);
% compute variance: mean square deviation from the true
% square the distance between measured and true
% find the average
Rinit = sum(vecnorm(measErr).^2)/(n+1);
disp(Rinit)

% create vector of stdevs for Q and R
function vector = constructParameterVector(Q,R)
vector = sqrt([Q;R]);
end

% fminunc works by iteratively changing a parameter vector
% so we put the params into a vector
initialParams = constructParameterVector(Qinit, Rinit);

% fminunc expects a function f(params) so wrap the measureRMSError method
% with an anonymous f(params) function
func = @(noiseParams) measureRMSError(noiseParams, KF, initStateTrain, initStateCov, posSelector, dtTrain, measPosTrain, truePosTrain);

optimalParams = fminunc(func,initialParams);
[QTuned, RTuned] = extractNoiseParameters(optimalParams);
disp("Auto-tuned parameters:");
disp("Q = " + QTuned + ", R = " + RTuned);

% evaluate the filter using the optimized params
KF.ProcessNoise = QTuned;
KF.MeasurementNoise = RTuned;
autoTunedRMSErrorTrain = evaluateFilter(KF, initStateTrain, initStateCov, posSelector, dtTrain, measPosTrain, truePosTrain);
autoTunedRMSErrorTest = evaluateFilter(KF, initStateTest, initStateCov, posSelector, dtTest, measPosTest, truePosTest);
disp("Auto-tuned RMS errors:");
disp("Train: " + autoTunedRMSErrorTrain);
disp("Test: " + autoTunedRMSErrorTest);

% you can see that the optimizer generalized pretty well

insfilterErrorState