function tunedRMSE = evaluateFilter(KF, initState, initStateCov, posSelector, dt, measPos, truePos)
initialize(KF, initState, initStateCov);
estPosTuned = zeros(2,size(measPos,2));
magPosError = zeros(1,size(measPos,2));

for i = 2:size(measPos,2)
    predict(KF, dt);
    x = correct(KF, measPos(:,i));
    estPosTuned(:,i) = posSelector * x(:);
    magPosError(i) = norm(estPosTuned(:,i) - truePos(:,i));
end
tunedRMSE = rms(magPosError(10:end));
end