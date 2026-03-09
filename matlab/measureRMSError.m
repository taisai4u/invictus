function rmse = measureRMSError(noiseParams, KF, initState, initCov, posSelector, dt, measPos, truePos)
    [Qtest, Rtest] = extractNoiseParameters(noiseParams);
    
    KF.ProcessNoise = Qtest;
    KF.MeasurementNoise = Rtest;

    rmse = evaluateFilter(KF, initState, initCov, posSelector, dt, measPos, truePos);
end