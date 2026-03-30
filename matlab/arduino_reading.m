Fs = 100;
samplesPerRead = 10;
runTime = 200;
isVerbose = false;

if ~(exist('a', 'var') && isvalid(a))
    a = arduino;
end
if ~exist('imu', 'var')
    imu = bno055(a, 'SampleRate', Fs, 'OutputFormat', 'matrix', ...
        'SamplesPerRead', samplesPerRead, 'I2CAddress', 0x28, 'OperatingMode','amg');
end
compFilt = complementaryFilter('SampleRate', Fs);

tuner = HelperOrientationFilterTuner(compFilt);
tic

while toc <= runTime
    [accel, gyro, mag, t, overrun] = imu();
    

    if (isVerbose && overrun > 0)
        fprintf('%d samples overrun ...\n', overrun);
    end

    q = compFilt(accel, gyro, mag);
    update(tuner, q);
end