D = gpuDevice(1);

% Setup a sample frequency and number of samples
sampleFreq = 1000;
sampleTime = 1/sampleFreq;
numSamples = 2^25;

% Create a gpuArray that will store the empty data structure in GPU memory
timeVec = gpuArray((0:numSamples-1) * sampleTime);

% Create a signal with harmonic components at frequencies 50 and 120 Hz, and add some noise
freq1   = 2 * pi * 50;
freq2   = 2 * pi * 120;
signal  = sin( freq1 .* timeVec ) + sin( freq2 .* timeVec );
signal  = signal + 2 * randn( size( timeVec ), 'gpuArray');

% Perform FFT on the GPU
transformedSignal = fft( signal );

% Compute the Power Spectral Density
powerSpectrum = transformedSignal .* conj(transformedSignal) ./ numSamples;

% Display the Power Spectral Density
frequencyVector = sampleFreq/2 * linspace( 0, 1, numSamples/2 + 1 );

clearvars -except D

% Setup a sample frequency and number of samples
sampleFreq = 1000;
sampleTime = 1/sampleFreq;
numSamples = 2^25;


pause(3);

tic

% Create a gpuArray that will store the empty data structure in GPU memory
timeVec = gpuArray((0:numSamples-1) * sampleTime);

% Create a signal with harmonic components at frequencies 50 and 120 Hz, and add some noise
freq1   = 2 * pi * 50;
freq2   = 2 * pi * 120;
signal  = sin( freq1 .* timeVec ) + sin( freq2 .* timeVec );
signal  = signal + 2 * randn( size( timeVec ), 'gpuArray');

% Perform FFT on the GPU
transformedSignal = fft( signal );

% Compute the Power Spectral Density
powerSpectrum = transformedSignal .* conj(transformedSignal) ./ numSamples;

% Display the Power Spectral Density
frequencyVector = sampleFreq/2 * linspace( 0, 1, numSamples/2 + 1 );

x = gather(frequencyVector);
wait(D);

toc

disp('Done!')

pause(1)

gpuDevice([])

pause(1)

