function [results,samples,metadata] = benchmarkMatlab(repetitions,outputFile)
%BENCHMARKMATLAB Reproducible performance suite for major MATLAB routines.

if nargin < 1
    repetitions = 3;
end
if nargin < 2
    outputFile = '';
end
if ~isscalar(repetitions) || repetitions < 1 || rem(repetitions,1) ~= 0
    error('repetitions must be a positive integer')
end

repositoryRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(repositoryRoot));
warningState = warning;
cleanup = onCleanup(@() warning(warningState));
warning('off','all');

caseNames = {
    'AR(1), 9 states, 4 moments'
    'VAR(1), 2D, 81 states'
    'GMAR(1), 9 states'
    'GMAR(2), 81 states'
    'CIR, 9 states'
    'SV, 45 states'
    };
caseFunctions = {
    @benchmarkAR
    @benchmarkVAR
    @benchmarkGMAR1
    @benchmarkGMAR2
    @benchmarkCIR
    @benchmarkSV
    };

caseCount = numel(caseNames);
samples = zeros(caseCount,repetitions);
checksums = zeros(caseCount,1);

for caseIndex = 1:caseCount
    checksums(caseIndex) = caseFunctions{caseIndex}(); % JIT warm-up
    for repetition = 1:repetitions
        startTime = tic;
        checksums(caseIndex) = caseFunctions{caseIndex}();
        samples(caseIndex,repetition) = toc(startTime);
    end
end

results = table(caseNames,median(samples,2),min(samples,[],2),...
    max(samples,[],2),checksums,'VariableNames',...
    {'Case','MedianSeconds','MinimumSeconds','MaximumSeconds','Checksum'});
metadata = struct('MatlabRelease',version('-release'),...
    'MatlabVersion',version,'Computer',computer,...
    'Repetitions',repetitions,'Timestamp',datetime('now'));

disp(results)
if ~isempty(outputFile)
    save(outputFile,'results','samples','metadata')
end

end


function checksum = benchmarkAR
[transition,grid] = discretization.discreteAR(0,0.9,0.1,9,'even',4);
checksum = sum(transition,'all') + sum(grid,'all');
end


function checksum = benchmarkVAR
lag = [0.9809 0.0028; 0.0410 0.9648];
covariance = [0.0087^2 0; 0 0.0262^2];
[transition,grid] = discretization.discreteVAR( ...
    zeros(2,1),lag,covariance,9,2,'even');
checksum = sum(transition,'all') + sum(grid,'all');
end


function checksum = benchmarkGMAR1
[transition,grid] = discretization.discreteGaussianMixtureAR( ...
    0.0555,0.5854,...
    [0.1628 0.8372],[-0.0039 0.0008],[0.1293 0.0300],9,2,'even');
checksum = sum(transition,'all') + sum(grid,'all');
end


function checksum = benchmarkGMAR2
[transition,grid] = discretization.discreteGaussianMixtureAR( ...
    0.0555,[0.8959 -0.3990],...
    [0.1628 0.8372],[-0.0039 0.0008],[0.1293 0.0300],9,2,'even');
checksum = sum(transition,'all') + sum(grid,'all');
end


function checksum = benchmarkCIR
[transition,grid] = discretization.discreteCIR( ...
    -log(0.58),0.028,0.1,0.25,9,0.995,...
    'exponential');
checksum = sum(transition,'all') + sum(grid,'all');
end


function checksum = benchmarkSV
[transition,grid] = discretization.discreteStochasticVolatilityAR( ...
    0.95,0.9,0.007,0.06,9,5);
checksum = sum(transition,'all') + sum(grid,'all');
end
