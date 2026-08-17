function tests = testProcesses
% Automated structural tests for public Markov-process discretizers.
tests = functiontests(localfunctions);
end

function setupOnce(~)
repositoryRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(repositoryRoot));
end

function testDiscreteARDefaultArguments(testCase)
[transition, grid] = discretization.discreteAR(0, 0.9, 0.1, 5);
verifyMarkovApproximation(testCase, transition, grid, 5);
end

function testDiscreteVARDefaultArguments(testCase)
[transition, grid] = discretization.discreteVAR(0, 0.9, 0.01, 5);
verifyMarkovApproximation(testCase, transition, grid, 5);
end

function testDiscreteCIRDefaultArguments(testCase)
a = -log(0.58);
b = 0.028;
sigma = 0.1;
warningState = warning;
cleanup = onCleanup(@() warning(warningState));
warning('off','all');
[transition, grid] = discretization.discreteCIR(a, b, sigma, 0.25, 9);
verifyMarkovApproximation(testCase, transition, grid, 9);
verifyGreaterThan(testCase, grid, zeros(size(grid)));
end

function testDiscreteGMARDefaultArguments(testCase)
[transition, grid] = discretization.discreteGaussianMixtureAR( ...
    0.0555, 0.5854, [0.1628, 0.8372], ...
    [-0.0039, 0.0008], [0.1293, 0.0300], 5);
verifyMarkovApproximation(testCase, transition, grid, 5);
end

function testDiscreteSVDefaultArguments(testCase)
[transition, grid] = discretization.discreteStochasticVolatilityAR( ...
    0.95, 0.9, 0.007, 0.06, 5, 3);

verifySize(testCase, transition, [15, 15]);
verifySize(testCase, grid, [2, 15]);
verifyTrue(testCase, all(isfinite(transition), 'all'));
verifyGreaterThanOrEqual(testCase, transition, zeros(size(transition)));
verifyEqual(testCase, sum(transition, 2), ones(15, 1), ...
    'AbsTol', 1e-10);
end

function testARReferenceParity(testCase)
repositoryRoot = fileparts(fileparts(mfilename('fullpath')));
fixture = load(fullfile(repositoryRoot, ...
    'reference_data', 'matlab_r2024b_baseline.mat'));

[transition, grid] = discretization.discreteAR( ...
    fixture.arMu, fixture.arRho, ...
    fixture.arSigma, fixture.arStateCount, fixture.arMethod, ...
    fixture.arMomentCount);

verifyEqual(testCase, grid, fixture.arGrid, 'AbsTol', 1e-10);
verifyEqual(testCase, transition, fixture.arTransition, 'AbsTol', 1e-7);
end

function verifyMarkovApproximation(testCase, transition, grid, stateCount)
verifySize(testCase, transition, [stateCount, stateCount]);
verifyEqual(testCase, numel(grid), stateCount);
verifyTrue(testCase, all(isfinite(transition), 'all'));
verifyTrue(testCase, all(isfinite(grid), 'all'));
verifyGreaterThanOrEqual(testCase, transition, zeros(size(transition)));
verifyEqual(testCase, sum(transition, 2), ones(stateCount, 1), ...
    'AbsTol', 1e-10);
end
