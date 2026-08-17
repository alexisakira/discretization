function tests = testPerformanceParity
% Numerical parity tests for routines targeted by the performance pass.
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
repositoryRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(repositoryRoot));
testCase.TestData.Fixture = load(fullfile(repositoryRoot,...
    'reference_data','matlab_r2024b_performance.mat'));
testCase.TestData.CIRFixture = load(fullfile(repositoryRoot,...
    'reference_data','matlab_r2024b_milestone2.mat'));
warningState = warning;
testCase.TestData.WarningCleanup = onCleanup(@() warning(warningState));
warning('off','all');
end

function testARParity(testCase)
fixture = testCase.TestData.Fixture;
[transition,grid] = discretization.discreteAR(0,0.9,0.1,9,'even',4);
verifyEqual(testCase,grid,fixture.ar4Grid,'AbsTol',1e-10)
verifyEqual(testCase,transition,fixture.ar4Transition,'AbsTol',1e-8)
end

function testVARParity(testCase)
fixture = testCase.TestData.Fixture;
lag = [0.9809 0.0028; 0.0410 0.9648];
covariance = [0.0087^2 0; 0 0.0262^2];
[transition,grid] = discretization.discreteVAR( ...
    zeros(2,1),lag,covariance,9,2,'even');
verifyEqual(testCase,grid,fixture.varGrid,'AbsTol',1e-9)
verifyEqual(testCase,transition,fixture.varTransition,'AbsTol',1e-8)
end

function testGMARParity(testCase)
fixture = testCase.TestData.Fixture;
mixtureProbability = [0.1628 0.8372];
mixtureMean = [-0.0039 0.0008];
mixtureStd = [0.1293 0.0300];
[transition1,grid1] = ...
    discretization.discreteGaussianMixtureAR( ...
    0.0555,0.5854,mixtureProbability,...
    mixtureMean,mixtureStd,9,2,'even');
[transition2,grid2] = ...
    discretization.discreteGaussianMixtureAR( ...
    0.0555,[0.8959 -0.3990],...
    mixtureProbability,mixtureMean,mixtureStd,9,2,'even');
verifyEqual(testCase,grid1,fixture.gmar1Grid,'AbsTol',1e-10)
verifyEqual(testCase,transition1,fixture.gmar1Transition,'AbsTol',1e-8)
verifyEqual(testCase,grid2,fixture.gmar2Grid,'AbsTol',1e-10)
verifyEqual(testCase,transition2,fixture.gmar2Transition,'AbsTol',1e-8)
end

function testSVParity(testCase)
fixture = testCase.TestData.Fixture;
[transition,grid] = discretization.discreteStochasticVolatilityAR( ...
    0.95,0.9,0.007,0.06,9,5);
verifyEqual(testCase,grid,fixture.svGrid,'AbsTol',1e-10)
verifyEqual(testCase,transition,fixture.svTransition,'AbsTol',1e-8)
end

function testCIRParity(testCase)
fixture = testCase.TestData.CIRFixture;
[transition,grid] = discretization.discreteCIR( ...
    fixture.cirA,fixture.cirB,...
    fixture.cirSigma,fixture.cirDelta,fixture.cirStateCount,...
    fixture.cirCoverage,'exponential');
verifyEqual(testCase,grid,fixture.cirGridExponential,'AbsTol',1e-10)
verifyEqual(testCase,transition,fixture.cirTransitionExponential,...
    'AbsTol',1e-8)
end
