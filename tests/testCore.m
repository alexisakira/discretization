function tests = testCore
% Automated tests for the distribution and quadrature building blocks.
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
repositoryRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(repositoryRoot));
testCase.TestData.RepositoryRoot = repositoryRoot;
end

function testDiscreteApproximationMatchesMoments(testCase)
grid = [-1, 0, 1];
targetMoments = [0; 0.5];

[probability, ~, momentError] = discretization.maximumEntropyWeights( ...
    grid, @(x) [x; x.^2], targetMoments);

verifyEqual(testCase, probability, [0.25, 0.5, 0.25], ...
    'AbsTol', 1e-8);
verifyEqual(testCase, sum(probability), 1, 'AbsTol', 1e-12);
verifyLessThan(testCase, norm(momentError), 1e-8);
end

function testEntropyObjectiveIsStableForLargeExponents(testCase)
lambda = 1000;
evaluatedMoments = [1000, -1000];
targetMoment = 0;
prior = [0.5, 0.5];

[objective, gradient, hessian, probability] = entropyObjective( ...
    lambda, evaluatedMoments, targetMoment, prior);

verifyTrue(testCase, isfinite(objective));
verifyTrue(testCase, all(isfinite(gradient)));
verifyTrue(testCase, all(isfinite(hessian), 'all'));
verifyEqual(testCase, objective, 1e6-log(2), 'AbsTol', 1e-8);
verifyEqual(testCase, gradient, 1000, 'AbsTol', 1e-10);
verifyEqual(testCase, probability, [1, 0], 'AbsTol', realmin);
verifyGreaterThanOrEqual(testCase, hessian, -eps);
end

function testDiscreteNonparametricDistribution(testCase)
centeredMoments = [0, 1, 0, 3];
[grid, probability] = discretization.momentMatchedDistribution( ...
    9, centeredMoments);

verifySize(testCase, grid, [1, 9]);
verifySize(testCase, probability, [1, 9]);
verifyGreaterThanOrEqual(testCase, probability, zeros(size(probability)));
verifyEqual(testCase, sum(probability), 1, 'AbsTol', 1e-10);
verifyEqual(testCase, probability * (grid'.^(1:4)), ...
    centeredMoments, 'AbsTol', 1e-5);
end

function testNonparametricGaussianQuadrature(testCase)
data = linspace(-2, 2, 101);
[nodes, weights] = discretization.dataDrivenGaussianQuadrature(data, 3);

verifySize(testCase, nodes, [1, 3]);
verifySize(testCase, weights, [1, 3]);
verifyTrue(testCase, issorted(nodes));
verifyGreaterThan(testCase, weights, zeros(size(weights)));
verifyEqual(testCase, sum(weights), 1, 'AbsTol', 1e-12);
verifyEqual(testCase, weights * nodes', mean(data), 'AbsTol', 1e-12);
end

function testCoreReferenceParity(testCase)
fixture = load(fullfile(testCase.TestData.RepositoryRoot, ...
    'reference_data', 'matlab_r2024b_baseline.mat'));

[probability, ~, momentError] = discretization.maximumEntropyWeights( ...
    fixture.coreGrid, @(x) [x; x.^2], fixture.coreTargetMoments);

verifyEqual(testCase, probability, fixture.coreProbability, ...
    'AbsTol', 1e-8);
verifyEqual(testCase, momentError, fixture.coreMomentError, ...
    'AbsTol', 1e-8);
end
