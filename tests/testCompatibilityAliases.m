function tests = testCompatibilityAliases
% Ensure historical entry points continue to forward to the package API.
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
repositoryRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(repositoryRoot));
testCase.TestData.RepositoryRoot = repositoryRoot;
end

function testPackageWorksWithRootOnly(testCase)
originalPath = path;
cleanup = onCleanup(@() path(originalPath));
repositoryFolders = strsplit(genpath(testCase.TestData.RepositoryRoot),...
    pathsep);
for folderIndex = 1:numel(repositoryFolders)
    if ~isempty(repositoryFolders{folderIndex})
        rmpath(repositoryFolders{folderIndex});
    end
end
addpath(testCase.TestData.RepositoryRoot)

[transition,grid] = discretization.discreteAR(0,0.9,0.1,5);
verifySize(testCase,transition,[5 5])
verifySize(testCase,grid,[1 5])
verifyEqual(testCase,sum(transition,2),ones(5,1),'AbsTol',1e-10)
end

function testLegacyModelAliases(testCase)
warningState = warning;
cleanup = onCleanup(@() warning(warningState));
warning('off','all');

[actualP,actualX] = discreteAR(0,0.9,0.1,5);
[expectedP,expectedX] = discretization.discreteAR(0,0.9,0.1,5);
verifyEqual(testCase,actualP,expectedP)
verifyEqual(testCase,actualX,expectedX)

[actualP,actualX] = discreteVAR(0,0.9,0.01,5);
[expectedP,expectedX] = discretization.discreteVAR(0,0.9,0.01,5);
verifyEqual(testCase,actualP,expectedP)
verifyEqual(testCase,actualX,expectedX)

cirArguments = {-log(0.58),0.028,0.1,0.25,5,0.995,'exponential'};
[actualP,actualX] = discreteCIR(cirArguments{:});
[expectedP,expectedX] = discretization.discreteCIR(cirArguments{:});
verifyEqual(testCase,actualP,expectedP)
verifyEqual(testCase,actualX,expectedX)

mixtureArguments = {0.0555,0.5854,[0.1628 0.8372],...
    [-0.0039 0.0008],[0.1293 0.0300],5,2,'even'};
[actualP,actualX] = discreteGMAR(mixtureArguments{:});
[expectedP,expectedX] = ...
    discretization.discreteGaussianMixtureAR(mixtureArguments{:});
verifyEqual(testCase,actualP,expectedP)
verifyEqual(testCase,actualX,expectedX)

svArguments = {0.95,0.9,0.007,0.06,5,3};
[actualP,actualX] = discreteSV(svArguments{:});
[expectedP,expectedX] = ...
    discretization.discreteStochasticVolatilityAR(svArguments{:});
verifyEqual(testCase,actualP,expectedP)
verifyEqual(testCase,actualX,expectedX)
end

function testLegacyUtilityAliases(testCase)
warningState = warning;
cleanup = onCleanup(@() warning(warningState));
warning('off','all');

[actualX,actualP] = discreteNP(7,[0 1 0 3]);
[expectedX,expectedP] = ...
    discretization.momentMatchedDistribution(7,[0 1 0 3]);
verifyEqual(testCase,actualX,expectedX)
verifyEqual(testCase,actualP,expectedP)

data = linspace(-2,2,101);
[actualX,actualW] = NPGQ(data,3);
[expectedX,expectedW] = ...
    discretization.dataDrivenGaussianQuadrature(data,3);
verifyEqual(testCase,actualX,expectedX)
verifyEqual(testCase,actualW,expectedW)

mixtureArguments = {[0.25 0.75],[-1 0.5],[0.4 0.8],3};
[actualX,actualW] = GaussianMixtureQuadrature(mixtureArguments{:});
[expectedX,expectedW] = ...
    discretization.gaussianMixtureQuadrature(mixtureArguments{:});
verifyEqual(testCase,actualX,expectedX)
verifyEqual(testCase,actualW,expectedW)

grid = [-1 0 1];
momentFunction = @(x) [x;x.^2];
[actualP,actualDual,actualError] = discreteApproximation( ...
    grid,momentFunction,[0;0.5]);
[expectedP,expectedDual,expectedError] = ...
    discretization.maximumEntropyWeights(grid,momentFunction,[0;0.5]);
verifyEqual(testCase,actualP,expectedP)
verifyEqual(testCase,actualDual,expectedDual)
verifyEqual(testCase,actualError,expectedError)

densityArguments = {[0.01 0.02 0.03],0.02,0.5,0.03,0.1,0.25};
actualDensity = CIRpdf(densityArguments{:});
expectedDensity = discretization.cirTransitionDensity(densityArguments{:});
verifyEqual(testCase,actualDensity,expectedDensity)
end

function testLegacyEntryPointWarning(testCase)
verifyWarning(testCase,@callLegacyAR,'discretization:legacyEntryPoint')
end

function testDeprecatedNameWarning(testCase)
verifyWarning(testCase,@callLegacyDistribution,...
    'discretization:deprecatedFunction')
end

function testPreviousPackageNameWarnings(testCase)
verifyWarning(testCase,@callPreviousMixtureName,...
    'discretization:deprecatedFunction')
verifyWarning(testCase,@callPreviousStochasticVolatilityName,...
    'discretization:deprecatedFunction')
verifyWarning(testCase,@callPreviousDistributionName,...
    'discretization:deprecatedFunction')
verifyWarning(testCase,@callPreviousQuadratureName,...
    'discretization:deprecatedFunction')
end

function callLegacyAR
[~,~] = discreteAR(0,0.9,0.1,5);
end

function callLegacyDistribution
[~,~] = discreteNP(5,[0 1 0 3]);
end

function callPreviousMixtureName
[~,~] = discretization.discreteARWithGaussianMixtureShocks( ...
    0,0.5,1,0,0.1,5);
end

function callPreviousStochasticVolatilityName
[~,~] = discretization.discreteARWithStochasticVolatility( ...
    0.5,0.5,0.1,0.1,3,3);
end

function callPreviousDistributionName
[~,~] = discretization.discreteDistributionFromMoments(5,[0 1 0 3]);
end

function callPreviousQuadratureName
[~,~] = discretization.nonparametricGaussianQuadrature( ...
    linspace(-2,2,101),3);
end
