function tests = testNumericalRegressions
tests = functiontests(localfunctions);
end

function setupOnce(~)
addpath(fileparts(fileparts(mfilename('fullpath'))));
end

function testCIRSmallNoncentralityAndCentralLimit(testCase)
% Independent density reference for the previously underflowing case.
verifyEqual(testCase,discretization.cirTransitionDensity(1,1,1,1,.1,10), ...
    5.639545542967239,'RelTol',1e-11);
x = [.9,1,1.1];
centralDensity = exp(200*log(200)-gammaln(200)+199*log(x)-200*x);
verifyEqual(testCase,discretization.cirTransitionDensity(x,1,1,1,.1,1000), ...
    centralDensity,'RelTol',1e-11);
verifyEqual(testCase,discretization.cirTransitionDensity(x',1,1,1,.1,10), ...
    discretization.cirTransitionDensity(x,1,1,1,.1,10)','RelTol',1e-12);
end

function testCIRLongGapChains(testCase)
for gap = [10,1000]
    [P,X] = discretization.discreteCIR(1,1,.1,gap,51);
    rho = exp(-gap); c = -expm1(-gap);
    mean = rho*X+c; variance = .01*c*(rho*X+c/2);
    verifyTrue(testCase,all(isfinite(P),'all'));
    verifyEqual(testCase,sum(P,2),ones(51,1),'AbsTol',1e-12);
    verifyLessThan(testCase,max(abs((P*X'-mean')./sqrt(variance'))),1e-7);
    verifyLessThan(testCase,max(abs(sum(P.*(X-mean').^2,2)./variance'-1)),1e-7);
end
end

function testAllMixtureGridsLocationInvariant(testCase)
for method = {'even','gauss-legendre','clenshaw-curtis','gauss-hermite','GMQ'}
    for moments = [2,4]
        [P,X,d] = discretization.discreteGaussianMixtureAR( ...
            0,.4,[.4,.6],[-.1,.1],[.1,.2],21,moments,method{1},4);
        for location = [.5,10]
            [Q,Y,info] = discretization.discreteGaussianMixtureAR( ...
                location,.4,[.4,.6],[-.1,.1],[.1,.2],21,moments,method{1},4);
            verifyEqual(testCase,Y-X,location*ones(size(X)),'AbsTol',1e-12);
            verifyEqual(testCase,Q,P,'AbsTol',1e-7);
            verifyEqual(testCase,info.matchedMoments,d.matchedMoments);
        end
    end
end
end

function testSVNestedVolatilityFallback(testCase)
state = warning; cleanup = onCleanup(@() warning(state));
warning('off','all');
[~,~,d] = discretization.discreteStochasticVolatilityAR(0,.99,.007,.06,9,3,'quantile');
verifyEqual(testCase,d.matchedMoments,2*ones(27,1));
verifyEmpty(testCase,d.downgradedRows);
verifyEqual(testCase,d.volatility.matchedMoments,[1;2;1]);
verifyEqual(testCase,d.volatility.downgradedRows,[1;3]);
end
