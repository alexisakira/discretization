function tests = testGaussianMixtureAREven
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
repository = fileparts(fileparts(mfilename('fullpath')));
addpath(repository);
addpath(fullfile(repository,'Discretize Gaussian mixture AR'));
state = warning;
testCase.TestData.WarningCleanup = onCleanup(@() warning(state));
warning('off','all');
end

function testGaussianLimit(testCase)
sigma = 0.1;
[P,X,d] = discretization.discreteGaussianMixtureAR( ...
    0,0.4,1,0,sigma,31,4,'even',4);
[mixtureP,mixtureX] = discretization.discreteGaussianMixtureAR( ...
    0,0.4,[0.3,0.7],[0,0],[sigma,sigma],31,4,'even',4);
verifyEqual(testCase,mixtureX,X,'AbsTol',1e-14);
verifyEqual(testCase,mixtureP,P,'AbsTol',1e-8);
verifyMarkov(testCase,P);
increments = X-0.4*X';
verifyEqual(testCase,sum(P.*increments,2),zeros(31,1),'AbsTol',sigma*1e-5);
verifyEqual(testCase,sum(P.*increments.^2,2),sigma^2*ones(31,1),'AbsTol',sigma^2*1e-5);
center = 16;
verifyEqual(testCase,d.matchedMoments(center),4);
verifyEqual(testCase,sum(P(center,:).*increments(center,:).^4), ...
    3*sigma^4,'AbsTol',sigma^4*1e-5);
end

function testLocationAndUnitInvariance(testCase)
p = [0.25,0.75]; means = [-0.3,0.1]; sds = [0.15,0.08];
[reference,grid,d] = discretization.discreteGaussianMixtureAR( ...
    0,0.4,p,means,sds,31,4,'even',4);
cases = [1,10;1e-3,0;1e3,0];
for row = 1:size(cases,1)
    units = cases(row,1); location = cases(row,2);
    [P,X,diagnostics] = discretization.discreteGaussianMixtureAR( ...
        location,0.4,p,units*means,units*sds,31,4,'even',4);
    verifyEqual(testCase,(X-location)/units,grid,'AbsTol',1e-12);
    verifyEqual(testCase,P,reference,'AbsTol',2e-7);
    verifyEqual(testCase,diagnostics.matchedMoments,d.matchedMoments);
end
end

function testNearUnitRoot(testCase)
rho = 0.995; sigma = 0.1;
[P,X,d] = discretization.discreteGaussianMixtureAR( ...
    0,rho,1,0,sigma,63,2,'even',2.5);
verifyMarkov(testCase,P);
verifyEmpty(testCase,d.downgradedRows);
increments = X-rho*X';
verifyEqual(testCase,sum(P.*increments,2),zeros(63,1),'AbsTol',sigma*1e-5);
verifyEqual(testCase,sum(P.*increments.^2,2),sigma^2*ones(63,1),'AbsTol',sigma^2*1e-5);
end

function testSkewMixtureTailResolution(testCase)
p = [0.05,0.95]; means = [-0.8,0.8*0.05/0.95]; sds = [0.2,0.03];
exactTail = sum(p.*normcdf(-0.5,means,sds));
errors = zeros(1,2);
for entry = 1:2
    counts = [63,127]; n = counts(entry);
    [P,X,d] = discretization.discreteGaussianMixtureAR( ...
        0,0.95,p,means,sds,n,4,'even',4);
    center = (n+1)/2;
    verifyMarkov(testCase,P);
    verifyEqual(testCase,d.matchedMoments(center),4);
    verifyLessThanOrEqual(testCase,max(abs(d.standardizedMomentErrors(center,:))),d.momentTolerance);
    errors(entry) = abs(sum(P(center,X < -0.5))-exactTail);
end
verifyLessThan(testCase,errors(2),errors(1));
verifyLessThan(testCase,errors(2),6e-4);
end

function testFallbackDiagnostics(testCase)
p = [0.05,0.95]; means = [-0.8,0.8*0.05/0.95]; sds = [0.2,0.03];
[P,X,d] = discretization.discreteGaussianMixtureAR( ...
    0,0.95,p,means,sds,31,4,'even');
verifyMarkov(testCase,P);
verifyNotEmpty(testCase,d.downgradedRows);
verifyEqual(testCase,d.downgradedRows,find(d.matchedMoments < 4));
verifyEqual(testCase,d.requestedMoments,4);
verifySize(testCase,d.standardizedMomentErrors,[31,4]);
meanInnovation = sum(p.*means);
sd = sqrt(sum(p.*(sds.^2+(means-meanInnovation).^2)));
z = (X-0.95*X'-meanInnovation)/sd;
centeredMeans = (means-meanInnovation)/sd;
target = [0,1,sum(p.*(centeredMeans.^3+3*centeredMeans.*(sds/sd).^2)), ...
    sum(p.*(centeredMeans.^4+6*centeredMeans.^2.*(sds/sd).^2+3*(sds/sd).^4))];
for order = 1:4
    expected = sum(P.*z.^order,2)-target(order);
    verifyEqual(testCase,d.standardizedMomentErrors(:,order),expected,'AbsTol',1e-10);
end
for row = 1:31
    matched = d.matchedMoments(row);
    if matched > 0
        verifyLessThanOrEqual(testCase,max(abs(d.standardizedMomentErrors(row,1:matched))),d.momentTolerance);
    end
end
end

function testAggregateFallbackWarning(testCase)
identifier = 'discretization:discreteGaussianMixtureAR:MomentFallback';
state = warning('on',identifier);
cleanup = onCleanup(@() warning(state));
verifyWarning(testCase,@() discretization.discreteGaussianMixtureAR( ...
    0,0.95,[0.05,0.95],[-0.8,0.8*0.05/0.95],[0.2,0.03],9,4,'even'),identifier);
end

function testInfeasibleVarianceIsReported(testCase)
[P,X,d] = discretization.discreteGaussianMixtureAR( ...
    0,0.4,1,0,0.1,9,2,'even',0.5);
verifyMarkov(testCase,P);
verifyEqual(testCase,d.matchedMoments,ones(9,1));
verifyEqual(testCase,d.downgradedRows,(1:9)');
verifyLessThanOrEqual(testCase,max(abs(d.standardizedMomentErrors(:,1))),d.momentTolerance);
verifyLessThan(testCase,d.standardizedMomentErrors(:,2),-0.5*ones(9,1));
verifyEqual(testCase,P*X',0.4*X','AbsTol',0.1*d.momentTolerance);
end

function testInfeasibleAR2MeanIsReported(testCase)
A = [0.8959,-0.3990];
[P,X,d] = discretization.discreteGaussianMixtureAR( ...
    0,A,1,0,0.1,5,2,'even',3);
outside = abs(A*X)' > max(abs(X(1,:)));
verifyTrue(testCase,any(outside));
verifyMarkov(testCase,P);
verifyEqual(testCase,d.matchedMoments(outside),zeros(nnz(outside),1));
end

function testProbabilityRoundoffAndZeroWeight(testCase)
base = {0,0.4,[0.25,0.75],[-0.3,0.1],[0.15,0.08],15,2,'even',4};
[P,X] = discretization.discreteGaussianMixtureAR(base{:});
rounded = base;
rounded{3} = [0.25;0.75+5e-14];
rounded{4} = rounded{4}'; rounded{5} = rounded{5}';
[roundP,roundX] = discretization.discreteGaussianMixtureAR(rounded{:});
verifyEqual(testCase,roundX,X,'AbsTol',1e-12);
verifyEqual(testCase,roundP,P,'AbsTol',1e-8);
[zeroP,zeroX] = discretization.discreteGaussianMixtureAR( ...
    0,0.4,[0.25,0.75,0],[-0.3,0.1,10],[0.15,0.08,0.05],15,2,'even',4);
verifyEqual(testCase,zeroX,X);
verifyEqual(testCase,zeroP,P,'AbsTol',1e-12);
end

function testNonzeroInnovationMeanConvention(testCase)
[P,X,d] = discretization.discreteGaussianMixtureAR( ...
    1,0.4,[0.25,0.75],[-0.1,0.06],[0.15,0.08],31,2,'even',4);
verifyEqual(testCase,d.innovationMean,0.02,'AbsTol',1e-14);
verifyEqual(testCase,d.unconditionalMean,1+0.02/(1-0.4),'AbsTol',1e-14);
rows = d.matchedMoments >= 1;
verifyEqual(testCase,P(rows,:)*X',1*(1-0.4)+0.4*X(rows)'+0.02, ...
    'AbsTol',d.innovationSD*d.momentTolerance);
end

function testAR2LagStructure(testCase)
[P,X,d] = discretization.discreteGaussianMixtureAR( ...
    0,[0.5,0.2],[0.25,0.75],[-0.3,0.1],[0.15,0.08],5,2,'even',3);
verifySize(testCase,P,[25,25]);
verifySize(testCase,X,[2,25]);
verifyMarkov(testCase,P);
for row = 1:25
    verifyEqual(testCase,P(row,X(2,:) ~= X(1,row)),zeros(1,20));
end
verifySize(testCase,d.matchedMoments,[25,1]);
end

function testLegacyDiagnostics(testCase)
args = {0,0.4,[0.25,0.75],[-0.3,0.1],[0.15,0.08],9,2,'even',4};
[P,X,d] = discretization.discreteGaussianMixtureAR(args{:});
[legacyP,legacyX,legacyDiagnostics] = discreteGMAR(args{:});
verifyEqual(testCase,legacyP,P);
verifyEqual(testCase,legacyX,X);
verifyEqual(testCase,legacyDiagnostics,d);
[aliasP,aliasX,aliasDiagnostics] = discretization.discreteARWithGaussianMixtureShocks(args{:});
verifyEqual(testCase,aliasP,P);
verifyEqual(testCase,aliasX,X);
verifyEqual(testCase,aliasDiagnostics,d);
end

function testInvalidInputs(testCase)
args = {0,0.4,[0.25,0.75],[-0.3,0.1],[0.15,0.08],9,2,'even',4};
cases = {
    1,NaN,'MATLAB:discreteGaussianMixtureAR:expectedFinite';
    2,NaN,'MATLAB:discreteGaussianMixtureAR:expectedFinite';
    3,[-0.1,1.1],'MATLAB:discreteGaussianMixtureAR:expectedNonnegative';
    3,[0.2,0.2],'discretization:discreteGaussianMixtureAR:InvalidProbabilities';
    4,0,'discretization:discreteGaussianMixtureAR:InvalidMixture';
    4,[NaN,0.1],'MATLAB:discreteGaussianMixtureAR:expectedFinite';
    5,[0,0.08],'MATLAB:discreteGaussianMixtureAR:expectedPositive';
    6,2,'MATLAB:discreteGaussianMixtureAR:notGreaterEqual';
    6,Inf,'MATLAB:discreteGaussianMixtureAR:expectedFinite';
    7,5,'MATLAB:discreteGaussianMixtureAR:notLessEqual';
    8,'invalid','discretization:discreteGaussianMixtureAR:InvalidMethod';
    9,-2,'MATLAB:discreteGaussianMixtureAR:expectedPositive';
    2,1,'discretization:discreteGaussianMixtureAR:NonstationaryProcess'
};
for row = 1:size(cases,1)
    invalid = args;
    invalid{cases{row,1}} = cases{row,2};
    verifyError(testCase,@() discretization.discreteGaussianMixtureAR(invalid{:}),cases{row,3});
end
end

function verifyMarkov(testCase,P)
verifyTrue(testCase,all(isfinite(P),'all'));
verifyGreaterThanOrEqual(testCase,P,zeros(size(P)));
verifyEqual(testCase,sum(P,2),ones(size(P,1),1),'AbsTol',1e-12);
end
