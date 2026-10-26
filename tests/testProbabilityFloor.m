function tests = testProbabilityFloor
tests = functiontests(localfunctions);
end
function setupOnce(~)
addpath(fileparts(fileparts(mfilename('fullpath'))));
end
function testGaussianTailsAndDensity(testCase)
rho=.9; sigma=.1;
[P,X,d]=discretization.discreteAR(0,rho,sigma,81);
z=(X-rho*X')/sigma;
verifyLessThan(testCase,max(P(abs(z)>10)),1e-15);
verifyEqual(testCase,d.matchedMoments,2*ones(81,1));
Q=exp(-.5*z.^2); Q=Q./sum(Q,2);
% Density and exact moments differ at truncated boundary rows. Where the
% Gaussian tail outside the grid is negligible, no tilt is necessary.
interior=min((X(end)-rho*X')/sigma,(rho*X'-X(1))/sigma)>8;
verifyTrue(testCase,any(interior));
verifyEqual(testCase,P(interior,:),Q(interior,:),'AbsTol',1e-11);
for row=1:81
    verifyLessThanOrEqual(testCase,max(abs(d.standardizedMomentErrors(row,1:d.matchedMoments(row)))),d.momentTolerance);
end
[legacy,~]=discretization.discreteAR(0,rho,sigma,81,'ProbabilityFloor',1e-8);
verifyGreaterThan(testCase,max(legacy(abs(z)>10)),1e-12);
end
function testExplicitLegacyReference(testCase)
root=fileparts(fileparts(mfilename('fullpath')));
f=load(fullfile(root,'reference_data','matlab_r2024b_performance.mat'));
[P,X]=discretization.discreteAR(0,.9,.1,9,'even',4,[],'ProbabilityFloor',1e-8);
verifyEqual(testCase,P,f.ar4Transition,'AbsTol',1e-8);
verifyEqual(testCase,X,f.ar4Grid,'AbsTol',1e-12);
[P,X]=discretization.discreteVAR(zeros(2,1),[.9809,.0028;.0410,.9648],diag([.0087^2,.0262^2]),9,'ProbabilityFloor',1e-8);
verifyEqual(testCase,P,f.varTransition,'AbsTol',1e-8);
verifyEqual(testCase,X,f.varGrid,'AbsTol',1e-9);
[P,X]=discretization.discreteGaussianMixtureAR(.0555,.5854,[.1628,.8372],[-.0039,.0008],[.1293,.0300],9,'ProbabilityFloor',1e-8);
verifyEqual(testCase,P,f.gmar1Transition,'AbsTol',1e-8);
verifyEqual(testCase,X,f.gmar1Grid,'AbsTol',1e-12);
[P,X]=discretization.discreteStochasticVolatilityAR(.95,.9,.007,.06,9,5,'ProbabilityFloor',1e-8);
verifyEqual(testCase,P,f.svTransition,'AbsTol',1e-8);
verifyEqual(testCase,X,f.svGrid,'AbsTol',1e-12);
end
function testOptionalMixtureMethods(testCase)
for method={'gauss-legendre','clenshaw-curtis','gauss-hermite','GMQ'}
    [P,~,d]=discretization.discreteGaussianMixtureAR(0,.4,[.4,.6],[-.1,.1],[.1,.2],21,4,method{1},4);
    verifyEqual(testCase,sum(P,2),ones(21,1),'AbsTol',1e-12);
    for row=1:21
        if d.matchedMoments(row)>0
            verifyLessThanOrEqual(testCase,max(abs(d.standardizedMomentErrors(row,1:d.matchedMoments(row)))),d.momentTolerance);
        end
    end
end
end
function testFallbackPointMass(testCase)
[P,X,d]=discretization.discreteGaussianMixtureAR(0,.4,1,10,.1,9);
verifyEqual(testCase,d.matchedMoments,zeros(9,1));
verifyEqual(testCase,P(:,end),ones(9,1));
verifyGreaterThan(testCase,X(end),X(1));
end
function testJumpOUMomentsAndZeroJump(testCase)
[P,X,info]=discretization.discreteOUAsymmetricLaplaceJumps(.7,.3,.1,2.7,1.05,.25,21);
means=P*X'; variance=sum(P.*(X-info.rho*X').^2,2);
verifyEqual(testCase,means,info.rho*X','AbsTol',1e-7*info.stationarySD);
verifyEqual(testCase,variance,info.conditionalVariance*ones(21,1),'AbsTol',1e-7*info.stationaryVariance);
[zero,Z]=discretization.discreteOUAsymmetricLaplaceJumps(.7,.3,0,2.7,1.05,.25,21);
[gaussian,G]=discretization.discreteOU(.7,.3,.25,21);
verifyEqual(testCase,Z,G,'AbsTol',1e-14);
verifyEqual(testCase,zero,gaussian,'AbsTol',1e-8);
verifyError(testCase,@() discretization.discreteOU(.7,.3,.01,[-1,0,1]),'discretization:discreteOU:InfeasibleGrid');
end
function testCIRFellerEqualityAndInfeasibleGrid(testCase)
% Exact binary inputs avoid an ambiguous floating-point Feller boundary.
a=.5; b=.0625; sigma=.25; gap=.25; current=.02;
density=discretization.cirTransitionDensity([0,.01],current,a,b,sigma,gap);
c=2*a/(-expm1(-a*gap)*sigma^2);
verifyEqual(testCase,density(1),c*exp(-c*current*exp(-a*gap)),'AbsTol',1e-12);
verifyError(testCase,@() discretization.discreteCIR(.5,.03,.1,.0001,3), ...
    'discretization:discreteCIR:EmptyFeasibleSet');
end
