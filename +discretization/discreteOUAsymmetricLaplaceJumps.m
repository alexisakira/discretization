function [P,X,info] = discreteOUAsymmetricLaplaceJumps( ...
    kappa,sigma,jumpIntensity,upperRate,lowerRate,Delta,gridOrN)
%DISCRETEOUASYMMETRICLAPLACEJUMPS ME discretization of a jump-OU process.
%   [P,X,INFO] = discretization.discreteOUAsymmetricLaplaceJumps(...
%       KAPPA,SIGMA,JUMPINTENSITY,UPPERRATE,LOWERRATE,DELTA,N)
%   constructs an N-state maximum-entropy approximation to
%
%       X(t+Delta) = rho*X(t) + eta_Delta + J_Delta - E[J_Delta],
%
%   where rho = exp(-KAPPA*DELTA), eta_Delta is the exact Gaussian OU
%   innovation, and J_Delta is an aggregate asymmetric-Laplace jump.
%   JUMPINTENSITY is a jump-arrival intensity per unit of calendar time;
%   UPPERRATE and LOWERRATE are the positive- and negative-tail rates of a
%   primitive asymmetric-Laplace jump.
%
%   The number M_Delta of primitive jumps in a step has the geometric law
%
%       Pr(M_Delta=m) = q_Delta*(1-q_Delta)^m,  m = 0,1,...,
%       q_Delta = exp(-JUMPINTENSITY*DELTA).
%
%   Conditional on M_Delta >= 1, geometric closure implies that the sum is
%   asymmetric Laplace with rates a_Delta and b_Delta satisfying
%
%       a_Delta*b_Delta = q_Delta*UPPERRATE*LOWERRATE,
%       a_Delta-b_Delta = UPPERRATE-LOWERRATE.
%
%   This finite-Delta model permits multiple jumps within a step. As Delta
%   tends to zero, it converges locally to a compensated compound-Poisson OU
%   process with intensity JUMPINTENSITY and primitive asymmetric-Laplace
%   jumps. For a horizon m*Delta, use P^m. At a finite Delta this is a
%   splitting discretization: jumps are aggregated at the end of each step,
%   so taking a smaller Delta better approximates continuous jump timing.
%
%   Supplying an increasing vector GRIDORN instead of N uses that fixed
%   grid. This is the appropriate form during likelihood optimization. The
%   automatic N-point grid has endpoints +/-OMEGA*sqrt(N-1), where OMEGA is
%   the stationary standard deviation implied by the Delta-step moments.
%
%   Each row of P exactly matches the conditional mean rho*x and the total
%   conditional variance, subject to numerical solver tolerance. The prior
%   in the relative-entropy projection is the exact mixture of a Gaussian
%   no-jump density and a normal--asymmetric-Laplace jump density.
%
%   Example (quarterly matrix and annual transition):
%       [Pq,X] = discretization.discreteOUAsymmetricLaplaceJumps( ...
%           0.05,0.27,0.10,2.7,1.05,0.25,101);
%       Pannual = Pq^4;
%
%   INFO reports the Delta-step parameters, feasibility margins, and moment
%   errors. Setting JUMPINTENSITY=0 reproduces the Gaussian OU construction.

if nargin < 7
    error('discretization:discreteOUAsymmetricLaplaceJumps:NotEnoughInputs', ...
        ['Supply kappa, sigma, jumpIntensity, upperRate, lowerRate, ', ...
        'Delta, and N or a grid.'])
end

validateattributes(kappa,{'numeric'}, ...
    {'real','finite','scalar','positive'},mfilename,'kappa',1)
validateattributes(sigma,{'numeric'}, ...
    {'real','finite','scalar','positive'},mfilename,'sigma',2)
validateattributes(jumpIntensity,{'numeric'}, ...
    {'real','finite','scalar','nonnegative'},mfilename,'jumpIntensity',3)
validateattributes(upperRate,{'numeric'}, ...
    {'real','finite','scalar','positive'},mfilename,'upperRate',4)
validateattributes(lowerRate,{'numeric'}, ...
    {'real','finite','scalar','positive'},mfilename,'lowerRate',5)
validateattributes(Delta,{'numeric'}, ...
    {'real','finite','scalar','positive'},mfilename,'Delta',6)

rho = exp(-kappa*Delta);
oneMinusRhoSquared = -expm1(-2*kappa*Delta);
gaussianVariance = (sigma^2/(2*kappa))*oneMinusRhoSquared;
gaussianSD = sqrt(gaussianVariance);

noJumpProbability = exp(-jumpIntensity*Delta);
jumpProbability = -expm1(-jumpIntensity*Delta);
if noJumpProbability == 0
    error('discretization:discreteOUAsymmetricLaplaceJumps:StepTooLarge', ...
        ['jumpIntensity*Delta is too large to retain a positive no-jump ', ...
        'probability. Reduce Delta.'])
end

[aggregateUpperRate,aggregateLowerRate] = aggregateTailRates( ...
    noJumpProbability,upperRate,lowerRate);
aggregateJumpMean = 1/aggregateUpperRate-1/aggregateLowerRate;
aggregateJumpVariance = ...
    1/aggregateUpperRate^2+1/aggregateLowerRate^2;

centeredJumpVariance = jumpProbability*aggregateJumpVariance+ ...
    jumpProbability*noJumpProbability*aggregateJumpMean^2;
conditionalVariance = gaussianVariance+centeredJumpVariance;
stationaryVariance = conditionalVariance/oneMinusRhoSquared;
stationarySD = sqrt(stationaryVariance);

if any(~isfinite([gaussianVariance,aggregateUpperRate, ...
        aggregateLowerRate,aggregateJumpMean,aggregateJumpVariance, ...
        conditionalVariance,stationaryVariance])) || ...
        conditionalVariance <= 0 || stationarySD <= 0
    error('discretization:discreteOUAsymmetricLaplaceJumps:InvalidScale', ...
        'The implied transition or stationary moments are invalid.')
end

automaticGrid = isscalar(gridOrN);
if automaticGrid
    validateattributes(gridOrN,{'numeric'}, ...
        {'real','finite','scalar','integer','>=',3},mfilename,'N',7)
    N = gridOrN;
    gridLimit = stationarySD*sqrt(N-1);
    if ~isfinite(gridLimit)
        error('discretization:discreteOUAsymmetricLaplaceJumps:InvalidGrid', ...
            'The automatic grid endpoints must be finite.')
    end
    X = linspace(-gridLimit,gridLimit,N);
else
    validateattributes(gridOrN,{'numeric'}, ...
        {'real','finite','vector'},mfilename,'X',7)
    X = gridOrN(:)';
    N = numel(X);
    if N < 3 || any(diff(X) <= 0)
        error('discretization:discreteOUAsymmetricLaplaceJumps:InvalidGrid', ...
            'The grid must contain at least three strictly increasing points.')
    end
end

gridDifference = diff(X);
quadratureWeight = [gridDifference(1), ...
    (X(3:end)-X(1:end-2))/2,gridDifference(end)];

minimumSlack = Inf;
maximumSlack = Inf;
for row = 1:N
    conditionalMean = rho*X(row);
    [lowerSlack,upperSlack] = checkMomentFeasibility( ...
        X,conditionalMean,conditionalVariance,row);
    minimumSlack = min(minimumSlack,lowerSlack);
    maximumSlack = min(maximumSlack,upperSlack);
end

P = NaN(N);
dualStart = zeros(2,1);
scaledMomentError = NaN(2,N);
mixtureCenterAdjustment = jumpProbability*aggregateJumpMean;

for row = 1:N
    conditionalMean = rho*X(row);
    mixtureCenter = conditionalMean-mixtureCenterAdjustment;
    z = X-mixtureCenter;

    logGaussian = -0.5*log(2*pi)-log(gaussianSD) ...
        -0.5*(z/gaussianSD).^2;
    if jumpProbability > 0
        logJump = normalAsymmetricLaplaceLogPdf( ...
            z,gaussianSD,aggregateUpperRate,aggregateLowerRate);
        logDensity = logAddExp( ...
            log(noJumpProbability)+logGaussian, ...
            log(jumpProbability)+logJump);
    else
        logDensity = logGaussian;
    end

    logPrior = log(quadratureWeight)+logDensity;
    if ~any(isfinite(logPrior))
        error('discretization:discreteOUAsymmetricLaplaceJumps:InvalidPrior', ...
            'The transition-density prior underflowed in row %d.',row)
    end
    prior = exp(logPrior-max(logPrior));
    prior = max(prior,realmin);
    prior = prior/sum(prior);

    momentFunction = @(state) [ ...
        (state-conditionalMean)/stationarySD; ...
        ((state-conditionalMean)/stationarySD).^2];
    targetMoments = [0;conditionalVariance/stationaryVariance];
    [probability,dualStart,rowMomentError] = maximumEntropyWeightsCore( ...
        X,momentFunction,targetMoments,prior,dualStart);

    if any(~isfinite(probability)) || norm(rowMomentError,Inf) > 1e-7
        error(['discretization:discreteOUAsymmetricLaplaceJumps:', ...
            'MomentMatchingFailure'], ...
            'Two-moment matching failed in transition row %d.',row)
    end
    P(row,:) = probability;
    scaledMomentError(:,row) = rowMomentError;
end

P = max(P,0);
P = P./sum(P,2);

conditionalMeanVector = rho*X(:);
discreteMean = P*X(:);
discreteVariance = P*(X(:).^2)-discreteMean.^2;

info = struct;
info.Delta = Delta;
info.rho = rho;
info.jumpIntensity = jumpIntensity;
info.noJumpProbability = noJumpProbability;
info.jumpProbability = jumpProbability;
info.primitiveUpperRate = upperRate;
info.primitiveLowerRate = lowerRate;
info.aggregateUpperRate = aggregateUpperRate;
info.aggregateLowerRate = aggregateLowerRate;
info.aggregateJumpMean = aggregateJumpMean;
info.aggregateJumpVariance = aggregateJumpVariance;
info.gaussianVariance = gaussianVariance;
info.centeredJumpVariance = centeredJumpVariance;
info.conditionalVariance = conditionalVariance;
info.stationaryVariance = stationaryVariance;
info.stationarySD = stationarySD;
info.automaticGrid = automaticGrid;
info.minimumLowerFeasibilitySlack = minimumSlack;
info.minimumUpperFeasibilitySlack = maximumSlack;
info.maximumRowSumError = max(abs(sum(P,2)-1));
info.maximumMeanError = max(abs(discreteMean-conditionalMeanVector));
info.maximumVarianceError = max(abs( ...
    discreteVariance-conditionalVariance));
info.maximumScaledSolverMomentError = max(abs(scaledMomentError),[],'all');

end


function [upperRateDelta,lowerRateDelta] = aggregateTailRates( ...
    noJumpProbability,upperRate,lowerRate)
% Rates of a geometric sum conditional on at least one primitive jump.
rateDifference = upperRate-lowerRate;
rateProduct = noJumpProbability*upperRate*lowerRate;
discriminant = hypot(rateDifference,2*sqrt(rateProduct));

% Avoid cancellation when the primitive rates differ substantially.
if rateDifference >= 0
    upperRateDelta = 0.5*(discriminant+rateDifference);
    lowerRateDelta = rateProduct/upperRateDelta;
else
    lowerRateDelta = 0.5*(discriminant-rateDifference);
    upperRateDelta = rateProduct/lowerRateDelta;
end
end


function logDensity = normalAsymmetricLaplaceLogPdf(z,s,a,b)
% Log density of N(0,s^2) plus AL(a,b), evaluated without normcdf.
logCoefficient = log(a)+log(b)-log(a+b);
rightArgument = z/s-a*s;
leftArgument = -z/s-b*s;

rightTerm = -a*z+0.5*(a*s)^2+standardNormalLogCdf(rightArgument);
leftTerm = b*z+0.5*(b*s)^2+standardNormalLogCdf(leftArgument);
logDensity = logCoefficient+logAddExp(rightTerm,leftTerm);
end


function value = standardNormalLogCdf(z)
% Stable log(Phi(z)) using erfcx in the lower tail.
value = zeros(size(z));
negative = z < 0;
value(negative) = log(0.5)-0.5*z(negative).^2+ ...
    log(erfcx(-z(negative)/sqrt(2)));
value(~negative) = log1p(-0.5*erfc(z(~negative)/sqrt(2)));
end


function value = logAddExp(a,b)
% Elementwise log(exp(a)+exp(b)).
maximum = max(a,b);
value = maximum+log(exp(a-maximum)+exp(b-maximum));
end


function [lowerSlack,upperSlack] = checkMomentFeasibility( ...
    grid,meanValue,varianceValue,row)
% Strict necessary and sufficient bounds for two-moment matching.
scale = max([1,max(abs(grid))^2,varianceValue]);
tolerance = 1e-12*scale;

if meanValue <= grid(1)+tolerance || meanValue >= grid(end)-tolerance
    error('discretization:discreteOUAsymmetricLaplaceJumps:InfeasibleGrid', ...
        'The conditional mean lies outside the grid in row %d.',row)
end

leftIndex = find(grid <= meanValue,1,'last');
if grid(leftIndex) == meanValue
    minimumVariance = 0;
else
    minimumVariance = (meanValue-grid(leftIndex))* ...
        (grid(leftIndex+1)-meanValue);
end
maximumVariance = (meanValue-grid(1))*(grid(end)-meanValue);
lowerSlack = varianceValue-minimumVariance;
upperSlack = maximumVariance-varianceValue;

if lowerSlack <= tolerance || upperSlack <= tolerance
    error('discretization:discreteOUAsymmetricLaplaceJumps:InfeasibleGrid', ...
        ['The conditional variance is not strictly feasible on the grid ', ...
        'in row %d.'],row)
end
end
