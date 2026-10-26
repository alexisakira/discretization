function [P,X] = discreteOU(kappa,sigma,Delta,gridOrN)
%DISCRETEOU Moment-matched discretization of a centered OU process.
%   [P,X] = discretization.discreteOU(kappa,sigma,Delta,N) discretizes
%   dX_t = -kappa*X_t*dt + sigma*dW_t on N evenly spaced points. The
%   automatic grid has endpoints +/-omega*sqrt(N-1), where
%   omega = sigma/sqrt(2*kappa) is the stationary standard deviation.
%   Its spacing d/omega = 2/sqrt(N-1) is the upper value in (C.5) and
%   is feasible for every Delta > 0.
%
%   [P,X] = discretization.discreteOU(kappa,sigma,Delta,X) instead uses
%   the supplied increasing grid. This form permits a common grid across
%   parameter values and observation gaps in a likelihood calculation.
%
%   P(i,j) is the probability of moving from X(i) to X(j). Each row
%   matches the exact conditional mean and variance by maximum entropy,
%   using the Gaussian transition density as its prior.

if nargin < 4
    error('discretization:discreteOU:NotEnoughInputs', ...
        'You must supply kappa, sigma, Delta, and N or a grid.')
end
validateattributes(kappa,{'numeric'}, ...
    {'real','finite','scalar','positive'},mfilename,'kappa',1)
validateattributes(sigma,{'numeric'}, ...
    {'real','finite','scalar','positive'},mfilename,'sigma',2)
validateattributes(Delta,{'numeric'}, ...
    {'real','finite','scalar','positive'},mfilename,'Delta',3)

omega = sigma/sqrt(2*kappa);
if ~isfinite(omega) || omega <= 0
    error('discretization:discreteOU:InvalidScale', ...
        'The stationary standard deviation must be finite and positive.')
end

if isscalar(gridOrN)
    validateattributes(gridOrN,{'numeric'}, ...
        {'real','finite','scalar','integer','>=',3},mfilename,'N',4)
    N = gridOrN;
    gridLimit = omega*sqrt(N-1);
    if ~isfinite(gridLimit)
        error('discretization:discreteOU:InvalidGrid', ...
            'The automatic grid endpoints must be finite.')
    end
    X = linspace(-gridLimit,gridLimit,N);
else
    validateattributes(gridOrN,{'numeric'}, ...
        {'real','finite','vector'},mfilename,'X',4)
    X = gridOrN(:)';
    N = numel(X);
    if N < 3 || any(diff(X) <= 0)
        error('discretization:discreteOU:InvalidGrid', ...
            'The grid must contain at least three increasing points.')
    end
end

rho = exp(-kappa*Delta);
conditionalVariance = omega^2*(-expm1(-2*kappa*Delta));
conditionalSD = sqrt(conditionalVariance);
gridDifference = diff(X);
quadratureWeight = [gridDifference(1), ...
    (X(3:end)-X(1:end-2))/2,gridDifference(end)];

% Reject an infeasible fixed grid before solving any maximum-entropy row.
% During likelihood optimization a trial parameter can make only a late row
% infeasible. Checking every row first avoids spending most of the objective
% evaluation on transition rows that will ultimately be discarded.
for row = 1:N
    conditionalMean = rho*X(row);
    checkMomentFeasibility(X,conditionalMean,conditionalVariance,row)
end

P = NaN(N);
lambda0 = zeros(2,1);
for row = 1:N
    conditionalMean = rho*X(row);

    logPrior = log(quadratureWeight) ...
        -0.5*((X-conditionalMean)/conditionalSD).^2;
    if ~any(isfinite(logPrior))
        error('discretization:discreteOU:InvalidPrior', ...
            'The Gaussian prior underflowed in transition row %d.',row)
    end
    prior = exp(logPrior-max(logPrior));
    prior = max(prior,realmin);
    prior = prior/sum(prior);

    momentFunction = @(state) [ ...
        (state-conditionalMean)/omega; ...
        ((state-conditionalMean)/omega).^2];
    targetMoments = [0;conditionalVariance/omega^2];
    [probability,lambda0,momentError] = maximumEntropyWeightsCore( ...
        X,momentFunction,targetMoments,prior,lambda0);
    if any(~isfinite(probability)) || norm(momentError,Inf) > 1e-7
        error('discretization:discreteOU:MomentMatchingFailure', ...
            'Two-moment matching failed in transition row %d.',row)
    end
    P(row,:) = probability;
end

P = max(P,0);
P = P./sum(P,2);

end


function checkMomentFeasibility(grid,meanValue,varianceValue,row)
if meanValue <= grid(1) || meanValue >= grid(end)
    error('discretization:discreteOU:InfeasibleGrid', ...
        'The conditional mean is outside the grid in row %d.',row)
end

leftIndex = find(grid <= meanValue,1,'last');
if grid(leftIndex) == meanValue
    minimumVariance = 0;
else
    minimumVariance = (meanValue-grid(leftIndex))* ...
        (grid(leftIndex+1)-meanValue);
end
maximumVariance = (meanValue-grid(1))*(grid(end)-meanValue);
if varianceValue <= minimumVariance || varianceValue >= maximumVariance
    error('discretization:discreteOU:InfeasibleGrid', ...
        'The conditional variance is infeasible in row %d.',row)
end
end
