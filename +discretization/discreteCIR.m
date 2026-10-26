%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% discreteCIR
% (c) 2019--2026 Alexis Akira Toda
%
% Purpose:
%       Compute a finite-state Markov chain approximation to the
%       Cox-Ingersoll-Ross model
%       dX_t = kappa*(mu-X_t)*dt + sigma*sqrt(X_t)*dW_t.
%       The routine uses an automatically selected evenly spaced grid
%       x_n = d*n and matches the first two conditional moments exactly.
%
% Usage:
%       [P,X] = discretization.discreteCIR(...
%           kappa,mu,sigma,Delta,N)
%
% Inputs:
% kappa     - mean reversion parameter
% mu        - unconditional mean
% sigma     - volatility parameter
% Delta     - length of one period
% N         - number of grid points (integer greater than or equal to 3)
%
% Outputs:
% P         - N-by-N transition probability matrix
% X         - 1-by-N vector of grid points
%
% Version 2.0, September 17, 2026
%
% Select the grid from the sufficient two-moment feasibility conditions.
% Among guaranteed-feasible grids, maximize stationary coverage subject to
% a small numerical margin from the boundary of the feasible interval.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [P,X] = discreteCIR(kappa,mu,sigma,Delta,N)

%% Input validation
if nargin < 5
    error('discretization:discreteCIR:NotEnoughInputs', ...
        'You must supply kappa, mu, sigma, Delta, and N.')
end

validateattributes(kappa,{'numeric'}, ...
    {'real','finite','scalar','positive'},mfilename,'kappa',1)
validateattributes(mu,{'numeric'}, ...
    {'real','finite','scalar','positive'},mfilename,'mu',2)
validateattributes(sigma,{'numeric'}, ...
    {'real','finite','scalar','positive'},mfilename,'sigma',3)
validateattributes(Delta,{'numeric'}, ...
    {'real','finite','scalar','positive'},mfilename,'Delta',4)
validateattributes(N,{'numeric'}, ...
    {'real','finite','scalar','integer','>=',3},mfilename,'N',5)

if sigma^2 > 2*kappa*mu
    error('discretization:discreteCIR:FellerCondition', ...
        'The Feller condition sigma^2 <= 2*kappa*mu must hold.')
end

%% Select the evenly spaced grid
rho = exp(-kappa*Delta);
c = -expm1(-kappa*Delta); % numerically stable evaluation of 1-rho
eta = sigma^2/(kappa*mu);

[rLower,rUpper] = feasibleSpacingInterval(rho,eta,N);
if isnan(rLower)
    error('discretization:discreteCIR:EmptyFeasibleSet', ...
        ['No grid x_n = d*n satisfies the sufficient two-moment ', ...
        'feasibility conditions for N = %d. Increase N.'],N)
end

% The strict feasibility conditions define an open interval. Stay a small
% distance from its boundary to avoid an ill-conditioned dual problem.
% Let the trim shrink with N so the selected grid still refines asymptotically.
trimFraction = min(1e-4,1/N);
intervalWidth = rUpper-rLower;
rLowerSafe = rLower+trimFraction*intervalWidth;
rUpperSafe = rUpper-trimFraction*intervalWidth;

% Unconstrained maximizer of stationary Gamma coverage on [d,N*d].
rCoverage = log(N)/(N-1);
r = min(rUpperSafe,max(rLowerSafe,rCoverage));
d = mu*r;
X = d*(1:N);

if any(feasibilityFunctions(r,rho,eta,N) <= 0)
    error('discretization:discreteCIR:GridSelectionFailure', ...
        'The automatically selected grid is not strictly feasible.')
end

%% Construct the transition matrix
P = NaN(N);
scalingFactor = X(end);
momentErrorTolerance = 1e-7;

for ii = 1:N
    x = X(ii);
    condMean = rho*x+c*mu;
    condVar = (sigma^2/kappa)*c*(rho*x+(mu/2)*c);

    % On an evenly spaced grid, equal quadrature weights cancel after
    % normalization, so the prior is proportional to the transition density.
    q = relativeReferenceWeights(cirLogDensity(X,x,kappa,mu,sigma,Delta));

    momentFunction = @(state) [ ...
        (state-condMean)./scalingFactor; ...
        ((state-condMean)./scalingFactor).^2];
    targetMoments = [0;condVar/scalingFactor^2];

    [probability,~,momentError] = maximumEntropyWeightsCore( ...
        X,momentFunction,targetMoments,q,zeros(2,1));
    if any(~isfinite(probability)) || ...
            norm(momentError,Inf) > momentErrorTolerance
        error('discretization:discreteCIR:MomentMatchingFailure', ...
            ['Two-moment matching failed numerically in row %d even ', ...
            'though the selected grid is theoretically feasible.'],ii)
    end
    P(ii,:) = probability;
end

end


function [rLower,rUpper] = feasibleSpacingInterval(rho,eta,N)
% Return the open interval of normalized feasible spacings r = d/mu.
domainLower = 1/N;
domainUpper = 1;
c = 1-rho;

coefficients = [ ...
    -1, 4*eta*c*rho, 2*eta*c^2; ...
    -(N-rho), N+1-2*rho-eta*rho, -c*(1+eta/2); ...
    N*(rho*N-1), N+1-rho*N*(2+eta), -c*(1+eta/2)];

breakpoints = [domainLower,domainUpper];
for row = 1:size(coefficients,1)
    candidateRoots = realRootsInInterval( ...
        coefficients(row,:),domainLower,domainUpper);
    breakpoints = [breakpoints,candidateRoots]; %#ok<AGROW>
end
breakpoints = mergeClosePoints(sort(breakpoints));

feasibleIntervals = zeros(0,2);
for interval = 1:(numel(breakpoints)-1)
    left = breakpoints(interval);
    right = breakpoints(interval+1);
    midpoint = (left+right)/2;
    if all(feasibilityFunctions(midpoint,rho,eta,N) > 0)
        feasibleIntervals(end+1,:) = [left,right]; %#ok<AGROW>
    end
end

if isempty(feasibleIntervals)
    rLower = NaN;
    rUpper = NaN;
elseif size(feasibleIntervals,1) == 1
    rLower = feasibleIntervals(1,1);
    rUpper = feasibleIntervals(1,2);
else
    error('discretization:discreteCIR:DisconnectedFeasibleSet', ...
        'The computed feasible set is unexpectedly disconnected.')
end

end


function values = feasibilityFunctions(r,rho,eta,N)
% Evaluate S(r), L_N(r), and U_N(r) from the grid-feasibility corollary.
c = 1-rho;
S = 4*eta*c*(rho*r+c/2)-r^2;
L = (1-r)*((N-rho)*r-c)-eta*(rho*r+c/2);
U = ((rho*N-1)*r+c)*(N*r-1)-eta*(rho*N*r+c/2);
values = [S,L,U];
end


function intervalRoots = realRootsInInterval(coefficients,lower,upper)
% Compute real roots strictly inside an interval, allowing a linear case.
scale = max(abs(coefficients));
coefficientTolerance = 100*eps(max(1,scale));
firstNonzero = find(abs(coefficients) > coefficientTolerance,1,'first');

if isempty(firstNonzero) || firstNonzero == numel(coefficients)
    intervalRoots = [];
    return
end

allRoots = roots(coefficients(firstNonzero:end));
realMask = abs(imag(allRoots)) <= ...
    1e-10*max(1,abs(real(allRoots)));
realRoots = real(allRoots(realMask));
intervalRoots = realRoots(realRoots > lower & realRoots < upper).';
end


function merged = mergeClosePoints(points)
% Merge numerically indistinguishable polynomial roots.
merged = points(1);
for index = 2:numel(points)
    tolerance = 1e-11*max([1,abs(points(index)),abs(merged(end))]);
    if points(index)-merged(end) > tolerance
        merged(end+1) = points(index); %#ok<AGROW>
    end
end
end
