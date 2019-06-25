%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% discreteCIR
% (c) 2019 Alexis Akira Toda
%
% Purpose:
%       Compute a finite-state Markov chain approximation to the
%       Cox-Ingersoll-Ross model
%       dr_t = a*(b-r_t)*dt + sigma*sqrt(r_t)*dW_t
%
% Usage:
%       [P,X] = discreteCIR(a,b,sigma,Delta,N,Coverage,method)
%
% Inputs:
% a         - mean reversion parameter
% b         - unconditional mean of interest rate
% sigma     - volatility parameter
% Delta     - length of one period; for example, if parameters a, b, sigma
%           are calibrated at annual frequency and you want to discretize
%           the model at quarterly frequency, set Delta = 1/4.
% N         - number of grid points
% Optional:
% Coverage  - coverage probability of grid (default = 0.999)
% method    - grid specification ('even' or 'exponential' (default))
%
% Outputs:
% P         - N x N transition probability matrix
% X         - 1 x N vector of grid
%
% Version 1.1, May 28, 2019
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

function [P,X] = discreteCIR(a,b,sigma,Delta,N,Coverage,method)

%% some error checking
% Make sure the correct number of arguments are provided
if nargin < 5
    error('You must supply at least 5 arguments to discreteCIR')
elseif nargin < 6 % Coverage not provided
    Coverage = 0.999; % cover 99.9% of state space
elseif nargin < 7 % grid choice not provided
    method = 'exponential';
elseif nargin >= 7
    if ~strcmp(method,'even')&&~strcmp(method,'exponential')
        error('Method must be either even or exponential')
    end
end

if (a <= 0)||(b <= 0)||(sigma <= 0)||(Delta <= 0)
    error('a, b, sigma, Delta must be positive')
end

if 2*a*b - sigma^2 <= 0
    error('It must be sigma^2 < 2*a*b')
end

% Check that N is a valid number of grid points
if ~isnumeric(N) || N < 2 || rem(N,1) ~= 0
    error('N must be a positive integer greater than 2')
end

if (Coverage <= 0)||(Coverage >= 1)
    error('coverage must be between 0 and 1')
end

%% construct the grid

alpha = 2*a*b/sigma^2;
beta = 2*a/sigma^2;

p = [1-Coverage 1+Coverage]/2;
pd = makedist('Gamma','a',alpha,'b',1/beta);
quantiles = icdf(pd,p);

if strcmp(method,'even')
    X = linspace(quantiles(1),quantiles(2),N); % evenly-spaced grid to cover coverage
    W = ones(size(X));
elseif strcmp(method,'exponential')
    X = exp(linspace(log(quantiles(1)),log(quantiles(2)),N)); % exponential grid
    W = X; % change of variable formula when using exponential grid
end
W = W/sum(W); % normalization (not necessary; just in case for numerical stability)

P = NaN(N);
scalingFactor = max(abs(X));
ea = exp(-a*Delta); % constant used in following calculation

for ii = 1:N
    r = X(ii); % current interest rate
    condMean = r*ea + b*(1-ea); % conditional mean
    condVar = (sigma^2/a)*(1 - ea)*(r*ea + b/2); % conditional variance
    TBar = [0 condVar]'; % conditional first and second moments (centered)
    q = W.*CIRpdf(X,r,a,b,sigma,Delta); % weighted conditional pdf (prior)
    % do maximum entropy discretization
    [p,~,momentError] = discreteApproximation(X,@(x) [(x-condMean)./scalingFactor;...
        ((x-condMean)./scalingFactor).^2],TBar./(scalingFactor.^(1:2)'),q,zeros(2,1));
    if norm(momentError) > 1e-5 % if 2 moments fail, then just match 1 moment
        warning('Failed to match first 2 moments. Just matching 1.')
        P(ii,:) = discreteApproximation(X,@(x)(x-condMean)/scalingFactor,0,q,0);
    else
        P(ii,:) = p;
    end
end

end