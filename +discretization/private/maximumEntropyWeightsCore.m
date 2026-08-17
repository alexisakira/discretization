%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% maximumEntropyWeightsCore
% (c) 2016 Leland E. Farmer and Alexis Akira Toda
% 
% Purpose: 
%       Compute a discrete state approximation to a distribution with known
%       moments, using the maximum entropy procedure proposed in Tanaka and
%       Toda (2013)
%
% Usage:
%       [p,lambdaBar,momentError] = maximumEntropyWeightsCore(...)
%           D,T,TBar,q,lambda0)
%
% Inputs:
% D         - (K x N) matrix of grid points. K is the dimension of the
%             domain. N is the number of points at which an approximation
%             is to be constructed.
% T         - A function handle which should accept arguments of dimension
%             (K x N) and return an (L x N) matrix of moments evaluated at
%             each grid point, where L is the number of moments to be
%             matched.
% TBar      - (L x 1) vector of moments of the underlying distribution
%             which should be matched
% Optional:
% q         - (1 X N) vector of prior weights for each point in D. The
%             default is for each point to have an equal weight.
% lambda0   - (L x 1) vector of initial guesses for the dual problem
%             variables. The default is a vector of zeros.
%
% Outputs:
% p         - (1 x N) vector of probabilties assigned to each grid point in
%             D.
% lambdaBar - (L x 1) vector of dual problem variables which solve the
%             maximum entropy problem
% momentError - (L x 1) vector of errors in moments (defined by moments of
%               discretization minus actual moments)
%
% Version 1.2: June 7, 2016
%
% Version 1.3: May 26, 2019
%
% Changed algorithm to 'trust-region' to use Hessian
%
% Version 1.4: September 27, 2023
%
% Changed fminunc option for Matlab 2023b
% Display warning if moment error is large
%
% Version 2.0: August 17, 2026
%
% Use a stable log-sum-exp objective and its analytic Hessian.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

function [p,lambdaBar,momentError] = maximumEntropyWeightsCore(D,T,TBar,q,lambda0)

% Input error checking

if nargin < 3
    error('You must provide at least 3 arguments to maximumEntropyWeights.')
end

N = size(D,2);

Tx = T(D);
L = size(Tx,1);

if size(Tx,2) ~= N || length(TBar) ~= L
    error('Dimension mismatch')
end

% Set default parameters if not provided
if nargin < 4
    q = ones(1,N)./N; % uniform distribution
end
if nargin < 5
    lambda0 = zeros(L,1);
end

TBar = TBar(:);
q = q(:)';
lambda0 = lambda0(:);

if numel(q) ~= N || any(~isfinite(q)) || any(q < 0) || ~any(q > 0)
    error('q must contain N nonnegative finite weights, including at least one positive weight.')
end
if numel(lambda0) ~= L || any(~isfinite(lambda0))
    error('lambda0 must contain one finite value for each target moment.')
end

% Compute maximum entropy discrete distribution

%options = optimset('TolFun',1e-10,'TolX',1e-10,'Display','off','GradObj','on','Hessian','on');
%options = optimset('TolFun',1e-10,'TolX',1e-10,'Display','off','Algorithm','trust-region');
persistent options fallbackOptions
if isempty(options)
    options = optimoptions('fminunc','TolFun',1e-10,'TolX',1e-10,...
        'Display','off','Algorithm','trust-region',...
        'SpecifyObjectiveGradient',true,'HessianFcn','objective');
    fallbackOptions = optimoptions('fminunc','TolFun',1e-10,'TolX',1e-10,...
        'Display','off','Algorithm','quasi-newton',...
        'SpecifyObjectiveGradient',true);
end

% Keep the best finite candidate across solver attempts. Infeasible moment
% targets can make the dual problem non-coercive, so a failed optimization
% must not replace a usable candidate with NaN or Inf.
lambdaBar = zeros(L,1);
[~,initialError] = entropyObjective(lambdaBar,Tx,TBar,q);
bestError = norm(initialError);

[candidate,candidateError] = tryEntropySolve( ...
    lambda0,Tx,TBar,q,options);
if candidateError < bestError
    lambdaBar = candidate;
    bestError = candidateError;
end

if bestError > 1e-5 && any(lambda0 ~= 0)
    [candidate,candidateError] = tryEntropySolve( ...
        zeros(L,1),Tx,TBar,q,options);
    if candidateError < bestError
        lambdaBar = candidate;
        bestError = candidateError;
    end
end

if bestError > 1e-5
    [candidate,candidateError] = tryEntropySolve( ...
        zeros(L,1),Tx,TBar,q,fallbackOptions);
    if candidateError < bestError
        lambdaBar = candidate;
    end
end

% Compute final probability weights and moment errors
[~,momentError,~,p] = entropyObjective(lambdaBar,Tx,TBar,q);

if norm(momentError) > 1e-5
    warning('Large moment error. Consider increasing number of points or expanding domain')
end

end


function [lambda,errorNorm] = tryEntropySolve(lambda0,Tx,TBar,q,options)
% Run one solver attempt without allowing a numerical failure to escape.
lambda = lambda0;
errorNorm = Inf;

try
    candidate = fminunc( ...
        @(dual) entropyObjective(dual,Tx,TBar,q),lambda0,options);
    if all(isfinite(candidate))
        [~,candidateError] = entropyObjective(candidate,Tx,TBar,q);
        candidateErrorNorm = norm(candidateError);
        if isfinite(candidateErrorNorm)
            lambda = candidate;
            errorNorm = candidateErrorNorm;
        end
    end
catch
    % The caller will retain a better finite candidate or the zero vector.
end

end

