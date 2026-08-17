%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% entropyObjective
% (c) 2016 Leland E. Farmer and Alexis Akira Toda
% 
% Purpose: 
%       Compute the maximum entropy objective function used in
%       maximumEntropyWeights
%
% Usage:
%       obj = entropyObjective(lambda,Tx,TBar,q)
%
% Inputs:
% lambda    - (L x 1) vector of values of the dual problem variables
% Tx        - (L x N) matrix of moments evaluated at the grid points
%             specified in maximumEntropyWeights
% TBar      - (L x 1) vector of moments of the underlying distribution
%             which should be matched 
% q         - (1 X N) vector of prior weights for each point in the grid.
%
% Outputs:
% obj       - scalar value of the log objective evaluated at lambda
% Optional (useful for optimization routines):
% gradObj   - (L x 1) gradient vector of the objective function evaluated
%             at lambda
% hessianObj- (L x L) hessian matrix of the objective function evaluated at
%             lambda
% p         - (1 x N) normalized probability weights at lambda
%
% Version 1.2: June 7, 2016
%
% Version 2.0: August 17, 2026
%
% Evaluate the equivalent log-sum-exp objective to avoid numerical
% overflow and return its analytic gradient and Hessian.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

function [obj,gradObj,hessianObj,p] = entropyObjective(lambda,Tx,TBar,q)

% Some error checking

if nargin < 4
    error('You must provide 4 arguments to entropyObjective.')
end

[L,N] = size(Tx);

if numel(lambda) ~= L || numel(TBar) ~= L || numel(q) ~= N
    error('Dimensions of inputs are not compatible.')
end
if any(~isfinite(lambda)) || any(~isfinite(Tx),'all') || ...
        any(~isfinite(TBar)) || any(~isfinite(q))
    error('Inputs to entropyObjective must be finite.')
end
if any(q < 0) || ~any(q > 0)
    error('Prior weights must be nonnegative with at least one positive value.')
end

lambda = lambda(:);
TBar = TBar(:);
q = q(:)';

% Compute the log-sum-exp objective. Subtracting the largest log weight
% prevents overflow without changing the normalized probabilities.

Tdiff = Tx-TBar;
logWeights = log(q) + lambda'*Tdiff;
shift = max(logWeights);
scaledWeights = exp(logWeights-shift);
normalizer = sum(scaledWeights);
obj = shift + log(normalizer);

% Compute gradient of objective function

if nargout > 1
    p = scaledWeights/normalizer;
    weightedDifferences = Tdiff.*p;
    gradObj = sum(weightedDifferences,2);
end

% Compute hessian of objective function

if nargout > 2
    hessianObj = weightedDifferences*Tdiff' - gradObj*gradObj';
    hessianObj = (hessianObj+hessianObj')/2;
end

end
