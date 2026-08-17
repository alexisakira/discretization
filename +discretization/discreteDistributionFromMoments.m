function [x,p] = discreteDistributionFromMoments(varargin)
%DISCRETEDISTRIBUTIONFROMMOMENTS Compatibility name.
warning('discretization:deprecatedFunction', ...
    ['discretization.discreteDistributionFromMoments is deprecated. ' ...
    'Use discretization.momentMatchedDistribution instead.'])
[x,p] = discretization.momentMatchedDistribution(varargin{:});
end
