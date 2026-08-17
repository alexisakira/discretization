function [x,p] = discreteNP(varargin)
%DISCRETENP Compatibility alias for the descriptive package function.
warning('discretization:deprecatedFunction', ...
    ['discreteNP is deprecated. Use ' ...
    'discretization.momentMatchedDistribution instead.'])
[x,p] = discretization.momentMatchedDistribution(varargin{:});
end
