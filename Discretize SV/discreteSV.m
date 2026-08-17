function [P,yxGrids] = discreteSV(varargin)
%DISCRETESV Compatibility alias for the descriptive package function.
warning('discretization:deprecatedFunction', ...
    ['discreteSV is deprecated. Use ' ...
    'discretization.discreteStochasticVolatilityAR instead.'])
[P,yxGrids] = discretization.discreteStochasticVolatilityAR(varargin{:});
end
