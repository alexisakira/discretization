function [P,yxGrids] = discreteARWithStochasticVolatility(varargin)
%DISCRETEARWITHSTOCHASTICVOLATILITY Compatibility name.
warning('discretization:deprecatedFunction', ...
    ['discretization.discreteARWithStochasticVolatility is deprecated. ' ...
    'Use discretization.discreteStochasticVolatilityAR instead.'])
[P,yxGrids] = discretization.discreteStochasticVolatilityAR(varargin{:});
end
