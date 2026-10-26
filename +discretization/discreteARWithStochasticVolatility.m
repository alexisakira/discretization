function varargout = discreteARWithStochasticVolatility(varargin)
%DISCRETEARWITHSTOCHASTICVOLATILITY Compatibility name.
warning('discretization:deprecatedFunction', ...
    ['discretization.discreteARWithStochasticVolatility is deprecated. ' ...
    'Use discretization.discreteStochasticVolatilityAR instead.'])
[varargout{1:nargout}] = discretization.discreteStochasticVolatilityAR(varargin{:});
end
