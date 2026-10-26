function varargout = discreteSV(varargin)
% Compatibility wrapper; all options and diagnostics are forwarded.
warning('discretization:legacyEntryPoint', ...
    'Use discretization.discreteStochasticVolatilityAR instead of discreteSV.');
[varargout{1:nargout}] = discretization.discreteStochasticVolatilityAR(varargin{:});
end
