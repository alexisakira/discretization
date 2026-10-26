function varargout = discreteAR(varargin)
% Compatibility wrapper; all options and diagnostics are forwarded.
warning('discretization:legacyEntryPoint', ...
    'Use discretization.discreteAR instead of discreteAR.');
[varargout{1:nargout}] = discretization.discreteAR(varargin{:});
end
