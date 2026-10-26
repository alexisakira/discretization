function varargout = discreteVAR(varargin)
% Compatibility wrapper; all options and diagnostics are forwarded.
warning('discretization:legacyEntryPoint', ...
    'Use discretization.discreteVAR instead of discreteVAR.');
[varargout{1:nargout}] = discretization.discreteVAR(varargin{:});
end
