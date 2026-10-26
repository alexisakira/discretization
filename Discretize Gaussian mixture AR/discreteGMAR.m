function [P,X,diagnostics] = discreteGMAR(varargin)
%DISCRETEGMAR Compatibility alias for the descriptive package function.
warning('discretization:deprecatedFunction', ...
    ['discreteGMAR is deprecated. Use ' ...
    'discretization.discreteGaussianMixtureAR instead.'])
[P,X,diagnostics] = discretization.discreteGaussianMixtureAR(varargin{:});
end
