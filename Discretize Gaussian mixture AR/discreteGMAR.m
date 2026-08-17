function [P,X] = discreteGMAR(varargin)
%DISCRETEGMAR Compatibility alias for the descriptive package function.
warning('discretization:deprecatedFunction', ...
    ['discreteGMAR is deprecated. Use ' ...
    'discretization.discreteGaussianMixtureAR instead.'])
[P,X] = discretization.discreteGaussianMixtureAR(varargin{:});
end
