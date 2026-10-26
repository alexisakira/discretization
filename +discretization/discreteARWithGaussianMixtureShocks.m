function [P,X,diagnostics] = discreteARWithGaussianMixtureShocks(varargin)
%DISCRETEARWITHGAUSSIANMIXTURESHOCKS Compatibility name.
warning('discretization:deprecatedFunction', ...
    ['discretization.discreteARWithGaussianMixtureShocks is deprecated. ' ...
    'Use discretization.discreteGaussianMixtureAR instead.'])
[P,X,diagnostics] = discretization.discreteGaussianMixtureAR(varargin{:});
end
