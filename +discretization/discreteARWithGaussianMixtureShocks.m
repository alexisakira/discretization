function [P,X] = discreteARWithGaussianMixtureShocks(varargin)
%DISCRETEARWITHGAUSSIANMIXTURESHOCKS Compatibility name.
warning('discretization:deprecatedFunction', ...
    ['discretization.discreteARWithGaussianMixtureShocks is deprecated. ' ...
    'Use discretization.discreteGaussianMixtureAR instead.'])
[P,X] = discretization.discreteGaussianMixtureAR(varargin{:});
end
