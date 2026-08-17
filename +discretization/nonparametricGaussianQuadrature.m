function [x,w] = nonparametricGaussianQuadrature(varargin)
%NONPARAMETRICGAUSSIANQUADRATURE Compatibility name.
warning('discretization:deprecatedFunction', ...
    ['discretization.nonparametricGaussianQuadrature is deprecated. ' ...
    'Use discretization.dataDrivenGaussianQuadrature instead.'])
[x,w] = discretization.dataDrivenGaussianQuadrature(varargin{:});
end
