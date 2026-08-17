function [x,w] = GaussianMixtureQuadrature(varargin)
%GAUSSIANMIXTUREQUADRATURE Compatibility alias for the package function.
warning('discretization:deprecatedFunction', ...
    ['GaussianMixtureQuadrature is deprecated. Use ' ...
    'discretization.gaussianMixtureQuadrature instead.'])
[x,w] = discretization.gaussianMixtureQuadrature(varargin{:});
end
