function [x,w] = NPGQ(varargin)
%NPGQ Compatibility alias for the descriptive package function.
warning('discretization:deprecatedFunction', ...
    ['NPGQ is deprecated. Use ' ...
    'discretization.dataDrivenGaussianQuadrature instead.'])
[x,w] = discretization.dataDrivenGaussianQuadrature(varargin{:});
end
