function [p,lambdaBar,momentError] = discreteApproximation(varargin)
%DISCRETEAPPROXIMATION Compatibility alias for maximumEntropyWeights.
warning('discretization:deprecatedFunction', ...
    ['discreteApproximation is deprecated. Use ' ...
    'discretization.maximumEntropyWeights instead.'])
[p,lambdaBar,momentError] = discretization.maximumEntropyWeights(varargin{:});
end
