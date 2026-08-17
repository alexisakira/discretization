function [p,lambdaBar,momentError] = maximumEntropyWeights(varargin)
%MAXIMUMENTROPYWEIGHTS Compute moment-matching weights on a fixed grid.
%   [p,lambdaBar,momentError] = discretization.maximumEntropyWeights(...
%       D,T,TBar,q,lambda0)
[p,lambdaBar,momentError] = maximumEntropyWeightsCore(varargin{:});
end
