function [P,X] = discreteOU(varargin)
%DISCRETEOU Compatibility entry point for discretization.discreteOU.
warning('discretization:legacyEntryPoint', ...
    'Use discretization.discreteOU instead of the unqualified entry point.')
[P,X] = discretization.discreteOU(varargin{:});
end
