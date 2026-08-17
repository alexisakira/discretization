function [P,X] = discreteAR(varargin)
%DISCRETEAR Compatibility entry point for discretization.discreteAR.
warning('discretization:legacyEntryPoint', ...
    'Use discretization.discreteAR instead of the unqualified entry point.')
[P,X] = discretization.discreteAR(varargin{:});
end
