function [P,X] = discreteVAR(varargin)
%DISCRETEVAR Compatibility entry point for discretization.discreteVAR.
warning('discretization:legacyEntryPoint', ...
    'Use discretization.discreteVAR instead of the unqualified entry point.')
[P,X] = discretization.discreteVAR(varargin{:});
end
