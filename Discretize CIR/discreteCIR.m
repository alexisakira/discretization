function [P,X] = discreteCIR(varargin)
%DISCRETECIR Compatibility entry point for discretization.discreteCIR.
warning('discretization:legacyEntryPoint', ...
    'Use discretization.discreteCIR instead of the unqualified entry point.')
[P,X] = discretization.discreteCIR(varargin{:});
end
