function density = CIRpdf(varargin)
%CIRPDF Compatibility alias for cirTransitionDensity.
warning('discretization:deprecatedFunction', ...
    ['CIRpdf is deprecated. Use ' ...
    'discretization.cirTransitionDensity instead.'])
density = discretization.cirTransitionDensity(varargin{:});
end
