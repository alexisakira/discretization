function value = probabilityFloorOption(options)
% Parse an explicit legacy reference-weight floor; zero is the new default.
parser = inputParser;
addParameter(parser,'ProbabilityFloor',0,@(x) isnumeric(x) && isscalar(x) && ...
    isreal(x) && isfinite(x) && x >= 0 && x <= 1);
parse(parser,options{:});
value = parser.Results.ProbabilityFloor;
end
