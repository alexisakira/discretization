function prior = relativeReferenceWeights(logPrior)
% Stabilize relative log weights only. This is not an absolute density floor.
if ~any(isfinite(logPrior)) || any(isnan(logPrior)) || any(logPrior == Inf)
    error('discretization:InvalidPrior','Reference log weights must have a finite maximum.')
end
prior = exp(max(logPrior-max(logPrior),log(realmin)));
prior = prior/sum(prior);
end
