function [P,X,diagnostics] = discreteGaussianMixtureAREven(mu,A,pC,muC,sigmaC,Nm,nMoments,nSigmas,F)
% Even-grid implementation. All moment checks use innovation-SD units.
active = pC > 0;
pC = pC(active);
muC = muC(active);
sigmaC = sigmaC(active);
innovationMean = pC'*muC;
innovationVariance = pC'*(sigmaC.^2+(muC-innovationMean).^2);
innovationSD = sqrt(innovationVariance);
if ~isfinite(innovationSD) || innovationSD <= 0
    error('discretization:discreteGaussianMixtureAR:InvalidScale', ...
        'The innovation standard deviation must be finite and positive.')
end

order = length(A);
basis = zeros(order^2,1);
basis(1) = 1;
stationaryCovariance = (eye(order^2)-kron(F,F))\basis;
stationarySD = innovationSD*sqrt(stationaryCovariance(1));
limit = nSigmas*stationarySD;
X1 = linspace(mu-limit,mu+limit,Nm);
if ~isfinite(limit) || limit <= 0 || any(~isfinite(X1)) || any(diff(X1) <= 0)
    error('discretization:discreteGaussianMixtureAR:InvalidGrid', ...
        'The grid must have finite, distinct, increasing points.')
end
X = allcomb2(ones(order,1)*X1)';
stateCount = Nm^order;
lagStateCount = Nm^(order-1);
P = zeros(stateCount);

componentMeans = (muC-innovationMean)/innovationSD;
componentSDs = sigmaC/innovationSD;
targets = [0;1; ...
    pC'*(componentMeans.^3+3*componentMeans.*componentSDs.^2); ...
    pC'*(componentMeans.^4+6*componentMeans.^2.*componentSDs.^2+3*componentSDs.^4)];
momentTolerance = 1e-7;
matchedMoments = zeros(stateCount,1);
momentErrors = zeros(stateCount,4);

for row = 1:stateCount
    conditionalMean = mu*(1-sum(A))+A*X(:,row)+innovationMean;
    standardizedGrid = (X1-conditionalMean)/innovationSD;
    powers = standardizedGrid.^((1:4)');
    if any(~isfinite(powers),'all') || any(~isfinite(targets))
        error('discretization:discreteGaussianMixtureAR:InvalidScale', ...
            'Standardized moments must be finite.')
    end

    % Summing component densities in log space preserves rare shocks. The
    % floor is relative to the largest density and independent of units.
    z = (standardizedGrid'-componentMeans')./componentSDs';
    logComponents = log(pC')-log(componentSDs')-0.5*z.^2;
    largest = max(logComponents,[],2);
    logPrior = -Inf(Nm,1);
    finite = isfinite(largest);
    logPrior(finite) = largest(finite)+log(sum( ...
        exp(logComponents(finite,:)-largest(finite)),2));
    if ~any(isfinite(logPrior))
        error('discretization:discreteGaussianMixtureAR:InvalidPrior', ...
            'No finite Gaussian-mixture density in transition row %d.',row)
    end
    prior = relativeReferenceWeights(logPrior');

    [probability,matchedMoments(row)] = sequentialMomentMatch( ...
        X1,powers,targets,prior,nMoments,momentTolerance);
    momentErrors(row,:) = (powers*probability'-targets)';
    lagIndex = ceil(row/Nm);
    P(row,lagIndex:lagStateCount:stateCount) = probability;
end

diagnostics = struct('requestedMoments',nMoments, ...
    'matchedMoments',matchedMoments, ...
    'standardizedMomentErrors',momentErrors, ...
    'momentTolerance',momentTolerance, ...
    'downgradedRows',find(matchedMoments < nMoments), ...
    'innovationMean',innovationMean,'innovationSD',innovationSD, ...
    'unconditionalMean',mu+innovationMean/(1-sum(A)), ...
    'stationarySD',stationarySD,'gridHalfWidthSD',nSigmas, ...
    'gridSpacing',X1(2)-X1(1), ...
    'spacingToSmallestComponentSD',(X1(2)-X1(1))/min(sigmaC));
if ~isempty(diagnostics.downgradedRows)
    warning('discretization:discreteGaussianMixtureAR:MomentFallback', ...
        ['Matched fewer than %d moments in %d of %d transition rows. ' ...
        'Inspect diagnostics and adjust grid width or resolution.'], ...
        nMoments,numel(diagnostics.downgradedRows),stateCount)
end
end
