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
    prior = max(exp(logPrior'-max(logPrior)),realmin);
    prior = prior/sum(prior);

    [probability,matchedMoments(row)] = matchMoments( ...
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


function [probability,matched] = matchMoments(grid,powers,targets,prior,requested,tolerance)
% Start with the lowest feasible moments and retain them on any fallback.
if powers(1,1) > 0 || powers(1,end) < 0
    probability = zeros(size(grid));
    [~,nearest] = min(abs(powers(1,:)));
    probability(nearest) = 1;
    matched = 0;
    return
end

matched = 0;
probability = prior;
dual = 0;
for count = 1:requested
    evaluated = powers(1:count,:);
    target = targets(1:count);
    if count > 1 && ~momentTargetFeasible(evaluated,target)
        break
    end
    initial = [dual;0];
    if count == 1
        initial = 0;
    end
    [candidate,candidateDual,error] = maximumEntropyWeightsCore( ...
        grid,@(unused) evaluated,target,prior,initial);
    % Refine the residual when the optimizer stops on objective change.
    [refined,refinedDual,refinedError] = refineMoments( ...
        candidateDual,evaluated,target,prior);
    if norm(refinedError,Inf) < norm(error,Inf)
        candidate = refined;
        candidateDual = refinedDual;
        error = refinedError;
    end
    if any(~isfinite(candidate)) || norm(error,Inf) > tolerance
        break
    end
    probability = candidate;
    dual = candidateDual;
    matched = count;
end
end


function [probability,dual,error] = refineMoments(dual,evaluated,target,prior)
% A few damped Newton steps resolve small residuals without altering the
% shared solver. Accept only steps that improve the moment error.
[objective,error,hessian,probability] = entropyObjective(dual,evaluated,target,prior);
for iteration = 1:8
    if norm(error,Inf) <= 1e-10 || rcond(hessian) < 1e-14
        break
    end
    step = hessian\error;
    accepted = false;
    for damping = 0:20
        candidate = dual-step*2^(-damping);
        if any(~isfinite(candidate))
            continue
        end
        [newObjective,newError,newHessian,newProbability] = ...
            entropyObjective(candidate,evaluated,target,prior);
        if isfinite(newObjective) && norm(newError,Inf) < norm(error,Inf) && ...
                newObjective <= objective+1e-12*max(1,abs(objective))
            dual = candidate;
            objective = newObjective;
            error = newError;
            hessian = newHessian;
            probability = newProbability;
            accepted = true;
            break
        end
    end
    if ~accepted
        break
    end
end
end
