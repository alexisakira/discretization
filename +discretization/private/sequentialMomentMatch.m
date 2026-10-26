function [probability,matched] = sequentialMomentMatch(grid,powers,targets,prior,requested,tolerance)
% Start with the lowest feasible moments and retain them on any fallback.
if min(powers(1,:)) >= 0 || max(powers(1,:)) <= 0
    probability = zeros(size(grid));
    [~,nearest] = min(abs(powers(1,:)));
    probability(nearest) = 1;
    matched = 0;
    return
end

matched = 0;
probability = zeros(size(prior));
[~,nearest] = min(abs(powers(1,:)));
probability(nearest) = 1;
dual = 0;
for count = 1:requested
    evaluated = powers(1:count,:);
    target = targets(1:count);
    if count == 2
        z = powers(1,:);
        left = max(z(z <= 0)); right = min(z(z >= 0));
        minimumVariance = -left*right;
        maximumVariance = -min(z)*max(z);
        if targets(2) < minimumVariance || targets(2) > maximumVariance
            break
        end
    elseif count > 2 && ~momentTargetFeasible(evaluated,target)
        break
    end
    initial = [dual;0];
    if count == 1
        initial = 0;
    end
    [~,priorError,~,candidate] = entropyObjective(zeros(count,1),evaluated,target,prior);
    if norm(priorError,Inf) <= 1e-12
        probability = candidate;
        dual = zeros(count,1);
        matched = count;
        continue
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
