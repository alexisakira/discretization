function feasible = momentTargetFeasible(evaluatedMoments,targetMoments,tolerance)
%MOMENTTARGETFEASIBLE Check whether moments lie in the grid's convex hull.

if nargin < 3
    tolerance = 1e-10;
end

pointCount = size(evaluatedMoments,2);
constraintMatrix = [ones(1,pointCount);evaluatedMoments];
constraintTarget = [1;targetMoments(:)];
[~,residualSquared] = lsqnonneg(constraintMatrix,constraintTarget);
residual = sqrt(residualSquared);
feasible = residual <= tolerance*(1+norm(constraintTarget));

end

