function value = cirLogDensity(x,r,a,b,sigma,Delta)
% Scaled Bessel evaluation with a positive log-series fallback. Scaling alone
% prevents overflow, but I_q can still underflow at large q and small argument.
c = 2*a/(-expm1(-a*Delta)*sigma^2);
q = 2*a*b/sigma^2-1;
u = c*r*exp(-a*Delta);
v = c*x;
value = -Inf(size(v));
positive = v > 0;
if u == 0
    % Exact central Gamma limit when the noncentrality has decayed to zero.
    value(positive) = log(c)-v(positive)+q*log(v(positive))-gammaln(q+1);
else
    argument = 2*sqrt(u)*sqrt(v(positive));
    value(positive) = log(c)-(sqrt(u)-sqrt(v(positive))).^2 ...
        +(q/2)*(log(v(positive))-log(u))+log(besseli(q,argument,1));
    for index = reshape(find(positive & ~isfinite(value)),1,[])
        % I_q(2*sqrt(u*v)) = (u*v)^(q/2)/Gamma(q+1)
        %   * sum_k (u*v)^k / (k!*(q+1)_k). Sum only positive terms,
        % in log space, until the descending tail is below machine epsilon.
        logProduct = log(u)+log(v(index));
        logTerm = 0; logSum = 0;
        converged = false;
        for k = 1:100000
            logRatio = logProduct-log(k)-log(q+k);
            logTerm = logTerm+logRatio;
            logSum = max(logSum,logTerm)+log1p(exp(-abs(logSum-logTerm)));
            if logRatio < 0 && logTerm-logSum < log(eps)
                converged = true;
                break
            end
        end
        if ~converged
            error('discretization:CIRLogDensity:NoConvergence', ...
                'The CIR log-density series did not converge.')
        end
        value(index) = log(c)-u-v(index)+q*log(v(index))-gammaln(q+1)+logSum;
    end
end
if q == 0
    value(v == 0) = log(c)-u;
else
    value(v == 0) = -Inf;
end
end
