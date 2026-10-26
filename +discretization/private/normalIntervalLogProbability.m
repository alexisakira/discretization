function value = normalIntervalLogProbability(lower,upper)
% Subtract CDFs in the smaller tail, using log1p to avoid cancellation.
positive = lower > 0;
a = lower; b = upper;
a(positive) = -upper(positive); b(positive) = -lower(positive);
logA = normalLogCDF(a); logB = normalLogCDF(b);
value = logB+log(-expm1(logA-logB));
end
function value = normalLogCDF(x)
value = log(normcdf(x));
tail = x < -8;
value(tail) = log(erfcx(-x(tail)/sqrt(2))/2)-x(tail).^2/2;
value(x == -Inf) = -Inf;
end
