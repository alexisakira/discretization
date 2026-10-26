function density = cirTransitionDensity(x,r,a,b,sigma,Delta)
%CIRTRANSITIONDENSITY Conditional Cox-Ingersoll-Ross transition density.
% x: vector of future interest rates
% r: current interest rate
% a, b, sigma: parameters of CIR model
% Delta: time difference

%% some error checking
if any(x < 0)
    error('x must be a nonnegative vector')
end
if (r <= 0)||(a <= 0)||(b <= 0)||(sigma <= 0)||(Delta <= 0)
    error('r, a, b, sigma, Delta must be positive')
end

if sigma^2 > 2*a*b
    error('It must be sigma^2 <= 2*a*b')
end

validateattributes(x,{'numeric'},{'real','finite','nonnegative'});
validateattributes([r,a,b,sigma,Delta],{'numeric'},{'real','finite','positive'});
density = exp(cirLogDensity(x,r,a,b,sigma,Delta));
end
