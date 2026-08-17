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

if 2*a*b - sigma^2 <= 0
    error('It must be sigma^2 < 2*a*b')
end

%% define parameters

c = (2*a)/((1-exp(-a*Delta))*sigma^2);
q = 2*a*b/sigma^2 - 1;
u = c*r*exp(-a*Delta);
v = c*x;

%% compute density

Z = 2*sqrt(u*v);
density = c*exp(-u-v).*(v/u).^(q/2).*besseli(q,Z);

end

