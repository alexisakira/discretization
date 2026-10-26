function value = cirLogDensity(x,r,a,b,sigma,Delta)
c = 2*a/(-expm1(-a*Delta)*sigma^2);
q = 2*a*b/sigma^2-1;
u = c*r*exp(-a*Delta);
v = c*x;
argument = 2*sqrt(u)*sqrt(v);
value = log(c)-(sqrt(u)-sqrt(v)).^2 ...
    +(q/2)*(log(v)-log(u))+log(besseli(q,argument,1));
if q == 0
    value(v == 0) = log(c)-u;
else
    value(v == 0) = -Inf;
end
end
