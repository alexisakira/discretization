clear
close all
clc;

% just pick some numbers
b = 4*0.007;
a = -log(0.58);
sigma = 0.1;

r = 0.04;
Delta = 1/4;
x = linspace(0,0.1,101);
temp = CIRpdf(x,r,a,b,sigma,Delta);

figure
plot(x,CIRpdf(x,r,a,b,sigma,Delta));

N = 9;
Coverage = 0.995; % coverage rate of the grid (optional)
%method = 'even';
method = 'exponential'; % grid choice (optional)

tic
[P,X] = discreteCIR(a,b,sigma,Delta,N,Coverage,method);
toc

if strcmp(method,'even')
    h = X(2) - X(1);
    dX = ones(size(X))/h;
elseif strcmp(method,'exponential')
    h = log(X(2)) - log(X(1)); % grid spacing in log
    dX = 1./(h*X);
end

figure
plot(X,P(floor(N/4),:).*dX,X,P(floor(N/2),:).*dX,X,P(ceil(3*N/4),:).*dX)
title('Transition probabilities (density scale)')

[v,~] = eigs(P',1,1);
pi = v'/sum(v); % stationary distribution

figure
plot(X,pi.*dX)
title('Stationary distribution (density scale)')