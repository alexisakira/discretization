mu = 0;
rho = 0.9;
sigma = 0.1;
nMoments = 4;
Nm = 9;

tic
[P,X] = discreteAR(mu,rho,sigma,Nm,'even',nMoments);
toc