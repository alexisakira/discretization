%% Stochastic volatility - example in Appendix B.2 of Farmer & Toda (2017)

lambda = 0.95;
rho = 0.9;
sigmaU = 0.007;
sigmaE = 0.06;
Ny = 9;
Nx = 5;

tic
[P,yxGrids] = discreteSV(lambda,rho,sigmaU,sigmaE,Ny,Nx);
toc