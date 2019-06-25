%% Gaussian AR(1)

% Parameter initialization

rho = 0.99; % persistence
N = 9;  % # of grid points
nMoments = 2; % # of moments to match
sigma2 = 0.1; % conditional variance

tic
[P1,D1] = discreteVAR(0,rho,sigma2,N,nMoments,'even');
% 'method' can be 'even', 'quantile', or 'quadrature'. 'quantile' is not
% recommended.
toc

%% Gaussian 2-D VAR(1) - example in Appendix B.1

% Set parameters

A = [0.9809 0.0028; 0.0410 0.9648]; % lag matrix
Sigma = [0.0087^2 0; 0 0.0262^2]; % conditional covariance matrix
N = 9; % # of grid points
nMoments = 2; % # of moments to match

% Discretize VAR

tic
[P2,D2] = discreteVAR(zeros(2,1),A,Sigma,N,nMoments,'even');
toc