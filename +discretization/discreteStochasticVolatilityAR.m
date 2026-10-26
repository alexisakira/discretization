%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% discreteStochasticVolatilityAR
% (c) 2016 Leland E. Farmer and Alexis Akira Toda
% 
% Purpose:
%       Discretize an AR(1) process with log AR(1) stochastic volatility
%       y_t = lambda*y_{t-1} + u_t
%       x_t = (1-rho)*mu + rho*x_{t-1} + epsilon_t
%       u_t ~ N(0,exp(x_t)); epsilon_t ~ N(0,sigma_e^2)
%
% Usage:
%       [P,yxGrids] = discretization.discreteStochasticVolatilityAR(...
%           lambda,rho,sigmaU,sigmaE,Ny,Nx,method,nSigmaY,varargin)
%
% Inputs:
% lambda    - persistence of y process
% rho       - persistence of x process
% sigmaU    - unconditional standard deviation of u_t
% sigmaE    - standard deviation of epsilon_t
% Ny        - number of grid points for y process
% Nx        - number of grid points for x process
% Optional:
% method    - quadrature method for x process (default = 'even')
% nSigmaY   - grid spacing parameter for y (default = sqrt((Ny-1)/2)
%
% Version 1.2: May 27, 2019
%
% - Added nSigmaY as optional input variable
% - Avoided combvec function, which requires Deep Learning Toolbox
% - Changed output grid name from xyGrids to yxGrids because order is y, x
% - Changed kappa to 1e-8
function [P,yxGrids,diagnostics] = discreteStochasticVolatilityAR(lambda,rho,sigmaU,sigmaE,Ny,Nx,varargin)
[options,floorValue] = processOptions(varargin,{'even',sqrt((Ny-1)/2)});
[method,nSigmaY] = options{:};

if floorValue > 0
    [P,yxGrids] = legacyDiscreteStochasticVolatilityAR(lambda,rho,sigmaU,sigmaE,Ny,Nx,method,nSigmaY,floorValue);
    diagnostics = struct('probabilityFloor',floorValue,'matchedMoments',[]);
    return
end
%% Compute some uncondtional moments

sigmaX = (sigmaE^2)/(1-rho^2); % unconditional variance of variance process
xBar = 2*log(sigmaU)-sigmaX/2; % unconditional mean of variance process, targeted to match a mean standard deviation of sigmaU
sigmaY = sqrt(exp(xBar+sigmaX/2)/(1-lambda^2)); % uncondtional standard deviation of technology shock

%% Construct technology process approximation

[Px,xGrid,volatilityDiagnostics] = discretization.discreteVAR(xBar*(1-rho),rho,sigmaE^2,Nx,2,method); % discretization of variance process

yGrid = linspace(-nSigmaY*sigmaY,nSigmaY*sigmaY,Ny);

Nm = Nx*Ny; % total number of state variable pairs
%yxGrids = flipud(combvec(xGrid,yGrid))';
temp1 = repmat(xGrid,1,Ny);
temp2 = kron(yGrid,ones(1,Nx));
yxGrids = flipud([temp1; temp2])'; % avoid using combvec, which requires deep learning toolbox
P = zeros(Nm);
matched = zeros(Nm,1);
errors = zeros(Nm,2);
for row = 1:Nm
    conditionalMean = lambda*yxGrids(row,1);
    conditionalVariance = exp((1-rho)*xBar+rho*yxGrids(row,2)+sigmaE^2/2);
    z = (yGrid-conditionalMean)/sqrt(conditionalVariance);
    prior = relativeReferenceWeights(-0.5*z.^2);
    powers = [z;z.^2];
    [p,matched(row)] = sequentialMomentMatch(yGrid,powers,[0;1],prior,2,1e-7);
    errors(row,:) = (powers*p'-[0;1])';
    P(row,:) = kron(p,Px(mod(row-1,Nx)+1,:));
end
yxGrids = yxGrids';
diagnostics = struct('requestedMoments',2,'matchedMoments',matched, ...
    'standardizedMomentErrors',errors,'momentTolerance',1e-7, ...
    'downgradedRows',find(matched < 2),'volatility',volatilityDiagnostics,'probabilityFloor',0);
end
