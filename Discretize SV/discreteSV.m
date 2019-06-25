%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% discreteSV
% (c) 2016 Leland E. Farmer and Alexis Akira Toda
% 
% Purpose:
%       Discretize an AR(1) process with log AR(1) stochastic volatility
%       y_t = lambda*y_{t-1} + u_t
%       x_t = (1-rho)*mu + rho*x_{t-1} + epsilon_t
%       u_t ~ N(0,exp(x_t)); epsilon_t ~ N(0,sigma_e^2)
%
% Usage:
%       [P,yxGrids] = discreteSV(lambda,rho,sigmaU,sigmaE,Ny,Nx,method,nSigmaY)
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
function [P,yxGrids] = discreteSV(lambda,rho,sigmaU,sigmaE,Ny,Nx,method,nSigmaY)

if nargin <= 6
    method = 'even'; % default is evenly spaced grid
end

if nargin <= 7
    nSigmaY = sqrt((Ny-1)/2); % spacing parameter for Y process
end

%% Compute some uncondtional moments

sigmaX = (sigmaE^2)/(1-rho^2); % unconditional variance of variance process
xBar = 2*log(sigmaU)-sigmaX/2; % unconditional mean of variance process, targeted to match a mean standard deviation of sigmaU
sigmaY = sqrt(exp(xBar+sigmaX/2)/(1-lambda^2)); % uncondtional standard deviation of technology shock

%% Construct technology process approximation

[Px,xGrid] = discreteVAR(xBar*(1-rho),rho,sigmaE^2,Nx,2,method); % discretization of variance process

yGrid = linspace(-nSigmaY*sigmaY,nSigmaY*sigmaY,Ny);

Nm = Nx*Ny; % total number of state variable pairs
%yxGrids = flipud(combvec(xGrid,yGrid))';
temp1 = repmat(xGrid,1,Ny);
temp2 = kron(yGrid,ones(1,Nx));
yxGrids = flipud([temp1; temp2])'; % avoid using combvec, which requires deep learning toolbox
P = zeros(Nm);
lambdaGuess = zeros(2,1);
scalingFactor = max(abs(yGrid));
kappa = 1e-8; % small positive constant for numerical stability

for ii = 1:Nm
    
    q = normpdf(yGrid,lambda*yxGrids(ii,1),sqrt(exp((1-rho)*xBar+rho*yxGrids(ii,2)+(sigmaE^2)/2)));
    if sum(q<kappa) > 0
        q(q<kappa) = kappa;
    end
    [p,~,momentError] = discreteApproximation(yGrid,@(X) [(X-lambda*yxGrids(ii,1))./scalingFactor; ((X-lambda*yxGrids(ii,1))./scalingFactor).^2],[0; (exp((1-rho)*xBar+rho*yxGrids(ii,2)+(sigmaE^2)/2))./(scalingFactor^2)],q,lambdaGuess);
    % If trying to match two conditional moments fails, just match the conditional mean
    if norm(momentError) > 1e-5
        warning('Failed to match first 2 moments. Just matching 1.')
        p = discreteApproximation(yGrid,@(X) (X-lambda*yxGrids(ii,1))./scalingFactor,0,q,0);
    end
    P(ii,:) = kron(p,ones(1,Nx));
    P(ii,:) = P(ii,:).*repmat(Px(mod(ii-1,Nx)+1,:),1,Ny);
 
end
yxGrids = yxGrids';
end