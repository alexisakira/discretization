%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% discreteGMAR
% (c) 2019 Alexis Akira Toda
% 
% Purpose:
%       Discretize an AR(p) process with Gaussian mixture shocks
%
% Usage:
%       [P,X] = discreteGMAR(mu,A,pC,muC,sigmaC,Nm,nMoments,method,nSigmas)
%
% Inputs:
% mu        - unconditional mean
% A         - vector of coefficients of AR(p)
% pC        - vector of proportions of Gaussian mixtures
% muC       - vector of means of Gaussian mixtures
% sigmaC    - vector of standard deviations of Gaussian mixtures
% Nm        - number of grid points in one dimension
% Optional:
% nMoments  - number of moments to match (default = 2)
% method    - quadrature method (default = 'even')
% nSigmas	- grid spacing when using even-spaced grid

function [P,X] = discreteGMAR(mu,A,pC,muC,sigmaC,Nm,nMoments,method,nSigmas)

%% some error checking
if any(pC < 0)
    error('mixture proportions must be positive')
end
if any(sigmaC < 0)
    error('standard deviations must be positive')
end
if sum(pC) ~= 1
    error('mixture proportions must add up to 1')
end

if size(pC,1) < size(pC,2)
    pC = pC'; % convert to column vector
end
if size(muC,1) < size(muC,2)
    muC = muC'; % convert to column vector
end
if size(sigmaC,1) < size(sigmaC,2)
    sigmaC = sigmaC'; % convert to column vector
end

K = length(A);

if size(A,1) > size(A,2)
    A = A'; % convert A to row vector
end

F = [A;eye(K-1,K)]; % matrix to represent AR(p) by VAR(1);
rho = abs(eigs(F,1)); % spectral radius of F

if rho >= 1
    error('spectral radius must be less than one')
end

% compute conditional moments
sigmaC2 = sigmaC.^2;
T1 = pC'*muC; % mean
T2 = pC'*(muC.^2+sigmaC2); % uncentered second moment
T3 = pC'*(muC.^3+3*muC.*sigmaC2); % uncentered third moment
T4 = pC'*(muC.^4+6*(muC.^2).*sigmaC2+3*sigmaC2.^2); % uncentered fourth moment

TBar = [T1 T2 T3 T4]';

nComp = length(pC); % number of mixture components
temp = zeros(1,1,nComp);
temp(1,1,:) = sigmaC2;
gmObj = gmdistribution(muC,temp,pC); % define the Gaussian mixture object

% Default number of moments is 2
if nargin == 6
    nMoments = 2;
end

% Check that Nm is a valid number of grid points
if ~isnumeric(Nm) || Nm < 3 || rem(Nm,1) ~= 0
    error('Nm must be a positive integer greater than 3')
end

% Check that nMoments is a valid number
if ~isnumeric(nMoments) || nMoments < 1 || nMoments > 4 || ~((rem(nMoments,1) == 0) || (nMoments == 1))
    error('nMoments must be either 1, 2, 3, 4')
end

% set default nSigmas if not supplied
if nargin < 9
    if rho <= 1-2/(Nm-1)
    nSigmas = sqrt(2*(Nm-1));
    else nSigmas = sqrt(Nm-1);
    end
end

sigma = sqrt(T2-T1^2); % conditional standard deviation
temp = (eye(K^2)-kron(F,F))\eye(K^2);
sigmaX = sigma*sqrt(temp(1,1)); % unconditional standard deviation

% construct the one dimensional grid
switch method
    case 'even' % evenly-spaced grid
        X1 = linspace(mu-nSigmas*sigmaX,mu+nSigmas*sigmaX,Nm);
        W = ones(1,Nm);
    case 'gauss-legendre' % Gauss-Legendre quadrature
        [X1,W] = legpts(Nm,[mu-nSigmas*sigmaX,mu+nSigmas*sigmaX]);
        X1 = X1';
    case 'clenshaw-curtis' % Clenshaw-Curtis quadrature
        [X1,W] = fclencurt(Nm,mu-nSigmas*sigmaX,mu+nSigmas*sigmaX);
        X1 = fliplr(X1');
        W = fliplr(W');
    case 'gauss-hermite' % Gauss-Hermite quadrature
        if rho > 0.8
            warning('Model is persistent; even-spaced grid is recommended')
        end
        [X1,W] = GaussHermite(Nm);
        X1 = mu+sqrt(2)*sigma*X1';
        W = W'./sqrt(pi);
    case 'GMQ' % Gaussian Mixture Quadrature
        if rho > 0.8
            warning('Model is persistent; even-spaced grid is recommended')
        end
        [X1,W] = GaussianMixtureQuadrature(pC,muC,sigmaC,Nm);
        X1 = X1 + mu;
end

X = allcomb2(ones(K,1)*X1)'; % K*Nm^K matrix of grid points

P = NaN(Nm^K); % transition probability matrix
P1 = NaN(Nm^K,Nm); % matrix to store transition probability
P2 = kron(eye(Nm^(K-1)),ones(Nm,1)); % Nm^K * Nm^(K-1) matrix used to construct P
scalingFactor = max(abs(X1));
kappa = 1e-8;

for ii = 1:Nm^K
    
    condMean = mu*(1-sum(A))+A*X(:,ii);
    xPDF = (X1-condMean)';
    switch method
        case 'gauss-hermite'
            q = W.*(pdf(gmObj,xPDF)./normpdf(xPDF,0,sigma))';
        case 'GMQ'
            q = W.*(pdf(gmObj,xPDF)./pdf(gmObj,X1'))';
        otherwise
            q = W.*(pdf(gmObj,xPDF))';
    end
    
    if any(q < kappa)
        q(q < kappa) = kappa;
    end

if nMoments == 1 % match only 1 moment
            P1(ii,:) = discreteApproximation(X1,@(x)(x-condMean)/scalingFactor,TBar(1)./scalingFactor,q,0);
else % match 2 moments first
    [p,lambda,momentError] = discreteApproximation(X1,@(x) [(x-condMean)./scalingFactor;...
        ((x-condMean)./scalingFactor).^2],...
        TBar(1:2)./(scalingFactor.^(1:2)'),q,zeros(2,1));
    if norm(momentError) > 1e-5 % if 2 moments fail, then just match 1 moment
                warning('Failed to match first 2 moments. Just matching 1.')
                P1(ii,:) = discreteApproximation(X1,@(x)(x-condMean)/scalingFactor,TBar(1)./scalingFactor,q,0);
    elseif nMoments == 2
        P1(ii,:) = p;
    elseif nMoments == 3
    [pnew,~,momentError] = discreteApproximation(X1,@(x) [(x-condMean)./scalingFactor;...
        ((x-condMean)./scalingFactor).^2;((x-condMean)./scalingFactor).^3],...
        TBar(1:3)./(scalingFactor.^(1:3)'),q,[lambda;0]);
    if norm(momentError) > 1e-5
        warning('Failed to match first 3 moments.  Just matching 2.')
        P1(ii,:) = p;
    else P1(ii,:) = pnew;
    end
    else % 4 moments
    [pnew,~,momentError] = discreteApproximation(X1,@(x) [(x-condMean)./scalingFactor;...
        ((x-condMean)./scalingFactor).^2; ((x-condMean)./scalingFactor).^3;...
        ((x-condMean)./scalingFactor).^4],TBar./(scalingFactor.^(1:4)'),q,[lambda;0;0]);
    if norm(momentError) > 1e-5
        %warning('Failed to match first 4 moments.  Just matching 3.')
        [pnew,~,momentError] = discreteApproximation(X1,@(x) [(x-condMean)./scalingFactor;...
        ((x-condMean)./scalingFactor).^2;((x-condMean)./scalingFactor).^3],...
        TBar(1:3)./(scalingFactor.^(1:3)'),q,[lambda;0]);
    if norm(momentError) > 1e-5
        warning('Failed to match first 3 moments.  Just matching 2.')
        P1(ii,:) = p;
    else P1(ii,:) = pnew;
        warning('Failed to match first 4 moments.  Just matching 3.')
    end
    else P1(ii,:) = pnew;
    end
    end
    P(ii,:) = kron(P1(ii,:),P2(ii,:));
end

end