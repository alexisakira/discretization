function [P,X] = discreteAR(mu,rho,sigma,Nm,method,nMoments,nSigmas)

% discretize AR(1) process with Gaussian shocks and various grids
% no need to use this because it is a special case of discreteGMAR.m

% define conditional central moments
T1 = 0;
T2 = sigma^2;
T3 = 0;
T4 = 3*sigma^4;

TBar = [T1 T2 T3 T4]'; % vector of conditional central moments

% Default number of moments to match is 2
if nargin == 4
    nMoments = 2;
end

% define grid spacing parameter if not provided
if nargin < 7
    if abs(rho) <= 1-2/(Nm-1)
        nSigmas = sqrt(2*(Nm-1));
    else
        nSigmas = sqrt(Nm-1);
    end
end

% Check that Nm is a valid number of grid points
if ~isnumeric(Nm) || Nm < 3 || rem(Nm,1) ~= 0
    error('Nm must be a positive integer greater than 3')
end

% Check that nMoments is a valid number
if ~isnumeric(nMoments) || nMoments < 1 || nMoments > 4 || ~((rem(nMoments,1) == 0) || (nMoments == 1))
    error('nMoments must be either 1, 2, 3, 4')
end

sigmaX = sigma/sqrt(1-rho^2); % unconditional standard deviation

switch method
    case 'even'
        X = linspace(mu-nSigmas*sigmaX,mu+nSigmas*sigmaX,Nm);
        W = ones(1,Nm);
    case 'gauss-legendre'
        [X,W] = legpts(Nm,[mu-nSigmas*sigmaX,mu+nSigmas*sigmaX]);
        X = X';
    case 'clenshaw-curtis'
        [X,W] = fclencurt(Nm,mu-nSigmas*sigmaX,mu+nSigmas*sigmaX);
        X = fliplr(X');
        W = fliplr(W');
    case 'gauss-hermite'
        [X,W] = GaussHermite(Nm);
        X = mu+sqrt(2)*sigma*X';
        W = W'./sqrt(pi);
end

P = NaN(Nm);
scalingFactor = max(abs(X));
kappa = 1e-8;

for ii = 1:Nm
    
    condMean = mu*(1-rho)+rho*X(ii); % conditional mean
    switch method % define prior probabilities
        case 'gauss-hermite'
            q = W;
        otherwise
            q = W.*normpdf(X,condMean,sigma);
    end
    
    if any(q < kappa)
        q(q < kappa) = kappa; % replace by small number for numerical stability
    end

if nMoments == 1 % match only 1 moment
            P(ii,:) = discreteApproximation(X,@(x)(x-condMean)/scalingFactor,TBar(1)./scalingFactor,q,0);
else % match 2 moments first
    [p,lambda,momentError] = discreteApproximation(X,@(x) [(x-condMean)./scalingFactor;...
        ((x-condMean)./scalingFactor).^2],...
        TBar(1:2)./(scalingFactor.^(1:2)'),q,zeros(2,1));
    if norm(momentError) > 1e-5 % if 2 moments fail, then just match 1 moment
                warning('Failed to match first 2 moments. Just matching 1.')
                P(ii,:) = discreteApproximation(X,@(x)(x-condMean)/scalingFactor,0,q,0);
    elseif nMoments == 2
        P(ii,:) = p;
    elseif nMoments == 3 % 3 moments
    [pnew,~,momentError] = discreteApproximation(X,@(x) [(x-condMean)./scalingFactor;...
        ((x-condMean)./scalingFactor).^2;((x-condMean)./scalingFactor).^3],...
        TBar(1:3)./(scalingFactor.^(1:3)'),q,[lambda;0]);
    if norm(momentError) > 1e-5
        warning('Failed to match first 3 moments.  Just matching 2.')
        P(ii,:) = p;
    else P(ii,:) = pnew;
    end
    else % 4 moments
    [pnew,~,momentError] = discreteApproximation(X,@(x) [(x-condMean)./scalingFactor;...
        ((x-condMean)./scalingFactor).^2; ((x-condMean)./scalingFactor).^3;...
        ((x-condMean)./scalingFactor).^4],TBar./(scalingFactor.^(1:4)'),q,[lambda;0;0]);
    if norm(momentError) > 1e-5
        %warning('Failed to match first 4 moments.  Just matching 3.')
        [pnew,~,momentError] = discreteApproximation(X,@(x) [(x-condMean)./scalingFactor;...
        ((x-condMean)./scalingFactor).^2;((x-condMean)./scalingFactor).^3],...
        TBar(1:3)./(scalingFactor.^(1:3)'),q,[lambda;0]);
    if norm(momentError) > 1e-5
        warning('Failed to match first 3 moments.  Just matching 2.')
        P(ii,:) = p;
    else P(ii,:) = pnew;
        warning('Failed to match first 4 moments.  Just matching 3.')
    end
    else P(ii,:) = pnew;
    end
    end
end

end