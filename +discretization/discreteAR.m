function [P,X] = discreteAR(mu,rho,sigma,Nm,method,nMoments,nSigmas)

%DISCRETEAR Discretize a Gaussian AR(1) process.
%   [P,X] = discretization.discreteAR(...
%       mu,rho,sigma,Nm,method,nMoments,nSigmas)
%
% This optimized Gaussian specialization complements
% discreteGaussianMixtureAR.

% define conditional central moments
T1 = 0;
T2 = sigma^2;
T3 = 0;
T4 = 3*sigma^4;

TBar = [T1 T2 T3 T4]'; % vector of conditional central moments

% Set defaults for optional arguments.
if nargin < 5
    method = 'even';
end
if nargin < 6
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
    
    q(q < kappa) = kappa; % replace small values for numerical stability

    standardizedGrid = (X-condMean)/scalingFactor;
    standardizedPowers = (standardizedGrid'.^(1:4))';
    scaledTargets = TBar./(scalingFactor.^(1:4)');

    if nMoments == 1
        P(ii,:) = maximumEntropyWeightsCore(X,...
            @(unused) standardizedPowers(1,:),scaledTargets(1),q,0);
        continue
    end

    [p,lambda,momentError] = maximumEntropyWeightsCore(X,...
        @(unused) standardizedPowers(1:2,:),scaledTargets(1:2),...
        q,zeros(2,1));
    if norm(momentError) > 1e-5
        warning('Failed to match first 2 moments. Just matching 1.')
        P(ii,:) = maximumEntropyWeightsCore(X,...
            @(unused) standardizedPowers(1,:),scaledTargets(1),q,0);
        continue
    end
    if nMoments == 2
        P(ii,:) = p;
        continue
    end

    if nMoments == 4 && momentTargetFeasible(...
            standardizedPowers,scaledTargets)
        [pnew,~,momentError] = maximumEntropyWeightsCore(X,...
            @(unused) standardizedPowers,scaledTargets,q,[lambda;0;0]);
        if norm(momentError) <= 1e-5
            P(ii,:) = pnew;
            continue
        end
    end

    if momentTargetFeasible(standardizedPowers(1:3,:),scaledTargets(1:3))
        [pnew,~,momentError] = maximumEntropyWeightsCore(X,...
            @(unused) standardizedPowers(1:3,:),scaledTargets(1:3),...
            q,[lambda;0]);
    else
        momentError = Inf;
    end
    if norm(momentError) > 1e-5
        warning('Failed to match first 3 moments. Just matching 2.')
        P(ii,:) = p;
    else
        P(ii,:) = pnew;
        if nMoments == 4
            warning('Failed to match first 4 moments. Just matching 3.')
        end
    end
end

end
