%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% discreteGaussianMixtureAR
% (c) 2019 Alexis Akira Toda
% 
% Purpose:
%       Discretize an AR(p) process with Gaussian mixture shocks
%
% Usage:
%       [P,X] = discretization.discreteGaussianMixtureAR(...
%           mu,A,pC,muC,sigmaC,Nm,nMoments,method,nSigmas)
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

function [P,X] = discreteGaussianMixtureAR(mu,A,pC,muC,sigmaC,Nm,nMoments,method,nSigmas)

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
rho = max(abs(eig(F))); % spectral radius of F

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

mixturePDF = @(values) sum(normpdf(values,muC',sigmaC').*pC',2);

% Set defaults for optional arguments.
if nargin < 7
    nMoments = 2;
end
if nargin < 8
    method = 'even';
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
    else
        nSigmas = sqrt(Nm-1);
    end
end

sigma = sqrt(T2-T1^2); % conditional standard deviation
firstBasisVector = [1;zeros(K^2-1,1)];
inverseFirstColumn = (eye(K^2)-kron(F,F))\firstBasisVector;
sigmaX = sigma*sqrt(inverseFirstColumn(1)); % unconditional standard deviation

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
        [X1,W] = discretization.gaussianMixtureQuadrature(pC,muC,sigmaC,Nm);
        X1 = X1 + mu;
end

X = allcomb2(ones(K,1)*X1)'; % K*Nm^K matrix of grid points

stateCount = Nm^K;
lagStateCount = Nm^(K-1);
P = zeros(stateCount); % transition probability matrix
scalingFactor = max(abs(X1));
kappa = 1e-8;

if strcmp(method,'gauss-hermite')
    referencePDF = normpdf(X1',0,sigma);
elseif strcmp(method,'GMQ')
    referencePDF = mixturePDF(X1');
end

for ii = 1:stateCount
    
    condMean = mu*(1-sum(A))+A*X(:,ii);
    xPDF = (X1-condMean)';
    switch method
        case 'gauss-hermite'
            q = W.*(mixturePDF(xPDF)./referencePDF)';
        case 'GMQ'
            q = W.*(mixturePDF(xPDF)./referencePDF)';
        otherwise
            q = W.*mixturePDF(xPDF)';
    end
    
    q(q < kappa) = kappa;
    standardizedGrid = (X1-condMean)/scalingFactor;
    standardizedPowers = (standardizedGrid'.^(1:4))';
    scaledTargets = TBar./(scalingFactor.^(1:4)');

    if nMoments == 1
        p1Row = maximumEntropyWeightsCore(X1,...
            @(unused) standardizedPowers(1,:),scaledTargets(1),q,0);
    else
        [p,lambda,momentError] = maximumEntropyWeightsCore(X1,...
            @(unused) standardizedPowers(1:2,:),scaledTargets(1:2),...
            q,zeros(2,1));
        if norm(momentError) > 1e-5
            warning('Failed to match first 2 moments. Just matching 1.')
            p1Row = maximumEntropyWeightsCore(X1,...
                @(unused) standardizedPowers(1,:),scaledTargets(1),q,0);
        elseif nMoments == 2
            p1Row = p;
        else
            matchedHigherMoment = false;
            if nMoments == 4 && momentTargetFeasible(...
                    standardizedPowers,scaledTargets)
                [pnew,~,momentError] = maximumEntropyWeightsCore(X1,...
                    @(unused) standardizedPowers,scaledTargets,...
                    q,[lambda;0;0]);
                if norm(momentError) <= 1e-5
                    p1Row = pnew;
                    matchedHigherMoment = true;
                end
            end
            if ~matchedHigherMoment
                if momentTargetFeasible(standardizedPowers(1:3,:),...
                        scaledTargets(1:3))
                    [pnew,~,momentError] = maximumEntropyWeightsCore(X1,...
                        @(unused) standardizedPowers(1:3,:),...
                        scaledTargets(1:3),q,[lambda;0]);
                else
                    momentError = Inf;
                end
                if norm(momentError) > 1e-5
                    warning('Failed to match first 3 moments. Just matching 2.')
                    p1Row = p;
                else
                    p1Row = pnew;
                    if nMoments == 4
                        warning('Failed to match first 4 moments. Just matching 3.')
                    end
                end
            end
        end
    end
    lagIndex = ceil(ii/Nm);
    P(ii,lagIndex:lagStateCount:stateCount) = p1Row;
end

end
