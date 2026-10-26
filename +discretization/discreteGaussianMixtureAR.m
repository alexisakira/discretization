%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% discreteGaussianMixtureAR
% (c) 2019 Alexis Akira Toda
% 
% Purpose:
%       Discretize an AR(p) process with Gaussian mixture shocks
%
% Usage:
%       [P,X,diagnostics] = discretization.discreteGaussianMixtureAR(...
%           mu,A,pC,muC,sigmaC,Nm,nMoments,method,nSigmas)
%
% Inputs:
% mu        - AR location (unconditional mean if the innovations have mean 0)
% A         - vector of coefficients of AR(p)
% pC        - vector of proportions of Gaussian mixtures
% muC       - vector of means of Gaussian mixtures
% sigmaC    - vector of standard deviations of Gaussian mixtures
% Nm        - number of grid points in one dimension
% Optional:
% nMoments  - number of moments to match (default = 2)
% method    - quadrature method (default = 'even')
% nSigmas   - grid half-width in stationary standard deviations
%
% The process is X_t = mu*(1-sum(A)) + A*[X_{t-1};...] + epsilon_t.
% Its unconditional mean is mu + sum(pC.*muC)/(1-sum(A)). To make mu
% the unconditional mean, supply mixture means whose weighted mean is zero.
%
% For method='even', diagnostics reports requestedMoments, matchedMoments
% and standardizedMomentErrors (one row per transition row, columns 1:4).
% Errors are in powers of the innovation standard deviation, centered on
% the conditional mean. momentTolerance is the acceptance tolerance;
% downgradedRows identifies rows that match fewer moments than requested.
% Grid width and resolution should be checked separately for spiky mixtures.
% The historical default width is retained; pass nSigmas explicitly when
% comparing resolutions. Diagnostics are empty for the optional methods.

function [P,X,diagnostics] = discreteGaussianMixtureAR(mu,A,pC,muC,sigmaC,Nm,nMoments,method,nSigmas)

if nargin < 6
    error('discretization:discreteGaussianMixtureAR:NotEnoughInputs', ...
        'Supply mu, A, pC, muC, sigmaC, and Nm.')
end
validateattributes(mu,{'numeric'},{'real','finite','scalar'},mfilename,'mu',1)
validateattributes(A,{'numeric'},{'real','finite','vector','nonempty'},mfilename,'A',2)
validateattributes(pC,{'numeric'},{'real','finite','vector','nonempty','nonnegative'},mfilename,'pC',3)
validateattributes(muC,{'numeric'},{'real','finite','vector','nonempty'},mfilename,'muC',4)
validateattributes(sigmaC,{'numeric'},{'real','finite','vector','nonempty','positive'},mfilename,'sigmaC',5)
validateattributes(Nm,{'numeric'},{'real','finite','scalar','integer','>=',3},mfilename,'Nm',6)
if numel(pC) ~= numel(muC) || numel(pC) ~= numel(sigmaC)
    error('discretization:discreteGaussianMixtureAR:InvalidMixture', ...
        'pC, muC, and sigmaC must have the same length.')
end
probabilitySum = sum(double(pC));
if abs(probabilitySum-1) > 1e-12
    error('discretization:discreteGaussianMixtureAR:InvalidProbabilities', ...
        'Mixture probabilities must sum to one within 1e-12.')
end

mu = double(mu);
Nm = double(Nm);
A = double(A(:)');
pC = double(pC(:))/probabilitySum;
muC = double(muC(:));
sigmaC = double(sigmaC(:));

if nargin < 7
    nMoments = 2;
end
if nargin < 8
    method = 'even';
end
validateattributes(nMoments,{'numeric'}, ...
    {'real','finite','scalar','integer','>=',1,'<=',4},mfilename,'nMoments',7)
nMoments = double(nMoments);
if isstring(method) && isscalar(method)
    method = char(method);
end
if ~ischar(method) || ~isrow(method) || ...
        ~ismember(method,{'even','gauss-legendre','clenshaw-curtis','gauss-hermite','GMQ'})
    error('discretization:discreteGaussianMixtureAR:InvalidMethod', ...
        'Unknown discretization method.')
end
K = length(A);
F = [A;eye(K-1,K)]; % matrix to represent AR(p) by VAR(1);
rho = max(abs(eig(F))); % spectral radius of F
if rho >= 1
    error('discretization:discreteGaussianMixtureAR:NonstationaryProcess', ...
        'The AR companion matrix must have spectral radius less than one.')
end
if nargin < 9
    if rho <= 1-2/(Nm-1)
        nSigmas = sqrt(2*(Nm-1));
    else
        nSigmas = sqrt(Nm-1);
    end
end
validateattributes(nSigmas,{'numeric'}, ...
    {'real','finite','scalar','positive'},mfilename,'nSigmas',9)
nSigmas = double(nSigmas);

diagnostics = [];
if strcmp(method,'even')
    [P,X,diagnostics] = discreteGaussianMixtureAREven( ...
        mu,A,pC,muC,sigmaC,Nm,nMoments,nSigmas,F);
    return
end

% Historical optional quadrature methods; their numerical cleanup is deferred.
sigmaC2 = sigmaC.^2;
T1 = pC'*muC; % mean
T2 = pC'*(muC.^2+sigmaC2); % uncentered second moment
T3 = pC'*(muC.^3+3*muC.*sigmaC2); % uncentered third moment
T4 = pC'*(muC.^4+6*(muC.^2).*sigmaC2+3*sigmaC2.^2); % uncentered fourth moment

TBar = [T1 T2 T3 T4]';

mixturePDF = @(values) sum(normpdf(values,muC',sigmaC').*pC',2);

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
