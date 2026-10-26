%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% discreteGaussianMixtureAR
% (c) 2019 Alexis Akira Toda
% 
% Purpose:
%       Discretize an AR(p) process with Gaussian mixture shocks
%
% Usage:
%       [P,X,diagnostics] = discretization.discreteGaussianMixtureAR(...
%           mu,A,pC,muC,sigmaC,Nm,nMoments,method,nSigmas,varargin)
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
% Name-value: 'ProbabilityFloor' (default 0); 1e-8 selects the 0.1.0 algorithm.
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
% comparing resolutions. Optional quadrature methods also return diagnostics.

function [P,X,diagnostics] = discreteGaussianMixtureAR(mu,A,pC,muC,sigmaC,Nm,varargin)
[options,floorValue] = processOptions(varargin,{2,'even',[]});
[nMoments,method,nSigmas] = options{:};

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
if isempty(nSigmas)
    if rho <= 1-2/(Nm-1)
        nSigmas = sqrt(2*(Nm-1));
    else
        nSigmas = sqrt(Nm-1);
    end
end
validateattributes(nSigmas,{'numeric'}, ...
    {'real','finite','scalar','positive'},mfilename,'nSigmas',9)
nSigmas = double(nSigmas);

if floorValue > 0
    [P,X] = legacyDiscreteGaussianMixtureAR(mu,A,pC,muC,sigmaC,Nm,nMoments,method,nSigmas,floorValue);
    diagnostics = struct('probabilityFloor',floorValue,'matchedMoments',[]);
    return
end
diagnostics = [];
if strcmp(method,'even')
    [P,X,diagnostics] = discreteGaussianMixtureAREven( ...
        mu,A,pC,muC,sigmaC,Nm,nMoments,nSigmas,F);
    return
end

% Optional quadrature methods use the same log-weight and sequential matching rules.
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
matched = zeros(stateCount,1);
errors = zeros(stateCount,4);
innovationMean = T1;
centralMeans = muC-innovationMean;
targets = [0;1;pC'*(centralMeans.^3+3*centralMeans.*sigmaC2)/sigma^3; ...
    pC'*(centralMeans.^4+6*centralMeans.^2.*sigmaC2+3*sigmaC2.^2)/sigma^4];
for row = 1:stateCount
    baseMean = mu*(1-sum(A))+A*X(:,row);
    logPrior = log(W)+gaussianMixtureLogDensity(X1-baseMean,pC,muC,sigmaC)';
    if strcmp(method,'gauss-hermite')
        logPrior = logPrior-(-0.5*(X1/sigma).^2-log(sigma)-0.5*log(2*pi));
    elseif strcmp(method,'GMQ')
        logPrior = logPrior-gaussianMixtureLogDensity(X1,pC,muC,sigmaC)';
    end
    prior = relativeReferenceWeights(logPrior);
    z = (X1-baseMean-innovationMean)/sigma;
    powers = z.^((1:4)');
    [probability,matched(row)] = sequentialMomentMatch(X1,powers,targets,prior,nMoments,1e-7);
    errors(row,:) = (powers*probability'-targets)';
    lagIndex = ceil(row/Nm);
    P(row,lagIndex:lagStateCount:stateCount) = probability;
end
diagnostics = struct('requestedMoments',nMoments,'matchedMoments',matched, ...
    'standardizedMomentErrors',errors,'momentTolerance',1e-7, ...
    'downgradedRows',find(matched < nMoments),'probabilityFloor',0);
end
