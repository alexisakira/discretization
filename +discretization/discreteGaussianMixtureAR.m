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
active = pC > 0;
pC = pC(active); muC = muC(active); sigmaC = sigmaC(active);
innovationMean = pC'*muC;
innovationVariance = pC'*(sigmaC.^2+(muC-innovationMean).^2);
innovationSD = sqrt(innovationVariance);
if ~isfinite(innovationSD) || innovationSD <= 0
    error('discretization:discreteGaussianMixtureAR:InvalidScale', ...
        'The innovation standard deviation must be finite and positive.')
end
basis = zeros(K^2,1); basis(1) = 1;
stationaryCovariance = (eye(K^2)-kron(F,F))\basis;
stationarySD = innovationSD*sqrt(stationaryCovariance(1));
limit = nSigmas*stationarySD;
switch method
    case 'even'
        X1 = linspace(mu-limit,mu+limit,Nm);
        W = ones(1,Nm);
    case 'gauss-legendre'
        [X1,W] = legpts(Nm,[mu-limit,mu+limit]); X1 = X1';
    case 'clenshaw-curtis'
        [X1,W] = fclencurt(Nm,mu-limit,mu+limit);
        X1 = fliplr(X1'); W = fliplr(W');
    case 'gauss-hermite'
        [X1,W] = GaussHermite(Nm);
        X1 = mu+sqrt(2)*innovationSD*X1'; W = W'/sqrt(pi);
    case 'GMQ'
        [X1,W] = discretization.gaussianMixtureQuadrature(pC,muC,sigmaC,Nm);
        X1 = X1+mu;
end
if ~isfinite(limit) || limit <= 0 || any(~isfinite(X1)) || any(diff(X1) <= 0)
    error('discretization:discreteGaussianMixtureAR:InvalidGrid', ...
        'The grid must have finite, distinct, increasing points.')
end
if ismember(method,{'gauss-hermite','GMQ'}) && rho > .8
    warning('Model is persistent; even-spaced grid is recommended')
end
X = allcomb2(ones(K,1)*X1)';
stateCount = Nm^K; lagStateCount = Nm^(K-1);
P = zeros(stateCount);
componentMeans = (muC-innovationMean)/innovationSD;
componentSDs = sigmaC/innovationSD;
targets = [0;1;pC'*(componentMeans.^3+3*componentMeans.*componentSDs.^2); ...
    pC'*(componentMeans.^4+6*componentMeans.^2.*componentSDs.^2+3*componentSDs.^4)];
matched = zeros(stateCount,1); errors = zeros(stateCount,4);
for row = 1:stateCount
    baseMean = mu*(1-sum(A))+A*X(:,row);
    z = (X1-baseMean-innovationMean)/innovationSD;
    % Standardized mixture densities avoid cancellation and preserve units.
    logPrior = log(W)+gaussianMixtureLogDensity(z,pC,componentMeans,componentSDs)';
    if strcmp(method,'gauss-hermite')
        logPrior = logPrior-(-.5*((X1-mu)/innovationSD).^2-.5*log(2*pi));
    elseif strcmp(method,'GMQ')
        logPrior = logPrior-gaussianMixtureLogDensity( ...
            (X1-mu-innovationMean)/innovationSD,pC,componentMeans,componentSDs)';
    end
    prior = relativeReferenceWeights(logPrior);
    powers = z.^((1:4)');
    [probability,matched(row)] = sequentialMomentMatch(X1,powers,targets,prior,nMoments,1e-7);
    errors(row,:) = (powers*probability'-targets)';
    lagIndex = ceil(row/Nm);
    P(row,lagIndex:lagStateCount:stateCount) = probability;
end
diagnostics = struct('requestedMoments',nMoments,'matchedMoments',matched, ...
    'standardizedMomentErrors',errors,'momentTolerance',1e-7, ...
    'downgradedRows',find(matched < nMoments),'probabilityFloor',0, ...
    'innovationMean',innovationMean,'innovationSD',innovationSD, ...
    'unconditionalMean',mu+innovationMean/(1-sum(A)), ...
    'stationarySD',stationarySD,'gridHalfWidthSD',nSigmas);
if strcmp(method,'even')
    diagnostics.gridSpacing = X1(2)-X1(1);
    diagnostics.spacingToSmallestComponentSD = diagnostics.gridSpacing/min(sigmaC);
end
if ~isempty(diagnostics.downgradedRows)
    warning('discretization:discreteGaussianMixtureAR:MomentFallback', ...
        ['Matched fewer than %d moments in %d of %d transition rows. ' ...
        'Inspect diagnostics and adjust grid width or resolution.'], ...
        nMoments,numel(diagnostics.downgradedRows),stateCount)
end
end
