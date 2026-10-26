function [P,X,diagnostics] = discreteAR(mu,rho,sigma,Nm,varargin)
%DISCRETEAR Discretize a Gaussian AR(1) process.
%   [P,X] = discretization.discreteAR(...
%       mu,rho,sigma,Nm,method,nMoments,nSigmas)
%
% This optimized Gaussian specialization complements
% discreteGaussianMixtureAR.
% Optional positional arguments: method, nMoments, nSigmas (unchanged defaults).
% Append 'ProbabilityFloor',1e-8 for the 0.1.0 algorithm (default: 0).
% The third output reports matched moments and standardized errors by row.

[options,floorValue] = processOptions(varargin,{'even',2,[]});
[method,nMoments,nSigmas] = options{:};

validateattributes(mu,{'numeric'},{'real','finite','scalar'},mfilename,'mu',1)
validateattributes(rho,{'numeric'},{'real','finite','scalar','>',-1,'<',1},mfilename,'rho',2)
validateattributes(sigma,{'numeric'},{'real','finite','scalar','positive'},mfilename,'sigma',3)
validateattributes(Nm,{'numeric'},{'real','finite','scalar','integer','>=',3},mfilename,'Nm',4)
method = validatestring(method,{'even','gauss-legendre','clenshaw-curtis','gauss-hermite'});
% define grid spacing parameter if not provided
if isempty(nSigmas)
    if abs(rho) <= 1-2/(Nm-1)
        nSigmas = sqrt(2*(Nm-1));
    else
        nSigmas = sqrt(Nm-1);
    end
end

% Check that nMoments is a valid number
if ~isnumeric(nMoments) || nMoments < 1 || nMoments > 4 || ~((rem(nMoments,1) == 0) || (nMoments == 1))
    error('nMoments must be either 1, 2, 3, 4')
end

validateattributes(nSigmas,{'numeric'},{'real','finite','scalar','positive'},mfilename,'nSigmas')
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

if floorValue > 0
    [P,X] = legacyDiscreteAR(mu,rho,sigma,Nm,method,nMoments,nSigmas,floorValue);
    diagnostics = struct('probabilityFloor',floorValue,'matchedMoments',[]);
    return
end
P = zeros(Nm);
matched = zeros(Nm,1);
errors = zeros(Nm,4);
targets = [0;1;0;3];
for row = 1:Nm
    conditionalMean = mu*(1-rho)+rho*X(row);
    z = (X-conditionalMean)/sigma;
    if strcmp(method,'gauss-hermite')
        logPrior = log(W);
    else
        logPrior = log(W)-0.5*z.^2-log(sigma)-0.5*log(2*pi);
    end
    prior = relativeReferenceWeights(logPrior);
    powers = z.^((1:4)');
    [P(row,:),matched(row)] = sequentialMomentMatch(X,powers,targets,prior,nMoments,1e-7);
    errors(row,:) = (powers*P(row,:)'-targets)';
end
diagnostics = struct('requestedMoments',nMoments,'matchedMoments',matched, ...
    'standardizedMomentErrors',errors,'momentTolerance',1e-7, ...
    'downgradedRows',find(matched < nMoments),'probabilityFloor',0);
end
