%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% discreteVAR
% (c) 2015 Leland E. Farmer and Alexis Akira Toda
% 
% Purpose: 
%       Compute a finite-state Markov chain approximation to a VAR(1)
%       process of the form 
%
%           y_(t+1) = b + B*y_(t) + Psi^(1/2)*epsilon_(t+1)
%
%       where epsilon_(t+1) is an (M x 1) vector of independent standard
%       normal innovations
%
% Usage:
%       [P,X] = discretization.discreteVAR(...
%           b,B,Psi,Nm,nMoments,method,nSigmas,varargin)
%
% Inputs:
% b         - (M x 1) constant vector
% B         - (M x M) matrix of impact coefficients
% Psi       - (M x M) variance-covariance matrix of the innovations
% Nm        - Desired number of discrete points in each dimension
% Optional:
% nMoments  - Desired number of moments to match. The default is 2.
% method    - String specifying the method used to determine the grid
%             points. Accepted inputs are 'even,' 'quantile,' and
%             'quadrature.' The default option is 'even.' Please see the
%             paper for more details.
% nSigmas   - If the 'even' option is specified, nSigmas is used to
%             determine the number of unconditional standard deviations
%             used to set the endpoints of the grid. The default is
%             sqrt(Nm-1).
%
% Outputs:
% P         - (Nm^M x Nm^M) probability transition matrix. Each row
%             corresponds to a discrete conditional probability 
%             distribution over the state M-tuples in X
% X         - (M x Nm^M) matrix of states. Each column corresponds to an
%             M-tuple of values which correspond to the state associated 
%             with each row of P
%
% NOTES:
% - discreteVAR only accepts non-singular variance-covariance matrices.
% - discreteVAR only constructs tensor product grids where each dimension
%   contains the same number of points. For this reason it is recommended
%   that this code not be used for problems of more than about 4 or 5
%   dimensions due to curse of dimensionality issues.
% Future updates will allow for singular variance-covariance matrices and
% sparse grid specifications.
%
% Version 1.1: May 5, 2015
%
% Changed default spacing for even-grid method to sqrt((Nm-1)/2) 
%
% Version 1.2: March 4, 2016
%
% - Changed default spacing for even-grid method to sqrt(Nm-1)
% - Use alternative diagonalization method to make the variance of each VAR
%   component approximately equal
% - Target arbitrarily many conditional moments up to 4
%
% Version 1.3: May 24, 2019
%
% - Added warning message for using 'quantile' method because it is poor
% - Decreased warning threshold for 'quadrature' method to spectral radius 0.8
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

function [P,X,diagnostics] = discreteVAR(b,B,Psi,Nm,varargin)
[options,floorValue] = processOptions(varargin,{2,'even',sqrt(Nm-1)});
[nMoments,method,nSigmas] = options{:};

%% Input error checks

warningState = warning;
warningCleanup = onCleanup(@() warning(warningState));
warning('off','MATLAB:singularMatrix');

% Make sure the correct number of arguments are provided
if nargin < 4
    error('You must supply at least 4 arguments to discreteVAR')
else
    if ~strcmp(method,'quantile') && ~strcmp(method,'even') && ~strcmp(method,'quadrature')
        error('Method must be one of quantile, even, or quadrature')
    end
    if strcmp(method,'quantile')
        warning('quantile method is poor and not recommended')
    end
end

bSize = size(B);
M = bSize(1);
% Check to see if user has provided an explicit spacing for the 'even'
% method. If not, use default value.
if isempty(nSigmas)
    nSigmas = sqrt(Nm-1);
end

% Check size restrictions on matrices
if bSize(1) ~= bSize(2)
    error('B must be a square matrix')
end

if size(b,2) ~= 1
    error('b must be a column vector')
end

if size(b,1) ~= bSize(1)
    error('b must have the same number of rows as B')
end

% Check that Psi is a valid covariance matrix
[~,posDefCheck] = chol(Psi);
if posDefCheck
    error('Psi must be a positive definite matrix')
end

% Check that Nm is a valid number of grid points
if ~isnumeric(Nm) || Nm < 3 || rem(Nm,1) ~= 0
    error('Nm must be a positive integer greater than 3')
end

% Check that nMoments is a valid number
if ~isnumeric(nMoments) || nMoments < 1 || ~((rem(nMoments,2) == 0) || (nMoments == 1))
    error('nMoments must be either 1 or a positive even integer')
   %~isnumeric(nMoments) || nMoments < 1 || rem(nMoments,1) ~= 0
   % error('nMoments must be a positive integer')
end

% Warning about persistence for quadrature method
if strcmp(method,'quadrature') && any(eig(B) > 0.8)
    warning('The quadrature method may perform poorly for persistent processes.')
end

%% Compute polynomial moments of standard normal distribution
gaussianMoment = zeros(nMoments,1);
c = 1;
for k=1:floor(nMoments/2)
    c = (2*k-1)*c;
    gaussianMoment(2*k) = c;
end

%% Compute standardized VAR(1) representation (zero mean and diagonal covariance matrix)

if M == 1
    
    C = sqrt(Psi);
    A = B;
    mu = b/(1-B);
    Sigma = 1/(1-B^2);
    
else
    
    C1 = chol(Psi,'lower');
    mu = (eye(M)-B)\b;
    A1 = C1\(B*C1);
    sigmaVector = (eye(M^2)-kron(A1,A1))\reshape(eye(M),M^2,1);
    Sigma1 = reshape(sigmaVector,M,M); % unconditional variance
    U = minVarTrace(Sigma1);
    A = U'*A1*U;
    Sigma = U'*Sigma1*U;
    C = C1*U;
    
end

%% Construct 1-D grids

sigmas = sqrt(diag(Sigma));
y1D = zeros(M,Nm);

switch method
    case 'even'
        minSigmas = sqrt(min(eigs(Sigma)));
        for ii = 1:M
            y1D(ii,:) = linspace(-minSigmas*nSigmas,minSigmas*nSigmas,Nm);
        end
    case 'quantile'
        y1DBounds = zeros(M,Nm+1);
        for ii = 1:M
            y1D(ii,:) = norminv((2*(1:Nm)-1)./(2*Nm),0,sigmas(ii));
            y1DBounds(ii,:) = [-Inf, norminv((1:Nm-1)./Nm,0,sigmas(ii)), Inf];
        end
    case 'quadrature'
        [nodes,weights] = GaussHermite(Nm);
        for ii = 1:M
            y1D(ii,:) = sqrt(2)*nodes;
        end
end

% Construct all possible combinations of elements of the 1-D grids
D = allcomb2(y1D)';

%% Construct finite-state Markov chain approximation
if floorValue > 0
    [P,X] = legacyDiscreteVAR(b,B,Psi,Nm,nMoments,method,nSigmas,floorValue);
    diagnostics = struct('probabilityFloor',floorValue,'matchedMoments',[]);
    return
end
condMean = A*D;
P = zeros(Nm^M);
matched = zeros(Nm^M,M);
errors = zeros(Nm^M,M,nMoments);
for row = 1:Nm^M
    rowProbability = 1;
    for component = 1:M
        grid = y1D(component,:);
        center = condMean(component,row);
        switch method
            case 'even'
                logPrior = -0.5*(grid-center).^2-0.5*log(2*pi);
            case 'quantile'
                logPrior = normalIntervalLogProbability(y1DBounds(component,1:end-1)-center, ...
                    y1DBounds(component,2:end)-center);
            case 'quadrature'
                logPrior = log(weights'/sqrt(pi))-0.5*(grid-center).^2+0.5*grid.^2;
        end
        prior = relativeReferenceWeights(logPrior);
        powers = (grid-center).^((1:nMoments)');
        [probability,matched(row,component)] = sequentialMomentMatch( ...
            grid,powers,gaussianMoment(1:nMoments),prior,nMoments,1e-7);
        errors(row,component,:) = powers*probability'-gaussianMoment(1:nMoments);
        rowProbability = kron(rowProbability,probability);
    end
    P(row,:) = rowProbability;
end
X = C*D+repmat(mu,1,Nm^M);
diagnostics = struct('requestedMoments',nMoments,'matchedMoments',matched, ...
    'standardizedMomentErrors',errors,'momentTolerance',1e-7, ...
    'downgradedRows',find(any(matched < nMoments,2)),'probabilityFloor',0);
end
