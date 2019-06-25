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
%       [P,X] = discreteVAR(b,B,Psi,Nm,nMoments,method,nSigmas)
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

function [P,X] = discreteVAR(b,B,Psi,Nm,nMoments,method,nSigmas)

%% Input error checks

warning off MATLAB:singularMatrix % surpress inversion warnings

% Make sure the correct number of arguments are provided
if nargin < 4
    error('You must supply at least 4 arguments to discreteVAR')
elseif nargin >= 6
    if ~strcmp(method,'quantile') && ~strcmp(method,'even') && ~strcmp(method,'quadrature')
        error('Method must be one of quantile, even, or quadrature')
    end
    if strcmp(method,'quantile')
        warning('quantile method is poor and not recommended')
    end
end

% Default number of moments is 2
if nargin == 4
    nMoments = 2;
end

% Default method is 'even'
if nargin <= 5
    method = 'even';
end

bSize = size(B);
M = bSize(1);
% Check to see if user has provided an explicit spacing for the 'even'
% method. If not, use default value.
if strcmp(method,'even') && nargin < 7
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
    mu = ((eye(M)-B)\eye(M))*b;
    A1 = C1\(B*C1);
    Sigma1 = reshape(((eye(M^2)-kron(A1,A1))\eye(M^2))*reshape(eye(M),M^2,1),M,M); % unconditional variance
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
        for ii = 1:M
            minSigmas = sqrt(min(eigs(Sigma)));
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

condMean = A*D; % conditional mean of the VAR process at each grid point
P = ones(Nm^M); % probability transition matrix
scalingFactor = y1D(:,end); % normalizing constant for maximum entropy computations
temp = zeros(M,Nm); % used to store some intermediate calculations
lambdaBar = zeros(2*M,Nm^M); % store optimized values of lambda (2 moments) to improve initial guesses
kappa = 1e-8; % small positive constant for numerical stability

for ii = 1:(Nm^M)
   
    % Construct prior guesses for maximum entropy optimizations
    switch method
        case 'even'
            q = normpdf(y1D,repmat(condMean(:,ii),1,Nm),1);
        case 'quantile'
            q = normcdf(y1DBounds(:,2:end),repmat(condMean(:,ii),1,Nm),1)...
                - normcdf(y1DBounds(:,1:end-1),repmat(condMean(:,ii),1,Nm),1);
        case 'quadrature'
            q = bsxfun(@times,(normpdf(y1D,repmat(condMean(:,ii),1,Nm),1)./normpdf(y1D,0,1)),...
                (weights'./sqrt(pi)));
    end
    
    % Make sure all elements of the prior are stricly positive
    q(q<kappa) = kappa;
    
    for jj = 1:M
        
        % Try to use intelligent initial guesses
        if ii == 1
            lambdaGuess = zeros(2,1);
        else
            lambdaGuess = lambdaBar((jj-1)*2+1:jj*2,ii-1);
        end
        
        % Maximum entropy optimization
        if nMoments == 1 % match only 1 moment
            temp(jj,:) = discreteApproximation(y1D(jj,:),...
                @(X)(X-condMean(jj,ii))/scalingFactor(jj),0,q(jj,:),0);
        else % match 2 moments first
            [p,lambda,momentError] = discreteApproximation(y1D(jj,:),...
                @(X) polynomialMoment(X,condMean(jj,ii),scalingFactor(jj),2),...
                [0; 1]./(scalingFactor(jj).^(1:2)'),q(jj,:),lambdaGuess);
            if norm(momentError) > 1e-5 % if 2 moments fail, then just match 1 moment
                warning('Failed to match first 2 moments. Just matching 1.')
                temp(jj,:) = discreteApproximation(y1D(jj,:),...
                    @(X)(X-condMean(jj,ii))/scalingFactor(jj),0,q(jj,:),0);
                lambdaBar((jj-1)*2+1:jj*2,ii) = zeros(2,1);
            elseif nMoments == 2
                lambdaBar((jj-1)*2+1:jj*2,ii) = lambda;
                temp(jj,:) = p;
            else % solve maximum entropy problem sequentially from low order moments
                lambdaBar((jj-1)*2+1:jj*2,ii) = lambda;
                for mm = 4:2:nMoments
                    lambdaGuess = [lambda;0;0]; % add zero to previous lambda
                    [pnew,lambda,momentError] = discreteApproximation(y1D(jj,:),...
                        @(X) polynomialMoment(X,condMean(jj,ii),scalingFactor(jj),mm),...
                        gaussianMoment(1:mm)./(scalingFactor(jj).^(1:mm)'),q(jj,:),lambdaGuess);
                    if norm(momentError) > 1e-5
                        warning(sprintf('Failed to match first %d moments.  Just matching %d.',mm,mm-2))
                        break;
                    else
                        p = pnew;
                    end
                end
                temp(jj,:) = p;
            end
        end
    end
    
    P(ii,:) = prod(allcomb2(temp),2)';
    
end

X = C*D + repmat(mu,1,Nm^M); % map grids back to original space

warning on MATLAB:singularMatrix

end