%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NPGQ
% (c) 2019 Alexis Akira Toda
% 
% Purpose: 
%       Discretize a nonparametric distribution directly from data
%
% Usage:
%       [x,w] = NPGQ(data,N)
%
% Inputs:
%       - data: data (one dimensional). Because sufficiently high-order
%       moments need to exist, if you know data is positive, take the
%       logarithm before feeding in. Afterwards you can recover the grid by
%       computing exp(x) instead of x.
%       - N: number of grid points
%
% Outputs:
%       - x: nodes of nonparametric Gaussian quadrature
%       - w: weights (probability) of nonparametric Gaussian quadrature
%
% Version 1.1: May 27, 2019
%
function [x,w] = NPGQ(data,N)

%% standardize data for numerical stability
I = length(data);
if I <= 1
    error('sample size must exceed 1')
end

if all(data > 0)
    warning('Your data is positive and may not have sufficiently high moments. Consider using log(data) instead of data')
end

mu = sum(data)/I; % sample mean
sigma = sqrt(sum((data-mu).^2)/I); % sample standard deviation

z = (data-mu)/sigma; % standardize data for numerical stability

if size(z,1) > size(z,2)
    z = z'; % convert to row vector
end

%% Precompute polynomial moments of Gaussian mixture
temp = bsxfun(@power,z,[0:2*N]'); % matrix that stores powers of data
PolyMoments = sum(temp,2)/I; % column vector of polynomial moments

%% Implement Golub-Welsch algorithm
M = zeros(N+1); % matrix of moments
for n = 1:N+1
    M(n,:) = PolyMoments(n:N+n);
end
R = chol(M); % Cholesky factorization
temp0 = diag(R);
temp0(end) = [];
beta = temp0(2:N)./temp0(1:N-1);
temp1 = diag(R,1);
temp2 = temp1./temp0;
alpha = temp2 - [0;temp2(1:N-1)];

T = diag(alpha) + diag(beta,-1) + diag(beta,1);
[V,D] = eig(T);

%% Compute nodes and weights of Gaussian quadrature
x = diag(D)';
[x,ind] = sort(x); % sort nodes in ascending order

w = zeros(1,N);
for n = 1:N
    v = V(:,n);
    w(n) = v(1)^2/dot(v,v);
end
w = w(ind);

%% convert back to original scale
x = mu + sigma*x;

end

