%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% discreteNP
% (c) 2023 Alexis Akira Toda
%
% Purpose:
%       discretize a nonparametric distribution given centered moments
% Usage:
%       [x,p] = discreteNP(N,cMoments,q)
%
% Inputs:
% N         - number of grid points
% cMoments  - vector of centered moments
% q         - vector of initial probability (optional)
%
% Outputs:
% x         - grid points
% p         - probability
%
% Version 1.0: September 27, 2023
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [x,p] = discreteNP(N,cMoments,q)

% some error checking
L = length(cMoments); % number of moments to be matched
if L < 2
    error('You need to provide at least two moments')
end
if L > N+1
    error('There are not enough points to match moments')
end
if cMoments(1) ~= 0
    error('Moments must be centered')
end
if cMoments(2) <= 0
    error('Second moment must be positive')
end
sigma = sqrt(cMoments(2)); % standard deviation
x = linspace(-sigma*sqrt(2*N),sigma*sqrt(2*N),N); % grid points
% using sqrt(2*N) times standard deviation recommended by Farmer-Toda

if nargin < 3
    q = normpdf(x,0,sigma);
    q = q/sum(q); % if initial prior not provided, use Gaussian
end
if length(q) ~= N
    error('Length of q must equal N')
end
if size(cMoments,1) < size(cMoments,2)
    cMoments = cMoments'; % convert to column vector
end

T = @(x)bsxfun(@power,x,[1:L]'); % moment defining function
p = discreteApproximation(x,T,cMoments,q);

end