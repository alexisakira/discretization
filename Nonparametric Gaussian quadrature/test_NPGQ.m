I = 10000; % sample size

mu = 0;
sigma = 0.2;
alpha = 2;
beta = 1;
U = rand(I,1); % uniform random variable
Normal = normrnd(mu,sigma,[I,1]); % normal random variable
p = alpha/(alpha+beta);
Laplace = (U <= p).*exprnd(1/alpha,[I,1]) - (U > p).*exprnd(1/beta,[I,1]); % Laplace random variable
data = Normal + Laplace; % normal-Laplace random variable

histogram(data);

N = 9; % number of grid points

tic
[x,w] = NPGQ(data,N);
toc