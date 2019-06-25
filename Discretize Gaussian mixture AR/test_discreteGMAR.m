Nm = 9; % number of points for discretization

mu = 0.0555;
A1 = 0.5854;
A2 = [0.8959 -0.3990];
pC = [0.1628 0.8372];
muC = [-0.0039 0.0008];
sigmaC = [0.1293 0.0300];

nMoments = 2;
tic
[PEven1,XEven1] = discreteGMAR(mu,A1,pC,muC,sigmaC,Nm,nMoments,'even');
toc
tic
[PEven2,XEven2] = discreteGMAR(mu,A2,pC,muC,sigmaC,Nm,nMoments,'even');
toc

tic
[PGMQ1,XGMQ1] = discreteGMAR(mu,A1,pC,muC,sigmaC,Nm,nMoments,'GMQ');
toc
tic
[PGMQ2,XGMQ2] = discreteGMAR(mu,A2,pC,muC,sigmaC,Nm,nMoments,'GMQ');
toc

[v,~] = eigs(PEven1',1,1);
pi = v/sum(v); % stationary distribution

figure
plot(XEven1,pi)
xlabel('x')
ylabel('Stationary distribution')