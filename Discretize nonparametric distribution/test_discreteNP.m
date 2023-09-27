%% figure formatting

set(0,'DefaultTextInterpreter','latex')
set(0,'DefaultAxesTickLabelInterpreter','latex');
set(0,'DefaultLegendInterpreter', 'latex')
   
set(0,'DefaultTextFontSize', 14)
set(0,'DefaultAxesFontSize', 14)
set(0,'DefaultLineLineWidth',1)

temp = get(gca,'ColorOrder');
c1 = temp(1,:);
c2 = temp(2,:);
clear temp
close all


N = 9; % number of grid points
cMoments1 = [0, 1, 0, 3]; % Gaussian
cMoments2 = [0, 1, 0, 5]; % some example

tic
[x1,p1] = discreteNP(N,cMoments1);
[x2,p2] = discreteNP(N,cMoments2);
toc

figure
plot(x1,p1); hold on
plot(x2,p2)
xlabel('$x$')
ylabel('Probability mass')
legend(['$E[X^4]=$' num2str(cMoments1(end))], ['$E[X^4]=$' num2str(cMoments2(end))])